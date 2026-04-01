use std::ffi::c_char;
use std::slice;

use fpzip_rs::{FpZipError, FpZipType};

/// C-compatible header struct.
#[repr(C)]
pub struct FpZipHeaderC {
    pub data_type: u8,
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub nf: u32,
}

// Error codes
const FPZIP_OK: i32 = 0;
const FPZIP_ERR_NULL_PTR: i32 = -1;
const FPZIP_ERR_INVALID_MAGIC: i32 = -2;
const FPZIP_ERR_UNSUPPORTED_VERSION: i32 = -3;
const FPZIP_ERR_INVALID_TYPE: i32 = -4;
const FPZIP_ERR_DIMENSION_MISMATCH: i32 = -5;
const FPZIP_ERR_TYPE_MISMATCH: i32 = -6;
const FPZIP_ERR_BUFFER_TOO_SMALL: i32 = -7;
const FPZIP_ERR_UNEXPECTED_EOF: i32 = -8;
const FPZIP_ERR_IO: i32 = -9;
fn error_to_code(e: &FpZipError) -> i32 {
    match e {
        FpZipError::InvalidMagic(_) => FPZIP_ERR_INVALID_MAGIC,
        FpZipError::UnsupportedVersion(_) => FPZIP_ERR_UNSUPPORTED_VERSION,
        FpZipError::InvalidDataType(_) => FPZIP_ERR_INVALID_TYPE,
        FpZipError::DimensionMismatch { .. } => FPZIP_ERR_DIMENSION_MISMATCH,
        FpZipError::TypeMismatch { .. } => FPZIP_ERR_TYPE_MISMATCH,
        FpZipError::BufferTooSmall { .. } => FPZIP_ERR_BUFFER_TOO_SMALL,
        FpZipError::UnexpectedEof => FPZIP_ERR_UNEXPECTED_EOF,
        FpZipError::Io(_) => FPZIP_ERR_IO,
    }
}

/// Compresses float data.
///
/// Returns 0 on success, negative error code on failure.
/// On success, `*out_len` is set to the number of bytes written.
///
/// # Safety
/// All pointers must be valid. `data` must point to `len` floats.
/// `out_buf` must point to `out_capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn fpzip_compress_float(
    data: *const f32,
    len: usize,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
    out_buf: *mut u8,
    out_capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || out_buf.is_null() || out_len.is_null() {
        return FPZIP_ERR_NULL_PTR;
    }

    let input = unsafe { slice::from_raw_parts(data, len) };
    match fpzip_rs::compress_f32(input, nx, ny, nz, nf) {
        Ok(compressed) => {
            if compressed.len() > out_capacity {
                return FPZIP_ERR_BUFFER_TOO_SMALL;
            }
            let output = unsafe { slice::from_raw_parts_mut(out_buf, out_capacity) };
            output[..compressed.len()].copy_from_slice(&compressed);
            unsafe { *out_len = compressed.len() };
            FPZIP_OK
        }
        Err(e) => error_to_code(&e),
    }
}

/// Compresses double data.
///
/// # Safety
/// Same requirements as `fpzip_compress_float`.
#[no_mangle]
pub unsafe extern "C" fn fpzip_compress_double(
    data: *const f64,
    len: usize,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
    out_buf: *mut u8,
    out_capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || out_buf.is_null() || out_len.is_null() {
        return FPZIP_ERR_NULL_PTR;
    }

    let input = unsafe { slice::from_raw_parts(data, len) };
    match fpzip_rs::compress_f64(input, nx, ny, nz, nf) {
        Ok(compressed) => {
            if compressed.len() > out_capacity {
                return FPZIP_ERR_BUFFER_TOO_SMALL;
            }
            let output = unsafe { slice::from_raw_parts_mut(out_buf, out_capacity) };
            output[..compressed.len()].copy_from_slice(&compressed);
            unsafe { *out_len = compressed.len() };
            FPZIP_OK
        }
        Err(e) => error_to_code(&e),
    }
}

/// Decompresses float data.
///
/// # Safety
/// `compressed` must point to `compressed_len` bytes.
/// `out_buf` must point to `out_capacity` floats.
#[no_mangle]
pub unsafe extern "C" fn fpzip_decompress_float(
    compressed: *const u8,
    compressed_len: usize,
    out_buf: *mut f32,
    out_capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if compressed.is_null() || out_buf.is_null() || out_len.is_null() {
        return FPZIP_ERR_NULL_PTR;
    }

    let input = unsafe { slice::from_raw_parts(compressed, compressed_len) };
    match fpzip_rs::decompress_f32(input) {
        Ok(data) => {
            if data.len() > out_capacity {
                return FPZIP_ERR_BUFFER_TOO_SMALL;
            }
            let output = unsafe { slice::from_raw_parts_mut(out_buf, out_capacity) };
            output[..data.len()].copy_from_slice(&data);
            unsafe { *out_len = data.len() };
            FPZIP_OK
        }
        Err(e) => error_to_code(&e),
    }
}

/// Decompresses double data.
///
/// # Safety
/// Same requirements as `fpzip_decompress_float`.
#[no_mangle]
pub unsafe extern "C" fn fpzip_decompress_double(
    compressed: *const u8,
    compressed_len: usize,
    out_buf: *mut f64,
    out_capacity: usize,
    out_len: *mut usize,
) -> i32 {
    if compressed.is_null() || out_buf.is_null() || out_len.is_null() {
        return FPZIP_ERR_NULL_PTR;
    }

    let input = unsafe { slice::from_raw_parts(compressed, compressed_len) };
    match fpzip_rs::decompress_f64(input) {
        Ok(data) => {
            if data.len() > out_capacity {
                return FPZIP_ERR_BUFFER_TOO_SMALL;
            }
            let output = unsafe { slice::from_raw_parts_mut(out_buf, out_capacity) };
            output[..data.len()].copy_from_slice(&data);
            unsafe { *out_len = data.len() };
            FPZIP_OK
        }
        Err(e) => error_to_code(&e),
    }
}

/// Reads the header from compressed data.
///
/// # Safety
/// `data` must point to `data_len` bytes. `header` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn fpzip_read_header(
    data: *const u8,
    data_len: usize,
    header: *mut FpZipHeaderC,
) -> i32 {
    if data.is_null() || header.is_null() {
        return FPZIP_ERR_NULL_PTR;
    }

    let input = unsafe { slice::from_raw_parts(data, data_len) };
    match fpzip_rs::read_header(input) {
        Ok(h) => {
            let out = unsafe { &mut *header };
            out.data_type = h.data_type as u8;
            out.nx = h.nx;
            out.ny = h.ny;
            out.nz = h.nz;
            out.nf = h.nf;
            FPZIP_OK
        }
        Err(e) => error_to_code(&e),
    }
}

/// Returns the maximum possible compressed size.
#[no_mangle]
pub extern "C" fn fpzip_max_compressed_size(element_count: usize, data_type: u8) -> usize {
    let dt = if data_type == 0 {
        FpZipType::Float
    } else {
        FpZipType::Double
    };
    fpzip_rs::max_compressed_size(element_count, dt)
}

/// Returns a static error message string for the given error code.
#[no_mangle]
pub extern "C" fn fpzip_error_message(code: i32) -> *const c_char {
    let msg: &[u8] = match code {
        FPZIP_OK => b"success\0",
        FPZIP_ERR_NULL_PTR => b"null pointer\0",
        FPZIP_ERR_INVALID_MAGIC => b"invalid magic number\0",
        FPZIP_ERR_UNSUPPORTED_VERSION => b"unsupported version\0",
        FPZIP_ERR_INVALID_TYPE => b"invalid data type\0",
        FPZIP_ERR_DIMENSION_MISMATCH => b"dimension mismatch\0",
        FPZIP_ERR_TYPE_MISMATCH => b"type mismatch\0",
        FPZIP_ERR_BUFFER_TOO_SMALL => b"buffer too small\0",
        FPZIP_ERR_UNEXPECTED_EOF => b"unexpected end of input\0",
        FPZIP_ERR_IO => b"I/O error\0",
        _ => b"unknown error\0",
    };
    msg.as_ptr() as *const c_char
}
