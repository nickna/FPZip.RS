use crate::codec::range_decoder::RangeDecoder;
use crate::codec::range_encoder::RangeEncoder;
use crate::decoder;
use crate::encoder;
use crate::error::{FpZipError, Result};
use crate::header::{FpZipHeader, FpZipType};

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Compresses a float slice.
pub fn compress_f32(data: &[f32], nx: u32, ny: u32, nz: u32, nf: u32) -> Result<Vec<u8>> {
    validate_dimensions(data.len(), nx, ny, nz, nf)?;

    let header = FpZipHeader::new(FpZipType::Float, nx, ny, nz, nf);
    let mut enc = RangeEncoder::with_capacity(data.len() * 4);

    // Write header through the range encoder (C++ compatible)
    header.write_to_encoder(&mut enc);

    encoder::compress_4d_float(&mut enc, data, nx as i32, ny as i32, nz as i32, nf as i32);
    let output = enc.finish();

    Ok(output)
}

/// Compresses a double slice.
pub fn compress_f64(data: &[f64], nx: u32, ny: u32, nz: u32, nf: u32) -> Result<Vec<u8>> {
    validate_dimensions(data.len(), nx, ny, nz, nf)?;

    let header = FpZipHeader::new(FpZipType::Double, nx, ny, nz, nf);
    let mut enc = RangeEncoder::with_capacity(data.len() * 8);

    header.write_to_encoder(&mut enc);

    encoder::compress_4d_double(&mut enc, data, nx as i32, ny as i32, nz as i32, nf as i32);
    let output = enc.finish();

    Ok(output)
}

/// Decompresses data to a float vector.
pub fn decompress_f32(data: &[u8]) -> Result<Vec<f32>> {
    let mut dec = RangeDecoder::new(data);
    dec.init();

    let header = FpZipHeader::read_from_decoder(&mut dec)?;
    if header.data_type != FpZipType::Float {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Float,
            actual: header.data_type,
        });
    }

    let total = header.total_elements() as usize;
    let mut result = vec![0.0f32; total];

    decoder::decompress_4d_float(
        &mut dec,
        &mut result,
        header.nx as i32,
        header.ny as i32,
        header.nz as i32,
        header.nf as i32,
    );

    Ok(result)
}

/// Decompresses data to a double vector.
pub fn decompress_f64(data: &[u8]) -> Result<Vec<f64>> {
    let mut dec = RangeDecoder::new(data);
    dec.init();

    let header = FpZipHeader::read_from_decoder(&mut dec)?;
    if header.data_type != FpZipType::Double {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Double,
            actual: header.data_type,
        });
    }

    let total = header.total_elements() as usize;
    let mut result = vec![0.0f64; total];

    decoder::decompress_4d_double(
        &mut dec,
        &mut result,
        header.nx as i32,
        header.ny as i32,
        header.nz as i32,
        header.nf as i32,
    );

    Ok(result)
}

/// Decompresses float data into a pre-allocated buffer. Returns the header.
pub fn decompress_f32_into(data: &[u8], output: &mut [f32]) -> Result<FpZipHeader> {
    let mut dec = RangeDecoder::new(data);
    dec.init();

    let header = FpZipHeader::read_from_decoder(&mut dec)?;
    if header.data_type != FpZipType::Float {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Float,
            actual: header.data_type,
        });
    }

    let total = header.total_elements() as usize;
    if output.len() < total {
        return Err(FpZipError::BufferTooSmall {
            needed: total,
            available: output.len(),
        });
    }

    decoder::decompress_4d_float(
        &mut dec,
        output,
        header.nx as i32,
        header.ny as i32,
        header.nz as i32,
        header.nf as i32,
    );

    Ok(header)
}

/// Decompresses double data into a pre-allocated buffer. Returns the header.
pub fn decompress_f64_into(data: &[u8], output: &mut [f64]) -> Result<FpZipHeader> {
    let mut dec = RangeDecoder::new(data);
    dec.init();

    let header = FpZipHeader::read_from_decoder(&mut dec)?;
    if header.data_type != FpZipType::Double {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Double,
            actual: header.data_type,
        });
    }

    let total = header.total_elements() as usize;
    if output.len() < total {
        return Err(FpZipError::BufferTooSmall {
            needed: total,
            available: output.len(),
        });
    }

    decoder::decompress_4d_double(
        &mut dec,
        output,
        header.nx as i32,
        header.ny as i32,
        header.nz as i32,
        header.nf as i32,
    );

    Ok(header)
}

/// Compresses float data into a pre-allocated byte buffer.
/// Returns number of bytes written, or error if buffer too small.
pub fn compress_f32_into(
    data: &[f32],
    destination: &mut [u8],
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<usize> {
    let compressed = compress_f32(data, nx, ny, nz, nf)?;
    if compressed.len() > destination.len() {
        return Err(FpZipError::BufferTooSmall {
            needed: compressed.len(),
            available: destination.len(),
        });
    }
    destination[..compressed.len()].copy_from_slice(&compressed);
    Ok(compressed.len())
}

/// Compresses double data into a pre-allocated byte buffer.
pub fn compress_f64_into(
    data: &[f64],
    destination: &mut [u8],
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<usize> {
    let compressed = compress_f64(data, nx, ny, nz, nf)?;
    if compressed.len() > destination.len() {
        return Err(FpZipError::BufferTooSmall {
            needed: compressed.len(),
            available: destination.len(),
        });
    }
    destination[..compressed.len()].copy_from_slice(&compressed);
    Ok(compressed.len())
}

/// Reads the header from compressed data without decompressing.
pub fn read_header(data: &[u8]) -> Result<FpZipHeader> {
    let mut dec = RangeDecoder::new(data);
    dec.init();
    FpZipHeader::read_from_decoder(&mut dec)
}

/// Gets the maximum possible compressed size for the given element count.
pub fn max_compressed_size(element_count: usize, data_type: FpZipType) -> usize {
    let element_size = match data_type {
        FpZipType::Float => 4,
        FpZipType::Double => 8,
    };
    let data_size = element_count * element_size;
    // Header is encoded through range coder so no separate header size needed,
    // but add margin for worst-case expansion
    data_size + (data_size / 20) + 128
}

/// Builder for advanced compression usage.
pub struct FpZipCompressor {
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
}

impl FpZipCompressor {
    pub fn new(nx: u32) -> Self {
        Self {
            nx,
            ny: 1,
            nz: 1,
            nf: 1,
        }
    }

    pub fn ny(mut self, ny: u32) -> Self {
        self.ny = ny;
        self
    }

    pub fn nz(mut self, nz: u32) -> Self {
        self.nz = nz;
        self
    }

    pub fn nf(mut self, nf: u32) -> Self {
        self.nf = nf;
        self
    }

    pub fn compress_f32(&self, data: &[f32]) -> Result<Vec<u8>> {
        compress_f32(data, self.nx, self.ny, self.nz, self.nf)
    }

    pub fn compress_f64(&self, data: &[f64]) -> Result<Vec<u8>> {
        compress_f64(data, self.nx, self.ny, self.nz, self.nf)
    }

    pub fn compress_f32_into(&self, data: &[f32], dest: &mut [u8]) -> Result<usize> {
        compress_f32_into(data, dest, self.nx, self.ny, self.nz, self.nf)
    }

    pub fn compress_f64_into(&self, data: &[f64], dest: &mut [u8]) -> Result<usize> {
        compress_f64_into(data, dest, self.nx, self.ny, self.nz, self.nf)
    }
}

/// Stream-based compression for float data.
#[cfg(feature = "std")]
pub fn compress_f32_to_writer<W: std::io::Write>(
    data: &[f32],
    writer: &mut W,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<u64> {
    let compressed = compress_f32(data, nx, ny, nz, nf)?;
    writer.write_all(&compressed)?;
    Ok(compressed.len() as u64)
}

/// Stream-based compression for double data.
#[cfg(feature = "std")]
pub fn compress_f64_to_writer<W: std::io::Write>(
    data: &[f64],
    writer: &mut W,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<u64> {
    let compressed = compress_f64(data, nx, ny, nz, nf)?;
    writer.write_all(&compressed)?;
    Ok(compressed.len() as u64)
}

/// Stream-based decompression for float data.
#[cfg(feature = "std")]
pub fn decompress_f32_from_reader<R: std::io::Read>(
    reader: &mut R,
) -> Result<(FpZipHeader, Vec<f32>)> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    let header_copy = read_header(&data)?;
    if header_copy.data_type != FpZipType::Float {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Float,
            actual: header_copy.data_type,
        });
    }
    let result = decompress_f32(&data)?;
    Ok((header_copy, result))
}

/// Stream-based decompression for double data.
#[cfg(feature = "std")]
pub fn decompress_f64_from_reader<R: std::io::Read>(
    reader: &mut R,
) -> Result<(FpZipHeader, Vec<f64>)> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    let header_copy = read_header(&data)?;
    if header_copy.data_type != FpZipType::Double {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Double,
            actual: header_copy.data_type,
        });
    }
    let result = decompress_f64(&data)?;
    Ok((header_copy, result))
}

fn validate_dimensions(data_length: usize, nx: u32, ny: u32, nz: u32, nf: u32) -> Result<()> {
    let expected = nx as u64 * ny as u64 * nz as u64 * nf as u64;
    if data_length as u64 != expected {
        return Err(FpZipError::DimensionMismatch {
            actual: data_length,
            expected,
            nx,
            ny,
            nz,
            nf,
        });
    }
    Ok(())
}
