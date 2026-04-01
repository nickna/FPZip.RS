use fpzip_rs::{FpZipError, FpZipHeader, FpZipType};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Compresses float data and writes to an async writer.
pub async fn compress_f32<W: AsyncWrite + Unpin>(
    data: &[f32],
    writer: &mut W,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<u64, FpZipError> {
    let compressed = fpzip_rs::compress_f32(data, nx, ny, nz, nf)?;
    writer
        .write_all(&compressed)
        .await
        .map_err(FpZipError::Io)?;
    Ok(compressed.len() as u64)
}

/// Compresses double data and writes to an async writer.
pub async fn compress_f64<W: AsyncWrite + Unpin>(
    data: &[f64],
    writer: &mut W,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<u64, FpZipError> {
    let compressed = fpzip_rs::compress_f64(data, nx, ny, nz, nf)?;
    writer
        .write_all(&compressed)
        .await
        .map_err(FpZipError::Io)?;
    Ok(compressed.len() as u64)
}

/// Decompresses float data from an async reader.
pub async fn decompress_f32<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<(FpZipHeader, Vec<f32>), FpZipError> {
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .await
        .map_err(FpZipError::Io)?;

    let header = fpzip_rs::read_header(&data)?;
    if header.data_type != FpZipType::Float {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Float,
            actual: header.data_type,
        });
    }

    let result = fpzip_rs::decompress_f32(&data)?;
    Ok((header, result))
}

/// Decompresses double data from an async reader.
pub async fn decompress_f64<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<(FpZipHeader, Vec<f64>), FpZipError> {
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .await
        .map_err(FpZipError::Io)?;

    let header = fpzip_rs::read_header(&data)?;
    if header.data_type != FpZipType::Double {
        return Err(FpZipError::TypeMismatch {
            expected: FpZipType::Double,
            actual: header.data_type,
        });
    }

    let result = fpzip_rs::decompress_f64(&data)?;
    Ok((header, result))
}

/// Compresses float data on a blocking thread (for large datasets).
pub async fn compress_f32_blocking(
    data: Vec<f32>,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<Vec<u8>, FpZipError> {
    tokio::task::spawn_blocking(move || fpzip_rs::compress_f32(&data, nx, ny, nz, nf))
        .await
        .map_err(|e| FpZipError::Io(std::io::Error::other(e)))?
}

/// Compresses double data on a blocking thread (for large datasets).
pub async fn compress_f64_blocking(
    data: Vec<f64>,
    nx: u32,
    ny: u32,
    nz: u32,
    nf: u32,
) -> Result<Vec<u8>, FpZipError> {
    tokio::task::spawn_blocking(move || fpzip_rs::compress_f64(&data, nx, ny, nz, nf))
        .await
        .map_err(|e| FpZipError::Io(std::io::Error::other(e)))?
}
