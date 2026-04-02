/// Errors that can occur during FPZip compression or decompression.
#[derive(Debug)]
pub enum FpZipError {
    /// The compressed data does not start with the fpzip magic number.
    InvalidMagic(u32),
    /// The format version in the header is not supported.
    UnsupportedVersion(u16),
    /// The data type byte in the header is not 0 (float) or 1 (double).
    InvalidDataType(u8),
    /// The input data length does not match `nx * ny * nz * nf`.
    DimensionMismatch {
        actual: usize,
        expected: u64,
        nx: u32,
        ny: u32,
        nz: u32,
        nf: u32,
    },
    /// Attempted to decompress float data as double, or vice versa.
    TypeMismatch {
        expected: crate::header::FpZipType,
        actual: crate::header::FpZipType,
    },
    /// The output buffer is too small for the result.
    BufferTooSmall { needed: usize, available: usize },
    /// The compressed data ended unexpectedly.
    UnexpectedEof,
    /// An I/O error occurred during stream-based operations.
    #[cfg(feature = "std")]
    Io(std::io::Error),
}

impl core::fmt::Display for FpZipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMagic(m) => {
                write!(f, "invalid magic number: expected 0x007A7066, got 0x{m:08X}")
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported version {v}, maximum supported: 1")
            }
            Self::InvalidDataType(t) => write!(f, "invalid data type: {t}"),
            Self::DimensionMismatch {
                actual,
                expected,
                nx,
                ny,
                nz,
                nf,
            } => write!(
                f,
                "dimension mismatch: data length {actual} != {expected} (nx={nx} * ny={ny} * nz={nz} * nf={nf})"
            ),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::BufferTooSmall { needed, available } => {
                write!(
                    f,
                    "buffer too small: need {needed} bytes, got {available}"
                )
            }
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FpZipError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for FpZipError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A specialized `Result` type for fpzip operations.
pub type Result<T> = core::result::Result<T, FpZipError>;
