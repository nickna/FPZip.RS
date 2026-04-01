use crate::codec::range_decoder::RangeDecoder;
use crate::codec::range_encoder::RangeEncoder;
use crate::error::{FpZipError, Result};

/// C++ fpzip major version.
pub const FPZ_MAJ_VERSION: u32 = 0x0110;

/// C++ fpzip minor version (FPZIP_FP_INT = 4).
pub const FPZ_MIN_VERSION: u32 = 4;

/// Type of floating-point data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FpZipType {
    Float = 0,
    Double = 1,
}

/// Header structure for FpZip compressed data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpZipHeader {
    pub data_type: FpZipType,
    pub prec: u32,
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub nf: u32,
}

impl FpZipHeader {
    pub fn new(data_type: FpZipType, nx: u32, ny: u32, nz: u32, nf: u32) -> Self {
        let prec = match data_type {
            FpZipType::Float => 32,
            FpZipType::Double => 64,
        };
        Self {
            data_type,
            prec,
            nx,
            ny,
            nz,
            nf,
        }
    }

    /// Total number of elements.
    pub fn total_elements(&self) -> u64 {
        self.nx as u64 * self.ny as u64 * self.nz as u64 * self.nf as u64
    }

    /// Writes the header through the range encoder (C++ compatible format).
    pub fn write_to_encoder(&self, enc: &mut RangeEncoder) {
        // magic: 'f', 'p', 'z', '\0'
        enc.encode_uint(b'f' as u32, 8);
        enc.encode_uint(b'p' as u32, 8);
        enc.encode_uint(b'z' as u32, 8);
        enc.encode_uint(0, 8);

        // format version
        enc.encode_uint(FPZ_MAJ_VERSION, 16);
        enc.encode_uint(FPZ_MIN_VERSION, 8);

        // type (1 bit) and precision (7 bits)
        enc.encode_uint(self.data_type as u32, 1);
        enc.encode_uint(self.prec, 7);

        // array dimensions
        enc.encode_uint(self.nx, 32);
        enc.encode_uint(self.ny, 32);
        enc.encode_uint(self.nz, 32);
        enc.encode_uint(self.nf, 32);
    }

    /// Reads the header through the range decoder (C++ compatible format).
    pub fn read_from_decoder(dec: &mut RangeDecoder) -> Result<Self> {
        // magic
        let f = dec.decode_uint(8);
        let p = dec.decode_uint(8);
        let z = dec.decode_uint(8);
        let nul = dec.decode_uint(8);
        if f != b'f' as u32 || p != b'p' as u32 || z != b'z' as u32 || nul != 0 {
            let magic = f | (p << 8) | (z << 16) | (nul << 24);
            return Err(FpZipError::InvalidMagic(magic));
        }

        // format version
        let maj = dec.decode_uint(16);
        let min = dec.decode_uint(8);
        if maj != FPZ_MAJ_VERSION {
            return Err(FpZipError::UnsupportedVersion(maj as u16));
        }
        if min != FPZ_MIN_VERSION {
            return Err(FpZipError::UnsupportedVersion(min as u16));
        }

        // type (1 bit) and precision (7 bits)
        let type_bit = dec.decode_uint(1);
        let prec = dec.decode_uint(7);

        let data_type = match type_bit {
            0 => FpZipType::Float,
            1 => FpZipType::Double,
            _ => return Err(FpZipError::InvalidDataType(type_bit as u8)),
        };

        // array dimensions
        let nx = dec.decode_uint(32);
        let ny = dec.decode_uint(32);
        let nz = dec.decode_uint(32);
        let nf = dec.decode_uint(32);

        Ok(Self {
            data_type,
            prec,
            nx,
            ny,
            nz,
            nf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_header_via_encoder() {
        let h = FpZipHeader::new(FpZipType::Float, 10, 20, 30, 2);
        let mut enc = RangeEncoder::new();
        h.write_to_encoder(&mut enc);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let h2 = FpZipHeader::read_from_decoder(&mut dec).unwrap();
        assert_eq!(h.data_type, h2.data_type);
        assert_eq!(h.nx, h2.nx);
        assert_eq!(h.ny, h2.ny);
        assert_eq!(h.nz, h2.nz);
        assert_eq!(h.nf, h2.nf);
    }

    #[test]
    fn total_elements() {
        let h = FpZipHeader::new(FpZipType::Double, 10, 20, 30, 2);
        assert_eq!(h.total_elements(), 12000);
    }

    #[test]
    fn invalid_magic_returns_error() {
        // Encode garbage instead of 'fpz\0'
        let mut enc = RangeEncoder::new();
        enc.encode_uint(b'X' as u32, 8);
        enc.encode_uint(b'Y' as u32, 8);
        enc.encode_uint(b'Z' as u32, 8);
        enc.encode_uint(0, 8);
        // fill rest so decoder doesn't EOF
        enc.encode_uint(FPZ_MAJ_VERSION, 16);
        enc.encode_uint(FPZ_MIN_VERSION, 8);
        enc.encode_uint(0, 1);
        enc.encode_uint(32, 7);
        enc.encode_uint(1, 32);
        enc.encode_uint(1, 32);
        enc.encode_uint(1, 32);
        enc.encode_uint(1, 32);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert!(matches!(
            FpZipHeader::read_from_decoder(&mut dec),
            Err(FpZipError::InvalidMagic(_))
        ));
    }

    #[test]
    fn round_trip_double_header() {
        let h = FpZipHeader::new(FpZipType::Double, 65, 64, 63, 3);
        let mut enc = RangeEncoder::new();
        h.write_to_encoder(&mut enc);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let h2 = FpZipHeader::read_from_decoder(&mut dec).unwrap();
        assert_eq!(h, h2);
    }
}
