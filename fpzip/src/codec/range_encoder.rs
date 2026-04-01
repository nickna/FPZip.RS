use super::rc_qs_model::RCQsModel;

extern crate alloc;
use alloc::vec::Vec;

/// Range coder (arithmetic encoder) for entropy coding.
pub struct RangeEncoder {
    output: Vec<u8>,
    low: u32,
    range: u32,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    /// Creates a new range encoder.
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            low: 0,
            range: 0xFFFFFFFF,
        }
    }

    /// Creates a new range encoder with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            output: Vec::with_capacity(capacity),
            low: 0,
            range: 0xFFFFFFFF,
        }
    }

    /// Returns the number of bytes written so far.
    pub fn bytes_written(&self) -> usize {
        self.output.len()
    }

    /// Finishes encoding and returns the compressed data.
    pub fn finish(mut self) -> Vec<u8> {
        self.put(4);
        self.output
    }

    /// Returns reference to the output buffer.
    pub fn output(&self) -> &[u8] {
        &self.output
    }

    /// Consumes and returns the output buffer without finishing.
    pub fn into_output(self) -> Vec<u8> {
        self.output
    }

    /// Encodes a single bit.
    #[inline]
    pub fn encode_bit(&mut self, bit: bool) {
        self.range >>= 1;
        if bit {
            self.low = self.low.wrapping_add(self.range);
        }
        self.normalize();
    }

    /// Encodes a symbol using a probability model.
    #[inline]
    pub fn encode_with_model(&mut self, symbol: u32, model: &mut RCQsModel) {
        let (l, r) = model.encode(symbol);
        model.normalize(&mut self.range);
        self.low = self.low.wrapping_add(self.range.wrapping_mul(l));
        self.range = self.range.wrapping_mul(r);
        self.normalize();
    }

    /// Encodes an n-bit unsigned integer (0 <= s < 2^n).
    #[inline]
    pub fn encode_uint(&mut self, s: u32, n: i32) {
        if n <= 0 {
            return;
        }
        let mut s = s;
        let mut n = n;
        if n > 16 {
            self.encode_shift(s & 0xFFFF, 16);
            s >>= 16;
            n -= 16;
        }
        self.encode_shift(s, n);
    }

    /// Encodes a 64-bit unsigned integer with n bits.
    pub fn encode_ulong(&mut self, s: u64, n: i32) {
        if n <= 0 {
            return;
        }
        let mut s = s;
        let mut n = n;
        while n > 16 {
            self.encode_shift((s & 0xFFFF) as u32, 16);
            s >>= 16;
            n -= 16;
        }
        self.encode_shift(s as u32, n);
    }

    /// Encodes an integer using shift (for power-of-2 ranges).
    #[inline]
    fn encode_shift(&mut self, s: u32, n: i32) {
        self.range >>= n as u32;
        self.low = self.low.wrapping_add(self.range.wrapping_mul(s));
        self.normalize();
    }

    /// Normalizes the range and outputs fixed bits.
    #[inline]
    fn normalize(&mut self) {
        while ((self.low ^ self.low.wrapping_add(self.range)) >> 24) == 0 {
            self.put(1);
            self.range <<= 8;
        }
        if (self.range >> 16) == 0 {
            self.put(2);
            self.range = 0u32.wrapping_sub(self.low);
        }
    }

    /// Outputs n bytes from the high bits of low.
    #[inline]
    fn put(&mut self, n: i32) {
        for _ in 0..n {
            self.output.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::range_decoder::RangeDecoder;

    #[test]
    fn encode_decode_bits() {
        let mut enc = RangeEncoder::new();
        enc.encode_bit(true);
        enc.encode_bit(false);
        enc.encode_bit(true);
        enc.encode_bit(true);
        enc.encode_bit(false);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert!(dec.decode_bit());
        assert!(!dec.decode_bit());
        assert!(dec.decode_bit());
        assert!(dec.decode_bit());
        assert!(!dec.decode_bit());
    }

    #[test]
    fn encode_decode_uint8() {
        let mut enc = RangeEncoder::new();
        enc.encode_uint(0xAB, 8);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert_eq!(dec.decode_uint(8), 0xAB);
    }

    #[test]
    fn encode_decode_uint16() {
        let mut enc = RangeEncoder::new();
        enc.encode_uint(0xABCD, 16);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert_eq!(dec.decode_uint(16), 0xABCD);
    }

    #[test]
    fn encode_decode_uint32() {
        let mut enc = RangeEncoder::new();
        enc.encode_uint(0xDEADBEEF, 32);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert_eq!(dec.decode_uint(32), 0xDEADBEEF);
    }

    #[test]
    fn encode_decode_ulong64() {
        let mut enc = RangeEncoder::new();
        enc.encode_ulong(0xDEADBEEFCAFEBABE, 64);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert_eq!(dec.decode_ulong(64), 0xDEADBEEFCAFEBABE);
    }

    #[test]
    fn encode_decode_with_model() {
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, 65);
        enc.encode_with_model(32, &mut model); // bias symbol
        enc.encode_with_model(0, &mut model);
        enc.encode_with_model(64, &mut model);
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, 65);
        assert_eq!(dec.decode_with_model(&mut dmodel), 32);
        assert_eq!(dec.decode_with_model(&mut dmodel), 0);
        assert_eq!(dec.decode_with_model(&mut dmodel), 64);
    }

    #[test]
    fn bytes_written_tracking() {
        let mut enc = RangeEncoder::new();
        assert_eq!(enc.bytes_written(), 0);
        enc.encode_bit(true);
        // After encoding, some bytes may have been output
        let data = enc.finish();
        assert!(data.len() > 0);
    }

    #[test]
    fn mixed_encoding_modes() {
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, 10);

        enc.encode_bit(true);
        enc.encode_uint(42, 8);
        enc.encode_with_model(5, &mut model);
        enc.encode_uint(0x1234, 16);
        enc.encode_bit(false);

        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, 10);

        assert!(dec.decode_bit());
        assert_eq!(dec.decode_uint(8), 42);
        assert_eq!(dec.decode_with_model(&mut dmodel), 5);
        assert_eq!(dec.decode_uint(16), 0x1234);
        assert!(!dec.decode_bit());
    }

    #[test]
    fn large_symbol_sequence() {
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, 65);

        let symbols: Vec<u32> = (0..1000).map(|i| (i % 65) as u32).collect();
        for &s in &symbols {
            enc.encode_with_model(s, &mut model);
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, 65);

        for &expected in &symbols {
            assert_eq!(dec.decode_with_model(&mut dmodel), expected);
        }
    }
}
