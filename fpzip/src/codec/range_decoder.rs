use super::rc_qs_model::RCQsModel;

/// Range coder (arithmetic decoder) for entropy decoding.
pub struct RangeDecoder<'a> {
    input: &'a [u8],
    pos: usize,
    low: u32,
    range: u32,
    code: u32,
    error: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Creates a new range decoder reading from the given byte slice.
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            low: 0,
            range: 0xFFFFFFFF,
            code: 0,
            error: false,
        }
    }

    /// Whether an EOF error was encountered.
    pub fn error(&self) -> bool {
        self.error
    }

    /// Number of bytes read so far.
    pub fn bytes_read(&self) -> usize {
        self.pos
    }

    /// Initializes the decoder by reading the first 4 bytes.
    pub fn init(&mut self) {
        self.error = false;
        self.get(4);
    }

    /// Decodes a single bit.
    #[inline]
    pub fn decode_bit(&mut self) -> bool {
        self.range >>= 1;
        let s = self.code >= self.low.wrapping_add(self.range);
        if s {
            self.low = self.low.wrapping_add(self.range);
        }
        self.normalize();
        s
    }

    /// Decodes a symbol using a probability model.
    #[inline]
    pub fn decode_with_model(&mut self, model: &mut RCQsModel) -> u32 {
        model.normalize(&mut self.range);
        let mut l = self.code.wrapping_sub(self.low) / self.range;
        let mut r = 0u32;
        let s = model.decode(&mut l, &mut r);
        self.low = self.low.wrapping_add(self.range.wrapping_mul(l));
        self.range = self.range.wrapping_mul(r);
        self.normalize();
        s
    }

    /// Decodes an n-bit unsigned integer (0 <= result < 2^n).
    #[inline]
    pub fn decode_uint(&mut self, n: i32) -> u32 {
        if n <= 0 {
            return 0;
        }
        let mut s = 0u32;
        let mut m = 0;
        let mut n = n;

        while n > 16 {
            s += self.decode_shift(16) << m;
            m += 16;
            n -= 16;
        }

        (self.decode_shift(n) << m) + s
    }

    /// Decodes a 64-bit unsigned integer with n bits.
    pub fn decode_ulong(&mut self, n: i32) -> u64 {
        if n <= 0 {
            return 0;
        }
        let mut s = 0u64;
        let mut m = 0;
        let mut n = n;

        while n > 16 {
            s += (self.decode_shift(16) as u64) << m;
            m += 16;
            n -= 16;
        }

        ((self.decode_shift(n) as u64) << m) + s
    }

    /// Decodes using shift (for power-of-2 ranges).
    #[inline]
    fn decode_shift(&mut self, n: i32) -> u32 {
        self.range >>= n as u32;
        let s = self.code.wrapping_sub(self.low) / self.range;
        self.low = self.low.wrapping_add(self.range.wrapping_mul(s));
        self.normalize();
        s
    }

    /// Normalizes the range and inputs new data.
    #[inline]
    fn normalize(&mut self) {
        while ((self.low ^ self.low.wrapping_add(self.range)) >> 24) == 0 {
            self.get(1);
            self.range <<= 8;
        }
        if (self.range >> 16) == 0 {
            self.get(2);
            self.range = 0u32.wrapping_sub(self.low);
        }
    }

    /// Inputs n bytes from the stream.
    #[inline]
    fn get(&mut self, n: i32) {
        for _ in 0..n {
            self.code <<= 8;
            self.code |= self.get_byte() as u32;
            self.low <<= 8;
        }
    }

    /// Reads a single byte.
    #[inline]
    fn get_byte(&mut self) -> u8 {
        if self.pos >= self.input.len() {
            self.error = true;
            return 0;
        }
        let b = self.input[self.pos];
        self.pos += 1;
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eof_sets_error() {
        let data = [0u8; 4]; // Just enough for init
        let mut dec = RangeDecoder::new(&data);
        dec.init();
        assert!(!dec.error());
        // Decoding beyond available data should eventually set error
        for _ in 0..100 {
            dec.decode_bit();
        }
        assert!(dec.error());
    }
}
