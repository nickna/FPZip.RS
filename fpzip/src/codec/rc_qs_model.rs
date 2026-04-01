extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

const TABLE_SHIFT: i32 = 7;

/// Quasi-Static adaptive probability model for the range coder.
pub struct RCQsModel {
    symbols: usize,
    bits: i32,
    left: i32,
    more: i32,
    incr: u32,
    rescale: i32,
    target_rescale: i32,
    symf: Vec<u32>,
    cumf: Vec<u32>,
    search_shift: i32,
    search: Option<Vec<u32>>,
}

impl RCQsModel {
    /// Creates a new quasi-static probability model.
    ///
    /// `compress`: true for compression, false for decompression.
    /// `symbols`: number of symbols.
    /// `bits`: log2 of total frequency count (must be <= 16).
    /// `period`: max symbols between normalizations.
    pub fn new(compress: bool, symbols: usize, bits: i32, period: i32) -> Self {
        assert!(bits <= 16, "bits must be <= 16");
        assert!(period < (1 << (bits + 1)), "period too large");

        let n = symbols;
        let symf = vec![0u32; n + 1];
        let mut cumf = vec![0u32; n + 1];
        cumf[0] = 0;
        cumf[n] = 1u32 << bits;

        let (search, search_shift) = if compress {
            (None, 0)
        } else {
            let ss = bits - TABLE_SHIFT;
            let s = vec![0u32; (1 << TABLE_SHIFT) + 1];
            (Some(s), ss)
        };

        let mut model = Self {
            symbols,
            bits,
            left: 0,
            more: 0,
            incr: 0,
            rescale: 0,
            target_rescale: period,
            symf,
            cumf,
            search_shift,
            search,
        };
        model.reset();
        model
    }

    /// Creates a new model with default bits=16 and period=0x400.
    pub fn with_defaults(compress: bool, symbols: usize) -> Self {
        Self::new(compress, symbols, 16, 0x400)
    }

    pub fn symbols(&self) -> usize {
        self.symbols
    }

    /// Reinitializes the model to uniform distribution.
    pub fn reset(&mut self) {
        let n = self.symbols;
        self.rescale = (n as i32 >> 4) | 2;
        self.more = 0;

        let total_freq = self.cumf[n];
        let f = total_freq / n as u32;
        let m = total_freq % n as u32;

        for i in 0..m as usize {
            self.symf[i] = f + 1;
        }
        for i in m as usize..n {
            self.symf[i] = f;
        }

        self.update();
    }

    /// Gets the cumulative and individual frequencies for encoding symbol s.
    #[inline]
    pub fn encode(&mut self, s: u32) -> (u32, u32) {
        let cum_freq = self.cumf[s as usize];
        let freq = self.cumf[s as usize + 1] - cum_freq;
        self.update_symbol(s);
        (cum_freq, freq)
    }

    /// Returns the symbol corresponding to cumulative frequency l.
    /// Updates l to the cumulative frequency and returns (symbol, freq).
    pub fn decode(&mut self, l: &mut u32, r: &mut u32) -> u32 {
        let search = self.search.as_ref().unwrap();
        let i = (*l >> self.search_shift) as usize;
        let mut s = search[i];
        let mut h = search[i + 1] + 1;

        // Binary search
        while s + 1 < h {
            let m = (s + h) >> 1;
            if *l < self.cumf[m as usize] {
                h = m;
            } else {
                s = m;
            }
        }

        *l = self.cumf[s as usize];
        *r = self.cumf[s as usize + 1] - *l;
        self.update_symbol(s);

        s
    }

    /// Normalizes the range by shifting right by bits.
    #[inline]
    pub fn normalize(&self, r: &mut u32) {
        *r >>= self.bits;
    }

    /// Main update routine - rescales frequencies and rebuilds tables.
    fn update(&mut self) {
        if self.more > 0 {
            self.left = self.more;
            self.more = 0;
            self.incr += 1;
            return;
        }

        if self.rescale != self.target_rescale {
            self.rescale *= 2;
            if self.rescale > self.target_rescale {
                self.rescale = self.target_rescale;
            }
        }

        let n = self.symbols;
        let mut cf = self.cumf[n];
        let mut count = cf;

        for i in (0..n).rev() {
            let mut sf = self.symf[i];
            cf -= sf;
            self.cumf[i] = cf;
            sf = (sf >> 1) | 1; // halve with odd bit set
            count -= sf;
            self.symf[i] = sf;
        }

        self.incr = count / self.rescale as u32;
        self.more = (count % self.rescale as u32) as i32;
        self.left = self.rescale - self.more;

        // Build lookup table
        if let Some(ref mut search) = self.search {
            let mut h = 1i32 << TABLE_SHIFT;
            for i in (0..n).rev() {
                let new_h = (self.cumf[i] >> self.search_shift) as i32;
                for l in new_h..=h {
                    search[l as usize] = i as u32;
                }
                h = new_h;
            }
        }
    }

    /// Updates frequency for a single symbol.
    #[inline]
    fn update_symbol(&mut self, s: u32) {
        if self.left == 0 {
            self.update();
        }
        self.left -= 1;
        self.symf[s as usize] += self.incr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_default_params() {
        let m = RCQsModel::with_defaults(true, 65);
        assert_eq!(m.symbols(), 65);
    }

    #[test]
    #[should_panic(expected = "bits must be <= 16")]
    fn bits_too_large() {
        RCQsModel::new(true, 10, 17, 0x400);
    }

    #[test]
    #[should_panic(expected = "period too large")]
    fn period_too_large() {
        RCQsModel::new(true, 10, 16, 1 << 17);
    }

    #[test]
    fn compress_mode_no_search_table() {
        let m = RCQsModel::new(true, 10, 16, 0x400);
        assert!(m.search.is_none());
    }

    #[test]
    fn decompress_mode_has_search_table() {
        let m = RCQsModel::new(false, 10, 16, 0x400);
        assert!(m.search.is_some());
    }

    #[test]
    fn encode_returns_valid_frequencies() {
        let mut m = RCQsModel::with_defaults(true, 65);
        for s in 0..65u32 {
            let (cum, freq) = m.encode(s);
            assert!(freq > 0, "freq must be > 0 for symbol {s}");
            assert!(
                cum + freq <= (1 << 16),
                "cumulative overflow for symbol {s}"
            );
        }
    }

    #[test]
    fn reset_restores_uniform() {
        let mut m = RCQsModel::with_defaults(true, 10);
        // Encode some symbols to change distribution
        for _ in 0..100 {
            m.encode(0);
        }
        m.reset();
        // After reset, frequencies should be roughly uniform
        let (_, f0) = m.encode(0);
        let (_, f5) = m.encode(5);
        // They should be close after reset
        let diff = (f0 as i64 - f5 as i64).unsigned_abs();
        assert!(diff <= 2, "frequencies should be roughly equal after reset");
    }
}
