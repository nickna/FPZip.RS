use crate::codec::range_encoder::RangeEncoder;
use crate::codec::rc_qs_model::RCQsModel;
use crate::core::pc_map;

/// PC_BIT_MAX threshold: bits <= 8 use small alphabet, bits > 8 use wide alphabet.
const PC_BIT_MAX: u32 = 8;

/// Returns the number of symbols for the given bit width.
pub fn symbol_count(bits: u32) -> usize {
    if bits > PC_BIT_MAX {
        // Wide alphabet: 2 * bits + 1
        (2 * bits + 1) as usize
    } else {
        // Small alphabet: 2 * (1 << bits) - 1
        (2 * (1u32 << bits) - 1) as usize
    }
}

/// Predictive coder encoder for float values.
pub struct PCEncoderFloat<'a> {
    encoder: &'a mut RangeEncoder,
    model: &'a mut RCQsModel,
    bits: u32,
}

impl<'a> PCEncoderFloat<'a> {
    pub fn new(encoder: &'a mut RangeEncoder, model: &'a mut RCQsModel, bits: u32) -> Self {
        Self {
            encoder,
            model,
            bits,
        }
    }

    /// Encodes a mapped value with prediction. Returns the actual value (masked).
    #[inline]
    pub fn encode(&mut self, actual: u32, predicted: u32) -> u32 {
        let actual = pc_map::mask_u32(actual, self.bits);
        let predicted = pc_map::mask_u32(predicted, self.bits);

        if self.bits > PC_BIT_MAX {
            self.encode_wide(actual, predicted);
        } else {
            self.encode_narrow(actual, predicted);
        }
        actual
    }

    #[inline]
    fn encode_wide(&mut self, actual: u32, predicted: u32) {
        let bias = self.bits;
        if predicted < actual {
            let d = actual - predicted;
            let k = 31 - d.leading_zeros();
            self.encoder.encode_with_model(bias + 1 + k, self.model);
            self.encoder.encode_uint(d - (1u32 << k), k as i32);
        } else if predicted > actual {
            let d = predicted - actual;
            let k = 31 - d.leading_zeros();
            self.encoder.encode_with_model(bias - 1 - k, self.model);
            self.encoder.encode_uint(d - (1u32 << k), k as i32);
        } else {
            self.encoder.encode_with_model(bias, self.model);
        }
    }

    #[inline]
    fn encode_narrow(&mut self, actual: u32, predicted: u32) {
        let bias = (1u32 << self.bits) - 1;
        let symbol = bias.wrapping_add(actual).wrapping_sub(predicted);
        self.encoder.encode_with_model(symbol, self.model);
    }
}

/// Predictive coder encoder for double values.
pub struct PCEncoderDouble<'a> {
    encoder: &'a mut RangeEncoder,
    model: &'a mut RCQsModel,
    bits: u32,
}

impl<'a> PCEncoderDouble<'a> {
    pub fn new(encoder: &'a mut RangeEncoder, model: &'a mut RCQsModel, bits: u32) -> Self {
        Self {
            encoder,
            model,
            bits,
        }
    }

    /// Encodes a mapped value with prediction. Returns the actual value (masked).
    #[inline]
    pub fn encode(&mut self, actual: u64, predicted: u64) -> u64 {
        let actual = pc_map::mask_u64(actual, self.bits);
        let predicted = pc_map::mask_u64(predicted, self.bits);

        if self.bits > PC_BIT_MAX {
            self.encode_wide(actual, predicted);
        } else {
            self.encode_narrow(actual, predicted);
        }
        actual
    }

    #[inline]
    fn encode_wide(&mut self, actual: u64, predicted: u64) {
        let bias = self.bits;
        if predicted < actual {
            let d = actual - predicted;
            let k = 63 - d.leading_zeros();
            self.encoder.encode_with_model(bias + 1 + k, self.model);
            self.encoder.encode_ulong(d - (1u64 << k), k as i32);
        } else if predicted > actual {
            let d = predicted - actual;
            let k = 63 - d.leading_zeros();
            self.encoder.encode_with_model(bias - 1 - k, self.model);
            self.encoder.encode_ulong(d - (1u64 << k), k as i32);
        } else {
            self.encoder.encode_with_model(bias, self.model);
        }
    }

    #[inline]
    fn encode_narrow(&mut self, actual: u64, predicted: u64) {
        let bias = (1u64 << self.bits) - 1;
        let symbol = bias.wrapping_add(actual).wrapping_sub(predicted);
        self.encoder.encode_with_model(symbol as u32, self.model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::range_decoder::RangeDecoder;
    use crate::core::pc_decoder::{PCDecoderDouble, PCDecoderFloat};

    fn round_trip_float(actual: u32, predicted: u32, bits: u32) -> u32 {
        let symbols = symbol_count(bits);
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, symbols);
        {
            let mut pc = PCEncoderFloat::new(&mut enc, &mut model, bits);
            pc.encode(actual, predicted);
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, symbols);
        let mut pcd = PCDecoderFloat::new(&mut dec, &mut dmodel, bits);
        pcd.decode(predicted)
    }

    fn round_trip_double(actual: u64, predicted: u64, bits: u32) -> u64 {
        let symbols = symbol_count(bits);
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, symbols);
        {
            let mut pc = PCEncoderDouble::new(&mut enc, &mut model, bits);
            pc.encode(actual, predicted);
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, symbols);
        let mut pcd = PCDecoderDouble::new(&mut dec, &mut dmodel, bits);
        pcd.decode(predicted)
    }

    #[test]
    fn float_perfect_prediction() {
        assert_eq!(round_trip_float(100, 100, 32), 100);
    }

    #[test]
    fn float_underprediction() {
        assert_eq!(round_trip_float(200, 100, 32), 200);
    }

    #[test]
    fn float_overprediction() {
        assert_eq!(round_trip_float(50, 200, 32), 50);
    }

    #[test]
    fn float_all_delta_sizes() {
        let predicted = 0u32;
        for k in 0..31 {
            let delta = 1u32 << k;
            let actual = predicted.wrapping_add(delta);
            assert_eq!(
                round_trip_float(actual, predicted, 32),
                actual,
                "k={k} under"
            );
            let actual2 = predicted.wrapping_sub(delta);
            assert_eq!(
                round_trip_float(actual2, predicted, 32),
                actual2,
                "k={k} over"
            );
        }
    }

    #[test]
    fn double_perfect_prediction() {
        assert_eq!(round_trip_double(100, 100, 64), 100);
    }

    #[test]
    fn double_underprediction() {
        assert_eq!(round_trip_double(200, 100, 64), 200);
    }

    #[test]
    fn double_overprediction() {
        assert_eq!(round_trip_double(50, 200, 64), 50);
    }

    #[test]
    fn float_sequence() {
        let symbols = symbol_count(32);
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, symbols);
        let pairs: Vec<(u32, u32)> =
            vec![(100, 100), (200, 100), (50, 200), (0, 0), (0xFFFFFFFF, 0)];
        {
            let mut pc = PCEncoderFloat::new(&mut enc, &mut model, 32);
            for &(a, p) in &pairs {
                pc.encode(a, p);
            }
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, symbols);
        let mut pcd = PCDecoderFloat::new(&mut dec, &mut dmodel, 32);
        for &(a, p) in &pairs {
            assert_eq!(pcd.decode(p), a);
        }
    }

    #[test]
    fn double_all_delta_sizes() {
        let predicted = 0u64;
        for k in 0..63 {
            let delta = 1u64 << k;
            let actual = predicted.wrapping_add(delta);
            assert_eq!(
                round_trip_double(actual, predicted, 64),
                actual,
                "k={k} under"
            );
        }
    }

    // Small alphabet tests (bits <= 8)
    #[test]
    fn float_narrow_round_trip() {
        for bits in [2, 4, 8] {
            let mask = (1u32 << bits) - 1;
            for a in 0..=mask.min(15) {
                for p in 0..=mask.min(15) {
                    let result = round_trip_float(a, p, bits);
                    assert_eq!(result, a, "bits={bits} a={a} p={p}");
                }
            }
        }
    }

    #[test]
    fn float_reduced_precision_wide() {
        // bits=16 uses wide alphabet
        let mask = 0xFFFFu32;
        assert_eq!(round_trip_float(0, 0, 16), 0);
        assert_eq!(round_trip_float(mask, 0, 16), mask);
        assert_eq!(round_trip_float(100, 200, 16), 100);
        assert_eq!(round_trip_float(200, 100, 16), 200);
    }
}
