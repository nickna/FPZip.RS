use crate::codec::range_encoder::RangeEncoder;
use crate::codec::rc_qs_model::RCQsModel;

/// Number of symbols for float predictive coder (2 * 32 + 1 = 65).
pub const FLOAT_SYMBOLS: usize = 65;
/// Number of symbols for double predictive coder (2 * 64 + 1 = 129).
pub const DOUBLE_SYMBOLS: usize = 129;

const FLOAT_BIAS: u32 = 32;
const DOUBLE_BIAS: u32 = 64;

/// Predictive coder encoder for float values.
pub struct PCEncoderFloat<'a> {
    encoder: &'a mut RangeEncoder,
    model: &'a mut RCQsModel,
}

impl<'a> PCEncoderFloat<'a> {
    pub fn new(encoder: &'a mut RangeEncoder, model: &'a mut RCQsModel) -> Self {
        Self { encoder, model }
    }

    /// Encodes a mapped value with prediction. Returns the actual value.
    #[inline]
    pub fn encode(&mut self, actual: u32, predicted: u32) -> u32 {
        if predicted < actual {
            let d = actual - predicted;
            let k = bit_scan_reverse_u32(d);
            self.encoder
                .encode_with_model(FLOAT_BIAS + 1 + k, self.model);
            self.encoder.encode_uint(d - (1u32 << k), k as i32);
        } else if predicted > actual {
            let d = predicted - actual;
            let k = bit_scan_reverse_u32(d);
            self.encoder
                .encode_with_model(FLOAT_BIAS - 1 - k, self.model);
            self.encoder.encode_uint(d - (1u32 << k), k as i32);
        } else {
            self.encoder.encode_with_model(FLOAT_BIAS, self.model);
        }
        actual
    }
}

/// Predictive coder encoder for double values.
pub struct PCEncoderDouble<'a> {
    encoder: &'a mut RangeEncoder,
    model: &'a mut RCQsModel,
}

impl<'a> PCEncoderDouble<'a> {
    pub fn new(encoder: &'a mut RangeEncoder, model: &'a mut RCQsModel) -> Self {
        Self { encoder, model }
    }

    /// Encodes a mapped value with prediction. Returns the actual value.
    #[inline]
    pub fn encode(&mut self, actual: u64, predicted: u64) -> u64 {
        if predicted < actual {
            let d = actual - predicted;
            let k = bit_scan_reverse_u64(d);
            self.encoder
                .encode_with_model(DOUBLE_BIAS + 1 + k, self.model);
            self.encoder.encode_ulong(d - (1u64 << k), k as i32);
        } else if predicted > actual {
            let d = predicted - actual;
            let k = bit_scan_reverse_u64(d);
            self.encoder
                .encode_with_model(DOUBLE_BIAS - 1 - k, self.model);
            self.encoder.encode_ulong(d - (1u64 << k), k as i32);
        } else {
            self.encoder.encode_with_model(DOUBLE_BIAS, self.model);
        }
        actual
    }
}

/// Returns the position of the highest set bit (0-indexed from LSB).
#[inline(always)]
fn bit_scan_reverse_u32(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// Returns the position of the highest set bit (0-indexed from LSB).
#[inline(always)]
fn bit_scan_reverse_u64(x: u64) -> u32 {
    63 - x.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::range_decoder::RangeDecoder;
    use crate::core::pc_decoder::{PCDecoderDouble, PCDecoderFloat};

    fn round_trip_float(actual: u32, predicted: u32) -> u32 {
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, FLOAT_SYMBOLS);
        {
            let mut pc = PCEncoderFloat::new(&mut enc, &mut model);
            pc.encode(actual, predicted);
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, FLOAT_SYMBOLS);
        let mut pcd = PCDecoderFloat::new(&mut dec, &mut dmodel);
        pcd.decode(predicted)
    }

    fn round_trip_double(actual: u64, predicted: u64) -> u64 {
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, DOUBLE_SYMBOLS);
        {
            let mut pc = PCEncoderDouble::new(&mut enc, &mut model);
            pc.encode(actual, predicted);
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, DOUBLE_SYMBOLS);
        let mut pcd = PCDecoderDouble::new(&mut dec, &mut dmodel);
        pcd.decode(predicted)
    }

    #[test]
    fn float_perfect_prediction() {
        assert_eq!(round_trip_float(100, 100), 100);
    }

    #[test]
    fn float_underprediction() {
        assert_eq!(round_trip_float(200, 100), 200);
    }

    #[test]
    fn float_overprediction() {
        assert_eq!(round_trip_float(50, 200), 50);
    }

    #[test]
    fn float_all_delta_sizes() {
        let predicted = 0u32;
        for k in 0..31 {
            let delta = 1u32 << k;
            let actual = predicted.wrapping_add(delta);
            assert_eq!(round_trip_float(actual, predicted), actual, "k={k} under");
            let actual2 = predicted.wrapping_sub(delta);
            assert_eq!(round_trip_float(actual2, predicted), actual2, "k={k} over");
        }
    }

    #[test]
    fn double_perfect_prediction() {
        assert_eq!(round_trip_double(100, 100), 100);
    }

    #[test]
    fn double_underprediction() {
        assert_eq!(round_trip_double(200, 100), 200);
    }

    #[test]
    fn double_overprediction() {
        assert_eq!(round_trip_double(50, 200), 50);
    }

    #[test]
    fn float_sequence() {
        let mut enc = RangeEncoder::new();
        let mut model = RCQsModel::with_defaults(true, FLOAT_SYMBOLS);
        let pairs: Vec<(u32, u32)> =
            vec![(100, 100), (200, 100), (50, 200), (0, 0), (0xFFFFFFFF, 0)];
        {
            let mut pc = PCEncoderFloat::new(&mut enc, &mut model);
            for &(a, p) in &pairs {
                pc.encode(a, p);
            }
        }
        let data = enc.finish();

        let mut dec = RangeDecoder::new(&data);
        dec.init();
        let mut dmodel = RCQsModel::with_defaults(false, FLOAT_SYMBOLS);
        let mut pcd = PCDecoderFloat::new(&mut dec, &mut dmodel);
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
            assert_eq!(round_trip_double(actual, predicted), actual, "k={k} under");
        }
    }
}
