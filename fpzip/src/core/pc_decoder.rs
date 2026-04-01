use crate::codec::range_decoder::RangeDecoder;
use crate::codec::rc_qs_model::RCQsModel;

const FLOAT_BIAS: u32 = 32;
const DOUBLE_BIAS: u32 = 64;

/// Predictive coder decoder for float values.
pub struct PCDecoderFloat<'a, 'b> {
    decoder: &'a mut RangeDecoder<'b>,
    model: &'a mut RCQsModel,
}

impl<'a, 'b> PCDecoderFloat<'a, 'b> {
    pub fn new(decoder: &'a mut RangeDecoder<'b>, model: &'a mut RCQsModel) -> Self {
        Self { decoder, model }
    }

    /// Decodes a value given a prediction.
    #[inline]
    pub fn decode(&mut self, predicted: u32) -> u32 {
        let s = self.decoder.decode_with_model(self.model);

        if s > FLOAT_BIAS {
            // Underprediction
            let k = (s - FLOAT_BIAS - 1) as i32;
            let d = (1u32 << k) + self.decoder.decode_uint(k);
            predicted.wrapping_add(d)
        } else if s < FLOAT_BIAS {
            // Overprediction
            let k = (FLOAT_BIAS - 1 - s) as i32;
            let d = (1u32 << k) + self.decoder.decode_uint(k);
            predicted.wrapping_sub(d)
        } else {
            // Perfect prediction
            predicted
        }
    }
}

/// Predictive coder decoder for double values.
pub struct PCDecoderDouble<'a, 'b> {
    decoder: &'a mut RangeDecoder<'b>,
    model: &'a mut RCQsModel,
}

impl<'a, 'b> PCDecoderDouble<'a, 'b> {
    pub fn new(decoder: &'a mut RangeDecoder<'b>, model: &'a mut RCQsModel) -> Self {
        Self { decoder, model }
    }

    /// Decodes a value given a prediction.
    #[inline]
    pub fn decode(&mut self, predicted: u64) -> u64 {
        let s = self.decoder.decode_with_model(self.model);

        if s > DOUBLE_BIAS {
            // Underprediction
            let k = (s - DOUBLE_BIAS - 1) as i32;
            let d = (1u64 << k) + self.decoder.decode_ulong(k);
            predicted.wrapping_add(d)
        } else if s < DOUBLE_BIAS {
            // Overprediction
            let k = (DOUBLE_BIAS - 1 - s) as i32;
            let d = (1u64 << k) + self.decoder.decode_ulong(k);
            predicted.wrapping_sub(d)
        } else {
            // Perfect prediction
            predicted
        }
    }
}
