use crate::codec::range_decoder::RangeDecoder;
use crate::codec::rc_qs_model::RCQsModel;
use crate::core::pc_map;

const PC_BIT_MAX: u32 = 8;

/// Predictive coder decoder for float values.
pub struct PCDecoderFloat<'a, 'b> {
    decoder: &'a mut RangeDecoder<'b>,
    model: &'a mut RCQsModel,
    bits: u32,
}

impl<'a, 'b> PCDecoderFloat<'a, 'b> {
    pub fn new(decoder: &'a mut RangeDecoder<'b>, model: &'a mut RCQsModel, bits: u32) -> Self {
        Self {
            decoder,
            model,
            bits,
        }
    }

    #[inline]
    pub fn decode(&mut self, predicted: u32) -> u32 {
        let predicted = pc_map::mask_u32(predicted, self.bits);
        if self.bits > PC_BIT_MAX {
            self.decode_wide(predicted)
        } else {
            self.decode_narrow(predicted)
        }
    }

    #[inline]
    fn decode_wide(&mut self, predicted: u32) -> u32 {
        let bias = self.bits;
        let s = self.decoder.decode_with_model(self.model);

        if s > bias {
            let k = (s - bias - 1) as i32;
            let d = (1u32 << k) + self.decoder.decode_uint(k);
            pc_map::mask_u32(predicted.wrapping_add(d), self.bits)
        } else if s < bias {
            let k = (bias - 1 - s) as i32;
            let d = (1u32 << k) + self.decoder.decode_uint(k);
            pc_map::mask_u32(predicted.wrapping_sub(d), self.bits)
        } else {
            predicted
        }
    }

    #[inline]
    fn decode_narrow(&mut self, predicted: u32) -> u32 {
        let bias = (1u32 << self.bits) - 1;
        let s = self.decoder.decode_with_model(self.model);
        let r = predicted.wrapping_add(s).wrapping_sub(bias);
        pc_map::mask_u32(r, self.bits)
    }
}

/// Predictive coder decoder for double values.
pub struct PCDecoderDouble<'a, 'b> {
    decoder: &'a mut RangeDecoder<'b>,
    model: &'a mut RCQsModel,
    bits: u32,
}

impl<'a, 'b> PCDecoderDouble<'a, 'b> {
    pub fn new(decoder: &'a mut RangeDecoder<'b>, model: &'a mut RCQsModel, bits: u32) -> Self {
        Self {
            decoder,
            model,
            bits,
        }
    }

    #[inline]
    pub fn decode(&mut self, predicted: u64) -> u64 {
        let predicted = pc_map::mask_u64(predicted, self.bits);
        if self.bits > PC_BIT_MAX {
            self.decode_wide(predicted)
        } else {
            self.decode_narrow(predicted)
        }
    }

    #[inline]
    fn decode_wide(&mut self, predicted: u64) -> u64 {
        let bias = self.bits;
        let s = self.decoder.decode_with_model(self.model);

        if s > bias {
            let k = (s - bias - 1) as i32;
            let d = (1u64 << k) + self.decoder.decode_ulong(k);
            pc_map::mask_u64(predicted.wrapping_add(d), self.bits)
        } else if s < bias {
            let k = (bias - 1 - s) as i32;
            let d = (1u64 << k) + self.decoder.decode_ulong(k);
            pc_map::mask_u64(predicted.wrapping_sub(d), self.bits)
        } else {
            predicted
        }
    }

    #[inline]
    fn decode_narrow(&mut self, predicted: u64) -> u64 {
        let bias = (1u64 << self.bits) - 1;
        let s = self.decoder.decode_with_model(self.model) as u64;
        let r = predicted.wrapping_add(s).wrapping_sub(bias);
        pc_map::mask_u64(r, self.bits)
    }
}
