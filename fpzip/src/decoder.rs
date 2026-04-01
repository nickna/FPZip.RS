use crate::codec::range_decoder::RangeDecoder;
use crate::codec::rc_qs_model::RCQsModel;
use crate::core::front::Front;
use crate::core::pc_decoder::{PCDecoderDouble, PCDecoderFloat};
use crate::core::pc_encoder::{DOUBLE_SYMBOLS, FLOAT_SYMBOLS};
use crate::core::pc_map;

/// Decompresses a 3D float array.
pub(crate) fn decompress_3d_float(
    decoder: &mut RangeDecoder,
    data: &mut [f32],
    nx: i32,
    ny: i32,
    nz: i32,
) {
    let mut model = RCQsModel::with_defaults(false, FLOAT_SYMBOLS);
    let zero_mapped = pc_map::forward_f32(0.0f32);
    let mut front = Front::<u32>::new(nx, ny, zero_mapped);

    let mut data_index = 0;

    front.advance(0, 0, 1);
    for _ in 0..nz {
        front.advance(0, 1, 0);
        for _ in 0..ny {
            front.advance(1, 0, 0);
            for _ in 0..nx {
                let p = front
                    .get(1, 0, 0)
                    .wrapping_sub(front.get(0, 1, 1))
                    .wrapping_add(front.get(0, 1, 0))
                    .wrapping_sub(front.get(1, 0, 1))
                    .wrapping_add(front.get(0, 0, 1))
                    .wrapping_sub(front.get(1, 1, 0))
                    .wrapping_add(front.get(1, 1, 1));

                let mut pc = PCDecoderFloat::new(decoder, &mut model);
                let a = pc.decode(p);

                data[data_index] = pc_map::inverse_f32(a);
                data_index += 1;

                front.push(a);
            }
        }
    }
}

/// Decompresses a 3D double array.
pub(crate) fn decompress_3d_double(
    decoder: &mut RangeDecoder,
    data: &mut [f64],
    nx: i32,
    ny: i32,
    nz: i32,
) {
    let mut model = RCQsModel::with_defaults(false, DOUBLE_SYMBOLS);
    let zero_mapped = pc_map::forward_f64(0.0f64);
    let mut front = Front::<u64>::new(nx, ny, zero_mapped);

    let mut data_index = 0;

    front.advance(0, 0, 1);
    for _ in 0..nz {
        front.advance(0, 1, 0);
        for _ in 0..ny {
            front.advance(1, 0, 0);
            for _ in 0..nx {
                let p = front
                    .get(1, 0, 0)
                    .wrapping_sub(front.get(0, 1, 1))
                    .wrapping_add(front.get(0, 1, 0))
                    .wrapping_sub(front.get(1, 0, 1))
                    .wrapping_add(front.get(0, 0, 1))
                    .wrapping_sub(front.get(1, 1, 0))
                    .wrapping_add(front.get(1, 1, 1));

                let mut pc = PCDecoderDouble::new(decoder, &mut model);
                let a = pc.decode(p);

                data[data_index] = pc_map::inverse_f64(a);
                data_index += 1;

                front.push(a);
            }
        }
    }
}

/// Decompresses a 4D float array (multiple fields).
pub(crate) fn decompress_4d_float(
    decoder: &mut RangeDecoder,
    data: &mut [f32],
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
) {
    let field_size = (nx as usize) * (ny as usize) * (nz as usize);
    for f in 0..nf as usize {
        let start = f * field_size;
        let end = start + field_size;
        decompress_3d_float(decoder, &mut data[start..end], nx, ny, nz);
    }
}

/// Decompresses a 4D double array (multiple fields).
pub(crate) fn decompress_4d_double(
    decoder: &mut RangeDecoder,
    data: &mut [f64],
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
) {
    let field_size = (nx as usize) * (ny as usize) * (nz as usize);
    for f in 0..nf as usize {
        let start = f * field_size;
        let end = start + field_size;
        decompress_3d_double(decoder, &mut data[start..end], nx, ny, nz);
    }
}
