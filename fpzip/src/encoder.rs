use crate::codec::range_encoder::RangeEncoder;
use crate::codec::rc_qs_model::RCQsModel;
use crate::core::front::Front;
use crate::core::pc_encoder::{symbol_count, PCEncoderDouble, PCEncoderFloat};
use crate::core::pc_map;

#[cfg(feature = "rayon")]
extern crate alloc;
#[cfg(feature = "rayon")]
use alloc::vec::Vec;

/// Compresses a 3D float array at the given bit precision.
pub(crate) fn compress_3d_float(
    encoder: &mut RangeEncoder,
    data: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
    bits: u32,
) {
    let symbols = symbol_count(bits);
    let mut model = RCQsModel::with_defaults(true, symbols);
    let zero_mapped = pc_map::forward_f32(0.0f32, bits);
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

                let a = pc_map::forward_f32(data[data_index], bits);
                data_index += 1;

                let mut pc = PCEncoderFloat::new(encoder, &mut model, bits);
                let a = pc.encode(a, p);
                front.push(a);
            }
        }
    }
}

/// Compresses a 3D double array at the given bit precision.
pub(crate) fn compress_3d_double(
    encoder: &mut RangeEncoder,
    data: &[f64],
    nx: i32,
    ny: i32,
    nz: i32,
    bits: u32,
) {
    let symbols = symbol_count(bits);
    let mut model = RCQsModel::with_defaults(true, symbols);
    let zero_mapped = pc_map::forward_f64(0.0f64, bits);
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

                let a = pc_map::forward_f64(data[data_index], bits);
                data_index += 1;

                let mut pc = PCEncoderDouble::new(encoder, &mut model, bits);
                let a = pc.encode(a, p);
                front.push(a);
            }
        }
    }
}

/// Compresses a 4D float array (multiple fields).
pub(crate) fn compress_4d_float(
    encoder: &mut RangeEncoder,
    data: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
    bits: u32,
) {
    let field_size = (nx as usize) * (ny as usize) * (nz as usize);
    for f in 0..nf as usize {
        let start = f * field_size;
        let end = start + field_size;
        compress_3d_float(encoder, &data[start..end], nx, ny, nz, bits);
    }
}

/// Compresses a 4D double array (multiple fields).
pub(crate) fn compress_4d_double(
    encoder: &mut RangeEncoder,
    data: &[f64],
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
    bits: u32,
) {
    let field_size = (nx as usize) * (ny as usize) * (nz as usize);
    for f in 0..nf as usize {
        let start = f * field_size;
        let end = start + field_size;
        compress_3d_double(encoder, &data[start..end], nx, ny, nz, bits);
    }
}

#[allow(dead_code)]
/// Parallel compression of 4D float data (each field compressed independently).
///
/// **Note**: Produces different byte output than sequential mode because each field
/// starts with a fresh encoder state. The output is only decompressible by
/// `decompress_4d_float_parallel`.
#[cfg(feature = "rayon")]
pub(crate) fn compress_4d_float_parallel(
    data: &[f32],
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
    bits: u32,
) -> Vec<Vec<u8>> {
    use rayon::prelude::*;

    let field_size = (nx as usize) * (ny as usize) * (nz as usize);
    (0..nf as usize)
        .into_par_iter()
        .map(|f| {
            let start = f * field_size;
            let end = start + field_size;
            let mut encoder = RangeEncoder::new();
            compress_3d_float(&mut encoder, &data[start..end], nx, ny, nz, bits);
            encoder.finish()
        })
        .collect()
}

#[allow(dead_code)]
/// Parallel compression of 4D double data.
#[cfg(feature = "rayon")]
pub(crate) fn compress_4d_double_parallel(
    data: &[f64],
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
    bits: u32,
) -> Vec<Vec<u8>> {
    use rayon::prelude::*;

    let field_size = (nx as usize) * (ny as usize) * (nz as usize);
    (0..nf as usize)
        .into_par_iter()
        .map(|f| {
            let start = f * field_size;
            let end = start + field_size;
            let mut encoder = RangeEncoder::new();
            compress_3d_double(&mut encoder, &data[start..end], nx, ny, nz, bits);
            encoder.finish()
        })
        .collect()
}
