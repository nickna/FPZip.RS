/// Maps a float to a uint preserving ordering, at the given bit precision.
#[inline(always)]
pub fn forward_f32(d: f32, bits: u32) -> u32 {
    let shift = 32 - bits;
    let mut r = d.to_bits();
    r = !r;
    r >>= shift;
    r ^= (r >> (bits - 1)).wrapping_neg() >> (shift + 1);
    r
}

/// Maps a uint back to a float (inverse of forward_f32).
#[inline(always)]
pub fn inverse_f32(mut r: u32, bits: u32) -> f32 {
    let shift = 32 - bits;
    r ^= (r >> (bits - 1)).wrapping_neg() >> (shift + 1);
    r = !r;
    r <<= shift;
    f32::from_bits(r)
}

/// Reduces a float to its identity at the given precision (lossy round-trip).
#[inline(always)]
pub fn identity_f32(d: f32, bits: u32) -> f32 {
    let shift = 32 - bits;
    let mut r = d.to_bits();
    r >>= shift;
    r <<= shift;
    f32::from_bits(r)
}

/// Maps a double to a ulong preserving ordering, at the given bit precision.
#[inline(always)]
pub fn forward_f64(d: f64, bits: u32) -> u64 {
    let shift = 64 - bits;
    let mut r = d.to_bits();
    r = !r;
    r >>= shift;
    r ^= (r >> (bits - 1)).wrapping_neg() >> (shift + 1);
    r
}

/// Maps a ulong back to a double (inverse of forward_f64).
#[inline(always)]
pub fn inverse_f64(mut r: u64, bits: u32) -> f64 {
    let shift = 64 - bits;
    r ^= (r >> (bits - 1)).wrapping_neg() >> (shift + 1);
    r = !r;
    r <<= shift;
    f64::from_bits(r)
}

/// Reduces a double to its identity at the given precision (lossy round-trip).
#[inline(always)]
pub fn identity_f64(d: f64, bits: u32) -> f64 {
    let shift = 64 - bits;
    let mut r = d.to_bits();
    r >>= shift;
    r <<= shift;
    f64::from_bits(r)
}

/// Masks a u32 value to the given bit precision (identity map for INT mode).
#[inline(always)]
pub fn mask_u32(v: u32, bits: u32) -> u32 {
    v & (u32::MAX >> (32 - bits))
}

/// Masks a u64 value to the given bit precision (identity map for INT mode).
#[inline(always)]
pub fn mask_u64(v: u64, bits: u32) -> u64 {
    v & (u64::MAX >> (64 - bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_round_trip_full_precision() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1e10,
            -1e10,
            1e-10,
            -1e-10,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
        ];
        for &v in &values {
            let mapped = forward_f32(v, 32);
            let back = inverse_f32(mapped, 32);
            assert_eq!(v.to_bits(), back.to_bits(), "round-trip failed for {v}");
        }
    }

    #[test]
    fn double_round_trip_full_precision() {
        let values = [
            0.0f64,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1e100,
            -1e100,
            1e-100,
            -1e-100,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
        ];
        for &v in &values {
            let mapped = forward_f64(v, 64);
            let back = inverse_f64(mapped, 64);
            assert_eq!(v.to_bits(), back.to_bits(), "round-trip failed for {v}");
        }
    }

    #[test]
    fn float_ordering_preserved() {
        let values = [-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0];
        for i in 0..values.len() - 1 {
            let a = forward_f32(values[i], 32);
            let b = forward_f32(values[i + 1], 32);
            assert!(
                a < b,
                "ordering not preserved: f({}) = {} >= f({}) = {}",
                values[i],
                a,
                values[i + 1],
                b
            );
        }
    }

    #[test]
    fn double_ordering_preserved() {
        let values = [-1.0f64, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0];
        for i in 0..values.len() - 1 {
            let a = forward_f64(values[i], 64);
            let b = forward_f64(values[i + 1], 64);
            assert!(
                a < b,
                "ordering not preserved: f({}) = {} >= f({}) = {}",
                values[i],
                a,
                values[i + 1],
                b
            );
        }
    }

    #[test]
    fn special_values_float() {
        let nan = forward_f32(f32::NAN, 32);
        let back = inverse_f32(nan, 32);
        assert!(back.is_nan());

        assert_eq!(
            inverse_f32(forward_f32(f32::INFINITY, 32), 32).to_bits(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            inverse_f32(forward_f32(f32::NEG_INFINITY, 32), 32).to_bits(),
            f32::NEG_INFINITY.to_bits()
        );

        let neg_zero = -0.0f32;
        assert_eq!(
            inverse_f32(forward_f32(neg_zero, 32), 32).to_bits(),
            neg_zero.to_bits()
        );
    }

    #[test]
    fn special_values_double() {
        let nan = forward_f64(f64::NAN, 64);
        let back = inverse_f64(nan, 64);
        assert!(back.is_nan());

        assert_eq!(
            inverse_f64(forward_f64(f64::INFINITY, 64), 64).to_bits(),
            f64::INFINITY.to_bits()
        );
        assert_eq!(
            inverse_f64(forward_f64(f64::NEG_INFINITY, 64), 64).to_bits(),
            f64::NEG_INFINITY.to_bits()
        );

        let neg_zero = -0.0f64;
        assert_eq!(
            inverse_f64(forward_f64(neg_zero, 64), 64).to_bits(),
            neg_zero.to_bits()
        );
    }

    #[test]
    fn reduced_precision_float() {
        // At reduced precision, identity(forward(inverse(forward(v)))) should equal identity(v)
        for bits in [8, 16, 24] {
            let v = 3.14f32;
            let mapped = forward_f32(v, bits);
            let back = inverse_f32(mapped, bits);
            let id = identity_f32(v, bits);
            assert_eq!(back.to_bits(), id.to_bits(), "bits={bits}");
        }
    }

    #[test]
    fn mask_u32_works() {
        assert_eq!(mask_u32(0xFFFF_FFFF, 8), 0xFF);
        assert_eq!(mask_u32(0xFFFF_FFFF, 16), 0xFFFF);
        assert_eq!(mask_u32(0xFFFF_FFFF, 32), 0xFFFF_FFFF);
    }
}
