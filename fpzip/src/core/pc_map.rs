/// Maps a float to a uint preserving ordering.
#[inline(always)]
pub fn forward_f32(d: f32) -> u32 {
    let mut r = d.to_bits();
    r = !r;
    r ^= (r >> 31).wrapping_neg() >> 1;
    r
}

/// Maps a uint back to a float (inverse of forward_f32).
#[inline(always)]
pub fn inverse_f32(mut r: u32) -> f32 {
    r ^= (r >> 31).wrapping_neg() >> 1;
    r = !r;
    f32::from_bits(r)
}

/// Maps a double to a ulong preserving ordering.
#[inline(always)]
pub fn forward_f64(d: f64) -> u64 {
    let mut r = d.to_bits();
    r = !r;
    r ^= (r >> 63).wrapping_neg() >> 1;
    r
}

/// Maps a ulong back to a double (inverse of forward_f64).
#[inline(always)]
pub fn inverse_f64(mut r: u64) -> f64 {
    r ^= (r >> 63).wrapping_neg() >> 1;
    r = !r;
    f64::from_bits(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_round_trip() {
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
            let mapped = forward_f32(v);
            let back = inverse_f32(mapped);
            assert_eq!(v.to_bits(), back.to_bits(), "round-trip failed for {v}");
        }
    }

    #[test]
    fn double_round_trip() {
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
            let mapped = forward_f64(v);
            let back = inverse_f64(mapped);
            assert_eq!(v.to_bits(), back.to_bits(), "round-trip failed for {v}");
        }
    }

    #[test]
    fn float_ordering_preserved() {
        let values = [-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0];
        for i in 0..values.len() - 1 {
            let a = forward_f32(values[i]);
            let b = forward_f32(values[i + 1]);
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
            let a = forward_f64(values[i]);
            let b = forward_f64(values[i + 1]);
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
        // NaN, Inf, -Inf, -0 should all round-trip
        let nan = forward_f32(f32::NAN);
        let back = inverse_f32(nan);
        assert!(back.is_nan());

        assert_eq!(
            inverse_f32(forward_f32(f32::INFINITY)).to_bits(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            inverse_f32(forward_f32(f32::NEG_INFINITY)).to_bits(),
            f32::NEG_INFINITY.to_bits()
        );

        let neg_zero = -0.0f32;
        assert_eq!(
            inverse_f32(forward_f32(neg_zero)).to_bits(),
            neg_zero.to_bits()
        );
    }

    #[test]
    fn special_values_double() {
        let nan = forward_f64(f64::NAN);
        let back = inverse_f64(nan);
        assert!(back.is_nan());

        assert_eq!(
            inverse_f64(forward_f64(f64::INFINITY)).to_bits(),
            f64::INFINITY.to_bits()
        );
        assert_eq!(
            inverse_f64(forward_f64(f64::NEG_INFINITY)).to_bits(),
            f64::NEG_INFINITY.to_bits()
        );

        let neg_zero = -0.0f64;
        assert_eq!(
            inverse_f64(forward_f64(neg_zero)).to_bits(),
            neg_zero.to_bits()
        );
    }
}
