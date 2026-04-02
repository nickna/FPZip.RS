//! SIMD-accelerated batch PCMap operations.
//!
//! Converts arrays of `f32`/`f64` to their integer-mapped representations
//! in bulk using SIMD intrinsics. On x86/x86_64, this uses SSE2 to process
//! 4 floats or 2 doubles at a time. Falls back to scalar code on other
//! architectures or when the `simd` feature is disabled.
//!
//! These are useful for pre-processing data before compression or
//! post-processing after decompression, though the main compression loop
//! itself is inherently sequential (each element depends on previously
//! encoded neighbors).

use super::pc_map;

/// Batch-converts an array of `f32` values to their integer-mapped `u32` representations.
///
/// This is equivalent to calling [`pc_map::forward_f32`] on each element, but uses
/// SIMD intrinsics when available for higher throughput.
///
/// # Arguments
/// * `input` - Source float array.
/// * `output` - Destination u32 array. Must be at least as long as `input`.
/// * `bits` - Bit precision (32 for lossless).
pub fn forward_batch_f32(input: &[f32], output: &mut [u32], bits: u32) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if bits == 32 {
            forward_batch_f32_sse2_full(input, output);
            return;
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if bits == 32 {
            forward_batch_f32_sse2_full(input, output);
            return;
        }
    }

    // Scalar fallback
    for (i, &d) in input.iter().enumerate() {
        output[i] = pc_map::forward_f32(d, bits);
    }
}

/// Batch-converts an array of integer-mapped `u32` values back to `f32`.
///
/// Inverse of [`forward_batch_f32`].
pub fn inverse_batch_f32(input: &[u32], output: &mut [f32], bits: u32) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if bits == 32 {
            inverse_batch_f32_sse2_full(input, output);
            return;
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if bits == 32 {
            inverse_batch_f32_sse2_full(input, output);
            return;
        }
    }

    for (i, &r) in input.iter().enumerate() {
        output[i] = pc_map::inverse_f32(r, bits);
    }
}

/// Batch-converts an array of `f64` values to their integer-mapped `u64` representations.
pub fn forward_batch_f64(input: &[f64], output: &mut [u64], bits: u32) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    // SSE2 handles 2 doubles at a time (128-bit / 64-bit), but the bit manipulation
    // requires integer ops. The gain is marginal for doubles so we use scalar.
    for (i, &d) in input.iter().enumerate() {
        output[i] = pc_map::forward_f64(d, bits);
    }
}

/// Batch-converts an array of integer-mapped `u64` values back to `f64`.
pub fn inverse_batch_f64(input: &[u64], output: &mut [f64], bits: u32) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    for (i, &r) in input.iter().enumerate() {
        output[i] = pc_map::inverse_f64(r, bits);
    }
}

// --- SSE2 implementation for f32 at full precision (bits=32, shift=0) ---

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn forward_batch_f32_sse2_full(input: &[f32], output: &mut [u32]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Safety: SSE2 is guaranteed on x86_64. We reinterpret f32/u32 slices
    // as __m128i through pointer casts, respecting alignment and bounds.
    unsafe {
        let ones = _mm_set1_epi32(-1i32);
        let zero = _mm_setzero_si128();

        for i in 0..chunks {
            let offset = i * 4;
            let r = _mm_loadu_si128(input[offset..].as_ptr() as *const __m128i);

            // r = ~r
            let r = _mm_xor_si128(r, ones);

            // r ^= -(r >> 31) >> 1
            // (r >> 31) is 0 or 1 per lane. -(0 or 1) is 0 or 0xFFFFFFFF.
            // Then >> 1 gives 0 or 0x7FFFFFFF.
            let sign = _mm_srli_epi32(r, 31); // 0 or 1
            let neg_sign = _mm_sub_epi32(zero, sign); // 0 or 0xFFFFFFFF
            let xor_mask = _mm_srli_epi32(neg_sign, 1); // 0 or 0x7FFFFFFF

            let r = _mm_xor_si128(r, xor_mask);

            _mm_storeu_si128(output[offset..].as_mut_ptr() as *mut __m128i, r);
        }
    }

    let scalar_start = chunks * 4;
    for i in 0..remainder {
        output[scalar_start + i] = pc_map::forward_f32(input[scalar_start + i], 32);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn inverse_batch_f32_sse2_full(input: &[u32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let ones = _mm_set1_epi32(-1i32);
        let zero = _mm_setzero_si128();

        for i in 0..chunks {
            let offset = i * 4;
            let r = _mm_loadu_si128(input[offset..].as_ptr() as *const __m128i);

            // r ^= -(r >> 31) >> 1
            let sign = _mm_srli_epi32(r, 31);
            let neg_sign = _mm_sub_epi32(zero, sign);
            let xor_mask = _mm_srli_epi32(neg_sign, 1);
            let r = _mm_xor_si128(r, xor_mask);

            // r = ~r
            let r = _mm_xor_si128(r, ones);

            _mm_storeu_si128(output[offset..].as_mut_ptr() as *mut __m128i, r);
        }
    }

    let scalar_start = chunks * 4;
    for i in 0..remainder {
        output[scalar_start + i] = pc_map::inverse_f32(input[scalar_start + i], 32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_batch_f32_matches_scalar() {
        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.7).collect();
        let mut simd_out = vec![0u32; input.len()];
        let mut scalar_out = vec![0u32; input.len()];

        forward_batch_f32(&input, &mut simd_out, 32);
        for (i, &d) in input.iter().enumerate() {
            scalar_out[i] = pc_map::forward_f32(d, 32);
        }
        assert_eq!(simd_out, scalar_out);
    }

    #[test]
    fn inverse_batch_f32_matches_scalar() {
        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.7).collect();
        let mut mapped = vec![0u32; input.len()];
        for (i, &d) in input.iter().enumerate() {
            mapped[i] = pc_map::forward_f32(d, 32);
        }

        let mut simd_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];

        inverse_batch_f32(&mapped, &mut simd_out, 32);
        for (i, &r) in mapped.iter().enumerate() {
            scalar_out[i] = pc_map::inverse_f32(r, 32);
        }
        for i in 0..input.len() {
            assert_eq!(
                simd_out[i].to_bits(),
                scalar_out[i].to_bits(),
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn forward_batch_f32_round_trip() {
        let input: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            42.5,
            -1e10,
            f32::EPSILON,
            f32::MAX,
            f32::MIN,
        ];
        let mut mapped = vec![0u32; input.len()];
        let mut output = vec![0.0f32; input.len()];

        forward_batch_f32(&input, &mut mapped, 32);
        inverse_batch_f32(&mapped, &mut output, 32);

        for i in 0..input.len() {
            if input[i].is_nan() {
                assert!(output[i].is_nan(), "expected NaN at index {i}");
            } else {
                assert_eq!(
                    input[i].to_bits(),
                    output[i].to_bits(),
                    "mismatch at index {i}: {} vs {}",
                    input[i],
                    output[i]
                );
            }
        }
    }

    #[test]
    fn forward_batch_f64_matches_scalar() {
        let input: Vec<f64> = (-50..50).map(|i| i as f64 * 0.7).collect();
        let mut batch_out = vec![0u64; input.len()];
        let mut scalar_out = vec![0u64; input.len()];

        forward_batch_f64(&input, &mut batch_out, 64);
        for (i, &d) in input.iter().enumerate() {
            scalar_out[i] = pc_map::forward_f64(d, 64);
        }
        assert_eq!(batch_out, scalar_out);
    }

    #[test]
    fn forward_batch_f32_non_aligned_length() {
        // Test with lengths that aren't multiples of 4
        for len in [1, 2, 3, 5, 7, 13, 17] {
            let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let mut simd_out = vec![0u32; len];
            let mut scalar_out = vec![0u32; len];

            forward_batch_f32(&input, &mut simd_out, 32);
            for (i, &d) in input.iter().enumerate() {
                scalar_out[i] = pc_map::forward_f32(d, 32);
            }
            assert_eq!(simd_out, scalar_out, "failed for len={len}");
        }
    }

    #[test]
    fn forward_batch_f32_reduced_precision() {
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 1.5).collect();
        for bits in [8, 16, 24] {
            let mut batch_out = vec![0u32; input.len()];
            let mut scalar_out = vec![0u32; input.len()];

            forward_batch_f32(&input, &mut batch_out, bits);
            for (i, &d) in input.iter().enumerate() {
                scalar_out[i] = pc_map::forward_f32(d, bits);
            }
            assert_eq!(batch_out, scalar_out, "failed for bits={bits}");
        }
    }
}
