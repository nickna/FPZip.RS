#[allow(dead_code)]
mod test_helpers;

use test_helpers::*;

// Standard test dimensions from C++ reference (65x64x63)
const NX: u32 = 65;
const NY: u32 = 64;
const NZ: u32 = 63;

// Expected checksums for byte-identical C++ output (FPZIP_FP_INT lossless mode)
// From testfpzip.c cksum[3][2][0..2] indexed as [1D, 2D, 3D]
const EXPECTED_FLOAT_CHECKSUM_1D: u32 = 0x3e32e8c1;
const EXPECTED_FLOAT_CHECKSUM_2D: u32 = 0x8bb6d562;
const EXPECTED_FLOAT_CHECKSUM_3D: u32 = 0x5d710559;
const EXPECTED_DOUBLE_CHECKSUM_1D: u32 = 0x7cc58c60;
const EXPECTED_DOUBLE_CHECKSUM_2D: u32 = 0xc5f53ff4;
const EXPECTED_DOUBLE_CHECKSUM_3D: u32 = 0xbfc5a355;

// C++ baseline compression ratios (bits per value)
const CPP_FLOAT_TRILINEAR_BPV: f64 = 23.92;
const CPP_DOUBLE_TRILINEAR_BPV: f64 = 52.98;
const CPP_FLOAT_GRADIENT_BPV: f64 = 1.79;
const CPP_DOUBLE_GRADIENT_BPV: f64 = 2.15;
const CPP_FLOAT_SINE_BPV: f64 = 22.04;
const CPP_DOUBLE_SINE_BPV: f64 = 51.10;
const CPP_FLOAT_RANDOM_BPV: f64 = 31.67;
const CPP_DOUBLE_RANDOM_BPV: f64 = 62.18;

// 1% tolerance
const TOLERANCE: f64 = 1.01;

// --- Trilinear Field Tests ---

#[test]
fn float_trilinear_field_meets_cpp_baseline() {
    let field = generate_float_field(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let stats = compress_float_stats(&field, NX, NY, NZ, 1);

    // Verify round-trip
    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);

    let max_allowed = CPP_FLOAT_TRILINEAR_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Float trilinear: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_FLOAT_TRILINEAR_BPV
    );
}

#[test]
fn double_trilinear_field_meets_cpp_baseline() {
    let field = generate_double_field(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let stats = compress_double_stats(&field, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);

    let max_allowed = CPP_DOUBLE_TRILINEAR_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Double trilinear: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_DOUBLE_TRILINEAR_BPV
    );
}

// --- C++ byte-identical checksum tests (FPZIP_FP_INT lossless) ---
// The C++ test compresses the same field data with 3 dimension layouts:
//   1D: (nx*ny*nz, 1, 1)
//   2D: (nx, ny*nz, 1)
//   3D: (nx, ny, nz)

#[test]
fn float_trilinear_matches_cpp_checksum_1d() {
    let field = generate_float_field(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let stats = compress_float_stats(&field, NX * NY * NZ, 1, 1, 1);
    assert_eq!(EXPECTED_FLOAT_CHECKSUM_1D, jenkins_hash(&stats.compressed_data));
}

#[test]
fn float_trilinear_matches_cpp_checksum_2d() {
    let field = generate_float_field(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let stats = compress_float_stats(&field, NX, NY * NZ, 1, 1);
    assert_eq!(EXPECTED_FLOAT_CHECKSUM_2D, jenkins_hash(&stats.compressed_data));
}

#[test]
fn float_trilinear_matches_cpp_checksum_3d() {
    let field = generate_float_field(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let stats = compress_float_stats(&field, NX, NY, NZ, 1);
    assert_eq!(EXPECTED_FLOAT_CHECKSUM_3D, jenkins_hash(&stats.compressed_data));
}

#[test]
fn double_trilinear_matches_cpp_checksum_1d() {
    // C++ uses a static seed that persists across float_field() and double_field() calls.
    // The float test runs first, consuming PRNG state.
    let (_, seed) =
        generate_float_field_with_seed(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let field = generate_double_field(NX as usize, NY as usize, NZ as usize, 0.0, seed);
    let stats = compress_double_stats(&field, NX * NY * NZ, 1, 1, 1);
    assert_eq!(EXPECTED_DOUBLE_CHECKSUM_1D, jenkins_hash(&stats.compressed_data));
}

#[test]
fn double_trilinear_matches_cpp_checksum_2d() {
    let (_, seed) =
        generate_float_field_with_seed(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let field = generate_double_field(NX as usize, NY as usize, NZ as usize, 0.0, seed);
    let stats = compress_double_stats(&field, NX, NY * NZ, 1, 1);
    assert_eq!(EXPECTED_DOUBLE_CHECKSUM_2D, jenkins_hash(&stats.compressed_data));
}

#[test]
fn double_trilinear_matches_cpp_checksum_3d() {
    let (_, seed) =
        generate_float_field_with_seed(NX as usize, NY as usize, NZ as usize, 0.0, 1);
    let field = generate_double_field(NX as usize, NY as usize, NZ as usize, 0.0, seed);
    let stats = compress_double_stats(&field, NX, NY, NZ, 1);
    assert_eq!(EXPECTED_DOUBLE_CHECKSUM_3D, jenkins_hash(&stats.compressed_data));
}

// --- Data Pattern Tests - Float ---

#[test]
fn float_zeros_achieves_high_compression() {
    let data = vec![0.0f32; 1000];
    let stats = compress_float_stats(&data, 10, 10, 10, 1);

    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    assert!(
        stats.bits_per_value() < 1.0,
        "Zeros should compress to < 1 bit/value, got {:.2}",
        stats.bits_per_value()
    );
}

#[test]
fn float_constant_value_achieves_high_compression() {
    let data = vec![3.14159f32; 1000];
    let stats = compress_float_stats(&data, 10, 10, 10, 1);

    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    assert!(
        stats.compression_ratio() >= 4.0,
        "Constant data should achieve >= 4:1 ratio, got {:.2}",
        stats.compression_ratio()
    );
}

#[test]
fn float_linear_gradient_meets_cpp_baseline() {
    let data: Vec<f32> = (0..(NX * NY * NZ) as usize)
        .map(|i| i as f32 * 0.001)
        .collect();
    let stats = compress_float_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    let max_allowed = CPP_FLOAT_GRADIENT_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Float gradient: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_FLOAT_GRADIENT_BPV
    );
}

#[test]
fn float_random_meets_cpp_baseline() {
    // Use a simple deterministic RNG
    let mut rng_seed = 42u64;
    let n = (NX * NY * NZ) as usize;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            rng_seed = rng_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_seed >> 33) as f64 / (1u64 << 31) as f64 * 1000.0 - 500.0) as f32
        })
        .collect();
    let stats = compress_float_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    let max_allowed = CPP_FLOAT_RANDOM_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Float random: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_FLOAT_RANDOM_BPV
    );
}

#[test]
fn float_sine_wave_meets_cpp_baseline() {
    let data: Vec<f32> = (0..(NX * NY * NZ) as usize)
        .map(|i| (i as f32 * 0.01).sin() * 100.0)
        .collect();
    let stats = compress_float_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    let max_allowed = CPP_FLOAT_SINE_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Float sine wave: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_FLOAT_SINE_BPV
    );
}

// --- Data Pattern Tests - Double ---

#[test]
fn double_zeros_achieves_high_compression() {
    let data = vec![0.0f64; (NX * NY * NZ) as usize];
    let stats = compress_double_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    assert!(
        stats.bits_per_value() < 1.0,
        "Zeros should compress to < 1 bit/value, got {:.2}",
        stats.bits_per_value()
    );
}

#[test]
fn double_constant_value_achieves_high_compression() {
    let data = vec![3.14159265358979f64; (NX * NY * NZ) as usize];
    let stats = compress_double_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    assert!(
        stats.compression_ratio() >= 4.0,
        "Constant data should achieve >= 4:1 ratio, got {:.2}",
        stats.compression_ratio()
    );
}

#[test]
fn double_linear_gradient_meets_cpp_baseline() {
    let data: Vec<f64> = (0..(NX * NY * NZ) as usize)
        .map(|i| i as f64 * 0.001)
        .collect();
    let stats = compress_double_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    let max_allowed = CPP_DOUBLE_GRADIENT_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Double gradient: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_DOUBLE_GRADIENT_BPV
    );
}

#[test]
fn double_random_meets_cpp_baseline() {
    let mut rng_seed = 42u64;
    let n = (NX * NY * NZ) as usize;
    let data: Vec<f64> = (0..n)
        .map(|_| {
            rng_seed = rng_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_seed >> 33) as f64 / (1u64 << 31) as f64 * 1000.0 - 500.0
        })
        .collect();
    let stats = compress_double_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    let max_allowed = CPP_DOUBLE_RANDOM_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Double random: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_DOUBLE_RANDOM_BPV
    );
}

#[test]
fn double_sine_wave_meets_cpp_baseline() {
    let data: Vec<f64> = (0..(NX * NY * NZ) as usize)
        .map(|i| (i as f64 * 0.01).sin() * 100.0)
        .collect();
    let stats = compress_double_stats(&data, NX, NY, NZ, 1);

    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(data, decompressed);

    let max_allowed = CPP_DOUBLE_SINE_BPV * TOLERANCE;
    assert!(
        stats.bits_per_value() <= max_allowed,
        "Double sine wave: {:.2} bits/value exceeds C++ baseline ({:.2}) by more than 1%",
        stats.bits_per_value(),
        CPP_DOUBLE_SINE_BPV
    );
}

// --- Multi-dimensional Tests ---

#[test]
fn float_trilinear_1d_round_trip() {
    let field = generate_float_field(1000, 1, 1, 0.0, 1);
    let stats = compress_float_stats(&field, 1000, 1, 1, 1);
    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);
    assert!(stats.compression_ratio() > 1.0);
}

#[test]
fn float_trilinear_2d_round_trip() {
    let field = generate_float_field(100, 10, 1, 0.0, 1);
    let stats = compress_float_stats(&field, 100, 10, 1, 1);
    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);
    assert!(stats.compression_ratio() > 1.0);
}

#[test]
fn float_trilinear_3d_round_trip() {
    let field = generate_float_field(10, 10, 10, 0.0, 1);
    let stats = compress_float_stats(&field, 10, 10, 10, 1);
    let decompressed = fpzip_rs::decompress_f32(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);
    assert!(stats.compression_ratio() > 1.0);
}

#[test]
fn double_trilinear_1d_round_trip() {
    let field = generate_double_field(1000, 1, 1, 0.0, 1);
    let stats = compress_double_stats(&field, 1000, 1, 1, 1);
    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);
    assert!(stats.compression_ratio() > 1.0);
}

#[test]
fn double_trilinear_2d_round_trip() {
    let field = generate_double_field(100, 10, 1, 0.0, 1);
    let stats = compress_double_stats(&field, 100, 10, 1, 1);
    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);
    assert!(stats.compression_ratio() > 1.0);
}

#[test]
fn double_trilinear_3d_round_trip() {
    let field = generate_double_field(10, 10, 10, 0.0, 1);
    let stats = compress_double_stats(&field, 10, 10, 10, 1);
    let decompressed = fpzip_rs::decompress_f64(&stats.compressed_data).unwrap();
    assert_eq!(field, decompressed);
    assert!(stats.compression_ratio() > 1.0);
}
