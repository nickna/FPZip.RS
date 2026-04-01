#[allow(dead_code)]
mod test_helpers;

use fpzip_rs::{
    compress_f32, compress_f64, decompress_f32, decompress_f64, read_header, FpZipCompressor,
    FpZipError, FpZipType,
};

#[test]
fn float_simple_array_round_trip() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let compressed = compress_f32(&original, 8, 1, 1, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn double_simple_array_round_trip() {
    let original = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let compressed = compress_f64(&original, 8, 1, 1, 1).unwrap();
    let decompressed = decompress_f64(&compressed).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn float_2d_array_round_trip() {
    let original: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let compressed = compress_f32(&original, 4, 4, 1, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn float_3d_array_round_trip() {
    let original: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let compressed = compress_f32(&original, 4, 4, 4, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn float_smooth_gradient_compresses() {
    let original: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
    let compressed = compress_f32(&original, 10, 10, 10, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn float_random_data_round_trip() {
    // Simple LCG for deterministic "random" data matching C# Random(42) is not possible,
    // so we use our own deterministic generator
    let mut data = vec![0.0f32; 1000];
    let mut seed = 42u64;
    for v in data.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((seed >> 33) as f64 / (1u64 << 31) as f64 * 1000.0 - 500.0) as f32;
    }
    let compressed = compress_f32(&data, 10, 10, 10, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(data, decompressed);
}

#[test]
fn float_special_values_round_trip() {
    let original = vec![
        0.0f32,
        -0.0f32,
        1.0,
        -1.0,
        f32::EPSILON,
        -f32::EPSILON,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let compressed = compress_f32(&original, original.len() as u32, 1, 1, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();

    assert_eq!(original.len(), decompressed.len());
    for i in 0..original.len() {
        if original[i].is_nan() {
            assert!(decompressed[i].is_nan());
        } else {
            assert_eq!(original[i].to_bits(), decompressed[i].to_bits());
        }
    }
}

#[test]
fn double_special_values_round_trip() {
    let original = vec![
        0.0f64,
        -0.0f64,
        1.0,
        -1.0,
        f64::EPSILON,
        -f64::EPSILON,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let compressed = compress_f64(&original, original.len() as u32, 1, 1, 1).unwrap();
    let decompressed = decompress_f64(&compressed).unwrap();

    assert_eq!(original.len(), decompressed.len());
    for i in 0..original.len() {
        if original[i].is_nan() {
            assert!(decompressed[i].is_nan());
        } else {
            assert_eq!(original[i].to_bits(), decompressed[i].to_bits());
        }
    }
}

#[test]
fn float_large_array_round_trip() {
    let (nx, ny, nz) = (100u32, 100, 10);
    let mut original = vec![0.0f32; (nx * ny * nz) as usize];
    for i in 0..original.len() {
        original[i] = (i as f32 * 0.01).sin() * 100.0;
    }
    let compressed = compress_f32(&original, nx, ny, nz, 1).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn header_reads_correct_dimensions() {
    let data = vec![1.0f32; 24];
    let compressed = compress_f32(&data, 2, 3, 4, 1).unwrap();
    let header = read_header(&compressed).unwrap();

    assert_eq!(header.data_type, FpZipType::Float);
    assert_eq!(header.nx, 2);
    assert_eq!(header.ny, 3);
    assert_eq!(header.nz, 4);
    assert_eq!(header.nf, 1);
    assert_eq!(header.total_elements(), 24);
}

#[test]
fn float_multiple_fields_round_trip() {
    let (nx, ny, nz, nf) = (4u32, 4, 4, 2);
    let total = (nx * ny * nz * nf) as usize;
    let original: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();

    let compressed = compress_f32(&original, nx, ny, nz, nf).unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);

    let header = read_header(&compressed).unwrap();
    assert_eq!(header.nf, nf);
}

#[test]
fn stream_compress_decompress_works() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = Vec::new();
    fpzip_rs::compress_f32_to_writer(&original, &mut output, 8, 1, 1, 1).unwrap();
    let decompressed = decompress_f32(&output).unwrap();
    assert_eq!(original, decompressed);
}

#[test]
fn invalid_dimensions_returns_error() {
    let data = vec![0.0f32; 10];
    let result = compress_f32(&data, 5, 5, 1, 1); // 5*5=25 != 10
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        FpZipError::DimensionMismatch { .. }
    ));
}

#[test]
fn wrong_type_returns_error() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0];
    let compressed = compress_f32(&original, 4, 1, 1, 1).unwrap();
    let result = decompress_f64(&compressed);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        FpZipError::TypeMismatch { .. }
    ));
}

#[test]
fn builder_api_works() {
    let original: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let compressed = FpZipCompressor::new(4)
        .ny(4)
        .nz(4)
        .compress_f32(&original)
        .unwrap();
    let decompressed = decompress_f32(&compressed).unwrap();
    assert_eq!(original, decompressed);
}
