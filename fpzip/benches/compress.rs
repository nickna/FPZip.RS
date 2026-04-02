use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// --- Data generators ---

fn generate_trilinear_f32(nx: usize, ny: usize, nz: usize) -> Vec<f32> {
    let n = nx * ny * nz;
    let mut field = vec![0.0f32; n];
    let mut seed = 1u32;

    for i in 1..n {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFFFFFF;
        let val = (seed as f64) * f64::exp2(-31.0);
        let val = 2.0 * val - 1.0;
        let val = val * val * val;
        let val = val * val * val;
        field[i] = val as f32;
    }

    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx {
                field[x + nx * (y + ny * z)] += field[(x - 1) + nx * (y + ny * z)];
            }
        }
    }
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                field[x + nx * (y + ny * z)] += field[x + nx * ((y - 1) + ny * z)];
            }
        }
    }
    for z in 1..nz {
        for y in 0..ny {
            for x in 0..nx {
                field[x + nx * (y + ny * z)] += field[x + nx * (y + ny * (z - 1))];
            }
        }
    }
    field
}

fn generate_trilinear_f64(nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n = nx * ny * nz;
    let mut field = vec![0.0f64; n];
    let mut seed = 1u32;

    for i in 1..n {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFFFFFF;
        let val = (seed as f64) * f64::exp2(-31.0);
        let val = 2.0 * val - 1.0;
        let val = val * val * val;
        let val = val * val * val;
        field[i] = val;
    }

    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx {
                field[x + nx * (y + ny * z)] += field[(x - 1) + nx * (y + ny * z)];
            }
        }
    }
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                field[x + nx * (y + ny * z)] += field[x + nx * ((y - 1) + ny * z)];
            }
        }
    }
    for z in 1..nz {
        for y in 0..ny {
            for x in 0..nx {
                field[x + nx * (y + ny * z)] += field[x + nx * (y + ny * (z - 1))];
            }
        }
    }
    field
}

fn generate_zeros_f32(n: usize) -> Vec<f32> {
    vec![0.0f32; n]
}

fn generate_gradient_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.001).collect()
}

fn generate_sine_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.01).sin() * 100.0).collect()
}

fn generate_random_f32(n: usize) -> Vec<f32> {
    let mut seed = 42u64;
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 * 1000.0 - 500.0) as f32
        })
        .collect()
}

// --- Benchmarks ---

fn bench_compress_f32_patterns(c: &mut Criterion) {
    let n = 65 * 64 * 63; // 262,080 elements
    let (nx, ny, nz) = (65u32, 64, 63);

    let mut group = c.benchmark_group("compress_f32");
    group.throughput(Throughput::Elements(n as u64));

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("trilinear", generate_trilinear_f32(65, 64, 63)),
        ("zeros", generate_zeros_f32(n)),
        ("gradient", generate_gradient_f32(n)),
        ("sine", generate_sine_f32(n)),
        ("random", generate_random_f32(n)),
    ];

    for (name, data) in &cases {
        group.bench_with_input(BenchmarkId::new("pattern", *name), data, |b, data| {
            b.iter(|| fpzip_rs::compress_f32(data, nx, ny, nz, 1).unwrap())
        });
    }
    group.finish();
}

fn bench_decompress_f32_patterns(c: &mut Criterion) {
    let n = 65 * 64 * 63;
    let (nx, ny, nz) = (65u32, 64, 63);

    let mut group = c.benchmark_group("decompress_f32");
    group.throughput(Throughput::Elements(n as u64));

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("trilinear", generate_trilinear_f32(65, 64, 63)),
        ("zeros", generate_zeros_f32(n)),
        ("gradient", generate_gradient_f32(n)),
        ("sine", generate_sine_f32(n)),
        ("random", generate_random_f32(n)),
    ];

    for (name, data) in &cases {
        let compressed = fpzip_rs::compress_f32(data, nx, ny, nz, 1).unwrap();
        group.bench_with_input(
            BenchmarkId::new("pattern", *name),
            &compressed,
            |b, compressed| b.iter(|| fpzip_rs::decompress_f32(compressed).unwrap()),
        );
    }
    group.finish();
}

fn bench_compress_f64(c: &mut Criterion) {
    let (nx, ny, nz) = (65u32, 64, 63);
    let data = generate_trilinear_f64(65, 64, 63);

    let mut group = c.benchmark_group("compress_f64");
    group.throughput(Throughput::Elements(data.len() as u64));

    group.bench_function("trilinear_65x64x63", |b| {
        b.iter(|| fpzip_rs::compress_f64(&data, nx, ny, nz, 1).unwrap())
    });
    group.finish();
}

fn bench_decompress_f64(c: &mut Criterion) {
    let (nx, ny, nz) = (65u32, 64, 63);
    let data = generate_trilinear_f64(65, 64, 63);
    let compressed = fpzip_rs::compress_f64(&data, nx, ny, nz, 1).unwrap();

    let mut group = c.benchmark_group("decompress_f64");
    group.throughput(Throughput::Elements(data.len() as u64));

    group.bench_function("trilinear_65x64x63", |b| {
        b.iter(|| fpzip_rs::decompress_f64(&compressed).unwrap())
    });
    group.finish();
}

fn bench_compress_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_f32_size");

    for &(name, nx, ny, nz) in &[
        ("1k_1d", 1000, 1, 1),
        ("10k_3d", 22, 22, 22),
        ("262k_3d", 65, 64, 63),
        ("1M_3d", 100, 100, 100),
    ] {
        let n = (nx * ny * nz) as usize;
        let data = generate_trilinear_f32(nx as usize, ny as usize, nz as usize);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("trilinear", name), &data, |b, data| {
            b.iter(|| fpzip_rs::compress_f32(data, nx, ny, nz, 1).unwrap())
        });
    }
    group.finish();
}

fn bench_precision(c: &mut Criterion) {
    let (nx, ny, nz) = (65u32, 64, 63);
    let data = generate_trilinear_f32(65, 64, 63);

    let mut group = c.benchmark_group("compress_f32_precision");
    group.throughput(Throughput::Elements(data.len() as u64));

    for &prec in &[8, 16, 24, 32] {
        let comp = fpzip_rs::FpZipCompressor::new(nx).ny(ny).nz(nz).prec(prec);
        group.bench_with_input(BenchmarkId::new("prec", prec), &data, |b, data| {
            b.iter(|| comp.compress_f32(data).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compress_f32_patterns,
    bench_decompress_f32_patterns,
    bench_compress_f64,
    bench_decompress_f64,
    bench_compress_sizes,
    bench_precision,
);
criterion_main!(benches);
