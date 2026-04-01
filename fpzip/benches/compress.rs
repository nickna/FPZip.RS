use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

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
                let idx = x + nx * (y + ny * z);
                let prev = (x - 1) + nx * (y + ny * z);
                field[idx] += field[prev];
            }
        }
    }
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let prev = x + nx * ((y - 1) + ny * z);
                field[idx] += field[prev];
            }
        }
    }
    for z in 1..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let prev = x + nx * (y + ny * (z - 1));
                field[idx] += field[prev];
            }
        }
    }
    field
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_f32");

    for &(name, nx, ny, nz) in &[
        ("small_1d", 1000, 1, 1),
        ("medium_3d", 65, 64, 63),
        ("large_3d", 100, 100, 100),
    ] {
        let data = generate_trilinear_f32(nx, ny, nz);
        group.bench_with_input(BenchmarkId::new("trilinear", name), &data, |b, data| {
            b.iter(|| fpzip_rs::compress_f32(data, nx as u32, ny as u32, nz as u32, 1).unwrap())
        });
    }
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_f32");

    for &(name, nx, ny, nz) in &[
        ("small_1d", 1000, 1, 1),
        ("medium_3d", 65, 64, 63),
        ("large_3d", 100, 100, 100),
    ] {
        let data = generate_trilinear_f32(nx, ny, nz);
        let compressed = fpzip_rs::compress_f32(&data, nx as u32, ny as u32, nz as u32, 1).unwrap();
        group.bench_with_input(
            BenchmarkId::new("trilinear", name),
            &compressed,
            |b, compressed| b.iter(|| fpzip_rs::decompress_f32(compressed).unwrap()),
        );
    }
    group.finish();
}

fn bench_round_trip(c: &mut Criterion) {
    let data = generate_trilinear_f32(65, 64, 63);
    c.bench_function("round_trip_65x64x63", |b| {
        b.iter(|| {
            let compressed = fpzip_rs::compress_f32(&data, 65, 64, 63, 1).unwrap();
            fpzip_rs::decompress_f32(&compressed).unwrap()
        })
    });
}

criterion_group!(benches, bench_compress, bench_decompress, bench_round_trip);
criterion_main!(benches);
