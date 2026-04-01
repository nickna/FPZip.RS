/// LCG multiplier from C++ reference.
const MULTIPLIER: u32 = 1103515245;
/// LCG increment from C++ reference.
const INCREMENT: u32 = 12345;

/// Generates the next random double using the C++ LCG algorithm.
/// Range: [-1, 1], shaped by cubing twice (val^9).
fn next_double(seed: &mut u32) -> f64 {
    *seed = seed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
    *seed &= 0x7FFFFFFF;

    // Convert to [0, 1)
    let val = (*seed as f64) * f64::exp2(-31.0);

    // Convert to [-1, 1]
    let val = 2.0 * val - 1.0;

    // Shape distribution by cubing twice (val^9)
    let val = val * val * val; // val^3
    val * val * val // val^9
}

fn next_float(seed: &mut u32) -> f32 {
    next_double(seed) as f32
}

/// Generates a trilinear float field matching C++ float_field().
/// Returns (field, final_seed) so the seed state can be passed to the next generator
/// (matching C++'s static seed behavior).
pub fn generate_float_field_with_seed(
    nx: usize,
    ny: usize,
    nz: usize,
    offset: f32,
    seed: u32,
) -> (Vec<f32>, u32) {
    let n = nx * ny * nz;
    let mut field = vec![0.0f32; n];
    let mut seed = seed;

    field[0] = offset;
    for i in 1..n {
        field[i] = next_float(&mut seed);
    }

    // Integrate along X
    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx {
                let idx = x + nx * (y + ny * z);
                let prev = (x - 1) + nx * (y + ny * z);
                field[idx] += field[prev];
            }
        }
    }

    // Integrate along Y
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let prev = x + nx * ((y - 1) + ny * z);
                field[idx] += field[prev];
            }
        }
    }

    // Integrate along Z
    for z in 1..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let prev = x + nx * (y + ny * (z - 1));
                field[idx] += field[prev];
            }
        }
    }

    (field, seed)
}

pub fn generate_float_field(nx: usize, ny: usize, nz: usize, offset: f32, seed: u32) -> Vec<f32> {
    let n = nx * ny * nz;
    let mut field = vec![0.0f32; n];
    let mut seed = seed;

    field[0] = offset;
    for i in 1..n {
        field[i] = next_float(&mut seed);
    }

    // Integrate along X
    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx {
                let idx = x + nx * (y + ny * z);
                let prev = (x - 1) + nx * (y + ny * z);
                field[idx] += field[prev];
            }
        }
    }

    // Integrate along Y
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let prev = x + nx * ((y - 1) + ny * z);
                field[idx] += field[prev];
            }
        }
    }

    // Integrate along Z
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

/// Generates a trilinear double field matching C++ double_field().
pub fn generate_double_field(nx: usize, ny: usize, nz: usize, offset: f64, seed: u32) -> Vec<f64> {
    let n = nx * ny * nz;
    let mut field = vec![0.0f64; n];
    let mut seed = seed;

    field[0] = offset;
    for i in 1..n {
        field[i] = next_double(&mut seed);
    }

    // Integrate along X
    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx {
                let idx = x + nx * (y + ny * z);
                let prev = (x - 1) + nx * (y + ny * z);
                field[idx] += field[prev];
            }
        }
    }

    // Integrate along Y
    for z in 0..nz {
        for y in 1..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let prev = x + nx * ((y - 1) + ny * z);
                field[idx] += field[prev];
            }
        }
    }

    // Integrate along Z
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

pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub value_count: usize,
    pub compressed_data: Vec<u8>,
}

impl CompressionStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_bytes == 0 {
            f64::INFINITY
        } else {
            self.original_bytes as f64 / self.compressed_bytes as f64
        }
    }

    pub fn bits_per_value(&self) -> f64 {
        if self.value_count == 0 {
            0.0
        } else {
            (self.compressed_bytes as f64 * 8.0) / self.value_count as f64
        }
    }
}

pub fn compress_float_stats(data: &[f32], nx: u32, ny: u32, nz: u32, nf: u32) -> CompressionStats {
    let original_bytes = data.len() * 4;
    let compressed = fpzip_rs::compress_f32(data, nx, ny, nz, nf).unwrap();
    CompressionStats {
        original_bytes,
        compressed_bytes: compressed.len(),
        value_count: data.len(),
        compressed_data: compressed,
    }
}

pub fn compress_double_stats(data: &[f64], nx: u32, ny: u32, nz: u32, nf: u32) -> CompressionStats {
    let original_bytes = data.len() * 8;
    let compressed = fpzip_rs::compress_f64(data, nx, ny, nz, nf).unwrap();
    CompressionStats {
        original_bytes,
        compressed_bytes: compressed.len(),
        value_count: data.len(),
        compressed_data: compressed,
    }
}

/// Jenkins one-at-a-time hash (same as C++ reference).
pub fn jenkins_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in data {
        h = h.wrapping_add(b as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}
