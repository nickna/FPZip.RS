# fpzip-rs

Lossless compression for multi-dimensional floating-point arrays, implemented in pure Rust.

A faithful port of Peter Lindstrom's [FPZip](https://computing.llnl.gov/projects/fpzip) algorithm. Compresses `f32` and `f64` arrays with 1D, 2D, 3D, and 4D support. Designed for scientific and numerical data with high spatial correlation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- Lossless compression for `f32` and `f64` arrays
- Multi-dimensional support (1D, 2D, 3D, 4D)
- Pure Rust with no unsafe in the core library
- `no_std` compatible (with `alloc`)
- C FFI layer for calling from C/C++/Python/etc.
- Async support via tokio (`fpzip-async` crate)
- Optional parallel compression via rayon

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fpzip-rs = "0.1"
```

### Compress and decompress

```rust
use fpzip_rs::{compress_f32, decompress_f32};

// 10x10x10 grid of float data
let data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();

let compressed = compress_f32(&data, 10, 10, 10, 1).unwrap();
let decompressed = decompress_f32(&compressed).unwrap();

assert_eq!(data, decompressed); // lossless!
```

### Builder API

```rust
use fpzip_rs::{FpZipCompressor, decompress_f64};

let data: Vec<f64> = vec![3.14; 64];

let compressed = FpZipCompressor::new(4)
    .ny(4)
    .nz(4)
    .compress_f64(&data)
    .unwrap();

let decompressed = decompress_f64(&compressed).unwrap();
assert_eq!(data, decompressed);
```

### Stream-based I/O

```rust
use fpzip_rs::{compress_f32_to_writer, decompress_f32_from_reader};
use std::io::Cursor;

let data = vec![1.0f32, 2.0, 3.0, 4.0];
let mut buf = Vec::new();
compress_f32_to_writer(&data, &mut buf, 4, 1, 1, 1).unwrap();

let mut reader = Cursor::new(&buf);
let (header, decompressed) = decompress_f32_from_reader(&mut reader).unwrap();
assert_eq!(data, decompressed);
```

### Pre-allocated buffers

```rust
use fpzip_rs::{compress_f32_into, decompress_f32_into, max_compressed_size, FpZipType};

let data = vec![0.0f32; 1000];
let max_size = max_compressed_size(1000, FpZipType::Float);
let mut buf = vec![0u8; max_size];

let written = compress_f32_into(&data, &mut buf, 10, 10, 10, 1).unwrap();
let compressed = &buf[..written];

let mut output = vec![0.0f32; 1000];
let header = decompress_f32_into(compressed, &mut output).unwrap();
```

## API Reference

### Free Functions

| Function | Description |
|----------|-------------|
| `compress_f32(data, nx, ny, nz, nf)` | Compress `&[f32]` to `Vec<u8>` |
| `compress_f64(data, nx, ny, nz, nf)` | Compress `&[f64]` to `Vec<u8>` |
| `decompress_f32(data)` | Decompress `&[u8]` to `Vec<f32>` |
| `decompress_f64(data)` | Decompress `&[u8]` to `Vec<f64>` |
| `compress_f32_into(data, buf, ...)` | Compress into pre-allocated `&mut [u8]` |
| `decompress_f32_into(data, buf)` | Decompress into pre-allocated `&mut [f32]` |
| `compress_f32_to_writer(data, w, ...)` | Compress to `impl Write` |
| `decompress_f32_from_reader(r)` | Decompress from `impl Read` |
| `read_header(data)` | Read header without decompressing |
| `max_compressed_size(count, type)` | Upper bound on compressed size |

Double variants (`f64`) are available for all functions.

### Builder

```rust
FpZipCompressor::new(nx)
    .ny(ny)    // default: 1
    .nz(nz)    // default: 1
    .nf(nf)    // default: 1
    .compress_f32(data)
```

## Workspace Crates

| Crate | Description |
|-------|-------------|
| `fpzip-rs` | Core compression library |
| `fpzip-ffi` | C FFI layer (`cdylib` + `staticlib`) with `fpzip.h` header |
| `fpzip-async` | Async wrappers using tokio `AsyncRead`/`AsyncWrite` |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | `std::io::Read/Write` streaming APIs |
| `alloc` | yes | `Vec`-based return types (implied by `std`) |
| `rayon` | no | Parallel 4D field compression |

## C FFI

The `fpzip-ffi` crate builds a C-compatible shared/static library. A header file is provided at `fpzip-ffi/include/fpzip.h`.

```c
#include "fpzip.h"

float data[1000] = { /* ... */ };
uint8_t buf[8192];
size_t compressed_len;

int rc = fpzip_compress_float(data, 1000, 10, 10, 10, 1, buf, sizeof(buf), &compressed_len);
if (rc != 0) {
    printf("Error: %s\n", fpzip_error_message(rc));
}
```

## Algorithm

FPZip combines three techniques:

1. **Lorenzo predictor** -- predicts each value from 7 neighbors in a 3D wavefront
2. **Integer mapping** -- bijectively maps IEEE 754 floats to unsigned integers preserving ordering
3. **Adaptive arithmetic coding** -- range coder with quasi-static probability model

The predictor formula in integer domain:

```
p = f[1,0,0] - f[0,1,1] + f[0,1,0] - f[1,0,1] + f[0,0,1] - f[1,1,0] + f[1,1,1]
```

Only the residual (actual - predicted) is entropy coded, achieving high compression on spatially correlated data.

## Compressed Format

24-byte header (little-endian):

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `0x007A7066` (`fpz\0`) |
| 4 | 2 | Version |
| 6 | 1 | Type (0=float, 1=double) |
| 7 | 1 | Reserved |
| 8 | 4 | nx |
| 12 | 4 | ny |
| 16 | 4 | nz |
| 20 | 4 | nf |

Followed by the arithmetic-coded compressed stream.

## License

MIT -- see [LICENSE](LICENSE).

## Acknowledgments

Based on [FPZip](https://computing.llnl.gov/projects/fpzip) by Peter Lindstrom, Lawrence Livermore National Laboratory.
