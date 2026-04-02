//! # fpzip-rs
//!
//! Lossless and lossy compression for multi-dimensional floating-point arrays.
//!
//! A faithful Rust port of Peter Lindstrom's
//! [FPZip](https://computing.llnl.gov/projects/fpzip) algorithm, producing
//! byte-identical output with the C++ reference implementation. Compresses
//! `f32` and `f64` arrays with 1D, 2D, 3D, and 4D support.
//!
//! # Quick Start
//!
//! ```
//! use fpzip_rs::{compress_f32, decompress_f32};
//!
//! let data: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
//! let compressed = compress_f32(&data, 10, 10, 10, 1).unwrap();
//! let decompressed = decompress_f32(&compressed).unwrap();
//! assert_eq!(data, decompressed);
//! ```
//!
//! # Lossy Compression
//!
//! Use [`FpZipCompressor`] with [`prec`](FpZipCompressor::prec) to set reduced
//! bit precision for lossy compression with better ratios:
//!
//! ```
//! use fpzip_rs::FpZipCompressor;
//!
//! let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
//! let compressed = FpZipCompressor::new(4)
//!     .ny(4)
//!     .nz(4)
//!     .prec(16) // 16-bit precision (lossy)
//!     .compress_f32(&data)
//!     .unwrap();
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod codec;
pub mod compressor;
pub mod core;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod header;

// Re-exports for convenience
pub use compressor::{
    compress_f32, compress_f32_into, compress_f64, compress_f64_into, decompress_f32,
    decompress_f32_into, decompress_f64, decompress_f64_into, max_compressed_size, read_header,
    FpZipCompressor,
};
pub use error::{FpZipError, Result};
pub use header::{FpZipHeader, FpZipType};

#[cfg(feature = "std")]
pub use compressor::{
    compress_f32_to_writer, compress_f64_to_writer, decompress_f32_from_reader,
    decompress_f64_from_reader,
};
