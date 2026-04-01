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
