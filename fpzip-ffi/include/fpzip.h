/* fpzip-rs C FFI header */
#ifndef FPZIP_RS_H
#define FPZIP_RS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define FPZIP_OK                    0
#define FPZIP_ERR_NULL_PTR         -1
#define FPZIP_ERR_INVALID_MAGIC    -2
#define FPZIP_ERR_UNSUPPORTED_VER  -3
#define FPZIP_ERR_INVALID_TYPE     -4
#define FPZIP_ERR_DIM_MISMATCH     -5
#define FPZIP_ERR_TYPE_MISMATCH    -6
#define FPZIP_ERR_BUF_TOO_SMALL    -7
#define FPZIP_ERR_UNEXPECTED_EOF   -8
#define FPZIP_ERR_IO               -9

/* Data types */
#define FPZIP_TYPE_FLOAT  0
#define FPZIP_TYPE_DOUBLE 1

/* Header structure */
typedef struct {
    uint8_t  data_type;
    uint32_t nx;
    uint32_t ny;
    uint32_t nz;
    uint32_t nf;
} fpzip_header_t;

/* Compress float data. Returns 0 on success. */
int32_t fpzip_compress_float(
    const float* data, size_t len,
    uint32_t nx, uint32_t ny, uint32_t nz, uint32_t nf,
    uint8_t* out_buf, size_t out_capacity, size_t* out_len);

/* Compress double data. Returns 0 on success. */
int32_t fpzip_compress_double(
    const double* data, size_t len,
    uint32_t nx, uint32_t ny, uint32_t nz, uint32_t nf,
    uint8_t* out_buf, size_t out_capacity, size_t* out_len);

/* Decompress float data. Returns 0 on success. */
int32_t fpzip_decompress_float(
    const uint8_t* compressed, size_t compressed_len,
    float* out_buf, size_t out_capacity, size_t* out_len);

/* Decompress double data. Returns 0 on success. */
int32_t fpzip_decompress_double(
    const uint8_t* compressed, size_t compressed_len,
    double* out_buf, size_t out_capacity, size_t* out_len);

/* Read header from compressed data. Returns 0 on success. */
int32_t fpzip_read_header(
    const uint8_t* data, size_t data_len,
    fpzip_header_t* header);

/* Returns maximum possible compressed size. */
size_t fpzip_max_compressed_size(size_t element_count, uint8_t data_type);

/* Returns a static error message string for the given error code. */
const char* fpzip_error_message(int32_t code);

#ifdef __cplusplus
}
#endif

#endif /* FPZIP_RS_H */
