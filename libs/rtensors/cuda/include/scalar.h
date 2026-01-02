#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void launch_elementwise_contiguous_f32(float* data, size_t n, uint8_t op, float value, unsigned int block_size);
void launch_elementwise_contiguous_f64(double* data, size_t n, uint8_t op, double value, unsigned int block_size);
void launch_elementwise_contiguous_u8(uint8_t* data, size_t n, uint8_t op, uint8_t value, unsigned int block_size);
void launch_elementwise_contiguous_u16(uint16_t* data, size_t n, uint8_t op, uint16_t value, unsigned int block_size);
void launch_elementwise_contiguous_u32(uint32_t* data, size_t n, uint8_t op, uint32_t value, unsigned int block_size);
void launch_elementwise_contiguous_u64(uint64_t* data, size_t n, uint8_t op, uint64_t value, unsigned int block_size);
void launch_elementwise_contiguous_u128(__uint128_t* data, size_t n, uint8_t op, __uint128_t value, unsigned int block_size);
void launch_elementwise_contiguous_i8(int8_t* data, size_t n, uint8_t op, int8_t value, unsigned int block_size);
void launch_elementwise_contiguous_i16(int16_t* data, size_t n, uint8_t op, int16_t value, unsigned int block_size);
void launch_elementwise_contiguous_i32(int32_t* data, size_t n, uint8_t op, int32_t value, unsigned int block_size);
void launch_elementwise_contiguous_i64(int64_t* data, size_t n, uint8_t op, int64_t value, unsigned int block_size);
void launch_elementwise_contiguous_i128(__int128_t* data, size_t n, uint8_t op, __int128_t value, unsigned int block_size);
void launch_elementwise_contiguous_boolean(bool* data, size_t n, uint8_t op, bool value, unsigned int block_size);

void launch_elementwise_strided_f32(float* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, float value, unsigned int block_size);
void launch_elementwise_strided_f64(double* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, double value, unsigned int block_size);
void launch_elementwise_strided_u8(uint8_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, uint8_t value, unsigned int block_size);
void launch_elementwise_strided_u16(uint16_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, uint16_t value, unsigned int block_size);
void launch_elementwise_strided_u32(uint32_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, uint32_t value, unsigned int block_size);
void launch_elementwise_strided_u64(uint64_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, uint64_t value, unsigned int block_size);
void launch_elementwise_strided_u128(__uint128_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, __uint128_t value, unsigned int block_size);
void launch_elementwise_strided_i8(int8_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, int8_t value, unsigned int block_size);
void launch_elementwise_strided_i16(int16_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, int16_t value, unsigned int block_size);
void launch_elementwise_strided_i32(int32_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, int32_t value, unsigned int block_size);
void launch_elementwise_strided_i64(int64_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, int64_t value, unsigned int block_size);
void launch_elementwise_strided_i128(__int128_t* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, __int128_t value, unsigned int block_size);
void launch_elementwise_strided_boolean(bool* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, bool value, unsigned int block_size);

void launch_elementwise_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, float value, unsigned int block_size);
void launch_elementwise_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, double value, unsigned int block_size);
void launch_elementwise_nd_affine_u8(uint8_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, uint8_t value, unsigned int block_size);
void launch_elementwise_nd_affine_u16(uint16_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, uint16_t value, unsigned int block_size);
void launch_elementwise_nd_affine_u32(uint32_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, uint32_t value, unsigned int block_size);
void launch_elementwise_nd_affine_u64(uint64_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, uint64_t value, unsigned int block_size);
void launch_elementwise_nd_affine_u128(__uint128_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, __uint128_t value, unsigned int block_size);
void launch_elementwise_nd_affine_i8(int8_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, int8_t value, unsigned int block_size);
void launch_elementwise_nd_affine_i16(int16_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, int16_t value, unsigned int block_size);
void launch_elementwise_nd_affine_i32(int32_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, int32_t value, unsigned int block_size);
void launch_elementwise_nd_affine_i64(int64_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, int64_t value, unsigned int block_size);
void launch_elementwise_nd_affine_i128(__int128_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, __int128_t value, unsigned int block_size);
void launch_elementwise_nd_affine_boolean(bool* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, uint8_t op, bool value, unsigned int block_size);

// Negation operations
void launch_negate_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i8(int8_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i16(int16_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i32(int32_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i64(int64_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i128(__int128_t* data, size_t start, size_t len, unsigned int block_size);


void launch_relu_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i8(int8_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i16(int16_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i32(int32_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i64(int64_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i128(__int128_t* data, size_t start, size_t len, unsigned int block_size);

void launch_relu_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i8(int8_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i16(int16_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i32(int32_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i64(int64_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i128(__int128_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);


void launch_relu_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i8(int8_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i16(int16_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i32(int32_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i64(int64_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i128(__int128_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);


void launch_negate_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i8(int8_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i16(int16_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i32(int32_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i64(int64_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i128(__int128_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);

void launch_negate_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i8(int8_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i16(int16_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i32(int32_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i64(int64_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i128(__int128_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);



void launch_sigmoid_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_sigmoid_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);




void launch_sigmoid_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_sigmoid_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);

void launch_sigmoid_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_sigmoid_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);

void launch_tanh_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_tanh_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);

void launch_tanh_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_tanh_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);

void launch_tanh_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_tanh_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);

#define DECLARE_UNARY_FLOAT_OP_HEADERS(op)                                                     \
    void launch_##op##_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size); \
    void launch_##op##_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size); \
                                                                                                \
    void launch_##op##_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size); \
    void launch_##op##_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size); \
                                                                                                \
    void launch_##op##_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size); \
    void launch_##op##_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);


DECLARE_UNARY_FLOAT_OP_HEADERS(abs)

void launch_flat_contiguous_reduce_f64(const double *data, double *out, size_t start, size_t len, ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);
void launch_nd_reduce_contiguous_f64(double *data, double *out, size_t offset, size_t outer, size_t r, size_t inner, ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);
// void launch_elementwise_scattered_f32(float* data, const size_t* offsets, size_t n, uint8_t op, float value, unsigned int block_size);
// void launch_elementwise_scattered_f64(double* data, const size_t* offsets, size_t n, uint8_t op, double value, unsigned int block_size);
// void launch_elementwise_scattered_u8(uint8_t* data, const size_t* offsets, size_t n, uint8_t op, uint8_t value, unsigned int block_size);
// void launch_elementwise_scattered_u16(uint16_t* data, const size_t* offsets, size_t n, uint8_t op, uint16_t value, unsigned int block_size);
// void launch_elementwise_scattered_u32(uint32_t* data, const size_t* offsets, size_t n, uint8_t op, uint32_t value, unsigned int block_size);
// void launch_elementwise_scattered_u64(uint64_t* data, const size_t* offsets, size_t n, uint8_t op, uint64_t value, unsigned int block_size);
// void launch_elementwise_scattered_u128(__uint128_t* data, const size_t* offsets, size_t n, uint8_t op, __uint128_t value, unsigned int block_size);
// void launch_elementwise_scattered_i8(int8_t* data, const size_t* offsets, size_t n, uint8_t op, int8_t value, unsigned int block_size);
// void launch_elementwise_scattered_i16(int16_t* data, const size_t* offsets, size_t n, uint8_t op, int16_t value, unsigned int block_size);
// void launch_elementwise_scattered_i32(int32_t* data, const size_t* offsets, size_t n, uint8_t op, int32_t value, unsigned int block_size);
// void launch_elementwise_scattered_i64(int64_t* data, const size_t* offsets, size_t n, uint8_t op, int64_t value, unsigned int block_size);
// void launch_elementwise_scattered_i128(__int128_t* data, const size_t* offsets, size_t n, uint8_t op, __int128_t value, unsigned int block_size);

#ifdef __cplusplus
}
#endif
