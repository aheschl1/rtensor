#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar binary operations - Add
#define DECLARE_SCALAR_OP_HEADERS(op, TYPE, SUFFIX) \
    void launch_##op##_contiguous_##SUFFIX(TYPE* data, size_t n, TYPE value, unsigned int block_size); \
    void launch_##op##_strided_##SUFFIX(TYPE* data, size_t start, ptrdiff_t stride, size_t len, TYPE value, unsigned int block_size); \
    void launch_##op##_nd_affine_##SUFFIX(TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, TYPE value, unsigned int block_size);

// Add operation - all types
DECLARE_SCALAR_OP_HEADERS(add, float, f32)
DECLARE_SCALAR_OP_HEADERS(add, double, f64)
DECLARE_SCALAR_OP_HEADERS(add, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(add, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(add, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(add, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(add, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(add, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(add, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(add, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(add, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(add, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(add, bool, boolean)

// Sub operation - all types
DECLARE_SCALAR_OP_HEADERS(sub, float, f32)
DECLARE_SCALAR_OP_HEADERS(sub, double, f64)
DECLARE_SCALAR_OP_HEADERS(sub, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(sub, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(sub, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(sub, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(sub, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(sub, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(sub, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(sub, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(sub, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(sub, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(sub, bool, boolean)

// Mul operation - all types
DECLARE_SCALAR_OP_HEADERS(mul, float, f32)
DECLARE_SCALAR_OP_HEADERS(mul, double, f64)
DECLARE_SCALAR_OP_HEADERS(mul, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(mul, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(mul, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(mul, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(mul, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(mul, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(mul, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(mul, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(mul, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(mul, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(mul, bool, boolean)

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
DECLARE_UNARY_FLOAT_OP_HEADERS(sqrt)

// Reduction operations
#define DECLARE_REDUCTION_OP_HEADERS(TYPE, SUFFIX)                                                \
    void launch_flat_contiguous_reduce_##SUFFIX(                                                  \
        const TYPE *data, TYPE *out, size_t start, size_t len,                                    \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);        \
    void launch_nd_reduce_contiguous_##SUFFIX(                                                    \
        TYPE *data, TYPE *out, size_t offset, size_t outer, size_t r, size_t inner,              \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);

// Float types
DECLARE_REDUCTION_OP_HEADERS(float,  f32)
DECLARE_REDUCTION_OP_HEADERS(double, f64)


// Reduction operations
#define DECLARE_ARGMAX_OP_HEADERS(TYPE, SUFFIX)                                                \
    void launch_flat_contiguous_argmax_##SUFFIX(                                                  \
        const TYPE *data, uint64_t *out, size_t start, size_t len,                                    \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);        \
    void launch_nd_argmax_contiguous_##SUFFIX(                                                    \
        TYPE *data, uint64_t *out, size_t offset, size_t outer, size_t r, size_t inner,              \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);

// Float types
DECLARE_ARGMAX_OP_HEADERS(float,  f32)
DECLARE_ARGMAX_OP_HEADERS(double, f64)

// // Unsigned integer types
// DECLARE_REDUCTION_OP_HEADERS(uint8_t,  u8)
// DECLARE_REDUCTION_OP_HEADERS(uint16_t, u16)
// DECLARE_REDUCTION_OP_HEADERS(uint32_t, u32)
// DECLARE_REDUCTION_OP_HEADERS(uint64_t, u64)
// DECLARE_REDUCTION_OP_HEADERS(__uint128_t, u128)

// // Signed integer types
// DECLARE_REDUCTION_OP_HEADERS(int8_t,  i8)
// DECLARE_REDUCTION_OP_HEADERS(int16_t, i16)
// DECLARE_REDUCTION_OP_HEADERS(int32_t, i32)
// DECLARE_REDUCTION_OP_HEADERS(int64_t, i64)
// DECLARE_REDUCTION_OP_HEADERS(__int128_t, i128)


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
