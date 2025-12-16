#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void launch_binary_broadcast_elementwise_f32(const float* lbuf, const float* rbuf, float* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_f64(const double* lbuf, const double* rbuf, double* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_u8(const uint8_t* lbuf, const uint8_t* rbuf, uint8_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_u16(const uint16_t* lbuf, const uint16_t* rbuf, uint16_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_u32(const uint32_t* lbuf, const uint32_t* rbuf, uint32_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_u64(const uint64_t* lbuf, const uint64_t* rbuf, uint64_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_u128(const __uint128_t* lbuf, const __uint128_t* rbuf, __uint128_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_i8(const int8_t* lbuf, const int8_t* rbuf, int8_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_i16(const int16_t* lbuf, const int16_t* rbuf, int16_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_i32(const int32_t* lbuf, const int32_t* rbuf, int32_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_i64(const int64_t* lbuf, const int64_t* rbuf, int64_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_i128(const __int128_t* lbuf, const __int128_t* rbuf, __int128_t* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);
void launch_binary_broadcast_elementwise_boolean(const bool* lbuf, const bool* rbuf, bool* dbuf, size_t loff, size_t roff, size_t doff, size_t rank, size_t size, const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, const size_t* shape, uint8_t op, unsigned int block_size);

#ifdef __cplusplus
}
#endif
