#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void launch_matmul_u8(const uint8_t* A, const uint8_t* B, uint8_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_u16(const uint16_t* A, const uint16_t* B, uint16_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_u32(const uint32_t* A, const uint32_t* B, uint32_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_u64(const uint64_t* A, const uint64_t* B, uint64_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_u128(const __uint128_t* A, const __uint128_t* B, __uint128_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_i8(const int8_t* A, const int8_t* B, int8_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_i16(const int16_t* A, const int16_t* B, int16_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_i32(const int32_t* A, const int32_t* B, int32_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_i64(const int64_t* A, const int64_t* B, int64_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_i128(const __int128_t* A, const __int128_t* B, __int128_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);
void launch_matmul_boolean(const bool* A, const bool* B, bool* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, ContiguityType contiguity, unsigned int block_size);



#ifdef __cplusplus
}
#endif