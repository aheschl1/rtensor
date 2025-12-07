// This file contains a CUDA kernel for generalized matrix multiplication (GEMM) operations.
// Shoud NOT be used for types supported by cuBLAS
#include "../../include/matmul.h"

template <typename T>
__global__ void matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    size_t M,
    size_t N,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc
) {
    for (size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
         row < M; 
         row += blockDim.y * gridDim.y) {
        for (size_t col = blockIdx.x * blockDim.x + threadIdx.x; 
             col < N; 
             col += blockDim.x * gridDim.x) {
            
            T value = static_cast<T>(0);
            for (size_t k = 0; k < K; ++k) {
                value += A[row * lda + k] * B[k * ldb + col];
            }
            C[row * ldc + col] = value;
        }
    }
}

template <typename T>
void launch_matmul_op(
    const T* A,
    const T* B,
    T* C,
    size_t M,
    size_t N,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc,
    unsigned int block_size
) {
    // block_size = ALIGN_BLOCK_SIZE(block_size);

    // dim3 block(block_size, block_size);
    dim3 block(16, 16);
    dim3 grid(
        std::min((unsigned int)((N + block.x - 1) / block.x), 65535u), 
        std::min((unsigned int)((M + block.y - 1) / block.y), 65535u)
    );
    
    matmul_kernel<T><<<grid, block>>>(A, B, C, M, N, K, lda, ldb, ldc);
}

#define DECLARE_MATMUL_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_matmul_##SUFFIX( \
        const TYPE* A, const TYPE* B, TYPE* C, \
        size_t M, size_t N, size_t K, \
        size_t lda, size_t ldb, size_t ldc, \
        unsigned int block_size \
    ) { \
        launch_matmul_op<TYPE>(A, B, C, M, N, K, lda, ldb, ldc, block_size); \
    }

DECLARE_MATMUL_LAUNCHER(uint8_t,  u8)
DECLARE_MATMUL_LAUNCHER(uint16_t, u16)
DECLARE_MATMUL_LAUNCHER(uint32_t, u32)
DECLARE_MATMUL_LAUNCHER(uint64_t, u64)
DECLARE_MATMUL_LAUNCHER(__uint128_t, u128)
DECLARE_MATMUL_LAUNCHER(int8_t,  i8)
DECLARE_MATMUL_LAUNCHER(int16_t, i16)
DECLARE_MATMUL_LAUNCHER(int32_t, i32)
DECLARE_MATMUL_LAUNCHER(int64_t, i64)
DECLARE_MATMUL_LAUNCHER(__int128_t, i128)