#include "../../../include/unary.h"

template <typename T>
__global__ void elementwise_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size,
    uint8_t op,
    T value
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {
        size_t linear = idx;
        ptrdiff_t phys = (ptrdiff_t)offset;

        for (int dim = (int)rank - 1; dim >= 0; --dim) {
            size_t coord = linear % shape[dim];
            linear /= shape[dim];
            phys += (ptrdiff_t)coord * stride[dim];
        }

        size_t final_idx = (size_t)phys; 

        T x = data[final_idx];
        switch (op) {
            case OP_ADD: data[final_idx] = x + value; break;
            case OP_SUB: data[final_idx] = x - value; break;
            case OP_MUL: data[final_idx] = x * value; break;
        }
    }
}

template <typename T>
void launch_elementwise_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    uint8_t op,
    T value,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    elementwise_nd_affine_kernel<T><<<grid, block_size>>>(data, offset, stride, shape, rank, size, op, value);
}

#define DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_elementwise_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, uint8_t op, TYPE value, unsigned int block_size \
    ) { \
        launch_elementwise_nd_affine_op<TYPE>(data, offset, stride, shape, rank, size, op, value, block_size); \
    }

DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(float,  f32)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(double, f64)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(uint8_t,  u8)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(uint16_t, u16)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(uint32_t, u32)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(uint64_t, u64)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(__uint128_t, u128)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(int8_t,  i8)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(int16_t, i16)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(int32_t, i32)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(int64_t, i64)
DECLARE_ELEMENTWISE_ND_AFFINE_LAUNCHER(__int128_t, i128)
