#include "../../../include/scalar.h"

template <typename T>
__global__ void elementwise_strided_kernel(
    T* __restrict__ data,
    size_t start,
    ptrdiff_t stride,
    size_t len,
    uint8_t op,
    T value
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)start + (ptrdiff_t)i * stride);
        T x = data[idx];
        switch (op) {
            case OP_ADD: data[idx] = x + value; break;
            case OP_SUB: data[idx] = x - value; break;
            case OP_MUL: data[idx] = x * value; break;
        }
    }
}

template <typename T>
void launch_elementwise_strided_op(
    T* data,
    size_t start,
    ptrdiff_t stride,
    size_t len,
    uint8_t op,
    T value,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    elementwise_strided_kernel<T><<<grid, block_size>>>(data, start, stride, len, op, value);
}

#define DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_elementwise_strided_##SUFFIX( \
        TYPE* data, size_t start, ptrdiff_t stride, size_t len, uint8_t op, TYPE value, unsigned int block_size \
    ) { \
        launch_elementwise_strided_op<TYPE>(data, start, stride, len, op, value, block_size); \
    }

DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(float,  f32)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(double, f64)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(uint8_t,  u8)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(uint16_t, u16)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(uint32_t, u32)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(uint64_t, u64)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(__uint128_t, u128)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(int8_t,  i8)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(int16_t, i16)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(int32_t, i32)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(int64_t, i64)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(__int128_t, i128)
DECLARE_ELEMENTWISE_STRIDED_LAUNCHER(bool, boolean)
