#include "../../../include/scalar.h"

template <typename T>
__global__ void elementwise_contiguous_kernel(
    T* __restrict__ data,
    size_t n,
    uint8_t op,
    T value
) {
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {

        T x = data[i];
        switch (op) {
            case OP_ADD: data[i] = x + value; break;
            case OP_SUB: data[i] = x - value; break;
            case OP_MUL: data[i] = x * value; break;
        }
    }
}

template <typename T>
void launch_elementwise_contiguous_op(
    T* data,
    size_t n,
    uint8_t op,
    T value,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((n + block_size - 1) / block_size), 65535u);
    elementwise_contiguous_kernel<T><<<grid, block_size>>>(data, n, op, value);
}

#define DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_elementwise_contiguous_##SUFFIX( \
        TYPE* data, size_t n, uint8_t op, TYPE value, unsigned int block_size \
    ) { \
        launch_elementwise_contiguous_op<TYPE>(data, n, op, value, block_size); \
    }


DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(float,  f32)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(double, f64)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(uint8_t,  u8)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(uint16_t, u16)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(uint32_t, u32)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(uint64_t, u64)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(__uint128_t, u128)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(int8_t,  i8)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(int16_t, i16)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(int32_t, i32)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(int64_t, i64)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(__int128_t, i128)
DECLARE_ELEMENTWISE_CONTIGUOUS_LAUNCHER(bool, boolean)
