#include "../../../include/scalar.h"

/*
    FUNCTORS
*/

struct AddOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return x + value;
    }
};

struct SubOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return x - value;
    }
};

struct MulOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return x * value;
    }
};

/*
    KERNELS
*/

template <typename T, typename Op>
__global__ void elementwise_strided_kernel(
    T* __restrict__ data,
    size_t start,
    ptrdiff_t stride,
    size_t len,
    T value,
    Op op
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)start + (ptrdiff_t)i * stride);
        data[idx] = op(data[idx], value);
    }
}

/*
    LAUNCHERS
*/

template <typename T, typename Op>
void launch_scalar_strided_op(
    T* data,
    size_t start,
    ptrdiff_t stride,
    size_t len,
    T value,
    unsigned int block_size,
    Op op
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    elementwise_strided_kernel<T, Op><<<grid, block_size>>>(data, start, stride, len, value, op);
}

#define DECLARE_SCALAR_STRIDED_LAUNCHER(OPNAME, OP_TYPE, TYPE, SUFFIX) \
    extern "C" void launch_##OPNAME##_strided_##SUFFIX( \
        TYPE* data, size_t start, ptrdiff_t stride, size_t len, TYPE value, unsigned int block_size \
    ) { \
        launch_scalar_strided_op<TYPE, OP_TYPE>( \
            data, start, stride, len, value, block_size, OP_TYPE{}); \
    }

// add: all types
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, float,  f32)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, double, f64)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, uint8_t,  u8)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, uint16_t, u16)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, uint32_t, u32)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, uint64_t, u64)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, __uint128_t, u128)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, int8_t,  i8)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, int16_t, i16)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, int32_t, i32)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, int64_t, i64)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, __int128_t, i128)
DECLARE_SCALAR_STRIDED_LAUNCHER(add, AddOp, bool, boolean)

// sub: all types
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, float,  f32)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, double, f64)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, uint8_t,  u8)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, uint16_t, u16)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, uint32_t, u32)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, uint64_t, u64)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, __uint128_t, u128)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, int8_t,  i8)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, int16_t, i16)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, int32_t, i32)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, int64_t, i64)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, __int128_t, i128)
DECLARE_SCALAR_STRIDED_LAUNCHER(sub, SubOp, bool, boolean)

// mul: all types
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, float,  f32)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, double, f64)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, uint8_t,  u8)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, uint16_t, u16)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, uint32_t, u32)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, uint64_t, u64)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, __uint128_t, u128)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, int8_t,  i8)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, int16_t, i16)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, int32_t, i32)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, int64_t, i64)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, __int128_t, i128)
DECLARE_SCALAR_STRIDED_LAUNCHER(mul, MulOp, bool, boolean)
