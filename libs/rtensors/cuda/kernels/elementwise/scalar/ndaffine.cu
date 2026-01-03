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
__global__ void elementwise_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size,
    T value,
    Op op
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

        data[final_idx] = op(data[final_idx], value);
    }
}

/*
    LAUNCHERS
*/

template <typename T, typename Op>
void launch_scalar_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    T value,
    unsigned int block_size,
    Op op
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    elementwise_nd_affine_kernel<T, Op><<<grid, block_size>>>(data, offset, stride, shape, rank, size, value, op);
}

#define DECLARE_SCALAR_ND_AFFINE_LAUNCHER(OPNAME, OP_TYPE, TYPE, SUFFIX) \
    extern "C" void launch_##OPNAME##_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, TYPE value, unsigned int block_size \
    ) { \
        launch_scalar_nd_affine_op<TYPE, OP_TYPE>( \
            data, offset, stride, shape, rank, size, value, block_size, OP_TYPE{}); \
    }

// add: all types
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, float,  f32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, double, f64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, uint8_t,  u8)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, uint16_t, u16)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, uint32_t, u32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, uint64_t, u64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, __uint128_t, u128)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, int8_t,  i8)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, int16_t, i16)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, int32_t, i32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, int64_t, i64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, __int128_t, i128)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(add, AddOp, bool, boolean)

// sub: all types
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, float,  f32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, double, f64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, uint8_t,  u8)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, uint16_t, u16)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, uint32_t, u32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, uint64_t, u64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, __uint128_t, u128)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, int8_t,  i8)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, int16_t, i16)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, int32_t, i32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, int64_t, i64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, __int128_t, i128)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(sub, SubOp, bool, boolean)

// mul: all types
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, float,  f32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, double, f64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, uint8_t,  u8)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, uint16_t, u16)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, uint32_t, u32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, uint64_t, u64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, __uint128_t, u128)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, int8_t,  i8)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, int16_t, i16)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, int32_t, i32)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, int64_t, i64)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, __int128_t, i128)
DECLARE_SCALAR_ND_AFFINE_LAUNCHER(mul, MulOp, bool, boolean)
