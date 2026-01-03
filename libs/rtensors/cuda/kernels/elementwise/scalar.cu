#include "../../include/scalar.h"

/*
    FUNCTORS
*/

struct AddOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return x + value;
    }
};

struct EluOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return x >= T(0) ? x : value * expm1(x);
    }
};

// log base n
struct LogOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return log(x) / log(value);
    }
};

struct Log1POp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T value) const {
        return log1p(x) / log(value);
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

struct LeakyReluOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x, T slope) const {
        return x > T(0) ? x : x * slope;
    }
};

/*
    KERNELS
*/

template <typename T, typename Op>
__global__ void elementwise_contiguous_kernel(
    T* __restrict__ data,
    size_t n,
    T value,
    Op op
) {
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {

        data[i] = op(data[i], value);
    }
}

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
void launch_scalar_contiguous_op(
    T* data,
    size_t n,
    T value,
    unsigned int block_size,
    Op op
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((n + block_size - 1) / block_size), 65535u);
    elementwise_contiguous_kernel<T, Op><<<grid, block_size>>>(data, n, value, op);
}

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

#define DECLARE_SCALAR_LAUNCHERS(OPNAME, OP_TYPE, TYPE, SUFFIX) \
    extern "C" void launch_##OPNAME##_contiguous_##SUFFIX( \
        TYPE* data, size_t n, TYPE value, unsigned int block_size \
    ) { \
        launch_scalar_contiguous_op<TYPE, OP_TYPE>( \
            data, n, value, block_size, OP_TYPE{}); \
    } \
    \
    extern "C" void launch_##OPNAME##_strided_##SUFFIX( \
        TYPE* data, size_t start, ptrdiff_t stride, size_t len, TYPE value, unsigned int block_size \
    ) { \
        launch_scalar_strided_op<TYPE, OP_TYPE>( \
            data, start, stride, len, value, block_size, OP_TYPE{}); \
    } \
    \
    extern "C" void launch_##OPNAME##_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, TYPE value, unsigned int block_size \
    ) { \
        launch_scalar_nd_affine_op<TYPE, OP_TYPE>( \
            data, offset, stride, shape, rank, size, value, block_size, OP_TYPE{}); \
    }

// add: all types
DECLARE_SCALAR_LAUNCHERS(add, AddOp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, double, f64)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, uint8_t,  u8)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, uint16_t, u16)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, uint32_t, u32)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, uint64_t, u64)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, __uint128_t, u128)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, int8_t,  i8)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, int16_t, i16)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, int32_t, i32)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, int64_t, i64)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, __int128_t, i128)
DECLARE_SCALAR_LAUNCHERS(add, AddOp, bool, boolean)

// sub: all types
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, double, f64)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, uint8_t,  u8)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, uint16_t, u16)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, uint32_t, u32)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, uint64_t, u64)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, __uint128_t, u128)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, int8_t,  i8)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, int16_t, i16)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, int32_t, i32)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, int64_t, i64)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, __int128_t, i128)
DECLARE_SCALAR_LAUNCHERS(sub, SubOp, bool, boolean)

// mul: all types
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, double, f64)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, uint8_t,  u8)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, uint16_t, u16)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, uint32_t, u32)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, uint64_t, u64)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, __uint128_t, u128)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, int8_t,  i8)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, int16_t, i16)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, int32_t, i32)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, int64_t, i64)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, __int128_t, i128)
DECLARE_SCALAR_LAUNCHERS(mul, MulOp, bool, boolean)

DECLARE_SCALAR_LAUNCHERS(log, LogOp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(log, LogOp, double,  f64)

DECLARE_SCALAR_LAUNCHERS(log1p, Log1POp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(log1p, Log1POp, double,  f64)

DECLARE_SCALAR_LAUNCHERS(elu, EluOp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(elu, EluOp, double,  f64)

// leaky_relu: all types
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, float,  f32)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, double, f64)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, uint8_t,  u8)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, uint16_t, u16)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, uint32_t, u32)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, uint64_t, u64)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, __uint128_t, u128)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, int8_t,  i8)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, int16_t, i16)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, int32_t, i32)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, int64_t, i64)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, __int128_t, i128)
DECLARE_SCALAR_LAUNCHERS(leaky_relu, LeakyReluOp, bool, boolean)
