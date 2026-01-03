#include "../../include/scalar.h"
#include <stdio.h>

/*
    FUNCTORS
*/

struct SigmoidOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return ((T) 1) / ((T) 1 + exp(-x));
    }
};

struct SiluOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return x / ((T) 1 + exp(-x));
    }
};

struct LnOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return log(x);
    }
};

struct Ln1pOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return log1p(x);
    }
};

struct FloorOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return floor(x);
    }
};

struct CeilOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return ceil(x);
    }
};

struct RoundOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return round(x);
    }
};

struct TruncOp {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return trunc(x);
    }
};

struct ExpM1Op {
    template<typename  T>
    __device__ __forceinline__ T operator()(T x) const {
         return expm1(x);
    }
};

struct ReluOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return x < (T)0 ? (T)0 : x;
    }
};

struct SqrtOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return sqrt(x);
    }
};

struct TanhOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        T a = exp(x);
        T b = exp(-x);
        return (a - b) / (a + b);
    }
};

struct NegateOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return -x;
    }
};

struct AbsOp {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        if(x < (T) 0) {
            return -x;
        } else {
            return x;
        }
    }
};

/*
    KERNELS
*/

template <typename T, typename Op>
__global__ void elementwise_strided_kernel(
    T* __restrict__ data,
    size_t offset,
    ptrdiff_t stride,
    size_t len,
    Op op
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)offset + (ptrdiff_t)i * stride);
        data[idx] = op(data[idx]);
    }
}

template <typename T, typename Op>
__global__ void elementwise_contiguous_kernel(
    T *__restrict__ data,
    size_t start,
    size_t len,
    Op op)
{
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x)
    {

        size_t idx = start + i;
        data[idx] = op(data[idx]);
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

        data[final_idx] = op(data[final_idx]);
    }
}

/*
    LAUNCHERS
*/

template <typename T, typename Op>
void launch_unary_nd_affine_op(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    unsigned int block_size,
    Op op
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    elementwise_nd_affine_kernel<T, Op><<<grid, block_size>>>(data, offset, stride, shape, rank, size, op);
}

template <typename T, typename Op>
void launch_unary_contiguous_op(
    T *data,
    size_t start,
    size_t len,
    unsigned int block_size,
    Op op)
{

    block_size = ALIGN_BLOCK_SIZE(block_size);
    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    elementwise_contiguous_kernel<T, Op>
        <<<grid, block_size>>>(data, start, len, op);
}

template <typename T, typename Op>
void launch_unary_strided_op(
    T* data,
    size_t offset,
    ptrdiff_t stride,
    size_t len,
    unsigned int block_size,
    Op op
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    elementwise_strided_kernel<T, Op><<<grid, block_size>>>(data, offset, stride, len, op);
}



#define DECLARE_UNARY_LAUNCHERS(OPNAME, OP_TYPE, TYPE, SUFFIX)                    \
    extern "C" void launch_##OPNAME##_contiguous_##SUFFIX(                        \
        TYPE* data, size_t start, size_t len, unsigned int block_size)            \
    {                                                                             \
        launch_unary_contiguous_op<TYPE, OP_TYPE>(                                \
            data, start, len, block_size, OP_TYPE{});                             \
    }                                                                             \
                                                                                  \
    extern "C" void launch_##OPNAME##_strided_##SUFFIX(                           \
        TYPE* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size) \
    {                                                                             \
        launch_unary_strided_op<TYPE, OP_TYPE>(                                   \
            data, offset, stride, len, block_size, OP_TYPE{});                    \
    }                                                                             \
                                                                                  \
    extern "C" void launch_##OPNAME##_nd_affine_##SUFFIX(                         \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape,  \
        size_t rank, size_t size, unsigned int block_size)                        \
    {                                                                             \
        launch_unary_nd_affine_op<TYPE, OP_TYPE>(                                  \
            data, offset, stride, shape, rank, size, block_size, OP_TYPE{});      \
    }

// sigmoid: float/double only
DECLARE_UNARY_LAUNCHERS(sigmoid, SigmoidOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(sigmoid, SigmoidOp, double, f64)

DECLARE_UNARY_LAUNCHERS(silu, SiluOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(silu, SiluOp, double, f64)

// negate: all signed + floats
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, float,   f32)
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, double,  f64)
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, int8_t,  i8)
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, int16_t, i16)
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, int32_t, i32)
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, int64_t, i64)
DECLARE_UNARY_LAUNCHERS(negate, NegateOp, __int128_t, i128)

// relu: all signed + floats
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, float,   f32)
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, double,  f64)
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, int8_t,  i8)
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, int16_t, i16)
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, int32_t, i32)
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, int64_t, i64)
DECLARE_UNARY_LAUNCHERS(relu, ReluOp, __int128_t, i128)

// tanh: float/double only
DECLARE_UNARY_LAUNCHERS(tanh, TanhOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(tanh, TanhOp, double, f64)

DECLARE_UNARY_LAUNCHERS(abs, AbsOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(abs, AbsOp, double, f64)

DECLARE_UNARY_LAUNCHERS(sqrt, SqrtOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(sqrt, SqrtOp, double, f64)

DECLARE_UNARY_LAUNCHERS(ln, LnOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(ln, LnOp, double, f64)

DECLARE_UNARY_LAUNCHERS(ln1p, Ln1pOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(ln1p, Ln1pOp, double, f64)

DECLARE_UNARY_LAUNCHERS(floor, FloorOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(floor, FloorOp, double, f64)

DECLARE_UNARY_LAUNCHERS(ceil, CeilOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(ceil, CeilOp, double, f64)

DECLARE_UNARY_LAUNCHERS(round, RoundOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(round, RoundOp, double, f64)

DECLARE_UNARY_LAUNCHERS(trunc, TruncOp, float,  f32)
DECLARE_UNARY_LAUNCHERS(trunc, TruncOp, double, f64)

DECLARE_UNARY_LAUNCHERS(expm1, ExpM1Op, float,  f32)
DECLARE_UNARY_LAUNCHERS(expm1, ExpM1Op, double, f64)


// extern "C" void launch_test_summy(double *data, size_t start, size_t len, unsigned int block_size)
// {
//     // launch_test_sum_op(data, start, len, block_size);
// }
