#include <cuda_runtime.h>

#include <stdio.h>
#include "../../include/common.h"
#include <limits.h>
#include <cub/device/device_reduce.cuh>
// #include <cub/iterator/transform_input_iterator.cuh>

/// @brief The summation functor.
struct SumOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

struct ProdOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a * b;
    }
};

struct MaxOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return (a > b) ? a : b;
    }
};

struct MinOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return (a < b) ? a : b;
    }
};

struct PostNothing
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(T out, size_t n) const
    {
        return out;
    }
};

struct PostDivTotal
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(T out, size_t n) const
    {

        return out /= (T)n;
    }
};

template <typename T>
struct WelfordState
{
    T mean;
    T m2;
    int count;
};

template <typename T>
__host__ __device__ __forceinline__ WelfordState<T> welford_init()
{
    return WelfordState<T>{(T)0, (T)0, 0};
}

template <typename T>
__host__ __device__ __forceinline__ void welford_update(WelfordState<T> *state, T x)
{
    state->count += 1; // increment the counter.

    T delta = x - state->mean;
    state->mean += delta / (T)state->count;

    T delta2 = x - state->mean;
    state->m2 += delta * delta2;
}

template <typename T>
__host__ __device__ __forceinline__ WelfordState<T> welford_combine(WelfordState<T> a, WelfordState<T> b)
{
    if (a.count == 0)
    {
        return b;
    }
    if (b.count == 0)
    {
        return a;
    }

    WelfordState<T> result;

    T delta = b.mean - a.mean;
    int n = a.count + b.count;

    result.mean = a.mean + delta * (T(b.count) / T(n));
    result.m2 = a.m2 + b.m2 + (delta * delta) * ((T(a.count) * T(b.count)) / T(n));
    result.count = n;

    return result;
}

/// only ever launched with one element to be processed
template <typename T, typename PostOp>
__global__ void post_transform_kernel(T *out, size_t n, PostOp post)
{

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // printf("HI i = %d, out[%d] = %g\n", i, i, out[i]);
        out[i] = post(out[i], n);
    }
}

/// @brief Designed to run a reduction operation on a contiguous array where we would like to reduce
/// everything to a single element.
/// @tparam T the datatype to use
/// @tparam Op the functor
/// @param data the pointer to the input data
/// @param d_out the poitner to the result buffer
/// @param start the start of where we should read
/// @param num_items how big the buffer is
/// @param block_size the block size
/// @param op the operator, i.e., the actual functor.
/// @param init the base case for reduction.
template <typename T, typename Op, typename PostOp>
void launch_flat_contiguous_reduce(
    const T *data,
    T *d_out,
    size_t start,
    size_t num_items,
    unsigned int block_size,
    Op op,
    T init,
    PostOp post)
{
    // Declare, allocate, and initialize device-accessible pointers for
    // input and output
    const T *d_in = &data[start];

    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, op, init);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, op, init);

    // Free the temporary storage allocation.
    cudaFree(&d_temp_storage);
    
    // only ever one element so this is okay
    post_transform_kernel<<<1, block_size>>>(d_out, num_items, post);
}

template <typename T, typename Op, typename PostOp>
__global__ void sum_axis_contig_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner,
    Op op,
    T init,
    PostOp post)
{
    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    // Each thread accumulates a partial sum over k = threadIdx.x, threadIdx.x+blockDim.x, ...
    T thread_sum = init;
    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {
        thread_sum = op(thread_sum, base[k * inner]);
    }

    // Reduce thread_sum across the block (simple shared-memory reduction)
    __shared__ T smem[256]; // assumes blockDim.x <= 256; adjust if you want bigger
    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    // power-of-two reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] = op(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[out_linear] = post((T)smem[0], R);
    }
}

template <typename T, typename PostOp>
__global__ void var_axis_contig_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner,
    PostOp post,
    bool unbiased,
    bool is_std)
{
    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    // Each thread accumulates a partial sum over k = threadIdx.x, threadIdx.x+blockDim.x, ...
    WelfordState<T> state = welford_init<T>();
    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {
        welford_update(&state, base[k * inner]);
    }

    // Reduce thread_sum across the block (simple shared-memory reduction)
    __shared__ WelfordState<T> smem[256]; // assumes blockDim.x <= 256; adjust if you want bigger
    smem[threadIdx.x] = state;
    __syncthreads();

    // power-of-two reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] = welford_combine(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        WelfordState<T> reso = smem[0];

        // UNBIASED
        T denom;
        if (unbiased)
        {
            denom = (reso.count > 1) ? (T)(reso.count - 1) : (T)1;
        }
        else
        {
            denom = (reso.count > 0) ? (T)(reso.count) : (T)1;
        }

        //

        if(is_std) {
            out[out_linear] = sqrt(reso.m2 / denom);
        } else {
            out[out_linear] = reso.m2 / denom;
        }
        
        // out[out_linear] = (T) smem[0];
        // out[out_linear] = post((T)smem[0], R);
    }
}

template <typename T>
void sum_axis_strided_fast_launch(
    const T *d_in,
    T *d_out,
    size_t offset,
    size_t outer,
    size_t r,
    size_t inner,
    ReductionOpCode op,
    const ReductionSettings *settings,
    unsigned int blocksize)
{
    size_t out_elems = outer * inner;
    if (out_elems == 0)
    {
        // This is the fast path.
        return;
    }

    int block = blocksize;

    dim3 grid((unsigned)out_elems);

    switch (op)
    {
    case OP_SUM:
        sum_axis_contig_kernel<T, SumOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, SumOp{}, (T)0, PostNothing{});
        break;
    case OP_MAX:
        sum_axis_contig_kernel<T, MaxOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, MaxOp{}, std::numeric_limits<T>::lowest(), PostNothing{});
        break;
    case OP_MIN:
        sum_axis_contig_kernel<T, MinOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, MinOp{}, std::numeric_limits<T>::max(), PostNothing{});
        break;
    case OP_PROD:
        sum_axis_contig_kernel<T, ProdOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, ProdOp{}, (T)1, PostNothing{});
        break;
    case OP_MEAN:
        sum_axis_contig_kernel<T, SumOp, PostDivTotal><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, SumOp{}, (T)0, PostDivTotal{});
        break;
    case OP_VARIANCE:
        var_axis_contig_kernel<T, PostDivTotal><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, PostDivTotal{}, settings->unbiased, settings->is_std);
        break;
    // case OP_VARIANCE_UNBIASED:
    //     var_axis_contig_kernel<T, PostDivTotal, true><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, PostDivTotal{});
    //     break;
    // case OP_POP_VARIANCE:
    //     var_axis_contig_kernel<T, PostDivTotal, false><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, PostDivTotal{});
    //     break;
    default:
        return;
        // case OP_ADD: dbuf[dof] = lbuf[lo] + rbuf[ro]; break;
        // case OP_SUB: dbuf[dof] = lbuf[lo] - rbuf[ro]; break;
        // case OP_MUL: dbuf[dof] = lbuf[lo] * rbuf[ro]; break;
    }
}


template <typename T>
void dispatch_flat_contiguous_reduce(
    const T *data,
    T *out,
    size_t start,
    size_t len,
    ReductionOpCode op,
    const ReductionSettings *settings,
    unsigned int block_size)
{

    switch (op)
    {
    case OP_SUM:
        launch_flat_contiguous_reduce<T, SumOp, PostNothing>(data, out, start, len, block_size, SumOp{}, (T)0, PostNothing{});
        break;
    case OP_MIN:
        launch_flat_contiguous_reduce<T, MinOp, PostNothing>(data, out, start, len, block_size, MinOp{}, (T)std::numeric_limits<T>::max(), PostNothing{});
        break;
    case OP_MAX:
        launch_flat_contiguous_reduce<T, MaxOp, PostNothing>(data, out, start, len, block_size, MaxOp{}, (T)std::numeric_limits<T>::lowest(), PostNothing{});
        break;
    case OP_PROD:
        launch_flat_contiguous_reduce<T, ProdOp, PostNothing>(data, out, start, len, block_size, ProdOp{}, (T)1, PostNothing{});
        break;
    case OP_MEAN:
        launch_flat_contiguous_reduce<T, SumOp, PostDivTotal>(data, out, start, len, block_size, SumOp{}, (T)0, PostDivTotal{});
        break;
    case OP_VARIANCE:
        launch_flat_contiguous_reduce_variance<T>(data, out, start, len, settings, block_size);
        break;

    default:
        return;
    }
}

extern "C" void launch_flat_contiguous_reduce_f64(const double *data, double *out, size_t start, size_t len, ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size)
{
    dispatch_flat_contiguous_reduce<double>(data, out, start, len, code, settings, block_size);
}

extern "C" void launch_nd_reduce_contiguous_f64(double *data, double *out, size_t offset, size_t outer, size_t r, size_t inner, ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size)
{
    sum_axis_strided_fast_launch<double>(
        data,
        out,
        offset,
        outer,
        r,
        inner,
        code,
        settings,
        block_size);
}

