#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>

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
template <typename T, typename Op>
void launch_flat_contiguous_reduce(
    T *data,
    T* d_out,
    size_t start,
    size_t num_items,
    unsigned int block_size,
    Op op,
    T init
)
{
    // Declare, allocate, and initialize device-accessible pointers for
    // input and output
    T *d_in = &data[start];

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
}


template <typename T>
void launch_test_sum_op(
    T *data,
    size_t start,
    size_t len,
    unsigned int block_size)
{

    T* output;
    cudaMalloc(&output, sizeof(T));
    launch_flat_contiguous_reduce<T, SumOp>(data, output, start, len, block_size, SumOp {}, (T) 0);
    T h0;
    cudaMemcpy(&h0, output, sizeof(T), cudaMemcpyDeviceToHost);
    printf("first element = %g\n", (double)h0); // cast for safe printf

    cudaDeviceSynchronize();
    // printf("Hello\n");
    // cudaDeviceSynchronize();

    // // Declare, allocate, and initialize device-accessible pointers for
    // // input and output
    // int num_items = len;    // e.g., 7
    // T *d_in = &data[start]; // e.g., [8, 6, 7, 5, 3, 0, 9]
    // T *d_out = nullptr;     // e.g., [-]
    // CustomMin min_op;
    // T init = (T)0; // e.g., INT_MAX

    // cudaMalloc(&d_out, sizeof(T));

    // // Determine temporary device storage requirements
    // void *d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    // cub::DeviceReduce::Reduce(
    //     d_temp_storage, temp_storage_bytes,
    //     d_in, d_out, num_items, min_op, init);

    // // Allocate temporary storage
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // // Run reduction
    // cub::DeviceReduce::Reduce(
    //     d_temp_storage, temp_storage_bytes,
    //     d_in, d_out, num_items, min_op, init);

    // T h0;
    // cudaMemcpy(&h0, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    // printf("first element = %g\n", (double)h0); // cast for safe printf

    // cudaDeviceSynchronize();

    // // block_size = ALIGN_BLOCK_SIZE(block_size);
    // // const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    // // test_sum_kernel<T><<<grid, block_size>>>(data, start, len);
}

// #define D+ECLARE_TANH_CONTIGUOUS_LAUNCHER(TYPE, SUFFIX) \
//     extern "C" void launch_tanh_contiguous_##SUFFIX( \
//         TYPE* data, size_t start, size_t len, unsigned int block_size \
//     ) { \
//         launch_tanh_contiguous_op<TYPE>(data, start, len, block_size); \
//     }


extern "C" void launch_flat_contiguous_reduce_sum_double(double *data, double *out, size_t start, size_t len, unsigned int block_size)
{
    launch_flat_contiguous_reduce<double, SumOp>(data, out, start, len, block_size, SumOp {}, (double) 0);
}

extern "C" void launch_test_summy(double *data, size_t start, size_t len, unsigned int block_size)
{
    launch_test_sum_op(data, start, len, block_size);
}