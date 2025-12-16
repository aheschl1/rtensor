// #include "../../../include/scalar.h"

// template <typename T>
// __global__ void elementwise_scattered_kernel(
//     T* __restrict__ data,
//     const size_t* __restrict__ offsets,
//     size_t n,
//     uint8_t op,
//     T value
// ) {
//     for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//          i < n;
//          i += blockDim.x * gridDim.x) {

//         const size_t idx = offsets[i];
//         T x = data[idx];
//         switch (op) {
//             case OP_ADD: data[idx] = x + value; break;
//             case OP_SUB: data[idx] = x - value; break;
//             case OP_MUL: data[idx] = x * value; break;
//         }
//     }
// }

// template <typename T>
// void launch_elementwise_scattered_op(
//     T* data,
//     const size_t* offsets,
//     size_t n,
//     uint8_t op,
//     T value,
//     unsigned int block_size
// ) {
//     block_size = ALIGN_BLOCK_SIZE(block_size);

//     const unsigned int grid = (unsigned int)((n + block_size - 1) / block_size);
//     elementwise_scattered_kernel<T><<<grid, block_size>>>(data, offsets, n, op, value);
// }

// #define DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(TYPE, SUFFIX) \
//     extern "C" void launch_elementwise_scattered_##SUFFIX( \
//         TYPE* data, const size_t* offsets, size_t n, uint8_t op, TYPE value, unsigned int block_size \
//     ) { \
//         launch_elementwise_scattered_op<TYPE>(data, offsets, n, op, value, block_size); \
//     }


// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(float,  f32)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(double, f64)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(uint8_t,  u8)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(uint16_t, u16)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(uint32_t, u32)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(uint64_t, u64)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(__uint128_t, u128)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(int8_t,  i8)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(int16_t, i16)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(int32_t, i32)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(int64_t, i64)
// DECLARE_ELEMENTWISE_SCATTERED_LAUNCHER(__int128_t, i128)
