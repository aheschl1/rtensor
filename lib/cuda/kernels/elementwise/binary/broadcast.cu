#include "../../../include/binary.h"

template <typename T>
__global__ void binary_broadcast_elementwise(
    const T* __restrict__ lbuf,
    const T* __restrict__ rbuf,
    T* __restrict__ dbuf,
    size_t loff,
    size_t roff,
    size_t doff,
    size_t rank,
    size_t size,  // prod(shape)
    const ptrdiff_t* __restrict__ lstride,
    const ptrdiff_t* __restrict__ rstride,
    const ptrdiff_t* __restrict__ dstride,
    const size_t * __restrict__ shape,
    uint8_t op
) {
    size_t coords[128];
    for(
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {

        size_t rem = idx;
        

        for (int d = rank - 1; d >= 0; --d) {
            size_t dim = shape[d];
            coords[d] = rem % dim;
            rem /= dim;
        }

        ptrdiff_t lo = (ptrdiff_t) loff; 
        ptrdiff_t ro = (ptrdiff_t) roff; 
        ptrdiff_t dof = (ptrdiff_t) doff;
        
        for (size_t d = 0; d < rank; ++d) {
            lo += (ptrdiff_t)coords[d] * lstride[d];
        }
        for (size_t d = 0; d < rank; ++d) {
            ro += (ptrdiff_t)coords[d] * rstride[d];
        }
        for (size_t d = 0; d < rank; ++d) {
            dof += (ptrdiff_t)coords[d] * dstride[d];
        }

        switch (op) {
            case OP_ADD: dbuf[dof] = lbuf[lo] + rbuf[ro]; break;
            case OP_SUB: dbuf[dof] = lbuf[lo] - rbuf[ro]; break;
            case OP_MUL: dbuf[dof] = lbuf[lo] * rbuf[ro]; break;
        }
    }
}

template <typename T>
void launch_binary_broadcast_elementwise_op(
    const T* lbuf,
    const T* rbuf,
    T* dbuf,
    size_t loff,
    size_t roff,
    size_t doff,
    size_t rank,
    size_t size,
    const ptrdiff_t* lstride,
    const ptrdiff_t* rstride,
    const ptrdiff_t* dstride,
    const size_t* shape,
    uint8_t op,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    binary_broadcast_elementwise<T><<<grid, block_size>>>(
        lbuf, rbuf, dbuf, loff, roff, doff, rank, size, lstride, rstride, dstride, shape, op
    );
}

#define DECLARE_BINARY_BROADCAST_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_binary_broadcast_elementwise_##SUFFIX( \
        const TYPE* lbuf, const TYPE* rbuf, TYPE* dbuf, \
        size_t loff, size_t roff, size_t doff, \
        size_t rank, size_t size, \
        const ptrdiff_t* lstride, const ptrdiff_t* rstride, const ptrdiff_t* dstride, \
        const size_t* shape, uint8_t op, unsigned int block_size \
    ) { \
        launch_binary_broadcast_elementwise_op<TYPE>( \
            lbuf, rbuf, dbuf, loff, roff, doff, rank, size, \
            lstride, rstride, dstride, shape, op, block_size \
        ); \
    }

DECLARE_BINARY_BROADCAST_LAUNCHER(float,  f32)
DECLARE_BINARY_BROADCAST_LAUNCHER(double, f64)
DECLARE_BINARY_BROADCAST_LAUNCHER(uint8_t,  u8)
DECLARE_BINARY_BROADCAST_LAUNCHER(uint16_t, u16)
DECLARE_BINARY_BROADCAST_LAUNCHER(uint32_t, u32)
DECLARE_BINARY_BROADCAST_LAUNCHER(uint64_t, u64)
DECLARE_BINARY_BROADCAST_LAUNCHER(__uint128_t, u128)
DECLARE_BINARY_BROADCAST_LAUNCHER(int8_t,  i8)
DECLARE_BINARY_BROADCAST_LAUNCHER(int16_t, i16)
DECLARE_BINARY_BROADCAST_LAUNCHER(int32_t, i32)
DECLARE_BINARY_BROADCAST_LAUNCHER(int64_t, i64)
DECLARE_BINARY_BROADCAST_LAUNCHER(__int128_t, i128)
DECLARE_BINARY_BROADCAST_LAUNCHER(bool, boolean)
