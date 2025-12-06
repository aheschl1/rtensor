

use crate::{core::{tensor::TensorError, value::TensorValue, MetaTensor, MetaTensorView}, ops::{base::OpType}};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;

pub trait Backend<T: TensorValue> {
    type Buf;

    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError>;
    fn alloc(&self, len: usize) -> Result<Self::Buf, TensorError>;
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError>;
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError>;
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError>;
    fn len(&self, buf: &Self::Buf) -> usize;
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError>;
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError>;
    fn new() -> Self;

    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: (OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>;

    fn matmul_float32(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError>;

    fn matmul_float64(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError>;

    fn matmul_generic(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError>;

    /// Matrix multiplication of two tensors
    /// 
    /// # Parameters
    /// - `lhs_buf`: Buffer of the left-hand side tensor
    /// - `rhs_buf`: Buffer of the right-hand side tensor
    /// - `lhs_offset`: Offset into the left-hand side buffer
    /// - `rhs_offset`: Offset into the right-hand side buffer
    /// - `b`: Batch size
    /// - `m`: Number of rows in the left-hand side matrix
    /// - `k`: Number of columns in the left-hand side matrix (and rows in
    /// 
    /// # Requirements
    /// - The left-hand side tensor must have shape `[b, m, k]`
    /// - The right-hand side tensor must have shape `[b, k, n]`
    /// - The output tensor will have shape `[b, m, n]`
    /// - contiguous C-order layout with offset 0
    fn matmul(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError>{
        // switch on type id
        match std::any::TypeId::of::<T>() {
            x if x == std::any::TypeId::of::<f32>() => {
                self.matmul_float32(
                    lhs_buf,
                    rhs_buf,
                    lhs_offset,
                    rhs_offset,
                    b,
                    m,
                    k,
                    n,
                )
            },
            x if x == std::any::TypeId::of::<f64>() => {
                self.matmul_float64(
                    lhs_buf,
                    rhs_buf,
                    lhs_offset,
                    rhs_offset,
                    b,
                    m,
                    k,
                    n,
                )
            },
            _ => {
                self.matmul_generic(
                    lhs_buf,
                    rhs_buf,
                    lhs_offset,
                    rhs_offset,
                    b,
                    m,
                    k,
                    n,
                )
            }
        }
    }

    /// Broadcast two tensors into a destination tensor according to broadcasting rules
    /// 
    /// # Safety
    /// The caller must ensure that the pointers and metatensors are valid and that the destination
    /// tensor has the correct shape to hold the broadcasted result of the two source tensors.
    /// Dst and left may be the same buffer for in-place operations. It is vital that the caller ensures
    /// the stride for the left buffer contains no zeros in this case.
    unsafe fn broadcast(
        &self, 
        left: (*const Self::Buf, &MetaTensor), 
        right: (*const Self::Buf, &MetaTensor),
        dst: (*mut Self::Buf, &MetaTensor),
        op: OpType
    ) -> Result<(), TensorError>;

    fn apply_elementwise(&self, buf: &mut Self::Buf, op: (OpType, T), meta: &MetaTensor) -> Result<(), TensorError> {
        if meta.is_contiguous() {
            return self.apply_elementwise_contiguous(buf, op, meta.offset, meta.size())
        }

        let non_singleton_dims = meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            // essentially 1D
            // optimial because we can just stride along the single non-singleton dimension
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return self.apply_elementwise_1d_strided(
                buf, 
                op, 
                meta.offset, 
                dim_stride, 
                dim_size
            )
        }

        // worst case because we have to handle full nD striding
        // better than a full gather scatter
        self.apply_elementwise_nd(
            buf,
            op,
            meta.offset,
            meta.shape.as_slice(),
            meta.strides.as_ref(),
        )
    }
}