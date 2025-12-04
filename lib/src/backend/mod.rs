

use crate::{core::{tensor::TensorError, value::{TensorValue}, MetaTensor, MetaTensorView}, ops::{binary::ElementwiseBinaryTensorOp, unary::ElementwiseUnaryTensorOp}};

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
        op: &ElementwiseUnaryTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseUnaryTensorOp<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: &ElementwiseUnaryTensorOp<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>;

    fn broadcast(
        &self, 
        left: (&Self::Buf, &MetaTensor), 
        right: (&Self::Buf, &MetaTensor),
        dst: (&mut Self::Buf, &MetaTensor),
        op: ElementwiseBinaryTensorOp<T>
    ) -> Result<(), TensorError>;

    fn apply_elementwise(&self, buf: &mut Self::Buf, op: ElementwiseUnaryTensorOp<T>, meta: &MetaTensor) -> Result<(), TensorError> {
        if meta.is_contiguous() {
            return self.apply_elementwise_contiguous(buf, &op, meta.offset, meta.size())
        }

        let non_singleton_dims = meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            // essentially 1D
            // optimial because we can just stride along the single non-singleton dimension
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return self.apply_elementwise_1d_strided(
                buf, 
                &op, 
                meta.offset, 
                dim_stride, 
                dim_size
            )
        }

        // worst case because we have to handle full nD striding
        // better than a full gather scatter
        self.apply_elementwise_nd(
            buf,
            &op,
            meta.offset,
            meta.shape.as_slice(),
            &meta.stride,
        )

    }
}