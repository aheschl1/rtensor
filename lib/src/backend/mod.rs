

use crate::{core::{meta::ContiguityTypes, tensor::TensorError, value::TensorValue, MetaTensor, MetaTensorView}, ops::base::OpType};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;
#[cfg(feature = "remote")]
pub mod remote;

pub trait Backend: Send + Sync + 'static + Clone {
    type Buf<T: TensorValue>: Send + Sync;

    fn device_type() -> crate::core::primitives::DeviceType;
    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, TensorError>;
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, TensorError>;
    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), TensorError>;
    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, TensorError>;
    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), TensorError>;
    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize;
    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, TensorError>;
    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, TensorError>;
    fn new() -> Self;

    fn apply_elementwise_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>;

    /// Broadcast two tensors into a destination tensor according to broadcasting rules
    /// 
    /// # Safety
    /// The caller must ensure that the pointers and metatensors are valid and that the destination
    /// tensor has the correct shape to hold the broadcasted result of the two source tensors.
    /// Dst and left may be the same buffer for in-place operations. It is vital that the caller ensures
    /// the stride for the left buffer contains no zeros in this case.
    unsafe fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &MetaTensor), 
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: OpType
    ) -> Result<(), TensorError>;

    fn apply_elementwise<T: TensorValue>(&self, buf: &mut Self::Buf<T>, op: (OpType, T), meta: &MetaTensor) -> Result<(), TensorError> {
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

pub trait BackendMatMul<T: TensorValue>: Backend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &MetaTensor),
        rhs: (&Self::Buf<T>, &MetaTensor),
        dst: &mut Self::Buf<T>,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: ContiguityTypes,
    ) -> Result<(), TensorError>;
}