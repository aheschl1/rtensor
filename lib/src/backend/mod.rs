

use crate::{core::{meta::ContiguityTypes, tensor::TensorError, value::TensorValue, MetaTensor, MetaTensorView}, ops::base::BinaryOpType};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;
#[cfg(feature = "remote")]
pub mod remote;

macro_rules! elementwise_binary_dispatch {
    (
        $self:expr,
        $buf:expr,
        $op:expr,
        $meta:expr,
        contiguous = $contig:ident,
        strided1d  = $strided1d:ident,
        nd         = $nd:ident
    ) => {{
        if $meta.is_contiguous() {
            return $self.$contig(
                $buf,
                $op,
                $meta.offset,
                $meta.size(),
            );
        }

        let non_singleton_dims = $meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return $self.$strided1d(
                $buf,
                $op,
                $meta.offset,
                dim_stride,
                dim_size,
            );
        }

        $self.$nd(
            $buf,
            $op,
            $meta.offset,
            $meta.shape.as_slice(),
            $meta.strides.as_ref(),
        )
    }};
}

macro_rules! elementwise_unary_dispatch {
    (
        $self:expr,
        $buf:expr,
        $meta:expr,
        contiguous = $contig:ident,
        strided1d  = $strided1d:ident,
        nd         = $nd:ident
    ) => {{
        if $meta.is_contiguous() {
            return $self.$contig(
                $buf,
                $meta.offset,
                $meta.size(),
            );
        }

        let non_singleton_dims = $meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return $self.$strided1d(
                $buf,
                $meta.offset,
                dim_stride,
                dim_size,
            );
        }

        $self.$nd(
            $buf,
            $meta.offset,
            $meta.shape.as_slice(),
            $meta.strides.as_ref(),
        )
    }};
}


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
    /// Broadcast two tensors into a destination tensor according to broadcasting rules
    /// 
    /// # Safety
    /// The caller must ensure that the pointers and metatensors are valid and that the destination
    /// tensor has the correct shape to hold the broadcasted result of the two source tensors.
    /// Dst and left may be the same buffer for in-place operations. It is vital that the caller ensures
    /// the stride for the left buffer contains no zeros in this case.
    fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &MetaTensor), 
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType
    ) -> Result<(), TensorError>;
    fn apply_elementwise_binary_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (BinaryOpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_binary_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (BinaryOpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_elementwise_binary_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (BinaryOpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>;

    fn apply_elementwise_binary<T: TensorValue>(&self, buf: &mut Self::Buf<T>, op: (BinaryOpType, T), meta: &MetaTensor) -> Result<(), TensorError> {
        elementwise_binary_dispatch!(
            self,
            buf,
            op,
            meta,
            contiguous = apply_elementwise_binary_contiguous,
            strided1d  = apply_elementwise_binary_1d_strided,
            nd         = apply_elementwise_binary_nd
        )
    }

    fn apply_neg_contiguous<T: TensorValue + std::ops::Neg<Output = T>>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_neg_1d_strided<T: TensorValue + std::ops::Neg<Output = T>>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError>;

    fn apply_neg_nd<T: TensorValue + std::ops::Neg<Output = T>>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>;

    fn apply_neg<T: TensorValue + std::ops::Neg<Output = T>>(&self, buf: &mut Self::Buf<T>, meta: &MetaTensor) -> Result<(), TensorError> {
        elementwise_unary_dispatch!(
            self,
            buf,
            meta,
            contiguous = apply_neg_contiguous,
            strided1d  = apply_neg_1d_strided,
            nd         = apply_neg_nd
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
