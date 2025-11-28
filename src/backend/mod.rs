

use crate::{core::{meta::MemRegion, tensor::TensorError, value::{TensorValue, TensorValueElementwise}}, ops::elementwise::ElementwiseTensorOp};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;

pub trait Backend<T: TensorValue> {
    type Buf;

    fn from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError>;
    fn alloc(&self, len: usize) -> Result<Self::Buf, TensorError>;
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError>;
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError>;
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError>;
    fn len(&self, buf: &Self::Buf) -> usize;
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError>;
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError>;

    // fn apply_unary(&self, buf: &mut Self::Buf, op: UnaryTensorOp<T>, offsets: Vec<usize>) -> Result<(), TensorError>    
    fn new() -> Self;
}

pub trait BackendElementwise<T: TensorValue + TensorValueElementwise>: Backend<T> {
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>;
    fn apply_elementwise_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        stride: usize,
        len: usize
    ) -> Result<(), TensorError>;
    fn apply_elementwise_scattered(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        offsets: &[usize]
    ) -> Result<(), TensorError>;

    fn apply_elementwise(&self, buf: &mut Self::Buf, op: ElementwiseTensorOp<T>, regions: &[MemRegion]) -> Result<(), TensorError>{
        for region in regions {
            match region {
                MemRegion::Contiguous{start, len} => {
                    self.apply_elementwise_contiguous(buf, &op, *start, *len)?
                },
                MemRegion::Strided{start, stride, len} => {
                    self.apply_elementwise_strided(buf, &op, *start, *stride, *len)?
                },
                MemRegion::Scattered{offsets} => {
                    self.apply_elementwise_scattered(buf, &op, offsets)?
                },
            }
        }
        Ok(())
    }
}