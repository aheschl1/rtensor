
use crate::{backend::{Backend, BackendElementwise}, core::{tensor::TensorError, value::{TensorValue, TensorValueElementwise}}, ops::elementwise::ElementwiseTensorOp};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Cpu;

impl<T: TensorValue> Backend<T> for Cpu {
    type Buf = Box<[T]>;



    fn alloc(&self, len: usize) -> Result<Box<[T]>, TensorError> {
        Ok(vec![T::default(); len].into())
    }

    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError> {
        if dst.len() != src.len() {
            return Err(TensorError::SizeMismatch);
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError> {
        Ok(*buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds
        )?)
    }

    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError> {
        match buf.get_mut(offset) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(TensorError::IdxOutOfBounds),
        }
    }
    
    fn from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError> {
        Ok(src)
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }

    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
    
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError> {
        Ok(src.clone())
    }
}

impl<T: TensorValue + TensorValueElementwise> BackendElementwise<T> for Cpu {
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        for i in start..(start + len) {
            bufptr[i] = op.apply(bufptr[i]);
        }
        Ok(())
    }
    
    fn apply_elementwise_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        start: usize,
        stride: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let mut idx = start;
        for _ in 0..len {
            bufptr[idx] = op.apply(bufptr[idx]);
            idx += stride;
        }
        Ok(())
    }
    
    fn apply_elementwise_scattered(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
        offsets: &[usize]
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        for &idx in offsets {
            bufptr[idx] = op.apply(bufptr[idx]);
        }
        Ok(())
    }
}