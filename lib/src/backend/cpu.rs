
use crate::{backend::Backend, core::{meta::TensorOffsetIterator, tensor::TensorError, value::TensorValue, MetaTensor}, ops::{binary::ElementwiseBinaryTensorOp, unary::ElementwiseUnaryTensorOp}};

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
    
    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError> {
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


        fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseUnaryTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        for item in bufptr.iter_mut().skip(start).take(len) {
            *item = op.apply(*item);
        }
        Ok(())
    }
    
    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseUnaryTensorOp<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let mut idx: isize = offset as isize;
        for _ in 0..len {
            bufptr[idx as usize] = op.apply(bufptr[idx as usize]);
            idx += stride;
        }
        Ok(())
    }
    
    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: &ElementwiseUnaryTensorOp<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let iterator = TensorOffsetIterator::new(
            shape,
            stride,
            offset,
        );
        for idx in iterator {
            bufptr[idx] = op.apply(bufptr[idx]);
        }
        Ok(())
    }

    fn broadcast(
        &self, 
        left: (&Self::Buf, &MetaTensor), 
        right: (&Self::Buf, &MetaTensor),
        dst: (&mut Self::Buf, &MetaTensor),
        op: ElementwiseBinaryTensorOp<T>
    ) -> Result<(), TensorError> {
        // this is a stupid algorithm which is O(rank*size)
        // it can be optimized to O(size) later
        // a cleaner O(rank*size) algorithm just uses the coordinate iterator
        // and converts the, to full offsets
        let (left_buf, left_meta) = left;
        let (right_buf, right_meta) = right;
        let (dst_buf, dst_meta) = dst;

        let rank = dst_meta.rank();

        let sl = left_meta.stride();
        let sr = right_meta.stride();
        let sd = dst_meta.stride();

        let mut ol = left_meta.offset() as isize;
        let mut or = right_meta.offset() as isize;
        let mut od = dst_meta.offset() as isize;

        let mut coords = vec![0; rank];

        let mut first = true;

        for new_coord in dst_meta.iter_coords() {
            if first {
                first = false;
            } else{
                for d in (0..rank).rev() {
                    if new_coord[d] != coords[d] {
                        let delta = new_coord[d] as isize - coords[d] as isize;
                        ol += delta * sl[d];
                        or += delta * sr[d];
                        od += delta * sd[d];
                    }
                }
            }
            coords = new_coord;
            debug_assert!(od >= 0);
            debug_assert!(ol >= 0);
            debug_assert!(or >= 0);
            dst_buf[od as usize] = op.apply(left_buf[ol as usize], right_buf[or as usize]);
        }

        Ok(())
    }
}