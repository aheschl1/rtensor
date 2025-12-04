
use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, value::TensorValue, Dim, MetaTensor, Stride, TensorView, TensorViewMut}};
use super::slice::{Slice, compute_sliced_parameters};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum TensorError {
    #[error("index out of bounds")]
    IdxOutOfBounds,

    #[error("wrong number of dimensions")]
    WrongDims,

    #[error("invalid tensor shape")]
    InvalidShape,

    #[error("invalid dimension")]
    InvalidDim,

    #[error("size mismatch between tensors")]
    SizeMismatch,

    #[error("backend error: {0}")]
    BackendError(String),

    #[error("broadcast error: {0}")]
    BroadcastError(String),

    #[cfg(feature = "cuda")]
    #[error("cuda error: {0}")]
    CudaError(String),
}

pub trait AsView<T: TensorValue, B: Backend<T>> {
    /// Returns an immutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view(&self) -> TensorView<'_, T, B>;
}

pub trait AsViewMut<T: TensorValue, B: Backend<T>> : AsView<T, B> {
    /// Returns a mutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view_mut<'a>(&'a mut self) -> TensorViewMut<'a, T, B>;
}

pub trait AsTensor<T: TensorValue, B: Backend<T>> {
    /// Converts to an owned tensor, copying data.
    fn owned(&self) -> TensorBase<T, B>;
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorBase<T, B> {
    fn view(&self) -> TensorView<'_, T, B> {
        TensorView::<T, B>::from_parts(
            &self.raw, 
            &self.backend, 
            self.meta.clone()
        )
    }
} 

impl<T: TensorValue, B: Backend<T>> AsViewMut<T, B> for TensorBase<T, B> {
    fn view_mut<'a>(&'a mut self) -> TensorViewMut<'a, T, B> {
        TensorViewMut::<T, B>::from_parts(
            &mut self.raw, 
            &self.backend, 
            self.meta.clone()
        )
    }
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorView<'_, T, B> 
{
    fn view(&self) -> TensorView<'_, T, B> {
        TensorView::from_parts(
            self.raw, 
            self.backend,
            self.meta.clone()
        )
    }
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorViewMut<'_, T, B> 
{
    fn view(&self) -> TensorView<'_, T, B> {
        TensorView::from_parts(
            self.raw,
            self.backend,
            self.meta.clone()
        )
    }
}

impl<T: TensorValue, B: Backend<T>> AsViewMut<T, B> for TensorViewMut<'_, T, B> 
{
    fn view_mut<'a>(&'a mut self) -> TensorViewMut<'a, T, B> {
        TensorViewMut::from_parts(
            self.raw,
            self.backend,
            self.meta.clone()
        )
    }
}


impl <T: TensorValue, B: Backend<T>> AsTensor<T, B> for TensorBase<T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        self.clone()
    }
}

impl<'a, T: TensorValue, B: Backend<T>> AsTensor<T, B> for TensorView<'a, T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        view_to_owned(&self.meta, self.raw, self.backend).unwrap()
    }
}

impl<'a, T: TensorValue, B: Backend<T>> AsTensor<T, B> for TensorViewMut<'a, T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        view_to_owned(&self.meta, self.raw, self.backend).unwrap()
    }
}

#[inline]
fn view_to_owned<T: TensorValue, B: Backend<T>>(meta: &MetaTensor, raw: &B::Buf, backend: &B) -> Result<TensorBase<T, B>, TensorError> {
    let size = meta.size();
    let new_backend = B::new();
    let mut new_buf = new_backend.alloc(size)?;
    
    // Copy element by element from the view to the new contiguous buffer
    // The view might be non-contiguous (e.g., a column slice), so we iterate
    // through all logical positions and copy to sequential positions in the new buffer
    for (new_idx, old_offset) in meta.iter_offsets().enumerate() {
        let value = backend.read(raw, old_offset)?;
        new_backend.write(&mut new_buf, new_idx, value)?;
    }
    
    // Create a new tensor with contiguous layout (standard row-major stride)
    let new_shape = meta.shape().clone();
    let new_stride = super::shape_to_stride(&new_shape);
    let new_meta = MetaTensor::new(new_shape, new_stride, 0);
    
    Ok(TensorBase::from_parts(new_backend, new_buf, new_meta))
}

pub trait TensorAccess<T: TensorValue, B: Backend<T>>: Sized {
    /// Get element at given index
    fn get<I: Into<Idx>>(&self, idx: I) -> Result<T, TensorError>;

    fn item(&self) -> Result<T, TensorError> {
        self.get(Idx::Item)
    }
    /// Create a slice/view of the tensor along a specific dimension at a given index
    fn slice<S: Into<Slice>>(&self, dim: Dim, idx: S) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized;
    /// take a slice at given index
    fn slice_at(&self, dim: Dim, at: usize) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized{
        self.slice(dim, at)
    }
}

pub trait TensorAccessMut<T: TensorValue, B: Backend<T>>: TensorAccess<T, B> {
    /// Slice mutable tensor to get a mutable view
    fn slice_mut<S: Into<Slice>>(&mut self, dim: Dim, idx: S) -> Result<TensorViewMut<'_, T, B>, TensorError> where Self: Sized;
    /// sets a value at given index
    fn set<I: Into<Idx>>(&mut self, idx: I, value: T) -> Result<(), TensorError>;
    /// take a mutable slice at given index
    fn slice_at_mut(&mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError> where Self: Sized{
        self.slice_mut(dim, idx)
    }
}

impl<T: TensorValue, B: Backend<T>, V> TensorAccess<T, B> for V
where B: Backend<T>, V: AsView<T, B>
{
    /// Returns a reference to the element at a logical index, converting
    /// coordinates into a buffer position via stride and offset.
    ///
    /// Errors
    /// - `WrongDims` if the index rank doesn't match the tensor rank.
    /// - `IdxOutOfBounds` if the computed buffer index is outside the backing slice.
    fn get<I: Into<Idx>>(&self, idx: I) -> Result<T, TensorError> {
        let view = self.view();
        let idx = logical_to_buffer_idx(&idx.into(), view.meta.strides(), view.meta.offset())?;
        view.backend.read(view.raw, idx)
    }

    /// Creates a new immutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    fn slice<S: Into<Slice>>(&self, dim: Dim, idx: S) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized {
        let view = self.view();
        let (new_shape, new_stride, offset) = compute_sliced_parameters(
            view.meta.shape(), 
            view.meta.strides(), 
            view.meta.offset(),
            dim, 
            idx
        )?;
        
        let v = TensorView::from_parts(view.raw, view.backend, MetaTensor::new(new_shape, new_stride, offset));
        Ok(v)
    }
}

impl<T: TensorValue, B: Backend<T>, V> TensorAccessMut<T, B> for V
where V: AsViewMut<T, B>
{
    /// Creates a new mutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    fn slice_mut<S: Into<Slice>>(&mut self, dim: Dim, idx: S) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let view = self.view_mut();
        let (new_shape, new_stride, offset) =
            compute_sliced_parameters(view.meta.shape(), view.meta.strides(), view.meta.offset(), dim, idx)?;
    
        Ok(TensorViewMut::from_parts(view.raw, view.backend, MetaTensor::new(new_shape, new_stride, offset)))
    }
    
    fn set<I: Into<Idx>>(&mut self, idx: I, value: T) -> Result<(), TensorError> {
        let view = self.view_mut();
        let idx = idx.into();
        let buf_idx = logical_to_buffer_idx(&idx, view.meta.strides(), view.meta.offset())?;
        view.backend.write(view.raw, buf_idx, value)
    }

}

/// Converts a logical index (coordinate, single position, or scalar) into a
/// linear buffer index using the provided stride and offset.
///
/// Behavior
/// - `Coord(&[d0, d1, ...])` computes `offset + sum(di*stride[i])`.
/// - `At(i)` is treated as `Coord(&[i])`.
/// - `Item` is only valid for scalars (rank 0).
///
/// Errors
/// - `WrongDims` if index rank differs from stride length, or `Item` is used on non-scalars.
/// - `IdxOutOfBounds` is not checked here (caller validates against buffer length).
fn logical_to_buffer_idx(idx: &Idx, stride: &Stride, offset: usize) -> Result<usize, TensorError> {
    match idx {
        Idx::Coord(idx) => {
            if idx.len() != stride.len() {
                Err(TensorError::WrongDims)
            }else{
                let bidx = idx
                    .iter()
                    .zip(stride)
                    .fold(offset as isize, |acc, (a, b)| acc + (*a as isize) * *b);
                if bidx < 0 {
                    return Err(TensorError::IdxOutOfBounds);
                }
                Ok(bidx as usize)
            }
        },
        Idx::Item => {
            if stride.is_empty() {
                Ok(offset)
            }else{
                Err(TensorError::WrongDims)
            }
        },
        Idx::At(i) => {
            // Single-dimensional index; only valid when there is exactly one dimension
            logical_to_buffer_idx(&Idx::Coord(vec![*i]), stride, offset)
        }
    }
}
