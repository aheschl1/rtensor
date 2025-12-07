
use crate::{backend::Backend, core::{idx::Idx, meta::is_contiguous_relaxed, primitives::TensorBase, value::TensorValue, Dim, MetaTensor, MetaTensorView, Shape, Strides, TensorView, TensorViewMut}};
use super::slice::{Slice, compute_sliced_parameters};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum TensorError {
    #[error("index out of bounds {0}")]
    IdxOutOfBounds(String),

    #[error("wrong number of dimensions {0}")]
    WrongDims(String),

    #[error("invalid tensor shape {0}")]
    InvalidShape(String),

    #[error("invalid dimension {0}")]
    InvalidDim(String),

    #[error("size mismatch between tensors {0}")]
    SizeMismatch(String),

    #[error("backend error: {0}")]
    BackendError(String),

    #[error("broadcast error: {0}")]
    BroadcastError(String),

    #[error("tensor is not contiguous {0}")]
    ContiguityError(String),

    #[cfg(feature = "cuda")]
    #[error("cuda error: {0}")]
    CudaError(String),
}

pub trait AsView<T: TensorValue, B: Backend<T>> {
    /// Returns an immutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view(&self) -> TensorView<'_, T, B>;
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError>;
}

pub trait AsViewMut<T: TensorValue, B: Backend<T>> : AsView<T, B> {
    /// Returns a mutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B>;
    fn view_as_mut(&'_ mut self, shape: Shape) -> Result<TensorViewMut<'_, T, B>, TensorError>;
}

pub trait AsTensor<T: TensorValue, B: Backend<T>> {
    /// Converts to an owned tensor, copying data.
    fn owned(&self) -> TensorBase<T, B>;
    fn contiguous(&self) -> TensorBase<T, B>;
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorBase<T, B> {
    fn view(&self) -> TensorView<'_, T, B> {
        TensorView::<T, B>::from_parts(
            &self.buf, 
            &self.backend, 
            self.meta.clone()
        )
    }
    
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError> {
        // collapse into shape
        if !is_contiguous_relaxed(&self.meta.shape, &self.meta.strides){
            return Err(TensorError::ContiguityError("Cannot view_as non contiguous tensor".to_string()));
        }

        panic!()
    }
} 

impl<T: TensorValue, B: Backend<T>> AsViewMut<T, B> for TensorBase<T, B> {
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> {
        TensorViewMut::<T, B>::from_parts(
            &mut self.buf, 
            &self.backend, 
            self.meta.clone()
        )
    }
    
    fn view_as_mut(&'_ mut self, shape: Shape) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        todo!()
    }
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorView<'_, T, B> 
{
    fn view(&self) -> TensorView<'_, T, B> {
        TensorView::from_parts(
            self.buf, 
            self.backend,
            self.meta.clone()
        )
    }
    
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError> {
        todo!()
    }
}

impl<T: TensorValue, B: Backend<T>> AsView<T, B> for TensorViewMut<'_, T, B> 
{
    fn view(&self) -> TensorView<'_, T, B> {
        TensorView::from_parts(
            self.buf,
            self.backend,
            self.meta.clone()
        )
    }
    
    fn view_as(&self, shape: Shape) -> Result<TensorView<'_, T, B>, TensorError> {
        todo!()
    }
}

impl<T: TensorValue, B: Backend<T>> AsViewMut<T, B> for TensorViewMut<'_, T, B> 
{
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> {
        TensorViewMut::from_parts(
            self.buf,
            self.backend,
            self.meta.clone()
        )
    }
    
    fn view_as_mut(&'_ mut self, shape: Shape) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        todo!()
    }
}


impl <T: TensorValue, B: Backend<T>> AsTensor<T, B> for TensorBase<T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        self.clone()
    }
    
    fn contiguous(&self) -> TensorBase<T, B> {
        if self.meta.is_contiguous() {
            // fast path: already contiguous
            self.clone()
        } else {
            view_to_contiguous(&self.meta, &self.buf, &self.backend).unwrap()
        }
    }
}

impl<'a, T: TensorValue, B: Backend<T>> AsTensor<T, B> for TensorView<'a, T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        view_to_contiguous(&self.meta, self.buf, self.backend).unwrap()
    }

    fn contiguous(&self) -> TensorBase<T, B> {
        self.owned()
    }
}

impl<'a, T: TensorValue, B: Backend<T>> AsTensor<T, B> for TensorViewMut<'a, T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        view_to_contiguous(&self.meta, self.buf, self.backend).unwrap()
    }

    fn contiguous(&self) -> TensorBase<T, B> {
        self.owned()
    }
}

#[inline]
fn view_to_contiguous<T: TensorValue, B: Backend<T>>(meta: &MetaTensor, raw: &B::Buf, backend: &B) -> Result<TensorBase<T, B>, TensorError> {
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

    fn permute(&self, dims: impl Into<Idx>) -> Result<TensorView<'_, T, B>, TensorError>;
    fn transpose(&self) -> Result<TensorView<'_, T, B>, TensorError>;
    fn unsqueeze_at(&self, dim: Dim) -> Result<TensorView<'_, T, B>, TensorError>;
    fn unsqueeze(&self) -> Result<TensorView<'_, T, B>, TensorError> {
        self.unsqueeze_at(0)
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

    fn permute_mut(&mut self, dims: impl Into<Idx>) -> Result<TensorViewMut<'_, T, B>, TensorError> ;
    fn transpose_mut(&mut self) -> Result<TensorViewMut<'_, T, B>, TensorError>;
    fn unsqueeze_at_mut(&mut self, dim: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError>;
    fn unsqueeze_mut(&mut self) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        self.unsqueeze_at_mut(0)
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
        view.backend.read(view.buf, idx)
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
        
        let v = TensorView::from_parts(view.buf, view.backend, MetaTensor::new(new_shape, new_stride, offset));
        Ok(v)
    }
    
    fn permute(&self, dims: impl Into<Idx>) -> Result<TensorView<'_, T, B>, TensorError> {
        let mut view = self.view();
        let (new_shape, new_stride) = compute_permuted_parameters(
            view.meta.shape(),
            view.meta.strides(),
            &dims.into()
        )?;

        view.meta.shape = new_shape;
        view.meta.strides = new_stride;

        Ok(view)
    }
    
    /// permute all dims
    fn transpose(&self) -> Result<TensorView<'_, T, B>, TensorError> {
        let rank = self.view().meta.rank();
        let dims: Idx = Idx::Coord((0..rank).rev().collect());
        self.permute(dims)
    }

    fn unsqueeze_at(&self, dim: Dim) -> Result<TensorView<'_, T, B>, TensorError> {
        let view = self.view();
        let (new_shape, new_strides) = compute_unsqueezed_parameters(
            view.meta.shape(),
            view.meta.strides(),
            dim
        )?;

        let res = TensorView::from_parts(
            view.buf, 
            view.backend, 
            MetaTensor::new(new_shape, new_strides, view.meta.offset())
        );
        Ok(res)
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
    
        Ok(TensorViewMut::from_parts(view.buf, view.backend, MetaTensor::new(new_shape, new_stride, offset)))
    }
    
    fn set<I: Into<Idx>>(&mut self, idx: I, value: T) -> Result<(), TensorError> {
        let view = self.view_mut();
        let idx = idx.into();
        let buf_idx = logical_to_buffer_idx(&idx, view.meta.strides(), view.meta.offset())?;
        view.backend.write(view.buf, buf_idx, value)
    }

    fn permute_mut(&mut self, dims: impl Into<Idx>) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let mut view = self.view_mut();
        let (new_shape, new_stride) = compute_permuted_parameters(
            view.meta.shape(),
            view.meta.strides(),
            &dims.into()
        )?;

        view.meta.shape = new_shape;
        view.meta.strides = new_stride;

        Ok(view)
    }

    fn transpose_mut(&mut self) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let rank = self.view().meta.rank();
        let dims: Idx = Idx::Coord((0..rank).rev().collect());
        self.permute_mut(dims)
    }

    fn unsqueeze_at_mut(&mut self, dim: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let view = self.view_mut();
        let (new_shape, new_strides) = compute_unsqueezed_parameters(
            view.meta.shape(),
            view.meta.strides(),
            dim
        )?;

        let res = TensorViewMut::from_parts(
            view.buf, 
            view.backend, 
            MetaTensor::new(new_shape, new_strides, view.meta.offset())
        );
        Ok(res)
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
#[inline]
fn logical_to_buffer_idx(idx: &Idx, stride: &Strides, offset: usize) -> Result<usize, TensorError> {
    match idx {
        Idx::Coord(idx) => {
            if idx.len() != stride.len() {
                Err(TensorError::WrongDims(format!(
                    "Index rank {} does not match tensor rank {}",
                    idx.len(),
                    stride.len()
                )))
            }else{
                let bidx = idx
                    .iter()
                    .zip(stride.iter())
                    .fold(offset as isize, |acc, (a, b)| acc + (*a as isize) * *b);
                if bidx < 0 {
                    return Err(TensorError::IdxOutOfBounds("Buffer index is negative".to_string()));
                }
                Ok(bidx as usize)
            }
        },
        Idx::Item => {
            if stride.is_empty() {
                Ok(offset)
            }else{
                Err(TensorError::WrongDims(format!(
                    "Item index used on non-scalar tensor with rank {}",
                    stride.len()
                )))
            }
        },
        Idx::At(i) => {
            // Single-dimensional index; only valid when there is exactly one dimension
            logical_to_buffer_idx(&Idx::Coord(vec![*i]), stride, offset)
        }
    }
}

#[inline]
fn compute_permuted_parameters(shape: &Shape, stride: &Strides, dims: &Idx) -> Result<(Shape, Strides), TensorError> {
    let rank = shape.len();
    let dims_vec = match dims {
        Idx::Coord(v) => v.clone(),
        Idx::At(i) => vec![*i],
        Idx::Item => vec![],
    };

    if dims_vec.len() != rank {
        return Err(TensorError::WrongDims(format!(
            "Permutation dims length {} does not match tensor rank {}",
            dims_vec.len(),
            rank
        )));
    }

    let mut new_shape = Vec::with_capacity(rank);
    let mut new_stride = Vec::with_capacity(rank);

    for &d in &dims_vec {
        if d >= rank {
            return Err(TensorError::InvalidDim(format!(
                "Permutation dim {} is out of bounds for tensor rank {}",
                d,
                rank
            )));
        }
        new_shape.push(shape[d]);
        new_stride.push(stride[d]);
    }

    Ok((new_shape.into(), new_stride.into()))
}

#[inline]
fn compute_unsqueezed_parameters(shape: &Shape, stride: &Strides, dim: Dim) -> Result<(Shape, Strides), TensorError> {
    if dim > shape.len() {
        return Err(TensorError::InvalidDim(format!(
            "Unsqueeze dim {} is out of bounds for tensor rank {}",
            dim,
            shape.len()
        )));
    }
    let mut new_strides = stride.clone();
    let mut new_shape = shape.clone();

    let lstr = *new_strides.0.get(dim).unwrap_or(&1);
    let lsh = *new_shape.0.get(dim).unwrap_or(&1) as isize;
    new_strides.0.insert(dim, lstr * lsh);
    new_shape.0.insert(dim, 1);

    Ok((new_shape, new_strides))
}