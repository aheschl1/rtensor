use std::{marker::PhantomData, ops::{Index, IndexMut}};

use crate::ndarray::{Dim, Shape, Stride, TensorOwned, TensorView, TensorViewBase, TensorViewMut};


#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum TensorError {
    IdxOutOfBounds,
    WrongDims,
    InvalidShape,
    InvalidDim
}

pub enum Idx {
    Coord(Vec<usize>),
    At(usize),
    Item
}

pub trait ViewableTensor<T: Sized> {
    fn view(&self) -> TensorView<'_, T>;
}
pub trait ViewableTensorMut<T: Sized>: ViewableTensor<T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T>;
}

impl<T: Sized> ViewableTensor<T> for TensorOwned<T> {
    fn view(&self) -> TensorView<'_, T> {
        TensorView{
            raw: &self.raw,
            stride: self.stride.clone(),
            offset: 0,
            shape: self.shape.clone(),
            _l: PhantomData,
            _t: PhantomData,
        }
    }
}

impl<'a, T, B> ViewableTensor<T> for TensorViewBase<'a, T, B> 
where B: AsRef<[T]> + 'a
{
    fn view(&self) -> TensorView<'_, T> {
        TensorView{
            raw: self.raw.as_ref(),
            stride: self.stride.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
            _l: PhantomData,
            _t: PhantomData,
        }
    }
}

impl<T: Sized> ViewableTensorMut<T> for TensorViewMut<'_, T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut{
            raw: self.raw,
            stride: self.stride.clone(),
            offset: self.offset,
            shape: self.shape.clone(),
            _l: PhantomData,
            _t: PhantomData,
        }
    }
}

impl<T: Sized> ViewableTensorMut<T> for TensorOwned<T> {
    fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut{
            raw: &mut self.raw,
            stride: self.stride.clone(),
            offset: 0,
            shape: self.shape.clone(),
            _l: PhantomData,
            _t: PhantomData,
        }
    }
}

pub trait Tensor<T>: Sized {
    /// Get the size of a specific dimension
    fn dim(&self, dim: Dim) -> Dim;
    /// Get the shape of the tensor
    fn shape(&self) -> Shape;
    /// Get the number of dimensions
    fn dims(&self) -> usize {
        self.stride().len()
    }
    /// Get element at given index
    fn get(&self, idx: &Idx) -> Result<&T, TensorError>;
    /// Check if tensor is a scalar (0-dimensional)
    fn is_scalar(&self) -> bool {
        self.stride().is_empty()
    }

    fn item(&self) -> Result<&T, TensorError> {
        self.get(&Idx::Item)
    }
    /// Get the stride of the tensor
    fn stride(&self) -> Stride;
    /// Get the total number of elements in the tensor
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
    /// Create a slice/view of the tensor along a specific dimension at a given index
    fn slice<'a>(&'a self, dim: Dim, idx: Dim) -> Result<TensorView<'a, T>, TensorError> where Self: Sized;

    fn is_row(&self) -> bool {
        self.shape().len() == 2 && self.shape()[0] == 1
    }

    fn is_column(&self) -> bool {
        self.shape().len() == 1
    }
}
pub trait TensorMut<T>: Tensor<T> {
    /// Get mutable element at given index
    fn get_mut(&mut self, idx: &Idx) -> Result<&mut T, TensorError>;
    /// Slice mutable tensor to get a mutable view
    fn slice_mut<'a>(&'a mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'a, T>, TensorError> where Self: Sized;
    /// sets a value at given index
    fn set(&mut self, idx: &Idx, value: T) -> Result<(), TensorError> {
        let slot = self.get_mut(idx)?;
        *slot = value;
        Ok(())
    }
}

impl<T, W> Tensor<T> for W 
    where W: ViewableTensor<T>
{
    fn shape(&self) -> Shape {
        self.view().shape.clone()
    }

    fn stride(&self) -> Stride {
        self.view().stride.clone()
    }

    fn get(&self, idx: &Idx) -> Result<&T, TensorError> {
        match idx {
            Idx::Coord(idx) => {
                if idx.len() != self.dims() {
                    return Err(TensorError::WrongDims)
                }else{
                    let view = self.view();
                    let bidx = idx
                        .iter()
                        .zip(&view.stride)
                        .fold(view.offset, |acc, (a, b)| acc + *a*b);
                    view.raw.get(bidx).ok_or(TensorError::IdxOutOfBounds)
                }
            },
            Idx::Item => {
                if self.is_scalar() {
                    self.get(&Idx::Coord(vec![]))
                }else{
                    Err(TensorError::WrongDims)
                }
            },
            Idx::At(i) => {
                self.get(&Idx::Coord(vec![*i]))
            }
        }
    }

    fn slice(&self, dim: Dim, idx: Dim) -> Result<TensorView<'_, T>, TensorError> where Self: Sized {
        if dim >= self.dims() {
            return Err(TensorError::InvalidDim);
        }
        if idx >= self.dim(dim) {
            return Err(TensorError::IdxOutOfBounds);
        }
        let mut new_shape = self.shape();
        new_shape.remove(dim);
        let mut new_stride = self.stride();
        new_stride.remove(dim);
        let mut v = self.view();
        v.offset = v.offset + v.stride()[dim] * idx;
        v.stride = new_stride.clone();
        v.shape = new_shape.clone();
        Ok(v)
    }
    
    fn dim(&self, dim: Dim) -> Dim {
        self.shape()[dim]
    }
}

impl<T, W> TensorMut<T> for W
    where W: ViewableTensorMut<T>
{
    fn get_mut(&mut self, idx: &Idx) -> Result<&mut T, TensorError> {
        match idx {
            Idx::Coord(idx) => {
                if idx.len() != self.dims() {
                    return Err(TensorError::WrongDims)
                }else{
                    let view = self.view_mut();
                    let bidx = idx
                        .iter()
                        .zip(&view.stride)
                        .fold(view.offset, |acc, (a, b)| acc + *a*b);
                    view.raw.get_mut(bidx).ok_or(TensorError::IdxOutOfBounds)
                }
            },
            Idx::Item => {
                if self.is_scalar() {
                    self.get_mut(&Idx::Item)
                }else{
                    Err(TensorError::WrongDims)
                }
            },
            Idx::At(i) => {
                self.get_mut(&Idx::Coord(vec![*i]))
            },
        }
    }

    fn slice_mut<'a>(&'a mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'a, T>, TensorError> where Self: Sized {
        if dim >= self.dims() {
            return Err(TensorError::InvalidDim);
        }
        if idx >= self.dim(dim) {
            return Err(TensorError::IdxOutOfBounds);
        }
        let mut new_shape = self.shape().clone();
        new_shape.remove(dim);
        let mut new_stride = self.stride().clone();
        new_stride.remove(dim);

        let mut v = self.view_mut();
        v.offset = v.offset + v.stride()[dim] * idx;
        v.stride = new_stride.clone();
        v.shape = new_shape.clone();
        Ok(v)
    }
}


impl<'a, T, S: Into<Vec<usize>>> Index<S> for TensorOwned<T> {
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        self.get(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, B, S: Into<Vec<usize>>> Index<S> for TensorViewBase<'a, T, B> 
    where B: AsRef<[T]> + 'a
{
    type Output = T;

    fn index(&self, index: S) -> &Self::Output {
        self.get(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, S: Into<Vec<usize>>> IndexMut<S> for TensorOwned<T> {
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        self.get_mut(&Idx::Coord(index.into())).unwrap()
    }
}

impl<'a, T, S: Into<Vec<usize>>> IndexMut<S> for TensorViewMut<'_, T> {
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        self.get_mut(&Idx::Coord(index.into())).unwrap()
    }
}

