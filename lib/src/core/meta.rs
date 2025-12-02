use std::ops::{Index, IndexMut};

use crate::{backend::Backend, core::{primitives::TensorBase, value::TensorValue, TensorViewMut}};

use super::primitives::TensorView;

pub type Dim = usize;
pub type Stride = Vec<isize>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Shape(pub Vec<Dim>);


impl Shape {
    pub fn empty() -> Self {
        Shape(vec![])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Dim> {
        self.0.iter()
    }

    pub fn as_slice(&self) -> &[Dim] {
        &self.0
    }
    pub fn remove(&mut self, index: usize) -> Dim {
        self.0.remove(index)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, dim: &Dim) -> bool {
        self.0.contains(dim)
    }
}

impl Index<usize> for Shape {
    type Output = Dim;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl PartialEq<Vec<Dim>> for Shape {
    fn eq(&self, other: &Vec<Dim>) -> bool {
        &self.0 == other
    }
}


#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MetaTensor {
    pub shape: Shape,
    /// Affine matrix wo offset
    pub stride: Stride,
    pub offset: usize,
}

impl MetaTensor {
    /// Creates tensor metadata with explicit shape, stride and offset.
    pub fn new(shape: impl Into<Shape>, stride: Stride, offset: usize) -> Self {
        Self { shape: shape.into(), stride, offset }
    }

    /// Returns true when the metadata describes a scalar (rank 0).
    pub fn is_scalar(&self) -> bool { self.stride.is_empty() }
    /// Returns true when the metadata describes a 1xN row tensor.
    pub fn is_row(&self) -> bool { self.shape.len() == 2 && self.shape[0] == 1 }
    /// Returns true when the metadata describes a 1-D column tensor.
    pub fn is_column(&self) -> bool { self.shape.len() == 1 }
    /// Number of dimensions (rank).
    pub fn dims(&self) -> usize { self.shape.len() }
    /// Total number of elements (product of all dimensions).
    pub fn size(&self) -> usize { self.shape.iter().product() }
    /// Whether the layout is contiguous in row-major order, allowing strides of
    /// one for non-singleton dims and ignoring dims of size 1.
    pub fn is_contiguous(&self) -> bool { is_contiguous_relaxed(&self.shape, &self.stride) }
    /// Borrow the shape vector.
    pub fn shape(&self) -> &Shape { &self.shape }
    /// Borrow the stride vector.
    pub fn stride(&self) -> &Stride { &self.stride }
    /// Return the starting offset (in elements) into the underlying buffer.
    pub fn offset(&self) -> usize { self.offset }
    /// Returns the size of a single dimension by index.
    pub fn dim(&self, dim: Dim) -> Dim { self.shape[dim] }
    pub fn rank(&self) -> usize { self.shape.len() }
    /// Returns an iterator over all offsets in the underlying buffer for this tensor/view.
    pub fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let offset = self.offset;
        TensorOffsetIterator::new(self.shape.as_slice(), self.stride.as_slice(), offset)
    }
    pub fn iter_coords(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        CoordIter::new(self.shape.as_slice())
    }
    pub fn ith_offset(&self, i: usize) -> usize {
        let mut idx = i;
        let mut coords = vec![0; self.rank()];

        for d in (0..self.rank()).rev() {
            let dim = self.shape[d];
            coords[d] = idx % dim;
            idx /= dim;
        }

        let mut offset = self.offset as isize;
        for (c, stride) in coords.iter().zip(self.stride.iter()) {
            offset += *c as isize * *stride;
        }

        offset as usize
    }
}

pub struct TensorOffsetIterator<'a> {
    shape:   &'a [usize],
    stride:  &'a [isize],
    offset0: isize,
    index:   Vec<usize>,
    started: bool,
    done:    bool,
}

impl<'a> TensorOffsetIterator<'a> {
    pub fn new(shape: &'a [usize], stride: &'a [isize], base_offset: usize) -> Self {
        let rank = shape.len();
        TensorOffsetIterator {
            shape,
            stride,
            offset0: base_offset as isize,
            index: vec![0; rank],
            started: false,
            done: false,
        }
    }
}

impl<'a> Iterator for TensorOffsetIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.shape.is_empty() {
            if self.started {
                self.done = true;
                return None;
            }
            self.started = true;
            return Some(self.offset0 as usize);
        }

        if !self.started {
            self.started = true;
        }

        let mut off = self.offset0;
        for (coord, stride) in self.index.iter().zip(self.stride.iter()) {
            off += (*coord as isize) * *stride;
        }
        debug_assert!(off >= 0, "Negative offset in ND iterator is illegal.");
        let phys = off as usize;

        for dim in (0..self.index.len()).rev() {
            self.index[dim] += 1;
            if self.index[dim] < self.shape[dim] {
                return Some(phys);
            }
            self.index[dim] = 0;
        }

        self.done = true;
        Some(phys)
    }
}

pub struct CoordIter<'a> {
    shape: &'a [usize],
    coords: Vec<usize>,
    started: bool,
    done: bool,
}

impl<'a> CoordIter<'a> {
    pub fn new(shape: &'a [usize]) -> Self {
        let r = shape.len();
        Self {
            shape,
            coords: vec![0; r],
            started: false,
            done: false,
        }
    }
}

impl<'a> Iterator for CoordIter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.shape.is_empty() {
            if self.started {
                self.done = true;
                return None;
            }
            self.started = true;
            return Some(vec![]);
        }

        if !self.started {
            self.started = true;
            return Some(self.coords.clone());
        }

        // Multi-dimensional counting
        for d in (0..self.coords.len()).rev() {
            self.coords[d] += 1;
            if self.coords[d] < self.shape[d] {
                return Some(self.coords.clone());
            }
            self.coords[d] = 0;
        }

        self.done = true;
        Some(self.coords.clone())
    }
}


/// Computes the standard row-major stride for a given shape.
pub fn shape_to_stride(shape: &Shape) -> Stride {
    let mut stride: Vec<isize> = vec![1; shape.len()];
    for i in (0..shape.len()).rev() {
        if i < shape.len() - 1 {
            stride[i] = stride[i + 1] * shape[i + 1] as isize;
        }
    }
    stride 
}


/// Checks whether a layout (shape/stride) is contiguous in a relaxed sense:
/// ignores singleton dimensions and accepts empty shapes.
#[inline]
pub(crate) fn is_contiguous_relaxed(shape: &Shape, stride: &Stride) -> bool {
    if shape.is_empty() { return true; }
    if shape.contains(&0) { return true; }
    if shape.len() != stride.len() { return false; }

    let mut expected = 1usize;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        let s = stride[i];
        if dim != 1 {
            if s.unsigned_abs() != expected { return false; }
            expected = expected.saturating_mul(dim);
        }
    }
    true
}

/// Read-only metadata view for tensors and views.
pub trait MetaTensorView {

    fn meta(&self) -> &MetaTensor;
    /// Borrow the shape vector.
    fn shape(&self) -> &Shape { &self.meta().shape }
    /// Borrow the stride vector.
    fn stride(&self) -> &Stride { &self.meta().stride }
    /// Starting offset (in elements) into the underlying buffer.
    fn offset(&self) -> usize { self.meta().offset }
    /// Number of dimensions (rank).
    fn dims(&self) -> usize { self.meta().dims() }
    /// Size of one dimension by index.
    fn dim(&self, dim: Dim) -> Dim { self.meta().dim(dim) }
    /// Total number of elements (product of all dimensions).
    fn size(&self) -> usize { self.meta().size() }
    /// True for rank-0 (scalar) tensors.
    fn is_scalar(&self) -> bool { self.meta().is_scalar() }
    /// True for 1xN row tensors.
    fn is_row(&self) -> bool { self.meta().is_row() }
    /// True for 1-D column tensors.
    fn is_column(&self) -> bool { self.meta().is_column() }

    fn rank(&self) -> usize { self.meta().rank() }
    /// Whether the layout is contiguous in row-major order under relaxed rules
    /// (ignoring singleton dimensions).
    fn is_contiguous(&self) -> bool { self.meta().is_contiguous() }

    fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ { self.meta().iter_offsets() }

    /// Returns a vector of (dim_index, dim_size, dim_stride) for all non-singleton dimensions.
    fn non_singleton_dims(&self) -> Vec<(usize, Dim, isize)> {
        self.shape().iter()
            .zip(self.stride().iter())
            .enumerate()
            .filter(|(_, (&d, _))| d > 1)
            .map(|(i, (&d, &s))| (i, d, s))
            .collect()
    }
}

impl MetaTensorView for MetaTensor {
    fn meta(&self) -> &MetaTensor {
        self
    }
}

impl<B, T: TensorValue> MetaTensorView for TensorBase<T, B> 
where
    B: Backend<T>,
{
    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

impl<T: TensorValue, B> MetaTensorView for TensorView<'_, T, B>
where
    B: Backend<T>,
{
    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

impl <T: TensorValue, B> MetaTensorView for TensorViewMut<'_, T, B>
where
    B: Backend<T>,
{
    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}


impl From<Shape> for Vec<Dim> {
    fn from(val: Shape) -> Self {
        val.0
    }
}

impl From<Vec<Dim>> for Shape {
    fn from(v: Vec<Dim>) -> Self {
        Shape(v)
    }
}

impl AsRef<[Dim]> for Shape {
    fn as_ref(&self) -> &[Dim] {
        &self.0
    }
}

impl From<(Dim,)> for Shape {
    fn from(t: (Dim,)) -> Self {
        Shape(vec![t.0])
    }
}

impl From<(Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1])
    }
}

impl From<(Dim, Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1, t.2])
    }
}

impl From<(Dim, Dim, Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim, Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1, t.2, t.3])
    }
}

impl From<(Dim, Dim, Dim, Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim, Dim, Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1, t.2, t.3, t.4])
    }
}