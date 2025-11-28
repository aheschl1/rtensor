use crate::{backend::Backend, core::{primitives::TensorBase, value::TensorValue, TensorViewMut}};

use super::primitives::TensorView;

pub type Dim = usize;
pub type Stride = Vec<usize>;
pub type Shape = Vec<Dim>;

pub enum MemRegion {
    Contiguous { start: usize, len: usize },
    Strided { start: usize, stride: usize, len: usize },
    Scattered { offsets: Vec<usize> },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MetaTensor {
    shape: Shape,
    stride: Stride,
    offset: usize,
}

impl MetaTensor {
    /// Creates tensor metadata with explicit shape, stride and offset.
    pub fn new(shape: Shape, stride: Stride, offset: usize) -> Self {
        Self { shape, stride, offset }
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
    /// Returns all contiguous memory ranges (in elements) that cover this tensor/view.
    pub fn mem_regions(&self) -> Vec<MemRegion> {
        if self.size() == 0 {
            return vec![];
        }

        if self.is_contiguous() {
            return vec![MemRegion::Contiguous {
                start: self.offset,
                len: self.size(),
            }];
        }

        let mut non_singleton_dims = Vec::new();
        for (i, &dim) in self.shape.iter().enumerate() {
            if dim != 1 {
                non_singleton_dims.push((i, dim, self.stride[i]));
            }
        }

        // case of a single non-singleton dimension
        // for example, a row or column vector
        // basically a row/column vector with arbitrary n dimensions
        if non_singleton_dims.len() == 1 {
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return vec![MemRegion::Strided {
                start: self.offset,
                stride: dim_stride,
                len: dim_size,
            }];
        }

        // find a contiguous inner dimension
        if let Some(inner) = innermost_contiguous_dim(self) {
            return regions_by_inner_dim(self.clone(), inner);
        }

        vec![MemRegion::Scattered { offsets: self.iter_offsets().collect() }]
    }

    /// Returns an iterator over all offsets in the underlying buffer for this tensor/view.
    pub fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let shape = self.shape.clone();
        let stride = self.stride.clone();
        let offset = self.offset;
        TensorOffsetIterator::new(shape, stride, offset)
    }
}

struct TensorOffsetIterator {
    shape: Shape,
    stride: Stride,
    current_indices: Vec<usize>,
    done: bool,
    base_offset: usize,
}

impl TensorOffsetIterator {
    fn new(shape: Shape, stride: Stride, base_offset: usize) -> Self {
        let dims = shape.len();
        Self {
            shape,
            stride,
            current_indices: vec![0; dims],
            done: false, // Start as not done - even scalars need one iteration
            base_offset,
        }
    }
}

impl Iterator for TensorOffsetIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Special case for scalars (0 dimensions)
        if self.shape.is_empty() {
            self.done = true;
            return Some(self.base_offset);
        }

        let mut offset = self.base_offset;
        for (idx, stride) in self.current_indices.iter().zip(self.stride.iter()) {
            offset += idx * stride;
        }

        // Increment indices
        for i in (0..self.current_indices.len()).rev() {
            self.current_indices[i] += 1;
            if self.current_indices[i] < self.shape[i] {
                break;
            } else {
                self.current_indices[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(offset)
    }
}

/// Computes the standard row-major stride for a given shape.
pub fn shape_to_stride(shape: &Shape) -> Stride {
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len()).rev() {
        if i < shape.len() - 1 {
            stride[i] = stride[i + 1] * shape[i + 1];
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
            if s != expected { return false; }
            expected = expected.saturating_mul(dim);
        }
    }
    true
}

/// Read-only metadata view for tensors and views.
pub trait MetaTensorView {

    fn meta(&self) -> &MetaTensor;
    /// Borrow the shape vector.
    fn shape(&self) -> &Shape { self.meta().shape() }
    /// Borrow the stride vector.
    fn stride(&self) -> &Stride { self.meta().stride() }
    /// Starting offset (in elements) into the underlying buffer.
    fn offset(&self) -> usize { self.meta().offset() }
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

    fn mem_regions(&self) -> Vec<MemRegion> { self.meta().mem_regions() }
}

impl MetaTensorView for MetaTensor {
    fn meta(&self) -> &MetaTensor {
        self
    }
}

impl<B, T: TensorValue> MetaTensorView for TensorBase<B, T> 
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


fn innermost_contiguous_dim(meta_tensor: &MetaTensor) -> Option<usize> {
    let rank = meta_tensor.rank();
    (0..rank).rev().find(|&d| meta_tensor.stride[d] == 1)
}

/// Computes memory regions for a given inner dimension.
/// For each combination of outer dimensions, a region covering the inner
/// dimension is created, either as contiguous or strided.
/// For example, for a tensor of shape [2,3,4] and inner=1 (the '3' dimension),
/// the resulting regions will cover all 2*4=8 rows of length 3.
fn regions_by_inner_dim(meta_tensor: MetaTensor, inner: usize) -> Vec<MemRegion> {
    let rank = meta_tensor.rank();
    let mut regions = Vec::new();

    let inner_len = meta_tensor.shape[inner];
    let inner_stride = meta_tensor.stride[inner];

    // Build outer dims
    let mut outer_shape = Vec::new();
    let mut outer_strides = Vec::new();

    for d in 0..rank {
        if d == inner { continue; }
        outer_shape.push(meta_tensor.shape[d]);
        outer_strides.push(meta_tensor.stride[d]);
    }

    let mut outer_idx = vec![0; outer_shape.len()];

    if outer_idx.is_empty() {
        // This is a pure 1D tensor
        if inner_stride == 1 {
            regions.push(MemRegion::Contiguous {
                start: meta_tensor.offset,
                len: inner_len,
            });
        } else {
            regions.push(MemRegion::Strided {
                start: meta_tensor.offset,
                stride: inner_stride,
                len: inner_len,
            });
        }
        return regions;
    }

    // Iterate outer indices
    'outer_loop: loop {
        // Compute base offset for the current outer coordinates
        let mut base = meta_tensor.offset;
        let mut oi = 0;

        for d in 0..rank {
            if d == inner { continue; }
            base += outer_idx[oi] * meta_tensor.stride[d];
            oi += 1;
        }

        // Emit region
        if inner_stride == 1 {
            regions.push(MemRegion::Contiguous {
                start: base,
                len: inner_len,
            });
        } else {
            regions.push(MemRegion::Strided {
                start: base,
                stride: inner_stride,
                len: inner_len,
            });
        }

        // Increment outer indices like a multi-digit counter
        for i in (0..outer_idx.len()).rev() {
            outer_idx[i] += 1;
            if outer_idx[i] < outer_shape[i] {
                continue 'outer_loop;
            } else {
                outer_idx[i] = 0;
            }
        }

        break;
    }

    regions
}


// TODO tests for MetaTensor and MetaTensorView