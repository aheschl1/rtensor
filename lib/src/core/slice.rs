use std::ops::{Range, RangeBounds, RangeFrom, RangeTo, RangeInclusive, RangeFull};
use super::{Shape, Strides};
use super::tensor::TensorError;


#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Slice {
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub step: isize,
}

impl Slice {
    pub fn new(start: Option<usize>, end: Option<usize>, step: isize) -> Self {
        Slice { start, end, step }
    }

    /// Creates a full slice (equivalent to `..`).
    pub fn full() -> Self {
        Slice { start: None, end: None, step: 1 }
    }

    pub fn step(mut self, step: isize) -> Self {
        self.step = step;
        self
    }

    pub fn start(mut self, start: usize) -> Self {
        self.start = Some(start);
        self
    }

    pub fn end(mut self, end: usize) -> Self {
        self.end = Some(end);
        self
    }
}

impl From<Range<usize>> for Slice {
    fn from(range: Range<usize>) -> Self {
        let step = if range.start > range.end { -1 } else { 1 };
        Slice {
            start: Some(range.start),
            end: Some(range.end),
            step,
        }
    }
}

impl From<RangeInclusive<usize>> for Slice {
    fn from(range: RangeInclusive<usize>) -> Self {
        let start = *range.start();
        let end = *range.end();
        let step = if start > end { -1 } else { 1 };

        // make end exclusive
        let end = if step < 0 { if end > 0 { Some(end - 1) } else { None } } else { Some(end + 1) };
        Slice {
            start: Some(start),
            end, 
            step,
        }
    }
}

impl From<RangeFrom<usize>> for Slice {
    fn from(range: RangeFrom<usize>) -> Self {
        Slice {
            start: Some(range.start),
            end: None,
            step: 1,
        }
    }
}

impl From<RangeTo<usize>> for Slice {
    fn from(range: RangeTo<usize>) -> Self {
        Slice {
            start: None,
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Slice::full()
    }
}

impl From<usize> for Slice {
    fn from(idx: usize) -> Self {
        Slice {
            start: Some(idx),
            end: Some(idx),
            step: 1,
        }
    }
}

impl RangeBounds<usize> for Slice {
    fn start_bound(&self) -> std::ops::Bound<&usize> {
        match &self.start {
            Some(s) => std::ops::Bound::Included(s),
            None => std::ops::Bound::Unbounded,
        }
    }

    fn end_bound(&self) -> std::ops::Bound<&usize> {
        match &self.end {
            Some(e) => std::ops::Bound::Excluded(e),
            None => std::ops::Bound::Unbounded,
        }
    }
}

/// Computes the new shape, stride, and offset for a view obtained by slicing
/// along a specific dimension.
///
/// # Behavior
/// - For single-index slices (start == end), removes that dimension
/// - For range slices, keeps the dimension but adjusts its size
/// - Supports negative steps for reverse iteration
///
/// # Errors
/// - `InvalidDim` if `dim` is out of bounds
/// - `IdxOutOfBounds` if indices are out of range
/// - `InvalidShape` if step is 0
pub(crate) fn compute_sliced_parameters(
    shape: &Shape,
    stride: &Strides,
    offset: usize,
    dim: usize,
    slice: impl Into<Slice>
) -> Result<(Shape, Strides, usize), TensorError>
{
    let slice: Slice = slice.into();

    // Validate dimension for slicing over
    if dim >= shape.len() {
        return Err(TensorError::InvalidDim(format!(
            "Dimension {} out of bounds for shape with rank {}",
            dim,
            shape.len()
        )));
    }
    
    // Validate step, should be non-zero
    let step: isize = slice.step;
    if step == 0 {
        return Err(TensorError::InvalidShape(format!(
            "Slice step cannot be zero"
        )));
    }

    // (inclusive)
    let start: isize = match slice.start {
        Some(s) => s as isize,                  // as defined
        None if step > 0 => 0,                         // default to start of dimension
        None if step < 0 => (shape[dim] as isize) - 1, // default to end of dimension
        _ => unreachable!(),
    };

    // (exclusive)
    let end: isize = match slice.end {
        Some(e) => e as isize,              // as defined
        None if step > 0 => shape[dim] as isize,   // default to end of dimension 
        None if step < 0 => -1,                    // default to before start of dimension
        _ => unreachable!(),
    };

    // Range validation
    if step > 0 && (start < 0 || start >= shape[dim] as isize || end < 0 || end > shape[dim] as isize) {
        return Err(TensorError::IdxOutOfBounds(format!(
            "Slice indices out of bounds for dimension {} with size {}: start {}, end {}",
            dim,
            shape[dim],
            start,
            end
        )));
    }

    // start can be in full range, end can be no less than -1 (full), and must not exceed shape
    if step < 0 && (start < 0 || start >= shape[dim] as isize || end < -1 || end >= shape[dim] as isize) {
        return Err(TensorError::IdxOutOfBounds(format!(
            "Slice indices out of bounds for dimension {} with size {}: start {}, end {}",
            dim,
            shape[dim],
            start,
            end
        )));
    }

    // Calculate the length of the resulting slice
    let len: usize = {
        if (step > 0 && start >= end) || (step < 0 && start <= end) {
            0
        } else {
            let dist = (start - end).abs() - 1;
            (dist as usize / step.unsigned_abs()) + 1 // integer division rounding up
        }
    };

    if len == 0 {
        // Empty slice - collapse dimension
        let mut new_shape = shape.clone();
        let mut new_stride = stride.clone();
        new_shape.remove(dim);
        new_stride.remove(dim);

        let new_offset = offset + (start * stride[dim]).max(0) as usize;
        return Ok((new_shape, new_stride, new_offset));
    }

    let new_offset = offset + (start * stride[dim]).max(0) as usize;

    let mut new_shape = shape.clone();
    let mut new_stride = stride.clone();

    new_shape[dim] = len;
    // For negative steps, stride becomes negative (reverse iteration)
    new_stride[dim] = stride[dim] * step;

    Ok((new_shape, new_stride, new_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_from_range() {
        let slice: Slice = (2..5).into();
        assert_eq!(slice.start, Some(2));
        assert_eq!(slice.end, Some(5));
        assert_eq!(slice.step, 1);
    }

    #[test]
    fn test_slice_from_usize() {
        let slice: Slice = 3.into();
        assert_eq!(slice.start, Some(3));
        assert_eq!(slice.end, Some(3));
        assert_eq!(slice.step, 1);
    }

    #[test]
    fn test_slice_builder() {
        let slice = Slice::full().step(-2).start(10).end(2);
        assert_eq!(slice.start, Some(10));
        assert_eq!(slice.end, Some(2));
        assert_eq!(slice.step, -2);
    }

    #[test]
    fn test_slice_range_bounds() {
        let slice = Slice::new(Some(2), Some(8), 1);
        assert_eq!(slice.start_bound(), std::ops::Bound::Included(&2));
        assert_eq!(slice.end_bound(), std::ops::Bound::Excluded(&8));
    }
}
