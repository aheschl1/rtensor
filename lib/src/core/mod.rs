
pub mod primitives;
pub mod meta;
pub mod tensor;
pub mod idx;
pub mod value;
pub mod slice;

pub use meta::{Dim, Shape, Stride, MetaTensor, MetaTensorView, shape_to_stride};
pub use primitives::{Tensor, TensorView, CpuTensorView, TensorViewMut};
pub use slice::Slice;


#[cfg(test)]
mod tests {
    use crate::{backend::Backend, coord, core::{idx::Idx, tensor::{AsTensor, AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape, Slice, Stride, Tensor}};

    fn make_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Tensor<T> {
        Tensor::from_buf(buf, shape.into()).unwrap()
    }

    #[test]
    fn test_slice_matrix() {
        let buf = vec![
            1, 2, 3, 
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*slice.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 3);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap(); // slice along columns, should give a view of shape [2]
        assert_eq!(*slice2.shape(), vec![2]);
        assert_eq!(*slice2.stride(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(coord![1], &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
    }

    #[test]
    fn test_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(*slice.stride(), vec![2, 1]);
        assert_eq!(index_tensor(coord![0, 0], &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 5);

        // second depth
        let view = tensor.view();
        let slice_second_depth = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice_second_depth.shape(), vec![2, 2]);
        assert_eq!(*slice_second_depth.stride(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice_second_depth).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice_second_depth).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice_second_depth).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice_second_depth).unwrap(), 9);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap(); // slice along row, should give a view of shape [2, 2]
        assert_eq!(*slice2.shape(), vec![2, 2]);
        assert_eq!(*slice2.stride(), vec![4, 1]);
        assert_eq!(index_tensor(coord![0, 0], &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice2).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice2).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice2).unwrap(), 7);

        // column slice
        let view = tensor.view();
        let slice3 = view.slice(2, 0..0).unwrap(); // slice along column
        assert_eq!(*slice3.shape(), vec![2, 2]);
        assert_eq!(*slice3.stride(), vec![4, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice3).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice3).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice3).unwrap(), 8);
    }

    #[test]
    fn test_slice_of_slice() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);

        let slice_of_slice = slice.slice(0, 2..2).unwrap(); // slice along columns, should give a view of shape []
        assert_eq!(*slice_of_slice.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Coord(vec![]), &slice_of_slice).unwrap(), 6);
    }

    #[test]
    fn test_range_slice_matrix_rows() {
        // Test slicing a range of rows (keeps dimension)
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![4, 3]; // 4 rows, 3 columns
        let tensor = make_tensor(buf, shape);
        
        // Slice rows 1..3 (middle 2 rows)
        let view = tensor.view();
        let slice = view.slice(0, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![2, 3]); // 2 rows, 3 columns
        assert_eq!(*slice.stride(), vec![3, 1]);
        
        // Should contain rows 1 and 2 (4,5,6 and 7,8,9)
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &slice).unwrap(), 9);
    }

    #[test]
    fn test_range_slice_matrix_columns() {
        // Test slicing a range of columns (keeps dimension)
        let buf = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ];
        let shape = vec![3, 4]; // 3 rows, 4 columns
        let tensor = make_tensor(buf, shape);
        
        // Slice columns 1..3 (middle 2 columns)
        let view = tensor.view();
        let slice = view.slice(1, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![3, 2]); // 3 rows, 2 columns
        assert_eq!(*slice.stride(), vec![4, 1]);
        
        // Should contain columns 1 and 2 from each row
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &slice).unwrap(), 11);
    }

    #[test]
    fn test_range_slice_cube_depth() {
        // Test slicing a range along depth dimension
        let buf = vec![
            // Depth 0
            1, 2,
            3, 4,
            // Depth 1
            5, 6,
            7, 8,
            // Depth 2
            9, 10,
            11, 12,
            // Depth 3
            13, 14,
            15, 16
        ];
        let shape = vec![4, 2, 2]; // 4 depth, 2 rows, 2 columns
        let tensor = make_tensor(buf, shape);
        
        // Slice depth 1..3 (middle 2 slices)
        let view = tensor.view();
        let slice = view.slice(0, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![2, 2, 2]); // 2 depth, 2 rows, 2 columns
        assert_eq!(*slice.stride(), vec![4, 2, 1]);
        
        // Should contain depth slices 1 and 2
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &slice).unwrap(), 9);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 1]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 0]), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &slice).unwrap(), 12);
    }

    #[test]
    fn test_range_slice_first_n_elements() {
        // Test slicing from start: 0..n
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..5).unwrap(); // First 5 elements
        assert_eq!(*slice.shape(), vec![5]);
        assert_eq!(*slice.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(4), &slice).unwrap(), 5);
    }

    #[test]
    fn test_range_slice_last_n_elements() {
        // Test slicing to end: n..size
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 5..8).unwrap(); // Last 3 elements
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*slice.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 8);
    }

    #[test]
    fn test_range_slice_of_range_slice() {
        // Test chaining range slices
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // First slice: get elements 2..8
        let view = tensor.view();
        let slice1 = view.slice(0, 2..8).unwrap();
        assert_eq!(*slice1.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &slice1).unwrap(), 3);
        assert_eq!(index_tensor(Idx::At(5), &slice1).unwrap(), 8);
        
        // Second slice: get elements 1..4 from the previous slice
        let slice2 = slice1.slice(0, 1..4).unwrap();
        assert_eq!(*slice2.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 6);
    }

    #[test]
    fn test_range_slice_mut() {
        // Test mutable range slicing
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![4, 2];
        let mut tensor = make_tensor(buf, shape);
        
        let mut slice = tensor.slice_mut(0, 1..3).unwrap(); // Rows 1 and 2
        assert_eq!(*slice.shape(), vec![2, 2]);
        
        // Modify values in the slice
        slice.set(&Idx::Coord(vec![0, 0]), 30).unwrap();
        slice.set(&Idx::Coord(vec![1, 1]), 80).unwrap();
        
        // Check original tensor was modified
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &tensor.view()).unwrap(), 30);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &tensor.view()).unwrap(), 80);
    }

    #[test]
    fn test_range_slice_full_dimension() {
        // Test slicing with full range (should be same as original)
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..2).unwrap(); // All rows
        assert_eq!(*slice.shape(), vec![2, 3]);
        assert_eq!(*slice.stride(), vec![3, 1]);
        
        // Should contain all original data
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &slice).unwrap(), 6);
    }

    #[test]
    fn test_range_slice_single_element() {
        // Test range with single element (n..n+1 is same as empty range n..n after indexing)
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 2..3).unwrap(); // Single element at index 2
        assert_eq!(*slice.shape(), vec![1]); // Dimension remains, but size is 1
        assert_eq!(*slice.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 3);
    }

    #[test]
    fn test_range_slice_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        
        // Range end beyond dimension size
        assert!(matches!(
            tensor.view().slice(0, 0..3),
            Err(TensorError::IdxOutOfBounds)
        ));
        
        // Range start beyond dimension size
        assert!(matches!(
            tensor.view().slice(0, 3..4),
            Err(TensorError::IdxOutOfBounds)
        ));
        
        // Range with start > end
        assert!(matches!(
            tensor.view().slice(0, 2..1),
            Err(TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_negative_step_basic() {
        // Test basic negative step slicing on a 1D tensor
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // Slice with step -1 (reverse)
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 1);
    }

    #[test]
    fn test_negative_step_with_range() {
        // Test negative step with explicit start and end
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // Slice from index 6 down to index 2 with step -1
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(6), Some(2), -1)).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 7); // index 6
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 6); // index 5
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 5); // index 4
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 4); // index 3
    }

    #[test]
    fn test_negative_step_skip_elements() {
        // Test negative step -2 (every other element backwards)
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 8); // index 7
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 6); // index 5
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 4); // index 3
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 2); // index 1
    }

    #[test]
    fn test_negative_step_matrix_rows() {
        // Test negative step on matrix rows
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![4, 3];
        let tensor = make_tensor(buf, shape);
        
        // Reverse rows with step -1
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![4, 3]);
        
        // First row should be the last row of original
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &slice).unwrap(), 12);
        
        // Last row should be the first row of original
        assert_eq!(index_tensor(Idx::Coord(vec![3, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 2]), &slice).unwrap(), 3);
    }

    #[test]
    fn test_negative_step_with_start_less_than_end() {
        // When step is negative and start < end, result should be empty
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(2), Some(5), -1)).unwrap();
        assert_eq!(*slice.shape(), vec![]); // Empty slice
    }

    #[test]
    fn test_positive_step_with_start_greater_than_end() {
        // When step is positive and start > end, result should be empty
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(5), Some(2), 1)).unwrap();
        assert_eq!(*slice.shape(), vec![]); // Empty slice
    }

    #[test]
    fn test_negative_step_partial_range() {
        // Test negative step with only start specified
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // From index 5 going backwards to the beginning
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(5), None, -1)).unwrap();
        assert_eq!(*slice.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 6); // index 5
        assert_eq!(index_tensor(Idx::At(5), &slice).unwrap(), 1); // index 0
    }

    #[test]
    fn test_negative_step_partial_end() {
        // Test negative step with only end specified
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // From the end going backwards to index 3
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(None, Some(3), -1)).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 8); // index 7
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 5); // index 4
    }

    #[test]
    fn test_negative_step_3d_tensor() {
        // Test negative step on 3D tensor
        let buf = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ];
        let shape = vec![4, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        // Reverse along the depth dimension
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![4, 2, 2]);
        
        // First depth slice should be the last one of original
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &slice).unwrap(), 13);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &slice).unwrap(), 16);
        
        // Last depth slice should be the first one of original
        assert_eq!(index_tensor(Idx::Coord(vec![3, 0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 1, 1]), &slice).unwrap(), 4);
    }

    #[test]
    fn test_negative_step_chained() {
        // Test chaining slices with negative steps
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // First reverse the whole array
        let view = tensor.view();
        let slice1 = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(index_tensor(Idx::At(0), &slice1).unwrap(), 10);
        
        // Then take every other element
        let slice2 = slice1.slice(0, Slice::full().step(2)).unwrap();
        assert_eq!(*slice2.shape(), vec![5]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(4), &slice2).unwrap(), 2);
    }

    #[test]
    fn test_negative_step_funtax() {
        // Test chaining slices with negative steps
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // Then take every other element
        let view = tensor.view();
        let slice2 = view.slice(0, 9..=0).unwrap();
        assert_eq!(*slice2.shape(), vec![10]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 9);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 8);
    }

    #[test]
    fn test_negative_step_mut() {
        // Test mutable slicing with negative step
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let mut tensor = make_tensor(buf, shape);
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::full().step(-1)).unwrap();
        
        // Modify reversed view
        slice.set(&Idx::At(0), 60).unwrap(); // Should modify index 5
        slice.set(&Idx::At(5), 10).unwrap(); // Should modify index 0
        
        // Check original tensor
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(5), &tensor.view()).unwrap(), 60);
    }

    #[test]
    fn test_negative_step_mut_funtax() {
        // Test mutable slicing with negative step
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let mut tensor = make_tensor(buf, shape);
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(-1)).unwrap();
        
        // Modify reversed view
        slice.set(&Idx::At(0), 60).unwrap(); // Should modify index 5
        slice.set(&Idx::At(5), 10).unwrap(); // Should modify index 0
        
        // Check original tensor
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(5), &tensor.view()).unwrap(), 60);
    }

    #[test]
    fn test_negative_step_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![6]);
        
        // Invalid start index with negative step
        assert!(matches!(
            tensor.view().slice(0, Slice::new(Some(10), Some(2), -1)),
            Err(TensorError::IdxOutOfBounds)
        ));
        
        // Invalid end index with negative step
        assert!(matches!(
            tensor.view().slice(0, Slice::new(Some(5), Some(10), -1)),
            Err(TensorError::IdxOutOfBounds)
        ));
        
        // Step of 0 should error
        assert!(matches!(
            tensor.view().slice(0, Slice::new(Some(2), Some(5), 0)),
            Err(TensorError::InvalidShape)
        ));
    }

    #[test]
    fn test_reverse_range_auto_negative_step() {
        // Test that reversed ranges automatically get negative step
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // Exclusive range with start > end should auto-reverse
        let view = tensor.view();
        let slice = view.slice(0, 9..0).unwrap();
        assert_eq!(*slice.shape(), vec![9]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10); // index 9
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 9);  // index 8
        assert_eq!(index_tensor(Idx::At(8), &slice).unwrap(), 2);  // index 1
        
        // Inclusive range with start > end should auto-reverse
        let view = tensor.view();
        let slice = view.slice(0, 9..=0).unwrap();
        assert_eq!(*slice.shape(), vec![10]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10); // index 9
        assert_eq!(index_tensor(Idx::At(9), &slice).unwrap(), 1);  // index 0
    }

    #[test]
    fn test_custom_positive_step() {
        // Test slicing with custom positive step values (step > 1)
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_tensor(buf, shape);
        
        // Step by 2: take every other element from full range
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 14);
        
        // Step by 3: from index 1 to 8
        let slice2 = view.slice(0, Slice::from(1..8).step(3)).unwrap();
        assert_eq!(*slice2.shape(), vec![3]); // Indices 1, 4, 7
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 7);
        
        // Step by 4: from start to 12
        let slice3 = view.slice(0, Slice::from(..12).step(4)).unwrap();
        assert_eq!(*slice3.shape(), vec![3]); // Indices 0, 4, 8
        assert_eq!(index_tensor(Idx::At(0), &slice3).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice3).unwrap(), 8);
        
        // Step by 5: from 3 to end
        let slice4 = view.slice(0, Slice::from(3..).step(5)).unwrap();
        assert_eq!(*slice4.shape(), vec![3]); // Indices 3, 8, 13
        assert_eq!(index_tensor(Idx::At(0), &slice4).unwrap(), 3);
        assert_eq!(index_tensor(Idx::At(1), &slice4).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(2), &slice4).unwrap(), 13);
        
        // Large step: step by 10
        let slice5 = view.slice(0, Slice::from(2..).step(10)).unwrap();
        assert_eq!(*slice5.shape(), vec![2]); // Indices 2, 12
        assert_eq!(index_tensor(Idx::At(0), &slice5).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &slice5).unwrap(), 12);
    }

    #[test]
    fn test_custom_negative_step() {
        // Test slicing with custom negative step values (step < -1)
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_tensor(buf, shape);
        let view = tensor.view();
        
        // Step by -2: every other element, reversed
        let slice = view.slice(0, Slice::from(..).step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 15); // Start from end
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 13);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 1);
        
        // Step by -3: from index 10 to 1
        let slice2 = view.slice(0, Slice::from(10..1).step(-3)).unwrap();
        assert_eq!(*slice2.shape(), vec![3]); // Indices 10, 7, 4
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 4);
        
        // Step by -4: from end to index 2
        let slice3 = view.slice(0, Slice::from(..2).step(-4)).unwrap();
        assert_eq!(*slice3.shape(), vec![4]); // Indices 15, 11, 7, 3
        assert_eq!(index_tensor(Idx::At(0), &slice3).unwrap(), 15);
        assert_eq!(index_tensor(Idx::At(1), &slice3).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(2), &slice3).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(3), &slice3).unwrap(), 3);
        
        // Step by -5: from index 14 to start
        let slice4 = view.slice(0, Slice::from(14..).step(-5)).unwrap();
        assert_eq!(*slice4.shape(), vec![3]); // Indices 14, 9, 4
        assert_eq!(index_tensor(Idx::At(0), &slice4).unwrap(), 14);
        assert_eq!(index_tensor(Idx::At(1), &slice4).unwrap(), 9);
        assert_eq!(index_tensor(Idx::At(2), &slice4).unwrap(), 4);
    }

    #[test]
    fn test_custom_positive_step_mut() {
        // Test mutable slicing with custom positive step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
        // Step by 2: modify every other element
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        
        // Set values at step positions
        slice.set(&Idx::At(0), 100).unwrap(); // index 0
        slice.set(&Idx::At(1), 102).unwrap(); // index 2
        slice.set(&Idx::At(7), 114).unwrap(); // index 14
        
        // Verify changes in original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 100);
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 102);
        assert_eq!(index_tensor(Idx::At(3), &view).unwrap(), 3); // Unchanged
        assert_eq!(index_tensor(Idx::At(14), &view).unwrap(), 114);
        assert_eq!(index_tensor(Idx::At(15), &view).unwrap(), 15); // Unchanged
    }

    #[test]
    fn test_custom_positive_step_mut_with_range() {
        // Test mutable slicing with step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
        // Step by 3: from index 1 to 10
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(1..10).step(3)).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 1, 4, 7
        
        slice.set(&Idx::At(0), 101).unwrap(); // index 1
        slice.set(&Idx::At(1), 104).unwrap(); // index 4
        slice.set(&Idx::At(2), 107).unwrap(); // index 7
        
        // Verify
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 0);  // Unchanged
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 101);
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 104);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 107);
        assert_eq!(index_tensor(Idx::At(8), &view).unwrap(), 8);  // Unchanged
    }

    #[test]
    fn test_custom_negative_step_mut() {
        // Test mutable slicing with custom negative step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
        // Step by -2: every other element, reversed
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        
        slice.set(&Idx::At(0), 115).unwrap(); // index 15
        slice.set(&Idx::At(1), 113).unwrap(); // index 13
        slice.set(&Idx::At(7), 101).unwrap(); // index 1
        
        // Verify changes
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 101);
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 2);   // Unchanged
        assert_eq!(index_tensor(Idx::At(13), &view).unwrap(), 113);
        assert_eq!(index_tensor(Idx::At(14), &view).unwrap(), 14); // Unchanged
        assert_eq!(index_tensor(Idx::At(15), &view).unwrap(), 115);
    }

    #[test]
    fn test_custom_negative_step_mut_with_range() {
        // Test mutable slicing with negative step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
        // Step by -3: from index 12 to 3
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(12..3).step(-3)).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 12, 9, 6
        
        slice.set(&Idx::At(0), 212).unwrap(); // index 12
        slice.set(&Idx::At(1), 209).unwrap(); // index 9
        slice.set(&Idx::At(2), 206).unwrap(); // index 6
        
        // Verify
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 206);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 7);   // Unchanged
        assert_eq!(index_tensor(Idx::At(9), &view).unwrap(), 209);
        assert_eq!(index_tensor(Idx::At(10), &view).unwrap(), 10); // Unchanged
        assert_eq!(index_tensor(Idx::At(12), &view).unwrap(), 212);
    }

    #[test]
    fn slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);

        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 9);

        let slice_of_slice = slice.slice(1, 0..0).unwrap(); // slice along row, should give a view of shape [2]
        assert_eq!(*slice_of_slice.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &slice_of_slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(1), &slice_of_slice).unwrap(), 8);

        // slice of slice of slice
        let slice_of_slice_of_slice = slice_of_slice.slice(0, 1..1).unwrap(); // slice along column, should give a view of shape []
        assert_eq!(*slice_of_slice_of_slice.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Item, &slice_of_slice_of_slice).unwrap(), 8);
    }

    #[test]
    fn test_mut_slices() {
        // mut slice from owned tensor
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];

        let mut tensor = make_tensor(buf, shape);
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // slice along rows, should give a view of shape [3]
        
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);
        slice.set(&Idx::At(1), 50).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 50);
    }

    #[test]
    fn test_column() {
    let tensor = Tensor::column(vec![1, 2, 3]);
    assert_eq!(*tensor.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_row() {
    let tensor = Tensor::row(vec![1, 2, 3]);
    assert_eq!(*tensor.shape(), vec![1, 3]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 3);

        assert_eq!(tensor.view().get(&[0, 1]).unwrap(), 2);
    }

    #[test]
    fn test_scalar() {
        let buf = vec![42];
        let shape = vec![];
        let tensor = make_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::Item, &tensor.view()).unwrap(), 42);
        assert!(tensor.is_scalar());
        assert_eq!(Tensor::scalar(42), tensor);
    }

    #[test]
    fn test_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);

        tensor.view_mut().set(&Idx::At(1), 1).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 1);
    }

    #[test]
    fn test_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 6);

        tensor.view_mut().set(&Idx::Coord(vec![1, 2]), 100).unwrap();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 100);
    }

    #[test]
    fn test_cube() {
        //
        let buf = vec![
            1, 2,
            4, 5,

            6, 7,
            8, 9
        ];
        let shape = vec![2, 2, 2];
        let mut tensor = make_tensor(buf, shape);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor.view()).unwrap(), 1); // depth, row, column
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor.view()).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 1]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 0]), &tensor.view()).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &tensor.view()).unwrap(), 9);

        // modify
        tensor.view_mut().set(&Idx::Coord(vec![1, 0, 0]), 67).unwrap();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor.view()).unwrap(), 67);
    }

    // #[test]
    // fn test_view_as_owned_success() {
    //     let buf = vec![1, 2, 3, 4, 5, 6];
    //     let shape = vec![2, 3];
    //     let tensor = make_tensor(buf, shape);
    //     let reshaped = tensor.view().view_as(vec![3, 2]).unwrap();
    //     assert_eq!(*reshaped.shape(), vec![3, 2]);
    //     assert_eq!(*reshaped.stride(), vec![2, 1]);
    //     // Row-major sequence preserved
    //     assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &reshaped).unwrap(), 1);
    //     assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &reshaped).unwrap(), 2);
    //     assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &reshaped).unwrap(), 3);
    //     assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &reshaped).unwrap(), 4);
    //     assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &reshaped).unwrap(), 5);
    //     assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &reshaped).unwrap(), 6);
    // }

    // #[test]
    // fn test_view_as_owned_error() {
    //     let buf = vec![1, 2, 3, 4, 5, 6];
    //     let shape = vec![2, 3];
    //     let tensor = make_tensor(buf, shape);
    //     assert!(matches!(tensor.view().view_as(vec![4, 2]), Err(TensorError::InvalidShape)));
    // }

    // #[test]
    // fn test_view_as_slice_success() {
    //     let buf = vec![
    //         1, 2, 3, 
    //         4, 5, 6
    //     ];
    //     let shape = vec![2, 3];
    //     let tensor = make_tensor(buf, shape);

    //     let view = tensor.view();
    //     let slice = view.slice(0, 1..1).unwrap(); // shape [3]
    //     assert_eq!(*slice.shape(), vec![3]);
    //     let reshaped = slice.view_as(vec![1, 3]).unwrap();
    //     assert_eq!(*reshaped.shape(), vec![1, 3]);
    //     assert_eq!(*reshaped.stride(), vec![3, 1]);
    //     // Values should correspond to original slice elements 4,5,6
    //     assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &reshaped).unwrap(), 4);
    //     assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &reshaped).unwrap(), 5);
    //     assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &reshaped).unwrap(), 6);
    // }

    // #[test]
    // fn test_view_as_mut_view_modify() {
    //     let buf = vec![1, 2, 3, 4];
    //     let shape = vec![2, 2];
    //     let mut tensor = make_tensor(buf, shape);
    //     let mut view_mut = tensor.view_mut(); // shape [2,2]
    //     // Modify before reshaping to avoid borrow conflicts
    //     view_mut.set(&Idx::Coord(vec![1, 0]), 40).unwrap(); // coordinate [1,0] maps to linear index 2
    //     let reshaped = view_mut.view_as(vec![4]).unwrap(); // reshape to flat vector
    //     assert_eq!(*reshaped.shape(), vec![4]);
    //     assert_eq!(*reshaped.stride(), vec![1]);
    //     // Check reshaped view sees update at linear index 2
    //     assert_eq!(index_tensor(Idx::At(2), &reshaped).unwrap(), 40);
    // }

    // #[test]
    // fn test_view_as_scalar() {
    //     let tensor = CpuTensor::scalar(99); // shape []
    //     let view1 = tensor.view();
    //     assert_eq!(*view1.shape(), vec![]);
    //     let reshaped = view1.view_as(vec![1]).unwrap();
    //     assert_eq!(*reshaped.shape(), vec![1]);
    //     assert_eq!(*reshaped.stride(), vec![1]);
    //     assert_eq!(index_tensor(Idx::At(0), &reshaped).unwrap(), 99);

    //     // view as [1, 1, 1]

    //     let r2 = reshaped.view_as(vec![1, 1, 1]).unwrap();
    //     assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &r2).unwrap(), 99);

    // }

    fn index_tensor<'a, T: TensorValue + PartialEq + std::fmt::Debug, B: Backend<T>>(index: impl Into<Idx>, tensor: &'a impl TensorAccess<T, B>) -> Result<T, TensorError> {
        let index = index.into();
        let r: Result<T, TensorError> = tensor.get(&index);
        let a = match r.as_ref() {
            Ok(v) => Ok(*v),
            Err(e) => return Err(e.clone()),
        };
        let b = match &index {
            Idx::At(i) => tensor.get(Idx::Coord(vec![*i])),
            Idx::Coord(idx) => tensor.get(Idx::Coord(idx.clone())),
            Idx::Item => tensor.item(),
        };
        assert_eq!(a, b);
        r
    }

    #[test]
    fn test_shape_to_stride() {
        let shape = vec![2, 2, 3];
        let stride: Stride = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![6, 3, 1]);
    }

    #[test]
    fn test_shape_to_stride_single_dim() {
        let shape = vec![4];
        let stride: Stride = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_shape_to_stride_empty() {
        let shape: Shape = Shape::empty();
        let stride: Stride = super::shape_to_stride(&shape);

        assert!(stride.is_empty());
    }

    #[test]
    fn test_shape_to_stride_ones() {
        let shape = vec![1, 1, 1];
        let stride: Stride = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![1, 1, 1]);
    }

    #[test]
    fn test_shape_to_stride_mixed() {
        let shape = vec![5, 1, 2];
        let stride: Stride = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![2, 2, 1]);
    }

    #[test]
    fn test_shape_to_stride_larger() {
        let shape = vec![3, 4, 5];
        let stride: Stride = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![20, 5, 1]);
    }

    #[test]
    fn test_from_buf_error() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            Tensor::from_buf(buf, shape),
            Err(TensorError::InvalidShape)
        ));
    }

    #[test] 
    fn test_get_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor.view()),
            Err(TensorError::WrongDims)
        ));
        assert!(matches!(
            index_tensor(Idx::Coord(vec![2, 0]), &tensor.view()),
            Err(TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_slice_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            tensor.view().slice(2, 0..0),
            Err(TensorError::InvalidDim)
        ));
        assert!(matches!(
            tensor.view().slice(0, 2..2),
            Err(TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_index_and_index_mut() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

        // Test Index on TensorOwned
        assert_eq!(tensor.view().get(&Idx::Coord(vec![0, 1])).unwrap(), 2);
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 6);

        // Test IndexMut on TensorOwned
        tensor.view_mut().set(vec![1, 1], 55).unwrap();
        assert_eq!(tensor.view().get(&Idx::Coord(vec![1, 1])).unwrap(), 55);
        assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 55);

        // Test on a slice (TensorView)
        let view = tensor.view();
        let view = view.slice(0, 1..1).unwrap(); // second row
        assert_eq!(view.get(vec![0]).unwrap(), 4);
        assert_eq!(view.get(vec![1]).unwrap(), 55);
        assert_eq!(view.get(vec![2]).unwrap(), 6);

        // Test on a mutable slice (TensorViewMut)
        let mut mut_view = tensor.view_mut();
        let mut mut_slice = mut_view.slice_mut(0, 0..0).unwrap(); // first row
        mut_slice.set(vec![2], 33).unwrap();
        assert_eq!(mut_slice.get(&Idx::Coord(vec![2])).unwrap(), 33);
        assert_eq!(mut_slice.get(vec![2]).unwrap(), 33);

        // Verify original tensor was changed
        assert_eq!(tensor.view().get(vec![0, 2]).unwrap(), 33);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_panic() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_wrong_dims_panic() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![0, 0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_mut_out_of_bounds_panic() {
        let mut tensor = make_tensor(vec![1, 2, 3], vec![3]);
        tensor.view_mut().set(vec![3], 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_mut_wrong_dims_panic() {
        let mut tensor = make_tensor(vec![1, 2, 3], vec![3]);
        tensor.view_mut().set(vec![0, 0], 4).unwrap();
    }

    // --- Additional coverage tests ---
    #[test]
    fn test_set_method() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tensor.view_mut().set(&Idx::Coord(vec![1, 2]), 99), Ok(())));
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 99);
    }

    #[test]
    fn test_is_row_and_is_column() {
        let row = Tensor::row(vec![1, 2, 3]);
        let col = Tensor::column(vec![1, 2, 3]);
        let scalar = Tensor::scalar(10);
        assert!(row.is_row());
        assert!(!row.is_column());
        assert!(!col.is_row());
        assert!(col.is_column());
        assert!(scalar.is_scalar());
        assert!(!scalar.is_row());
        assert!(!scalar.is_column());
    }

    #[test]
    fn test_slice_scalar_error() {
        let scalar = Tensor::scalar(5);
        assert!(matches!(scalar.view().slice(0, 0..0), Err(TensorError::InvalidDim)));
    }

    // #[test]
    // fn test_view_as_slice_error() {
    //     let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

    //     let view = tensor.view();
    //     let slice = view.slice(0, 0..0).unwrap(); // shape [3]
    //     assert!(matches!(slice.view_as(vec![2, 2]), Err(TensorError::InvalidShape)));
    // }

    // #[test]
    // fn test_view_mut_as_error() {
    //     let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    //     let view_mut = tensor.view_mut();
    //     assert!(matches!(view_mut.view_as(vec![3, 2]), Err(TensorError::InvalidShape)));
    // }

    #[test]
    fn test_item_wrong_dims_error() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tensor.view().get(&Idx::Item), Err(TensorError::WrongDims)));
    }

    #[test]
    fn test_from_buf_empty_shape_error() {
        assert!(matches!(Tensor::from_buf(Vec::<i32>::new(), vec![]), Err(TensorError::InvalidShape)));
    }

    // #[test]
    // fn test_modify_after_reshape_reflects() {
    //     let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    //     {
    //         let view_mut = tensor.view_mut();
    //         let mut reshaped = view_mut.view_as(vec![4]).unwrap();
    //         reshaped.set(Idx::At(3), 40).unwrap(); // modify last element
    //         assert_eq!(reshaped.get(&Idx::At(3)).unwrap(), 40);
    //     }
    //     assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 40);
    // }

    #[test]
    fn test_slice_single_dim_to_scalar() {
        let tensor = make_tensor(vec![42], vec![1]);

        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.shape(), vec![]);
        assert_eq!(slice.get(&Idx::Item).unwrap(), 42);
        assert_eq!(slice.get(&Idx::Coord(vec![])).unwrap(), 42);
    }

    #[test]
    fn test_cube_slice_mut_chain() {
        let mut tensor = make_tensor(vec![
            1, 2,
            4, 5,
            6, 7,
            8, 9
        ], vec![2, 2, 2]);

        let mut view = tensor.view_mut();
        let mut depth_view = view.slice_mut(0, 1..1).unwrap(); // shape [2,2]
        assert_eq!(*depth_view.shape(), vec![2, 2]);
        let mut row_view = depth_view.slice_mut(0, 1..1).unwrap(); // shape [2]
        assert_eq!(*row_view.shape(), vec![2]);
        row_view.set(Idx::At(0), 800).unwrap(); // maps to original [1,1,0]
        assert_eq!(tensor.view().get(vec![1, 1, 0]).unwrap(), 800);
        assert_eq!(tensor.view().get(vec![1, 1, 1]).unwrap(), 9);
    }

    // --- Additional metadata coverage tests ---
    #[test]
    fn test_meta_dims_and_dim() {
        let scalar = Tensor::scalar(10);
        assert_eq!(scalar.dims(), 0);

        let vec = make_tensor(vec![1, 2, 3], vec![3]);
        assert_eq!(vec.dims(), 1);
        assert_eq!(vec.dim(0), 3);

        let mat = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert_eq!(mat.dims(), 2);
        assert_eq!(mat.dim(0), 2);
        assert_eq!(mat.dim(1), 2);

        let cube = make_tensor(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        assert_eq!(cube.dims(), 3);
        assert_eq!(cube.dim(0), 2);
        assert_eq!(cube.dim(1), 2);
        assert_eq!(cube.dim(2), 2);
    }

    #[test]
    fn test_meta_size() {
        let scalar = Tensor::scalar(42);
        assert_eq!(scalar.size(), 1);

        let vec = make_tensor(vec![1, 2, 3, 4], vec![4]);
        assert_eq!(vec.size(), 4);

        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        assert_eq!(mat.size(), 6);
    }

    #[test]
    fn test_meta_offsets_on_slices() {
        // 2x2 -> stride [2,1]
        let m = make_tensor(vec![1,2,3,4], vec![2,2]);
        let v_m = m.view();
        let s = v_m.slice(0, 1..1).unwrap(); // take second row

        assert_eq!(s.offset(), 2);

        // 2x2x2 -> stride [4,2,1]
        let c = make_tensor(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        let v_c = c.view();
        let s1 = v_c.slice(0, 1..1).unwrap(); // offset +4

        assert_eq!(s1.offset(), 4);
        let s2 = s1.slice(1, 1..1).unwrap(); // now stride [2,1], add +1
        assert_eq!(s2.offset(), 5);
    }

    #[test]
    fn test_is_contiguous_owned_and_slices() {
        // Owned are contiguous
        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        assert!(mat.is_contiguous());

        let v1 = mat.view();
        let row = v1.slice(0, 1..1).unwrap();

    // Row slice -> contiguous (stride [1])
        assert!(row.is_contiguous());

        // Column-like slice -> non-contiguous (stride [3])
        let v2 = mat.view();
        let col_like = v2.slice(1, 0..0).unwrap();
        assert!(!col_like.is_contiguous());
    }

    // #[test]
    // fn test_reshape_noncontiguous_slice_contiguous() {
    //     // Start non-contiguous 1D view (stride [3])
    //     let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
    //     let v3 = mat.view();
    //     let col_like = v3.slice(1, 0..0).unwrap(); // shape [2], stride [3]
    //     assert!(!col_like.is_contiguous());

    //     // Reshape to [1,2] -> becomes contiguous
    //     let reshaped = col_like.view_as(vec![1, 2]).unwrap();
    //     assert!(reshaped.is_contiguous());
    //     assert_eq!(*reshaped.shape(), vec![1,2]);
    //     // Note: reshaping a non-contiguous slice uses underlying memory order
    //     // starting at the slice's offset; here it's [1, 2]
    //     assert_eq!(reshaped.get(vec![0,0]).unwrap(), 1);
    //     assert_eq!(reshaped.get(vec![0,1]).unwrap(), 2);
    // }

    #[test]
    fn test_tensor_meta_direct_impl() {
        // contiguous meta
        let shape = vec![2, 3];
        let stride = super::shape_to_stride(&shape.clone().into());
        let meta = MetaTensor::new(shape.clone(), stride.clone(), 0);
        assert_eq!(meta.shape(), &shape);
        assert_eq!(meta.strides(), &stride);
        assert_eq!(meta.offset(), 0);
        assert_eq!(meta.dims(), 2);
        assert_eq!(meta.dim(0), 2);
        assert_eq!(meta.dim(1), 3);
        assert_eq!(meta.size(), 6);
        assert!(!meta.is_scalar());
        assert!(!meta.is_row());
        assert!(!meta.is_column());
        assert!(meta.is_contiguous());

        // non-contiguous meta (e.g., a column-like view of a [2,3] matrix)
        let nc_meta = MetaTensor::new(vec![2], vec![3], 0);
        assert_eq!(nc_meta.dims(), 1);
        assert_eq!(nc_meta.size(), 2);
        assert!(!nc_meta.is_contiguous());
    }

    #[test]
    fn test_idx_at_vs_coord_on_sliced_view() {
        // Create a non-contiguous 1D view and ensure At == Coord mapping
        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        let v4 = mat.view();
        let col_like = v4.slice(1, 0..0).unwrap(); // [2], stride [3]
        assert_eq!(col_like.get(vec![0]).unwrap(), col_like.get(&Idx::Coord(vec![0])).unwrap());
        assert_eq!(col_like.get(vec![1]).unwrap(), col_like.get(&Idx::Coord(vec![1])).unwrap());
    }

    #[test]
    fn test_dims_equals_stride_len() {
        let owned = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        assert_eq!(owned.dims(), owned.stride().len());

        let v5 = owned.view();
        let row = v5.slice(0, 1..1).unwrap();
        assert_eq!(row.dims(), row.stride().len());

        let v6 = owned.view();
        let col_like = v6.slice(1, 0..0).unwrap();
        assert_eq!(col_like.dims(), col_like.stride().len());
    }

    #[test]
    fn test_view_to_owned_contiguous() {
        // Test converting a contiguous view to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let owned = view.owned();
        
        // Should have same shape and values
        assert_eq!(owned.shape(), tensor.shape());
        assert_eq!(owned.raw, tensor.raw);
        assert!(owned.is_contiguous());
    }

    #[test]
    fn test_view_to_owned_noncontiguous_column() {
        // Test converting a non-contiguous column slice to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let col_slice = view.slice(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        // This is non-contiguous (stride [3])
        assert!(!col_slice.is_contiguous());
        assert_eq!(*col_slice.shape(), vec![2]);
        
        // Convert to owned
        let owned = col_slice.owned();
        
        // Owned should be contiguous with correct values
        assert!(owned.is_contiguous());
        assert_eq!(*owned.shape(), vec![2]);
        assert_eq!(owned.raw, vec![2, 5].into_boxed_slice());
        assert_eq!(index_tensor(Idx::At(0), &owned.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &owned.view()).unwrap(), 5);
    }

    #[test]
    fn test_view_to_owned_row_slice() {
        // Test converting a row slice to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let row_slice = view.slice(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        
        assert!(row_slice.is_contiguous());
        
        let owned = row_slice.owned();
        
        assert_eq!(*owned.shape(), vec![3]);
        assert_eq!(owned.raw, vec![4, 5, 6].into_boxed_slice());
    }

    // #[test]
    // fn test_view_to_owned_reshaped() {
    //     // Test converting a reshaped view to owned
    //     let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    //     let view = tensor.view();
    //     let reshaped = view.view_as(vec![3, 2]).unwrap();
        
    //     let owned = reshaped.owned();
        
    //     assert_eq!(*owned.shape(), vec![3, 2]);
    //     assert_eq!(owned.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    //     assert!(owned.is_contiguous());
    // }

    #[test]
    fn test_view_to_owned_3d_slice() {
        // Test converting a 3D slice to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        let view = tensor.view();
        let depth_slice = view.slice(0, 1..1).unwrap(); // Second depth: [5, 6, 7, 8]
        
        let owned = depth_slice.owned();
        
        assert_eq!(*owned.shape(), vec![2, 2]);
        assert_eq!(owned.raw, vec![5, 6, 7, 8].into_boxed_slice());
        assert!(owned.is_contiguous());
    }

    #[test]
    fn test_viewmut_to_owned() {
        // Test that mutable views can also be converted to owned
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view_mut = tensor.view_mut();
        let owned = view_mut.owned();
        
        assert_eq!(*owned.shape(), vec![2, 3]);
        assert_eq!(owned.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_instantiation() {
        let tensor = Tensor::<f32>::zeros((1, 1));

        assert_eq!(*tensor.shape(), vec![1, 1]);
        assert_eq!(tensor.raw.len(), 1);
        assert_eq!(tensor.raw[0], 0.0);

    }

    #[test]
    fn test_range_to_syntax() {
        // Test ..5 syntax (from start to 5)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, ..5).unwrap();
        assert_eq!(*slice.shape(), vec![5]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 40);
        assert_eq!(index_tensor(Idx::At(4), &slice).unwrap(), 50);
        
        // Test mutation - add to each element
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, ..5).unwrap();
        slice_mut.set(&Idx::At(0), 11).unwrap();
        slice_mut.set(&Idx::At(2), 33).unwrap();
        slice_mut.set(&Idx::At(4), 55).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 33);
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 55);
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 60); // Unchanged
    }

    #[test]
    fn test_range_from_syntax() {
        // Test 2.. syntax (from 2 to end)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, 2..).unwrap();
        assert_eq!(*slice.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 40);
        assert_eq!(index_tensor(Idx::At(5), &slice).unwrap(), 80);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 2..).unwrap();
        slice_mut.set(&Idx::At(0), 333).unwrap();
        slice_mut.set(&Idx::At(5), 888).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 333);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 888);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 10); // Unchanged
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 20); // Unchanged
    }

    #[test]
    fn test_range_full_syntax() {
        // Test .. syntax (full range)
        let buf = vec![10, 20, 30, 40];
        let shape = vec![4];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, ..).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 40);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, ..).unwrap();
        slice_mut.set(&Idx::At(0), 100).unwrap();
        slice_mut.set(&Idx::At(3), 400).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 100);
        assert_eq!(index_tensor(Idx::At(3), &view).unwrap(), 400);
    }

    #[test]
    fn test_inclusive_range_syntax() {
        // Test 2..=5 syntax (inclusive end)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, 2..=5).unwrap();
        assert_eq!(*slice.shape(), vec![4]); // Indices 2, 3, 4, 5
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 40);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 50);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 60);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 2..=5).unwrap();
        slice_mut.set(&Idx::At(0), 333).unwrap();
        slice_mut.set(&Idx::At(3), 666).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 333);
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 666);
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 20); // Unchanged
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 70); // Unchanged
    }

    #[test]
    fn test_inclusive_range_zero_end() {
        // Test 0..=2 syntax (first 3 elements)
        let buf = vec![10, 20, 30, 40, 50];
        let shape = vec![5];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, 0..=2).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 0, 1, 2
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 0..=2).unwrap();
        slice_mut.set(&Idx::At(1), 222).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 222);
    }

    #[test]
    fn test_single_index_syntax() {
        // Test single index 3 (reduces dimension)
        let buf = vec![10, 20, 30, 40, 50, 60];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);
        
        // Slice along dimension 1 (columns) at index 1
        let view = tensor.view();
        let slice = view.slice(1, 1).unwrap();
        assert_eq!(*slice.shape(), vec![2]); // Dimension reduced
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 20); // First row, column 1
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50); // Second row, column 1
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(1, 1).unwrap();
        slice_mut.set(&Idx::At(0), 222).unwrap();
        slice_mut.set(&Idx::At(1), 555).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &view).unwrap(), 222);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &view).unwrap(), 555);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &view).unwrap(), 10); // Unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &view).unwrap(), 30); // Unchanged
    }

    #[test]
    fn test_reverse_range_exclusive_with_values() {
        // Test 5..2 auto-reverse (step=-1)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in reversed slice
        let view = tensor.view();
        let slice = view.slice(0, 5..2).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 5, 4, 3
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 60); // Index 5
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50); // Index 4
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 40); // Index 3
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 5..2).unwrap();
        slice_mut.set(&Idx::At(0), 666).unwrap(); // Should modify index 5
        slice_mut.set(&Idx::At(2), 444).unwrap(); // Should modify index 3
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 666);
        assert_eq!(index_tensor(Idx::At(3), &view).unwrap(), 444);
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 30); // Unchanged
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 70); // Unchanged
    }

    #[test]
    fn test_slice_from_with_step() {
        // Test Slice::from(8..).step(-1) - from index 8 to start, reversed
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let shape = vec![10];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(8..).step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![9]); // Indices 8, 7, 6, 5, 4, 3, 2, 1, 0
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 90);  // Index 8
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 80);  // Index 7
        assert_eq!(index_tensor(Idx::At(8), &slice).unwrap(), 10);  // Index 0
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::from(8..).step(-1)).unwrap();
        slice_mut.set(&Idx::At(0), 999).unwrap();  // Should modify index 8
        slice_mut.set(&Idx::At(8), 111).unwrap();  // Should modify index 0
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(8), &view).unwrap(), 999);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 111);
        assert_eq!(index_tensor(Idx::At(9), &view).unwrap(), 100); // Unchanged
    }

    #[test]
    fn test_slice_to_with_step() {
        // Test Slice::from(..5).step(-1) - from end down to index 5 (exclusive)
        // With negative step and end=5, start defaults to last index (7)
        // So this gives indices [7, 6, 5] going backwards = 3 elements
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(..5).step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![2]); // Indices 7, 6 (end=5 is exclusive)
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 80); // Index 7
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 70); // Index 6
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::from(..5).step(-1)).unwrap();
        slice_mut.set(&Idx::At(0), 888).unwrap(); // Should modify index 7
        slice_mut.set(&Idx::At(1), 777).unwrap(); // Should modify index 6
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 888);
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 777);
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 60); // Unchanged (end is exclusive)
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 50); // Unchanged
    }

    #[test]
    fn test_reverse_first_n_elements() {
        // To reverse first N elements, use Slice::new(Some(N-1), None, Some(-1))
        // This gets elements from index N-1 down to 0
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Reverse first 5 elements: indices 4, 3, 2, 1, 0
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(4), None, -1)).unwrap();
        assert_eq!(*slice.shape(), vec![5]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 50); // Index 4
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 40); // Index 3
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30); // Index 2
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 20); // Index 1
        assert_eq!(index_tensor(Idx::At(4), &slice).unwrap(), 10); // Index 0
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::new(Some(4), None, -1)).unwrap();
        slice_mut.set(&Idx::At(0), 555).unwrap(); // Should modify index 4
        slice_mut.set(&Idx::At(4), 111).unwrap(); // Should modify index 0
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 555);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 111);
    }

    #[test]
    fn test_manual_slice_construction_values() {
        // Test Slice::new(Some(8), Some(2), Some(-2)) - from 8 to 2, step -2
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let shape = vec![10];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(8), Some(2), -2)).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 8, 6, 4
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 90);  // Index 8
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 70);  // Index 6
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 50);  // Index 4
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::new(Some(8), Some(2), -2)).unwrap();
        slice_mut.set(&Idx::At(0), 999).unwrap();  // Should modify index 8
        slice_mut.set(&Idx::At(2), 555).unwrap();  // Should modify index 4
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(8), &view).unwrap(), 999);
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 555);
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 70); // Unchanged until we check
    }

    #[test]
    fn test_empty_slice_same_indices() {
        // Test 5..5 (empty slice)
        let buf = vec![10, 20, 30, 40, 50, 60];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 5..5).unwrap();
        assert_eq!(*slice.shape(), vec![]); // Empty - dimension collapsed
    }

    #[test]
    fn test_matrix_column_slice_with_mutation() {
        // Test slicing a column from a matrix with value checks and mutation
        let buf = vec![
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12
        ];
        let shape = vec![3, 4]; // 3 rows, 4 columns
        let mut tensor = make_tensor(buf, shape);
        
        // Slice column 2 (third column)
        let view = tensor.view();
        let slice = view.slice(1, 2..3).unwrap();
        assert_eq!(*slice.shape(), vec![3, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &slice).unwrap(), 11);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(1, 2..3).unwrap();
        slice_mut.set(&Idx::Coord(vec![0, 0]), 33).unwrap();
        slice_mut.set(&Idx::Coord(vec![2, 0]), 111).unwrap();
        
        // Verify mutations in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &view).unwrap(), 33);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2]), &view).unwrap(), 111);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &view).unwrap(), 7); // Unchanged
    }

    #[test]
    fn test_3d_depth_slice_with_mutation() {
        // Test 3D tensor depth slicing with value checks and mutation
        let buf = vec![
            // Depth 0
            1, 2,
            3, 4,
            // Depth 1
            5, 6,
            7, 8,
            // Depth 2
            9, 10,
            11, 12
        ];
        let shape = vec![3, 2, 2]; // 3 depth, 2 rows, 2 columns
        let mut tensor = make_tensor(buf, shape);
        
        // Slice depth 1..=1 (second depth slice)
        let view = tensor.view();
        let slice = view.slice(0, 1..=1).unwrap();
        assert_eq!(*slice.shape(), vec![1, 2, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &slice).unwrap(), 8);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 1..=1).unwrap();
        slice_mut.set(&Idx::Coord(vec![0, 0, 0]), 55).unwrap();
        slice_mut.set(&Idx::Coord(vec![0, 1, 1]), 88).unwrap();
        
        // Verify mutations in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &view).unwrap(), 55);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &view).unwrap(), 88);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0, 0]), &view).unwrap(), 9); // Unchanged
    }

    #[test]
    fn test_chained_slices_with_mutation() {
        // Test chaining different slice syntaxes with mutation
        let buf = vec![
            1,  2,  3,  4,  5,  6,
            7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24
        ];
        let shape = vec![4, 6]; // 4 rows, 6 columns
        let mut tensor = make_tensor(buf, shape);
        
        // First slice: rows 1..3
        let view = tensor.view();
        let slice1 = view.slice(0, 1..3).unwrap();
        assert_eq!(*slice1.shape(), vec![2, 6]);
        
        // Second slice: columns 2..=4 (inclusive)
        let slice2 = slice1.slice(1, 2..=4).unwrap();
        assert_eq!(*slice2.shape(), vec![2, 3]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice2).unwrap(), 9);   // Row 1, Col 2
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &slice2).unwrap(), 11);  // Row 1, Col 4
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice2).unwrap(), 15);  // Row 2, Col 2
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &slice2).unwrap(), 17);  // Row 2, Col 4
        
        // Test mutation through chained slices
        let mut view_mut = tensor.view_mut();
        let mut slice1_mut = view_mut.slice_mut(0, 1..3).unwrap();
        let mut slice2_mut = slice1_mut.slice_mut(1, 2..=4).unwrap();
        slice2_mut.set(&Idx::Coord(vec![0, 1]), 999).unwrap(); // Row 1, Col 3 in original
        slice2_mut.set(&Idx::Coord(vec![1, 2]), 1717).unwrap(); // Row 2, Col 4 in original
        
        // Verify mutations in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 3]), &view).unwrap(), 999);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 4]), &view).unwrap(), 1717);
    }

    #[test]
    fn test_step_2_with_mutation() {
        // Test custom step with value checks and mutation
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let mut tensor = make_tensor(buf, shape);
        
        // Every other element starting from index 1
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(1..9).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![4]); // Indices 1, 3, 5, 7
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 8);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::from(1..9).step(2)).unwrap();
        slice_mut.set(&Idx::At(0), 22).unwrap();  // Index 1
        slice_mut.set(&Idx::At(3), 88).unwrap();  // Index 7
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 22);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 88);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 3); // Unchanged (skipped by step)
    }
}
