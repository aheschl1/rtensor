
pub mod primitives;
pub mod meta;
pub mod tensor;
pub mod idx;
pub mod value;

pub use meta::{Dim, Shape, Stride, MetaTensor, MetaTensorView, shape_to_stride};
pub use primitives::{CpuTensor, TensorView, CpuTensorView, TensorViewMut};


#[cfg(test)]
mod tests {
    use crate::{backend::Backend, core::{idx::Idx, value::TensorValue, tensor::{AsTensor, AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, CpuTensor, MetaTensor, MetaTensorView, Shape, Stride}};

    fn make_tensor<T: TensorValue>(buf: Vec<T>, shape: Shape) -> CpuTensor<T> {
        CpuTensor::from_buf(buf, shape).unwrap()
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
        assert_eq!(index_tensor(Idx::Coord(&[1]), &slice2).unwrap(), 4);
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
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice).unwrap(), 5);

        // second depth
        let view = tensor.view();
        let slice_second_depth = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice_second_depth.shape(), vec![2, 2]);
        assert_eq!(*slice_second_depth.stride(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice_second_depth).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice_second_depth).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice_second_depth).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice_second_depth).unwrap(), 9);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap(); // slice along row, should give a view of shape [2, 2]
        assert_eq!(*slice2.shape(), vec![2, 2]);
        assert_eq!(*slice2.stride(), vec![4, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice2).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice2).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice2).unwrap(), 7);

        // column slice
        let view = tensor.view();
        let slice3 = view.slice(2, 0..0).unwrap(); // slice along column
        assert_eq!(*slice3.shape(), vec![2, 2]);
        assert_eq!(*slice3.stride(), vec![4, 2]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice3).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice3).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice3).unwrap(), 8);
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
        assert_eq!(index_tensor(Idx::Coord(&[]), &slice_of_slice).unwrap(), 6);
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
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[0, 2]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 2]), &slice).unwrap(), 9);
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
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[2, 0]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(&[2, 1]), &slice).unwrap(), 11);
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
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 0]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 1]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1, 1]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 0]), &slice).unwrap(), 9);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 1]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1, 0]), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1, 1]), &slice).unwrap(), 12);
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
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..3).unwrap(); // Rows 1 and 2
        assert_eq!(*slice.shape(), vec![2, 2]);
        
        // Modify values in the slice
        slice.set(&Idx::Coord(&[0, 0]), 30).unwrap();
        slice.set(&Idx::Coord(&[1, 1]), 80).unwrap();
        
        // Check original tensor was modified
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &tensor.view()).unwrap(), 30);
        assert_eq!(index_tensor(Idx::Coord(&[2, 1]), &tensor.view()).unwrap(), 80);
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
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[1, 2]), &slice).unwrap(), 6);
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
    fn slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);

        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice).unwrap(), 9);

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
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &tensor.view()).unwrap(), 50);
    }

    #[test]
    fn test_column() {
    let tensor = CpuTensor::column(vec![1, 2, 3]);
    assert_eq!(*tensor.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_row() {
    let tensor = CpuTensor::row(vec![1, 2, 3]);
    assert_eq!(*tensor.shape(), vec![1, 3]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[0, 2]), &tensor.view()).unwrap(), 3);

        assert_eq!(tensor.view().get(&[0, 1]).unwrap(), 2);
    }

    #[test]
    fn test_scalar() {
        let buf = vec![42];
        let shape = vec![];
        let tensor = make_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::Item, &tensor.view()).unwrap(), 42);
        assert!(tensor.is_scalar());
        assert_eq!(CpuTensor::scalar(42), tensor);
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

        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[0, 2]), &tensor.view()).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[1, 2]), &tensor.view()).unwrap(), 6);

        tensor.view_mut().set(&Idx::Coord(&[1, 2]), 100).unwrap();
        assert_eq!(index_tensor(Idx::Coord(&[1, 2]), &tensor.view()).unwrap(), 100);
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
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 0]), &tensor.view()).unwrap(), 1); // depth, row, column
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 0]), &tensor.view()).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 1]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1, 0]), &tensor.view()).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1, 1]), &tensor.view()).unwrap(), 9);

        // modify
        tensor.view_mut().set(&Idx::Coord(&[1, 0, 0]), 67).unwrap();
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 0]), &tensor.view()).unwrap(), 67);
    }

    #[test]
    fn test_view_as_owned_success() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        let reshaped = tensor.view().view_as(vec![3, 2]).unwrap();
        assert_eq!(*reshaped.shape(), vec![3, 2]);
        assert_eq!(*reshaped.stride(), vec![2, 1]);
        // Row-major sequence preserved
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &reshaped).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &reshaped).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &reshaped).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &reshaped).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[2, 0]), &reshaped).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[2, 1]), &reshaped).unwrap(), 6);
    }

    #[test]
    fn test_view_as_owned_error() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        assert!(matches!(tensor.view().view_as(vec![4, 2]), Err(TensorError::InvalidShape)));
    }

    #[test]
    fn test_view_as_slice_success() {
        let buf = vec![
            1, 2, 3, 
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);

        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        let reshaped = slice.view_as(vec![1, 3]).unwrap();
        assert_eq!(*reshaped.shape(), vec![1, 3]);
        assert_eq!(*reshaped.stride(), vec![3, 1]);
        // Values should correspond to original slice elements 4,5,6
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &reshaped).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &reshaped).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[0, 2]), &reshaped).unwrap(), 6);
    }

    #[test]
    fn test_view_as_mut_view_modify() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let mut tensor = make_tensor(buf, shape);
        let mut view_mut = tensor.view_mut(); // shape [2,2]
        // Modify before reshaping to avoid borrow conflicts
        view_mut.set(&Idx::Coord(&[1, 0]), 40).unwrap(); // coordinate [1,0] maps to linear index 2
        let reshaped = view_mut.view_as(vec![4]).unwrap(); // reshape to flat vector
        assert_eq!(*reshaped.shape(), vec![4]);
        assert_eq!(*reshaped.stride(), vec![1]);
        // Check reshaped view sees update at linear index 2
        assert_eq!(index_tensor(Idx::At(2), &reshaped).unwrap(), 40);
    }

    #[test]
    fn test_view_as_scalar() {
        let tensor = CpuTensor::scalar(99); // shape []
        let view1 = tensor.view();
        assert_eq!(*view1.shape(), vec![]);
        let reshaped = view1.view_as(vec![1]).unwrap();
        assert_eq!(*reshaped.shape(), vec![1]);
        assert_eq!(*reshaped.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &reshaped).unwrap(), 99);

        // view as [1, 1, 1]

        let r2 = reshaped.view_as(vec![1, 1, 1]).unwrap();
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 0]), &r2).unwrap(), 99);

    }

    fn index_tensor<'a, T: TensorValue + PartialEq + std::fmt::Debug, B: Backend<T>>(index: Idx<'a>, tensor: &'a impl TensorAccess<T, B>) -> Result<T, TensorError> {
        let r: Result<T, TensorError> = tensor.get(&index);
        let a = match r.as_ref() {
            Ok(v) => Ok(*v),
            Err(e) => return Err(e.clone()),
        };
        let b = match &index {
            Idx::At(i) => tensor.get(Idx::Coord(&[*i])),
            Idx::Coord(idx) => tensor.get(Idx::Coord(idx)),
            Idx::Item => tensor.item(),
        };
        assert_eq!(a, b);
        r
    }

    #[test]
    fn test_shape_to_stride() {
        let shape = vec![2, 2, 3];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![6, 3, 1]);
    }

    #[test]
    fn test_shape_to_stride_single_dim() {
        let shape = vec![4];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_shape_to_stride_empty() {
        let shape: Shape = vec![];
        let stride: Stride = super::shape_to_stride(&shape);

        assert!(stride.is_empty());
    }

    #[test]
    fn test_shape_to_stride_ones() {
        let shape = vec![1, 1, 1];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![1, 1, 1]);
    }

    #[test]
    fn test_shape_to_stride_mixed() {
        let shape = vec![5, 1, 2];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![2, 2, 1]);
    }

    #[test]
    fn test_shape_to_stride_larger() {
        let shape = vec![3, 4, 5];
        let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![20, 5, 1]);
    }

    #[test]
    fn test_from_buf_error() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            CpuTensor::from_buf(buf, shape),
            Err(TensorError::InvalidShape)
        ));
    }

    #[test] 
    fn test_get_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            index_tensor(Idx::Coord(&[0, 0, 0]), &tensor.view()),
            Err(TensorError::WrongDims)
        ));
        assert!(matches!(
            index_tensor(Idx::Coord(&[2, 0]), &tensor.view()),
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
        assert_eq!(tensor.view().get(&Idx::Coord(&[0, 1])).unwrap(), 2);
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 6);

        // Test IndexMut on TensorOwned
        tensor.view_mut().set(vec![1, 1], 55).unwrap();
        assert_eq!(tensor.view().get(&Idx::Coord(&[1, 1])).unwrap(), 55);
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
        assert_eq!(mut_slice.get(&Idx::Coord(&[2])).unwrap(), 33);
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
        assert!(matches!(tensor.view_mut().set(&Idx::Coord(&[1, 2]), 99), Ok(())));
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 99);
    }

    #[test]
    fn test_is_row_and_is_column() {
        let row = CpuTensor::row(vec![1, 2, 3]);
        let col = CpuTensor::column(vec![1, 2, 3]);
        let scalar = CpuTensor::scalar(10);
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
        let scalar = CpuTensor::scalar(5);
        assert!(matches!(scalar.view().slice(0, 0..0), Err(TensorError::InvalidDim)));
    }

    #[test]
    fn test_view_as_slice_error() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap(); // shape [3]
        assert!(matches!(slice.view_as(vec![2, 2]), Err(TensorError::InvalidShape)));
    }

    #[test]
    fn test_view_mut_as_error() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let view_mut = tensor.view_mut();
        assert!(matches!(view_mut.view_as(vec![3, 2]), Err(TensorError::InvalidShape)));
    }

    #[test]
    fn test_item_wrong_dims_error() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tensor.view().get(&Idx::Item), Err(TensorError::WrongDims)));
    }

    #[test]
    fn test_from_buf_empty_shape_error() {
        assert!(matches!(CpuTensor::from_buf(Vec::<i32>::new(), vec![]), Err(TensorError::InvalidShape)));
    }

    #[test]
    fn test_modify_after_reshape_reflects() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        {
            let view_mut = tensor.view_mut();
            let mut reshaped = view_mut.view_as(vec![4]).unwrap();
            reshaped.set(Idx::At(3), 40).unwrap(); // modify last element
            assert_eq!(reshaped.get(&Idx::At(3)).unwrap(), 40);
        }
        assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 40);
    }

    #[test]
    fn test_slice_single_dim_to_scalar() {
        let tensor = make_tensor(vec![42], vec![1]);

        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.shape(), vec![]);
        assert_eq!(slice.get(&Idx::Item).unwrap(), 42);
        assert_eq!(slice.get(&Idx::Coord(&[])).unwrap(), 42);
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
        let scalar = CpuTensor::scalar(10);
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
        let scalar = CpuTensor::scalar(42);
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

    #[test]
    fn test_reshape_noncontiguous_slice_contiguous() {
        // Start non-contiguous 1D view (stride [3])
        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        let v3 = mat.view();
        let col_like = v3.slice(1, 0..0).unwrap(); // shape [2], stride [3]
        assert!(!col_like.is_contiguous());

        // Reshape to [1,2] -> becomes contiguous
        let reshaped = col_like.view_as(vec![1, 2]).unwrap();
        assert!(reshaped.is_contiguous());
        assert_eq!(*reshaped.shape(), vec![1,2]);
        // Note: reshaping a non-contiguous slice uses underlying memory order
        // starting at the slice's offset; here it's [1, 2]
        assert_eq!(reshaped.get(vec![0,0]).unwrap(), 1);
        assert_eq!(reshaped.get(vec![0,1]).unwrap(), 2);
    }

    #[test]
    fn test_tensor_meta_direct_impl() {
        // contiguous meta
        let shape = vec![2, 3];
        let stride = super::shape_to_stride(&shape);
        let meta = MetaTensor::new(shape.clone(), stride.clone(), 0);
        assert_eq!(meta.shape(), &shape);
        assert_eq!(meta.stride(), &stride);
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
        assert_eq!(col_like.get(vec![0]).unwrap(), col_like.get(&Idx::Coord(&[0])).unwrap());
        assert_eq!(col_like.get(vec![1]).unwrap(), col_like.get(&Idx::Coord(&[1])).unwrap());
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

    #[test]
    fn test_view_to_owned_reshaped() {
        // Test converting a reshaped view to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let reshaped = view.view_as(vec![3, 2]).unwrap();
        
        let owned = reshaped.owned();
        
        assert_eq!(*owned.shape(), vec![3, 2]);
        assert_eq!(owned.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
        assert!(owned.is_contiguous());
    }

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
}
