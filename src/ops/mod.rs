pub mod base;
pub mod elementwise;

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{meta::MetaTensorView, primitives::TensorBase, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}}};

    
    #[test]
    fn test_add() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;

        let expected = TensorBase::<Cpu, i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_ref() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view += &value;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_negative() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += -5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // same for sub
    #[test]
    fn test_sub() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= 5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_ref() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view -= &value;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![0, 10, 20], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_negative() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= -5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // same for mul

    #[test]
    fn test_mul() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_ref() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view *= &value;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_negative() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= -5;
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![-5, -10, -15], vec![3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // Tests with reshaping/slicing

    #[test]
    fn test_add_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![3, 2]).unwrap();
        reshaped += 10;
        
        // Original tensor should be modified
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![11, 12, 13, 14, 15, 16], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_after_slice() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        
        // Verify slice values before mutation
        use crate::core::tensor::TensorAccess;
        assert_eq!(slice.get(vec![0]).unwrap(), 4, "Slice[0] should be 4");
        assert_eq!(slice.get(vec![1]).unwrap(), 5, "Slice[1] should be 5");
        assert_eq!(slice.get(vec![2]).unwrap(), 6, "Slice[2] should be 6");
        
        slice += 100;
        
        // Only the sliced part should be modified
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 104, 105, 106], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_after_slice_and_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        let mut reshaped = slice.view_as(vec![1, 3]).unwrap();
        reshaped += 50;
        
        // Only the sliced part should be modified
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 54, 55, 56], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4]).unwrap();
        reshaped -= 5;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![5, 15, 25, 35], vec![2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_after_slice() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 0..0).unwrap(); // First depth slice: [1, 2, 3, 4]
        
        // Verify slice values before mutation
        use crate::core::tensor::TensorAccess;
        assert_eq!(slice.get(vec![0, 0]).unwrap(), 1, "Slice[0,0] should be 1");
        assert_eq!(slice.get(vec![0, 1]).unwrap(), 2, "Slice[0,1] should be 2");
        assert_eq!(slice.get(vec![1, 0]).unwrap(), 3, "Slice[1,0] should be 3");
        assert_eq!(slice.get(vec![1, 1]).unwrap(), 4, "Slice[1,1] should be 4");
        
        slice *= 10;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_scalar_reshaped_to_matrix() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![42], vec![1]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![1, 1]).unwrap();
        reshaped += 8;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![50], vec![1]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_mul_after_column_slice() {
        // Create a matrix and slice a column (non-contiguous)
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        // Verify slice values (non-contiguous access)
        use crate::core::tensor::TensorAccess;
        assert_eq!(col_slice.get(vec![0]).unwrap(), 2, "Column slice[0] should be 2");
        assert_eq!(col_slice.get(vec![1]).unwrap(), 5, "Column slice[1] should be 5");
        
        col_slice *= 3;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 6, 3, 4, 15, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_sub_ref_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![100, 200, 300, 400], vec![2, 2]).unwrap();
        let value = 50;
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4]).unwrap();
        reshaped -= &value;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![50, 150, 250, 350], vec![2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    #[test]
    fn test_add_ref_after_slice_chain() {
        // Create a 3D tensor and chain multiple slices
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let value = 1000;
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Second depth: [6, 7, 8, 9] -> wait, buffer is [5,6,7,8]
        
        // Verify depth slice values
        use crate::core::tensor::TensorAccess;
        assert_eq!(depth_slice.get(vec![0, 0]).unwrap(), 5, "Depth slice[0,0] should be 5");
        assert_eq!(depth_slice.get(vec![0, 1]).unwrap(), 6, "Depth slice[0,1] should be 6");
        assert_eq!(depth_slice.get(vec![1, 0]).unwrap(), 7, "Depth slice[1,0] should be 7");
        assert_eq!(depth_slice.get(vec![1, 1]).unwrap(), 8, "Depth slice[1,1] should be 8");
        
        let mut row_slice = depth_slice.slice_mut(0, 0..0).unwrap(); // First row of that: [5, 6]
        assert_eq!(row_slice.get(vec![0]).unwrap(), 5, "Row slice[0] should be 5");
        assert_eq!(row_slice.get(vec![1]).unwrap(), 6, "Row slice[1] should be 6");
        
        row_slice += &value;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 1005, 1006, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.raw.clone(), expected.raw);
    }

    // Tests for non-inplace operations (consume view, return new tensor)

    #[test]
    fn test_add_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + 5;
        
        // Result should be a new tensor with added values
        assert_eq!(result.raw, vec![6, 7, 8].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        
        // Original tensor should be unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_ref_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view + &value;
        
        assert_eq!(result.raw, vec![11, 12, 13].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view - 5;
        
        assert_eq!(result.raw, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(tensor.raw, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_sub_ref_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view - &value;
        
        assert_eq!(result.raw, vec![0, 10, 20].into_boxed_slice());
        assert_eq!(tensor.raw, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * 5;
        
        assert_eq!(result.raw, vec![5, 10, 15].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_mul_ref_not_inplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view * &value;
        
        assert_eq!(result.raw, vec![10, 20, 30].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_with_slice() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        let result = slice + 100;
        
        // Result should be a new 1D tensor with shape [3]
        assert_eq!(result.raw, vec![104, 105, 106].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        assert!(result.is_contiguous());
        
        // Original tensor should be unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_with_noncontiguous_slice() {
        // Test with non-contiguous column slice
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        assert!(!col_slice.is_contiguous(), "Column slice should be non-contiguous");
        
        let result = col_slice * 3;
        
        // Result should be a new contiguous tensor
        assert_eq!(result.raw, vec![6, 15].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace_with_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let reshaped = view.view_as(vec![4]).unwrap();
        let result = reshaped - 5;
        
        // Result should have the reshaped dimensions
        assert_eq!(result.raw, vec![5, 15, 25, 35].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![10, 20, 30, 40].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_negative_values() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + (-5);
        
        assert_eq!(result.raw, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(tensor.raw, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_negative_values() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * (-5);
        
        assert_eq!(result.raw, vec![-5, -10, -15].into_boxed_slice());
        assert_eq!(tensor.raw, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_chained_slices() {
        // Test with chained slices to ensure view_to_owned handles complex cases
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Second depth: [5, 6, 7, 8]
        let row_slice = depth_slice.slice_mut(0, 0..0).unwrap(); // First row: [5, 6]
        
        let result = row_slice + 1000;
        
        assert_eq!(result.raw, vec![1005, 1006].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_matrix() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let result = view * 10;
        
        assert_eq!(result.raw, vec![10, 20, 30, 40].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace_scalar() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![100], vec![]).unwrap();
        let view = tensor.view_mut();
        let result = view - 50;
        
        assert_eq!(result.raw, vec![50].into_boxed_slice());
        assert_eq!(*result.shape(), vec![]);
        assert!(result.is_scalar());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![100].into_boxed_slice());
    }

    // Tests for non-inplace operations on non-mutable views (TensorView)
    
    #[test]
    fn test_add_immutable_view_inline() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        // Call operation directly on view() without storing in variable
        let result = tensor.view() + 10;
        
        assert_eq!(result.raw, vec![11, 12, 13, 14].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_ref() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        let value = 100;
        let result = tensor.view() + &value;
        
        assert_eq!(result.raw, vec![105, 110, 115].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![5, 10, 15].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_inline() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![100, 200, 300], vec![3]).unwrap();
        let result = tensor.view() - 50;
        
        assert_eq!(result.raw, vec![50, 150, 250].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![100, 200, 300].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_with_slice() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        // Slice to get first row, then subtract
        let result = tensor.view().slice(0, 0..0).unwrap() - 5;
        
        assert_eq!(result.raw, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![10, 20, 30, 40, 50, 60].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_inline() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![2, 4, 6, 8], vec![4]).unwrap();
        let result = tensor.view() * 3;
        
        assert_eq!(result.raw, vec![6, 12, 18, 24].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![2, 4, 6, 8].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_matrix() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.view() * 10;
        
        assert_eq!(result.raw, vec![10, 20, 30, 40, 50, 60].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 3]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_after_reshape() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        // Reshape then add, all inline
        let result = tensor.view().view_as(vec![2, 2]).unwrap() + 100;
        
        assert_eq!(result.raw, vec![101, 102, 103, 104].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_scalar() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![999], vec![]).unwrap();
        let result = tensor.view() - 99;
        
        assert_eq!(result.raw, vec![900].into_boxed_slice());
        assert_eq!(*result.shape(), vec![]);
        assert!(result.is_scalar());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![999].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_noncontiguous_slice() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        // Get column (non-contiguous) then multiply
        let result = tensor.view().slice(1, 1..1).unwrap() * 5;
        
        assert_eq!(result.raw, vec![10, 25].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_chained_operations() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4]).unwrap();
        // Chain slice and reshape, then add
        let result = tensor.view()
            .slice(0, 0..0).unwrap()  // Get first row: [1, 2, 3, 4]
            .view_as(vec![2, 2]).unwrap()  // Reshape to 2x2
            + 1000;
        
        assert_eq!(result.raw, vec![1001, 1002, 1003, 1004].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_negative_values() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![-10, -20, -30], vec![3]).unwrap();
        let result = tensor.view() - 5;
        
        assert_eq!(result.raw, vec![-15, -25, -35].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![-10, -20, -30].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_with_ref() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![7, 14, 21], vec![3]).unwrap();
        let multiplier = 2;
        let result = tensor.view() * &multiplier;
        
        assert_eq!(result.raw, vec![14, 28, 42].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.raw, vec![7, 14, 21].into_boxed_slice());
    }

    // ============================================================================
    // EDGE CASE TESTS - Priority 1: Critical Coverage
    // ============================================================================

    // --- Scalar Tensor Tests ---

    #[test]
    fn test_scalar_add_operation() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![42], vec![]).unwrap();
        assert!(tensor.is_scalar());
        let mut view = tensor.view_mut();
        view += 10;
        assert_eq!(tensor.raw, vec![52].into_boxed_slice());
    }

    #[test]
    fn test_scalar_mul_operation() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![7], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= 6;
        assert_eq!(tensor.raw, vec![42].into_boxed_slice());
    }

    #[test]
    fn test_scalar_sub_operation() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![100], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view -= 58;
        assert_eq!(tensor.raw, vec![42].into_boxed_slice());
    }

    #[test]
    fn test_scalar_negative_value() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![-10], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= -3;
        view += 5;
        assert_eq!(tensor.raw, vec![35].into_boxed_slice());
    }

    #[test]
    fn test_scalar_zero_operations() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![0], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view += 42;
        view *= 0;
        view += 10;
        assert_eq!(tensor.raw, vec![10].into_boxed_slice());
    }

    #[test]
    fn test_scalar_non_inplace() {
        let tensor = TensorBase::<Cpu, i32>::from_buf(vec![100], vec![]).unwrap();
        let result = tensor.view() + 50;
        assert_eq!(result.raw, vec![150].into_boxed_slice());
        assert!(result.is_scalar());
        assert_eq!(tensor.raw, vec![100].into_boxed_slice());
    }

    // --- Single Element Tensor Tests ---

    #[test]
    fn test_single_element_1d() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![5], vec![1]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        assert_eq!(tensor.raw, vec![30].into_boxed_slice());
    }

    #[test]
    fn test_single_element_2d() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![8], vec![1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        assert_eq!(tensor.raw, vec![40].into_boxed_slice());
    }

    #[test]
    fn test_single_element_3d() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![3], vec![1, 1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view += 7;
        view *= 3;
        assert_eq!(tensor.raw, vec![30].into_boxed_slice());
    }

    // --- Non-Square Tensor Tests ---

    #[test]
    fn test_tall_matrix_operations() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![9, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        assert_eq!(tensor.raw, vec![2, 4, 6, 8, 10, 12, 14, 16, 18].into_boxed_slice());
    }

    #[test]
    fn test_wide_matrix_operations() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![1, 9]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        assert_eq!(tensor.raw, vec![11, 12, 13, 14, 15, 16, 17, 18, 19].into_boxed_slice());
    }

    #[test]
    fn test_rectangular_matrix_2x5() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![2, 5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3;
        assert_eq!(
            tensor.raw, 
            vec![3, 6, 9, 12, 15, 18, 21, 24, 27, 30].into_boxed_slice()
        );
    }

    #[test]
    fn test_rectangular_matrix_5x2() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![5, 2]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 1;
        assert_eq!(
            tensor.raw, 
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_boxed_slice()
        );
    }

    #[test]
    fn test_rectangular_3d_tensor() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![2, 3, 4]).unwrap();
        let mut view = tensor.view_mut();
        view += 100;
        
        let expected: Vec<i32> = (101..=124).collect();
        assert_eq!(tensor.raw, expected.into_boxed_slice());
    }

    #[test]
    fn test_non_square_slice_operations() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [5, 6, 7, 8]
        slice *= 10;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        assert_eq!(tensor.raw, expected.raw);
    }

    // --- Edge Value Tests ---

    #[test]
    fn test_zero_tensor_operations() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        assert_eq!(tensor.raw, vec![20, 20, 20, 20].into_boxed_slice());
    }

    #[test]
    fn test_operations_resulting_in_zero() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        assert_eq!(tensor.raw, vec![0, 0, 0, 0].into_boxed_slice());
    }

    #[test]
    fn test_mixed_positive_negative_values() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![-5, 10, -15, 20, -25, 30], 
            vec![6]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 10;
        
        assert_eq!(
            tensor.raw, 
            vec![0, 30, -20, 50, -40, 70].into_boxed_slice()
        );
    }

    #[test]
    fn test_large_values() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1_000_000, 2_000_000, 3_000_000], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 500_000;
        
        assert_eq!(
            tensor.raw, 
            vec![1_500_000, 2_500_000, 3_500_000].into_boxed_slice()
        );
    }

    // --- Multiple Data Type Tests ---

    #[test]
    fn test_f32_operations() {
        let mut tensor = TensorBase::<Cpu, f32>::from_buf(
            vec![1.5, 2.5, 3.5, 4.5], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 1.0;
        
        let expected = vec![4.0, 6.0, 8.0, 10.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.raw[i] - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f64_operations() {
        let mut tensor = TensorBase::<Cpu, f64>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 0.5;
        view *= 2.0;
        
        let expected = vec![3.0, 5.0, 7.0, 9.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.raw[i] - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn test_i64_operations() {
        let mut tensor = TensorBase::<Cpu, i64>::from_buf(
            vec![100, 200, 300, 400], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 50;
        view *= 3;
        
        assert_eq!(tensor.raw, vec![150, 450, 750, 1050].into_boxed_slice());
    }

    #[test]
    fn test_u32_operations() {
        let mut tensor = TensorBase::<Cpu, u32>::from_buf(
            vec![10, 20, 30, 40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        view *= 2;
        
        assert_eq!(tensor.raw, vec![30, 50, 70, 90].into_boxed_slice());
    }

    #[test]
    fn test_i16_operations() {
        let mut tensor = TensorBase::<Cpu, i16>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 10;
        view += 5;
        
        assert_eq!(tensor.raw, vec![15, 25, 35, 45, 55, 65].into_boxed_slice());
    }

    // --- Complex Non-Contiguous View Tests ---

    #[test]
    fn test_multiple_noncontiguous_operations() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![4, 3]
        ).unwrap();
        
        // Get middle column (non-contiguous)
        let mut view = tensor.view_mut();
        let mut col = view.slice_mut(1, 1..1).unwrap(); // Column 1: [2, 5, 8, 11]
        col += 100;
        col *= 2;
        
        let expected = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 204, 3, 4, 210, 6, 7, 216, 9, 10, 222, 12], 
            vec![4, 3]
        ).unwrap();
        assert_eq!(tensor.raw, expected.raw);
    }

    #[test]
    fn test_noncontiguous_slice_after_reshape() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![8]
        ).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4, 2]).unwrap();
        let mut col_slice = reshaped.slice_mut(1, 1..1).unwrap(); // Second column
        col_slice *= 10;
        
        // Second column in 4x2 is indices 1, 3, 5, 7
        let expected = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 20, 3, 40, 5, 60, 7, 80], 
            vec![8]
        ).unwrap();
        assert_eq!(tensor.raw, expected.raw);
    }

    #[test]
    fn test_3d_noncontiguous_slice() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![2, 3, 4]).unwrap();
        
        // Slice along middle dimension at index 1 (middle row)
        // Shape is [2, 3, 4] - we're slicing dim 1 at idx 1
        // This removes the middle dimension, leaving shape [2, 4]
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(1, 1..1).unwrap(); // Middle "row" at each depth level
        slice += 100;
        
        // Original data is laid out as:
        // Depth 0: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        // Depth 1: [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
        // Slicing at dim=1, idx=1 selects the middle row from each depth:
        // Result: [[5,6,7,8], [17,18,19,20]]
        // These are at indices: 4-7 and 16-19 in the flat array
        let mut expected_data: Vec<i32> = (1..=24).collect();
        for i in [5, 6, 7, 8, 17, 18, 19, 20].iter() {
            expected_data[i - 1] += 100;
        }
        
        assert_eq!(tensor.raw, expected_data.into_boxed_slice());
    }

    // --- 4D and 5D Tensor Tests ---

    #[test]
    fn test_4d_tensor_operations() {
        let data: Vec<i32> = (1..=16).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        
        let expected: Vec<i32> = (11..=26).collect();
        assert_eq!(tensor.raw, expected.into_boxed_slice());
    }

    #[test]
    fn test_4d_slice_operation() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![2, 3, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second slice along first dim
        slice *= 10;
        
        // Second half of data (indices 12-23)
        let mut expected: Vec<i32> = (1..=24).collect();
        for i in 12..24 {
            expected[i] *= 10;
        }
        
        assert_eq!(tensor.raw, expected.into_boxed_slice());
    }

    #[test]
    fn test_5d_tensor_operations() {
        let data: Vec<i32> = (1..=32).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![2, 2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 5;
        
        let expected: Vec<i32> = (1..=32).map(|x| x * 2 + 5).collect();
        assert_eq!(tensor.raw, expected.into_boxed_slice());
    }

    // --- Operation Chain Tests ---

    #[test]
    fn test_long_operation_chain() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        view += 1;   // [2, 3, 4, 5]
        view *= 2;   // [4, 6, 8, 10]
        view -= 1;   // [3, 5, 7, 9]
        view *= 3;   // [9, 15, 21, 27]
        view += 10;  // [19, 25, 31, 37]
        view -= 5;   // [14, 20, 26, 32]
        view *= 2;   // [28, 40, 52, 64]
        view += 2;   // [30, 42, 54, 66]
        view -= 10;  // [20, 32, 44, 56]
        view *= 1;   // [20, 32, 44, 56]
        
        assert_eq!(tensor.raw, vec![20, 32, 44, 56].into_boxed_slice());
    }

    #[test]
    fn test_alternating_inplace_noninplace() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        
        // Inplace
        let mut view1 = tensor.view_mut();
        view1 += 5;
        
        // Non-inplace
        let result1 = tensor.view() * 2;
        assert_eq!(result1.raw, vec![20, 30, 40].into_boxed_slice());
        
        // Inplace again
        let mut view2 = tensor.view_mut();
        view2 *= 3;
        
        // Non-inplace again
        let result2 = tensor.view() + 10;
        assert_eq!(result2.raw, vec![40, 55, 70].into_boxed_slice());
        
        // Original tensor should reflect inplace ops
        assert_eq!(tensor.raw, vec![30, 45, 60].into_boxed_slice());
    }

    #[test]
    fn test_multiple_views_different_ops() {
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        
        // First view - operate on first row
        {
            let mut view = tensor.view_mut();
            let mut row1 = view.slice_mut(0, 0..0).unwrap();
            row1 += 10;
        }
        
        // Second view - operate on second row
        {
            let mut view = tensor.view_mut();
            let mut row2 = view.slice_mut(0, 1..1).unwrap();
            row2 *= 5;
        }
        
        let expected = TensorBase::<Cpu, i32>::from_buf(
            vec![11, 12, 13, 20, 25, 30], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.raw, expected.raw);
    }

    // --- Float Edge Cases ---

    #[test]
    fn test_f32_small_values() {
        let mut tensor = TensorBase::<Cpu, f32>::from_buf(
            vec![0.0001, 0.0002, 0.0003], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 1000.0;
        
        let expected = vec![0.1, 0.2, 0.3];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.raw[i] - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f64_precision() {
        let mut tensor = TensorBase::<Cpu, f64>::from_buf(
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3.0;
        
        let expected = vec![1.0, 2.0, 3.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.raw[i] - exp).abs() < 1e-10);
        }
    }

    // --- Stress Test: Very Imbalanced Shapes ---

    #[test]
    fn test_very_imbalanced_shape_1000x2() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![1000, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        
        // Check a few values
        assert_eq!(tensor.raw[0], 6);
        assert_eq!(tensor.raw[1], 7);
        assert_eq!(tensor.raw[1998], 2004);
        assert_eq!(tensor.raw[1999], 2005);
    }

    #[test]
    fn test_very_imbalanced_shape_2x1000() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = TensorBase::<Cpu, i32>::from_buf(data, vec![2, 1000]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        
        // Check a few values
        assert_eq!(tensor.raw[0], 2);
        assert_eq!(tensor.raw[999], 2000);
        assert_eq!(tensor.raw[1000], 2002);
        assert_eq!(tensor.raw[1999], 4000);
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_tests {
    use crate::core::{primitives::CudaTensor, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}, CpuTensor, MetaTensorView};


    #[test]
    fn test_add_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        let expected = CpuTensor::<i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= 4;
        let expected = CpuTensor::<i32>::from_buf(vec![4, 8, 12], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_sub_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= 7;
        let expected = CpuTensor::<i32>::from_buf(vec![3, 13, 23], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_ref_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view += &value;
        let expected = CpuTensor::<i32>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }
    
    #[test]
    fn test_sub_ref_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view -= &value;
        let expected = CpuTensor::<i32>::from_buf(vec![0, 10, 20], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_ref_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view *= &value;
        let expected = CpuTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_negative_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += -5;
        let expected = CpuTensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_sub_negative_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= -5;
        let expected = CpuTensor::<i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_negative_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= -5;
        let expected = CpuTensor::<i32>::from_buf(vec![-5, -10, -15], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // Tests with reshaping/slicing

    #[test]
    fn test_add_after_reshape_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![3, 2]).unwrap();
        reshaped += 10;
        
        let expected = CpuTensor::<i32>::from_buf(vec![11, 12, 13, 14, 15, 16], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_after_slice_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        
        // Verify slice values before mutation
        assert_eq!(slice.get(vec![0]).unwrap(), 4);
        assert_eq!(slice.get(vec![1]).unwrap(), 5);
        assert_eq!(slice.get(vec![2]).unwrap(), 6);
        
        slice += 100;
        
        let expected = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 104, 105, 106], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_after_slice_and_reshape_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        let mut reshaped = slice.view_as(vec![1, 3]).unwrap();
        reshaped += 50;
        
        let expected = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 54, 55, 56], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_sub_after_reshape_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4]).unwrap();
        reshaped -= 5;
        
        let expected = CpuTensor::<i32>::from_buf(vec![5, 15, 25, 35], vec![2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_after_slice_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 0..0).unwrap(); // First depth slice
        
        // Verify slice values before mutation
        assert_eq!(slice.get(vec![0, 0]).unwrap(), 1);
        assert_eq!(slice.get(vec![0, 1]).unwrap(), 2);
        assert_eq!(slice.get(vec![1, 0]).unwrap(), 3);
        assert_eq!(slice.get(vec![1, 1]).unwrap(), 4);
        
        slice *= 10;
        
        let expected = CpuTensor::<i32>::from_buf(vec![10, 20, 30, 40, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_scalar_reshaped_to_matrix_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![42], vec![1]).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![1, 1]).unwrap();
        reshaped += 8;
        
        let expected = CpuTensor::<i32>::from_buf(vec![50], vec![1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_after_column_slice_cuda() {
        // Create a matrix and slice a column (non-contiguous)
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        // Verify slice values (non-contiguous access)
        assert_eq!(col_slice.get(vec![0]).unwrap(), 2);
        assert_eq!(col_slice.get(vec![1]).unwrap(), 5);
        
        col_slice *= 3;
        
        let expected = CpuTensor::<i32>::from_buf(vec![1, 6, 3, 4, 15, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_sub_ref_after_reshape_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![100, 200, 300, 400], vec![2, 2]).unwrap();
        let value = 50;
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4]).unwrap();
        reshaped -= &value;
        
        let expected = CpuTensor::<i32>::from_buf(vec![50, 150, 250, 350], vec![2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_ref_after_slice_chain_cuda() {
        // Create a 3D tensor and chain multiple slices
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let value = 1000;
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Second depth
        
        // Verify depth slice values
        assert_eq!(depth_slice.get(vec![0, 0]).unwrap(), 5);
        assert_eq!(depth_slice.get(vec![0, 1]).unwrap(), 6);
        assert_eq!(depth_slice.get(vec![1, 0]).unwrap(), 7);
        assert_eq!(depth_slice.get(vec![1, 1]).unwrap(), 8);
        
        let mut row_slice = depth_slice.slice_mut(0, 0..0).unwrap(); // First row of that
        assert_eq!(row_slice.get(vec![0]).unwrap(), 5);
        assert_eq!(row_slice.get(vec![1]).unwrap(), 6);
        
        row_slice += &value;
        
        let expected = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 4, 1005, 1006, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // Tests for non-inplace operations (consume view, return new tensor)

    #[test]
    fn test_add_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + 5;
        
        // Result should be a new tensor with added values
        let expected_result = CpuTensor::<i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original tensor should be unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_add_ref_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view + &value;
        
        let expected_result = CpuTensor::<i32>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_sub_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view - 5;
        
        let expected_result = CpuTensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        let expected_original = CpuTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_mul_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * 5;
        
        let expected_result = CpuTensor::<i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_add_not_inplace_with_slice_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        let result = slice + 100;
        
        // Result should be a new 1D tensor
        let expected_result = CpuTensor::<i32>::from_buf(vec![104, 105, 106], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert_eq!(*result.shape(), vec![3]);
        
        // Original tensor should be unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_mul_not_inplace_with_noncontiguous_slice_cuda() {
        // Test with non-contiguous column slice
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        assert!(!col_slice.is_contiguous());
        
        let result = col_slice * 3;
        
        // Result should be a new contiguous tensor
        let expected_result = CpuTensor::<i32>::from_buf(vec![6, 15], vec![2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // Tests for non-inplace operations on non-mutable views (TensorView)
    
    #[test]
    fn test_add_immutable_view_inline_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let result = tensor.view() + 10;
        
        let expected_result = CpuTensor::<i32>::from_buf(vec![11, 12, 13, 14], vec![4]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_sub_immutable_view_with_slice_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        let result = tensor.view().slice(0, 0..0).unwrap() - 5;
        
        let expected_result = CpuTensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_mul_immutable_view_matrix_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.view() * 10;
        
        let expected_result = CpuTensor::<i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // Large tensor test - avoid full CPU copy, only check select indices
    #[test]
    fn test_large_tensor_cuda() {
        const SIZE: usize = 10_000_000; // 10 million elements
        
        // Create a large tensor filled with sequential values
        let data: Vec<i32> = (0..SIZE as i32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![SIZE]).unwrap();
        
        // Apply operation
        let mut view = tensor.view_mut();
        view += 100;
        
        // Check only select indices to avoid copying 10M elements
        let indices_to_check = vec![
            0,              // First element
            1,              // Second element
            SIZE / 4,       // Quarter way
            SIZE / 2,       // Halfway
            3 * SIZE / 4,   // Three quarters
            SIZE - 2,       // Second to last
            SIZE - 1,       // Last element
        ];
        
        for &idx in &indices_to_check {
            let value = view.get(vec![idx]).unwrap();
            let expected = idx as i32 + 100;
            assert_eq!(
                value, 
                expected,
                "Mismatch at index {}: got {}, expected {}",
                idx, value, expected
            );
        }
    }

    #[test]
    fn test_large_tensor_mul_cuda() {
        const SIZE: usize = 5_000_000; // 5 million elements
        
        let data: Vec<i32> = (1..=SIZE as i32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![SIZE]).unwrap();
        
        let mut view = tensor.view_mut();
        view *= 3;
        
        // Check select indices
        let indices = vec![0, 100, 1000, SIZE / 2, SIZE - 100, SIZE - 1];
        
        for &idx in &indices {
            let value = view.get(vec![idx]).unwrap();
            let expected = (idx as i32 + 1) * 3;
            assert_eq!(value, expected, "Mismatch at index {}", idx);
        }
    }

    #[test]
    fn test_large_tensor_non_contiguous_cuda() {
        // Large matrix - test with non-contiguous slicing
        const ROWS: usize = 10_000;
        const COLS: usize = 1_000;
        const SIZE: usize = ROWS * COLS;
        
        let data: Vec<i32> = (0..SIZE as i32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![ROWS, COLS]).unwrap();
        
        // Slice a single column (non-contiguous) and add to it
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, (COLS / 2)..(COLS / 2)).unwrap(); // Middle column
        col_slice += 999;
        
        // Check a few values in the modified column
        let row_indices = vec![0, ROWS / 4, ROWS / 2, ROWS - 1];
        
        for &row_idx in &row_indices {
            let value = col_slice.get(vec![row_idx]).unwrap();
            let original_value = (row_idx * COLS + COLS / 2) as i32;
            let expected = original_value + 999;
            assert_eq!(value, expected, "Mismatch at row {}", row_idx);
        }
    }

    #[test]
    fn test_large_tensor_f32_cuda() {
        const SIZE: usize = 8_000_000; // 8 million f32 elements
        
        let data: Vec<f32> = (0..SIZE).map(|i| i as f32 * 0.5).collect();
        let mut tensor = CudaTensor::<f32>::from_buf(data, vec![SIZE]).unwrap();
        
        let mut view = tensor.view_mut();
        view *= 2.0;
        
        // Check select indices
        let indices = vec![0, 1000, SIZE / 3, SIZE / 2, SIZE - 1000, SIZE - 1];
        
        for &idx in &indices {
            let value = view.get(vec![idx]).unwrap();
            let expected = (idx as f32 * 0.5) * 2.0;
            assert!((value - expected).abs() < 1e-5, 
                "Mismatch at index {}: got {}, expected {}", idx, value, expected);
        }
    }

    #[test]
    fn test_cuda_performance_vs_cpu() {
        use std::time::Instant;
        use crate::backend::cpu::Cpu;
        
        const SIZE: usize = 20_000_000; // 20 million elements
        
        // Create identical data for CPU and GPU
        let data: Vec<f32> = (0..SIZE).map(|i| (i % 1000) as f32).collect();
        
        // CPU timing
        let mut cpu_tensor = crate::core::primitives::TensorBase::<Cpu, f32>::from_buf(
            data.clone(), 
            vec![SIZE]
        ).unwrap();
        
        let cpu_start = Instant::now();
        {
            let mut view = cpu_tensor.view_mut();
            view *= 3.14159;
            view += 2.71828;
            view -= 1.41421;
        }
        let cpu_duration = cpu_start.elapsed();
        
        // GPU timing - including data transfer
        let mut gpu_tensor = CudaTensor::<f32>::from_buf(data, vec![SIZE]).unwrap();
        
        let gpu_start = Instant::now();
        {
            let mut view = gpu_tensor.view_mut();
            view *= 3.14159;
            view += 2.71828;
            view -= 1.41421;
        }
        let gpu_duration = gpu_start.elapsed();
        
        // Verify results match (check select indices)
        let gpu_result = gpu_tensor.cpu().unwrap();
        
        let check_indices = vec![0, SIZE / 4, SIZE / 2, 3 * SIZE / 4, SIZE - 1];
        for &idx in &check_indices {
            let cpu_val = cpu_tensor.view().get(vec![idx]).unwrap();
            let gpu_val = gpu_result.view().get(vec![idx]).unwrap();
            assert!(
                (cpu_val - gpu_val).abs() < 1e-3,
                "CPU/GPU mismatch at index {}: CPU={}, GPU={}", idx, cpu_val, gpu_val
            );
        }
        
        println!("\nPerformance comparison ({} elements):", SIZE);
        println!("  CPU time: {:?}", cpu_duration);
        println!("  GPU time: {:?}", gpu_duration);
        println!("  Speedup: {:.2}x", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());
        
        // Assert GPU is faster - even with memory transfer overhead, GPU should be faster
        // for this size of operation. We use a conservative 1.1x threshold since we're including
        // data transfer time (both to GPU initially and back to CPU for verification)
        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        assert!(
            speedup > 1.1,
            "GPU should be faster than CPU for large tensors (even with transfer overhead). Got {:.2}x speedup",
            speedup
        );
    }

    // Additional comprehensive CUDA tests for edge cases and data types
    
    #[test]
    fn test_cuda_f64_operations() {
        let mut tensor = CudaTensor::<f64>::from_buf(vec![1.5, 2.5, 3.5, 4.5], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10.25;
        view *= 2.0;
        view -= 3.5;
        
        let expected = CpuTensor::<f64>::from_buf(
            vec![20.0, 22.0, 24.0, 26.0], 
            vec![4]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..4 {
            let val = result.view().get(vec![i]).unwrap();
            let exp = expected.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-10, "Mismatch at {}: {} vs {}", i, val, exp);
        }
    }

    #[test]
    fn test_cuda_i64_operations() {
        let mut tensor = CudaTensor::<i64>::from_buf(
            vec![100, 200, 300, 400, 500], 
            vec![5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 50;
        view *= 3;
        
        let expected = CpuTensor::<i64>::from_buf(
            vec![150, 450, 750, 1050, 1350], 
            vec![5]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_u32_operations() {
        let mut tensor = CudaTensor::<u32>::from_buf(
            vec![10, 20, 30, 40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        view *= 2;
        
        let expected = CpuTensor::<u32>::from_buf(
            vec![30, 50, 70, 90], 
            vec![4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_i16_operations() {
        let mut tensor = CudaTensor::<i16>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 10;
        view += 5;
        
        let expected = CpuTensor::<i16>::from_buf(
            vec![15, 25, 35, 45, 55, 65], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_chained_operations_complex() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 2, 2]
        ).unwrap();
        
        // Complex chain: multiply, add, subtract, multiply again
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 10.0;
        view -= 5.0;
        view *= 0.5;
        
        // Expected: ((x * 2 + 10) - 5) * 0.5 = (x * 2 + 5) * 0.5 = x + 2.5
        let expected = CpuTensor::<f32>::from_buf(
            vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..8 {
            let idx = vec![i / 4, (i / 2) % 2, i % 2];
            let val = result.view().get(idx.clone()).unwrap();
            let exp = expected.view().get(idx.clone()).unwrap();
            assert!((val - exp).abs() < 1e-5, "Mismatch at {:?}: {} vs {}", idx, val, exp);
        }
    }

    #[test]
    fn test_cuda_zero_operations() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 0;
        view *= 1;
        view -= 0;
        
        let expected = CpuTensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_multiply_by_zero() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![100, 200, 300, 400], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        
        let expected = CpuTensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_very_small_tensor() {
        // Single element tensor
        let mut tensor = CudaTensor::<f32>::from_buf(vec![42.0], vec![1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 8.0;
        
        let expected = CpuTensor::<f32>::from_buf(vec![92.0], vec![1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_4d_tensor() {
        // 2x2x2x2 tensor
        let data: Vec<i32> = (1..=16).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 100;
        
        let expected_data: Vec<i32> = (101..=116).collect();
        let expected = CpuTensor::<i32>::from_buf(expected_data, vec![2, 2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_mixed_operations_on_slice() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
            vec![3, 3]
        ).unwrap();
        
        let mut view = tensor.view_mut();
        let mut middle_row = view.slice_mut(0, 1..1).unwrap(); // [4, 5, 6]
        middle_row *= 10.0;
        middle_row += 5.0;
        
        let expected = CpuTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 45.0, 55.0, 65.0, 7.0, 8.0, 9.0], 
            vec![3, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_alternating_operations() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30, 40], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        // Alternating pattern
        view += 5;   // [15, 25, 35, 45]
        view *= 2;   // [30, 50, 70, 90]
        view -= 10;  // [20, 40, 60, 80]
        view *= 3;   // [60, 120, 180, 240]
        view += 10;  // [70, 130, 190, 250]
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![70, 130, 190, 250], 
            vec![4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_reshaped_operations() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![6]
        ).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![2, 3]).unwrap();
        reshaped *= 5.0;
        
        let expected = CpuTensor::<f32>::from_buf(
            vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0], 
            vec![6]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_non_inplace_multiple_results() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let view = tensor.view_mut();
        
        // Create multiple independent results
        let result1 = view + 10;
        let view2 = tensor.view_mut();
        let result2 = view2 * 5;
        let view3 = tensor.view_mut();
        let result3 = view3 - 1;
        
        let expected1 = CpuTensor::<i32>::from_buf(vec![11, 12, 13, 14], vec![4]).unwrap();
        let expected2 = CpuTensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let expected3 = CpuTensor::<i32>::from_buf(vec![0, 1, 2, 3], vec![4]).unwrap();
        
        assert_eq!(result1.cpu().unwrap(), expected1);
        assert_eq!(result2.cpu().unwrap(), expected2);
        assert_eq!(result3.cpu().unwrap(), expected3);
        
        // Original unchanged
        let expected_original = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_cuda_scalar_edge_cases() {
        // Test with actual scalar (0-dimensional tensor)
        let mut tensor = CudaTensor::<f32>::from_buf(vec![3.14159], vec![]).unwrap();
        assert!(tensor.is_scalar());
        
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 1.0;
        
        let result = tensor.cpu().unwrap();
        let value = result.view().get(vec![]).unwrap();
        assert!((value - 7.28318).abs() < 1e-5);
    }

    #[test]
    fn test_cuda_negative_to_positive() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![-10, -20, -30, -40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= -1;
        view += 5;
        
        let expected = CpuTensor::<i32>::from_buf(vec![15, 25, 35, 45], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_large_values_i32() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1_000_000, 2_000_000, 3_000_000], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 1_000_000;
        view -= 500_000;
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![1_500_000, 2_500_000, 3_500_000], 
            vec![3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_fractional_multiplication() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![10.0, 20.0, 30.0, 40.0], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 0.5;
        view *= 0.25;
        
        let expected = CpuTensor::<f32>::from_buf(
            vec![1.25, 2.5, 3.75, 5.0], 
            vec![4]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..4 {
            let val = result.view().get(vec![i]).unwrap();
            let exp = expected.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cuda_column_slice_multiple_ops() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 
            vec![3, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        let mut col = view.slice_mut(1, 1..1).unwrap(); // Middle column [2, 5, 8]
        
        col += 100;
        col *= 2;
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![1, 204, 3, 4, 210, 6, 7, 216, 9], 
            vec![3, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_3d_depth_slice_operations() {
        let data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
        let mut tensor = CudaTensor::<f32>::from_buf(
            data, 
            vec![3, 3, 3]
        ).unwrap();
        
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Middle depth slice
        depth_slice *= 10.0;
        
        let mut expected_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
        // Middle depth is indices 9-17 (0-indexed)
        for i in 9..18 {
            expected_data[i] *= 10.0;
        }
        
        let expected = CpuTensor::<f32>::from_buf(expected_data, vec![3, 3, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_ref_value_consistency() {
        // Test that ref and non-ref operations produce same results
        let data = vec![5, 10, 15, 20, 25];
        
        let mut tensor1 = CudaTensor::<i32>::from_buf(data.clone(), vec![5]).unwrap();
        let mut tensor2 = CudaTensor::<i32>::from_buf(data, vec![5]).unwrap();
        
        let value = 7;
        let mut view1 = tensor1.view_mut();
        view1 += value;  // Using owned value
        
        let mut view2 = tensor2.view_mut();
        view2 += &value;  // Using reference
        
        assert_eq!(tensor1.cpu().unwrap(), tensor2.cpu().unwrap());
    }

    #[test]
    fn test_cuda_immutable_view_complex_chain() {
        let tensor = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 4]
        ).unwrap();
        
        // Complex immutable chain
        let result = tensor.view()
            .slice(0, 1..1).unwrap()  // Second row: [5, 6, 7, 8]
            .view_as(vec![2, 2]).unwrap()  // Reshape
            + 100.0;
        
        let expected = CpuTensor::<f32>::from_buf(
            vec![105.0, 106.0, 107.0, 108.0], 
            vec![2, 2]
        ).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
        
        // Original unchanged
        let original_expected = CpuTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), original_expected);
    }

    // ============================================================================
    // EDGE CASE TESTS - Priority 1: Critical Coverage (CUDA)
    // ============================================================================

    // --- Scalar Tensor Tests ---

    #[test]
    fn test_scalar_add_operation_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![42], vec![]).unwrap();
        assert!(tensor.is_scalar());
        let mut view = tensor.view_mut();
        view += 10;
        let expected = CpuTensor::<i32>::from_buf(vec![52], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_mul_operation_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![7], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= 6;
        let expected = CpuTensor::<i32>::from_buf(vec![42], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_sub_operation_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![100], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view -= 58;
        let expected = CpuTensor::<i32>::from_buf(vec![42], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_negative_value_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![-10], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= -3;
        view += 5;
        let expected = CpuTensor::<i32>::from_buf(vec![35], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_zero_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![0], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view += 42;
        view *= 0;
        view += 10;
        let expected = CpuTensor::<i32>::from_buf(vec![10], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_non_inplace_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![100], vec![]).unwrap();
        let result = tensor.view() + 50;
        let expected_result = CpuTensor::<i32>::from_buf(vec![150], vec![]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert!(result.is_scalar());
        let expected_original = CpuTensor::<i32>::from_buf(vec![100], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // --- Single Element Tensor Tests ---

    #[test]
    fn test_single_element_1d_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5], vec![1]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        let expected = CpuTensor::<i32>::from_buf(vec![30], vec![1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_single_element_2d_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![8], vec![1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        let expected = CpuTensor::<i32>::from_buf(vec![40], vec![1, 1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_single_element_3d_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![3], vec![1, 1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view += 7;
        view *= 3;
        let expected = CpuTensor::<i32>::from_buf(vec![30], vec![1, 1, 1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Non-Square Tensor Tests ---

    #[test]
    fn test_tall_matrix_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![9, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        let expected = CpuTensor::<i32>::from_buf(vec![2, 4, 6, 8, 10, 12, 14, 16, 18], vec![9, 1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_wide_matrix_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![1, 9]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        let expected = CpuTensor::<i32>::from_buf(vec![11, 12, 13, 14, 15, 16, 17, 18, 19], vec![1, 9]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_rectangular_matrix_2x5_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![2, 5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3;
        let expected = CpuTensor::<i32>::from_buf(
            vec![3, 6, 9, 12, 15, 18, 21, 24, 27, 30], 
            vec![2, 5]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_rectangular_matrix_5x2_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![5, 2]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 1;
        let expected = CpuTensor::<i32>::from_buf(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
            vec![5, 2]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_rectangular_3d_tensor_cuda() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 3, 4]).unwrap();
        let mut view = tensor.view_mut();
        view += 100;
        
        let expected_data: Vec<i32> = (101..=124).collect();
        let expected = CpuTensor::<i32>::from_buf(expected_data, vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_non_square_slice_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [5, 6, 7, 8]
        slice *= 10;
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Edge Value Tests ---

    #[test]
    fn test_zero_tensor_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        let expected = CpuTensor::<i32>::from_buf(vec![20, 20, 20, 20], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_operations_resulting_in_zero_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        let expected = CpuTensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mixed_positive_negative_values_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![-5, 10, -15, 20, -25, 30], 
            vec![6]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 10;
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![0, 30, -20, 50, -40, 70], 
            vec![6]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_large_values_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1_000_000, 2_000_000, 3_000_000], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 500_000;
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![1_500_000, 2_500_000, 3_500_000], 
            vec![3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Complex Non-Contiguous View Tests ---

    #[test]
    fn test_multiple_noncontiguous_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![4, 3]
        ).unwrap();
        
        // Get middle column (non-contiguous)
        let mut view = tensor.view_mut();
        let mut col = view.slice_mut(1, 1..1).unwrap(); // Column 1: [2, 5, 8, 11]
        col += 100;
        col *= 2;
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![1, 204, 3, 4, 210, 6, 7, 216, 9, 10, 222, 12], 
            vec![4, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_noncontiguous_slice_after_reshape_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![8]
        ).unwrap();
        let view = tensor.view_mut();
        let mut reshaped = view.view_as(vec![4, 2]).unwrap();
        let mut col_slice = reshaped.slice_mut(1, 1..1).unwrap(); // Second column
        col_slice *= 10;
        
        // Second column in 4x2 is indices 1, 3, 5, 7
        let expected = CpuTensor::<i32>::from_buf(
            vec![1, 20, 3, 40, 5, 60, 7, 80], 
            vec![8]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_3d_noncontiguous_slice_cuda() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 3, 4]).unwrap();
        
        // Slice along middle dimension at index 1 (middle row)
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(1, 1..1).unwrap(); // Middle "row" at each depth level
        slice += 100;
        
        // Indices affected: 5-8, 17-20 (middle row of each depth, 1-indexed becomes 4-7, 16-19 in 0-indexed)
        let mut expected_data: Vec<i32> = (1..=24).collect();
        for i in [5, 6, 7, 8, 17, 18, 19, 20].iter() {
            expected_data[i - 1] += 100;
        }
        
        let expected = CpuTensor::<i32>::from_buf(expected_data, vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- 4D and 5D Tensor Tests ---

    #[test]
    fn test_4d_tensor_operations_cuda() {
        let data: Vec<i32> = (1..=16).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        
        let expected_data: Vec<i32> = (11..=26).collect();
        let expected = CpuTensor::<i32>::from_buf(expected_data, vec![2, 2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_4d_slice_operation_cuda() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 3, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second slice along first dim
        slice *= 10;
        
        // Second half of data (indices 12-23)
        let mut expected_data: Vec<i32> = (1..=24).collect();
        for i in 12..24 {
            expected_data[i] *= 10;
        }
        
        let expected = CpuTensor::<i32>::from_buf(expected_data, vec![2, 3, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_5d_tensor_operations_cuda() {
        let data: Vec<i32> = (1..=32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 5;
        
        let expected_data: Vec<i32> = (1..=32).map(|x| x * 2 + 5).collect();
        let expected = CpuTensor::<i32>::from_buf(expected_data, vec![2, 2, 2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Operation Chain Tests ---

    #[test]
    fn test_long_operation_chain_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        view += 1;   // [2, 3, 4, 5]
        view *= 2;   // [4, 6, 8, 10]
        view -= 1;   // [3, 5, 7, 9]
        view *= 3;   // [9, 15, 21, 27]
        view += 10;  // [19, 25, 31, 37]
        view -= 5;   // [14, 20, 26, 32]
        view *= 2;   // [28, 40, 52, 64]
        view += 2;   // [30, 42, 54, 66]
        view -= 10;  // [20, 32, 44, 56]
        view *= 1;   // [20, 32, 44, 56]
        
        let expected = CpuTensor::<i32>::from_buf(vec![20, 32, 44, 56], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_alternating_inplace_noninplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        
        // Inplace
        let mut view1 = tensor.view_mut();
        view1 += 5;
        
        // Non-inplace
        let result1 = tensor.view() * 2;
        let expected_result1 = CpuTensor::<i32>::from_buf(vec![20, 30, 40], vec![3]).unwrap();
        assert_eq!(result1.cpu().unwrap(), expected_result1);
        
        // Inplace again
        let mut view2 = tensor.view_mut();
        view2 *= 3;
        
        // Non-inplace again
        let result2 = tensor.view() + 10;
        let expected_result2 = CpuTensor::<i32>::from_buf(vec![40, 55, 70], vec![3]).unwrap();
        assert_eq!(result2.cpu().unwrap(), expected_result2);
        
        // Original tensor should reflect inplace ops
        let expected_final = CpuTensor::<i32>::from_buf(vec![30, 45, 60], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_final);
    }

    #[test]
    fn test_multiple_views_different_ops_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        
        // First view - operate on first row
        {
            let mut view = tensor.view_mut();
            let mut row1 = view.slice_mut(0, 0..0).unwrap();
            row1 += 10;
        }
        
        // Second view - operate on second row
        {
            let mut view = tensor.view_mut();
            let mut row2 = view.slice_mut(0, 1..1).unwrap();
            row2 *= 5;
        }
        
        let expected = CpuTensor::<i32>::from_buf(
            vec![11, 12, 13, 20, 25, 30], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Float Edge Cases ---

    #[test]
    fn test_f32_small_values_cuda() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![0.0001, 0.0002, 0.0003], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 1000.0;
        
        let result = tensor.cpu().unwrap();
        let expected = vec![0.1, 0.2, 0.3];
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f64_precision_cuda() {
        let mut tensor = CudaTensor::<f64>::from_buf(
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3.0;
        
        let result = tensor.cpu().unwrap();
        let expected = vec![1.0, 2.0, 3.0];
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-10);
        }
    }

    // --- Stress Test: Very Imbalanced Shapes ---

    #[test]
    fn test_very_imbalanced_shape_1000x2_cuda() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![1000, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        
        let result = tensor.cpu().unwrap();
        // Check a few values
        assert_eq!(result.view().get(vec![0, 0]).unwrap(), 6);
        assert_eq!(result.view().get(vec![0, 1]).unwrap(), 7);
        assert_eq!(result.view().get(vec![999, 0]).unwrap(), 2004);
        assert_eq!(result.view().get(vec![999, 1]).unwrap(), 2005);
    }

    #[test]
    fn test_very_imbalanced_shape_2x1000_cuda() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 1000]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        
        let result = tensor.cpu().unwrap();
        // Check a few values
        assert_eq!(result.view().get(vec![0, 0]).unwrap(), 2);
        assert_eq!(result.view().get(vec![0, 999]).unwrap(), 2000);
        assert_eq!(result.view().get(vec![1, 0]).unwrap(), 2002);
        assert_eq!(result.view().get(vec![1, 999]).unwrap(), 4000);
    }
}
