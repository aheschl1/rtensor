#![allow(non_snake_case)]

use std::sync::Arc;

use cudarc::{driver::{CudaContext, LaunchConfig, PushKernelArg}, nvrtc::Ptx};

use crate::core::tensor::TensorError;

// ignore styling for generated code
#[allow(non_camel_case_types)]
mod bindings{
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

const ADD_KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/add.ptx"));

pub struct CudaBackend {
    context: Arc<CudaContext>,
}

impl CudaBackend {
    /// Create a new CUDA backend instance
    /// 
    /// # Errors
    /// Returns an error if CUDA initialization fails or no CUDA device is found
    pub fn new() -> Result<Self, TensorError> {
        let context = CudaContext::new(0)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(Self { context })
    }
    
    pub fn context(&self) -> &CudaContext {
        &self.context
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new().expect("Failed to initialize CUDA backend")
    }
}


pub fn example_add() -> Result<Vec<f32>, TensorError> {
    // Initialize CUDA backend
    let backend = CudaBackend::new()?;
    let ctx = &backend.context;
    let stream = ctx.default_stream();
    
    // Compile and load PTX module
    let module = ctx.load_module(Ptx::from_src(ADD_KERNEL_PTX))
        .map_err(|e| TensorError::CudaError(e.to_string()))?;
    let kernel = module.load_function("add_kernel")
        .map_err(|e| TensorError::CudaError(e.to_string()))?;
    
    // Prepare host data
    let n = 10000_usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    
    // Copy to GPU using stream
    let dev_a = stream.clone_htod(&a)
        .map_err(|e| TensorError::CudaError(e.to_string()))?;
    let dev_b = stream.clone_htod(&b)
        .map_err(|e| TensorError::CudaError(e.to_string()))?;
    let mut dev_c = stream.alloc_zeros::<f32>(n)
        .map_err(|e| TensorError::CudaError(e.to_string()))?;
    
    // Launch kernel using builder pattern
    let mut launch_args = stream.launch_builder(&kernel);
    launch_args.arg(&dev_a);
    launch_args.arg(&dev_b);
    launch_args.arg(&mut dev_c);
    launch_args.arg(&n);
    
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        launch_args.launch(cfg)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
    }
    
    // Copy result back to host
    let result = stream.clone_dtoh(&dev_c)
        .map_err(|e| TensorError::CudaError(e.to_string()))?;
    
    Ok(result)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::core::{idx::Idx, primitives::CudaTensor, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, value::TensorValue, Shape};
    
    fn make_cuda_tensor<T: TensorValue + cudarc::driver::DeviceRepr>(buf: Vec<T>, shape: Shape) -> CudaTensor<T> {
        CudaTensor::from_buf(buf, shape).unwrap()
    }

    fn index_tensor<'a, T: TensorValue + PartialEq + std::fmt::Debug + cudarc::driver::DeviceRepr>(
        index: Idx<'a>, 
        tensor: &'a impl TensorAccess<T, crate::backend::cuda::CudaBackend>
    ) -> Result<T, TensorError> {
        let r: Result<T, TensorError> = tensor.get(&index);
        let a = match r.as_ref() {
            Ok(v) => Some(*v),
            Err(_) => None,
        };
        let b = match &index {
            Idx::Item => tensor.item().ok(),
            _ => tensor.get(&index).ok(),
        };
        assert_eq!(a, b);
        r
    }
    
    #[test]
    fn test_cuda_backend_init() {
        let backend = CudaBackend::new();
        assert!(backend.is_ok(), "Failed to initialize CUDA backend");
    }
    
    #[test]
    fn test_cuda_add() {
        let result = example_add();
        assert!(result.is_ok(), "CUDA addition failed");
        
        let result = result.unwrap();
        assert_eq!(result.len(), 10000);
        
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 6.0);
    }

    #[test]
    fn test_cuda_scalar() {
        let tensor = CudaTensor::scalar(42);
        assert_eq!(index_tensor(Idx::Item, &tensor.view()).unwrap(), 42);
        assert!(tensor.meta.is_scalar());
    }

    #[test]
    fn test_cuda_column() {
        let tensor = CudaTensor::column(vec![1, 2, 3]);
        assert_eq!(*tensor.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_cuda_row() {
        let tensor = CudaTensor::row(vec![1, 2, 3]);
        assert_eq!(*tensor.meta.shape(), vec![1, 3]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[0, 2]), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_cuda_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_cuda_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);

        tensor.view_mut().set(&Idx::At(1), 10).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 10);
    }

    #[test]
    fn test_cuda_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_cuda_tensor(buf, shape);

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
    fn test_cuda_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 0]), &tensor.view()).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 1]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1, 0]), &tensor.view()).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1, 1]), &tensor.view()).unwrap(), 9);

        tensor.view_mut().set(&Idx::Coord(&[1, 0, 0]), 67).unwrap();
        assert_eq!(index_tensor(Idx::Coord(&[1, 0, 0]), &tensor.view()).unwrap(), 67);
    }

    #[test]
    fn test_cuda_slice_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.meta.shape(), vec![3]);
        assert_eq!(*slice.meta.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 3);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap();
        assert_eq!(*slice2.meta.shape(), vec![2]);
        assert_eq!(*slice2.meta.stride(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[1]), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
    }

    #[test]
    fn test_cuda_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_cuda_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.meta.shape(), vec![2, 2]);
        assert_eq!(*slice.meta.stride(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice).unwrap(), 5);

        let view = tensor.view();
        let slice_second_depth = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice_second_depth.meta.shape(), vec![2, 2]);
        assert_eq!(*slice_second_depth.meta.stride(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice_second_depth).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice_second_depth).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice_second_depth).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice_second_depth).unwrap(), 9);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap();
        assert_eq!(*slice2.meta.shape(), vec![2, 2]);
        assert_eq!(*slice2.meta.stride(), vec![4, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice2).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice2).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice2).unwrap(), 7);

        let view = tensor.view();
        let slice3 = view.slice(2, 0..0).unwrap();
        assert_eq!(*slice3.meta.shape(), vec![2, 2]);
        assert_eq!(*slice3.meta.stride(), vec![4, 2]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice3).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice3).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice3).unwrap(), 8);
    }

    #[test]
    fn test_cuda_slice_of_slice() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);

        let slice_of_slice = slice.slice(0, 2..2).unwrap();
        assert_eq!(*slice_of_slice.meta.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Coord(&[]), &slice_of_slice).unwrap(), 6);
    }

    #[test]
    fn test_cuda_slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_cuda_tensor(buf, shape);

        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice.meta.shape(), vec![2, 2]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &slice).unwrap(), 9);

        let slice_of_slice = slice.slice(1, 0..0).unwrap();
        assert_eq!(*slice_of_slice.meta.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &slice_of_slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(1), &slice_of_slice).unwrap(), 8);

        let slice_of_slice_of_slice = slice_of_slice.slice(0, 1..1).unwrap();
        assert_eq!(*slice_of_slice_of_slice.meta.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Item, &slice_of_slice_of_slice).unwrap(), 8);
    }

    #[test]
    fn test_cuda_mut_slices() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap();
        
        assert_eq!(*slice.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);
        
        slice.set(&Idx::At(1), 50).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &tensor.view()).unwrap(), 50);
    }

    #[test]
    fn test_cuda_view_as_owned_success() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);
        let reshaped = tensor.view().view_as(vec![3, 2]).unwrap();
        
        assert_eq!(*reshaped.meta.shape(), vec![3, 2]);
        assert_eq!(*reshaped.meta.stride(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &reshaped).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &reshaped).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(&[1, 0]), &reshaped).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(&[1, 1]), &reshaped).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[2, 0]), &reshaped).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[2, 1]), &reshaped).unwrap(), 6);
    }

    #[test]
    fn test_cuda_view_as_owned_error() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);
        assert!(matches!(tensor.view().view_as(vec![4, 2]), Err(TensorError::InvalidShape)));
    }

    #[test]
    fn test_cuda_view_as_slice_success() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);

        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice.meta.shape(), vec![3]);
        let reshaped = slice.view_as(vec![1, 3]).unwrap();
        assert_eq!(*reshaped.meta.shape(), vec![1, 3]);
        assert_eq!(*reshaped.meta.stride(), vec![3, 1]);
        assert_eq!(index_tensor(Idx::Coord(&[0, 0]), &reshaped).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(&[0, 1]), &reshaped).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(&[0, 2]), &reshaped).unwrap(), 6);
    }

    #[test]
    fn test_cuda_view_as_mut_view_modify() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let mut tensor = make_cuda_tensor(buf, shape);
        let mut view_mut = tensor.view_mut();
        view_mut.set(&Idx::Coord(&[1, 0]), 40).unwrap();
        let reshaped = view_mut.view_as(vec![4]).unwrap();
        assert_eq!(*reshaped.meta.shape(), vec![4]);
        assert_eq!(*reshaped.meta.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(2), &reshaped).unwrap(), 40);
    }

    #[test]
    fn test_cuda_view_as_scalar() {
        let tensor = CudaTensor::scalar(99);
        let view1 = tensor.view();
        assert_eq!(*view1.meta.shape(), vec![]);
        let reshaped = view1.view_as(vec![1]).unwrap();
        assert_eq!(*reshaped.meta.shape(), vec![1]);
        assert_eq!(*reshaped.meta.stride(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &reshaped).unwrap(), 99);

        let r2 = reshaped.view_as(vec![1, 1, 1]).unwrap();
        assert_eq!(index_tensor(Idx::Coord(&[0, 0, 0]), &r2).unwrap(), 99);
    }

    #[test]
    fn test_cuda_from_buf_error() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            CudaTensor::from_buf(buf, shape),
            Err(TensorError::InvalidShape)
        ));
    }

    #[test]
    fn test_cuda_get_errors() {
        let tensor = make_cuda_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            tensor.view().get(vec![0, 0, 0]),
            Err(TensorError::WrongDims)
        ));
        assert!(matches!(
            tensor.view().get(vec![2, 0]),
            Err(TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_cuda_slice_errors() {
        let tensor = make_cuda_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            tensor.view().slice(3, 0..0),
            Err(TensorError::InvalidDim)
        ));
        assert!(matches!(
            tensor.view().slice(0, 5..5),
            Err(TensorError::IdxOutOfBounds)
        ));
    }

    #[test]
    fn test_cuda_index_and_index_mut() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_cuda_tensor(buf, shape);

        assert_eq!(tensor.view().get(&Idx::Coord(&[0, 1])).unwrap(), 2);
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 6);

        tensor.view_mut().set(vec![1, 1], 55).unwrap();
        assert_eq!(tensor.view().get(&Idx::Coord(&[1, 1])).unwrap(), 55);
        assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 55);

        let view = tensor.view();
        let view = view.slice(0, 1..1).unwrap();
        assert_eq!(view.get(vec![0]).unwrap(), 4);
        assert_eq!(view.get(vec![1]).unwrap(), 55);
        assert_eq!(view.get(vec![2]).unwrap(), 6);

        let mut mut_view = tensor.view_mut();
        let mut mut_slice = mut_view.slice_mut(0, 0..0).unwrap();
        mut_slice.set(vec![2], 33).unwrap();
        assert_eq!(mut_slice.get(&Idx::Coord(&[2])).unwrap(), 33);
        assert_eq!(mut_slice.get(vec![2]).unwrap(), 33);

        assert_eq!(tensor.view().get(vec![0, 2]).unwrap(), 33);
    }

    #[test]
    #[should_panic]
    fn test_cuda_index_out_of_bounds_panic() {
        let tensor = make_cuda_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_cuda_index_wrong_dims_panic() {
        let tensor = make_cuda_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![0, 0]).unwrap();
    }

    #[test]
    fn rediculously_large_cuda_tensor() {
        let n = 10_000_000_usize;
        let buf: Vec<u8> = vec![1; n];
        let shape = vec![n];
        let tensor = make_cuda_tensor(buf, shape);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(n - 1), &tensor.view()).unwrap(), 1);
    }
}
