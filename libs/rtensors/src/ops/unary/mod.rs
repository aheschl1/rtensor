use crate::{backend::Backend, core::{primitives::TensorBase, primops::{Exp, InvExp}, tensor::{AsTensor, AsViewMut}, value::TensorValue}};

pub mod neg;
pub mod relu;
mod sigmoid;
mod tanh;
mod abs;

pub use neg::Negate;
pub use relu::Relu;
pub use sigmoid::Sigmoid;
pub use tanh::Tanh;
pub use abs::Abs;

pub trait InplaceUnaryOp<T: TensorValue, B: Backend> {
    fn apply_relu(
        &mut self
    );
    fn apply_sigmoid(
        &mut self
    )
    where 
        T: Exp + InvExp
    ;
    fn apply_tanh(
        &mut self
    )
    where 
        T: Exp + InvExp;
    fn apply_abs(&mut self);
}

pub trait UnaryOp<T: TensorValue, B: Backend> {
    fn relu(
        &self,
    ) -> TensorBase<T, B>;
    fn sigmoid(
        &self,
    ) -> TensorBase<T, B>
    where 
        T: Exp + InvExp;
    fn tanh(
        &self,
    ) -> TensorBase<T, B>
    where 
        T: Exp + InvExp;
    fn abs(&self) -> TensorBase<T, B>;
}


impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> InplaceUnaryOp<T, B> for V {
    fn apply_relu(
        &mut self
    ) {
        self.relu_inplace();
    }

    fn apply_sigmoid(
        &mut self
    )
    where 
        T: Exp + InvExp
    {
        self.sigmoid_inplace();
    }

    fn apply_tanh(
        &mut self
    )
    where 
        T: Exp + InvExp
    {
        self.tanh_inplace();
    }
    fn apply_abs(&mut self) {
        self.abs_inplace();
    }
}


impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> UnaryOp<T, B> for V {
    fn relu(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_relu();
        result
    }

    fn sigmoid(
        &self,
    ) -> TensorBase<T, B>
    where 
        T: Exp + InvExp
    {
        let mut result = self.owned();
        result.apply_sigmoid();
        result
    }

    fn tanh(
        &self,
    ) -> TensorBase<T, B>
    where 
        T: Exp + InvExp
    {
        let mut result = self.owned();
        result.apply_tanh();
        result
    }
    fn abs(&self) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_abs();
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, ops::unary::{Negate, Relu, Sigmoid, Tanh}, testing::{unary_assert_1d_strided, unary_assert_contiguous, unary_assert_nd_strided}};

    #[test]
    fn test_unary_negate_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], std::ops::Neg::neg, |f| f.neg_inplace());
        
       
    }

    #[test]
    fn test_unary_negate_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], std::ops::Neg::neg, |f| f.neg_inplace());
    }

    #[test]
    fn test_unary_negate_nd_strided() {
         unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], std::ops::Neg::neg, |f| f.neg_inplace());
    }

     #[test]
    fn test_unary_relu_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([-1.0, 1.0], |f| f.max(0.0), Relu::relu_inplace);
    }

    #[test]
    fn test_unary_relu_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([-1.0, 1.0, -1.0], |f| f.max(0.0), |f| f.relu_inplace());
    }

    #[test]
    fn test_unary_relu_nd_strided() {
         unary_assert_nd_strided::<f64, _, _, Cpu>([
            -1.0, 1.0, 0.0, 2.0,
            1.0, 2.3, -0.3, 0.4,
            0.0, -0.3, 0.4, 0.5,
            -0.2, 0.1, 0.2, -0.5
        ], |f| f.max(0.0), |f| f.relu_inplace());
    }

  
    #[test]
    fn test_unary_sigmoid_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], |f| 1. / (1. + (-f).exp()), |f| f.sigmoid_inplace());
    }

    #[test]
    fn test_unary_sigmoid_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], |f| 1. / (1. + (-f).exp()), |f| f.sigmoid_inplace());
    }

    #[test]
    fn test_unary_sigmoid_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], |f| 1. / (1. + (-f).exp()), |f| f.sigmoid_inplace());
    }


    #[test]
    fn test_unary_tanh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()), |f| f.tanh_inplace());
    }

    #[test]
    fn test_unary_tanh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()), |f| f.tanh_inplace());
    }

    #[test]
    fn test_unary_tanh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()), |f| f.tanh_inplace());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::{backend::cuda::Cuda, core::{Tensor, primitives::{CudaTensor, TensorBase}, tensor::{AsTensor, TensorAccess, TensorAccessMut}}, ops::unary::{Abs, Negate, Relu, Sigmoid, Tanh}, testing::{test_with_contiguous_2_elem_tensor, unary_assert_1d_strided, unary_assert_contiguous, unary_assert_nd_strided}};




    #[test]
    fn test_unary_negate_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], std::ops::Neg::neg, |f| f.neg_inplace());
    }

    #[test]
    fn test_unary_negate_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], std::ops::Neg::neg, |f| f.neg_inplace());
    }

    #[test]
    fn test_unary_negate_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], std::ops::Neg::neg, |f| f.neg_inplace());
    }


    
    #[test]
    fn test_unary_abs_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.0, 1.3], |f| f.abs(), |f| f.abs_inplace());
    }

    #[test]
    fn test_unary_abs_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, -1.0, 3.0], |f| f.abs(), |f| f.abs_inplace());
    }

    #[test]
    fn test_unary_abs_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([-1.0; 16], |f| f.abs(), |f| f.abs_inplace());
    }


    #[test]
    fn test_unary_relu_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.0, 1.0], |f| f.max(0.0), Relu::relu_inplace);
    }

    #[test]
    fn test_unary_relu_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([-1.0, 1.0, -1.0], |f| f.max(0.0), |f| f.relu_inplace());
    }

    #[test]
    fn test_unary_relu_nd_strided_cuda() {
         unary_assert_nd_strided::<f64, _, _, Cuda>([
            -1.0, 1.0, 0.0, 2.0,
            1.0, 2.3, -0.3, 0.4,
            0.0, -0.3, 0.4, 0.5,
            -0.2, 0.1, 0.2, -0.5
        ], |f| f.max(0.0), |f| f.relu_inplace());
    }

    #[test]
    fn test_unary_sigmoid_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], |f| 1. / (1. + (-f).exp()), |f| f.sigmoid_inplace());
    }

    #[test]
    fn test_unary_sigmoid_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], |f| 1. / (1. + (-f).exp()), |f| f.sigmoid_inplace());
    }

    #[test]
    fn test_unary_sigmoid_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], |f| 1. / (1. + (-f).exp()), |f| f.sigmoid_inplace());
    }

    #[test]
    fn test_unary_tanh_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()), |f| f.tanh_inplace());
    }

    #[test]
    fn test_unary_tanh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()), |f| f.tanh_inplace());
    }

    #[test]
    fn test_unary_tanh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()), |f| f.tanh_inplace());
    }

}

#[cfg(all(test, feature = "remote"))]
mod remote_tests {
    use std::{sync::OnceLock, thread};

    use crate::{backend::{remote::{client::RemoteBackend, get_backend_default, server::RemoteServer}, Backend}, core::{primitives::{RemoteTensor, TensorBase}, tensor::TensorError, value::TensorValue, MetaTensor, Shape}};

    // Lazy static backend shared across all tests
    static BACKEND: OnceLock<RemoteBackend> = OnceLock::new();
    
    fn get_backend() -> RemoteBackend {

        BACKEND.get_or_init(|| {
            // Start the server
            let mut server = RemoteServer::new("127.0.0.1".parse().unwrap(), 7878);
            thread::spawn(move || {
                let _ = server.serve();
            });
            thread::sleep(std::time::Duration::from_millis(10));

            // Create and connect the backend
            let backend = get_backend_default().unwrap();
            
            backend
        }).clone()
    }
    
    fn make_remote_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Result<RemoteTensor<T>, TensorError> {
        let shape: Shape = shape.into();
        let buf_len = buf.len();
        let expected_len: usize = shape.iter().product();
        
        if buf_len != expected_len {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                expected_len,
                buf_len
            )));
        }
        
        let backend = get_backend();
        let buffer = backend.alloc_from_slice(buf.into())?;
        let stride = crate::core::shape_to_stride(&shape);
        
        // Clone the backend for this tensor
        let tensor_backend = backend.clone();
        drop(backend); // Release the lock
        
        Ok(TensorBase::from_parts(tensor_backend, buffer, MetaTensor::new(shape, stride, 0)))
    }

    #[test]
    fn test_remote_negate() {
        let tensor: TensorBase<f32, RemoteBackend> = make_remote_tensor(vec![1.0f32, -2.0, 3.0], (3,)).unwrap();
        let negated = -tensor;

        let expected = make_remote_tensor(vec![-1.0f32, 2.0, -3.0], (3,)).unwrap();
        assert_eq!(negated.cpu().unwrap(), expected.cpu().unwrap());
    }

}