use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
#[cfg(feature = "remote")]
use std::net::IpAddr;
use std::sync::Arc;

use crate::core::tensor::AsTensor;

use crate::backend::Backend;
use crate::backend::cpu::Cpu;
use crate::core::value::WeightValue;
use crate::grad::{self, GradNode, NodeKey};
use crate::core::value::TensorValue;
use crate::core::{shape_to_stride, MetaTensor, MetaTensorView, Shape};
use crate::core::tensor::{TensorError, compute_squeezed_parameters};

/// A generic tensor with backend-specific storage.
/// 
/// This is the base type for all tensors, parameterized by element type `T` and backend `B`.
/// Most users will use type aliases like `Tensor<T>` (CPU) or `CudaTensor<T>` (GPU).
#[derive(Debug, PartialEq, Eq)]
pub struct TensorBase<T: TensorValue, B: Backend> {
    pub(crate) backend: B,
    pub(crate) buf: B::Buf<T>,
    pub(crate) meta: MetaTensor,
}

#[derive(Debug, Clone)]
pub struct GradTensor<T: WeightValue, B: Backend> {
    pub(crate) inner: GradTensorRef<T, B>,
    pub(crate) node: NodeKey,
}

impl<T: WeightValue, B: Backend> GradTensor<T, B> {

    #[grad::when_enabled(ctx)]
    pub(crate) fn leaf(
        value: TensorBase<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            value,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        ctx.make_leaf(inner)

    }

    pub(crate) fn input( // a tensor that requires grad but is not a parameter, for example, input to the model
        value: TensorBase<T, B>
    ) -> Self {
        Self::from_op(value, GradNode::None)
    }

    #[inline]
    #[grad::when_enabled(ctx)]
    pub(crate) fn from_op(
        value: TensorBase<T, B>,
        op: GradNode<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            value,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        ctx.attach(inner, op)
    }

    #[inline]
    #[grad::when_enabled(ctx)]
    pub(crate) fn from_op_self_referential(
        value: TensorBase<T, B>,
        op_builder: impl FnOnce(GradTensorRef<T, B>) -> GradNode<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            value,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        let node = op_builder(inner.clone());
        ctx.attach(inner, node)
    } 

    #[grad::when_enabled(ctx)]
    pub(crate) fn is_leaf(&self) -> bool {
        let nodes = ctx.nodes.borrow();
        let node = nodes.get(self.node).expect("Node not found in grad context.");
        node.is_leaf()
    }

    pub(crate) fn copy_tensor(&self) -> TensorBase<T, B> {
        self.borrow().value.contiguous()
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, GradTensorInner<T, B>> {
        self.inner.borrow()
    }

    pub fn get_ref(&self) -> GradTensorRef<T, B> {
        self.inner.clone()
    }
}

pub struct GradTensorInner<T: TensorValue, B: Backend> {
    pub(crate) value: TensorBase<T, B>,
    pub(crate) grad: Option<TensorBase<T, B>>,
}

pub type GradTensorRef<T, B> = Arc<RefCell<GradTensorInner<T, B>>>;

impl<T: TensorValue, B: Backend> Debug for GradTensorInner<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradTensorInner")
            .field("grad", &self.grad)
            .field("value", &self.value)
            .finish()
    }
}

impl<T: WeightValue, B: Backend> Eq for GradTensor<T, B> {}
impl<T: WeightValue, B: Backend> PartialEq for GradTensor<T, B> { fn eq(&self, other: &Self) -> bool { true }}

impl<B: Backend, T: TensorValue> Clone for TensorBase<T, B> {
    fn clone(&self) -> Self {
        let new_backend = self.backend.clone();
        let new_buffer = new_backend.copy(&self.buf).unwrap();
        Self {
            backend: new_backend,
            buf: new_buffer,
            meta: self.meta.clone(),
        }

    }
}

/// An owned CPU tensor stored in row-major order.
/// 
/// # Examples
/// ```ignore
/// let tensor = Tensor::<f32>::zeros((3, 4));
/// let tensor = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], (2, 2)).unwrap();
/// ```
pub type Tensor<T> = TensorBase<T, Cpu>;

#[cfg(feature = "remote")]
use crate::backend::remote::client::RemoteBackend;

#[cfg(feature = "remote")]
pub type RemoteTensor<T> = TensorBase<T, RemoteBackend>;

#[cfg(feature = "cuda")]
/// An owned GPU tensor stored on CUDA device.
pub type CudaTensor<T> = TensorBase<T, crate::backend::cuda::Cuda>;

#[cfg(feature = "cuda")]
impl<T: TensorValue> CudaTensor<T> {
    /// Transfers this tensor from the CUDA device to CPU memory.
    pub fn cpu(&self) -> Result<Tensor<T>, TensorError> {
        let cpu_backend = Cpu;
        let cpu_buffer = self.backend.dump(&self.buf)?;
        let cpu = Tensor::from_parts(cpu_backend, cpu_buffer, self.meta.clone());
        Ok(cpu)
    }
}

#[cfg(feature = "cuda")]
impl<T: TensorValue> Tensor<T> {
    /// Transfers this tensor from CPU to the CUDA device.
    pub fn cuda(&self) -> Result<CudaTensor<T>, TensorError> {
        let cuda_backend = crate::backend::cuda::Cuda::construct(0)?;
        let cuda_buffer = cuda_backend.alloc_from_slice(self.backend.dump(&self.buf)?)?;
        let cuda = CudaTensor::from_parts(cuda_backend, cuda_buffer, self.meta.clone());
        Ok(cuda)
    }
}

#[cfg(feature = "remote")]
impl<T: TensorValue> RemoteTensor<T> {
    /// Transfers this tensor from remote backend to CPU memory.
    pub fn cpu(&self) -> Result<Tensor<T>, TensorError> {
        let cpu_backend = Cpu;
        let cpu_buffer = self.backend.dump(&self.buf)?;
        let cpu = Tensor::from_parts(cpu_backend, cpu_buffer, self.meta.clone());
        Ok(cpu)
    }

    pub fn with_remote(ip: IpAddr, port: u16) -> Result<Self, TensorError> {
        let remote_backend = RemoteBackend::new_with_address(ip, port)
            .map_err(|e| TensorError::RemoteError(format!("Failed to create remote backend: {}", e)))?;
        let buf = remote_backend.alloc::<T>(0)?;
        Ok(Self {
            backend: remote_backend,
            buf,
            meta: MetaTensor::new(vec![], vec![], 0),
            _t: PhantomData,
        })
    }
}

/// A non-owning immutable view over tensor data.
/// 
/// Views share the underlying buffer with the source tensor and have their own
/// metadata (shape, stride, offset) to represent different interpretations of the data.
pub struct TensorView<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    pub(crate) buf: &'a B::Buf<T>,
    pub(crate) backend: &'a B,
    pub(crate) meta: MetaTensor,
}

/// A non-owning mutable view over tensor data.
/// 
/// Like `TensorView` but allows mutation of the underlying data.
pub struct TensorViewMut<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    pub(crate) buf: &'a mut B::Buf<T>,
    pub(crate) backend: &'a B,
    pub(crate) meta: MetaTensor,
}

impl<'a, T, B> TensorView<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(
        buf: &'a B::Buf<T>,
        backend: &'a B,
        meta: MetaTensor
    ) -> Self {
        Self {
            buf,
            backend,
            meta,
        }
    }
}

impl<'a, T, B> TensorViewMut<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(
        raw: &'a mut B::Buf<T>,
        backend: &'a B,
        meta: MetaTensor
    ) -> Self {
        Self {
            buf: raw,
            backend,
            meta
        }
    }
}

pub type CpuTensorView<'a, T> = TensorView<'a, T, Cpu>;
pub type CpuTensorViewMut<'a, T> = TensorViewMut<'a, T, Cpu>;
#[cfg(feature = "cuda")]
pub type CudaTensorView<'a, T> = TensorView<'a, T, crate::backend::cuda::Cuda>;
#[cfg(feature = "cuda")]
pub type CudaTensorViewMut<'a, T> = TensorViewMut<'a, T, crate::backend::cuda::Cuda>;

#[cfg(feature = "remote")]
pub type RemoteTensorView<'a, T> = TensorView<'a, T, RemoteBackend>;
#[cfg(feature = "remote")]
pub type RemoteTensorViewMut<'a, T> = TensorViewMut<'a, T, RemoteBackend>;

impl<B, T: TensorValue> TensorBase<T, B> 
where 
    B: Backend,
{
    /// Internal constructor from raw parts. Used for creating tensors from
    /// existing backend buffers without copying.
    pub(crate) fn from_parts(backend: B, raw: B::Buf<T>, meta: MetaTensor) -> Self {
        Self {
            backend,
            buf: raw,
            meta
        }
    }

    /// Constructs a tensor from a buffer and shape.
    /// 
    /// The buffer must be contiguous and in row-major order.
    /// 
    /// # Errors
    /// - `InvalidShape` if the buffer size doesn't match the shape.
    /// - `InvalidShape` if the shape has more than 128 dimensions.
    /// 
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).unwrap();
    /// ```
    pub fn from_buf(raw: impl Into<Box<[T]>>, shape: impl Into<Shape>) -> Result<Self, TensorError> {
        let shape: Shape = shape.into();
        if shape.len() > 128 {
            // artificial cap due to broadcast cuda kernel...
            return Err(TensorError::InvalidShape(format!(
                "Tensors with more than 128 dimensions are not supported, got {} dimensions",
                shape.len()
            )));
        }
        let backend = B::new();
        let buffer = backend.alloc_from_slice(raw.into())?;
        if shape.iter().product::<usize>() != backend.len(&buffer) {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                shape.iter().product::<usize>(),
                backend.len(&buffer)
            )));
        }
        let stride = shape_to_stride(&shape);
        Ok(Self {
            backend,
            buf: buffer,
            meta: MetaTensor::new(shape, stride, 0)
        })
    }

    /// Creates a rank-0 (scalar) tensor.
    /// 
    /// # Examples
    /// ```ignore
    /// let scalar = Tensor::<f32>::scalar(42.0);
    /// ```
    pub fn scalar(value: T) -> Self {
        Self::from_buf(vec![value], vec![]).unwrap()
    }

    /// Creates a 1-D column tensor from values.
    /// 
    /// # Examples
    /// ```ignore
    /// let col = Tensor::<i32>::column(vec![1, 2, 3]);
    /// ```
    pub fn column(column: impl Into<Box<[T]>>) -> Self {
        let column = column.into();
        let shape = vec![column.len()];
        Self::from_buf(column, shape).unwrap()
    }

    /// Creates a 1xN row tensor from values.
    /// 
    /// # Examples
    /// ```ignore
    /// let row = Tensor::<f32>::row(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn row(row: impl Into<Box<[T]>>) -> Self {
        let row = row.into();
        let shape = vec![1, row.len()];
        Self::from_buf(row, shape).unwrap()
    }

    /// Creates a tensor filled with zeros.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// 
    /// # Examples
    /// ```ignore
    /// let zeros = Tensor::<f32>::zeros((3, 4));
    /// ```
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let zero_buf = vec![T::ZERO; element_count];
        Self::from_buf(zero_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with ones.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// 
    /// # Examples
    /// ```ignore
    /// let ones = Tensor::<f32>::ones((2, 2));
    /// ```
    pub fn ones(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let one_buf = vec![T::ONE; element_count];
        Self::from_buf(one_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with the maximum value for type `T`.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    pub fn max(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let max_buf = vec![T::MAX; element_count];
        Self::from_buf(max_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with the minimum value for type `T`.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    pub fn min(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let min_buf = vec![T::MIN; element_count];
        Self::from_buf(min_buf, shape).expect("Failed to allocate memory")
    }

    /// Squeezes the tensor in place, preventing a new allocation from being made.
    pub fn squeeze_in_place(&mut self) {
        let (new_shape, new_strides) = unsafe { compute_squeezed_parameters(self.shape(), self.strides(), None).unwrap_unchecked() };
        self.meta.shape = new_shape;
        self.meta.strides = new_strides;
    }
}

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

/// Indicates where a tensor's data resides.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "remote")]
    Remote {
        ip: IpAddr,
        port: u16,
        remote_type: Box<DeviceType>
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{primitives::GradTensor, tensor::{TensorAccess, WithGrad}, Tensor}, grad::{self, optim::{Optim, SGD}}, ops::broadcast::l1::l1_loss};

    #[test]
    fn playground() {

        fn model(wa: &GradTensor<f32, Cpu>, wb: &GradTensor<f32, Cpu>, target: &GradTensor<f32, Cpu>) -> GradTensor<f32, Cpu> {
            let c = wa + wb;
            let loss = l1_loss(&c, target);
            loss
        }

        fn modelv2(
            wa: &GradTensor<f32, Cpu>, 
            wb: &GradTensor<f32, Cpu>, 
            wc: &GradTensor<f32, Cpu>, 
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let inter = wb + wc;
            // println!("Intermediate: {:?}", inter);
            let c = wa + &inter;
            let loss = l1_loss(&c, target);
            loss
        }

        grad::with::<f32, Cpu>(|ctx| {
            
            let a = Tensor::<f32>::scalar(1.).grad();
            let b = Tensor::<f32>::ones((2, 2)).param();
            let target = Tensor::<f32>::zeros((2, 2)).grad();

            let mut optim = SGD::<f32, Cpu>::new(1.);
            // optim.register_parameter(&a).unwrap();
            optim.register_parameters(&[&b]).unwrap();
            
            for _ in 0..10 {
                let loss = model(&a, &b, &target);
                println!("Loss: {:?}", loss.borrow().value.item());
                println!("a: {:?}", a);
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }

            println!("{:?}", a);

            let a = Tensor::<f32>::ones((2, 2)).param();
            let b = Tensor::<f32>::ones((2, 2)).param();
            let c = Tensor::<f32>::ones((2, 2)).param();

            optim.register_parameter(&a).unwrap();
            optim.register_parameter(&b).unwrap();
            optim.register_parameter(&c).unwrap();

            for _ in 0..10 {
                let loss = modelv2(&a, &b, &c, &target);
                println!("Loss: {:?}", loss.borrow().value.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
        })
    }
}