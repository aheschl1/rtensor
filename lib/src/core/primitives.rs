use std::marker::PhantomData;

use crate::backend::Backend;
use crate::backend::cpu::Cpu;
use crate::core::value::TensorValue;
use crate::core::{shape_to_stride, Shape, MetaTensor};
use crate::core::tensor::TensorError;

#[derive(Debug, PartialEq, Eq)]
pub struct TensorBase<T: TensorValue, B: Backend<T>> {
    pub(crate) backend: B,
    pub(crate) raw: B::Buf,
    pub(crate) meta: MetaTensor,
    _t: PhantomData<T>,
}

impl<B: Backend<T>, T: TensorValue> Clone for TensorBase<T, B> {
    fn clone(&self) -> Self {
        let new_backend = B::new();
        let new_buffer = new_backend.copy(&self.raw).unwrap();
        Self {
            backend: new_backend,
            raw: new_buffer,
            meta: self.meta.clone(),
            _t: PhantomData,
        }
    }
}

/// An owned, contiguous tensor stored in row-major order.
///
/// Holds the backing buffer and the associated layout metadata. The `offset`
/// in `meta` is expected to be zero for owned tensors created via
/// `from_buf`/`row`/`column`/`scalar`.
pub type CpuTensor<T> = TensorBase<T, Cpu>;

#[cfg(feature = "cuda")]
pub type CudaTensor<T> = TensorBase<T, crate::backend::cuda::Cuda>;

#[cfg(feature = "cuda")]
impl<T: TensorValue> CudaTensor<T> {
    /// Transfers this tensor from the CUDA backend to a CPU tensor.
    pub fn cpu(&self) -> Result<CpuTensor<T>, TensorError> {
        let cpu_backend = Cpu;
        let cpu_buffer = self.backend.dump(&self.raw)?;
        let cpu = CpuTensor::from_parts(cpu_backend, cpu_buffer, self.meta.clone());
        Ok(cpu)
    }
}

#[cfg(feature = "cuda")]
impl<T: TensorValue> CpuTensor<T> {
    /// Transfers this tensor from the CPU backend to a CUDA tensor.
    pub fn cuda(&self) -> Result<CudaTensor<T>, TensorError> {
        let cuda_backend = crate::backend::cuda::Cuda::construct(0)?;
        let cuda_buffer = cuda_backend.alloc_from_slice(self.backend.dump(&self.raw)?)?;
        let cuda = CudaTensor::from_parts(cuda_backend, cuda_buffer, self.meta.clone());
        Ok(cuda)
    }
}

/// A non-owning view over tensor data with explicit layout metadata.
///
/// The `B` parameter abstracts over borrowed storage (e.g., `&[T]` or
/// `&mut [T]`), while `meta` carries shape, stride, and offset describing how
/// to interpret the underlying buffer.
pub struct TensorView<'a, T, B>
where
    T: TensorValue,
    B: Backend<T> + 'a,
{
    pub(crate) raw: &'a B::Buf,
    pub(crate) backend: &'a B,
    pub(crate) meta: MetaTensor,
}

pub struct TensorViewMut<'a, T, B>
where
    T: TensorValue,
    B: Backend<T> + 'a,
{
    pub(crate) raw: &'a mut B::Buf,
    pub(crate) backend: &'a B,
    pub(crate) meta: MetaTensor,
}

impl<'a, T, B> TensorView<'a, T, B>
where
    T: TensorValue,
    B: Backend<T> + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(
        raw: &'a B::Buf,
        backend: &'a B,
        meta: MetaTensor
    ) -> Self {
        Self {
            raw,
            backend,
            meta,
        }
    }
}

impl<'a, T, B> TensorViewMut<'a, T, B>
where
    T: TensorValue,
    B: Backend<T> + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(
        raw: &'a mut B::Buf,
        backend: &'a B,
        meta: MetaTensor
    ) -> Self {
        Self {
            raw,
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

impl<B, T: TensorValue> TensorBase<T, B> 
where 
    B: Backend<T>,
{
    /// Internal constructor from raw parts. Used for creating tensors from
    /// existing backend buffers without copying.
    pub(crate) fn from_parts(backend: B, raw: B::Buf, meta: MetaTensor) -> Self {
        Self {
            backend,
            raw,
            meta,
            _t: PhantomData,
        }
    }

    /// Constructs a tensor from a contiguous buffer in row-major order and a
    /// given shape. Validates that `shape` size equals `raw.len()`.
    ///
    /// Errors
    /// - `InvalidShape` if element count doesn't match.
    pub fn from_buf(raw: impl Into<Box<[T]>>, shape: impl Into<Shape>) -> Result<Self, TensorError> {
        let shape: Shape = shape.into();
        if shape.len() > 128 {
            // artificial cap due to broadcast cuda kernel...
            return Err(TensorError::InvalidShape)
        }
        let backend = B::new();
        let buffer = backend.alloc_from_slice(raw.into())?;
        if shape.iter().product::<usize>() != backend.len(&buffer) {
            return Err(TensorError::InvalidShape);
        }
        let stride = shape_to_stride(&shape);
        Ok(Self {
            backend,
            raw: buffer,
            meta: MetaTensor::new(shape, stride, 0),
            _t: PhantomData,
        })
    }

    /// Creates a rank-0 (scalar) tensor holding `value`.
    pub fn scalar(value: T) -> Self {
        Self::from_buf(vec![value], vec![]).unwrap()
    }

    /// Creates a 1-D column tensor from the provided values.
    pub fn column(column: impl Into<Box<[T]>>) -> Self {
        let column = column.into();
        let shape = vec![column.len()];
        Self::from_buf(column, shape).unwrap()
    }

    /// Creates a 1xN row tensor from the provided values.
    pub fn row(row: impl Into<Box<[T]>>) -> Self {
        let row = row.into();
        let shape = vec![1, row.len()];
        Self::from_buf(row, shape).unwrap()
    }

    /// Creates a tensor filled with zeroes for the given shape.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// Use `TensorBase::from_buf` for fallible allocation.
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let zero_buf = vec![T::zero(); element_count];
        Self::from_buf(zero_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with ones for the given shape.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// Use `TensorBase::from_buf` for fallible allocation.
    pub fn ones(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let one_buf = vec![T::one(); element_count];
        Self::from_buf(one_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with the maximum value for the given shape.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// Use `TensorBase::from_buf` for fallible allocation.
    pub fn max(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let max_buf = vec![T::max(); element_count];
        Self::from_buf(max_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with the minimum value for the given shape.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// Use `TensorBase::from_buf` for fallible allocation.
    pub fn min(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let min_buf = vec![T::min(); element_count];
        Self::from_buf(min_buf, shape).expect("Failed to allocate memory")
    }
}
