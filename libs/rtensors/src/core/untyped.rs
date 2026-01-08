use std::fmt::Debug;

use crate::{backend::Backend, core::{primitives::{DeviceType, TensorBase}, tensor::{AsView, AsViewMut}, value::{types, DType, TensorValue}, MetaTensor, TensorView, TensorViewMut}};

/// Trait for erased tensors, allowing dynamic dispatch on tensor types.
/// Implemented for all `TensorBase<T, B>` where `T: TensorValue` and `B: Backend<T>`.
pub trait UntypedTensor<B: Backend>: Send + Sync + Debug {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn device(&self) -> DeviceType;
    fn dtype(&self) -> DType;
    fn meta(&self) -> &MetaTensor;
}

impl<T, B> UntypedTensor<B> for TensorBase<T, B>
where
    T: crate::core::value::TensorValue + 'static,
    B: crate::backend::Backend + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceType {
        B::device_type()
    }

    fn dtype(&self) -> DType {
        T::DTYPE
    }

    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

/// Downcasting methods for `ErasedTensor`.
/// Allows retrieving the concrete tensor type from the erased trait object.
/// Requires knowing the original `T` and `B` types.
impl<B: Backend> dyn UntypedTensor<B> {
    pub fn typed_unknown(&self) -> UnknownTensor<'_, B> {
        match self.dtype() {
            DType::U8 => UnknownTensor::U8(self.typed::<u8>().unwrap()),
            DType::U16 => UnknownTensor::U16(self.typed::<u16>().unwrap()),
            DType::U32 => UnknownTensor::U32(self.typed::<u32>().unwrap()),
            DType::U64 => UnknownTensor::U64(self.typed::<u64>().unwrap()),
            DType::U128 => UnknownTensor::U128(self.typed::<u128>().unwrap()),
            DType::I8 => UnknownTensor::I8(self.typed::<i8>().unwrap()),
            DType::I16 => UnknownTensor::I16(self.typed::<i16>().unwrap()),
            DType::I32 => UnknownTensor::I32(self.typed::<i32>().unwrap()),
            DType::I64 => UnknownTensor::I64(self.typed::<i64>().unwrap()),
            DType::I128 => UnknownTensor::I128(self.typed::<i128>().unwrap()),
            DType::F32 => UnknownTensor::F32(self.typed::<f32>().unwrap()),
            DType::F64 => UnknownTensor::F64(self.typed::<f64>().unwrap()),
            DType::BOOL => UnknownTensor::BOOL(self.typed::<types::boolean>().unwrap()),
        }
    }

    pub fn typed_mut_unknown(&mut self) -> UnknownTensorMut<'_, B> {
        match self.dtype() {
            DType::U8 => UnknownTensorMut::U8(self.typed_mut::<u8>().unwrap()),
            DType::U16 => UnknownTensorMut::U16(self.typed_mut::<u16>().unwrap()),
            DType::U32 => UnknownTensorMut::U32(self.typed_mut::<u32>().unwrap()),
            DType::U64 => UnknownTensorMut::U64(self.typed_mut::<u64>().unwrap()),
            DType::U128 => UnknownTensorMut::U128(self.typed_mut::<u128>().unwrap()),
            DType::I8 => UnknownTensorMut::I8(self.typed_mut::<i8>().unwrap()),
            DType::I16 => UnknownTensorMut::I16(self.typed_mut::<i16>().unwrap()),
            DType::I32 => UnknownTensorMut::I32(self.typed_mut::<i32>().unwrap()),
            DType::I64 => UnknownTensorMut::I64(self.typed_mut::<i64>().unwrap()),
            DType::I128 => UnknownTensorMut::I128(self.typed_mut::<i128>().unwrap()),
            DType::F32 => UnknownTensorMut::F32(self.typed_mut::<f32>().unwrap()),
            DType::F64 => UnknownTensorMut::F64(self.typed_mut::<f64>().unwrap()),
            DType::BOOL => UnknownTensorMut::BOOL(self.typed_mut::<types::boolean>().unwrap()),
        }
    }

    pub fn typed<T>(&self) -> Option<&TensorBase<T, B>>
    where
        T: TensorValue + 'static,
    {
        self.as_any().downcast_ref::<TensorBase<T, B>>()
    }

    pub fn typed_mut<T>(&mut self) -> Option<&mut TensorBase<T, B>>
    where
        T: TensorValue + 'static,
    {
        self.as_any_mut().downcast_mut::<TensorBase<T, B>>()
    }

    pub fn view_typed<T>(&self) -> Option<TensorView<'_, T, B>>
    where
        T: TensorValue + 'static,
    {
        self.typed::<T>().map(|t| t.view())
    }

    pub fn view_typed_mut<T>(&mut self) -> Option<TensorViewMut<'_, T, B>>
    where
        T: TensorValue + 'static,
    {
        self.typed_mut::<T>().map(|t| t.view_mut())
    }
}

pub trait AsUntypedTensor<B: Backend> {
    fn as_untyped(self) -> Box<dyn UntypedTensor<B>>;
}

impl<T, B> AsUntypedTensor<B> for TensorBase<T, B>
where
    T: TensorValue + 'static,
    B: Backend + 'static,
{
    fn as_untyped(self) -> Box<dyn UntypedTensor<B>> {
        Box::new(self)
    }
}


// TODO: Do not duplicate this logic and find a better way 

pub enum UnknownTensor<'a, B: Backend> {
    U8(&'a TensorBase<u8, B>),
    U16(&'a TensorBase<u16, B>),
    U32(&'a TensorBase<u32, B>),
    U64(&'a TensorBase<u64, B>),
    U128(&'a TensorBase<u128, B>),
    I8(&'a TensorBase<i8, B>),
    I16(&'a TensorBase<i16, B>),
    I32(&'a TensorBase<i32, B>),
    I64(&'a TensorBase<i64, B>),
    I128(&'a TensorBase<i128, B>),
    F32(&'a TensorBase<f32, B>),
    F64(&'a TensorBase<f64, B>),
    BOOL(&'a TensorBase<types::boolean, B>),
}

pub enum UnknownTensorMut<'a, B: Backend> {
    U8(&'a mut TensorBase<u8, B>),
    U16(&'a mut TensorBase<u16, B>),
    U32(&'a mut TensorBase<u32, B>),
    U64(&'a mut TensorBase<u64, B>),
    U128(&'a mut TensorBase<u128, B>),
    I8(&'a mut TensorBase<i8, B>),
    I16(&'a mut TensorBase<i16, B>),
    I32(&'a mut TensorBase<i32, B>),
    I64(&'a mut TensorBase<i64, B>),
    I128(&'a mut TensorBase<i128, B>),
    F32(&'a mut TensorBase<f32, B>),
    F64(&'a mut TensorBase<f64, B>),
    BOOL(&'a mut TensorBase<types::boolean, B>),
}



#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{untyped::UntypedTensor, primitives::DeviceType, tensor::TensorError, value::TensorValue, Shape, Tensor}};

    #[test]
    fn test_erased_tensor_downcast() -> Result<(), TensorError> {
        // cpu, f32 tensor
        let tensor = Tensor::<f32>::zeros((2, 3));
        let erased: Box<dyn UntypedTensor<Cpu>> = Box::new(tensor);
        assert_eq!(erased.device(), DeviceType::Cpu);
        assert_eq!(erased.dtype(), f32::DTYPE);
        let downcasted = erased.typed::<f32>().unwrap();
        assert_eq!(downcasted.meta().shape, Shape::from((2, 3)));
        Ok(())
    }
}