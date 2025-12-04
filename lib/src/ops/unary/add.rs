use std::{borrow::Borrow, ops::{Add, AddAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::TensorValue, TensorView, TensorViewMut}, ops::unary::ElementwiseUnaryTensorOp};

impl<'a, T, B, O> AddAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn add_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseUnaryTensorOp::Add(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<T, B, O> AddAssign<O> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn add_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            &mut self.raw, 
            ElementwiseUnaryTensorOp::Add(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_add {
    ($type:ty) => {
        impl<'a, T, B, O> Add<O> for $type
        where
            T: TensorValue,
            B: Backend<T>,
            O: Borrow<T>,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: O) -> Self::Output {
                let mut result = self.owned();
                result += rhs;
                result
            }
        }
    };
}

impl_add!(&TensorViewMut<'a, T, B>);
impl_add!(TensorViewMut<'a, T, B>);
impl_add!(&TensorView<'a, T, B>);
impl_add!(TensorView<'a, T, B>);
impl_add!(&TensorBase<T, B>);
impl_add!(TensorBase<T, B>);