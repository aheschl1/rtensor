use std::{borrow::Borrow, ops::{Add, AddAssign}};

use crate::{backend::BackendUnaryElementwise, core::{primitives::TensorBase, tensor::AsTensor, value::{TensorValue, TensorValueElementwise}, TensorView, TensorViewMut}, ops::unary::ElementwiseTensorOp};

impl<'a, T, B, O> AddAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn add_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseTensorOp::Add(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<T, B, O> AddAssign<O> for TensorBase<B, T> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn add_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            &mut self.raw, 
            ElementwiseTensorOp::Add(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_add {
    ($type:ty) => {
        impl<'a, T, B, O> Add<O> for $type
        where
            T: TensorValueElementwise + TensorValue,
            B: BackendUnaryElementwise<T>,
            O: Borrow<T>,
        {
            type Output = TensorBase<B, T>;

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
impl_add!(&TensorBase<B, T>);
impl_add!(TensorBase<B, T>);