use std::{borrow::Borrow, ops::{Sub, SubAssign}};

use crate::{backend::BackendUnaryElementwise, core::{primitives::TensorBase, tensor::AsTensor, value::{TensorValue, TensorValueElementwise}, TensorView, TensorViewMut}, ops::unary::ElementwiseTensorOp};

impl<'a, T, B, O> SubAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn sub_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseTensorOp::Sub(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<T, B, O> SubAssign<O> for TensorBase<B, T> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn sub_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            &mut self.raw, 
            ElementwiseTensorOp::Sub(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_sub {
    ($type:ty) => {
        impl<'a, T, B, O> Sub<O> for $type
        where
            T: TensorValueElementwise + TensorValue,
            B: BackendUnaryElementwise<T>,
            O: Borrow<T>,
        {
            type Output = TensorBase<B, T>;

            fn sub(self, rhs: O) -> Self::Output {
                let mut result = self.owned();
                result -= rhs;
                result
            }
        }
    };
}

impl_sub!(&TensorViewMut<'a, T, B>);
impl_sub!(TensorViewMut<'a, T, B>);
impl_sub!(&TensorView<'a, T, B>);
impl_sub!(TensorView<'a, T, B>);
impl_sub!(&TensorBase<B, T>);
impl_sub!(TensorBase<B, T>);
