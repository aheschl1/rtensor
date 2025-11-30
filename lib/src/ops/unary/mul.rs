use std::{borrow::Borrow, ops::{Mul, MulAssign}};

use crate::{backend::BackendUnaryElementwise, core::{primitives::TensorBase, tensor::AsTensor, value::{TensorValue, TensorValueElementwise}, TensorView, TensorViewMut}, ops::unary::ElementwiseTensorOp};

impl<'a, T, B, O> MulAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn mul_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseTensorOp::Mul(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<T, B, O> MulAssign<O> for TensorBase<B, T> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendUnaryElementwise<T>,
          O: Borrow<T>
{
    fn mul_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            &mut self.raw, 
            ElementwiseTensorOp::Mul(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_mul {
    ($type:ty) => {
        impl<'a, T, B, O> Mul<O> for $type
        where
            T: TensorValueElementwise + TensorValue,
            B: BackendUnaryElementwise<T>,
            O: Borrow<T>,
        {
            type Output = TensorBase<B, T>;

            fn mul(self, rhs: O) -> Self::Output {
                let mut result = self.owned();
                result *= rhs;
                result
            }
        }
    };
}

impl_mul!(&TensorViewMut<'a, T, B>);
impl_mul!(TensorViewMut<'a, T, B>);
impl_mul!(&TensorView<'a, T, B>);
impl_mul!(TensorView<'a, T, B>);
impl_mul!(&TensorBase<B, T>);
impl_mul!(TensorBase<B, T>);