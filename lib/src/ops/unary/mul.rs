use std::{borrow::Borrow, ops::{Mul, MulAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::TensorValue, TensorView, TensorViewMut}, ops::unary::ElementwiseUnaryTensorOp};

impl<'a, T, B, O> MulAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn mul_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseUnaryTensorOp::Mul(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<T, B, O> MulAssign<O> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn mul_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            &mut self.raw, 
            ElementwiseUnaryTensorOp::Mul(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_mul {
    ($type:ty) => {
        impl<'a, T, B, O> Mul<O> for $type
        where
            T: TensorValue,
            B: Backend<T>,
            O: Borrow<T>,
        {
            type Output = TensorBase<T, B>;

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
impl_mul!(&TensorBase<T, B>);
impl_mul!(TensorBase<T, B>);