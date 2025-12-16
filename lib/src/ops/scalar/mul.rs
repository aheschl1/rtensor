use std::{ops::{Mul, MulAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::TensorValue, TensorView, TensorViewMut}, ops::base::BinaryOpType};

impl<'a, T, B> MulAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: T) {
        self.backend.apply_elementwise_binary(
            self.buf, 
            (BinaryOpType::Mul, rhs),
            &self.meta
        ).unwrap();
    }
}

impl<'a, T, B> MulAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.backend.apply_elementwise_binary(
            self.buf, 
            (BinaryOpType::Mul, *rhs),
            &self.meta
        ).unwrap();
    }
}

impl<T, B> MulAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: T) {
        self.backend.apply_elementwise_binary(
            &mut self.buf, 
            (BinaryOpType::Mul, rhs),
            &self.meta
        ).unwrap();
    }
}

impl<T, B> MulAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.backend.apply_elementwise_binary(
            &mut self.buf, 
            (BinaryOpType::Mul, *rhs),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_mul {
    ($type:ty) => {
        impl<'a, T, B> Mul<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn mul(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result *= rhs;
                result
            }
        }

        impl<'a, T, B> Mul<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn mul(self, rhs: &T) -> Self::Output {
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