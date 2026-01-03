use std::{ops::{Sub, SubAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::TensorValue, TensorView, TensorViewMut}, ops::base::BinaryOpType};

impl<'a, T, B> SubAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_sub(
            self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<'a, T, B> SubAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_sub(
            self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> SubAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_sub(
            &mut self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> SubAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_sub(
            &mut self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_sub {
    ($type:ty) => {
        impl<'a, T, B> Sub<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result -= rhs;
                result
            }
        }

        impl<'a, T, B> Sub<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: &T) -> Self::Output {
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
impl_sub!(&TensorBase<T, B>);
impl_sub!(TensorBase<T, B>);
