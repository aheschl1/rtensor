use std::{borrow::Borrow, ops::{Sub, SubAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::TensorValue, TensorView, TensorViewMut}, ops::unary::ElementwiseUnaryTensorOp};

impl<'a, T, B, O> SubAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn sub_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseUnaryTensorOp::Sub(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

impl<T, B, O> SubAssign<O> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend<T>,
          O: Borrow<T>
{
    fn sub_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            &mut self.raw, 
            ElementwiseUnaryTensorOp::Sub(*rhs.borrow()),
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_sub {
    ($type:ty) => {
        impl<'a, T, B, O> Sub<O> for $type
        where
            T: TensorValue,
            B: Backend<T>,
            O: Borrow<T>,
        {
            type Output = TensorBase<T, B>;

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
impl_sub!(&TensorBase<T, B>);
impl_sub!(TensorBase<T, B>);
