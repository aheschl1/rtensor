use std::{borrow::Borrow, ops::{Mul, MulAssign}};

use crate::{backend::BackendElementwise, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::{TensorValue, TensorValueElementwise}, TensorView, TensorViewMut}, ops::elementwise::ElementwiseTensorOp};

impl<'a, T, B, O> MulAssign<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendElementwise<T>,
          O: Borrow<T>
{
    fn mul_assign(&mut self, rhs: O) {
        self.backend.apply_elementwise(
            self.raw, 
            ElementwiseTensorOp::Mul(*rhs.borrow()),
            self.meta.mem_regions().as_slice()
        ).unwrap();
    }
}

impl<'a, T, B, O> Mul<O> for TensorViewMut<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendElementwise<T>,
          O: Borrow<T>
{
    type Output = TensorBase<B, T>;

    fn mul(self, rhs: O) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view *= rhs;
        result
    }
}

impl<'a, T, B, O> Mul<O> for TensorView<'a, T, B> 
    where T: TensorValueElementwise + TensorValue,
          B: BackendElementwise<T>,
          O: Borrow<T>
{
    type Output = TensorBase<B, T>;

    fn mul(self, rhs: O) -> Self::Output {
        let mut result = self.owned();
        let mut view = result.view_mut();
        view *= rhs;
        result
    }
}