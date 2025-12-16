use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::TensorValue, TensorView, TensorViewMut}};

pub trait Negate<T: TensorValue + std::ops::Neg<Output = T>, B: Backend> {
    fn neg_inplace(
        &mut self
    );
}

impl<T: TensorValue + std::ops::Neg<Output = T>, B: Backend, V: AsViewMut<T, B>> Negate<T, B> for V {
    fn neg_inplace(
        &mut self
    ) {
        let mut view = self.view_mut();
        // will never fail, could in theory ignore the result
        if let Err(e) = view.backend.apply_neg(view.buf, &view.meta) {
            panic!("Failed to apply negation: {}", e);
        }
    }
}

impl<T, B> std::ops::Neg for TensorBase<T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for TensorView<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for TensorViewMut<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}