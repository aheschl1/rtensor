use crate::{backend::Backend, core::{primops::{Exp, InvExp}, tensor::AsViewMut, value::TensorValue}};

pub trait Tanh<T: TensorValue + Exp, B: Backend> {
    fn tanh_inplace(
        &mut self
    );
}

impl<T: TensorValue + Exp + InvExp, B: Backend, V: AsViewMut<T, B>> Tanh<T, B> for V {
    fn tanh_inplace(
        &mut self
    ) {
        let view = self.view_mut();
        // will never fail, could in theory ignore the result
        if let Err(e) = view.backend.apply_tanh(view.buf, &view.meta) {
            panic!("Failed to apply relu: {}", e);
        }
    }
}

// impl<T, B> std::ops::Neg for TensorBase<T, B>
// where
//     T: TensorValue + std::ops::Neg<Output = T>,
//     B: Backend,
// {
//     type Output = TensorBase<T, B>;

//     fn neg(self) -> Self::Output {
//         let mut result = self.owned();
//         result.neg_inplace();
//         result
//     }
// }

// impl<'a, T, B> std::ops::Neg for TensorView<'a, T, B>
// where
//     T: TensorValue + std::ops::Neg<Output = T>,
//     B: Backend,
// {
//     type Output = TensorBase<T, B>;

//     fn neg(self) -> Self::Output {
//         let mut result = self.owned();
//         result.neg_inplace();
//         result
//     }
// }

// impl<'a, T, B> std::ops::Neg for TensorViewMut<'a, T, B>
// where
//     T: TensorValue + std::ops::Neg<Output = T>,
//     B: Backend,
// {
//     type Output = TensorBase<T, B>;

//     fn neg(self) -> Self::Output {
//         let mut result = self.owned();
//         result.neg_inplace();
//         result
//     }
// }

