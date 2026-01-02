use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::TensorValue}};

pub trait Abs<T: TensorValue, B: Backend> {
    fn abs_inplace(&mut self);
}

// pub trait Abs<T: TensorValue, B: Backend> : AsTensor<T, B> {
//     fn abs(&self) -> TensorBase<T, B>;
// }

impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> Abs<T, B> for V {
    fn abs_inplace(&mut self) {
        let view = self.view_mut();
        // will never fail, could in theory ignore the result
        if let Err(e) = view.backend.apply_abs(view.buf, &view.meta) {
            panic!("Failed to apply abs: {}", e);
        }
    }
}

// impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> Abs<T, B> for V {
//     fn abs(&self) -> TensorBase<T, B> {
//         let mut tensor = self.owned();
//         tensor.abs_inplace();
//         tensor
//     }
// }