use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::TensorValue, TensorView, TensorViewMut}};

pub trait Abs<T: TensorValue, B: Backend> {
    fn abs_inplace(
        &mut self
    );
}

impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> Abs<T, B> for V {
    fn abs_inplace(
        &mut self
    ) {
        let view = self.view_mut();
        // will never fail, could in theory ignore the result
        if let Err(e) = view.backend.apply_abs(view.buf, &view.meta) {
            panic!("Failed to apply abs: {}", e);
        }
    }
}


