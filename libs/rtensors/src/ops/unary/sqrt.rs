use crate::{backend::Backend, core::{primops, tensor::AsViewMut, value::TensorValue}};

pub trait Sqrt<T: TensorValue + primops::SquareRoot, B: Backend> {
    fn sqrt_inplace(
        &mut self
    );
}

impl<T: TensorValue + primops::SquareRoot, B: Backend, V: AsViewMut<T, B>> Sqrt<T, B> for V {
    fn sqrt_inplace(
        &mut self
    ) {
        let view = self.view_mut();
        // will never fail, could in theory ignore the result
        if let Err(e) = view.backend.apply_sqrt(view.buf, &view.meta) {
            panic!("Failed to apply sqrt: {}", e);
        }
    }
}