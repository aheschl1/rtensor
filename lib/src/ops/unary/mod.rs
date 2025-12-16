use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{AsTensor, AsViewMut}, value::TensorValue}};

pub mod neg;

pub trait InplaceUnaryOp<T: TensorValue, B: Backend> {
    fn apply_relu(
        &mut self
    );
    fn apply_sigmoid(
        &mut self
    );
    fn apply_tanh(
        &mut self
    );
}

pub trait UnaryOp<T: TensorValue, B: Backend> {
    fn relu(
        &self,
    ) -> TensorBase<T, B>;
    fn sigmoid(
        &self,
    ) -> TensorBase<T, B>;
    fn tanh(
        &self,
    ) -> TensorBase<T, B>;
}


impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> InplaceUnaryOp<T, B> for V {
    fn apply_relu(
        &mut self
    ) {
        todo!()
    }

    fn apply_sigmoid(
        &mut self
    ) {
        todo!()
    }

    fn apply_tanh(
        &mut self
    ) {
        todo!()
    }
}


impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> UnaryOp<T, B> for V {
    fn relu(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_relu();
        result
    }

    fn sigmoid(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_sigmoid();
        result
    }

    fn tanh(
        &self,
    ) -> TensorBase<T, B> {
        let mut result = self.owned();
        result.apply_tanh();
        result
    }
}