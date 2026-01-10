use crate::{backend::Backend, core::{primitives::{GradTensor, GradTensorRef}, tensor::TensorError, value::WeightValue}, grad};


    
pub trait Optim<T: WeightValue, B: Backend> {
    fn step(&mut self) -> Result<(), TensorError>;
    fn register_parameter(&mut self, param: &GradTensor<T, B>) -> Result<(), TensorError>;
    fn register_parameters(&mut self, param: &[&GradTensor<T, B>]) -> Result<(), TensorError> {
        for p in param {
            self.register_parameter(p)?;
        }
        Ok(())
    }
}

pub struct SGD<T: WeightValue, B: Backend> {
    parameters: Vec<GradTensorRef<T, B>>,
    learning_rate: T,
}

impl<T: WeightValue, B: Backend> SGD<T, B> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            parameters: Vec::new(),
            learning_rate,
        }
    }
}

impl<T: WeightValue, B: Backend> Optim<T, B> for SGD<T, B> {
    fn step(&mut self) -> Result<(), TensorError> {
        for param_ref in &self.parameters {
            let mut param = param_ref.borrow_mut();
            if let Some(grad) = &param.grad {
                // Update parameter: param = param - learning_rate * grad
                let update = grad * self.learning_rate;
                param.value -= &update;
                // Clear gradient after update
                param.grad = None;
            }
        }
        Ok(())  
    }

    #[grad::when_enabled(ctx, message = "Cannot register a parameter without a grad context.")]
    fn register_parameter(&mut self, param: &GradTensor<T, B>) -> Result<(), TensorError> {
        // check is leaf node
        let nodes = ctx.nodes.borrow();
        let node = nodes.get(param.node).ok_or_else(|| TensorError::GradError("Parameter not found in grad context.".into()))?;
        if !node.is_leaf() {
            return Err(TensorError::GradError("Only leaf tensors can be registered as parameters.".into()));
        }
        self.parameters.push(param.get_ref());
        Ok(())
    }
}