use crate::{backend::Backend, core::{primitives::TensorBase, tensor::TensorError, value::TensorValue}, grad::{GradContext, GradNode}};


pub(crate) fn accumulate_grad<T: TensorValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Leaf( grad_ref ) = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to accumulate_grad.".into()));
    };

    let mut grad_tensor = grad_ref.borrow_mut();
    if let Some(existing_grad) = &mut grad_tensor.grad {
        // Accumulate gradient
        *existing_grad += upstream;
    } else {
        // First gradient assignment
        grad_tensor.grad = Some(upstream.clone().into());
    }

    Ok(vec![upstream.clone()])
}

pub(crate) fn backwards_l1<T: TensorValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::L1 { input, target } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to L1 backwards.".into()));
    };

    todo!()
}