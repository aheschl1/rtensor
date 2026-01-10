use rand::distr::weighted::Weight;

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::TensorError, value::{TensorValue, WeightValue}}, grad::{GradContext, GradNode}};


pub(crate) fn accumulate_grad<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
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

pub(crate) fn backwards_add<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Add { left: _, right: _ } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Add backwards.".into()));
    };

    // Gradient of addition is just the upstream gradient for both inputs
    Ok(vec![upstream.clone(), upstream.clone()])
}

pub(crate) fn backwards_l1<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::L1 { grad_map, ..} = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to L1 backwards.".into()));
    };
    let result = vec![
        grad_map * upstream,
        -grad_map * upstream,
    ];
    Ok(result)
}