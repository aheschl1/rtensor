use rand::distr::weighted::Weight;

use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, tensor::{TensorAccess, TensorError}, value::{TensorValue, WeightValue}}, grad::{GradContext, GradNode}};


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

/// Backward function for the Permute operation
/// Reverses the permutation applied in the forward pass
pub fn backwards_permute<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Permute { dims, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Permute backwards.".into()));
    };
    let dims_vec = match dims {
        Idx::Coord(dims) => dims.clone(),
        Idx::At(i) => vec![*i],
        Idx::Item => vec![]
    };
    let mut inverse_dims = vec![0; dims_vec.len()];
    for (i, &d) in dims_vec.iter().enumerate() {
        inverse_dims[d] = i;
    }
    let permuted_grad = upstream.permute(inverse_dims)?;
    let mut grad = upstream.clone();
    grad.meta = permuted_grad.meta;
    Ok(vec![grad])
}

pub fn backwards_relu<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::ReLU { grad_map, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to ReLU backwards.".into()));
    };
    let grad = grad_map * upstream;
    Ok(vec![grad])
}

pub fn backwards_negate<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Negate { .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Negate backwards.".into()));
    };
    let grad = -upstream;
    Ok(vec![grad])
}

pub fn backwards_sqrt<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Sqrt { output, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sqrt backwards.".into()));
    };
    let two = T::from_f32(2.0);
    let grad = upstream / (output * two);
    Ok(vec![grad])
}

pub fn backwards_abs<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Abs { grad_map, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Abs backwards.".into()));
    };
    let grad = grad_map * upstream;
    Ok(vec![grad])
}

pub fn backwards_sigmoid<T: WeightValue, B: Backend>(
    node: &GradNode<T, B>, 
    upstream: &TensorBase<T, B>,
    _ctx: &GradContext<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Sigmoid { result, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sigmoid backwards.".into()));
    };
    let one = T::from_f32(1.0);
    // TODO allow scalars on left hand side so it is fewer kernel launches
    let grad = result * (-result + one) * upstream;
    Ok(vec![grad])
}