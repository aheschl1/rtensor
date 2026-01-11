use std::{cell::RefCell, sync::Arc};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{TensorAccess, TensorAccessMut}, value::{TensorValue, WeightValue}, MetaTensorView}, grad::{self, primitives::GradTensor, GradNode}, ops::{reduction::{ReductionOp, TotalReductionOp}, unary::UnaryOp}};



trait L1<T: WeightValue, B: Backend> {
    fn l1(&self, target: &Self) -> Self;
}

impl<T: WeightValue, B: Backend> L1<T, B> for TensorBase<T, B> {
    fn l1(&self, target: &Self) -> Self {
        let result = (self - target).abs().mean().expect("L1 loss computation failed");
        result
    }
}

impl<T: WeightValue, B: Backend> L1<T, B> for GradTensor<T, B> {
    fn l1(&self, target: &Self) -> Self {
        let self_inner = self.borrow();
        let target_inner = target.borrow();

        let self_tensor = &self_inner.tensor;
        let target_tensor = &target_inner.tensor;

        let diff = self_tensor - target_tensor;
        let mut grad_map = TensorBase::<T, B>::zeros(diff.shape());
        // TODO replace with sign operation
        for coord in diff.iter_coords() {
            let d = diff.get(&coord).unwrap();
            if d > T::ZERO {
                grad_map.set(&coord, T::ONE);
            } else if d < T::ZERO {
                grad_map.set(&coord, -T::ONE);
            } else {
                grad_map.set(&coord, T::ZERO);
            }
        }

        grad_map /= T::from_usize(self_tensor.size());

        let loss_tensor = self_tensor.l1(target_tensor);
        GradTensor::from_op_self_referential(loss_tensor, |inner| 
            GradNode::L1 {
                input: self.node,
                target: target.node,
                loss: inner,
                grad_map,
            }
        )
    }
}

pub fn l1_loss<T: WeightValue, B: Backend>(
    input: &GradTensor<T, B>,
    target: &GradTensor<T, B>,
) -> GradTensor<T, B> {
    input.l1(target)
}