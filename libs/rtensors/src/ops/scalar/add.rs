use std::{ops::{Add, AddAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, value::TensorValue, TensorView, TensorViewMut}};
use crate::core::tensor::AsTensor;

impl<'a, T, B> AddAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_add(
            self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<'a, T, B> AddAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_add(
            self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> AddAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_add(
            &mut self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> AddAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_add(
            &mut self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_add {
    ($type:ty) => {
        impl<'a, T, B> Add<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result += rhs;
                result
            }
        }

        impl<'a, T, B> Add<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: &T) -> Self::Output {
                let mut result = self.owned();
                result += rhs;
                result
            }
        }
    };
}

impl_add!(&TensorViewMut<'a, T, B>);
impl_add!(TensorViewMut<'a, T, B>);
impl_add!(&TensorView<'a, T, B>);
impl_add!(TensorView<'a, T, B>);
impl_add!(&TensorBase<T, B>);
impl_add!(TensorBase<T, B>);

// impl<T, B> std::ops::Add<GradTensor<T, B>> for GradTensor<T, B> 
//     where T: WeightValue,
//           B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn add(self, rhs: GradTensor<T, B>) -> Self::Output {
//         let value = &rhs.borrow().value + &self.borrow().value;
//         let op = GradNode::Add { left: self.node, right: rhs.node };
//         GradTensor::from_op(value, op)
//     }
// }

// impl<T, B> std::ops::Add<&GradTensor<T, B>> for &GradTensor<T, B> 
//     where T: WeightValue,
//           B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn add(self, rhs: &GradTensor<T, B>) -> Self::Output {
//         let value = &rhs.borrow().value + &self.borrow().value;
//         let op = GradNode::Add { left: self.node, right: rhs.node };
//         GradTensor::from_op(value, op)
//     }
// }