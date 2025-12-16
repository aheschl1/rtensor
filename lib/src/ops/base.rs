use crate::core::value::TensorValue;

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
#[derive(Clone, Copy)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul
}

#[cfg(feature = "cuda")]
impl BinaryOpType {
    #[inline(always)]
    pub(crate) fn to_op_code(&self) -> u8 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
        }
    }
}
    
impl BinaryOpType {
    #[inline(always)]
    pub fn apply<T: TensorValue>(&self, a: T, b: T) -> T 
    {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
        }
    }
}