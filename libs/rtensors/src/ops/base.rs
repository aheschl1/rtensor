use crate::core::value::TensorValue;

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
#[derive(Clone, Copy)]
pub enum BinaryOpType {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3
}

#[cfg(feature = "cuda")]
impl BinaryOpType {
    #[inline(always)]
    pub(crate) fn to_op_code(&self) -> u8 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
            Self::Div => 3
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
            Self::Div => a / b
        }
    }
}