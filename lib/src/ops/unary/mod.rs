use crate::core::value::TensorValue;

pub mod add;
pub mod sub;
pub mod mul;

pub enum ElementwiseUnaryTensorOp<T> 
where T: TensorValue
{
    Add(T),
    Sub(T),
    Mul(T),
}

impl<T> ElementwiseUnaryTensorOp<T> 
where T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + TensorValue
{
    #[inline(always)]
    pub fn apply(&self, x: T) -> T {
        match self {
            ElementwiseUnaryTensorOp::Add(val) => x + *val,
            ElementwiseUnaryTensorOp::Sub(val) => x - *val,
            ElementwiseUnaryTensorOp::Mul(val) => x * *val,
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> ElementwiseUnaryTensorOp<T>
where T: TensorValue
{
    /// Convert operation to op code (0=Add, 1=Sub, 2=Mul)
    #[inline(always)]
    pub fn to_op_code(&self) -> u8 {
        match self {
            ElementwiseUnaryTensorOp::Add(_) => 0,
            ElementwiseUnaryTensorOp::Sub(_) => 1,
            ElementwiseUnaryTensorOp::Mul(_) => 2,
        }
    }

    /// Get the operation value
    #[inline(always)]
    pub fn value(&self) -> T {
        match self {
            ElementwiseUnaryTensorOp::Add(v) => *v,
            ElementwiseUnaryTensorOp::Sub(v) => *v,
            ElementwiseUnaryTensorOp::Mul(v) => *v,
        }
    }
}
