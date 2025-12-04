use crate::core::value::TensorValue;


pub enum OpType {
    Add,
    Sub,
    Mul,
} 

#[cfg(feature = "cuda")]
impl OpType {
    #[inline(always)]
    pub(crate) fn to_op_code(&self) -> u8 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
            _ => panic!("Unsupported operation"),
        }
    }
}
    
impl OpType {
    #[inline(always)]
    pub fn apply<T: TensorValue>(&self, a: T, b: T) -> T 
    {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            _ => panic!("Unsupported operation"),
        }
    }
}