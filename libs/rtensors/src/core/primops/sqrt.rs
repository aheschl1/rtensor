use crate::core::value::{TensorValue, WeightValue};


pub trait SquareRoot : TensorValue {
    fn apply_sqrt(&self) -> Self;
}

impl<V: WeightValue> SquareRoot for V {
    #[inline(always)]
    fn apply_sqrt(&self) -> Self {
        self.square_root()
    }
}