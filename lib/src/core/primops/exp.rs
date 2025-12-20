use crate::core::value::{TensorValue, WeightValue};

pub trait Exp: TensorValue {
    fn apply_exp(&self) -> Self;
}


impl<V: WeightValue> Exp for V {
    #[inline]
    fn apply_exp(&self) -> Self {
        self.vexp()
    }
}

pub trait InvExp: TensorValue {
    fn apply_invexp(&self) -> Self;
}

impl<V: WeightValue> InvExp for V {
    #[inline]
    fn apply_invexp(&self) -> Self {
        (-*self).vexp()
    }
}

// TODO: Apply for other types.