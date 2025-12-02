
use std::marker::PhantomData;

use crate::{backend::{Backend, BackendBinaryElementwise}, core::{primitives::TensorBase, tensor::TensorError, value::{TensorValue, TensorValueElementwise}, MetaTensor, MetaTensorView, TensorView}};


pub enum ElementwiseBinaryTensorOp<T: TensorValueElementwise> {
    Add,
    _P(PhantomData<T>)
}

impl<T: TensorValueElementwise> ElementwiseBinaryTensorOp<T> {
    #[inline(always)]
    pub fn apply(&self, a: T, b: T) -> T 
    {
        match self {
            ElementwiseBinaryTensorOp::Add => a + b,
            _ => panic!("Unsupported operation"),
        }
    }

    pub fn to_op_code(&self) -> u8 {
        match self {
            ElementwiseBinaryTensorOp::Add => 0,
            _ => panic!("Unsupported operation"),
        }
    }
}


pub trait PointwiseTensorOp<T, B: Backend<T>> 
where T: TensorValue
{
    fn add(&'_ self, other: &TensorView<T, B>) -> Result<TensorBase<T, B>, TensorError>;
}

impl <T, B> PointwiseTensorOp<T, B> for TensorView<'_, T, B> 
where T: TensorValueElementwise,
      B: BackendBinaryElementwise<T>
{
    fn add(&'_ self, other: &TensorView<T, B>) -> Result<TensorBase<T, B>, TensorError> {
        let (out_shape, broadcast_stra, broadcast_strb) = compute_broadcasted_params(&self.meta, &other.meta)?;
        let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
        let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, other.offset());

        let view_a = TensorView::from_parts(self.raw, self.backend, meta_a);
        let view_b = TensorView::from_parts(other.raw, other.backend, meta_b);

        let mut result = TensorBase::<T, B>::zeros(out_shape);

        self.backend.broadcast(
            (view_a.raw, &view_a.meta), 
            (view_b.raw, &view_b.meta),
            (&mut result.raw, &result.meta),
            ElementwiseBinaryTensorOp::Add,
        )?;

        Ok(result)
    }
}

/// the new shape is the broadcasted shape
/// starting from the last dimension
/// 1. If match, take the dimension
/// 2. If one is 1, take the other dimension
/// 3. If neither is 1 and they don't match, error
/// 4. If one tensor has no more dimensions, take the other dimension
pub(crate) fn compute_broadcasted_params(
    a: &MetaTensor,
    b: &MetaTensor,
) -> Result<(Vec<usize>, Vec<isize>, Vec<isize>), TensorError>{
    let mut sa: Vec<usize> = a.shape().clone().into();
    let mut sb: Vec<usize> = b.shape().clone().into();
    let mut stra: Vec<isize> = a.stride().clone();
    let mut strb: Vec<isize> = b.stride().clone();

    if sa.len() < sb.len() {
        let pad = sb.len() - sa.len();
        sa.splice(0..0, std::iter::repeat_n(1, pad));
        stra.splice(0..0, std::iter::repeat_n(0, pad)); // Stride = 0 for inserted dims
    }
    if sb.len() < sa.len() {
        let pad = sa.len() - sb.len();
        sb.splice(0..0, std::iter::repeat_n(1, pad));
        strb.splice(0..0, std::iter::repeat_n(0, pad));
    }

    let mut out_shape = vec![];
    let mut out_stra = vec![];
    let mut out_strb = vec![];

    for i in 0..sa.len() {
        let a = sa[i];
        let b = sb[i];

        let out = match (a, b) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(TensorError::BroadcastError(format!("Shapes {:?} and {:?} are not broadcastable", sa, sb))),
        };

        out_shape.push(out);
        
        out_stra.push(if a == 1 && out > 1 { 0 } else { stra[i] });
        out_strb.push(if b == 1 && out > 1 { 0 } else { strb[i] });
    }

    Ok((out_shape, out_stra, out_strb))
}
