

use crate::core::{tensor::TensorError, MetaTensor};
pub mod add;
pub mod sub;
pub mod mul;

#[inline]
pub(crate) fn compute_broadcasted_params(
    a: &MetaTensor,
    b: &MetaTensor,
) -> Result<(Vec<usize>, Vec<isize>, Vec<isize>), TensorError>{
    let mut sa: Vec<usize> = a.shape().clone().into();
    let mut sb: Vec<usize> = b.shape().clone().into();
    let mut stra: Vec<isize> = a.strides().clone().into();
    let mut strb: Vec<isize> = b.strides().clone().into();

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
