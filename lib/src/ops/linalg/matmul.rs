use crate::{backend::{BackendMatMul}, core::{primitives::TensorBase, shape_to_stride, tensor::{AsTensor, AsView, TensorError}, value::TensorValue, Dim, MetaTensor, MetaTensorView, Shape}, ops::linalg::MatMul};

impl<L, R, T, B> MatMul<R, T, B> for L
where
    T: TensorValue,
    B: BackendMatMul<T>,
    L: AsView<T, B>,
    R: AsView<T, B>,
{
    fn matmul(&self, rhs: &R) -> Result<TensorBase<T, B>, TensorError> {
        let lhs_view0 = self.view();
        let rhs_view0 = rhs.view();

        let mut _lhs_storage = None;
        let lhs_view = if lhs_view0.is_contiguous(){
            lhs_view0
        } else {
            let c = lhs_view0.contiguous();
            _lhs_storage = Some(c);
            unsafe {_lhs_storage.as_ref().unwrap_unchecked().view()}
        };

        let mut _rhs_storage = None;
        let rhs_view = if rhs_view0.is_contiguous() {
            rhs_view0
        } else {
            let c = rhs_view0.contiguous();
            _rhs_storage = Some(c);
            unsafe {_rhs_storage.as_ref().unwrap_unchecked().view()}
        };

        let lhs = &lhs_view.meta;
        let rhs = &rhs_view.meta;

        let lr = lhs.rank();
        let rr = rhs.rank();

        if lr != rr || lr < 2 {
            return Err(TensorError::InvalidShape(format!(
                "Both tensors must have the same rank >= 2, got lhs rank {} and rhs rank {}",
                lr, rr
            )));
        }

        // batch dims are all leading dims except the last two
        let lhs_batch_dims: Vec<usize> = lhs.shape.0[..lr - 2].to_vec();
        let rhs_batch_dims: Vec<usize> = rhs.shape.0[..rr - 2].to_vec();

        if lhs_batch_dims != rhs_batch_dims {
            return Err(TensorError::SizeMismatch(format!(
                "Batch dimensions must match for matmul, got lhs batch dims {:?} and rhs batch dims {:?}",
                lhs_batch_dims, rhs_batch_dims
            )));
        }

        let b = if lhs_batch_dims.is_empty() {
            1
        } else {
            lhs_batch_dims.iter().product::<usize>()
        };

        // matrix dims: (..., M, K) @ (..., K, N)
        let m  = lhs.shape[lr - 2];
        let k_l = lhs.shape[lr - 1];
        let k_r = rhs.shape[rr - 2];
        let n  = rhs.shape[rr - 1];

        if k_l != k_r {
            return Err(TensorError::SizeMismatch(format!(
                "Inner matrix dimensions must match for matmul, got lhs K={} and rhs K={}",
                k_l, k_r
            )));
        }

        // -------- Output shape: (batch..., M, N) --------
        let mut out_shape_vec: Vec<Dim> = lhs_batch_dims;
        out_shape_vec.push(m);
        out_shape_vec.push(n);
        let out_shape: Shape = out_shape_vec.into();
        let out_strides = shape_to_stride(&out_shape);

        let buf = lhs_view.backend.matmul(
            lhs_view.buf,
            rhs_view.buf,
            0,  // lhs_offset
            0,  // rhs_offset
            b,
            m,
            k_l,
            n,
        )?;

        Ok(TensorBase::from_parts(
            B::new(),
            buf,
            MetaTensor::new(out_shape, out_strides, 0),
        ))
    }

}

