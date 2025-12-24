

use crate::{backend::{Backend, BackendMatMul}, core::{meta::TensorOffsetIterator, primops::{Exp, InvExp}, tensor::TensorError, value::{types, TensorValue}, MetaTensor}, openblas::{blasint, cblas_dgemm, cblas_sgemm, CBLAS_ORDER, CBLAS_TRANSPOSE}, ops::base::BinaryOpType};
use crate::backend::ContiguityTypes;
use crate::core::value::TypeConstants;

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Cpu;

macro_rules! elemwise_contiguous_loop {
    ($buf:expr, $start:expr, $len:expr, |$x:ident| $body:expr) => {{
        let slice = &mut $buf[$start .. $start + $len];
        for $x in slice.iter_mut() {
            *$x = $body;
        }
    }};
}

macro_rules! elemwise_1d_strided_loop {
    ($buf:expr, $offset:expr, $stride:expr, $len:expr, |$x:ident| $body:expr) => {{
        let mut idx: isize = $offset as isize;
        for _ in 0..$len {
            let $x = &mut $buf[idx as usize];
            *$x = $body;
            idx += $stride;
        }
    }};
}

macro_rules! elemwise_nd_loop {
    ($buf:expr, $offset:expr, $shape:expr, $stride:expr, |$x:ident| $body:expr) => {{
        let iter = TensorOffsetIterator::new(
            $shape,
            $stride,
            $offset,
        );
        for idx in iter {
            let $x = &mut $buf[idx];
            *$x = $body;
        }
    }};
}


macro_rules! impl_cpu_unary {
    ($name:ident, $func:ident $( where $($extra:tt)+ )?) => {
        paste::paste! {
            fn [<apply_ $name _1d_strided>]<T: TensorValue>(
                &self, buf: &mut Self::Buf<T>, 
                    offset: usize,
                    stride: isize,
                    len: usize
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                let bufptr = buf.as_mut();
                elemwise_1d_strided_loop!(bufptr, offset, stride, len, |x| $func(x));
                Ok(())
            }

            fn [<apply_ $name _contiguous>]<T: TensorValue>(
                &self, buf: &mut Self::Buf<T>, 
                    start: usize,
                    len: usize
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                let bufptr = buf.as_mut();
                elemwise_contiguous_loop!(bufptr, start, len, |x| $func(x) );
                Ok(())
            }

            fn [<apply_ $name _nd>]<T: TensorValue>(
                    &self,
                    buf: &mut Self::Buf<T>,
                    offset: usize,
                    shape: &[usize],
                    stride: &[isize],
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                let bufptr = buf.as_mut();
                elemwise_nd_loop!(bufptr, offset, shape, stride, |x| $func(x));
                Ok(())
            }
        }
    };
}


impl Backend for Cpu {
    type Buf<T: TensorValue> = Box<[T]>;


    fn device_type() -> crate::core::primitives::DeviceType {
        crate::core::primitives::DeviceType::Cpu
    }
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Box<[T]>, TensorError> {
        Ok(vec![T::default(); len].into())
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), TensorError> {
        if dst.len() != src.len() {
            return Err(TensorError::SizeMismatch(format!(
                "Buffer size mismatch in copy_from_slice: dst size {}, src size {}",
                dst.len(),
                src.len()
            )));
        }
        dst.copy_from_slice(src);
        Ok(())
    }


    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, TensorError> {
        Ok(*buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset,
                buf.len()
            )),
        )?)
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), TensorError> {
        match buf.get_mut(offset) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset,
                buf.len()
            ))),
        }
    }
    
    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, TensorError> {
        Ok(src)
    }
    
    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
    
    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, TensorError> {
        Ok(src.clone())
    }


    fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &MetaTensor), 
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType
    ) -> Result<(), TensorError> {
        // this is a stupid algorithm which is O(rank*size)
        // it can be optimized to O(size) later
        // a cleaner O(rank*size) algorithm just uses the coordinate iterator
        // and converts the, to full offsets
        let (left_buf, left_meta) = left;
        let (right_buf, right_meta) = right;
        let (dst_buf, dst_meta) = dst;

        let rank = dst_meta.rank();

        let sl = left_meta.strides();
        let sr = right_meta.strides();
        let sd = dst_meta.strides();

        
        let mut ol = left_meta.offset() as isize;
        let mut or = right_meta.offset() as isize;
        let mut od = dst_meta.offset() as isize;

        // println!("Strides: left: {:?}, right: {:?}, dst: {:?}", sl, sr, sd);
        // println!("Offsets: left: {}, right: {}, dst: {}", ol, or, od);

        let mut coords = vec![0; rank];

        let mut first = true;

        for new_coord in dst_meta.iter_coords() {
            // println!("Coords: {:?}", new_coord);
            if first {
                first = false;
            } else{
                for d in (0..rank).rev() {
                    if new_coord[d] != coords[d] {
                        let delta = new_coord[d] as isize - coords[d] as isize;
                        ol += delta * sl[d];
                        or += delta * sr[d];
                        od += delta * sd[d];
                    }
                }
            }
            coords = new_coord;
            debug_assert!(od >= 0);
            debug_assert!(ol >= 0);
            debug_assert!(or >= 0);
            // dst_buf[od as usize] = op.apply(left_buf[ol as usize], right_buf[or as usize]);
            unsafe {
                let lval = (*left_buf)[ol as usize];
                let rval = (*right_buf)[or as usize];
                (*dst_buf)[od as usize] = op.apply(lval, rval);
            }
        }

        Ok(())
    }

    fn apply_elementwise_binary_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (BinaryOpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        elemwise_contiguous_loop!(bufptr, start, len, |x| op.0.apply(*x, op.1));
        Ok(())
    }
    
    fn apply_elementwise_binary_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (BinaryOpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();

        elemwise_1d_strided_loop!(
            bufptr,
            offset,
            stride,
            len,
            |x| op.0.apply(*x, op.1)
        );

        Ok(())
    }
    
    fn apply_elementwise_binary_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (BinaryOpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        elemwise_nd_loop!(
            bufptr,
            offset,
            shape,
            stride,
            |x| op.0.apply(*x, op.1)
        );
        Ok(())
    }
    
    

    impl_cpu_unary!{ neg, _negate where T: std::ops::Neg<Output = T> }
    impl_cpu_unary!{ relu, _relu }
    impl_cpu_unary!{ sigmoid, _sigmoid where T: InvExp}
    impl_cpu_unary!{ tanh, _tanh where T: Exp + InvExp }

}

#[inline]
fn _tanh<T: TensorValue + InvExp + Exp>(x: &mut T) -> T {
    let a = x.apply_exp();
    let b = x.apply_invexp();
    (a - b) / (a + b)
}


#[inline]
fn _negate<T: TensorValue + std::ops::Neg<Output = T>>(x: &mut T) -> T {
    -*x
}


#[inline]
fn _relu<T: TensorValue>(x: &mut T) -> T {
    if *x > T::ZERO {
        *x
    } else {
        T::ZERO
    }
}

/// The sigmoid function, implemented based
/// on a quick google search.
#[inline]
fn _sigmoid<T: TensorValue>(x: &mut T) -> T
where 
    T: InvExp
{
    T::ONE / (T::ONE + x.apply_invexp())
}

macro_rules! blas_impl {
    ($t:ty, $gemm_fn:ident) => {
        impl BackendMatMul<$t> for Cpu {
            fn matmul(
                &self,
                lhs: (&Self::Buf<$t>, &MetaTensor),
                rhs: (&Self::Buf<$t>, &MetaTensor),
                dst: &mut Self::Buf<$t>,
                b: usize,
                m: usize,
                k: usize,
                n: usize,
                contiguity: ContiguityTypes
            ) -> Result<(), TensorError> {

                let (lhs_buf, lhs_meta): (&Self::Buf<$t>, &MetaTensor) = lhs;
                let (rhs_buf, rhs_meta): (&Self::Buf<$t>, &MetaTensor) = rhs;
                let lda = match &contiguity {
                    ContiguityTypes::ColumnMajor => lhs_meta.strides()[lhs_meta.rank() - 1] as blasint,
                    ContiguityTypes::RowMajor => lhs_meta.strides()[lhs_meta.rank() - 2] as blasint,
                    _ => panic!("Invalid contiguity for BLAS matmul")
                };

                let ldb = match &contiguity {
                    ContiguityTypes::ColumnMajor => rhs_meta.strides()[rhs_meta.rank() - 1] as blasint,
                    ContiguityTypes::RowMajor => rhs_meta.strides()[rhs_meta.rank() - 2] as blasint,
                    _ => panic!("Invalid contiguity for BLAS matmul")
                };
                let ldc = n as blasint; // row major

                let bstride_lhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides()[lhs_meta.rank() - 3] as usize
                } else {
                    m * k
                };
                
                let bstride_rhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides()[rhs_meta.rank() - 3] as usize
                } else {
                    k * n
                };
                
                let (order, trans, m, n, lda, ldb, lhs, rhs) = match contiguity {
                    ContiguityTypes::RowMajor => (
                        CBLAS_ORDER::CblasRowMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        lda,
                        ldb,
                        lhs_buf.as_ptr(),
                        rhs_buf.as_ptr(),
                    ),
                    ContiguityTypes::ColumnMajor => (
                        CBLAS_ORDER::CblasColMajor,
                        CBLAS_TRANSPOSE::CblasTrans,
                        m,
                        n,
                        ldb,
                        lda,
                        rhs_buf.as_ptr(),
                        lhs_buf.as_ptr(),
                    ),
                    _ => {
                        panic!("Invalid contiguity for BLAS matmul")
                    }
                };

                for batch in 0..b {
                    // base pointers
                    let lhs_batch = lhs_meta.offset + batch * bstride_lhs;
                    let rhs_batch = rhs_meta.offset + batch * bstride_rhs;

                    let out_batch = batch * m * n; // contiguous 0 offset

                    unsafe {
                        $gemm_fn(
                            order,
                            trans,
                            trans,
                            m as blasint,
                            n as blasint,
                            k as blasint,
                            1.0,
                            lhs.add(lhs_batch) as *const $t,
                            lda,
                            rhs.add(rhs_batch) as *const $t,
                            ldb,
                            0.0,
                            dst.as_mut_ptr().add(out_batch) as *mut $t,
                            ldc,
                        );
                    }
                }

                Ok(())
            }
        }
    };
}

macro_rules! generic_backend_matmul {
    ($t:ty) => {
        impl BackendMatMul<$t> for Cpu {
            fn matmul(
                &self,
                lhs: (&Self::Buf<$t>, &MetaTensor),
                rhs: (&Self::Buf<$t>, &MetaTensor),
                dst: &mut Self::Buf<$t>,
                b: usize,
                m: usize,
                k: usize,
                n: usize,
                contiguity: ContiguityTypes
            ) -> Result<(), TensorError> {
                // let mut out_buf = self.alloc(b * m * n)?;
                let (lhs_buf, lhs_meta): (&Self::Buf<$t>, &MetaTensor) = lhs;
                let (rhs_buf, rhs_meta): (&Self::Buf<$t>, &MetaTensor) = rhs;
                let lda = match contiguity {
                    ContiguityTypes::ColumnMajor => lhs_meta.strides[lhs_meta.rank() - 1] as usize,
                    ContiguityTypes::RowMajor => lhs_meta.strides[lhs_meta.rank() - 2] as usize,
                    _ => panic!("Invalid contiguity for generic matmul")
                };
                let ldb = match contiguity {
                    ContiguityTypes::ColumnMajor => rhs_meta.strides[rhs_meta.rank() - 1] as usize,
                    ContiguityTypes::RowMajor => rhs_meta.strides[rhs_meta.rank() - 2] as usize,
                    _ => panic!("Invalid contiguity for generic matmul")
                };

                let bstride_lhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides[lhs_meta.rank() - 3] as usize
                } else {
                    0 // only 1 batch, we won't stride
                };

                let bstride_rhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides[rhs_meta.rank() - 3] as usize
                } else {
                    0 // only 1 batch, we won't stride
                };

                for batch in 0..b {
                    let lhs_batch = lhs_meta.offset + batch * bstride_lhs;
                    let rhs_batch = rhs_meta.offset + batch * bstride_rhs;
                    let out_batch = batch * m * n;
                    // this is repeated code, yes, but we want to reduce indirection in the inner loop
                    // as this is a hot path. furthermore, branching in the inner loop will reduce chances of vectorization
                    if contiguity == ContiguityTypes::RowMajor {
                        for row in 0..m {
                            for col in 0..n {
                                let mut sum: $t = <$t>::ZERO;
                                for inner in 0..k {
                                    let lhs_idx = lhs_batch + row * lda + inner;
                                    let rhs_idx = rhs_batch + inner * ldb + col;
                                    sum += lhs_buf[lhs_idx] * rhs_buf[rhs_idx];
                                }
                                dst[out_batch + row * n + col] = sum;
                            }
                        }
                    }else {
                        for row in 0..m {
                            for col in 0..n {
                                let mut sum: $t = <$t>::ZERO;
                                for inner in 0..k {
                                    let lhs_idx = lhs_batch + row + inner * lda;
                                    let rhs_idx = rhs_batch + inner + col * ldb;
                                    sum += lhs_buf[lhs_idx] * rhs_buf[rhs_idx];
                                }
                                dst[out_batch + row * n + col] = sum;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
        
    };
}

// instead of specialization
blas_impl!(f32, cblas_sgemm);
blas_impl!(f64, cblas_dgemm);
generic_backend_matmul!(i8);
generic_backend_matmul!(i16);
generic_backend_matmul!(i32);
generic_backend_matmul!(i64);
generic_backend_matmul!(i128);
generic_backend_matmul!(u8);
generic_backend_matmul!(u16);
generic_backend_matmul!(u32);
generic_backend_matmul!(u64);
generic_backend_matmul!(u128);
generic_backend_matmul!(types::boolean);

#[cfg(test)]
mod tests {
    use crate::openblas::*;
    use std::ffi::CStr;

    #[test]
    fn test_openblas_info() {
        unsafe {
            // Get OpenBLAS information
            let config = openblas_get_config();
            let config_str = CStr::from_ptr(config).to_string_lossy();
            println!("OpenBLAS Config: {}", config_str);
            
            let corename = openblas_get_corename();
            let corename_str = CStr::from_ptr(corename).to_string_lossy();
            println!("OpenBLAS Core: {}", corename_str);
            
            let num_procs = openblas_get_num_procs();
            println!("Number of processors: {}", num_procs);
            assert!(num_procs > 0);
            
            let num_threads = openblas_get_num_threads();
            println!("Number of threads: {}", num_threads);
            assert!(num_threads > 0);
        }
    }

    #[test]
    fn test_openblas_set_threads() {
        unsafe {
            let original_threads = openblas_get_num_threads();
            
            // Set to 4 threads
            openblas_set_num_threads(4);
            assert_eq!(openblas_get_num_threads(), 4);
            
            // Restore original
            openblas_set_num_threads(original_threads);
            assert_eq!(openblas_get_num_threads(), original_threads);
        }
    }

    #[test]
    fn test_cblas_dot_product() {
        unsafe {
            // Test single precision dot product
            let x = vec![1.0f32, 2.0, 3.0, 4.0];
            let y = vec![5.0f32, 6.0, 7.0, 8.0];
            
            let result = cblas_sdot(
                x.len() as blasint,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1
            );
            
            // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
            let expected = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>();
            assert_eq!(result, expected);
            assert_eq!(result, 70.0);
        }
    }

    #[test]
    fn test_cblas_dot_product_double() {
        unsafe {
            // Test double precision dot product
            let x = vec![1.0f64, 2.0, 3.0, 4.0];
            let y = vec![5.0f64, 6.0, 7.0, 8.0];
            
            let result = cblas_ddot(
                x.len() as blasint,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1
            );
            
            let expected = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
            assert_eq!(result, expected);
            assert_eq!(result, 70.0);
        }
    }

    #[test]
    fn test_cblas_gemv() {
        unsafe {
            // Matrix-vector multiplication: y = A * x
            // A is 2x3, x is 3x1, result should be 2x1
            #[rustfmt::skip]
            let a = vec![
                1.0f32, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ];
            let x = vec![1.0f32, 2.0, 3.0];
            let mut y = vec![0.0f32, 0.0];
            
            cblas_sgemv(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                2,  // m: number of rows in A
                3,  // n: number of columns in A
                1.0,  // alpha
                a.as_ptr(),
                3,  // lda: leading dimension of A
                x.as_ptr(),
                1,  // incx
                0.0,  // beta
                y.as_mut_ptr(),
                1,  // incy
            );
            
            // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
            assert_eq!(y[0], 14.0);
            assert_eq!(y[1], 32.0);
        }
    }

    #[test]
    fn test_cblas_gemm() {
        unsafe {
            // Matrix-matrix multiplication: C = A * B
            // A is 2x3, B is 3x2, C should be 2x2
            #[rustfmt::skip]
            let a = vec![
                1.0f32, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ];
            #[rustfmt::skip]
            let b = vec![
                7.0f32, 8.0,
                9.0, 10.0,
                11.0, 12.0,
            ];
            let mut c = vec![0.0f32; 4];
            
            cblas_sgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                2,  // m: rows in A and C
                2,  // n: columns in B and C
                3,  // k: columns in A, rows in B
                1.0,  // alpha
                a.as_ptr(),
                3,  // lda
                b.as_ptr(),
                2,  // ldb
                0.0,  // beta
                c.as_mut_ptr(),
                2,  // ldc
            );
            
            // Expected:
            // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
            // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
            // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
            // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
            assert_eq!(c[0], 58.0);
            assert_eq!(c[1], 64.0);
            assert_eq!(c[2], 139.0);
            assert_eq!(c[3], 154.0);
        }
    }

    #[test]
    fn test_cblas_gemm_double() {
        unsafe {
            // Test double precision matrix multiplication
            #[rustfmt::skip]
            let a = vec![
                1.0f64, 2.0,
                3.0, 4.0,
            ];
            #[rustfmt::skip]
            let b = vec![
                5.0f64, 6.0,
                7.0, 8.0,
            ];
            let mut c = vec![0.0f64; 4];
            
            cblas_dgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                2,  // m
                2,  // n
                2,  // k
                1.0,  // alpha
                a.as_ptr(),
                2,  // lda
                b.as_ptr(),
                2,  // ldb
                0.0,  // beta
                c.as_mut_ptr(),
                2,  // ldc
            );
            
            // Expected:
            // C[0,0] = 1*5 + 2*7 = 19
            // C[0,1] = 1*6 + 2*8 = 22
            // C[1,0] = 3*5 + 4*7 = 43
            // C[1,1] = 3*6 + 4*8 = 50
            assert_eq!(c[0], 19.0);
            assert_eq!(c[1], 22.0);
            assert_eq!(c[2], 43.0);
            assert_eq!(c[3], 50.0);
        }
    }
}
