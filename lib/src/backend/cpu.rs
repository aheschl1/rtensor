
use crate::{backend::Backend, core::{meta::TensorOffsetIterator, tensor::TensorError, value::TensorValue, MetaTensor}, openblas::{blasint, cblas_dgemm, cblas_sgemm, CBLAS_ORDER, CBLAS_TRANSPOSE}, ops::base::OpType};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Cpu;

impl<T: TensorValue> Backend<T> for Cpu {
    type Buf = Box<[T]>;



    fn alloc(&self, len: usize) -> Result<Box<[T]>, TensorError> {
        Ok(vec![T::default(); len].into())
    }

    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), TensorError> {
        if dst.len() != src.len() {
            return Err(TensorError::SizeMismatch);
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, TensorError> {
        Ok(*buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds
        )?)
    }

    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), TensorError> {
        match buf.get_mut(offset) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(TensorError::IdxOutOfBounds),
        }
    }
    
    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, TensorError> {
        Ok(src)
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }

    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
    
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError> {
        Ok(src.clone())
    }


        fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        for item in bufptr.iter_mut().skip(start).take(len) {
            *item = op.0.apply(*item, op.1);
        }
        Ok(())
    }
    
    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let mut idx: isize = offset as isize;
        for _ in 0..len {
            bufptr[idx as usize] = op.0.apply(bufptr[idx as usize], op.1);
            idx += stride;
        }
        Ok(())
    }
    
    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: (OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let bufptr = buf.as_mut();
        let iterator = TensorOffsetIterator::new(
            shape,
            stride,
            offset,
        );
        for idx in iterator {
            bufptr[idx] = op.0.apply(bufptr[idx], op.1);
        }
        Ok(())
    }

    unsafe fn broadcast(
        &self, 
        left: (*const Self::Buf, &MetaTensor), 
        right: (*const Self::Buf, &MetaTensor),
        dst: (*mut Self::Buf, &MetaTensor),
        op: OpType
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
    
    fn matmul_generic(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError>{
        let mut out_buf = self.alloc(b * m * n)?;

        for batch in 0..b {
            let lhs_batch = lhs_offset + batch * m * k;
            let rhs_batch = rhs_offset + batch * k * n;
            let out_batch = batch * m * n;

            for row in 0..m {
                for col in 0..n {
                    let mut sum = T::default();
                    for inner in 0..k {
                        let lhs_idx = lhs_batch + row * k + inner;
                        let rhs_idx = rhs_batch + inner * n + col;
                        sum = sum + lhs_buf[lhs_idx] * rhs_buf[rhs_idx];
                    }
                    out_buf[out_batch + row * n + col] = sum;
                }
            }
        }
        Ok(out_buf)
    }
    
    fn matmul_float32(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError> {
        let mut out_buf: Box<[T]> = self.alloc(b * m * n)?;

        for batch in 0..b {
            let lhs_batch = lhs_offset + batch * m * k;
            let rhs_batch = rhs_offset + batch * k * n;
            let out_batch = batch * m * n;

            unsafe {
                cblas_sgemm(
                    CBLAS_ORDER::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as blasint,
                    n as blasint,
                    k as blasint,
                    1.0,
                    lhs_buf.as_ptr().add(lhs_batch) as *const f32,
                    k as blasint,
                    rhs_buf.as_ptr().add(rhs_batch) as *const f32,
                    n as blasint,
                    0.0,
                    out_buf.as_mut_ptr().add(out_batch) as *mut f32,
                    n as blasint,
                );
            }
        }

        Ok(out_buf)
    }
    
    fn matmul_float64(
        &self,
        lhs_buf: &Self::Buf,
        rhs_buf: &Self::Buf,
        lhs_offset: usize,
        rhs_offset: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::Buf, TensorError> {
        let mut out_buf: Box<[T]> = self.alloc(b * m * n)?;

        for batch in 0..b {
            let lhs_batch = lhs_offset + batch * m * k;
            let rhs_batch = rhs_offset + batch * k * n;
            let out_batch = batch * m * n;

            unsafe {
                cblas_dgemm(
                    CBLAS_ORDER::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as blasint,
                    n as blasint,
                    k as blasint,
                    1.0,
                    lhs_buf.as_ptr().add(lhs_batch) as *const f64,
                    k as blasint,
                    rhs_buf.as_ptr().add(rhs_batch) as *const f64,
                    n as blasint,
                    0.0,
                    out_buf.as_mut_ptr().add(out_batch) as *mut f64,
                    n as blasint,
                );
            }
        }

        Ok(out_buf)
    }

}

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
