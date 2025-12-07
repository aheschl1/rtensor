use std::sync::{atomic::{AtomicBool, Ordering}, Arc, LazyLock};

use cudarc::{cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig}, driver::{CudaContext, CudaSlice, DevicePtr}};

use crate::{backend::{Backend, BackendMatMul}, core::{tensor::TensorError, value::TensorValue, MetaTensor}, ops::base::OpType};

// Include bindgen-generated FFI declarations for CUDA kernel launchers
#[allow(non_camel_case_types)]
mod bindings{
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use bindings::*;

// Can be tuned based on kernel characteristics and GPU architecture
const DEFAULT_BLOCK_SIZE: u32 = 256;

const CUDA_BACKENDS: LazyLock<Vec<Cuda>> = LazyLock::new(|| {
    let mut backends = Vec::new();
    let device_count = cudarc::driver::CudaContext::device_count().unwrap_or(0);
    for device_id in 0..device_count {
        let ctx = CudaContext::new(device_id as usize).unwrap();
        let backend = Cuda { 
            ctx: ctx.clone(), 
            cublas: Arc::new(CudaBlas::new(ctx.default_stream()).unwrap()),
            dirty: AtomicBool::new(false).into(),
        };
        backends.push(backend);
    }
    backends
});


pub struct CudaBuf<T: TensorValue> {
    pub(crate) ptr: CudaSlice<T>,
    pub(crate) len: usize,
}

#[derive(Clone)]
pub struct Cuda {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) cublas: Arc<CudaBlas>,
    dirty: Arc<AtomicBool>,
}


impl Cuda {
    fn stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.ctx.default_stream()
    }

    pub(crate) fn construct(device: usize) -> Result<Self, TensorError> {
        // TODO multiple devices
        let backend = CUDA_BACKENDS.get(device)
            .ok_or_else(|| TensorError::CudaError(format!("CUDA device {} not found", device)))?
            .clone();
        Ok( backend )
    }

    pub fn flush(&self) -> Result<(), TensorError> {
        if self.dirty.swap(false, Ordering::AcqRel){
            // it was dirty, now its clean
            self.stream()
                .synchronize()
                .map_err(|e| TensorError::CudaError(e.to_string()))?;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn sync(&self) -> Result<(), TensorError> {
        if self.dirty.load(Ordering::Acquire){
            self.flush()?;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

impl<T: TensorValue> Backend<T> for Cuda {
    type Buf = CudaBuf<T>;
    
    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        let ptr = self.stream()
            .clone_htod(src.as_ref())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(CudaBuf { 
            ptr, 
            len: src.len(),
        })
    }
    
    fn alloc(&self, len: usize) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        let ptr;
        unsafe{
            ptr = self.stream()
                .alloc::<T>(len)
                .map_err(|e| TensorError::CudaError(e.to_string()))?;
        }
        Ok(CudaBuf { 
            ptr, 
            len, 
        })
    }
    
    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        self.sync()?;
        if src.len() != dst.len {
            return Err(TensorError::CudaError(format!("Source slice length {} does not match destination buffer length {}", src.len(), dst.len)));
        }

        self.stream()
            .memcpy_htod(src.as_ref(), &mut dst.ptr)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(())
    }
    
    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        self.sync()?;
        if offset >= buf.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset,
                buf.len
            )));
        }

        let mut host_buf = vec![T::default(); 1];
        self.stream()
            .memcpy_dtoh(&buf.ptr.slice(offset..offset+1), host_buf.as_mut_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(host_buf[0])
    }
    
    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        self.sync()?;
        if offset >= buf.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset,
                buf.len
            )));
        }

        self.stream()
            .memcpy_htod(&[value], &mut buf.ptr.slice_mut(offset..offset+1))
            .map_err(|e| TensorError::CudaError(e.to_string()))?;

        Ok(())
    }
    
    fn len(&self, buf: &Self::Buf) -> usize {
        buf.len
    }
    
    fn new() -> Self {
        // TODO multiple devices
        Self::construct(0).unwrap()
    }
    
    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, TensorError> {
        self.sync()?;
        let mut dst = self.alloc(src.len)?;
        self.stream()
            .memcpy_dtod(&src.ptr, &mut dst.ptr)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(dst)
    }
    
    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, TensorError> {
        self.sync()?;
        let mut host_buf = vec![T::default(); src.len];
        self.stream()
            .memcpy_dtoh(&src.ptr, host_buf.as_mut_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(host_buf.into_boxed_slice())
    }


        fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        
        let op_code = op.0.to_op_code();
        let value = op.1;
        let stream = self.stream();

        macro_rules! launch_elementwise {
            ($launch_fn:ident, $t:ty) => {{
                // transmute value from T to actual type
                let concrete_value: $t = unsafe { std::mem::transmute_copy(&value) };
                
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;
                let data_ptr = unsafe { data_ptr.add(start) };

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        len,
                        op_code,
                        concrete_value,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type
        let tid = std::any::TypeId::of::<T>();
        match tid {
            id if id == std::any::TypeId::of::<f32>() => launch_elementwise!(launch_elementwise_contiguous_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_elementwise!(launch_elementwise_contiguous_f64, f64),
            id if id == std::any::TypeId::of::<u8>() => launch_elementwise!(launch_elementwise_contiguous_u8, u8),
            id if id == std::any::TypeId::of::<u16>() => launch_elementwise!(launch_elementwise_contiguous_u16, u16),
            id if id == std::any::TypeId::of::<u32>() => launch_elementwise!(launch_elementwise_contiguous_u32, u32),
            id if id == std::any::TypeId::of::<u64>() => launch_elementwise!(launch_elementwise_contiguous_u64, u64),
            id if id == std::any::TypeId::of::<u128>() => launch_elementwise!(launch_elementwise_contiguous_u128, u128),
            id if id == std::any::TypeId::of::<i8>() => launch_elementwise!(launch_elementwise_contiguous_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_elementwise!(launch_elementwise_contiguous_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_elementwise!(launch_elementwise_contiguous_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_elementwise!(launch_elementwise_contiguous_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_elementwise!(launch_elementwise_contiguous_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
        }
    }
    
    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: (OpType, T), 
        start: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {

        let op_code = op.0.to_op_code();
        let value = op.1;
        let stream = self.stream();

        macro_rules! launch_elementwise {
            ($launch_fn:ident, $t:ty) => {{
                // transmute value from T to actual type
                let concrete_value: $t = unsafe { std::mem::transmute_copy(&value) };
                let (data_ptr, _) = buf.ptr.device_ptr(&stream);
                
                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        start,
                        stride,
                        len,
                        op_code,
                        concrete_value,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_elementwise!(launch_elementwise_strided_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_elementwise!(launch_elementwise_strided_f64, f64),
            id if id == std::any::TypeId::of::<u8>() => launch_elementwise!(launch_elementwise_strided_u8, u8),
            id if id == std::any::TypeId::of::<u16>() => launch_elementwise!(launch_elementwise_strided_u16, u16),
            id if id == std::any::TypeId::of::<u32>() => launch_elementwise!(launch_elementwise_strided_u32, u32),
            id if id == std::any::TypeId::of::<u64>() => launch_elementwise!(launch_elementwise_strided_u64, u64),
            id if id == std::any::TypeId::of::<u128>() => launch_elementwise!(launch_elementwise_strided_u128, u128),
            id if id == std::any::TypeId::of::<i8>() => launch_elementwise!(launch_elementwise_strided_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_elementwise!(launch_elementwise_strided_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_elementwise!(launch_elementwise_strided_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_elementwise!(launch_elementwise_strided_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_elementwise!(launch_elementwise_strided_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
        }
    }
    
    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: (OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let op_code = op.0.to_op_code();
        let value = op.1;
        let stream = self.stream();
        let rank = shape.len();
        let size = shape.iter().product::<usize>();
        // we need ptr to stridde and shape
        let shape_buf = self.alloc_from_slice(shape.to_vec().into_boxed_slice())?;
        let stride_buf = self.alloc_from_slice(stride.to_vec().into_boxed_slice())?;

        let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
        let (shape_ptr, _) = shape_buf.ptr.device_ptr(&stream);

        macro_rules! launch_elementwise {
            ($launch_fn:ident, $t:ty) => {{
                // transmute value from T to actual type
                let concrete_value: $t = unsafe { std::mem::transmute_copy(&value) };
                let (data_ptr, _) = buf.ptr.device_ptr(&stream);
                
                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride_ptr as *const isize,
                        shape_ptr as *const usize,
                        rank,
                        size,
                        op_code,
                        concrete_value,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_elementwise!(launch_elementwise_nd_affine_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_elementwise!(launch_elementwise_nd_affine_f64, f64),
            id if id == std::any::TypeId::of::<u8>() => launch_elementwise!(launch_elementwise_nd_affine_u8, u8),
            id if id == std::any::TypeId::of::<u16>() => launch_elementwise!(launch_elementwise_nd_affine_u16, u16),
            id if id == std::any::TypeId::of::<u32>() => launch_elementwise!(launch_elementwise_nd_affine_u32, u32),
            id if id == std::any::TypeId::of::<u64>() => launch_elementwise!(launch_elementwise_nd_affine_u64, u64),
            id if id == std::any::TypeId::of::<u128>() => launch_elementwise!(launch_elementwise_nd_affine_u128, u128),
            id if id == std::any::TypeId::of::<i8>() => launch_elementwise!(launch_elementwise_nd_affine_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_elementwise!(launch_elementwise_nd_affine_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_elementwise!(launch_elementwise_nd_affine_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_elementwise!(launch_elementwise_nd_affine_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_elementwise!(launch_elementwise_nd_affine_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
        }
    }

    unsafe fn broadcast(
        &self, 
        left: (*const Self::Buf, &MetaTensor), 
        right: (*const Self::Buf, &MetaTensor),
        dst: (*mut Self::Buf, &MetaTensor),
        op: OpType
    ) -> Result<(), TensorError> {
        let (lbuf, lmeta) = left;
        let (rbuf, rmeta) = right;
        let (dbuf, dmeta) = dst;

        let op_code = op.to_op_code();
        let stream = self.stream();
        
        let rank = dmeta.shape().len();
        let size = dmeta.shape().iter().product::<usize>();
        
        // Allocate device memory for strides and shapes
        // let lshape_buf = self.alloc_from_slice(lmeta.shape.0.clone().into_boxed_slice())?;
        // let rshape_buf = self.alloc_from_slice(rmeta.shape.0.clone().into_boxed_slice())?;
        let dshape_buf = self.alloc_from_slice(dmeta.shape.0.clone().into_boxed_slice())?;
        
        let lstride_buf = self.alloc_from_slice(lmeta.strides.0.clone().into_boxed_slice())?;
        let rstride_buf = self.alloc_from_slice(rmeta.strides.0.clone().clone().into_boxed_slice())?;
        let dstride_buf = self.alloc_from_slice(dmeta.strides.0.clone().into_boxed_slice())?;

        let (lstride_ptr, _) = lstride_buf.ptr.device_ptr(&stream);
        let (rstride_ptr, _) = rstride_buf.ptr.device_ptr(&stream);
        let (dstride_ptr, _) = dstride_buf.ptr.device_ptr(&stream);
        // let (lshape_ptr, _) = lshape_buf.ptr.device_ptr(&stream);
        // let (rshape_ptr, _) = rshape_buf.ptr.device_ptr(&stream);
        let (dshape_ptr, _) = dshape_buf.ptr.device_ptr(&stream);

        let loff = lmeta.offset();
        let roff = rmeta.offset();
        let doff = dmeta.offset();

        macro_rules! launch_broadcast {
            ($launch_fn:ident, $t:ty) => {{
                // let (lbuf_ptr, _) = lbuf.ptr.device_ptr(&stream);
                // let (rbuf_ptr, _) = rbuf.ptr.device_ptr(&stream);
                // let (dbuf_ptr, _) = dbuf.ptr.device_ptr(&stream);
                let lbuf_ptr = unsafe { (*lbuf).ptr.device_ptr(&stream).0 };
                let rbuf_ptr = unsafe { (*rbuf).ptr.device_ptr(&stream).0 };
                let dbuf_ptr = unsafe { (*dbuf).ptr.device_ptr(&stream).0 };
                
                unsafe {
                    $launch_fn(
                        lbuf_ptr as *const $t,
                        rbuf_ptr as *const $t,
                        dbuf_ptr as *mut $t,
                        loff,
                        roff,
                        doff,
                        rank,
                        size,
                        lstride_ptr as *const isize,
                        rstride_ptr as *const isize,
                        dstride_ptr as *const isize,
                        dshape_ptr as *const usize,
                        op_code,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_broadcast!(launch_binary_broadcast_elementwise_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_broadcast!(launch_binary_broadcast_elementwise_f64, f64),
            id if id == std::any::TypeId::of::<u8>() => launch_broadcast!(launch_binary_broadcast_elementwise_u8, u8),
            id if id == std::any::TypeId::of::<u16>() => launch_broadcast!(launch_binary_broadcast_elementwise_u16, u16),
            id if id == std::any::TypeId::of::<u32>() => launch_broadcast!(launch_binary_broadcast_elementwise_u32, u32),
            id if id == std::any::TypeId::of::<u64>() => launch_broadcast!(launch_binary_broadcast_elementwise_u64, u64),
            id if id == std::any::TypeId::of::<u128>() => launch_broadcast!(launch_binary_broadcast_elementwise_u128, u128),
            id if id == std::any::TypeId::of::<i8>() => launch_broadcast!(launch_binary_broadcast_elementwise_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_broadcast!(launch_binary_broadcast_elementwise_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_broadcast!(launch_binary_broadcast_elementwise_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_broadcast!(launch_binary_broadcast_elementwise_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_broadcast!(launch_binary_broadcast_elementwise_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA broadcast operation".to_string())),
        }
    }
}

macro_rules! generic_matmul_impl {
    ($t:ty, $launch_fn:ident) => {
        impl BackendMatMul<$t> for Cuda {
            fn matmul(
                &self,
                lhs: (&Self::Buf, &MetaTensor),
                rhs: (&Self::Buf, &MetaTensor),
                b: usize,
                m: usize,
                k: usize,
                n: usize,
            ) -> Result<Self::Buf, TensorError> {
                let stream = self.stream();
                let res = self.alloc(b * m * n)?;
                
                let (lhs_buf, lhs_meta) = lhs;
                let (rhs_buf, rhs_meta) = rhs;

                let (lhs_ptr, _) = lhs_buf.ptr.device_ptr(&stream);
                let (rhs_ptr, _) = rhs_buf.ptr.device_ptr(&stream);
                let (res_ptr, _) = res.ptr.device_ptr(&stream);
                
                let bstride_lhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides()[lhs_meta.rank() - 3] as usize
                } else{
                    m * k
                };

                let bstride_rhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides()[rhs_meta.rank() - 3] as usize
                } else{
                    k * n
                };

                let lda = lhs_meta.strides()[lhs_meta.rank() - 2] as usize;
                let ldb = rhs_meta.strides()[rhs_meta.rank() - 2] as usize;
                let ldc = n; // row-major
                // For batched matrix multiplication
                for batch_idx in 0..b {
                    let a_offset = lhs_meta.offset + batch_idx * bstride_lhs;
                    let b_offset = rhs_meta.offset + batch_idx * bstride_rhs;
                    let c_offset = batch_idx * m * n;
                    
                    let a_ptr = unsafe { (lhs_ptr as *const $t).add(a_offset) };
                    let b_ptr = unsafe { (rhs_ptr as *const $t).add(b_offset) };
                    let c_ptr = unsafe { (res_ptr as *mut $t).add(c_offset) };
                    
                    unsafe {
                        $launch_fn(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            m,
                            n,
                            k,
                            lda,
                            ldb,
                            ldc,
                            DEFAULT_BLOCK_SIZE,
                        );
                    }
                }
                
                self.dirty();
                Ok(res)
            }
        }
    };
}

macro_rules! cublas_impl {
    ($t:ty) => {
        impl BackendMatMul<$t> for Cuda {
            fn matmul(
                &self,
                lhs: (&Self::Buf, &MetaTensor),
                rhs: (&Self::Buf, &MetaTensor),
                b: usize,
                m: usize,
                k: usize,
                n: usize,
            ) -> Result<Self::Buf, TensorError> {
                // cuBLAS uses column-major order, but our tensors are row-major
                // To compute C = A * B in row-major, we compute C^T = B^T * A^T in column-major
                // This means we swap A and B, and swap m and n
                let (m, n) = (n, m);
                
                let (lhs_buf, lhs_meta) = lhs;
                let (rhs_buf, rhs_meta) = rhs;

                let lda = lhs_meta.strides()[rhs_meta.rank() - 2] as i32; // flipped 
                let ldb = rhs_meta.strides()[lhs_meta.rank() - 2] as i32;
                let ldc = m as i32;

                let bstride_lhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides()[lhs_meta.rank() - 3] as usize
                } else{
                    m as usize * k as usize
                };

                let bstride_rhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides()[rhs_meta.rank() - 3] as usize
                } else{
                    k as usize * n as usize
                };

                let cfg = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: m as i32,
                    n: n as i32,
                    k: k as i32,
                    alpha: 1.0,
                    lda: lda,  // leading dimension of B (now first operand)
                    ldb: ldb,  // leading dimension of A (now second operand)
                    beta: 0.0,
                    ldc: ldc,  // leading dimension of C
                };
                let cfg = StridedBatchedConfig {
                    gemm: cfg,
                    batch_size: b as i32,
                    stride_a: bstride_rhs as i64,  // stride for B
                    stride_b: bstride_lhs as i64,  // stride for A
                    stride_c: (n*m) as i64,
                };
                let mut res = self.alloc(b*n*m)?;

                unsafe{
                    // Note: operands are swapped (B, A instead of A, B)
                    self.cublas.gemm_strided_batched(
                        cfg, 
                        &rhs_buf.ptr.slice(rhs_meta.offset..),  // B comes first
                        &lhs_buf.ptr.slice(lhs_meta.offset..),  // A comes second
                        &mut res.ptr,
                    ).map_err(|e| TensorError::CudaError(e.to_string()))?;
                }
                self.dirty();
                Ok(res)
            }
        }
    };
}

cublas_impl!(f32);
cublas_impl!(f64);

generic_matmul_impl!(u8, launch_matmul_u8);
generic_matmul_impl!(u16, launch_matmul_u16);
generic_matmul_impl!(u32, launch_matmul_u32);
generic_matmul_impl!(u64, launch_matmul_u64);
generic_matmul_impl!(u128, launch_matmul_u128);
generic_matmul_impl!(i8, launch_matmul_i8);
generic_matmul_impl!(i16, launch_matmul_i16);
generic_matmul_impl!(i32, launch_matmul_i32);
generic_matmul_impl!(i64, launch_matmul_i64);
generic_matmul_impl!(i128, launch_matmul_i128);