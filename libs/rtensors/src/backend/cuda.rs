use std::sync::{atomic::{AtomicBool, Ordering}, Arc, LazyLock};

use cudarc::{cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig}, driver::{CudaContext, CudaSlice, DevicePtr}};

use crate::{backend::{cpu::Cpu, Backend, BackendMatMul}, core::{primitives::TensorBase, primops::{Exp, InvExp}, tensor::TensorError, value::{types, TensorValue}, MetaTensor, Tensor}, ops::base::BinaryOpType};
use crate::backend::ContiguityTypes;

// Include bindgen-generated FFI declarations for CUDA kernel launchers
#[allow(non_camel_case_types)]
mod bindings{
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
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
                todo!()
            }

            fn [<apply_ $name _contiguous>]<T: TensorValue>(
                &self, buf: &mut Self::Buf<T>, 
                    start: usize,
                    len: usize
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                todo!()
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
                todo!()
            }
        }
    };
}

impl Backend for Cuda {
    type Buf<T: TensorValue> = CudaBuf<T>;
    
    fn device_type() -> crate::core::primitives::DeviceType {
        crate::core::primitives::DeviceType::Cuda(0)
    }

    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let ptr = self.stream()
            .clone_htod(src.as_ref())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(CudaBuf { 
            ptr, 
            len: src.len(),
        })
    }

   
    
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
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
    
    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        self.sync()?;
        if src.len() != dst.len {
            return Err(TensorError::CudaError(format!("Source slice length {} does not match destination buffer length {}", src.len(), dst.len)));
        }

        self.stream()
            .memcpy_htod(src.as_ref(), &mut dst.ptr)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(())
    }

    fn copy_range_within<T: TensorValue>(
        &self, 
        buf: &mut Self::Buf<T>, 
        src: &Self::Buf<T>, 
        dst_offset: usize, 
        src_offset: usize, 
        len: usize
    ) -> Result<(), TensorError> {
        self.sync()?;
        if src_offset + len > src.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Source range (offset {} + length {}) out of bounds for buffer of length {}",
                src_offset,
                len,
                src.len
            )));
        }
        if dst_offset + len > buf.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Destination range (offset {} + length {}) out of bounds for buffer of length {}",
                dst_offset,
                len,
                buf.len
            )));
        }

        self.stream()
            .memcpy_dtod(
                &src.ptr.slice(src_offset..src_offset + len), 
                &mut buf.ptr.slice_mut(dst_offset..dst_offset + len)
            )
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(())
    }
    
    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
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
    
    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
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
    
    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        buf.len
    }
    
    fn new() -> Self {
        // TODO multiple devices
        Self::construct(0).unwrap()
    }
    
    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, TensorError> {
        self.sync()?;
        let mut dst = self.alloc(src.len)?;
        self.stream()
            .memcpy_dtod(&src.ptr, &mut dst.ptr)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(dst)
    }
    
    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, TensorError> {
        self.sync()?;
        let mut host_buf = vec![T::default(); src.len];
        self.stream()
            .memcpy_dtoh(&src.ptr, host_buf.as_mut_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(host_buf.into_boxed_slice())
    }


    fn apply_elementwise_binary_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (BinaryOpType, T), 
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
            id if id == std::any::TypeId::of::<types::boolean>() => launch_elementwise!(launch_elementwise_contiguous_boolean, bool),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
        }
    }
    
    fn apply_elementwise_binary_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (BinaryOpType, T), 
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
            id if id == std::any::TypeId::of::<types::boolean>() => launch_elementwise!(launch_elementwise_strided_boolean, bool),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
        }
    }
    
    fn apply_elementwise_binary_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (BinaryOpType, T),
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
        let shape_buf = self.alloc_from_slice(
            shape.to_vec()
            .into_iter()
            .map(|x| x as u64)
            .collect::<Vec<u64>>()
            .into_boxed_slice())?;
        let stride_buf = self.alloc_from_slice(
            stride.to_vec()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>()
            .into_boxed_slice())?;

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
            id if id == std::any::TypeId::of::<types::boolean>() => launch_elementwise!(launch_elementwise_nd_affine_boolean, bool),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
        }
    }

    fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &MetaTensor), 
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType
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
        let dshape_buf = self.alloc_from_slice(
            dmeta.shape.0.clone().into_iter().map(|x| x as u64).collect::<Vec<u64>>().into_boxed_slice()
        )?;
        
        let lstride_buf = self.alloc_from_slice(
            lmeta.strides.0.clone().into_iter().map(|x| x as i64).collect::<Vec<i64>>().into_boxed_slice()
        )?;
        let rstride_buf = self.alloc_from_slice(
            rmeta.strides.0.clone().into_iter().map(|x| x as i64).collect::<Vec<i64>>().into_boxed_slice()
        )?;
        let dstride_buf = self.alloc_from_slice(
            dmeta.strides.0.clone().into_iter().map(|x| x as i64).collect::<Vec<i64>>().into_boxed_slice()
        )?;

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
            id if id == std::any::TypeId::of::<types::boolean>() => launch_broadcast!(launch_binary_broadcast_elementwise_boolean, bool),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA broadcast operation".to_string())),
        }
    }


    
    fn apply_neg_contiguous<T: TensorValue + std::ops::Neg<Output = T>>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let stream = self.stream();


        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        start,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }
        

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_negate_contiguous_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_negate_contiguous_f64, f64),
            id if id == std::any::TypeId::of::<i8>() => launch_negate!(launch_negate_contiguous_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_negate!(launch_negate_contiguous_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_negate!(launch_negate_contiguous_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_negate!(launch_negate_contiguous_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_negate!(launch_negate_contiguous_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_neg_1d_strided<T: TensorValue + std::ops::Neg<Output = T>>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        let stream = self.stream();

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_negate_strided_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_negate_strided_f64, f64),
            id if id == std::any::TypeId::of::<i8>() => launch_negate!(launch_negate_strided_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_negate!(launch_negate_strided_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_negate!(launch_negate_strided_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_negate!(launch_negate_strided_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_negate!(launch_negate_strided_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_neg_nd<T: TensorValue + std::ops::Neg<Output = T>>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let stream = self.stream();
        let rank = shape.len();
        let size = shape.iter().product::<usize>();

        // Allocate device memory for strides and shapes
        let shape_buf = self.alloc_from_slice(
            shape.to_vec()
            .into_iter()
            .map(|x| x as u64)
            .collect::<Vec<u64>>()
            .into_boxed_slice())?;
        let stride_buf = self.alloc_from_slice(
            stride.to_vec()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>()
            .into_boxed_slice())?;

        let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
        let (shape_ptr, _) = shape_buf.ptr.device_ptr(&stream);

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride_ptr as *const isize,
                        shape_ptr as *const usize,
                        rank,
                        size,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_negate_nd_affine_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_negate_nd_affine_f64, f64),
            id if id == std::any::TypeId::of::<i8>() => launch_negate!(launch_negate_nd_affine_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_negate!(launch_negate_nd_affine_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_negate!(launch_negate_nd_affine_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_negate!(launch_negate_nd_affine_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_negate!(launch_negate_nd_affine_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }

    
    fn apply_relu_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let stream = self.stream();


        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        start,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }
        

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_relu_contiguous_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_relu_contiguous_f64, f64),
            id if id == std::any::TypeId::of::<i8>() => launch_negate!(launch_relu_contiguous_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_negate!(launch_relu_contiguous_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_negate!(launch_relu_contiguous_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_negate!(launch_relu_contiguous_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_negate!(launch_relu_contiguous_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_relu_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        // todo!()
        println!("Strided.");
        let stream = self.stream();

        macro_rules! launch_relu {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_relu!(launch_relu_strided_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_relu!(launch_relu_strided_f64, f64),
            id if id == std::any::TypeId::of::<i8>() => launch_relu!(launch_relu_strided_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_relu!(launch_relu_strided_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_relu!(launch_relu_strided_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_relu!(launch_relu_strided_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_relu!(launch_relu_strided_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_relu_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        // todo!()
        let stream = self.stream();
        let rank = shape.len();
        let size = shape.iter().product::<usize>();

        // Allocate device memory for strides and shapes
        let shape_buf = self.alloc_from_slice(
            shape.to_vec()
            .into_iter()
            .map(|x| x as u64)
            .collect::<Vec<u64>>()
            .into_boxed_slice())?;
        let stride_buf = self.alloc_from_slice(
            stride.to_vec()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>()
            .into_boxed_slice())?;

        let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
        let (shape_ptr, _) = shape_buf.ptr.device_ptr(&stream);

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride_ptr as *const isize,
                        shape_ptr as *const usize,
                        rank,
                        size,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_relu_nd_affine_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_relu_nd_affine_f64, f64),
            id if id == std::any::TypeId::of::<i8>() => launch_negate!(launch_relu_nd_affine_i8, i8),
            id if id == std::any::TypeId::of::<i16>() => launch_negate!(launch_relu_nd_affine_i16, i16),
            id if id == std::any::TypeId::of::<i32>() => launch_negate!(launch_relu_nd_affine_i32, i32),
            id if id == std::any::TypeId::of::<i64>() => launch_negate!(launch_relu_nd_affine_i64, i64),
            id if id == std::any::TypeId::of::<i128>() => launch_negate!(launch_relu_nd_affine_i128, i128),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }

    fn apply_sigmoid_contiguous<T: TensorValue + InvExp>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let stream = self.stream();


        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        start,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }
        

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_sigmoid_contiguous_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_sigmoid_contiguous_f64, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_sigmoid_1d_strided<T: TensorValue + InvExp>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        // todo!()
        println!("Strided.");
        let stream = self.stream();

        macro_rules! launch_relu {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_relu!(launch_sigmoid_strided_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_relu!(launch_sigmoid_strided_f64, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_sigmoid_nd<T: TensorValue + InvExp>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        // todo!()
        let stream = self.stream();
        let rank = shape.len();
        let size = shape.iter().product::<usize>();

        // Allocate device memory for strides and shapes
        let shape_buf = self.alloc_from_slice(
            shape.to_vec()
            .into_iter()
            .map(|x| x as u64)
            .collect::<Vec<u64>>()
            .into_boxed_slice())?;
        let stride_buf = self.alloc_from_slice(
            stride.to_vec()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>()
            .into_boxed_slice())?;

        let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
        let (shape_ptr, _) = shape_buf.ptr.device_ptr(&stream);

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride_ptr as *const isize,
                        shape_ptr as *const usize,
                        rank,
                        size,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_sigmoid_nd_affine_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_sigmoid_nd_affine_f64, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }

    fn apply_tanh_contiguous<T: TensorValue + Exp>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        let stream = self.stream();


        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        start,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }
        

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_tanh_contiguous_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_tanh_contiguous_f64, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_tanh_1d_strided<T: TensorValue + Exp>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {
        // todo!()
        println!("Strided.");
        let stream = self.stream();

        macro_rules! launch_relu {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_relu!(launch_tanh_strided_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_relu!(launch_tanh_strided_f64, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    fn apply_tanh_nd<T: TensorValue + Exp>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        // todo!()
        let stream = self.stream();
        let rank = shape.len();
        let size = shape.iter().product::<usize>();

        // Allocate device memory for strides and shapes
        let shape_buf = self.alloc_from_slice(
            shape.to_vec()
            .into_iter()
            .map(|x| x as u64)
            .collect::<Vec<u64>>()
            .into_boxed_slice())?;
        let stride_buf = self.alloc_from_slice(
            stride.to_vec()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>()
            .into_boxed_slice())?;

        let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
        let (shape_ptr, _) = shape_buf.ptr.device_ptr(&stream);

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        offset,
                        stride_ptr as *const isize,
                        shape_ptr as *const usize,
                        rank,
                        size,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_tanh_nd_affine_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_tanh_nd_affine_f64, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        }
    }
    
    // impl_cpu_unary!{ relu, _temp }
    // impl_cpu_unary! { neg, _temp }
    // impl_cpu_unary! { sigmoid, _temp }
}


impl Cuda {
    pub fn _test_apply_sum_flat_contiguous<T: TensorValue>(
        backend: &Cuda,
        buf: &mut <Cuda as Backend>::Buf<T>,
        out: &mut <Cuda as Backend>::Buf<T>,
        start: usize,   
        len: usize
    ) {
        apply_sum_flat_contiguous(backend, buf, out, start, len);
    }
}

fn apply_sum_flat_contiguous<T: TensorValue>(
    backend: &Cuda,
    buf: &mut <Cuda as Backend>::Buf<T>,
    out: &mut <Cuda as Backend>::Buf<T>,
    start: usize,   
    len: usize
) {
    // fn apply_tanh_contiguous<T: TensorValue + Exp>(
    //     &self, buf: &mut Self::Buf<T>, 
    //     start: usize,
    //     len: usize
    // ) -> Result<(), TensorError> {
    //     let stream = self.stream();

    let stream = backend.stream();

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                let (raw_output_ptr, _) = out.ptr.device_ptr(&stream);
                let out_ptr = raw_output_ptr as *mut $t;

                unsafe {
                    $launch_fn(
                        data_ptr as *mut $t,
                        out_ptr as *mut $t,
                        start,
                        len,
                        DEFAULT_BLOCK_SIZE,
                    );
                }
                backend.dirty();
                Ok(())
            }};
        }
        

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            // id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_tanh_contiguous_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_negate!(launch_flat_contiguous_reduce_sum_double, f64),
            _ => Err(TensorError::CudaError("Unsupported type for CUDA negation operation".to_string())),
        };

        println!("DONE");

    // }
}

pub fn _temp<T: TensorValue>(x: &mut T) -> T {
    *x
}

macro_rules! generic_matmul_impl {
    ($t:ty, $launch_fn:ident, $ptr_t:ty) => {
        impl BackendMatMul<$t> for Cuda {
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
                let stream = self.stream();
                // let res = self.alloc(b * m * n)?;
                
                let (lhs_buf, lhs_meta) = lhs;
                let (rhs_buf, rhs_meta) = rhs;

                let (lhs_ptr, _) = lhs_buf.ptr.device_ptr(&stream);
                let (rhs_ptr, _) = rhs_buf.ptr.device_ptr(&stream);
                let (dst_ptr, _) = dst.ptr.device_ptr(&stream);
                
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

                // let lda = lhs_meta.strides()[lhs_meta.rank() - 2] as usize;
                // let ldb = rhs_meta.strides()[rhs_meta.rank() - 2] as usize;
                let lda = match contiguity {
                    ContiguityTypes::RowMajor => lhs_meta.strides()[lhs_meta.rank() - 2] as usize,    // row-major
                    ContiguityTypes::ColumnMajor => lhs_meta.strides()[lhs_meta.rank() - 1] as usize, // column-major
                    ContiguityTypes::None => panic!("Matrix multiplication requires contiguous memory layout (either row-major or column-major)"),
                };
                let ldb = match contiguity {
                    ContiguityTypes::RowMajor => rhs_meta.strides()[rhs_meta.rank() - 2] as usize,    // row-major
                    ContiguityTypes::ColumnMajor => rhs_meta.strides()[rhs_meta.rank() - 1] as usize, // column-major
                    ContiguityTypes::None => panic!("Matrix multiplication requires contiguous memory layout (either row-major or column-major)"),
                };
                let ldc = n; // row-major
                
                // Convert Rust enum to C enum
                // These values match the ContiguityType enum in common.h
                let contiguity_c: u8 = match contiguity {
                    ContiguityTypes::RowMajor => 0,    // ROW_MAJOR
                    ContiguityTypes::ColumnMajor => 1, // COLUMN_MAJOR
                    ContiguityTypes::None => panic!("Matrix multiplication requires contiguous memory layout (either row-major or column-major)"),
                };
                
                // For batched matrix multiplication
                for batch_idx in 0..b {
                    let a_offset = lhs_meta.offset + batch_idx * bstride_lhs;
                    let b_offset = rhs_meta.offset + batch_idx * bstride_rhs;
                    let c_offset = batch_idx * m * n;
                    
                    let a_ptr = unsafe { (lhs_ptr as *const $ptr_t).add(a_offset) };
                    let b_ptr = unsafe { (rhs_ptr as *const $ptr_t).add(b_offset) };
                    let c_ptr = unsafe { (dst_ptr as *mut $ptr_t).add(c_offset) };
                    
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
                            contiguity_c,
                            DEFAULT_BLOCK_SIZE,
                        );
                    }
                }
                
                self.dirty();
                Ok(())
            }
        }
    };
}

macro_rules! cublas_impl {
    ($t:ty) => {
        impl BackendMatMul<$t> for Cuda {
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
                let (lhs_buf, lhs_meta) = lhs;
                let (rhs_buf, rhs_meta) = rhs;

                let lhs_ptr = rhs_buf.ptr.slice(rhs_meta.offset..);
                let rhs_ptr = lhs_buf.ptr.slice(lhs_meta.offset..);
                let ldc = n as i32;
                let stride_c = (m * n) as i64;
                
                let bstride_lhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides()[rhs_meta.rank() - 3] as i64
                } else {
                    n as i64 * k as i64
                };
                let bstride_rhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides()[lhs_meta.rank() - 3] as i64
                } else {
                    k as i64 * m as i64
                };

                // vary by contiguity
                let (stride_idx, transpose) = match contiguity {
                    ContiguityTypes::ColumnMajor => (1, cublasOperation_t::CUBLAS_OP_T),
                    ContiguityTypes::RowMajor => (2, cublasOperation_t::CUBLAS_OP_N),
                    _ => panic!("cuBLAS matmul only supports RowMajor and ColumnMajor contiguity"),
                };
                
                let lda = rhs_meta.strides()[rhs_meta.rank() - stride_idx] as i32;
                let ldb = lhs_meta.strides()[lhs_meta.rank() - stride_idx] as i32;
                let (m, n, k) = (n, m, k);

                let cfg = GemmConfig {
                    transa: transpose,
                    transb: transpose,
                    m: m as i32,
                    n: n as i32,
                    k: k as i32,
                    alpha: 1.0,
                    lda,  
                    ldb,
                    beta: 0.0,
                    ldc, 
                };
                let cfg = StridedBatchedConfig {
                    gemm: cfg,
                    batch_size: b as i32,
                    stride_a: bstride_lhs,
                    stride_b: bstride_rhs,
                    stride_c,
                };
                // let mut res = self.alloc(b*n*m)?;

                unsafe{
                    // Note: operands are swapped (B, A instead of A, B)
                    self.cublas.gemm_strided_batched(
                        cfg, 
                        &lhs_ptr,  // B comes first
                        &rhs_ptr,  // A comes second
                        &mut dst.ptr,
                    ).map_err(|e| TensorError::CudaError(e.to_string()))?;
                }
                self.dirty();
                Ok(())
            }
        }
    };
}

cublas_impl!(f32);
cublas_impl!(f64);

generic_matmul_impl!(u8, launch_matmul_u8, u8);
generic_matmul_impl!(u16, launch_matmul_u16, u16);
generic_matmul_impl!(u32, launch_matmul_u32, u32);
generic_matmul_impl!(u64, launch_matmul_u64, u64);
generic_matmul_impl!(u128, launch_matmul_u128, u128);
generic_matmul_impl!(i8, launch_matmul_i8, i8);
generic_matmul_impl!(i16, launch_matmul_i16, i16);
generic_matmul_impl!(i32, launch_matmul_i32, i32);
generic_matmul_impl!(i64, launch_matmul_i64, i64);
generic_matmul_impl!(i128, launch_matmul_i128, i128);
generic_matmul_impl!(types::boolean, launch_matmul_boolean, bool);


#[cfg(test)]
mod tests {
    use crate::{backend::cuda::Cuda, core::{MetaTensorView, primitives::CudaTensor, tensor::AsTensor}, ops::unary::{InplaceUnaryOp, Tanh}};



    #[test]
    pub fn test_reductio() {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> = CudaTensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2)).unwrap();
        println!("CUDA: {:?}", cuda.owned().cpu().unwrap());
        // cuda.tanh_inplace();

        let start = cuda.offset();
        let size = cuda.size();

        let sus = cuda.backend;

        let mut out = CudaTensor::<f64>::from_buf(vec![0.0], (1,)).unwrap();
        
        Cuda::_test_apply_sum_flat_contiguous(&sus, &mut cuda.buf, &mut out.buf, start, size);


        println!("OUT: {:?}", out.cpu());
    }
}