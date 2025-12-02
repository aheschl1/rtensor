use std::sync::{atomic::{AtomicBool, Ordering}, Arc, LazyLock};

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

use crate::{backend::{Backend, BackendBinaryElementwise, BackendUnaryElementwise}, core::{tensor::TensorError, value::{TensorValue, TensorValueElementwise}}, ops::unary::ElementwiseUnaryTensorOp};

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
        let backend = Cuda { 
            ctx: CudaContext::new(device_id as usize).unwrap(), 
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
            return Err(TensorError::IdxOutOfBounds);
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
            return Err(TensorError::IdxOutOfBounds);
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

    
}

impl<T: TensorValueElementwise> BackendUnaryElementwise<T> for Cuda {
    
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseUnaryTensorOp<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError> {
        
        let op_code = op.to_op_code();
        let value = op.value();
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
        op: &ElementwiseUnaryTensorOp<T>, 
        start: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {

        let op_code = op.to_op_code();
        let value = op.value();
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
        op: &ElementwiseUnaryTensorOp<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let op_code = op.to_op_code();
        let value = op.value();
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

}

impl<T: TensorValueElementwise> BackendBinaryElementwise<T> for Cuda {
    fn broadcast(
        &self, 
        left: (&Self::Buf, &crate::core::MetaTensor), 
        right: (&Self::Buf, &crate::core::MetaTensor),
        dst: (&mut Self::Buf, &crate::core::MetaTensor),
        op: crate::ops::binary::ElementwiseBinaryTensorOp<T>
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
        
        let lstride_buf = self.alloc_from_slice(lmeta.stride().clone().into_boxed_slice())?;
        let rstride_buf = self.alloc_from_slice(rmeta.stride().clone().into_boxed_slice())?;
        let dstride_buf = self.alloc_from_slice(dmeta.stride().clone().into_boxed_slice())?;

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
                let (lbuf_ptr, _) = lbuf.ptr.device_ptr(&stream);
                let (rbuf_ptr, _) = rbuf.ptr.device_ptr(&stream);
                let (dbuf_ptr, _) = dbuf.ptr.device_ptr(&stream);
                
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