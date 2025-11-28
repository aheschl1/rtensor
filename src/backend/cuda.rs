use std::sync::{atomic::{AtomicBool, Ordering}, Arc, LazyLock, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DeviceRepr};

use crate::{backend::{Backend, BackendUnaryElementwise}, core::{tensor::TensorError, value::{TensorValue, TensorValueElementwise}}, ops::unary::ElementwiseTensorOp};

// Include bindgen-generated FFI declarations for CUDA kernel launchers
#[allow(non_camel_case_types)]
mod bindings{
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use bindings::*;

// Can be tuned based on kernel characteristics and GPU architecture
const DEFAULT_BLOCK_SIZE: u32 = 256;

const CUDA_BACKENDS: LazyLock<Vec<CudaBackend>> = LazyLock::new(|| {
    let mut backends = Vec::new();
    let device_count = cudarc::driver::CudaContext::device_count().unwrap_or(0);
    for device_id in 0..device_count {
        let backend = CudaBackend { 
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
pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    dirty: Arc<AtomicBool>,
}


impl CudaBackend {
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

impl<T: TensorValue + DeviceRepr> Backend<T> for CudaBackend {
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

impl<T: TensorValueElementwise + DeviceRepr + 'static> BackendUnaryElementwise<T> for CudaBackend {
    
    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
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
        match std::any::TypeId::of::<T>() {
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
    
    fn apply_elementwise_strided(
        &self, buf: &mut Self::Buf, 
        op: &ElementwiseTensorOp<T>, 
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
    
    // fn apply_elementwise_scattered(
    //     &self, buf: &mut Self::Buf, 
    //     op: &ElementwiseTensorOp<T>, 
    //     offsets: &[usize]
    // ) -> Result<(), TensorError> {
    //     // Upload offsets to device
    //     let offsets_ptr_device = self.stream()
    //         .clone_htod(offsets)
    //         .map_err(|e| TensorError::CudaError(e.to_string()))?;

    //     let op_code = op.to_op_code();
    //     let value = op.value();
    //     let n = offsets.len();
    //     let stream = self.stream();

    //     macro_rules! launch_elementwise {
    //         ($launch_fn:ident, $t:ty) => {{
    //             // transmute value from T to actual type
    //             let concrete_value: $t = unsafe { std::mem::transmute_copy(&value) };
    //             let (data_ptr, _) = buf.ptr.device_ptr(&stream);
    //             let (offsets_ptr, _) = offsets_ptr_device.device_ptr(&stream);
                
    //             unsafe {
    //                 $launch_fn(
    //                     data_ptr as *mut $t,
    //                     offsets_ptr as *mut usize,
    //                     n,
    //                     op_code,
    //                     concrete_value,
    //                     DEFAULT_BLOCK_SIZE,
    //                 );
    //             }
    //             self.dirty();
    //             Ok(())
    //         }};
    //     }

    //     // Dispatch based on type
    //     match std::any::TypeId::of::<T>() {
    //         id if id == std::any::TypeId::of::<f32>() => launch_elementwise!(launch_elementwise_scattered_f32, f32),
    //         id if id == std::any::TypeId::of::<f64>() => launch_elementwise!(launch_elementwise_scattered_f64, f64),
    //         id if id == std::any::TypeId::of::<u8>() => launch_elementwise!(launch_elementwise_scattered_u8, u8),
    //         id if id == std::any::TypeId::of::<u16>() => launch_elementwise!(launch_elementwise_scattered_u16, u16),
    //         id if id == std::any::TypeId::of::<u32>() => launch_elementwise!(launch_elementwise_scattered_u32, u32),
    //         id if id == std::any::TypeId::of::<u64>() => launch_elementwise!(launch_elementwise_scattered_u64, u64),
    //         id if id == std::any::TypeId::of::<u128>() => launch_elementwise!(launch_elementwise_scattered_u128, u128),
    //         id if id == std::any::TypeId::of::<i8>() => launch_elementwise!(launch_elementwise_scattered_i8, i8),
    //         id if id == std::any::TypeId::of::<i16>() => launch_elementwise!(launch_elementwise_scattered_i16, i16),
    //         id if id == std::any::TypeId::of::<i32>() => launch_elementwise!(launch_elementwise_scattered_i32, i32),
    //         id if id == std::any::TypeId::of::<i64>() => launch_elementwise!(launch_elementwise_scattered_i64, i64),
    //         id if id == std::any::TypeId::of::<i128>() => launch_elementwise!(launch_elementwise_scattered_i128, i128),
    //         _ => Err(TensorError::CudaError("Unsupported type for CUDA elementwise operation".to_string())),
    //     }
    // }

}
