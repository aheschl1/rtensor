use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, LazyLock,
};

use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig},
    driver::{CudaContext, CudaSlice, DevicePtr},
};

use crate::{backend::ContiguityTypes, ops::reduction::{NormType, ReductionOpTypes}};
use crate::{
    backend::{cpu::Cpu, Backend, BackendMatMul},
    core::{
        primitives::TensorBase,
        primops::{Exp, InvExp},
        tensor::TensorError,
        value::{types, TensorValue},
        MetaTensor, Tensor,
    },
    ops::base::BinaryOpType,
};

// Include bindgen-generated FFI declarations for CUDA kernel launchers
#[allow(non_camel_case_types)]
mod bindings {
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
        let backend = CUDA_BACKENDS
            .get(device)
            .ok_or_else(|| TensorError::CudaError(format!("CUDA device {} not found", device)))?
            .clone();
        Ok(backend)
    }

    pub fn flush(&self) -> Result<(), TensorError> {
        if self.dirty.swap(false, Ordering::AcqRel) {
            // it was dirty, now its clean
            self.stream()
                .synchronize()
                .map_err(|e| TensorError::CudaError(e.to_string()))?;
        }
        Ok(())
    }

    #[inline(always)]
    pub fn sync(&self) -> Result<(), TensorError> {
        if self.dirty.load(Ordering::Acquire) {
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

// Put this somewhere visible (module scope), then invoke it inside your impl block.

// Cargo.toml:
// paste = "1"

macro_rules! specify_trait_unary_cabal {
    // ===== entry: no extra bounds =====
    { $op:ident } => {
        specify_trait_unary_cabal! { @impl $op where T: }
    };

    // ===== entry: with extra bounds =====
    { $op:ident where T: $($extra:tt)+ } => {
        specify_trait_unary_cabal! { @impl $op where T: $($extra)+ }
    };

    // ===== implementation =====
    ( @impl $op:ident where T: $($extra_bounds:tt)* ) => {
        paste::paste! {
            fn [<apply_ $op _contiguous>]<T: TensorValue $($extra_bounds)*>(
                &self,
                buf: &mut Self::Buf<T>,
                start: usize,
                len: usize,
            ) -> Result<(), TensorError> {
                let stream = self.stream();

                macro_rules! launch {
                    ($launch_fn:ident, $t:ty) => {{
                        let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                        let data_ptr = raw_ptr as *mut $t;
                        unsafe { $launch_fn(data_ptr, start, len, DEFAULT_BLOCK_SIZE); }
                        self.dirty();
                        Ok(())
                    }};
                }

                paste::paste! {
                    match std::any::TypeId::of::<T>() {
                        id if id == std::any::TypeId::of::<f32>() =>
                            launch!([<launch_ $op _contiguous_f32>], f32),
                        id if id == std::any::TypeId::of::<f64>() =>
                            launch!([<launch_ $op _contiguous_f64>], f64),
                        _ => Err(TensorError::CudaError(format!(
                            "Unsupported type for CUDA {} operation (float-only for now)",
                            stringify!($op),
                        ))),
                    }
                }
            }

            fn [<apply_ $op _1d_strided>]<T: TensorValue $($extra_bounds)*>(
                &self,
                buf: &mut Self::Buf<T>,
                offset: usize,
                stride: isize,
                len: usize,
            ) -> Result<(), TensorError> {
                let stream = self.stream();

                macro_rules! launch {
                    ($launch_fn:ident, $t:ty) => {{
                        let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                        let data_ptr = raw_ptr as *mut $t;
                        unsafe { $launch_fn(data_ptr, offset, stride, len, DEFAULT_BLOCK_SIZE); }
                        self.dirty();
                        Ok(())
                    }};
                }

                paste::paste! {
                    match std::any::TypeId::of::<T>() {
                        id if id == std::any::TypeId::of::<f32>() =>
                            launch!([<launch_ $op _strided_f32>], f32),
                        id if id == std::any::TypeId::of::<f64>() =>
                            launch!([<launch_ $op _strided_f64>], f64),
                        _ => Err(TensorError::CudaError(format!(
                            "Unsupported type for CUDA {} operation (float-only for now)",
                            stringify!($op),
                        ))),
                    }
                }
            }

            fn [<apply_ $op _nd>]<T: TensorValue $($extra_bounds)*>(
                &self,
                buf: &mut Self::Buf<T>,
                offset: usize,
                shape: &[usize],
                stride: &[isize],
            ) -> Result<(), TensorError> {
                let stream = self.stream();
                let rank = shape.len();
                let size = shape.iter().product::<usize>();

                // allocate device memory for shape/stride
                let shape_buf = self.alloc_from_slice(
                    shape.iter().copied().map(|x| x as u64).collect::<Vec<u64>>().into_boxed_slice()
                )?;
                let stride_buf = self.alloc_from_slice(
                    stride.iter().copied().map(|x| x as i64).collect::<Vec<i64>>().into_boxed_slice()
                )?;

                let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
                let (shape_ptr,  _) = shape_buf.ptr.device_ptr(&stream);

                macro_rules! launch {
                    ($launch_fn:ident, $t:ty) => {{
                        let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                        let data_ptr = raw_ptr as *mut $t;
                        unsafe {
                            $launch_fn(
                                data_ptr,
                                offset,
                                stride_ptr as *const isize,
                                shape_ptr  as *const usize,
                                rank,
                                size,
                                DEFAULT_BLOCK_SIZE,
                            );
                        }
                        self.dirty();
                        Ok(())
                    }};
                }

                paste::paste! {
                    match std::any::TypeId::of::<T>() {
                        id if id == std::any::TypeId::of::<f32>() =>
                            launch!([<launch_ $op _nd_affine_f32>], f32),
                        id if id == std::any::TypeId::of::<f64>() =>
                            launch!([<launch_ $op _nd_affine_f64>], f64),
                        _ => Err(TensorError::CudaError(format!(
                            "Unsupported type for CUDA {} operation (float-only for now)",
                            stringify!($op),
                        ))),
                    }
                }
            }
        }
    };
}



impl Backend for Cuda {
    type Buf<T: TensorValue> = CudaBuf<T>;

    fn device_type() -> crate::core::primitives::DeviceType {
        crate::core::primitives::DeviceType::Cuda(0)
    }

    fn alloc_from_slice<T: TensorValue>(
        &self,
        src: Box<[T]>,
    ) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let ptr = self
            .stream()
            .clone_htod(src.as_ref())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(CudaBuf {
            ptr,
            len: src.len(),
        })
    }

    fn alloc<T: TensorValue>(
        &self,
        len: usize,
    ) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let ptr;
        unsafe {
            ptr = self
                .stream()
                .alloc::<T>(len)
                .map_err(|e| TensorError::CudaError(e.to_string()))?;
        }
        Ok(CudaBuf { ptr, len })
    }

    fn copy_from_slice<T: TensorValue>(
        &self,
        dst: &mut Self::Buf<T>,
        src: &[T],
    ) -> Result<(), crate::core::tensor::TensorError> {
        self.sync()?;
        if src.len() != dst.len {
            return Err(TensorError::CudaError(format!(
                "Source slice length {} does not match destination buffer length {}",
                src.len(),
                dst.len
            )));
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
        len: usize,
    ) -> Result<(), TensorError> {
        self.sync()?;
        if src_offset + len > src.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Source range (offset {} + length {}) out of bounds for buffer of length {}",
                src_offset, len, src.len
            )));
        }
        if dst_offset + len > buf.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Destination range (offset {} + length {}) out of bounds for buffer of length {}",
                dst_offset, len, buf.len
            )));
        }

        self.stream()
            .memcpy_dtod(
                &src.ptr.slice(src_offset..src_offset + len),
                &mut buf.ptr.slice_mut(dst_offset..dst_offset + len),
            )
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(())
    }

    fn read<T: TensorValue>(
        &self,
        buf: &Self::Buf<T>,
        offset: usize,
    ) -> Result<T, crate::core::tensor::TensorError> {
        self.sync()?;
        if offset >= buf.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset, buf.len
            )));
        }

        let mut host_buf = vec![T::default(); 1];
        self.stream()
            .memcpy_dtoh(&buf.ptr.slice(offset..offset + 1), host_buf.as_mut_slice())
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(host_buf[0])
    }

    fn write<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        value: T,
    ) -> Result<(), crate::core::tensor::TensorError> {
        self.sync()?;
        if offset >= buf.len {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset, buf.len
            )));
        }

        self.stream()
            .memcpy_htod(&[value], &mut buf.ptr.slice_mut(offset..offset + 1))
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
        &self,
        buf: &mut Self::Buf<T>,
        op: (BinaryOpType, T),
        start: usize,
        len: usize,
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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_elementwise!(launch_elementwise_contiguous_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_elementwise!(launch_elementwise_contiguous_f64, f64)
            }
            id if id == std::any::TypeId::of::<u8>() => {
                launch_elementwise!(launch_elementwise_contiguous_u8, u8)
            }
            id if id == std::any::TypeId::of::<u16>() => {
                launch_elementwise!(launch_elementwise_contiguous_u16, u16)
            }
            id if id == std::any::TypeId::of::<u32>() => {
                launch_elementwise!(launch_elementwise_contiguous_u32, u32)
            }
            id if id == std::any::TypeId::of::<u64>() => {
                launch_elementwise!(launch_elementwise_contiguous_u64, u64)
            }
            id if id == std::any::TypeId::of::<u128>() => {
                launch_elementwise!(launch_elementwise_contiguous_u128, u128)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_elementwise!(launch_elementwise_contiguous_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_elementwise!(launch_elementwise_contiguous_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_elementwise!(launch_elementwise_contiguous_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_elementwise!(launch_elementwise_contiguous_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_elementwise!(launch_elementwise_contiguous_i128, i128)
            }
            id if id == std::any::TypeId::of::<types::boolean>() => {
                launch_elementwise!(launch_elementwise_contiguous_boolean, bool)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA elementwise operation".to_string(),
            )),
        }
    }

    fn apply_elementwise_binary_1d_strided<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (BinaryOpType, T),
        start: usize,
        stride: isize,
        len: usize,
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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_elementwise!(launch_elementwise_strided_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_elementwise!(launch_elementwise_strided_f64, f64)
            }
            id if id == std::any::TypeId::of::<u8>() => {
                launch_elementwise!(launch_elementwise_strided_u8, u8)
            }
            id if id == std::any::TypeId::of::<u16>() => {
                launch_elementwise!(launch_elementwise_strided_u16, u16)
            }
            id if id == std::any::TypeId::of::<u32>() => {
                launch_elementwise!(launch_elementwise_strided_u32, u32)
            }
            id if id == std::any::TypeId::of::<u64>() => {
                launch_elementwise!(launch_elementwise_strided_u64, u64)
            }
            id if id == std::any::TypeId::of::<u128>() => {
                launch_elementwise!(launch_elementwise_strided_u128, u128)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_elementwise!(launch_elementwise_strided_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_elementwise!(launch_elementwise_strided_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_elementwise!(launch_elementwise_strided_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_elementwise!(launch_elementwise_strided_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_elementwise!(launch_elementwise_strided_i128, i128)
            }
            id if id == std::any::TypeId::of::<types::boolean>() => {
                launch_elementwise!(launch_elementwise_strided_boolean, bool)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA elementwise operation".to_string(),
            )),
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
            shape
                .to_vec()
                .into_iter()
                .map(|x| x as u64)
                .collect::<Vec<u64>>()
                .into_boxed_slice(),
        )?;
        let stride_buf = self.alloc_from_slice(
            stride
                .to_vec()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
        )?;

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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_elementwise!(launch_elementwise_nd_affine_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_elementwise!(launch_elementwise_nd_affine_f64, f64)
            }
            id if id == std::any::TypeId::of::<u8>() => {
                launch_elementwise!(launch_elementwise_nd_affine_u8, u8)
            }
            id if id == std::any::TypeId::of::<u16>() => {
                launch_elementwise!(launch_elementwise_nd_affine_u16, u16)
            }
            id if id == std::any::TypeId::of::<u32>() => {
                launch_elementwise!(launch_elementwise_nd_affine_u32, u32)
            }
            id if id == std::any::TypeId::of::<u64>() => {
                launch_elementwise!(launch_elementwise_nd_affine_u64, u64)
            }
            id if id == std::any::TypeId::of::<u128>() => {
                launch_elementwise!(launch_elementwise_nd_affine_u128, u128)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_elementwise!(launch_elementwise_nd_affine_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_elementwise!(launch_elementwise_nd_affine_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_elementwise!(launch_elementwise_nd_affine_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_elementwise!(launch_elementwise_nd_affine_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_elementwise!(launch_elementwise_nd_affine_i128, i128)
            }
            id if id == std::any::TypeId::of::<types::boolean>() => {
                launch_elementwise!(launch_elementwise_nd_affine_boolean, bool)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA elementwise operation".to_string(),
            )),
        }
    }

    fn broadcast<T: TensorValue>(
        &self,
        left: (*const Self::Buf<T>, &MetaTensor),
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType,
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
            dmeta
                .shape
                .0
                .clone()
                .into_iter()
                .map(|x| x as u64)
                .collect::<Vec<u64>>()
                .into_boxed_slice(),
        )?;

        let lstride_buf = self.alloc_from_slice(
            lmeta
                .strides
                .0
                .clone()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
        )?;
        let rstride_buf = self.alloc_from_slice(
            rmeta
                .strides
                .0
                .clone()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
        )?;
        let dstride_buf = self.alloc_from_slice(
            dmeta
                .strides
                .0
                .clone()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_f64, f64)
            }
            id if id == std::any::TypeId::of::<u8>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_u8, u8)
            }
            id if id == std::any::TypeId::of::<u16>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_u16, u16)
            }
            id if id == std::any::TypeId::of::<u32>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_u32, u32)
            }
            id if id == std::any::TypeId::of::<u64>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_u64, u64)
            }
            id if id == std::any::TypeId::of::<u128>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_u128, u128)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_i128, i128)
            }
            id if id == std::any::TypeId::of::<types::boolean>() => {
                launch_broadcast!(launch_binary_broadcast_elementwise_boolean, bool)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA broadcast operation".to_string(),
            )),
        }
    }

    fn apply_neg_contiguous<T: TensorValue + std::ops::Neg<Output = T>>(
        &self,
        buf: &mut Self::Buf<T>,
        start: usize,
        len: usize,
    ) -> Result<(), TensorError> {
        let stream = self.stream();

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(data_ptr as *mut $t, start, len, DEFAULT_BLOCK_SIZE);
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_negate_contiguous_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_negate_contiguous_f64, f64)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_negate!(launch_negate_contiguous_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_negate!(launch_negate_contiguous_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_negate!(launch_negate_contiguous_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_negate!(launch_negate_contiguous_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_negate!(launch_negate_contiguous_i128, i128)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
        }
    }

    fn apply_neg_1d_strided<T: TensorValue + std::ops::Neg<Output = T>>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        stride: isize,
        len: usize,
    ) -> Result<(), TensorError> {
        let stream = self.stream();

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(data_ptr as *mut $t, offset, stride, len, DEFAULT_BLOCK_SIZE);
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_negate_strided_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_negate_strided_f64, f64)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_negate!(launch_negate_strided_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_negate!(launch_negate_strided_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_negate!(launch_negate_strided_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_negate!(launch_negate_strided_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_negate!(launch_negate_strided_i128, i128)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
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
            shape
                .to_vec()
                .into_iter()
                .map(|x| x as u64)
                .collect::<Vec<u64>>()
                .into_boxed_slice(),
        )?;
        let stride_buf = self.alloc_from_slice(
            stride
                .to_vec()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
        )?;

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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_negate_nd_affine_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_negate_nd_affine_f64, f64)
            }
            id if id == std::any::TypeId::of::<i8>() => {
                launch_negate!(launch_negate_nd_affine_i8, i8)
            }
            id if id == std::any::TypeId::of::<i16>() => {
                launch_negate!(launch_negate_nd_affine_i16, i16)
            }
            id if id == std::any::TypeId::of::<i32>() => {
                launch_negate!(launch_negate_nd_affine_i32, i32)
            }
            id if id == std::any::TypeId::of::<i64>() => {
                launch_negate!(launch_negate_nd_affine_i64, i64)
            }
            id if id == std::any::TypeId::of::<i128>() => {
                launch_negate!(launch_negate_nd_affine_i128, i128)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
        }
    }

    

    // fn apply_relu_contiguous<T: TensorValue>(
    //     &self,
    //     buf: &mut Self::Buf<T>,
    //     start: usize,
    //     len: usize,
    // ) -> Result<(), TensorError> {
    //     let stream = self.stream();

    //     macro_rules! launch_negate {
    //         ($launch_fn:ident, $t:ty) => {{
    //             let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
    //             let data_ptr = raw_ptr as *mut $t;

    //             unsafe {
    //                 $launch_fn(data_ptr as *mut $t, start, len, DEFAULT_BLOCK_SIZE);
    //             }
    //             self.dirty();
    //             Ok(())
    //         }};
    //     }

    //     // Dispatch based on type - only signed types support negation
    //     match std::any::TypeId::of::<T>() {
    //         id if id == std::any::TypeId::of::<f32>() => {
    //             launch_negate!(launch_relu_contiguous_f32, f32)
    //         }
    //         id if id == std::any::TypeId::of::<f64>() => {
    //             launch_negate!(launch_relu_contiguous_f64, f64)
    //         }
    //         id if id == std::any::TypeId::of::<i8>() => {
    //             launch_negate!(launch_relu_contiguous_i8, i8)
    //         }
    //         id if id == std::any::TypeId::of::<i16>() => {
    //             launch_negate!(launch_relu_contiguous_i16, i16)
    //         }
    //         id if id == std::any::TypeId::of::<i32>() => {
    //             launch_negate!(launch_relu_contiguous_i32, i32)
    //         }
    //         id if id == std::any::TypeId::of::<i64>() => {
    //             launch_negate!(launch_relu_contiguous_i64, i64)
    //         }
    //         id if id == std::any::TypeId::of::<i128>() => {
    //             launch_negate!(launch_relu_contiguous_i128, i128)
    //         }
    //         _ => Err(TensorError::CudaError(
    //             "Unsupported type for CUDA negation operation".to_string(),
    //         )),
    //     }
    // }

    // fn apply_relu_1d_strided<T: TensorValue>(
    //     &self,
    //     buf: &mut Self::Buf<T>,
    //     offset: usize,
    //     stride: isize,
    //     len: usize,
    // ) -> Result<(), TensorError> {
    //     // todo!()
    //     let stream = self.stream();

    //     macro_rules! launch_relu {
    //         ($launch_fn:ident, $t:ty) => {{
    //             let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
    //             let data_ptr = raw_ptr as *mut $t;

    //             unsafe {
    //                 $launch_fn(data_ptr as *mut $t, offset, stride, len, DEFAULT_BLOCK_SIZE);
    //             }
    //             self.dirty();
    //             Ok(())
    //         }};
    //     }

    //     // Dispatch based on type - only signed types support negation
    //     match std::any::TypeId::of::<T>() {
    //         id if id == std::any::TypeId::of::<f32>() => launch_relu!(launch_relu_strided_f32, f32),
    //         id if id == std::any::TypeId::of::<f64>() => launch_relu!(launch_relu_strided_f64, f64),
    //         id if id == std::any::TypeId::of::<i8>() => launch_relu!(launch_relu_strided_i8, i8),
    //         id if id == std::any::TypeId::of::<i16>() => launch_relu!(launch_relu_strided_i16, i16),
    //         id if id == std::any::TypeId::of::<i32>() => launch_relu!(launch_relu_strided_i32, i32),
    //         id if id == std::any::TypeId::of::<i64>() => launch_relu!(launch_relu_strided_i64, i64),
    //         id if id == std::any::TypeId::of::<i128>() => {
    //             launch_relu!(launch_relu_strided_i128, i128)
    //         }
    //         _ => Err(TensorError::CudaError(
    //             "Unsupported type for CUDA negation operation".to_string(),
    //         )),
    //     }
    // }

    // fn apply_relu_nd<T: TensorValue>(
    //     &self,
    //     buf: &mut Self::Buf<T>,
    //     offset: usize,
    //     shape: &[usize],
    //     stride: &[isize],
    // ) -> Result<(), TensorError> {
    //     // todo!()
    //     let stream = self.stream();
    //     let rank = shape.len();
    //     let size = shape.iter().product::<usize>();

    //     // Allocate device memory for strides and shapes
    //     let shape_buf = self.alloc_from_slice(
    //         shape
    //             .to_vec()
    //             .into_iter()
    //             .map(|x| x as u64)
    //             .collect::<Vec<u64>>()
    //             .into_boxed_slice(),
    //     )?;
    //     let stride_buf = self.alloc_from_slice(
    //         stride
    //             .to_vec()
    //             .into_iter()
    //             .map(|x| x as i64)
    //             .collect::<Vec<i64>>()
    //             .into_boxed_slice(),
    //     )?;

    //     let (stride_ptr, _) = stride_buf.ptr.device_ptr(&stream);
    //     let (shape_ptr, _) = shape_buf.ptr.device_ptr(&stream);

    //     macro_rules! launch_negate {
    //         ($launch_fn:ident, $t:ty) => {{
    //             let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
    //             let data_ptr = raw_ptr as *mut $t;

    //             unsafe {
    //                 $launch_fn(
    //                     data_ptr as *mut $t,
    //                     offset,
    //                     stride_ptr as *const isize,
    //                     shape_ptr as *const usize,
    //                     rank,
    //                     size,
    //                     DEFAULT_BLOCK_SIZE,
    //                 );
    //             }
    //             self.dirty();
    //             Ok(())
    //         }};
    //     }

    //     // Dispatch based on type - only signed types support negation
    //     match std::any::TypeId::of::<T>() {
    //         id if id == std::any::TypeId::of::<f32>() => {
    //             launch_negate!(launch_relu_nd_affine_f32, f32)
    //         }
    //         id if id == std::any::TypeId::of::<f64>() => {
    //             launch_negate!(launch_relu_nd_affine_f64, f64)
    //         }
    //         id if id == std::any::TypeId::of::<i8>() => {
    //             launch_negate!(launch_relu_nd_affine_i8, i8)
    //         }
    //         id if id == std::any::TypeId::of::<i16>() => {
    //             launch_negate!(launch_relu_nd_affine_i16, i16)
    //         }
    //         id if id == std::any::TypeId::of::<i32>() => {
    //             launch_negate!(launch_relu_nd_affine_i32, i32)
    //         }
    //         id if id == std::any::TypeId::of::<i64>() => {
    //             launch_negate!(launch_relu_nd_affine_i64, i64)
    //         }
    //         id if id == std::any::TypeId::of::<i128>() => {
    //             launch_negate!(launch_relu_nd_affine_i128, i128)
    //         }
    //         _ => Err(TensorError::CudaError(
    //             "Unsupported type for CUDA negation operation".to_string(),
    //         )),
    //     }
    // }

    specify_trait_unary_cabal!{relu}
    specify_trait_unary_cabal!{abs}

    fn apply_sigmoid_contiguous<T: TensorValue + InvExp>(
        &self,
        buf: &mut Self::Buf<T>,
        start: usize,
        len: usize,
    ) -> Result<(), TensorError> {
        let stream = self.stream();

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(data_ptr as *mut $t, start, len, DEFAULT_BLOCK_SIZE);
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_sigmoid_contiguous_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_sigmoid_contiguous_f64, f64)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
        }
    }

    fn apply_sigmoid_1d_strided<T: TensorValue + InvExp>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        stride: isize,
        len: usize,
    ) -> Result<(), TensorError> {
        // todo!()
        println!("Strided.");
        let stream = self.stream();

        macro_rules! launch_relu {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(data_ptr as *mut $t, offset, stride, len, DEFAULT_BLOCK_SIZE);
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                launch_relu!(launch_sigmoid_strided_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_relu!(launch_sigmoid_strided_f64, f64)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
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
            shape
                .to_vec()
                .into_iter()
                .map(|x| x as u64)
                .collect::<Vec<u64>>()
                .into_boxed_slice(),
        )?;
        let stride_buf = self.alloc_from_slice(
            stride
                .to_vec()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
        )?;

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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_sigmoid_nd_affine_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_sigmoid_nd_affine_f64, f64)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
        }
    }

    fn apply_tanh_contiguous<T: TensorValue + Exp>(
        &self,
        buf: &mut Self::Buf<T>,
        start: usize,
        len: usize,
    ) -> Result<(), TensorError> {
        let stream = self.stream();

        macro_rules! launch_negate {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(data_ptr as *mut $t, start, len, DEFAULT_BLOCK_SIZE);
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_tanh_contiguous_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_tanh_contiguous_f64, f64)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
        }
    }

    fn apply_tanh_1d_strided<T: TensorValue + Exp>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        stride: isize,
        len: usize,
    ) -> Result<(), TensorError> {
        // todo!()
        println!("Strided.");
        let stream = self.stream();

        macro_rules! launch_relu {
            ($launch_fn:ident, $t:ty) => {{
                let (raw_ptr, _) = buf.ptr.device_ptr(&stream);
                let data_ptr = raw_ptr as *mut $t;

                unsafe {
                    $launch_fn(data_ptr as *mut $t, offset, stride, len, DEFAULT_BLOCK_SIZE);
                }
                self.dirty();
                Ok(())
            }};
        }

        // Dispatch based on type - only signed types support negation
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => launch_relu!(launch_tanh_strided_f32, f32),
            id if id == std::any::TypeId::of::<f64>() => launch_relu!(launch_tanh_strided_f64, f64),
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
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
            shape
                .to_vec()
                .into_iter()
                .map(|x| x as u64)
                .collect::<Vec<u64>>()
                .into_boxed_slice(),
        )?;
        let stride_buf = self.alloc_from_slice(
            stride
                .to_vec()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice(),
        )?;

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
            id if id == std::any::TypeId::of::<f32>() => {
                launch_negate!(launch_tanh_nd_affine_f32, f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                launch_negate!(launch_tanh_nd_affine_f64, f64)
            }
            _ => Err(TensorError::CudaError(
                "Unsupported type for CUDA negation operation".to_string(),
            )),
        }
    }
    
    fn apply_reduce_contiguous_flat<T: TensorValue>(
        &self, 
        src: &Self::Buf<T>, 
        dst: &mut Self::Buf<T>, 
        start: usize, 
        len: usize, 
        op: crate::ops::reduction::ReductionOpTypes
    ) -> Result<(), TensorError> {
        apply_reduction_contiguous_single_elem(self, src, dst, start, len, op)
    }
    
    fn apply_reduce_contiguous_nd<T: TensorValue>(
        &self, 
        src: (&Self::Buf<T>, &MetaTensor), 
        dst: (&mut Self::Buf<T>, &MetaTensor), 
        dim: crate::core::Dim,
        op: crate::ops::reduction::ReductionOpTypes
    ) -> Result<(), TensorError> {
        apply_nd_reduction_contiguous(self, src, dst, dim, op)
    }
    

    fn apply_reduce_total<T: TensorValue>(
            &self, 
            src: (&Self::Buf<T>, &MetaTensor), 
            mut dst: (&mut Self::Buf<T>, &MetaTensor), 
            dim: crate::core::Dim,
            op: crate::ops::reduction::ReductionOpTypes
        ) -> Result<(), TensorError> {
        apply_reduction_contiguous_single_elem(self, &src.0, &mut dst.0, src.1.offset(), src.1.size(), op)
    }

    // impl_cpu_unary!{ relu, _temp }
    // impl_cpu_unary! { neg, _temp }
    // impl_cpu_unary! { sigmoid, _temp }
}





#[inline]
fn populate_reduction_settings(
    op: &ReductionOpTypes
) -> ReductionSettings {
    let mut settings = ReductionSettings {
        is_std: false,
        unbiased: false,
        norm_type: 0
    };

    match &op {
        ReductionOpTypes::Variance { unbiased } => {
            settings.unbiased = *unbiased;
        }
        ReductionOpTypes::Stdev { unbiased } => {
            settings.unbiased = *unbiased;
            settings.is_std = true;
        }
        ReductionOpTypes::Norm(ntype) => match ntype {
            NormType::L1 => settings.norm_type = 1,
            NormType::L2 => settings.norm_type = 2
        }
        _ => {}
    }
    settings
}


fn apply_reduction_contiguous_single_elem<T: TensorValue>(
    backend: &Cuda,
    buf: &<Cuda as Backend>::Buf<T>,
    out: &mut <Cuda as Backend>::Buf<T>,
    start: usize,
    len: usize,
    op: ReductionOpTypes,
    // settings: &ReductionSettings

) -> Result<(), TensorError> {
    let stream = backend.stream();


    let settings = populate_reduction_settings(&op);

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
                    op.get_code(),
                    &settings as *const ReductionSettings,
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
        id if id == std::any::TypeId::of::<f64>() => {
            launch_negate!(launch_flat_contiguous_reduce_f64, f64)
        }
        _ => Err(TensorError::CudaError(
            "Unsupported type for CUDA negation operation".to_string(),
        )),
    }
}



/// This assumes a  contiguous tensor.
fn apply_nd_reduction_contiguous<T: TensorValue>(
    backend: &Cuda,
    (in_d, in_d_meta): (&<Cuda as Backend>::Buf<T>, &MetaTensor),
    (out_d, _): (&mut <Cuda as Backend>::Buf<T>, &MetaTensor),
    axis: usize,
    code: ReductionOpTypes
) -> Result<(), TensorError> {


    let settings = populate_reduction_settings(&code);


    // This is a temporary limitation of the system.
    assert!(in_d_meta.is_contiguous(), "Currently the library only accepts contiguous tensors.");
    

    let stream = backend.stream();

    // let settings = unsafe {
    //     stream
    //         .alloc::<ReductionSettings>(size_of::<ReductionSettings>())?
    // };


    // Calculate the reduction length.
    let red_len = in_d_meta.shape()[axis];

    // Calculate the inner and outer dimensions.
    let inner = in_d_meta.inner_dimensions(axis);
    let outer = in_d_meta.outer_dimensions(axis);


    assert!(DEFAULT_BLOCK_SIZE <= 256, "We do not support this right now");

    macro_rules! launch_negate {
        ($launch_fn:ident, $t:ty) => {{
            let (raw_ptr, _) = in_d.ptr.device_ptr(&stream);
            let data_ptr = raw_ptr as *mut $t;

            let (raw_output_ptr, _) = out_d.ptr.device_ptr(&stream);
            let out_ptr = raw_output_ptr as *mut $t;

            unsafe {
                $launch_fn(
                    data_ptr as *mut $t,
                    out_ptr as *mut $t,
                    in_d_meta.offset(),
                    inner,
                    red_len,
                    outer,
                    code.get_code(),
                    &settings as *const ReductionSettings,
                    DEFAULT_BLOCK_SIZE
                );
            }
            backend.dirty();
            Ok(())
        }};
    }

    // Dispatch based on type - only signed types support negation
    match std::any::TypeId::of::<T>() {
        // id if id == std::any::TypeId::of::<f32>() => launch_negate!(launch_tanh_contiguous_f32, f32),
        id if id == std::any::TypeId::of::<f64>() => {
            launch_negate!(launch_nd_reduce_contiguous_f64, f64)
        }
        _ => Err(TensorError::CudaError(
            "Unsupported type for CUDA negation operation".to_string(),
        )),
    }
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
                contiguity: ContiguityTypes,
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

                unsafe {
                    // Note: operands are swapped (B, A instead of A, B)
                    self.cublas
                        .gemm_strided_batched(
                            cfg,
                            &lhs_ptr, // B comes first
                            &rhs_ptr, // A comes second
                            &mut dst.ptr,
                        )
                        .map_err(|e| TensorError::CudaError(e.to_string()))?;
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
    use std::error::Error;

    use crate::{
        backend::cuda::{Cuda, apply_nd_reduction_contiguous},
        core::{MetaTensorView, Tensor, idx::Idx, primitives::CudaTensor, tensor::{AsTensor, TensorAccess}, value::TensorValue},
        ops::{reduction::{NormType, ReductionOp, TotalReductionOp}, unary::{InplaceUnaryOp, Tanh}},
    };

    #[test]
    pub fn test_reduce_total_sum_case1() {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();
        assert_eq!(cuda.total_sum().unwrap().item().unwrap(), 1.1);
    }

    #[test]
    pub fn test_reduce_total_max_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();
        assert_eq!(cuda.max(&Idx::Item).unwrap().item().unwrap(), 0.3);
    }

     #[test]
    pub fn test_reduce_total_min_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, -0.9, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();
        assert_eq!(cuda.min(&Idx::Item).unwrap().item().unwrap(), -0.9);
    }


    #[test]
    pub fn test_reduce_total_prod_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.prod(&Idx::Item).unwrap().item().unwrap(), 40320.);
    }

    #[test]
    pub fn test_reduce_total_mean_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.mean(&Idx::Item).unwrap().item().unwrap(), 4.5);
    }

    #[test]
    pub fn test_reduce_total_variance_unbiased() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.var(&Idx::Item).unwrap().item().unwrap(), 6.0);
    }

    #[test]
    pub fn test_reduce_total_variance_biased() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.pop_var(&Idx::Item).unwrap().item().unwrap(), 5.25);
    }

    #[test]
    pub fn test_reduce_total_stdev_unbiased() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.std(&Idx::Item, true).unwrap().item().unwrap(), 2.449489742783178);
    }

    #[test]
    pub fn test_reduce_total_stdev_biased() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.std(&Idx::Item, false).unwrap().item().unwrap(), 2.29128784747792);
    }

    #[test]
    pub fn test_reduce_total_norm_l1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.norm(NormType::L1, &Idx::Item).unwrap().item().unwrap(), 36.);
    }

    #[test]
    pub fn test_reduce_total_norm_l2() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.norm(NormType::L2, &Idx::Item).unwrap().item().unwrap(), 14.2828568570857);
    }

    #[test]
    pub fn test_reduce_sum_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1., 0., 1., 0., 1., 1., 1., 0.], (4, 2))
                .unwrap();
        assert_eq!(cuda.sum(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![2., 3.], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_max_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![3., 5., 6., 8., 1., 2., -1., 4.], (4, 2))
                .unwrap();
        assert_eq!(cuda.max(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![8., 4.], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_min_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![3., 5., 6., 8., 1., 2., -1., 4.], (4, 2))
                .unwrap();
        assert_eq!(cuda.min(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![3., -1.], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_prod_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![3., 5., 6., 8., 1., 2., -1., 4.], (4, 2))
                .unwrap();
        assert_eq!(cuda.prod(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![720., -8.], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_mean_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.mean(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![2.5, 6.5], (1, 2))?.cpu()?);
        Ok(())
    }


    #[test]
    pub fn test_reduce_mean_case2() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.mean(&Idx::At(1))?.cpu()?, CudaTensor::from_buf(vec![3.0, 4.0, 5.0, 6.0], (4, 1))?.cpu()?);
        Ok(())
    }

 

    #[test]
    pub fn test_reduce_variance_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.var(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![1.6666666666666667f64, 1.6666666666666667], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_pop_var_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.pop_var(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![1.25, 1.25], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_stdev_unbiased() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.std(&Idx::At(0), true)?.cpu()?, CudaTensor::from_buf(vec![1.2909944487358056, 1.2909944487358056], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_stdev_biased() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.std(&Idx::At(0), false)?.cpu()?, CudaTensor::from_buf(vec![1.118033988749895, 1.118033988749895], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_logsumexp() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.logsumexp(&Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![4.440189698561196, 8.440189698561195], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_norm_l1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.norm(NormType::L1, &Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![10.0, 26.0], (1, 2))?.cpu()?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_norm_l2() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.norm(NormType::L2, &Idx::At(0))?.cpu()?, CudaTensor::from_buf(vec![5.477225575051661, 13.19090595827292], (1, 2))?.cpu()?);
        Ok(())
    }


    #[test]
    pub fn test_reductio_multi() {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cuda::Cuda> =
            CudaTensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();


        println!("Original: {:?}", cuda.cpu());

        let result = cuda.sum(&Idx::At(1));
        println!("Result: {:?}", result.unwrap().cpu());

        // let mut out: crate::core::primitives::TensorBase<f64, Cuda> = CudaTensor::from_buf(vec![0.0f64, 0.0f64], (2,))
        //     .unwrap();


        // let in_tensor = (&mut cuda.buf, cuda.meta.clone());
        // let out_tensor = (&mut out.buf, out.meta.clone());

        // _apply_sum_contiguous(&cuda.backend, in_tensor,  out_tensor, 1)
        //     .unwrap();
        

        // println!("Output: {:?}", out.cpu());

        // println!("CUDA: {:?}", cuda.owned().cpu().unwrap());
        // // cuda.tanh_inplace();

        // let start = cuda.offset();
        // let size = cuda.size();

        // let sus = cuda.backend;

        // let mut out = CudaTensor::<f64>::from_buf(vec![0.0], (1,)).unwrap();

        // Cuda::_test_apply_sum_flat_contiguous(&sus, &mut cuda.buf, &mut out.buf, start, size);

        // println!("OUT: {:?}", out.cpu());
    }
}
