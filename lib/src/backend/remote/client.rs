use std::{net::IpAddr, sync::{atomic::AtomicBool, Arc}};

use crate::{backend::Backend, core::value::TensorValue};


struct RemoteCtx {
    pub address: IpAddr,
    pub port: u16
}

pub struct RemoteBuf {
    pub id: u64,
    dirty: Arc<AtomicBool>,
}

pub struct RemoteBackend {
    remote: RemoteCtx,
}

impl<T: TensorValue> Backend<T> for RemoteBackend {
    type Buf = RemoteBuf;

    fn device_type() -> crate::core::primitives::DeviceType {
        todo!()
    }

    fn alloc_from_slice(&self, src: Box<[T]>) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        todo!()
    }

    fn alloc(&self, len: usize) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        todo!()
    }

    fn copy_from_slice(&self, dst: &mut Self::Buf, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn read(&self, buf: &Self::Buf, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        todo!()
    }

    fn write(&self, buf: &mut Self::Buf, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn len(&self, buf: &Self::Buf) -> usize {
        todo!()
    }

    fn copy(&self, src: &Self::Buf) -> Result<Self::Buf, crate::core::tensor::TensorError> {
        todo!()
    }

    fn dump(&self, src: &Self::Buf) -> Result<Box<[T]>, crate::core::tensor::TensorError> {
        todo!()
    }

    fn new() -> Self {
        todo!()
    }

    fn apply_elementwise_contiguous(
        &self, buf: &mut Self::Buf, 
        op: (crate::ops::base::OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn apply_elementwise_1d_strided(
        &self, buf: &mut Self::Buf, 
        op: (crate::ops::base::OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    fn apply_elementwise_nd(
        &self,
        buf: &mut Self::Buf,
        op: (crate::ops::base::OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }

    unsafe fn broadcast(
        &self, 
        left: (*const Self::Buf, &crate::core::MetaTensor), 
        right: (*const Self::Buf, &crate::core::MetaTensor),
        dst: (*mut Self::Buf, &crate::core::MetaTensor),
        op: crate::ops::base::OpType
    ) -> Result<(), crate::core::tensor::TensorError> {
        todo!()
    }
}