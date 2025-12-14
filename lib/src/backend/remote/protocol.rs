use serde::{Deserialize, Serialize};

use crate::{backend::remote::client::RemoteBuf, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, untyped::UntypedTensor, value::{DType, TensorValue}, MetaTensor}, ops::base::OpType};


#[derive(Serialize, Deserialize)]
pub(crate) struct Slice {
    pub(crate) data: Vec<u8>, // bytes
    pub(crate) dtype: DType,
}

impl Slice {
    #[inline(always)]
    pub(crate) fn from_boxed_slice<T: TensorValue>(boxed: Box<[T]>) -> Self {
        let dtype = T::DTYPE;
        let data = unsafe {
            let len = boxed.len() * std::mem::size_of::<T>();
            let ptr = Box::into_raw(boxed) as *mut u8;
            Vec::from_raw_parts(ptr, len, len)
        };
        Self { data, dtype }
    }

    #[inline(always)]
    pub(crate) fn from_slice<T: TensorValue>(slice: &[T]) -> Self {
        let dtype = T::DTYPE;
        let data = unsafe {
            let len = slice.len() * std::mem::size_of::<T>();
            let ptr = slice.as_ptr() as *const u8;
            let mut vec = Vec::with_capacity(len);
            vec.set_len(len);
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
            vec
        };
        Self { data, dtype }
    }

    #[inline(always)]
    pub(crate) fn to_boxed_slice<T: TensorValue>(self) -> Result<Box<[T]>, TensorError> {
        if self.dtype != T::DTYPE {
            return Err(TensorError::BackendError(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::DTYPE, self.dtype
            )));
        }
        let boxed = unsafe {
            let len = self.data.len() / std::mem::size_of::<T>();
            let ptr = self.data.as_ptr() as *mut T;
            std::mem::forget(self.data);
            Box::from_raw(std::slice::from_raw_parts_mut(ptr, len))
        };
        Ok(boxed)
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Value {
    data: Vec<u8>, // bytes
    dtype: DType,
}

impl Value {
    #[inline(always)]
    pub(crate) fn from_value<T: TensorValue>(value: T) -> Self {
        let dtype = T::DTYPE;
        let data = unsafe {
            let size = std::mem::size_of::<T>();
            let ptr = &value as *const T as *const u8;
            let mut vec = Vec::with_capacity(size);
            vec.set_len(size);
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), size);
            vec
        };
        Self { data, dtype }
    }

    #[inline(always)]
    pub(crate) fn to_value<T: TensorValue>(self) -> Result<T, TensorError> {
        if self.dtype != T::DTYPE {
            return Err(TensorError::BackendError(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::DTYPE, self.dtype
            )));
        }
        let value = unsafe {
            let ptr = self.data.as_ptr() as *const T;
            std::ptr::read(ptr)
        };
        Ok(value)
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TypelessBuf {
    pub(crate) id: u32,
    pub(crate) dtype: DType,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Response {
    pub(crate) asynchronous: bool,
    pub(crate) complete: bool,
    pub(crate) task_id: u32,
    pub(crate) message: Messages,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) task_id: u32,
    pub(crate) message: Messages,
}

impl Request {
    #[inline(always)]
    pub fn serialize(&self) -> Result<Vec<u8>, serde_json::Error> {
        debug_assert!(!self.message.is_response());
        serde_json::to_vec(self)
    }

    #[inline(always)]
    pub fn deserialize(data: &[u8]) -> Result<Self, serde_json::Error> {
        let resp: Request = serde_json::from_slice(data)?;
        debug_assert!(!resp.message.is_response());
        Ok(resp)
    }
}

impl Response {
    #[inline(always)]
    pub fn serialize(&self) -> Result<Vec<u8>, serde_json::Error> {
        debug_assert!(self.message.is_response());
        serde_json::to_vec(self)
    }

    #[inline(always)]
    pub fn deserialize(data: &[u8]) -> Result<Self, serde_json::Error> {
        let resp: Response = serde_json::from_slice(data)?;
        debug_assert!(resp.message.is_response());
        Ok(resp)
    }
}

impl<T: TensorValue> From<RemoteBuf<T>> for TypelessBuf {
    fn from(buf: RemoteBuf<T>) -> Self {
        Self {
            id: buf.id,
            dtype: buf.dtype,
        }
    }
}

macro_rules! impl_typeless_buf_conversions {
    ($($type:ty),+ $(,)?) => {
        $(
            impl From<TypelessBuf> for RemoteBuf<$type> {
                fn from(buf: TypelessBuf) -> Self {
                    Self {
                        id: buf.id,
                        dtype: buf.dtype,
                        _marker: std::marker::PhantomData::<$type>,
                    }
                }
            }
        )+
    };
}

impl_typeless_buf_conversions!(
    f32, f64,
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
);

#[derive(Serialize, Deserialize)]
pub (crate) enum Messages {
    DeviceType,
    DeviceTypeResponse {
        device_type: DeviceType
    },

    AllocFromSlice {
        slice: Slice
    },
    AllocFromSliceResponse {
        buf: Result<TypelessBuf, TensorError>
    },
    
    Alloc {
        len: usize,
        dtype: DType,
    },
    AllocResponse {
        buf: Result<TypelessBuf, TensorError>
    },

    CopyFromSlice {
        dst: TypelessBuf,
        src: Slice
    },
    CopyFromSliceResponse {
        result: Result<(), TensorError>
    },

    Read {
        buf: TypelessBuf,
        offset: usize,
    },
    ReadResponse {
        value: Result<Value, TensorError>,
    },

    Write {
        buf: TypelessBuf,
        offset: usize,
        value: Value,
    },
    WriteResponse {
        result: Result<(), TensorError>,
    },

    Len {
        buf: TypelessBuf,
    },
    LenResponse {
        len: usize,
    },

    Copy {
        src: TypelessBuf,
    },
    CopyResponse {
        buf: Result<TypelessBuf, TensorError>,
    },

    Dump {
        src: TypelessBuf,
    },
    DumpResponse {
        data: Result<Slice, TensorError>,
    },

    ApplyElementwise1DStrided {
        buf: TypelessBuf,
        op: (OpType, Value),
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyElementwise1DStridedResponse {
        result: Result<(), TensorError>,
    },

    ApplyElementwiseContiguous {
        buf: TypelessBuf,
        op: (OpType, Value),
        start: usize,
        len: usize,
    },
    ApplyElementwiseContiguousResponse {
        result: Result<(), TensorError>,
    },

    ApplyElementwiseND {
        buf: TypelessBuf,
        op: (OpType, Value),
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>,
    },
    ApplyElementwiseNDResponse {
        result: Result<(), TensorError>,
    },

    Broadcast {
        left: (TypelessBuf, MetaTensor),
        right: (TypelessBuf, MetaTensor),
        dst: (TypelessBuf, MetaTensor),
        op: OpType,
    },
    BroadcastResponse {
        result: Result<(), TensorError>,
    },

    ApplyElementwise {
        buf: TypelessBuf,
        op: (OpType, Value),
        meta: MetaTensor,
    },
    ApplyElementwiseResponse {
        result: Result<(), TensorError>,
    },

    Matmul {
        lhs: (TypelessBuf, MetaTensor),
        rhs: (TypelessBuf, MetaTensor),
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: ContiguityTypes
    },
    MatmulResponse {
        buf: Result<TypelessBuf, TensorError>,
    },

    ActionCompleted {
        task_id: u32,
    }

}

impl Messages {
    #[inline(always)]
    pub fn is_response(&self) -> bool {
        match self {
            Messages::DeviceTypeResponse { .. } |
            Messages::AllocFromSliceResponse { .. } |
            Messages::AllocResponse { .. } |
            Messages::CopyFromSliceResponse { .. } |
            Messages::ReadResponse { .. } |
            Messages::WriteResponse { .. } |
            Messages::LenResponse { .. } |
            Messages::CopyResponse { .. } |
            Messages::DumpResponse { .. } |
            Messages::ApplyElementwise1DStridedResponse { .. } |
            Messages::ApplyElementwiseContiguousResponse { .. } |
            Messages::ApplyElementwiseNDResponse { .. } |
            Messages::BroadcastResponse { .. } |
            Messages::ApplyElementwiseResponse { .. } |
            Messages::MatmulResponse { .. } => true,
            _ => false,
        }
    }
}