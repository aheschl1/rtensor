use serde::{Deserialize, Serialize};

use crate::{backend::remote::client::RemoteBuf, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::{DType, TensorValue}, MetaTensor}, ops::base::BinaryOpType};


#[derive(Serialize, Deserialize)]
pub(crate) struct Slice {
    pub(crate) data: Vec<u8>, // bytes
    pub(crate) dtype: DType,
}

impl<T: TensorValue> From<Slice> for Result<Box<[T]>, TensorError> {
    fn from(val: Slice) -> Self {
        val.to_boxed_slice::<T>()
    }
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
            let len = std::mem::size_of_val(slice);
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
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len))
        };
        Ok(boxed)
    }
}


impl<T: TensorValue> From<Box<[T]>> for Slice {
    fn from(boxed: Box<[T]>) -> Self {
        Slice::from_boxed_slice(boxed)
    }
}


impl<T: TensorValue> From<&[T]> for Slice {
    fn from(slice: &[T]) -> Self {
        Slice::from_slice(slice)
    }
}


#[derive(Serialize, Deserialize)]
pub(crate) struct Value {
    data: Vec<u8>, // bytes
    dtype: DType,
}

impl<T: TensorValue> From<Value> for Result<T, TensorError> {
    fn from(val: Value) -> Self {
        val.to_value::<T>()
    }
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

impl<T: TensorValue> From<T> for Value {
    fn from(value: T) -> Self {
        Value::from_value(value)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub(crate) struct TypelessBuf {
    pub(crate) id: u32,
    pub(crate) dtype: DType,
}

impl<T: TensorValue> From<TypelessBuf> for Result<RemoteBuf<T>, TensorError> {
    fn from(val: TypelessBuf) -> Self {
        Ok(RemoteBuf::from_typeless(val))
    }
}


#[derive(Serialize, Deserialize)]
pub(crate) struct Response {
    pub(crate) asynchronous: bool,
    pub(crate) complete: bool,
    pub(crate) task_id: u32,
    pub(crate) message: Messages,
    pub(crate) error: Option<TensorError>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) task_id: u32,
    pub(crate) message: Messages,
}

impl Request {
    #[inline(always)]
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        debug_assert!(!self.message.is_response());
        bincode::serialize(self)
    }

    #[inline(always)]
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        let resp: Request = bincode::deserialize(data)?;
        debug_assert!(!resp.message.is_response());
        Ok(resp)
    }
}

impl Response {
    #[inline(always)]
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        debug_assert!(self.message.is_response());
        bincode::serialize(self)
    }

    #[inline(always)]
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        let resp: Response = bincode::deserialize(data)?;
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
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
);

#[derive(Serialize, Deserialize)]
pub (crate) enum Messages {
    ErrorResponse {
        message: String,
    },
    DeviceType,
    DeviceTypeResponse {
        device_type: DeviceType
    },

    AllocFromSlice {
        src: Slice
    },
    AllocFromSliceResponse(Result<TypelessBuf, TensorError>),
    
    Alloc {
        len: usize,
        dtype: DType,
    },
    AllocResponse (Result<TypelessBuf, TensorError>),

    CopyFromSlice {
        dst: TypelessBuf,
        src: Slice
    },
    CopyFromSliceResponse (Result<(), TensorError>),

    Read {
        buf: TypelessBuf,
        offset: usize,
    },
    ReadResponse (Result<Value, TensorError>,),

    Write {
        buf: TypelessBuf,
        offset: usize,
        value: Value,
    },
    WriteResponse (Result<(), TensorError>),

    Len {
        buf: TypelessBuf,
    },
    LenResponse (usize),

    Copy {
        src: TypelessBuf,
    },
    CopyResponse(Result<TypelessBuf, TensorError>),

    Dump {
        src: TypelessBuf,
    },
    DumpResponse (Result<Slice, TensorError>),

    ApplyElementwiseBinary1dStrided {
        buf: TypelessBuf,
        op: (BinaryOpType, Value),
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyElementwiseBinary1dStridedResponse (Result<(), TensorError>),

    ApplyElementwiseBinaryContiguous {
        buf: TypelessBuf,
        op: (BinaryOpType, Value),
        start: usize,
        len: usize,
    },
    ApplyElementwiseBinaryContiguousResponse (Result<(), TensorError>),

    ApplyElementwiseBinaryNd {
        buf: TypelessBuf,
        op: (BinaryOpType, Value),
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>,
    },
    ApplyElementwiseBinaryNdResponse (Result<(), TensorError>),

    Broadcast {
        left: (TypelessBuf, MetaTensor),
        right: (TypelessBuf, MetaTensor),
        dst: (TypelessBuf, MetaTensor),
        op: BinaryOpType,
    },
    BroadcastResponse (Result<(), TensorError>),

    ApplyNegContiguous {
        buf: TypelessBuf,
        start: usize,
        len: usize,
    },
    ApplyNegContiguousResponse (Result<(), TensorError>),

    ApplyNeg1dStrided {
        buf: TypelessBuf,
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyNeg1dStridedResponse (Result<(), TensorError>),

    ApplyNegNd {
        buf: TypelessBuf,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>,
    },
    ApplyNegNdResponse (Result<(), TensorError>),

    Matmul {
        lhs: (TypelessBuf, MetaTensor),
        rhs: (TypelessBuf, MetaTensor),
        dst: TypelessBuf,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: ContiguityTypes
    },
    MatmulResponse (Result<(), TensorError>),

    ApplyReluNd {
        buf: TypelessBuf,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>
    },
    ApplyReluNdResponse(Result<(), TensorError>),

    ApplyRelu1dStrided {
        buf: TypelessBuf,
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyRelu1dStridedResponse(Result<(), TensorError>),

    ApplyReluContiguous {
        buf: TypelessBuf,
        offset: usize,
        len: usize,
    },
    ApplyReluContiguousResponse(Result<(), TensorError>),

    ApplySigmoidNd {
        buf: TypelessBuf,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>
    },
    ApplySigmoidNdResponse(Result<(), TensorError>),

    ApplySigmoid1dStrided {
        buf: TypelessBuf,
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplySigmoid1dStridedResponse(Result<(), TensorError>),

    ApplySigmoidContiguous {
        buf: TypelessBuf,
        offset: usize,
        len: usize,
    },
    ApplySigmoidContiguousResponse(Result<(), TensorError>),

    ApplyTanhNd {
        buf: TypelessBuf,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>
    },
    ApplyTanhNdResponse(Result<(), TensorError>),

    ApplyTanh1dStrided {
        buf: TypelessBuf,
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyTanh1dStridedResponse(Result<(), TensorError>),

    ApplyTanhContiguous {
        buf: TypelessBuf,
        offset: usize,
        len: usize,
    },
    ApplyTanhContiguousResponse(Result<(), TensorError>),

    CopyRangeWithin {
        dst: TypelessBuf,
        src: TypelessBuf,
        dst_offset: usize,
        src_offset: usize,
        len: usize,
    },
    CopyRangeWithinResponse(Result<(), TensorError>),

    ActionCompleted(u32)

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
            Messages::ApplyElementwiseBinary1dStridedResponse { .. } |
            Messages::ApplyElementwiseBinaryContiguousResponse { .. } |
            Messages::ApplyElementwiseBinaryNdResponse { .. } |
            Messages::BroadcastResponse { .. } |
            Messages::MatmulResponse { .. } |
            Messages::ApplyNeg1dStridedResponse { .. } |
            Messages::ApplyNegContiguousResponse { .. } |
            Messages::ApplyNegNdResponse { .. } |
            Messages::ErrorResponse { .. } |
            Messages::ActionCompleted { .. } |
            Messages::ApplyReluNdResponse { .. } |
            Messages::ApplyRelu1dStridedResponse { .. } |
            Messages::ApplyReluContiguousResponse { .. } |
            Messages::ApplySigmoidNdResponse { .. } |
            Messages::ApplySigmoid1dStridedResponse { .. } |
            Messages::ApplySigmoidContiguousResponse { .. } |
            Messages::ApplyTanhNdResponse { .. } |
            Messages::ApplyTanh1dStridedResponse { .. } |
            Messages::ApplyTanhContiguousResponse { .. } |
            Messages::CopyRangeWithinResponse { .. } => true,
            _ => false,
        }
    }
}