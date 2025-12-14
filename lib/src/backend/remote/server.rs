use std::{collections::HashMap, io::{Read, Write}, net::IpAddr, sync::{atomic::AtomicU32, Arc, RwLock}, thread};

use crate::{backend::{cpu::Cpu, remote::{client::RemoteBuf, protocol::{Messages, Request, Response, Slice, TypelessBuf}}, Backend, BackendMatMul, ContiguityTypes}, core::{primitives::DeviceType, tensor::TensorError, value::DType, MetaTensor}};
#[cfg(feature = "cuda")]
use crate::backend::cuda::Cuda;

pub(crate) struct RemoteServer {
    address: IpAddr,
    port: u16
}

// this is pure evil
struct BufferCollection<B:Backend> {
    u8_buffers: HashMap<u32, B::Buf<u8>>,
    u16_buffers: HashMap<u32, B::Buf<u16>>,
    u32_buffers: HashMap<u32, B::Buf<u32>>,
    u64_buffers: HashMap<u32, B::Buf<u64>>,
    u128_buffers: HashMap<u32, B::Buf<u128>>,
    i8_buffers: HashMap<u32, B::Buf<i8>>,
    i16_buffers: HashMap<u32, B::Buf<i16>>,
    i32_buffers: HashMap<u32, B::Buf<i32>>,
    i64_buffers: HashMap<u32, B::Buf<i64>>,
    i128_buffers: HashMap<u32, B::Buf<i128>>,
    f32_buffers: HashMap<u32, B::Buf<f32>>,
    f64_buffers: HashMap<u32, B::Buf<f64>>,
}

impl<B: Backend> Default for BufferCollection<B> {
    fn default() -> Self {
        Self {
            u8_buffers: HashMap::new(),
            u16_buffers: HashMap::new(),
            u32_buffers: HashMap::new(),
            u64_buffers: HashMap::new(),
            u128_buffers: HashMap::new(),
            i8_buffers: HashMap::new(),
            i16_buffers: HashMap::new(),
            i32_buffers: HashMap::new(),
            i64_buffers: HashMap::new(),
            i128_buffers: HashMap::new(),
            f32_buffers: HashMap::new(),
            f64_buffers: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct ClientConnection {
    #[cfg(feature = "cuda")]
    cuda_buffers: Arc<RwLock<BufferCollection<Cuda>>>,
    cpu_buffers: Arc<RwLock<BufferCollection<Cpu>>>,
    cpu: Cpu,
    next_buffer_id: Arc<AtomicU32>,
    #[cfg(feature = "cuda")]
    cuda: Cuda,
}

impl RemoteServer {
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            address,
            port,
        }
    }

    pub fn serve(&mut self) -> std::io::Result<()> {
        let listener = std::net::TcpListener::bind((self.address, self.port))?;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let connection = ClientConnection { 
                        cpu_buffers: Arc::new(RwLock::new(BufferCollection::default())),
                        #[cfg(feature = "cuda")]
                        cuda_buffers: Arc::new(RwLock::new(BufferCollection::default())),
                        cpu: Cpu::new(),
                        next_buffer_id: Arc::new(AtomicU32::new(0)),
                        #[cfg(feature = "cuda")]
                        cuda: Cuda::new(),
                    };
                    // launch a new thread 
                    std::thread::spawn(move || {
                        handle_connection(connection, stream);
                    });
                }
                Err(e) => {
                    eprintln!("Connection failed: {}", e);
                }
            }
        }
        Ok(())
    }
}


#[inline(always)]
fn select_buffer(_connection: &ClientConnection) -> DeviceType {
    // For simplicity, we always use CPU buffers in this example.
    DeviceType::Cpu
}

macro_rules! alloc_from_slice_for_dtype {
    ($slice:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let boxed_slice = $slice.to_boxed_slice::<$rust_type>()?;
        let device_type = select_buffer($connection);
        let buffer = match device_type {
            DeviceType::Cpu => {
                let buf = $connection.cpu.alloc_from_slice(boxed_slice)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cpu_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buf = $connection.cuda.alloc_from_slice(boxed_slice)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cuda_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        TypelessBuf::from(buffer)
    }};
}

macro_rules! alloc_for_dtype {
    ($len:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let buffer = match device_type {
            DeviceType::Cpu => {
                let buf = $connection.cpu.alloc::<$rust_type>($len)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cpu_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buf = $connection.cuda.alloc::<$rust_type>($len)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                $connection.cuda_buffers.write().unwrap().$buffer_field.insert(buffer_id, buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        TypelessBuf::from(buffer)
    }};
}

macro_rules! copy_from_slice_for_dtype {
    ($dst_id:expr, $src_slice:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let boxed_slice = $src_slice.to_boxed_slice::<$rust_type>()?;
        let src_slice_ref: &[$rust_type] = &boxed_slice;
        let device_type = select_buffer($connection);
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                let dst_buf = buffers.$buffer_field
                    .get_mut(&$dst_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $dst_id)))?;
                $connection.cpu.copy_from_slice(dst_buf, src_slice_ref)?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                let dst_buf = buffers.$buffer_field
                    .get_mut(&$dst_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $dst_id)))?;
                $connection.cuda.copy_from_slice(dst_buf, src_slice_ref)?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

macro_rules! read_for_dtype {
    ($buf_id:expr, $offset:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let value = match device_type {
            DeviceType::Cpu => {
                let buffers = $connection.cpu_buffers.read().unwrap();
                let buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.read(buf, $offset)?
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buffers = $connection.cuda_buffers.read().unwrap();
                let buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.read(buf, $offset)?
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        crate::backend::remote::protocol::Value::from_value(value)
    }};
}

macro_rules! write_for_dtype {
    ($buf_id:expr, $offset:expr, $value:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let typed_value = $value.to_value::<$rust_type>()?;
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.write(buf, $offset, typed_value)?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.write(buf, $offset, typed_value)?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

macro_rules! len_for_dtype {
    ($buf_id:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        match device_type {
            DeviceType::Cpu => {
                let buffers = $connection.cpu_buffers.read().unwrap();
                let buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.len(buf)
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buffers = $connection.cuda_buffers.read().unwrap();
                let buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.len(buf)
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
    }};
}

macro_rules! copy_for_dtype {
    ($buf_id:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let buffer = match device_type {
            DeviceType::Cpu => {
                let buffers = $connection.cpu_buffers.read().unwrap();
                let src_buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                let new_buf = $connection.cpu.copy(src_buf)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(buffers);
                $connection.cpu_buffers.write().unwrap().$buffer_field.insert(buffer_id, new_buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buffers = $connection.cuda_buffers.read().unwrap();
                let src_buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                let new_buf = $connection.cuda.copy(src_buf)?;
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(buffers);
                $connection.cuda_buffers.write().unwrap().$buffer_field.insert(buffer_id, new_buf);
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        TypelessBuf::from(buffer)
    }};
}

macro_rules! dump_for_dtype {
    ($buf_id:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let boxed_slice = match device_type {
            DeviceType::Cpu => {
                let buffers = $connection.cpu_buffers.read().unwrap();
                let buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.dump(buf)?
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buffers = $connection.cuda_buffers.read().unwrap();
                let buf = buffers.$buffer_field
                    .get(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.dump(buf)?
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        Slice::from_boxed_slice(boxed_slice)
    }};
}

macro_rules! apply_elementwise_contiguous_for_dtype {
    ($buf_id:expr, $op:expr, $value:expr, $start:expr, $len:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let typed_value = $value.to_value::<$rust_type>()?;
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.apply_elementwise_contiguous(buf, ($op, typed_value), $start, $len)?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.apply_elementwise_contiguous(buf, ($op, typed_value), $start, $len)?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

macro_rules! apply_elementwise_1d_strided_for_dtype {
    ($buf_id:expr, $op:expr, $value:expr, $offset:expr, $stride:expr, $len:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let typed_value = $value.to_value::<$rust_type>()?;
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.apply_elementwise_1d_strided(buf, ($op, typed_value), $offset, $stride, $len)?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.apply_elementwise_1d_strided(buf, ($op, typed_value), $offset, $stride, $len)?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

macro_rules! apply_elementwise_nd_for_dtype {
    ($buf_id:expr, $op:expr, $value:expr, $offset:expr, $shape:expr, $stride:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let typed_value = $value.to_value::<$rust_type>()?;
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cpu.apply_elementwise_nd(buf, ($op, typed_value), $offset, $shape, $stride)?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                let buf = buffers.$buffer_field
                    .get_mut(&$buf_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Buffer {} not found", $buf_id)))?;
                $connection.cuda.apply_elementwise_nd(buf, ($op, typed_value), $offset, $shape, $stride)?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

macro_rules! broadcast_for_dtype {
    ($left_id:expr, $left_meta:expr, $right_id:expr, $right_meta:expr, $dst_id:expr, $dst_meta:expr, $op:expr, $connection:expr, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        match device_type {
            DeviceType::Cpu => {
                let buffers = $connection.cpu_buffers.read().unwrap();
                let left_buf = buffers.$buffer_field
                    .get(&$left_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Left buffer {} not found", $left_id)))?;
                let right_buf = buffers.$buffer_field
                    .get(&$right_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Right buffer {} not found", $right_id)))?;
                let dst_buf = buffers.$buffer_field
                    .get(&$dst_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Dst buffer {} not found", $dst_id)))?;
                
                unsafe {
                    $connection.cpu.broadcast(
                        (left_buf as *const _, $left_meta),
                        (right_buf as *const _, $right_meta),
                        (dst_buf as *const _ as *mut _, $dst_meta),
                        $op
                    )?;
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buffers = $connection.cuda_buffers.read().unwrap();
                let left_buf = buffers.$buffer_field
                    .get(&$left_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Left buffer {} not found", $left_id)))?;
                let right_buf = buffers.$buffer_field
                    .get(&$right_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Right buffer {} not found", $right_id)))?;
                let dst_buf = buffers.$buffer_field
                    .get(&$dst_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("Dst buffer {} not found", $dst_id)))?;
                
                unsafe {
                    $connection.cuda.broadcast(
                        (left_buf as *const _, $left_meta),
                        (right_buf as *const _, $right_meta),
                        (dst_buf as *const _ as *mut _, $dst_meta),
                        $op
                    )?;
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        }
        Ok(())
    }};
}

macro_rules! matmul_for_dtype {
    ($lhs_id:expr, $lhs_meta:expr, $rhs_id:expr, $rhs_meta:expr, $b:expr, $m:expr, $k:expr, $n:expr, $contiguity:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        let buffer = match device_type {
            DeviceType::Cpu => {
                let buffers = $connection.cpu_buffers.read().unwrap();
                let lhs_buf = buffers.$buffer_field
                    .get(&$lhs_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("LHS buffer {} not found", $lhs_id)))?;
                let rhs_buf = buffers.$buffer_field
                    .get(&$rhs_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("RHS buffer {} not found", $rhs_id)))?;
                
                let result_buf = $connection.cpu.matmul(
                    (lhs_buf, $lhs_meta),
                    (rhs_buf, $rhs_meta),
                    $b, $m, $k, $n,
                    $contiguity
                )?;
                
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(buffers);
                $connection.cpu_buffers.write().unwrap().$buffer_field.insert(buffer_id, result_buf);
                
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let buffers = $connection.cuda_buffers.read().unwrap();
                let lhs_buf = buffers.$buffer_field
                    .get(&$lhs_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("LHS buffer {} not found", $lhs_id)))?;
                let rhs_buf = buffers.$buffer_field
                    .get(&$rhs_id)
                    .ok_or_else(|| TensorError::RemoteError(format!("RHS buffer {} not found", $rhs_id)))?;
                
                let result_buf = $connection.cuda.matmul(
                    (lhs_buf, $lhs_meta),
                    (rhs_buf, $rhs_meta),
                    $b, $m, $k, $n,
                    $contiguity
                )?;
                
                let buffer_id = $connection.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(buffers);
                $connection.cuda_buffers.write().unwrap().$buffer_field.insert(buffer_id, result_buf);
                
                RemoteBuf {
                    id: buffer_id,
                    dtype: DType::$dtype_variant,
                    _marker: std::marker::PhantomData::<$rust_type>,
                }
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
        TypelessBuf::from(buffer)
    }};
}

#[inline(always)]
fn handle_alloc_from_slice(
    slice: Slice,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let remote_buf = match slice.dtype {
        DType::U8 => alloc_from_slice_for_dtype!(slice, connection, U8, u8, u8_buffers),
        DType::U16 => alloc_from_slice_for_dtype!(slice, connection, U16, u16, u16_buffers),
        DType::U32 => alloc_from_slice_for_dtype!(slice, connection, U32, u32, u32_buffers),
        DType::U64 => alloc_from_slice_for_dtype!(slice, connection, U64, u64, u64_buffers),
        DType::U128 => alloc_from_slice_for_dtype!(slice, connection, U128, u128, u128_buffers),
        DType::I8 => alloc_from_slice_for_dtype!(slice, connection, I8, i8, i8_buffers),
        DType::I16 => alloc_from_slice_for_dtype!(slice, connection, I16, i16, i16_buffers),
        DType::I32 => alloc_from_slice_for_dtype!(slice, connection, I32, i32, i32_buffers),
        DType::I64 => alloc_from_slice_for_dtype!(slice, connection, I64, i64, i64_buffers),
        DType::I128 => alloc_from_slice_for_dtype!(slice, connection, I128, i128, i128_buffers),
        DType::F32 => alloc_from_slice_for_dtype!(slice, connection, F32, f32, f32_buffers),
        DType::F64 => alloc_from_slice_for_dtype!(slice, connection, F64, f64, f64_buffers),
    };
    Ok(remote_buf)
}

#[inline(always)]
fn handle_alloc(
    len: usize,
    dtype: DType,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let remote_buf = match dtype {
        DType::U8 => alloc_for_dtype!(len, connection, U8, u8, u8_buffers),
        DType::U16 => alloc_for_dtype!(len, connection, U16, u16, u16_buffers),
        DType::U32 => alloc_for_dtype!(len, connection, U32, u32, u32_buffers),
        DType::U64 => alloc_for_dtype!(len, connection, U64, u64, u64_buffers),
        DType::U128 => alloc_for_dtype!(len, connection, U128, u128, u128_buffers),
        DType::I8 => alloc_for_dtype!(len, connection, I8, i8, i8_buffers),
        DType::I16 => alloc_for_dtype!(len, connection, I16, i16, i16_buffers),
        DType::I32 => alloc_for_dtype!(len, connection, I32, i32, i32_buffers),
        DType::I64 => alloc_for_dtype!(len, connection, I64, i64, i64_buffers),
        DType::I128 => alloc_for_dtype!(len, connection, I128, i128, i128_buffers),
        DType::F32 => alloc_for_dtype!(len, connection, F32, f32, f32_buffers),
        DType::F64 => alloc_for_dtype!(len, connection, F64, f64, f64_buffers),
    };
    Ok(remote_buf)
}

#[inline(always)]
fn handle_copy_from_slice(
    dst: TypelessBuf,
    src: Slice,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match dst.dtype {
        DType::U8 => copy_from_slice_for_dtype!(dst.id, src, connection, u8, u8_buffers),
        DType::U16 => copy_from_slice_for_dtype!(dst.id, src, connection, u16, u16_buffers),
        DType::U32 => copy_from_slice_for_dtype!(dst.id, src, connection, u32, u32_buffers),
        DType::U64 => copy_from_slice_for_dtype!(dst.id, src, connection, u64, u64_buffers),
        DType::U128 => copy_from_slice_for_dtype!(dst.id, src, connection, u128, u128_buffers),
        DType::I8 => copy_from_slice_for_dtype!(dst.id, src, connection, i8, i8_buffers),
        DType::I16 => copy_from_slice_for_dtype!(dst.id, src, connection, i16, i16_buffers),
        DType::I32 => copy_from_slice_for_dtype!(dst.id, src, connection, i32, i32_buffers),
        DType::I64 => copy_from_slice_for_dtype!(dst.id, src, connection, i64, i64_buffers),
        DType::I128 => copy_from_slice_for_dtype!(dst.id, src, connection, i128, i128_buffers),
        DType::F32 => copy_from_slice_for_dtype!(dst.id, src, connection, f32, f32_buffers),
        DType::F64 => copy_from_slice_for_dtype!(dst.id, src, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_read(
    buf: TypelessBuf,
    offset: usize,
    connection: &ClientConnection,
) -> Result<crate::backend::remote::protocol::Value, TensorError> {
    let value = match buf.dtype {
        DType::U8 => read_for_dtype!(buf.id, offset, connection, u8, u8_buffers),
        DType::U16 => read_for_dtype!(buf.id, offset, connection, u16, u16_buffers),
        DType::U32 => read_for_dtype!(buf.id, offset, connection, u32, u32_buffers),
        DType::U64 => read_for_dtype!(buf.id, offset, connection, u64, u64_buffers),
        DType::U128 => read_for_dtype!(buf.id, offset, connection, u128, u128_buffers),
        DType::I8 => read_for_dtype!(buf.id, offset, connection, i8, i8_buffers),
        DType::I16 => read_for_dtype!(buf.id, offset, connection, i16, i16_buffers),
        DType::I32 => read_for_dtype!(buf.id, offset, connection, i32, i32_buffers),
        DType::I64 => read_for_dtype!(buf.id, offset, connection, i64, i64_buffers),
        DType::I128 => read_for_dtype!(buf.id, offset, connection, i128, i128_buffers),
        DType::F32 => read_for_dtype!(buf.id, offset, connection, f32, f32_buffers),
        DType::F64 => read_for_dtype!(buf.id, offset, connection, f64, f64_buffers),
    };
    Ok(value)
}

#[inline(always)]
fn handle_write(
    buf: TypelessBuf,
    offset: usize,
    value: crate::backend::remote::protocol::Value,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => write_for_dtype!(buf.id, offset, value, connection, u8, u8_buffers),
        DType::U16 => write_for_dtype!(buf.id, offset, value, connection, u16, u16_buffers),
        DType::U32 => write_for_dtype!(buf.id, offset, value, connection, u32, u32_buffers),
        DType::U64 => write_for_dtype!(buf.id, offset, value, connection, u64, u64_buffers),
        DType::U128 => write_for_dtype!(buf.id, offset, value, connection, u128, u128_buffers),
        DType::I8 => write_for_dtype!(buf.id, offset, value, connection, i8, i8_buffers),
        DType::I16 => write_for_dtype!(buf.id, offset, value, connection, i16, i16_buffers),
        DType::I32 => write_for_dtype!(buf.id, offset, value, connection, i32, i32_buffers),
        DType::I64 => write_for_dtype!(buf.id, offset, value, connection, i64, i64_buffers),
        DType::I128 => write_for_dtype!(buf.id, offset, value, connection, i128, i128_buffers),
        DType::F32 => write_for_dtype!(buf.id, offset, value, connection, f32, f32_buffers),
        DType::F64 => write_for_dtype!(buf.id, offset, value, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_len(
    buf: TypelessBuf,
    connection: &ClientConnection,
) -> Result<usize, TensorError> {
    let len = match buf.dtype {
        DType::U8 => len_for_dtype!(buf.id, connection, u8, u8_buffers),
        DType::U16 => len_for_dtype!(buf.id, connection, u16, u16_buffers),
        DType::U32 => len_for_dtype!(buf.id, connection, u32, u32_buffers),
        DType::U64 => len_for_dtype!(buf.id, connection, u64, u64_buffers),
        DType::U128 => len_for_dtype!(buf.id, connection, u128, u128_buffers),
        DType::I8 => len_for_dtype!(buf.id, connection, i8, i8_buffers),
        DType::I16 => len_for_dtype!(buf.id, connection, i16, i16_buffers),
        DType::I32 => len_for_dtype!(buf.id, connection, i32, i32_buffers),
        DType::I64 => len_for_dtype!(buf.id, connection, i64, i64_buffers),
        DType::I128 => len_for_dtype!(buf.id, connection, i128, i128_buffers),
        DType::F32 => len_for_dtype!(buf.id, connection, f32, f32_buffers),
        DType::F64 => len_for_dtype!(buf.id, connection, f64, f64_buffers),
    };
    Ok(len)
}

#[inline(always)]
fn handle_copy(
    src: TypelessBuf,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let new_buf = match src.dtype {
        DType::U8 => copy_for_dtype!(src.id, connection, U8, u8, u8_buffers),
        DType::U16 => copy_for_dtype!(src.id, connection, U16, u16, u16_buffers),
        DType::U32 => copy_for_dtype!(src.id, connection, U32, u32, u32_buffers),
        DType::U64 => copy_for_dtype!(src.id, connection, U64, u64, u64_buffers),
        DType::U128 => copy_for_dtype!(src.id, connection, U128, u128, u128_buffers),
        DType::I8 => copy_for_dtype!(src.id, connection, I8, i8, i8_buffers),
        DType::I16 => copy_for_dtype!(src.id, connection, I16, i16, i16_buffers),
        DType::I32 => copy_for_dtype!(src.id, connection, I32, i32, i32_buffers),
        DType::I64 => copy_for_dtype!(src.id, connection, I64, i64, i64_buffers),
        DType::I128 => copy_for_dtype!(src.id, connection, I128, i128, i128_buffers),
        DType::F32 => copy_for_dtype!(src.id, connection, F32, f32, f32_buffers),
        DType::F64 => copy_for_dtype!(src.id, connection, F64, f64, f64_buffers),
    };
    Ok(new_buf)
}

#[inline(always)]
fn handle_dump(
    src: TypelessBuf,
    connection: &ClientConnection,
) -> Result<Slice, TensorError> {
    let slice = match src.dtype {
        DType::U8 => dump_for_dtype!(src.id, connection, u8, u8_buffers),
        DType::U16 => dump_for_dtype!(src.id, connection, u16, u16_buffers),
        DType::U32 => dump_for_dtype!(src.id, connection, u32, u32_buffers),
        DType::U64 => dump_for_dtype!(src.id, connection, u64, u64_buffers),
        DType::U128 => dump_for_dtype!(src.id, connection, u128, u128_buffers),
        DType::I8 => dump_for_dtype!(src.id, connection, i8, i8_buffers),
        DType::I16 => dump_for_dtype!(src.id, connection, i16, i16_buffers),
        DType::I32 => dump_for_dtype!(src.id, connection, i32, i32_buffers),
        DType::I64 => dump_for_dtype!(src.id, connection, i64, i64_buffers),
        DType::I128 => dump_for_dtype!(src.id, connection, i128, i128_buffers),
        DType::F32 => dump_for_dtype!(src.id, connection, f32, f32_buffers),
        DType::F64 => dump_for_dtype!(src.id, connection, f64, f64_buffers),
    };
    Ok(slice)
}

#[inline(always)]
fn handle_apply_elementwise_contiguous(
    buf: TypelessBuf,
    op: crate::ops::base::OpType,
    value: crate::backend::remote::protocol::Value,
    start: usize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u8, u8_buffers),
        DType::U16 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u16, u16_buffers),
        DType::U32 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u32, u32_buffers),
        DType::U64 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u64, u64_buffers),
        DType::U128 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u128, u128_buffers),
        DType::I8 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i8, i8_buffers),
        DType::I16 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i16, i16_buffers),
        DType::I32 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i32, i32_buffers),
        DType::I64 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i64, i64_buffers),
        DType::I128 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i128, i128_buffers),
        DType::F32 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, f32, f32_buffers),
        DType::F64 => apply_elementwise_contiguous_for_dtype!(buf.id, op, value, start, len, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_apply_elementwise_1d_strided(
    buf: TypelessBuf,
    op: crate::ops::base::OpType,
    value: crate::backend::remote::protocol::Value,
    offset: usize,
    stride: isize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u8, u8_buffers),
        DType::U16 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u16, u16_buffers),
        DType::U32 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u32, u32_buffers),
        DType::U64 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u64, u64_buffers),
        DType::U128 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u128, u128_buffers),
        DType::I8 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i8, i8_buffers),
        DType::I16 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i16, i16_buffers),
        DType::I32 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i32, i32_buffers),
        DType::I64 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i64, i64_buffers),
        DType::I128 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i128, i128_buffers),
        DType::F32 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, f32, f32_buffers),
        DType::F64 => apply_elementwise_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_apply_elementwise_nd(
    buf: TypelessBuf,
    op: crate::ops::base::OpType,
    value: crate::backend::remote::protocol::Value,
    offset: usize,
    shape: &[usize],
    stride: &[isize],
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u8, u8_buffers),
        DType::U16 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u16, u16_buffers),
        DType::U32 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u32, u32_buffers),
        DType::U64 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u64, u64_buffers),
        DType::U128 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u128, u128_buffers),
        DType::I8 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i8, i8_buffers),
        DType::I16 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i16, i16_buffers),
        DType::I32 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i32, i32_buffers),
        DType::I64 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i64, i64_buffers),
        DType::I128 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i128, i128_buffers),
        DType::F32 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, f32, f32_buffers),
        DType::F64 => apply_elementwise_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_broadcast(
    left: (TypelessBuf, MetaTensor),
    right: (TypelessBuf, MetaTensor),
    dst: (TypelessBuf, MetaTensor),
    op: crate::ops::base::OpType,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    let (left_buf, left_meta) = left;
    let (right_buf, right_meta) = right;
    let (dst_buf, dst_meta) = dst;
    
    match dst_buf.dtype {
        DType::U8 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u8, u8_buffers),
        DType::U16 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u16, u16_buffers),
        DType::U32 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u32, u32_buffers),
        DType::U64 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u64, u64_buffers),
        DType::U128 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u128, u128_buffers),
        DType::I8 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i8, i8_buffers),
        DType::I16 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i16, i16_buffers),
        DType::I32 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i32, i32_buffers),
        DType::I64 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i64, i64_buffers),
        DType::I128 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i128, i128_buffers),
        DType::F32 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, f32, f32_buffers),
        DType::F64 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, f64, f64_buffers),
    }
}

#[inline(always)]
fn handle_matmul(
    lhs: (TypelessBuf, MetaTensor),
    rhs: (TypelessBuf, MetaTensor),
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    contiguity: ContiguityTypes,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let (lhs_buf, lhs_meta) = lhs;
    let (rhs_buf, rhs_meta) = rhs;
    
    let result_buf = match lhs_buf.dtype {
        DType::U8 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, U8, u8, u8_buffers),
        DType::U16 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, U16, u16, u16_buffers),
        DType::U32 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, U32, u32, u32_buffers),
        DType::U64 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, U64, u64, u64_buffers),
        DType::U128 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, U128, u128, u128_buffers),
        DType::I8 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, I8, i8, i8_buffers),
        DType::I16 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, I16, i16, i16_buffers),
        DType::I32 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, I32, i32, i32_buffers),
        DType::I64 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, I64, i64, i64_buffers),
        DType::I128 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, I128, i128, i128_buffers),
        DType::F32 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, F32, f32, f32_buffers),
        DType::F64 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, rhs_buf.id, &rhs_meta, b, m, k, n, contiguity, connection, F64, f64, f64_buffers),
    };
    
    Ok(result_buf)
}

#[inline(always)]
fn handle_request(
    request: Request, 
    connection: &ClientConnection,
    stream: &mut std::net::TcpStream,
) -> Result<(), TensorError> {
    let task_id = request.task_id;
    match request.message {
        Messages::DeviceType => {
            let device_type = select_buffer(connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::DeviceTypeResponse { device_type },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::AllocFromSlice { slice } => {
            let remote_buf = handle_alloc_from_slice(slice, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::AllocFromSliceResponse { buf: Ok(remote_buf) },
            };
            send_response(stream, &response)?;
            Ok(())
        },
        Messages::Alloc { len, dtype } => {
            let remote_buf = handle_alloc(len, dtype, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::AllocResponse { buf: Ok(remote_buf) },
            };
            send_response(stream, &response)?;
            Ok(())
        },
        Messages::CopyFromSlice { dst, src } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                message: Messages::CopyFromSliceResponse { result: Ok(()) },
            };
            send_response(stream, &ack_response)?;
            
            // Clone what we need for the background thread
            let connection_clone = connection.clone();
            let stream_clone = stream.try_clone()
                .map_err(|e| TensorError::RemoteError(format!("Failed to clone stream: {}", e)))?;
            
            // Spawn background task
            thread::spawn(move || {
                let result = handle_copy_from_slice(dst, src, &connection_clone);
                let completion_response = Response {
                    asynchronous: true,
                    complete: true,
                    task_id,
                    message: Messages::CopyFromSliceResponse { result },
                };
                let _ = send_response(&mut stream_clone.try_clone().unwrap(), &completion_response);
            });
            
            Ok(())
        },
        Messages::Read { buf, offset } => {
            let value = handle_read(buf, offset, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::ReadResponse { value: Ok(value) },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::Write { buf, offset, value } => {
            let result = handle_write(buf, offset, value, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::WriteResponse { result },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::Len { buf } => {
            let len = handle_len(buf, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::LenResponse { len },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::Copy { src } => {
            let new_buf = handle_copy(src, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::CopyResponse { buf: Ok(new_buf) },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::Dump { src } => {
            let slice = handle_dump(src, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::DumpResponse { data: Ok(slice) },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::ApplyElementwiseContiguous { buf, op, start, len } => {
            let (op_type, value) = op;
            let result = handle_apply_elementwise_contiguous(buf, op_type, value, start, len, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::ApplyElementwiseContiguousResponse { result },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::ApplyElementwise1DStrided { buf, op, offset, stride, len } => {
            let (op_type, value) = op;
            let result = handle_apply_elementwise_1d_strided(buf, op_type, value, offset, stride, len, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::ApplyElementwise1DStridedResponse { result },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::ApplyElementwiseND { buf, op, offset, shape, stride } => {
            let (op_type, value) = op;
            let result = handle_apply_elementwise_nd(buf, op_type, value, offset, &shape, &stride, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::ApplyElementwiseNDResponse { result },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::Broadcast { left, right, dst, op } => {
            let result = handle_broadcast(left, right, dst, op, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::BroadcastResponse { result },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        Messages::Matmul { lhs, rhs, b, m, k, n, contiguity } => {
            let result_buf = handle_matmul(lhs, rhs, b, m, k, n, contiguity, connection)?;
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                message: Messages::MatmulResponse { buf: Ok(result_buf) },
            };
            send_response(stream, &response)?;
            Ok(())
        }
        _ => {
            Err(TensorError::RemoteError("Unsupported operation".into()))
        }
    }
}

fn send_response(stream: &mut std::net::TcpStream, response: &Response) -> Result<(), TensorError> {
    let serialized = response.serialize()
        .map_err(|e| TensorError::RemoteError(format!("Failed to serialize response: {}", e)))?;
    let n = serialized.len() as u32;
    stream.write_all(&n.to_le_bytes())
        .map_err(|e| TensorError::RemoteError(format!("Failed to write response length: {}", e)))?;
    stream.write_all(&serialized)
        .map_err(|e| TensorError::RemoteError(format!("Failed to write response: {}", e)))?;
    Ok(())
}

fn handle_connection(connection: ClientConnection, mut stream: std::net::TcpStream) {
    // Handle communication with the client
    let mut n_buffer = [0u8; 4];
    loop {
        // Read from and write to connection.stream
        match stream.read_exact(&mut n_buffer) {
            Ok(_) => {
                let n = u32::from_le_bytes(n_buffer) as usize;
                let mut data_buffer = vec![0u8; n];
                match stream.read_exact(&mut data_buffer) {
                    Ok(_) => {
                        let request = Request::deserialize(&data_buffer).expect("Failed to deserialize request");
                        if let Err(e) = handle_request(request, &connection, &mut stream) {
                            eprintln!("Failed to handle request: {}", e);
                            // TODO: Send error response back to client
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to read data: {}", e);
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read size: {}", e);
                break;
            }
        }
    }
}