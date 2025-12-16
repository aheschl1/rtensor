use std::{collections::HashMap, io::{Read, Write}, net::IpAddr, sync::{atomic::AtomicU32, Arc, RwLock}, thread::{self, JoinHandle}};

use flume::Receiver;

use crate::{backend::{cpu::Cpu, remote::{enumdispatch::{dispatch_alloc, dispatch_alloc_from_slice, dispatch_apply_elementwise_1d_strided, dispatch_apply_elementwise_contiguous, dispatch_apply_elementwise_nd, dispatch_broadcast, dispatch_copy, dispatch_copy_from_slice, dispatch_dump, dispatch_len, dispatch_matmul, dispatch_read, dispatch_write}, protocol::{Messages, Request, Response, Slice, TypelessBuf}}, Backend}, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::types, MetaTensor}};
#[cfg(feature = "cuda")]
use crate::backend::cuda::Cuda;

pub(crate) struct RemoteServer {
    address: IpAddr,
    port: u16
}

// this is pure evil
pub(crate) struct BufferCollection<B:Backend> {
    pub(crate) u8_buffers: HashMap<u32, B::Buf<u8>>,
    pub(crate) u16_buffers: HashMap<u32, B::Buf<u16>>,
    pub(crate) u32_buffers: HashMap<u32, B::Buf<u32>>,
    pub(crate) u64_buffers: HashMap<u32, B::Buf<u64>>,
    pub(crate) u128_buffers: HashMap<u32, B::Buf<u128>>,
    pub(crate) i8_buffers: HashMap<u32, B::Buf<i8>>,
    pub(crate) i16_buffers: HashMap<u32, B::Buf<i16>>,
    pub(crate) i32_buffers: HashMap<u32, B::Buf<i32>>,
    pub(crate) i64_buffers: HashMap<u32, B::Buf<i64>>,
    pub(crate) i128_buffers: HashMap<u32, B::Buf<i128>>,
    pub(crate) f32_buffers: HashMap<u32, B::Buf<f32>>,
    pub(crate) f64_buffers: HashMap<u32, B::Buf<f64>>,
    pub(crate) bool_buffers: HashMap<u32, B::Buf<types::boolean>>,
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
            bool_buffers: HashMap::new(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ClientConnection {
    pub(crate) output_messages_sender: flume::Sender<Response>,
    pub(crate) output_messages_receiver: flume::Receiver<Response>,
    pub(crate) background_tasks_receiver: flume::Receiver<AsyncJob>,
    pub(crate) background_tasks_sender: flume::Sender<AsyncJob>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_buffers: Arc<RwLock<BufferCollection<Cuda>>>,
    pub(crate) cpu_buffers: Arc<RwLock<BufferCollection<Cpu>>>,
    pub(crate) cpu: Cpu,
    pub(crate) next_buffer_id: Arc<AtomicU32>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Cuda,
}

impl ClientConnection {
    pub fn new() -> Self {
        let (output_messages_sender, output_messages_receiver) = flume::unbounded();
        let (background_tasks_sender, background_tasks_receiver) = flume::unbounded();
        Self {
            output_messages_sender,
            output_messages_receiver,
            background_tasks_receiver,
            background_tasks_sender,
            cpu_buffers: Arc::new(RwLock::new(BufferCollection::default())),
            #[cfg(feature = "cuda")]
            cuda_buffers: Arc::new(RwLock::new(BufferCollection::default())),
            cpu: Cpu::new(),
            next_buffer_id: Arc::new(AtomicU32::new(0)),
            #[cfg(feature = "cuda")]
            cuda: Cuda::new(),
        }
    }

    pub fn queue_response(&self, response: Response) -> Result<(), TensorError> {
        self.output_messages_sender.send(response).map_err(|e| TensorError::RemoteError(format!("Failed to send response: {}", e)))
    }

    pub fn queue_job(&self, job: AsyncJob) -> Result<(), TensorError> {
        self.background_tasks_sender.send(job).map_err(|e| TensorError::RemoteError(format!("Failed to send job: {}", e)))
    }
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
                    let connection = ClientConnection::new();
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
pub(crate) fn select_buffer(_connection: &ClientConnection) -> DeviceType {
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
                let buffers = $connection.cpu_buffers.write().unwrap();
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
                let buffers = $connection.cuda_buffers.write().unwrap();
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
    ($lhs_id:expr, $lhs_meta:expr, $rhs_id:expr, $rhs_meta:expr, $dst_id:expr, $b:expr, $m:expr, $k:expr, $n:expr, $contiguity:expr, $connection:expr, $dtype_variant:ident, $rust_type:ty, $buffer_field:ident) => {{
        let device_type = select_buffer($connection);
        match device_type {
            DeviceType::Cpu => {
                let mut buffers = $connection.cpu_buffers.write().unwrap();
                // let lhs_buf = buffers.$buffer_field
                //     .get(&$lhs_id)
                //     .ok_or_else(|| TensorError::RemoteError(format!("LHS buffer {} not found", $lhs_id)))?;
                // let rhs_buf = buffers.$buffer_field
                //     .get(&$rhs_id)
                //     .ok_or_else(|| TensorError::RemoteError(format!("RHS buffer {} not found", $rhs_id)))?;
                // let dst_buf = buffers.$buffer_field
                //     .get_mut(&$dst_id)
                //     .ok_or_else(|| TensorError::RemoteError(format!("DST buffer {} not found", $dst_id)))?;
                let [Some(lhs_buf), Some(rhs_buf), Some(mut dst_buf)] = buffers.$buffer_field.get_disjoint_mut([&$lhs_id, &$rhs_id, &$dst_id]) else {
                    return Err(TensorError::RemoteError("Buffers missing.".into()));
                };
                $connection.cpu.matmul(
                    (lhs_buf, $lhs_meta),
                    (rhs_buf, $rhs_meta),
                    &mut dst_buf,
                    $b, $m, $k, $n,
                    $contiguity
                )?;
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                let mut buffers = $connection.cuda_buffers.write().unwrap();
                // let lhs_buf = buffers.$buffer_field
                //     .get(&$lhs_id)
                //     .ok_or_else(|| TensorError::RemoteError(format!("LHS buffer {} not found", $lhs_id)))?;
                // let rhs_buf = buffers.$buffer_field
                //     .get(&$rhs_id)
                //     .ok_or_else(|| TensorError::RemoteError(format!("RHS buffer {} not found", $rhs_id)))?;
                // let mut dst_buf = buffers.$buffer_field
                //     .get(&$dst_id)
                //     .ok_or_else(|| TensorError::RemoteError(format!("DST buffer {} not found", $dst_id)))?;
                let [Some(lhs_buf), Some(rhs_buf), Some(mut dst_buf)] = buffers.$buffer_field.get_disjoint_mut([&$lhs_id, &$rhs_id, &$dst_id]) else {
                    return Err(TensorError::RemoteError("Buffers missing.".into()));
                };
                $connection.cuda.matmul(
                    (lhs_buf, $lhs_meta),
                    (rhs_buf, $rhs_meta),
                    &mut dst_buf,
                    $b, $m, $k, $n,
                    $contiguity
                )?;
            },
            _ => {
                return Err(TensorError::RemoteError("Unsupported device type".into()));
            }
        };
    }};
}

pub(crate) enum AsyncJob {
    CopyFromSlice {
        task_id: u32,
        dst: TypelessBuf,
        src: Slice,
    },
    ApplyElementwiseContiguous {
        task_id: u32,
        buf: TypelessBuf,
        op: (crate::ops::base::OpType, crate::backend::remote::protocol::Value),
        start: usize,
        len: usize,
    },
    ApplyElementwise1DStrided {
        task_id: u32,
        buf: TypelessBuf,
        op: (crate::ops::base::OpType, crate::backend::remote::protocol::Value),
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyElementwiseND {
        task_id: u32,
        buf: TypelessBuf,
        op: (crate::ops::base::OpType, crate::backend::remote::protocol::Value),
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>,
    },
    Broadcast {
        task_id: u32,
        left: (TypelessBuf, MetaTensor),
        right: (TypelessBuf, MetaTensor),
        dst: (TypelessBuf, MetaTensor),
        op: crate::ops::base::OpType,
    },
    MatMul {
        task_id: u32,
        lhs: (TypelessBuf, MetaTensor),
        rhs: (TypelessBuf, MetaTensor),
        dst: TypelessBuf,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: ContiguityTypes
    },
}


#[inline(always)]
fn handle_request(
    request: Request, 
    connection: &ClientConnection,
){
    let task_id = request.task_id;
    match request.message {
        Messages::DeviceType => {
            let device_type = select_buffer(connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: None,
                message: Messages::DeviceTypeResponse { device_type },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
        Messages::AllocFromSlice { slice } => {
            let remote_buf = dispatch_alloc_from_slice(slice, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,                
                error: remote_buf.as_ref().err().cloned(),
                message: Messages::AllocFromSliceResponse { buf: remote_buf },
            };
            connection.queue_response(response).expect("Failed to send message");
        },
        Messages::Alloc { len, dtype } => {
            let remote_buf = dispatch_alloc(len, dtype, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: remote_buf.as_ref().err().cloned(),
                message: Messages::AllocResponse { buf: remote_buf },
            };
            connection.queue_response(response).expect("Failed to send message");
        },
        Messages::CopyFromSlice { dst, src } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                error: None,
                message: Messages::CopyFromSliceResponse { result: Ok(()) },
            };
            connection.queue_response(ack_response).expect("Failed to send message");
            
            // Clone what we need for the background thread            
            let job = AsyncJob::CopyFromSlice {
                task_id,
                dst,
                src,
            };
            connection.queue_job(job).expect("Failed to queue job");
        },
        Messages::Read { buf, offset } => {
            let value = dispatch_read(buf, offset, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: value.as_ref().err().cloned(),
                message: Messages::ReadResponse { value: value },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
        Messages::Write { buf, offset, value } => {
            let result = dispatch_write(buf, offset, value, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: result.as_ref().err().cloned(),
                message: Messages::WriteResponse { result },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
        Messages::Len { buf } => {
            let len = dispatch_len(buf, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: len.as_ref().err().cloned(),
                message: Messages::LenResponse { len: len.unwrap_or(0) },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
        Messages::Copy { src } => {
            let new_buf = dispatch_copy(src, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: new_buf.as_ref().err().cloned(),
                message: Messages::CopyResponse { buf: new_buf },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
        Messages::Dump { src } => {
            let slice = dispatch_dump(src, connection);
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: slice.as_ref().err().cloned(),
                message: Messages::DumpResponse { data: slice },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
        Messages::ApplyElementwiseContiguous { buf, op, start, len } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                error: None,
                message: Messages::ApplyElementwiseContiguousResponse { result: Ok(()) },
            };
            connection.queue_response(ack_response).expect("Failed to send message");
            
            let job = AsyncJob::ApplyElementwiseContiguous {
                task_id,
                buf,
                op,
                start,
                len,
            };
            connection.queue_job(job).expect("Failed to queue job");
        }
        Messages::ApplyElementwise1DStrided { buf, op, offset, stride, len } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                error: None,
                message: Messages::ApplyElementwise1DStridedResponse { result: Ok(()) },
            };
            connection.queue_response(ack_response).expect("Failed to send message");
            let job = AsyncJob::ApplyElementwise1DStrided {
                task_id,
                buf,
                op,
                offset,
                stride,
                len,
            };
            connection.queue_job(job).expect("Failed to queue job");
        }
        Messages::ApplyElementwiseND { buf, op, offset, shape, stride } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                error: None,
                message: Messages::ApplyElementwiseNDResponse { result: Ok(()) },
            };
            connection.queue_response(ack_response).expect("Failed to send message");
            
            let job = AsyncJob::ApplyElementwiseND {
                task_id,
                buf,
                op,
                offset,
                shape,
                stride,
            };
            connection.queue_job(job).expect("Failed to queue job");
        }
        Messages::Broadcast { left, right, dst, op } => {
            // Send initial acknowledgment
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                error: None,
                message: Messages::BroadcastResponse { result: Ok(()) },
            };
            connection.queue_response(ack_response).expect("Failed to send message");
            
            let job = AsyncJob::Broadcast {
                task_id,
                left,
                right,
                dst,
                op,
            };
            connection.queue_job(job).expect("Failed to queue job");            
        }
        Messages::Matmul { lhs, rhs, dst, b, m, k, n, contiguity } => {
            // let result = dispatch_matmul(lhs, rhs, dst, b, m, k, n, contiguity, connection);
            // let response = Response {
            //     asynchronous: false,
            //     complete: true,
            //     task_id,
            //     error: result.as_ref().err().cloned(),
            //     message: Messages::MatmulResponse { result: result },
            // };
            // connection.queue_response(response).expect("Failed to send message");
            let ack_response = Response {
                asynchronous: true,
                complete: false,
                task_id,
                error: None,
                message: Messages::MatmulResponse { result: Ok(()) },
            };
            connection.queue_response(ack_response).expect("Failed to send message");
            let job = AsyncJob::MatMul {
                task_id,
                lhs,
                rhs,
                dst,
                b,
                m,
                k,
                n,
                contiguity,
            };
            connection.queue_job(job).expect("Failed to queue job");
            // let result = dispatch_matmul(lhs, rhs, dst, b, m, k, n, contiguity, connection);
        }
        _ => {
            let response = Response {
                asynchronous: false,
                complete: true,
                task_id,
                error: Some(TensorError::RemoteError("Unsupported request".to_string())),
                message: Messages::ErrorResponse { message: "Unsupported request".to_string() },
            };
            connection.queue_response(response).expect("Failed to send message");
        }
    }
}


fn drain_messages(mut stream: std::net::TcpStream, receiver: Receiver<Response>) {
    loop {
        match receiver.recv() {
            Ok(response) => {
                let result: Result<(), TensorError> = || -> Result<(), TensorError> {
                    let serialized = response.serialize()
                        .map_err(|e| TensorError::RemoteError(format!("Failed to serialize response: {}", e)))?;
                    let n = serialized.len() as u32;
                    stream.write_all(&n.to_le_bytes())
                        .map_err(|e| TensorError::RemoteError(format!("Failed to write response length: {}", e)))?;
                    stream.write_all(&serialized)
                        .map_err(|e| TensorError::RemoteError(format!("Failed to write response: {}", e)))?;
                    Ok(())
                }();
                if let Err(e) = result {
                    eprintln!("{}", e);
                    break;
                }
            }
            Err(e) => {
                eprintln!("Failed to receive response: {}", e);
                break;
            }
        }
    }
}

fn handle_connection(connection: ClientConnection, mut stream: std::net::TcpStream) {
    // launch draining thread
    let stream_inner = stream.try_clone()
        .expect("Failed to clone stream for draining thread");
    let receiver = connection.output_messages_receiver.clone();
    thread::spawn(move || {
        drain_messages(stream_inner, receiver);
    });
    let connection_clone = connection.clone();
    thread::spawn(move || {
        drain_background_jobs(connection_clone);
    });
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
                        handle_request(request, &connection);
                    }
                    Err(e) => {
                        eprintln!("Failed to read request data: {}", e);
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read request size: {}", e);
                break;
            }
        }
    }
}

fn drain_background_jobs(connection: ClientConnection) {
    loop {
        let job = match connection.background_tasks_receiver.recv() {
            Ok(job) => job,
            Err(e) => {
                eprintln!("Failed to receive background job: {}", e);
                break;
            }
        };
        let task_id = match &job {
            AsyncJob::CopyFromSlice { task_id, .. } => *task_id,
            AsyncJob::ApplyElementwiseContiguous { task_id, .. } => *task_id,
            AsyncJob::ApplyElementwise1DStrided { task_id, .. } => *task_id,
            AsyncJob::ApplyElementwiseND { task_id, .. } => *task_id,
            AsyncJob::Broadcast { task_id, .. } => *task_id,
            AsyncJob::MatMul { task_id, .. } => *task_id,
        };
        let (message, error) = match job {
            AsyncJob::CopyFromSlice { dst, src, .. } => {
                let result = dispatch_copy_from_slice(dst, src, &connection);
                let err = result.as_ref().err().cloned();
                (Messages::CopyFromSliceResponse { result }, err)
            },
            AsyncJob::ApplyElementwiseContiguous { buf, op, start, len, .. } => {
                let (op_type, value) = op;
                let result = dispatch_apply_elementwise_contiguous(buf, op_type, value, start, len, &connection);
                let err = result.as_ref().err().cloned();
                (Messages::ApplyElementwiseContiguousResponse { result }, err)
            },
            AsyncJob::ApplyElementwise1DStrided { buf, op, offset, stride, len, .. } => {
                let (op_type, value) = op;
                let result = dispatch_apply_elementwise_1d_strided(buf, op_type, value, offset, stride, len, &connection);
                let err = result.as_ref().err().cloned();
                (Messages::ApplyElementwise1DStridedResponse { result }, err)
            },
            AsyncJob::ApplyElementwiseND { buf, op, offset, shape, stride, .. } => {
                let (op_type, value) = op;
                let result = dispatch_apply_elementwise_nd(buf, op_type, value, offset, &shape, &stride, &connection);
                let err = result.as_ref().err().cloned();
                (Messages::ApplyElementwiseNDResponse { result }, err)
            },
            AsyncJob::Broadcast { left, right, dst, op, .. } => {
                let result = dispatch_broadcast(left, right, dst, op, &connection);
                let err = result.as_ref().err().cloned();
                (Messages::BroadcastResponse { result }, err)
            },
            AsyncJob::MatMul { lhs, rhs, dst, b, m, k, n, contiguity, .. } => {
                let result = dispatch_matmul(lhs, rhs, dst, b, m, k, n, contiguity, &connection);
                let err = result.as_ref().err().cloned();
                (Messages::MatmulResponse { result }, err)
            },
        };
        let completion_response = Response {
            asynchronous: true,
            complete: true,
            task_id,
            error,
            message
        };
        let _ = connection.queue_response(completion_response);
    }
}

/// launch a new server in a background thread listening on the given IP and port
pub fn launch_server(ip: IpAddr, port: u16) -> Result<JoinHandle<()>, TensorError> {
    let mut server = RemoteServer::new(ip, port);
    let handle = thread::spawn(move || {
        server.serve().unwrap();
    });
    Ok(handle)
}