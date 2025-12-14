use std::{collections::HashMap, fmt::{Debug, Display}, io::{Read, Write}, net::IpAddr, sync::{atomic::{AtomicU32, Ordering}, mpsc, Arc, Condvar, Mutex, RwLock}};

use crate::{backend::{remote::{get_backend_default, protocol::{Messages, Request, Response, Slice, TypelessBuf, Value}}, Backend, BackendMatMul}, core::{primitives::DeviceType, tensor::TensorError, value::{DType, TensorValue}}};


#[derive(Debug, Clone)]
pub struct RemoteBuf<T: TensorValue> {
    pub(crate) id: u32,
    pub(crate) dtype: DType,
    pub (crate) _marker: std::marker::PhantomData<T>,
}

impl<T: TensorValue> RemoteBuf<T> {
    #[inline(always)]
    fn to_typeless(&self) -> TypelessBuf {
        TypelessBuf {
            id: self.id,
            dtype: self.dtype,
        }
    }

    #[inline(always)]
    fn from_typeless(buf: TypelessBuf) -> Self {
        Self {
            id: buf.id,
            dtype: buf.dtype,
            _marker: std::marker::PhantomData,
        }
    }
}

macro_rules! send_recv {
    ($self:expr, $message:expr, $response_pattern:pat => $result:expr) => {{
        let receiver = $self.send_message($message);
        let response = receiver.recv()
            .map_err(|_| TensorError::BackendError("Failed to receive response".to_string()))?;
        match response {
            $response_pattern => $result,
            _ => Err(TensorError::BackendError("Unexpected response type".to_string())),
        }
    }};
}

macro_rules! make_op {
    ($op:expr, $value:expr) => {
        ($op, Value::from_value($value))
    };
}

#[derive(Clone)]
pub struct RemoteBackend {
    remote_addr: IpAddr,
    remote_port: u16,
    message_id: Arc<AtomicU32>,
    pending: Arc<Pending>,
    outgoing: Arc<lfqueue::UnboundedQueue<Request>>,
    pending_response: Arc<RwLock<HashMap<u32, mpsc::Sender<Messages>>>>,
}
    
impl Debug for RemoteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBackend")
            .field("remote_addr", &self.remote_addr)
            .field("remote_port", &self.remote_port)
            .field("message_id", &self.message_id)
            .field("pending", &self.pending)
            .field("pending_response", &self.pending_response)
            .finish()
    }
}

#[derive(Debug)]
pub struct Pending {
    count: AtomicU32,
    lock: Mutex<()>,
    cv: Condvar,
}

impl Pending {
    #[inline(always)]
    pub fn inc(&self) {
        self.count.fetch_add(1, Ordering::AcqRel);
    }

    #[inline(always)]
    pub fn dec(&self) {
        if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.cv.notify_all();
        }
    }

    #[inline(always)]
    pub fn sync(&self) {
        let mut guard = self.lock.lock().unwrap();
        while self.count.load(Ordering::Acquire) != 0 {
            guard = self.cv.wait(guard).unwrap();
        }
    }
}


impl RemoteBackend {
    pub fn new_with_address(remote_addr: IpAddr, remote_port: u16) -> Result<Self, std::io::Error> {
        let pending = Pending {
            count: AtomicU32::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        };
        let res = Self {
            remote_addr,
            remote_port,
            pending: Arc::new(pending),
            message_id: Arc::new(AtomicU32::new(0)),
            outgoing: lfqueue::UnboundedQueue::new().into(),
            pending_response: Arc::new(RwLock::new(HashMap::new())),
        };
        Ok(res)
    }

    fn send_message(&self, msg: Messages) -> mpsc::Receiver<Messages>{
        let (sender, receiver) = mpsc::channel();
        let mid = self.next_message_id();
        {
            let mut pending = self.pending_response.write().unwrap();
            pending.insert(mid, sender);
        }
        self.pending.inc();
        let req = Request {
            task_id: mid,
            message: msg,
        };
        self.outgoing.enqueue(req);
        receiver
    }

    pub(crate) fn connect(&mut self) -> Result<(), std::io::Error> {
        let stream = std::net::TcpStream::connect((self.remote_addr, self.remote_port))?;
        stream.set_nodelay(true)?;

        let read_stream = stream.try_clone()?;
        let write_stream = stream;

        let remote = self.clone();
        std::thread::spawn(move || {
            drain_outgoing(remote, write_stream);
        });
        let remote = self.clone();
        std::thread::spawn(move || {
            read_incoming(remote, read_stream);
        });
        Ok(())
    }

    #[inline(always)]
    fn next_message_id(&self) -> u32 {
        self.message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

}

impl Backend for RemoteBackend {
    type Buf<T: TensorValue> = RemoteBuf<T>;

    fn new() -> Self {
        get_backend_default().expect("No default remote backend available")
    }
    
    fn device_type() -> crate::core::primitives::DeviceType {
        // let message = Messages::DeviceType;
        // let receiver = self.send_message(message);
        // let response = receiver.recv().unwrap();
        // match response {
        //     Messages::DeviceTypeResponse { device_type } => device_type,
        //     _ => panic!("Unexpected response type"),
        // }
        DeviceType::Remote {
            ip: todo!(),
            port: todo!(),
            remote_type: todo!(),
        }
    }

    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let message = Messages::AllocFromSlice {
            slice: Slice::from_boxed_slice(src),
        };
        send_recv!(self, message, Messages::AllocFromSliceResponse { buf } => {
            Ok(RemoteBuf::from_typeless(buf?))
        })
    }

    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        let message = Messages::Alloc {
            len,
            dtype: T::DTYPE,
        };
        send_recv!(self, message, Messages::AllocResponse { buf } => {
            Ok(RemoteBuf::from_typeless(buf?))
        })
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::CopyFromSlice {
            dst: dst.to_typeless(),
            src: Slice::from_slice(src),
        };
        send_recv!(self, message, Messages::CopyFromSliceResponse { result } => result)
    }

    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Read {
            buf: buf.to_typeless(),
            offset,
        };
        send_recv!(self, message, Messages::ReadResponse { value } => {
            value?.to_value::<T>()
        })
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::Write {
            buf: buf.to_typeless(),
            offset,
            value: Value::from_value(value),
        };
        send_recv!(self, message, Messages::WriteResponse { result } => result)
    }

    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        let message = Messages::Len {
            buf: buf.to_typeless(),
        };
        let receiver = self.send_message(message);
        match receiver.recv() {
            Ok(Messages::LenResponse { len }) => len,
            _ => panic!("Failed to get buffer length or unexpected response"),
        }
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Copy {
            src: src.to_typeless(),
        };
        send_recv!(self, message, Messages::CopyResponse { buf } => {
            Ok(RemoteBuf::from_typeless(buf?))
        })
    }

    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, crate::core::tensor::TensorError> {
        self.pending.sync();
        let message = Messages::Dump {
            src: src.to_typeless(),
        };
        send_recv!(self, message, Messages::DumpResponse { data } => {
            data?.to_boxed_slice::<T>()
        })
    }

    fn apply_elementwise_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::OpType, T), 
        start: usize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::ApplyElementwiseContiguous {
            buf: buf.to_typeless(),
            op: make_op!(op.0, op.1),
            start,
            len,
        };
        send_recv!(self, message, Messages::ApplyElementwiseContiguousResponse { result } => result)
    }

    fn apply_elementwise_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        op: (crate::ops::base::OpType, T), 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::ApplyElementwise1DStrided {
            buf: buf.to_typeless(),
            op: make_op!(op.0, op.1),
            offset,
            stride,
            len,
        };
        send_recv!(self, message, Messages::ApplyElementwise1DStridedResponse { result } => result)
    }

    fn apply_elementwise_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (crate::ops::base::OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::ApplyElementwiseND {
            buf: buf.to_typeless(),
            op: make_op!(op.0, op.1),
            offset,
            shape: shape.to_vec(),
            stride: stride.to_vec(),
        };
        send_recv!(self, message, Messages::ApplyElementwiseNDResponse { result } => result)
    }

    unsafe fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &crate::core::MetaTensor), 
        right: (*const Self::Buf<T>, &crate::core::MetaTensor),
        dst: (*mut Self::Buf<T>, &crate::core::MetaTensor),
        op: crate::ops::base::OpType
    ) -> Result<(), crate::core::tensor::TensorError> {
        let message = Messages::Broadcast {
            left: ((&*left.0).to_typeless(), left.1.clone()),
            right: ((&*right.0).to_typeless(), right.1.clone()),
            dst: ((&*dst.0).to_typeless(), dst.1.clone()),
            op,
        };
        send_recv!(self, message, Messages::BroadcastResponse { result } => result)
    }
}

impl<T: TensorValue> BackendMatMul<T> for RemoteBackend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &crate::core::MetaTensor),
        rhs: (&Self::Buf<T>, &crate::core::MetaTensor),
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: crate::core::meta::ContiguityTypes,
    ) -> Result<Self::Buf<T>, TensorError> {
        let message = Messages::Matmul {
            lhs: (lhs.0.to_typeless(), lhs.1.clone()),
            rhs: (rhs.0.to_typeless(), rhs.1.clone()),
            b,
            m,
            k,
            n,
            contiguity,
        };
        send_recv!(self, message, Messages::MatmulResponse { buf } => {
            Ok(RemoteBuf::from_typeless(buf?))
        })
    }
}

// todo, make this async
fn drain_outgoing(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let queue_handle = remote.outgoing.clone();
    loop {
        if let Some(req) = queue_handle.dequeue() {
            let serialized = req.serialize().unwrap();
            let n = serialized.len();
            let n_bytes = (n as u32).to_le_bytes();
            stream.write_all(&n_bytes).unwrap();
            stream.write_all(&serialized).unwrap();
        }
    }
}

fn read_incoming(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let mut len_buf = [0u8; 4];
    loop {
        stream.read_exact(&mut len_buf).unwrap();
        let msg_len = u32::from_le_bytes(len_buf) as usize;
        let mut msg_buf = vec![0u8; msg_len];
        stream.read_exact(&mut msg_buf).unwrap();
        let msg = Response::deserialize(&msg_buf).expect("Failed to deserialize response");
        let task_id = msg.task_id;
        if !msg.asynchronous {
            debug_assert!(msg.complete);
            let sender = {
                let mut pending = remote.pending_response.write().unwrap();
                pending.remove(&task_id)
            };
            if let Some(sender) = sender {
                sender.send(msg.message).unwrap();
            }
            remote.pending.dec();
        }else{
            if msg.complete {
                remote.pending.dec();
            }else{
                //send initial follow up to receiver, do not decrement pending yet
                let sender = {
                    let mut pending = remote.pending_response.write().unwrap();
                    pending.remove(&task_id)
                };
                if let Some(sender) = sender {
                    sender.send(msg.message).unwrap();
                }
            }
        }
    }
}
