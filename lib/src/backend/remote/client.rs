use std::{collections::HashMap, fmt::Debug, io::{Read, Write}, net::IpAddr, sync::{atomic::{AtomicBool, AtomicU32, Ordering}, Arc, Condvar, Mutex, RwLock}};

use crate::{backend::{remote::{get_backend_default, protocol::{Messages, Request, Response, Slice, TypelessBuf, Value}}, Backend, BackendMatMul}, core::{primitives::DeviceType, tensor::TensorError, value::{DType, TensorValue}}};
use flume;


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


#[derive(Debug)]
struct PendingHandler {
    count: Arc<AtomicU32>,
    cv: Condvar,
    mutex: Mutex<()>,
}

impl PendingHandler {
    fn inc(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
    fn dec(&self) {
        let prev = self.count.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            self.cv.notify_all();
        }
    }
    fn sync(&self) {
        while self.count.load(Ordering::SeqCst) > 0 {
            let lock = self.mutex.lock().unwrap();
            let _unused = self.cv.wait(lock).unwrap();
        }
    }
}

#[derive(Clone)]
pub struct RemoteBackend {
    remote_addr: IpAddr,
    remote_port: u16,
    message_id: Arc<AtomicU32>,
    pending: Arc<PendingHandler>,
    messages_outgoing_sender: flume::Sender<Request>,
    messages_outgoing_receiver: flume::Receiver<Request>,
    pending_response: Arc<RwLock<HashMap<u32, flume::Sender<Messages>>>>,
    poisoned: Arc<AtomicBool>,
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

impl RemoteBackend {
    pub fn new_with_address(remote_addr: IpAddr, remote_port: u16) -> Result<Self, std::io::Error> {
        let pending = PendingHandler {
            count: Arc::new(AtomicU32::new(0)),
            cv: Condvar::new(),
            mutex: Mutex::new(()),
        };
        let (sender, receiver) = flume::unbounded();
        let res = Self {
            remote_addr,
            remote_port,
            pending: Arc::new(pending),
            message_id: Arc::new(AtomicU32::new(0)),
            messages_outgoing_sender: sender,
            messages_outgoing_receiver: receiver,
            pending_response: Arc::new(RwLock::new(HashMap::new())),
            poisoned: Arc::new(AtomicBool::new(false)),
        };
        Ok(res)
    }

    fn poison(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    pub fn sync(&self) {
        self.pending.sync();
    }

    #[inline(always)]
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    fn send_message(&self, msg: Messages) -> flume::Receiver<Messages>{
        if self.is_poisoned() {
            panic!("Attempted to send message on poisoned RemoteBackend. Reasons for poison:
            1. An asynchronous operation reported an error from the remote backend.
            2. The RemoteBackend is in an inconsistent state and can no longer process messages safely.");
        }
        self.pending.inc();
        let (sender, receiver) = flume::bounded(1);
        let mid = self.next_message_id();
        {
            let mut pending = self.pending_response.write().unwrap();
            pending.insert(mid, sender);
        }
        let req = Request {
            task_id: mid,
            message: msg,
        };
        self.messages_outgoing_sender.send(req).expect("Failed to send message to outgoing channel");
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

    pub fn address(&self) -> (IpAddr, u16) {
        (self.remote_addr, self.remote_port)
    }

    #[inline(always)]
    fn next_message_id(&self) -> u32 {
        self.message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
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
            ip: "127.0.0.1".parse().unwrap(),
            port: 7878,
            remote_type: DeviceType::Cpu.into(),
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
        self.pending.sync();
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
    
    fn apply_elementwise_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        op: (crate::ops::base::OpType, T),
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError> {
        let message = Messages::ApplyElementwiseND {
            buf: buf.to_typeless(),
            op: make_op!(op.0, op.1),
            offset,
            shape: shape.to_vec(),
            stride: stride.to_vec(),
        };
        send_recv!(self, message, Messages::ApplyElementwiseNDResponse { result } => result)
    }
}

impl<T: TensorValue> BackendMatMul<T> for RemoteBackend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &crate::core::MetaTensor),
        rhs: (&Self::Buf<T>, &crate::core::MetaTensor),
        dst: &mut Self::Buf<T>,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        contiguity: crate::core::meta::ContiguityTypes,
    ) -> Result<(), TensorError> {
        let message: Messages = Messages::Matmul {
            lhs: (lhs.0.to_typeless(), lhs.1.clone()),
            rhs: (rhs.0.to_typeless(), rhs.1.clone()),
            dst: dst.to_typeless(),
            b,
            m,
            k,
            n,
            contiguity,
        };
        send_recv!(self, message, Messages::MatmulResponse { result } => result)
    }
}

#[inline]
fn read_response(stream: &mut std::net::TcpStream, len_buf: &mut  [u8; 4]) -> Result<Response, Box<bincode::ErrorKind>> {
    stream.read_exact(len_buf).unwrap();
    let msg_len = u32::from_le_bytes(*len_buf) as usize;
    let mut msg_buf = vec![0u8; msg_len];
    stream.read_exact(&mut msg_buf).unwrap();
    Response::deserialize(&msg_buf)
}

#[inline]
fn send_message_to_channel(remote: &RemoteBackend, msg: Response) {
    let task_id = msg.task_id;
    let sender = {
        let mut pending = remote.pending_response.write().unwrap();
        pending.remove(&task_id)
    };
    if let Some(sender) = sender {
        sender.send(msg.message).unwrap();
    }
}

// todo, make this async

/// Thread function to drain outgoing messages and send them over the TCP stream
/// # Arguments
/// * `remote` - The RemoteBackend instance
/// * `stream` - The TCP stream to send messages over
fn drain_outgoing(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let receiver = remote.messages_outgoing_receiver.clone();
    loop {        
        if let Ok(req) = receiver.recv() {
            if remote.is_poisoned() {
                break;
            }
            let serialized = req.serialize().unwrap();
            let n = serialized.len();
            let n_bytes = (n as u32).to_le_bytes();
            stream.write_all(&n_bytes).unwrap();
            stream.write_all(&serialized).unwrap();
        } else {
            // Channel closed, exit thread
            break;
        }
    }
}

/// Thread function to read incoming messages from the TCP stream
/// 
/// Upon receiving a message, if it is marked as not asynchronous, it sends the message to the waiting channel
/// Otherwise, it handles asynchronous messages accordingly. If the asynchronous message is complete and does not
/// indicate an error, it simply decrements the pending count. If it indicates an error, it poisons the RemoteBackend to prevent further operations.
/// If the asynchronous message is not complete, it sends a follow-up message to the waiting channel, and does not decrement the pending count.
/// 
/// # Arguments
/// * `remote` - The RemoteBackend instance
/// * `stream` - The TCP stream to read messages from
fn read_incoming(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let mut len_buf = [0u8; 4];
    loop {
        let msg = read_response(&mut stream, &mut len_buf).unwrap();
        if !msg.asynchronous {
            debug_assert!(msg.complete);
            send_message_to_channel(&remote, msg);
            remote.pending.dec();
        }else{
            if msg.complete {
                // no need to send follow up, just decrement pending. because, there was an incomplete one before
                // that sent the message. async followup is just a backend notification.
                // nobody waits on follow up of async
                if let Some(e) = msg.error {
                    remote.poison();
                    panic!("Inconsistent state detected. Received error in async message: {:?}", e);
                } 
                remote.pending.dec();
            }else{
                //send initial follow up to receiver, do not decrement pending yet
                send_message_to_channel(&remote, msg);
            }
        }
    }
}
