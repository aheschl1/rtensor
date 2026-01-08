use slotmap::new_key_type;

use crate::{backend::{cpu::Cpu, cuda::Cuda, Backend}, core::{primitives::{GradTensor, GradTensorRef}, untyped::UntypedTensor, value::TensorValue}};
use std::any::{Any, TypeId};
use std::collections::HashMap;

// struct NodeKey;

new_key_type! {
    pub(crate) struct NodeKey;
}

/// Each variant of a node holds parents and any tensors that need to be saved for backward.
pub(crate) enum GradNode<T: TensorValue, B: Backend> {
    Add { left: NodeKey, right: NodeKey },
    Leaf( GradTensorRef<T, B> ),

    // LOSSES
    L1 { 
        input: NodeKey, 
        // it is likely that this is leaf; however, it is not always the case
        // consider siamese networks
        target: NodeKey 
    },
}

impl<T: TensorValue, B: Backend> GradNode<T, B> {
    pub fn is_leaf(&self) -> bool {
        matches!(self, GradNode::Leaf(..))
    }

    pub fn leaf(inner: GradTensorRef<T, B>) -> Self {
        GradNode::Leaf(inner)
    }

    #[inline]
    pub fn parents(&self) -> Vec<NodeKey> {
        match self {
            GradNode::Add { left, right } => vec![left.clone(), right.clone()],
            GradNode::Leaf(_) => vec![],
            GradNode::L1 { input, target } => vec![input.clone(), target.clone()],
        }
    }

    #[inline]
    pub fn saved(&self) -> Vec<Box<dyn UntypedTensor<B>>> {
        match self {
            GradNode::Add { .. } => vec![],
            GradNode::Leaf(_) => vec![],
            GradNode::L1 { .. } => vec![],
        }
    }
}

pub struct GradContext<T: TensorValue, B: Backend> {
    // tape: Vec<NodeKey>, // holds references to all inner tensors that require gradients
    nodes: slotmap::SlotMap<NodeKey, GradNode<T, B>>,
}

impl<T: TensorValue, B: Backend> GradContext<T, B> {
    pub fn new() -> Self {
        Self { 
            nodes: slotmap::SlotMap::with_key(),
        }
    }

    /// Clears the tape, removing all recorded tensors.
    // pub fn clear(&mut self) {
    //     self.tape.clear();
    // }

    #[inline]
    pub(crate) fn make_leaf(
        &mut self,
        inner: GradTensorRef<T, B>,
    ) -> GradTensor<T, B> {
        let node = GradNode::leaf(inner.clone());
        self.attach(inner, node)
    }

    #[inline]
    pub(crate) fn attach(
        &mut self,
        inner: GradTensorRef<T, B>,
        op: GradNode<T, B>,
    ) -> GradTensor<T, B> {
        let node_id = self.nodes.insert(op);
        GradTensor {
            inner,
            node: node_id,
        }
    }
}

impl<T: TensorValue, B: Backend> std::fmt::Debug for GradContext<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradContext {{ nodes_len: {} }}", self.nodes.len())
    }
}

thread_local! {
    static GRAD_CONTEXT_CPU: std::cell::RefCell<HashMap<TypeId, Box<dyn Any>>> = std::cell::RefCell::new(HashMap::new());
    static GRAD_CONTEXT_CUDA: std::cell::RefCell<HashMap<TypeId, Box<dyn Any>>> = std::cell::RefCell::new(HashMap::new());
}


pub fn with<T: TensorValue, B: Backend>(
    f: impl FnOnce(&mut GradContext<T, B>)
) {
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let mut ctx_map = ctx_cell.borrow_mut();
            let ctx = ctx_map
                .entry(type_id)
                .or_insert_with(|| Box::new(GradContext::<T, Cpu>::new()));
            
            // SAFETY: We know the TypeId matches T, and we've verified B is Cpu
            let ctx = ctx.downcast_mut::<GradContext<T, Cpu>>().unwrap();
            let ctx = unsafe { &mut *(ctx as *mut GradContext<T, Cpu> as *mut GradContext<T, B>) };
            f(ctx);
        });
    } else if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
        GRAD_CONTEXT_CUDA.with(|ctx_cell| {
            let mut ctx_map = ctx_cell.borrow_mut();
            let ctx = ctx_map
                .entry(type_id)
                .or_insert_with(|| Box::new(GradContext::<T, Cuda>::new()));
            
            // SAFETY: We know the TypeId matches T, and we've verified B is Cuda
            let ctx = ctx.downcast_mut::<GradContext<T, Cuda>>().unwrap();
            let ctx = unsafe { &mut *(ctx as *mut GradContext<T, Cuda> as *mut GradContext<T, B>) };
            f(ctx);
        });
    } else {
        panic!("Unsupported backend for GradContext");
    }
}

#[inline]
pub fn is_enabled<T: TensorValue, B: Backend>() -> bool {
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            ctx_map.contains_key(&type_id)
        })
    } else if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
        GRAD_CONTEXT_CUDA.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            ctx_map.contains_key(&type_id)
        })
    } else {
        panic!("Unsupported backend for GradContext");
    }
}

/// Runs the provided closure if the gradient context is enabled for the given backend.
pub fn when_enabled<T: TensorValue, B: Backend, R>(
    f: impl FnOnce(&mut GradContext<T, B>) -> R
) -> Option<R>{
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let mut ctx_map = ctx_cell.borrow_mut();
            if let Some(ctx_box) = ctx_map.get_mut(&type_id) {
                // SAFETY: We know the TypeId matches T, and we've verified B is Cpu
                if let Some(ctx) = ctx_box.downcast_mut::<GradContext<T, Cpu>>() {
                    let ctx = unsafe { &mut *(ctx as *mut GradContext<T, Cpu> as *mut GradContext<T, B>) };
                    Some(f(ctx))
                }else{
                    None
                }
            } else {
                None
            }
        })
    } else if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
        GRAD_CONTEXT_CUDA.with(|ctx_cell| {
            let mut ctx_map = ctx_cell.borrow_mut();
            if let Some(ctx_box) = ctx_map.get_mut(&type_id) {
                // SAFETY: We know the TypeId matches T, and we've verified B is Cuda
                if let Some(ctx) = ctx_box.downcast_mut::<GradContext<T, Cuda>>() {
                    let ctx = unsafe { &mut *(ctx as *mut GradContext<T, Cuda> as *mut GradContext<T, B>) };
                    Some(f(ctx))
                } else {
                    None
                }
            } else {
                None
            }
        })
    } else {
        panic!("Unsupported backend for GradContext");
    }
}

pub fn when_disabled<T: TensorValue, B: Backend>(
    f: impl FnOnce()
) {
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            if !ctx_map.contains_key(&type_id) {
                f();
            }
        });
    } else if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
        GRAD_CONTEXT_CUDA.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            if !ctx_map.contains_key(&type_id) {
                f();
            }
        });
    } else {
        panic!("Unsupported backend for GradContext");
    }
}

/// same as anabled but warns if not enabled
pub fn enabled_or_warn<T: TensorValue, B: Backend>(
    f: impl FnOnce(&mut GradContext<T, B>),
    msg: &str,
) {
    when_disabled::<T, B>(|| {
        tracing::warn!("{}", msg);
    });
    with::<T, B>(f);
}