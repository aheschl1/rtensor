use slotmap::{new_key_type, SecondaryMap};

use crate::{backend::{cpu::Cpu, cuda::Cuda, Backend}, core::{primitives::{GradTensor, GradTensorRef, TensorBase}, tensor::TensorError, untyped::UntypedTensor, value::TensorValue}};
use std::{any::{Any, TypeId}, collections::VecDeque};
use std::collections::HashMap;

mod backwards;

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

    fn backwards(&self, upstream: &TensorBase<T, B>) -> Result<Vec<TensorBase<T, B>>, TensorError> {
        match self {
            GradNode::L1 { .. } => backwards::backwards_l1::<T, B>(self, upstream),
            GradNode::Leaf( .. ) => backwards::accumulate_grad::<T, B>(self, upstream),
            _ => Err(TensorError::UnsupportedOperation("Backward not implemented for this node type.".into())),
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

    pub fn backwards(&mut self, root: &GradTensor<T, B>) -> Result<(), TensorError> {
        // holds nodes to visit along with their upstream gradients
        // topo sort, because concider a graph like A->C<-B<-D where BFS should visit C too early
        let mut stack = Vec::new();
        let mut marks = SecondaryMap::new();
        let mut node_order = Vec::new();
        stack.push(StackState::Enter(root.node));

        enum StackState {
            Enter (NodeKey),
            Exit (NodeKey),
        }

        while let Some(state) = stack.pop() {
            match state {
                StackState::Enter (nkey) => {
                    match marks.get(nkey) {
                        Some(true) => continue,
                        Some(false) => return Err(TensorError::GradError("Graph contains a cycle.".into())),
                        None => {
                            marks.insert(nkey, false);
                            stack.push(StackState::Exit(nkey));
                            if let Some(node) = self.nodes.get(nkey) {
                                for parent in node.parents() {
                                    stack.push(StackState::Enter(parent));
                                }
                            } else {
                                return Err(TensorError::GradError("Node not found during backward pass.".into()));
                            }
                        }
                    }
                },
                StackState::Exit (nkey) => {
                    marks.insert(nkey, true);
                    node_order.push(nkey);
                }
            }
        }
        // could in theory move this into the above loop but this is clearer
        let mut accumulations = HashMap::new();
        accumulations.insert(root.node, vec![root.borrow().value.clone()]);

        for node_key in node_order {
            // accumulate grad. because of topo sort, we can assume to just sum the upstreams present to us
            // and then propagate downstream
            let dldy = accumulations.remove(&node_key)
                .unwrap() // t=since this node is in the visited list, it must have upstream grads
                .into_iter()
                .fold(None, |acc: Option<TensorBase<T, B>>, grad| {
                    if let Some(accum) = acc {
                        Some(accum + grad)
                    } else {
                        Some(grad)
                    }
                })
                .unwrap(); // must have at least one upstream grad

            let node = self.nodes.get(node_key).unwrap(); // we would never have discovered this node if it was not present
            let upstreams = node.backwards(&dldy)?;
            let parents = node.parents();
            for (parent, grad) in parents.into_iter().zip(upstreams.into_iter()) {
                accumulations.entry(parent).or_insert_with(Vec::new).push(grad);
            }
        }
        Ok(())
        // gradient is now accumulated in leaf nodes
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