use std::{cell::RefCell, sync::Arc};

use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, tensor::{AsTensor, TensorAccess, TensorError}, value::{TensorValue, WeightValue}, MetaTensorView}, grad::{self, GradNode, NodeKey}};

#[derive(Debug)]
pub struct GradTensor<T: WeightValue, B: Backend> {
    pub(crate) inner: GradTensorRef<T, B>,
    pub(crate) node: NodeKey,
}

// deep clone, not Rc clone
impl<T: WeightValue, B: Backend> Clone for GradTensor<T, B> {
    #[grad::when_enabled(ctx)]
    fn clone(&self) -> Self {
        let tensor = self.borrow().tensor.contiguous();
        let grad = match &self.borrow().grad {
            Some(g) => Some(g.contiguous()),
            None => None,
        };
        let inner = GradTensorInner {
            tensor,
            grad,
        };
        let inner_ref = Arc::new(RefCell::new(inner));
        let nodes = ctx.nodes.borrow();
        let curr_node = nodes.get(self.node).expect("Node not found in grad context");
        let node = if let GradNode::Leaf(_) = curr_node {
            GradNode::Leaf( inner_ref.clone() )
        } else {
            curr_node.clone()
        };
        drop(nodes);
        ctx.attach(inner_ref.clone(), node)
    }
}

impl<T: WeightValue, B: Backend> GradTensor<T, B> {

    #[grad::when_enabled(ctx)]
    pub(crate) fn leaf(
        tensor: TensorBase<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            tensor,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        ctx.make_leaf(inner)

    }

    pub(crate) fn input( // a tensor that requires grad but is not a parameter, for example, input to the model
        value: TensorBase<T, B>
    ) -> Self {
        Self::from_op(value, GradNode::None)
    }

    #[inline]
    #[grad::when_enabled(ctx)]
    pub(crate) fn from_op(
        tensor: TensorBase<T, B>,
        op: GradNode<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            tensor,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        ctx.attach(inner, op)
    }

    #[inline]
    #[grad::when_enabled(ctx)]
    pub(crate) fn from_op_self_referential(
        tensor: TensorBase<T, B>,
        op_builder: impl FnOnce(GradTensorRef<T, B>) -> GradNode<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            tensor,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        let node = op_builder(inner.clone());
        ctx.attach(inner, node)
    } 

    #[grad::when_enabled(ctx)]
    pub(crate) fn is_leaf(&self) -> bool {
        let nodes = ctx.nodes.borrow();
        let node = nodes.get(self.node).expect("Node not found in grad context.");
        node.is_leaf()
    }

    pub(crate) fn copy_tensor(&self) -> TensorBase<T, B> {
        self.borrow().tensor.contiguous()
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, GradTensorInner<T, B>> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GradTensorInner<T, B>> {
        self.inner.borrow_mut()
    }

    pub fn get_ref(&self) -> GradTensorRef<T, B> {
        self.inner.clone()
    }

    #[grad::when_enabled(ctx)]
    pub fn permute(self, dims: impl Into<Idx>) -> Result<Self, TensorError> {
        let idx = dims.into();
        let mut inner = self.borrow_mut();
        let new_view = inner.tensor.permute(idx.clone())?;
        inner.tensor.meta = new_view.meta.clone();
        drop(inner);
        // record node
        let new_node = GradNode::Permute {
            input: self.node,
            dims: idx,
        };

        Ok(ctx.attach(self.inner, new_node))
    }

    pub fn transpose(self) -> Self {
        let rank = self.borrow().tensor.rank();
        let dims: Idx = Idx::Coord((0..rank).rev().collect());
        unsafe { self.permute(dims).unwrap_unchecked() }
    }
    
}


pub struct GradTensorInner<T: TensorValue, B: Backend> {
    pub(crate) tensor: TensorBase<T, B>,
    pub(crate) grad: Option<TensorBase<T, B>>,
}

pub type GradTensorRef<T, B> = Arc<RefCell<GradTensorInner<T, B>>>;

impl<T: TensorValue, B: Backend> std::fmt::Debug for GradTensorInner<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradTensorInner")
            .field("grad", &self.grad)
            .field("value", &self.tensor)
            .finish()
    }
}

impl<T: WeightValue, B: Backend> Eq for GradTensor<T, B> {}
impl<T: WeightValue, B: Backend> PartialEq for GradTensor<T, B> { fn eq(&self, other: &Self) -> bool { true }}


#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{tensor::{TensorAccess, WithGrad}, Tensor}, grad::{self, optim::{Optim, SGD}, primitives::GradTensor}, ops::{broadcast::l1::l1_loss, scalar::ScalarGradOp, unary::UnaryGradOp}};

    #[test]
    fn playground() {

        fn model(wa: &GradTensor<f32, Cpu>, wb: &GradTensor<f32, Cpu>, target: &GradTensor<f32, Cpu>) -> GradTensor<f32, Cpu> {
            let c = wa + wb;
            let loss = l1_loss(&c, target);
            loss
        }

        fn modelv2(
            wa: &GradTensor<f32, Cpu>, 
            wb: &GradTensor<f32, Cpu>, 
            wc: &GradTensor<f32, Cpu>, 
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let inter = wb + wc;
            // println!("Intermediate: {:?}", inter);
            let c = wa + &inter;
            let loss = l1_loss(&c, target);
            loss
        }

        fn modelv3(
            input: &GradTensor<f32, Cpu>,  // [2, 3]
            wa: &GradTensor<f32, Cpu>, // [2, 3]
            wb: &GradTensor<f32, Cpu>,  // [3, 2]
            target: &GradTensor<f32, Cpu> // [3, 2]
        ) -> GradTensor<f32, Cpu> {
            let inter = input + wa; // [2, 3]
            let inter2 = inter.permute((1, 0)).unwrap().abs();
            // println!("Intermediate: {:?}", inter);
            let c = wb + &inter2;
            let loss = l1_loss(&c, target);
            loss
        }

        fn modelv4(
            input: &GradTensor<f32, Cpu>,  // [2, 3]
            wa: &GradTensor<f32, Cpu>, // [2, 3]
            wb: &GradTensor<f32, Cpu>,  // [3, 2]
            target: &GradTensor<f32, Cpu> // [3, 2]
        ) -> GradTensor<f32, Cpu> {
            let inter = input + wa; // [2, 3]
            let inter2 = inter.permute((1, 0)).unwrap().relu();
            // println!("Intermediate: {:?}", inter);
            let c = wb + &inter2;
            let loss = l1_loss(&c, target);
            loss
        }

        fn modelv5(
            input: &GradTensor<f32, Cpu>,  // [2, 3]
            target: &GradTensor<f32, Cpu> // [3, 2]
        ) -> GradTensor<f32, Cpu> {
            let loss = l1_loss(&-input.sqrt(), &target.clone().transpose());
            loss
        }

        fn modelv6(
            wa: &GradTensor<f32, Cpu>,  
            input: &GradTensor<f32, Cpu>,  
            target: &GradTensor<f32, Cpu> 
        ) -> GradTensor<f32, Cpu> {
            let temp = input + wa;
            let temp2 = temp.sigmoid().leaky_relu(1.); // grad should be identical even without leaky relu
            let loss = l1_loss(&-temp2.sqrt(), &target.clone().transpose());
            loss
        }

        fn modelv7(
            wa: &GradTensor<f32, Cpu>,  
            input: &GradTensor<f32, Cpu>,  
            target: &GradTensor<f32, Cpu> 
        ) -> GradTensor<f32, Cpu> {
            let temp = input + wa;
            let temp2 = temp.leaky_relu(0.1).ln();
            let loss = l1_loss(&-temp2, &target.clone().transpose());
            loss
        }

        grad::with::<f32, Cpu>(|ctx| {
            
            let a = Tensor::<f32>::scalar(1.).grad();
            let b = Tensor::<f32>::ones((2, 2)).param();
            let target = Tensor::<f32>::zeros((2, 2)).grad();

            let mut optim = SGD::<f32, Cpu>::new(1.);
            // optim.register_parameter(&a).unwrap();
            optim.register_parameters(&[&b]).unwrap();
            
            for _ in 0..10 {
                let loss = model(&a, &b, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }

            println!("{:?}", a);

            let a = Tensor::<f32>::ones((2, 2)).param();
            let b = Tensor::<f32>::ones((2, 2)).param();
            let c = Tensor::<f32>::ones((2, 2)).param();

            optim.register_parameter(&a).unwrap();
            optim.register_parameter(&b).unwrap();
            optim.register_parameter(&c).unwrap();

            for _ in 0..10 {
                let loss = modelv2(&a, &b, &c, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }

            println!("{:?}", a);

            let input = Tensor::<f32>::ones((2, 3)).grad();
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let wb = Tensor::<f32>::ones((3, 2)).param();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&wb).unwrap();
            for _ in 0..10 {
                let loss = modelv3(&input, &wa, &wb, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }

            println!("{:?}", input);
            println!("{:?}", wa);

            let input = Tensor::<f32>::ones((2, 3)).grad();
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let wb = Tensor::<f32>::ones((3, 2)).param();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&wb).unwrap();
            for _ in 0..10 {
                let loss = modelv4(&input, &wa, &wb, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }

            println!("{:?}", input);
            println!("{:?}", wa);

            let input = Tensor::<f32>::ones((2, 3)).param();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&input).unwrap();
            for _ in 0..12 {
                let loss = modelv5(&input, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }

            println!("{:?}", input);

            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            for _ in 0..10 {
                let loss = modelv6(&wa, &input, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            println!("{:?}", wa);

            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            for _ in 0..10 {
                let loss = modelv7(&wa, &input, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            println!("{:?}", wa);

            // use model, but do a broadcasted add
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::zeros((2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            for _ in 0..10 {
                let inter = &input + &wa; // broadcasted add
                let loss = l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            println!("{:?}", wa);

            // use model, but do a broadcasted sub
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::ones((2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            for _ in 0..10 {
                let inter = &input - &wa; // broadcasted sub
                let loss = l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            println!("{:?}", wa);

            // use model, but do a broadcasted mul
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::zeros((2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            for _ in 0..10 {
                let inter = &input * &wa; // broadcasted mul
                let loss = l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            println!("{:?}", wa);

            // use model, but do a broadcasted div
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::zeros((2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            for _ in 0..10 {
                let inter = &input / &wa; // broadcasted div
                let loss = l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            println!("{:?}", wa);

        })
    }
}