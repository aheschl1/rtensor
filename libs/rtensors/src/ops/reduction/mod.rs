use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, shape_to_stride, tensor::{AsTensor, AsView, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape}};


pub enum ReductionOpTypes {
    Sum,
    Prod,
    Max,
    Min,
    Mean,
    Variance {
        unbiased: bool
    },
    Stdev {
        unbiased: bool
    }
}

impl ReductionOpTypes {
    #[inline(always)]
    pub const fn get_code(&self) -> u8 {
        match self {
            Self::Sum => 1,
            Self::Prod => 2,
            Self::Max => 3,
            Self::Min => 4,
            Self::Mean => 5,
            Self::Variance {..} | Self::Stdev { .. } => 6
            
        }
    }
}

pub trait TotalReductionOp<T: TensorValue, B: Backend>: Sized + ReductionOp<T, B> {
    fn total_sum(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.sum(Idx::Item)
    }
    fn total_prod(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.prod(Idx::Item)
    }
    fn total_mean(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.mean(Idx::Item)
    }
    fn total_max(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.max(Idx::Item)
    }
    fn total_min(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.min(Idx::Item)
    }
    fn total_var(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.var(Idx::Item)
    }
    fn total_pop_var(&self) -> Result<TensorBase<T, B>, TensorError>{
        self.pop_var(Idx::Item)
    }
}

pub trait ReductionOp : Sized {
    
    fn sum(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn prod(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn mean(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn max(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn min(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn var(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn pop_var(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn std(&self, axes: &Idx, unbiased: bool) -> Result<Self, TensorError>;
}

impl<T: TensorValue, B: Backend, V> TotalReductionOp<T, B> for V where V: ReductionOp<T, B>{}


#[inline(always)]
pub fn do_reduce<T, B>(
    op: ReductionOpTypes,
    axes: &Idx,
    tensor: &impl AsView<T, B>,
) -> Result<TensorBase<T, B>, TensorError>
where
    T: TensorValue,
    B: Backend,
{
    let tensor = tensor.view();
    match axes {
        Idx::Item => {
            let mut output = TensorBase::from_buf(vec![T::ZERO], vec![])?;
            tensor.backend.apply_reduce_contiguous_flat(
                &tensor.buf,
                &mut output.buf,
                tensor.meta.offset,
                tensor.meta.size(),
                op,
            )?;
            Ok(output)
        }
        Idx::At(axis) => {
            let mut output =
                materialize_output::<T, B>(&tensor.meta, tensor.backend.clone(), axes)?;
            tensor.backend.apply_reduce(
                (&tensor.buf, &tensor.meta),
                (&mut output.buf, &output.meta),
                *axis,
                op,
            )?;
            Ok(output)
        }
        _ => Err(TensorError::WrongDims(
            "Reduction over multiple axes is not implemented yet.".to_string(),
        )),
    }
}


impl<T: TensorValue, B: Backend, V> ReductionOp<T, B> for V
where
    V: AsView<T, B>,
{
    fn sum(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Sum, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Sum, &axes, &t)
        }
    }

    fn prod(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Prod, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Prod, &axes, &t)
        }
    }

    fn max(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Max, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Max, &axes, &t)
        }
    }

    fn min(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Min, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Min, &axes, &t)
        }
    }

    fn mean(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Mean, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Mean, &axes, &t)
        }
    }
    fn var(&self, axes: &Idx) -> Result<Self, TensorError> {
        let code = ReductionOpTypes::Variance { unbiased: true };
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(code, axes, a)
        }else {
            do_reduce!(code, axes, self)
        }
    }
    fn pop_var(&self, axes: &Idx) -> Result<Self, TensorError> {
        let code = ReductionOpTypes::Variance { unbiased: false };
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(code, axes, a)
        }else {
            do_reduce!(code, axes, self)
        }
    }
     fn std(&self, axes: &Idx, unbiased: bool) -> Result<Self, TensorError> {
        let code = ReductionOpTypes::Stdev { unbiased };
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(code, axes, a)
        }else {
            do_reduce!(code, axes, self)
        }
    }
}


// implement_reductionop!(TensorBase);
// implement_reductionop!(TensorView<'a>);
// implement_reductionop!(TensorViewMut<'a>);

#[inline]
fn materialize_output<T: TensorValue, B: Backend>(input: &MetaTensor, backend: B, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>{
    let output_meta = reduction_output_meta(input.clone(), axes);
    let buf = backend.alloc(output_meta.size());

   
    Ok(TensorBase::from_parts(backend, buf?, output_meta))
}

#[inline]
fn reduction_output_meta(input: MetaTensor, axes: &Idx) -> MetaTensor {
    let mut output_shape = Vec::new();


    let idx_num = match axes {
        Idx::Item => 0,
        Idx::At(x) => *x,
        _ => panic!("Weird multi dimension reductions are not avaiable.")
    };




    for d in 0..input.rank() {
        if d == idx_num {
            // PyTorch and Numpy have a keep dim argument that removes this.
            output_shape.push(1);
        } else {
            output_shape.push(input.shape()[d]);

        }
    }

    println!("materialziing output: {:?}", output_shape);
    let output_shape: Shape = Shape::from(output_shape);
    let strides = shape_to_stride(&output_shape);

    MetaTensor {
        shape: output_shape,
        strides,
        offset: 0
    }
}