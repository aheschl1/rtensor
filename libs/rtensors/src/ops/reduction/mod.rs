use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, shape_to_stride, tensor::TensorError, value::TensorValue, MetaTensor, Shape}};


pub enum ReductionOpTypes {
    Sum = 1,
    Prod = 2,
    Max = 3,
    Min = 4,
}

pub trait ReductionOp : Sized {
    fn sum(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn prod(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn mean(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn max(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn min(&self, axes: &Idx) -> Result<Self, TensorError>;
}

macro_rules! do_reduce {
    ($op:expr, $axes:ident, $self:ident) => {
        if let Idx::At(axis) = $axes {
            let mut output = materialize_output::<T, B>(&$self.meta, $self.backend.clone())?;
            $self.backend.apply_reduce(
                (&$self.buf, &$self.meta), 
                (&mut output.buf, &output.meta), 
                *axis,
                $op,
            )?;
            Ok(output)
        } else{
            Err(TensorError::WrongDims(
                "Reduction over multiple axes is not implemented yet.".to_string(),
            ))
        }
    };
}

impl<T, B> ReductionOp for TensorBase<T, B>
where
    T: TensorValue,
    B: Backend,
{
    fn sum(&self, axes: &Idx) -> Result<Self, TensorError> {
        do_reduce!(ReductionOpTypes::Sum, axes, self)
    }

    fn prod(&self, axes: &Idx) -> Result<Self, TensorError> {
        do_reduce!(ReductionOpTypes::Prod, axes, self)
    }

    fn max(&self, axes: &Idx) -> Result<Self, TensorError> {
        do_reduce!(ReductionOpTypes::Max, axes, self)
    }

    fn min(&self, axes: &Idx) -> Result<Self, TensorError> {
        do_reduce!(ReductionOpTypes::Min, axes, self)
    }

    fn mean(&self, axes: &Idx) -> Result<Self, TensorError> {
        todo!()
    }
}

#[inline]
fn materialize_output<T: TensorValue, B: Backend>(input: &MetaTensor, backend: B) -> Result<TensorBase<T, B>, TensorError>{
    let output_meta = reduction_output_meta(input.clone(), vec![]);
    let buf = backend.alloc(output_meta.size());
    Ok(TensorBase::from_parts(backend, buf?, output_meta))
}

#[inline]
fn reduction_output_meta(input: MetaTensor, axes: Vec<usize>) -> MetaTensor {
    let mut output_shape = Vec::new();
    for (i, &dim) in input.shape.iter().enumerate() {
        if !axes.contains(&i) {
            output_shape.push(dim);
        }
    }
    let output_shape: Shape = Shape::from(output_shape);
    let strides = shape_to_stride(&output_shape);

    MetaTensor {
        shape: output_shape,
        strides,
        offset: 0
    }
}