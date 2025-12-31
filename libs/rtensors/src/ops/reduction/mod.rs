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

impl<T, B> ReductionOp for TensorBase<T, B>
where
    T: TensorValue,
    B: Backend,
{
    fn sum(&self, axes: &Idx) -> Result<Self, TensorError> {
        let output = materialize_output::<T, B>(&self.meta, self.backend.clone())?;
        self.backend.apply_reduce(&self.buf, &output.buf, ReductionOpTypes::Sum)?;
        Ok(output)
    }

    fn prod(&self, axes: &Idx) -> Result<Self, TensorError> {
        let output = materialize_output::<T, B>(&self.meta, self.backend.clone())?;
        self.backend.apply_reduce(&self.buf, &output.buf, ReductionOpTypes::Prod)?;
        Ok(output)
    }

    fn mean(&self, axes: &Idx) -> Result<Self, TensorError> {
        todo!()
    }

    fn max(&self, axes: &Idx) -> Result<Self, TensorError> {
        let output = materialize_output::<T, B>(&self.meta, self.backend.clone())?;
        self.backend.apply_reduce(&self.buf, &output.buf, ReductionOpTypes::Max)?;
        Ok(output)
    }

    fn min(&self, axes: &Idx) -> Result<Self, TensorError> {
        let output = materialize_output::<T, B>(&self.meta, self.backend.clone())?;
        self.backend.apply_reduce(&self.buf, &output.buf, ReductionOpTypes::Min)?;
        Ok(output)
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