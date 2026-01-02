use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, shape_to_stride, tensor::{AsTensor, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape}};


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
    },
    LogSumExp,
    Norm(NormType),
    ArgMax
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
            Self::Variance {..} | Self::Stdev { .. } => 6,
            Self::LogSumExp => 7,
            Self::Norm(_) => 8,
            Self::ArgMax => 9
            
        }
    }
}

pub enum NormType {
    L1,
    L2
}

pub trait TotalReductionOp: Sized {
    fn total_sum(&self) -> Result<Self, TensorError>;
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
    fn norm(&self, norm: NormType, axes: &Idx) -> Result<Self, TensorError>;
    fn logsumexp(&self, axes: &Idx) -> Result<Self, TensorError>;
    fn argmax(&self, axes: &Idx) -> Result<Self, TensorError>;
}

macro_rules! do_reduce {
    ($op:expr, $axes:ident, $tensor:ident) => {

        match $axes {
            Idx::Item => {
                let mut output = TensorBase::from_buf(vec![ T::ZERO ], vec![])?;
                $tensor.backend.apply_reduce_total(
                    (&$tensor.buf, &$tensor.meta), 
                    (&mut output.buf, &output.meta), 
                    0,
                    $op,
                )?;
                Ok(output)
                // Err(TensorError::WrongDims("test".to_string()))
            }
            Idx::At(axis) => {
                let mut output = materialize_output::<T, B>(&$tensor.meta, $tensor.backend.clone(), $axes)?;
                $tensor.backend.apply_reduce(
                    (&$tensor.buf, &$tensor.meta), 
                    (&mut output.buf, &output.meta), 
                    *axis,
                    $op,
                )?;
                Ok(output)
            }
            _ => Err(TensorError::WrongDims(
                "Reduction over multiple axes is not implemented yet.".to_string(),
            ))
        }

      
    };
}


impl<T, B> TotalReductionOp for TensorBase<T, B>
where 
    T: TensorValue,
    B: Backend
{
     fn total_sum(&self) -> Result<Self, TensorError> {
        self.sum(&Idx::Item)
    }
}

impl<T, B> ReductionOp for TensorBase<T, B>
where
    T: TensorValue,
    B: Backend,
{
   
    fn sum(&self, axes: &Idx) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(ReductionOpTypes::Sum, axes, a)
        }else {
            do_reduce!(ReductionOpTypes::Sum, axes, self)
        }
    }

    fn prod(&self, axes: &Idx) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(ReductionOpTypes::Prod, axes, a)
        }else {
            do_reduce!(ReductionOpTypes::Prod, axes, self)
        }
    }

    fn max(&self, axes: &Idx) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(ReductionOpTypes::Max, axes, a)
        }else {
            do_reduce!(ReductionOpTypes::Max, axes, self)
        }
    }

    fn min(&self, axes: &Idx) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(ReductionOpTypes::Min, axes, a)
        }else {
            do_reduce!(ReductionOpTypes::Min, axes, self)
        }
    }

    fn mean(&self, axes: &Idx) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(ReductionOpTypes::Mean, axes, a)
        }else {
            do_reduce!(ReductionOpTypes::Mean, axes, self)
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
    fn logsumexp(&self, axes: &Idx) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(ReductionOpTypes::LogSumExp, axes, a)
        }else {
            do_reduce!(ReductionOpTypes::LogSumExp, axes, self)
        }
    }
    fn norm(&self, norm: NormType, axes: &Idx) -> Result<Self, TensorError> {
        let code = ReductionOpTypes::Norm(norm);
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(code, axes, a)
        }else {
            do_reduce!(code, axes, self)
        }
    }
    fn argmax(&self, axes: &Idx) -> Result<Self, TensorError> {
        let code = ReductionOpTypes::ArgMax;
        if !self.is_contiguous() {
            let a = self.contiguous();
            do_reduce!(code, axes, a)
        }else {
            do_reduce!(code, axes, self)
        }
    }
}

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

    // for (i, &dim) in input.shape.iter().enumerate() {
    //     println!("i={i}, dim={dim}");
    //     if i < idx_num {
    //         output_shape.push(dim);
    //     }
    // }
    println!("materialziing output: {:?}", output_shape);
    let output_shape: Shape = Shape::from(output_shape);
    let strides = shape_to_stride(&output_shape);

    MetaTensor {
        shape: output_shape,
        strides,
        offset: 0
    }
}