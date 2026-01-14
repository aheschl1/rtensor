use crate::{
    backend::Backend,
    core::{
        TensorView, TensorViewMut, primitives::TensorBase, primops::{Exp, InvExp, SquareRoot}, tensor::{AsTensor, AsViewMut}, value::{TensorValue, WeightValue}
    },
};



// mod abs;
// pub mod neg;
// pub mod relu;
// mod sigmoid;
// mod sin;
// mod sqrt;
// mod tanh;

// pub use abs::Abs;
// pub use neg::Negate;
// pub use relu::Relu;
// pub use sigmoid::Sigmoid;
// pub use sqrt::Sqrt;
// pub use tanh::Tanh;
// pub use sin::Sin;

// macro_rules! specify_trait_unary_cabal {
//     // One or more items separated by ';' (optional trailing ';')
//     (
//         $(
//             $op:ident $(where T: $($extra:tt)+ )?
//         );+ $(;)?
//     ) => {
//         paste::paste! {
//             pub trait InplaceUnaryOp2<T, B>
//             where
//                 T: TensorValue,
//                 B: Backend,
//             {
//                 $(
//                     paste::paste! {
//                          // Adjust the signature to whatever your tensor type is.
//                     // This shows an inplace mutation pattern.
//                     fn [<apply_ $op>](&mut self)
//                     $( where T: $($extra)+ )?
//                     ;
//                     }

//                 )+
//             }

//             pub trait UnaryOp2<T: TensorValue, B: Backend> {
//                 $(
//                     paste::paste! {
//                          // Adjust the signature to whatever your tensor type is.
//                     // This shows an inplace mutation pattern.
//                     fn $op(&mut self)
//                     $( where T: $($extra)+ )?
//                     ;
//                     }

//                 )+
//             }
//         }
//     };
// }

macro_rules! specify_unary_op_template {
    (
        $(
            ($name:ident) $op:ident $(where T: $first:path $(, $extra:path)*)?;
        )+
    ) => {

        paste::paste! {

            pub trait InplaceUnaryOp<T: TensorValue, B: Backend> {
                $(
                    fn [<apply_ $op>](&mut self)
                    where 
                    $(
                        T: $first $(+ $extra)*
                    )?;
                )+

            }

            
            pub trait UnaryOp<T: TensorValue, B: Backend> {
                $(
                    fn $op(&self) -> TensorBase<T, B>
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?;
                )+
            }


            $(

                pub trait $name<T: TensorValue, B: Backend> {
                    fn [<$op _inplace>](&mut self)
                    where
                     $(
                        T: $first $(+ $extra)*
                    )?
                    ;
                }

                impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> $name<T, B> for V {
                    fn [<$op _inplace>](&mut self)
                    where
                     $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        let view = self.view_mut();
                        if let Err(e) = view.backend.[<apply_ $op>](view.buf, &view.meta) {
                            panic!("Failed to apply abs: {}", e);
                        }
                    }
                }

              
            )+



            impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> InplaceUnaryOp<T, B> for V
            {
                $(
                    fn [<apply_ $op>](&mut self)
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        self.[<$op _inplace>]();
                    }
                )+
            }

            
            impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> UnaryOp<T, B> for V {
                // fn relu(&self) -> TensorBase<T, B> {
                //     let mut result = self.owned();
                //     result.apply_relu();
                //     result
                // }
                $(
                    fn $op(&self) -> TensorBase<T, B>
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        let mut result = self.owned();
                        result.[<apply_ $op>]();
                        result
                    }
                )+
            }

            
            // impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> UnaryOp<T, B> for V {
            //     // fn relu(&self) -> TensorBase<T, B> {
            //     //     let mut result = self.owned();
            //     //     result.apply_relu();
            //     //     result
            //     // }
            //     $(
            //         fn $op(&self) -> TensorBase<T, B>
            //         where
            //         $(
            //             T: $first $(+ $extra)*
            //         )?
            //         {
            //             let mut result = self.owned();
            //             result.[<apply_ $op>]();
            //             // result.apply_relu();
            //             result
            //         }
            //     )+
            // }

           
        }
        
    };
}

specify_unary_op_template! {
    (Abs) abs;
    (Relu) relu;
    (Sigmoid) sigmoid where T: InvExp;
    (Tanh) tanh where T: Exp, InvExp;
    (Sqrt) sqrt where T: SquareRoot;
    (Negate) neg where T: std::ops::Neg<Output = T>;
    (Sin) sin where T: WeightValue;
    (Cos) cos where T: WeightValue;
    (Tan) tan where T: WeightValue;
    (Asin) asin where T: WeightValue;
    (Acos) acos where T: WeightValue;
    (Atan) atan where T: WeightValue;
    (Sinh) sinh where T: WeightValue;
    (Cosh) cosh where T: WeightValue;
    // (Tanh) tanh where T: WeightValue;
    (Asinh) asinh where T: WeightValue;
    (Acosh) acosh where T: WeightValue;
    (Atanh) atanh where T: WeightValue;
    (Rsqrt) rsqrt where T: WeightValue;
    (Reciprocal) reciprocal where T: WeightValue;
    (Square) square where T: WeightValue;
    (Cube) cube where T: WeightValue;
    (ExpV) exp where T: WeightValue;
    (Sign) sign where T: WeightValue;
}


impl<T, B> std::ops::Neg for TensorBase<T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for TensorView<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for TensorViewMut<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}



// macro_rules! specify_trait_unary_cabal {
//     // One or more items separated by ';' (optional trailing ';')
//     (
//         $(
//             $op:ident where T: $($extra:tt)+ )?
//         );+ $(;)?
//     ) => {
//         // hello
//         $(
//             fn $op<T>()
//             where
//                 $( T: $($extra)+ )?
//             {
                
//             }
//         )+
//     };
// }

// specify_trait_unary_cabal! {
//     relu where T: ;
//     sigmoid where T: Exp + InvExp
// }

// pub trait InplaceUnaryOp<T: TensorValue, B: Backend> {
//     fn apply_relu(&mut self);
//     fn apply_sigmoid(&mut self)
//     where
//         T: Exp + InvExp;
//     fn apply_tanh(&mut self)
//     where
//         T: Exp + InvExp;
//     fn apply_sqrt(&mut self)
//     where
//         T: SquareRoot;
//     fn apply_abs(&mut self);
//     // fn apply_sin(&mut self);
// }

// pub trait UnaryOp<T: TensorValue, B: Backend> {
//     fn relu(&self) -> TensorBase<T, B>;
//     fn sigmoid(&self) -> TensorBase<T, B>
//     where
//         T: Exp + InvExp;
//     fn tanh(&self) -> TensorBase<T, B>
//     where
//         T: Exp + InvExp;
//     fn abs(&self) -> TensorBase<T, B>;
//     // fn apply_sin(&self) -> TensorBase<T, B>;
// }

// impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> InplaceUnaryOp<T, B> for V {
//     fn apply_relu(&mut self) {
//         self.relu_inplace();
//     }

//     fn apply_sigmoid(&mut self)
//     where
//         T: Exp + InvExp,
//     {
//         self.sigmoid_inplace();
//     }

//     fn apply_tanh(&mut self)
//     where
//         T: Exp + InvExp,
//     {
//         self.tanh_inplace();
//     }
//     fn apply_abs(&mut self) {
//         self.abs_inplace();
//     }

//     fn apply_sqrt(&mut self)
//     where
//         T: SquareRoot,
//     {
//         self.sqrt_inplace();
//     }
//     // fn apply_sin(&mut self) {
//     //     self.sin_inplace();
//     // }
// }

// impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> UnaryOp<T, B> for V {
//     fn relu(&self) -> TensorBase<T, B> {
//         let mut result = self.owned();
//         result.apply_relu();
//         result
//     }

//     fn sigmoid(&self) -> TensorBase<T, B>
//     where
//         T: Exp + InvExp,
//     {
//         let mut result = self.owned();
//         result.apply_sigmoid();
//         result
//     }

//     fn tanh(&self) -> TensorBase<T, B>
//     where
//         T: Exp + InvExp,
//     {
//         let mut result = self.owned();
//         result.apply_tanh();
//         result
//     }
//     fn abs(&self) -> TensorBase<T, B> {
//         let mut result = self.owned();
//         result.apply_abs();
//         result
//     }
//     // fn apply_sin(&self) -> TensorBase<T, B> {
//     //     let mut result = self.owned();
//     //     result.apply_sin();
//     //     result
//     // }
// }

#[cfg(test)]
mod tests {
    use crate::{
        backend::cpu::Cpu,
        ops::unary::{Negate, Relu, Sigmoid, Sin, Sqrt, Tanh},
        testing::{unary_assert_1d_strided, unary_assert_contiguous, unary_assert_nd_strided},
    };

    #[test]
    fn test_unary_negate_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    // #[test]
    // fn test_unary_sin_contiguous() {
    //     unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], |f| f.sin(), |f| {
    //         f.sin_inplace()
    //     });
    // }

    // #[test]
    // fn test_unary_sin_1d_strided() {
    //     unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], |f| f.sin(), |f| {
    //         f.sin_inplace()
    //     });
    // }

    // #[test]
    // fn test_unary_sin_nd_strided() {
    //     unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], |f| f.sin(), |f| {
    //         f.sin_inplace()
    //     });
    // }

    #[test]
    fn test_unary_relu_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([-1.0, 1.0], |f| f.max(0.0), Relu::relu_inplace);
    }

    #[test]
    fn test_unary_relu_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [-1.0, 1.0, -1.0],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_relu_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [
                -1.0, 1.0, 0.0, 2.0, 1.0, 2.3, -0.3, 0.4, 0.0, -0.3, 0.4, 0.5, -0.2, 0.1, 0.2, -0.5,
            ],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>(
            [1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [1.0; 16],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>(
            [1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [1.0; 16],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| f.sqrt(),
            |f| f.sqrt_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0; 2], |f| f.sqrt(), |f| f.sqrt_inplace());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::{
        backend::cuda::Cuda,
        core::{
            Tensor, primitives::{CudaTensor, TensorBase}, tensor::{AsTensor, TensorAccess, TensorAccessMut}
        },
        ops::unary::*,
        testing::{
            test_with_contiguous_2_elem_tensor, unary_assert_1d_strided, unary_assert_contiguous,
            unary_assert_nd_strided,
        },
    };

    #[test]
    fn test_unary_negate_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_cos_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::cos, |f| {
            f.cos_inplace()
        });
    }

    #[test]
    fn test_unary_cos_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::cos, |f| {
            f.cos_inplace()
        });
    }

    #[test]
    fn test_unary_cos_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::cos, |f| {
            f.cos_inplace()
        });
    }

    #[test]
    fn test_unary_sin_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::sin, |f| {
            f.sin_inplace()
        });
    }

    #[test]
    fn test_unary_sin_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::sin, |f| {
            f.sin_inplace()
        });
    }

    #[test]
    fn test_unary_sin_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::sin, |f| {
            f.sin_inplace()
        });
    }

    #[test]
    fn test_unary_tan_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::tan, |f| {
            f.tan_inplace()
        });
    }

    #[test]
    fn test_unary_tan_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::tan, |f| {
            f.tan_inplace()
        });
    }

    #[test]
    fn test_unary_tan_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::tan, |f| {
            f.tan_inplace()
        });
    }


    /*
    ARCTRIG
     */

    #[test]
    fn test_unary_acos_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::acos, |f| {
            f.acos_inplace()
        });
    }

    #[test]
    fn test_unary_acos_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::acos, |f| {
            f.acos_inplace()
        });
    }

    #[test]
    fn test_unary_acos_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::acos, |f| {
            f.acos_inplace()
        });
    }

    #[test]
    fn test_unary_asin_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::asin, |f| {
            f.asin_inplace()
        });
    }

    #[test]
    fn test_unary_asin_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::asin, |f| {
            f.asin_inplace()
        });
    }

    #[test]
    fn test_unary_asin_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::asin, |f| {
            f.asin_inplace()
        });
    }

    #[test]
    fn test_unary_atan_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::atan, |f| {
            f.atan_inplace()
        });
    }

    #[test]
    fn test_unary_atan_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::atan, |f| {
            f.atan_inplace()
        });
    }

    #[test]
    fn test_unary_atan_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::atan, |f| {
            f.atan_inplace()
        });
    }


    /**
     * 
     * HYPERBOLIC ARC TRIG
     */

     #[test]
    fn test_unary_cosh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::cosh, |f| {
            f.cosh_inplace()
        });
    }

    #[test]
    fn test_unary_cosh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::cosh, |f| {
            f.cosh_inplace()
        });
    }

    #[test]
    fn test_unary_cosh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::cosh, |f| {
            f.cosh_inplace()
        });
    }

    #[test]
    fn test_unary_sinh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::sinh, |f| {
            f.sinh_inplace()
        });
    }

    #[test]
    fn test_unary_sinh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::sinh, |f| {
            f.sinh_inplace()
        });
    }

    #[test]
    fn test_unary_sinh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::sinh, |f| {
            f.sinh_inplace()
        });
    }

   


    /*
    ARCTRIG
     */

    #[test]
    fn test_unary_acosh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::acosh, |f| {
            f.acosh_inplace()
        });
    }

    #[test]
    fn test_unary_acosh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::acosh, |f| {
            f.acosh_inplace()
        });
    }

    #[test]
    fn test_unary_acosh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::acosh, |f| {
            f.acosh_inplace()
        });
    }

    #[test]
    fn test_unary_asinh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::asinh, |f| {
            f.asinh_inplace()
        });
    }

    #[test]
    fn test_unary_asinh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::asinh, |f| {
            f.asinh_inplace()
        });
    }

    #[test]
    fn test_unary_asinh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::asinh, |f| {
            f.asinh_inplace()
        });
    }

    #[test]
    fn test_unary_atanh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], f64::atanh, |f| {
            f.atanh_inplace()
        });
    }

    #[test]
    fn test_unary_atanh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], f64::atanh, |f| {
            f.atanh_inplace()
        });
    }

    #[test]
    fn test_unary_atanh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::atanh, |f| {
            f.atanh_inplace()
        });
    }


    /*
    RSQRT */

    #[test]
    fn test_unary_rsqrt_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| (1. / f.sqrt()), |f| {
            f.rsqrt_inplace()
        });
    }

    #[test]
    fn test_unary_rsqrt_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| (1. / f.sqrt()), |f| {
            f.rsqrt_inplace()
        });
    }

    #[test]
    fn test_unary_rsqrt_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| (1. / f.sqrt()), |f| {
            f.rsqrt_inplace()
        });
    }

    #[test]
    fn test_unary_sign_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], f64::signum, |f| {
            f.sign_inplace()
        });
    }

    #[test]
    fn test_unary_sign_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], f64::signum, |f| {
            f.sign_inplace()
        });
    }

    #[test]
    fn test_unary_sign_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::signum, |f| {
            f.sign_inplace()
        });
    }

    #[test]
    fn test_unary_exp_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], f64::exp, |f| {
            f.exp_inplace()
        });
    }

    #[test]
    fn test_unary_exp_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], f64::exp, |f| {
            f.exp_inplace()
        });
    }

    #[test]
    fn test_unary_exp_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::exp, |f| {
            f.exp_inplace()
        });
    }

    #[test]
    fn test_unary_square_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| f * f, |f| {
            f.square_inplace()
        });
    }

    #[test]
    fn test_unary_square_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| f * f, |f| {
            f.square_inplace()
        });
    }

    #[test]
    fn test_unary_square_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| f * f, |f| {
            f.square_inplace()
        });
    }

     #[test]
    fn test_unary_cube_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| f * f * f, |f| {
            f.cube_inplace()
        });
    }

    #[test]
    fn test_unary_cube_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| f * f * f, |f| {
            f.cube_inplace()
        });
    }

    #[test]
    fn test_unary_cube_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| f * f * f, |f| {
            f.cube_inplace()
        });
    }

     #[test]
    fn test_unary_reciprocal_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| 1. / f, |f| {
            f.reciprocal_inplace()
        });
    }

    #[test]
    fn test_unary_reciprocal_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| 1. / f, |f| {
            f.reciprocal_inplace()
        });
    }

    #[test]
    fn test_unary_reciprocal_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| 1. / f, |f| {
            f.reciprocal_inplace()
        });
    }


    #[test]
    fn test_unary_abs_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.0, 1.3], |f| f.abs(), |f| f.abs_inplace());
    }

    #[test]
    fn test_unary_abs_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, -1.0, 3.0],
            |f| f.abs(),
            |f| f.abs_inplace(),
        );
    }

    #[test]
    fn test_unary_abs_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([-1.0; 16], |f| f.abs(), |f| f.abs_inplace());
    }

    #[test]
    fn test_unary_relu_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.0, 1.0], |f| f.max(0.0), Relu::relu_inplace);
    }

    #[test]
    fn test_unary_relu_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [-1.0, 1.0, -1.0],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_relu_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [
                -1.0, 1.0, 0.0, 2.0, 1.0, 2.3, -0.3, 0.4, 0.0, -0.3, 0.4, 0.5, -0.2, 0.1, 0.2, -0.5,
            ],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>(
            [1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [1.0; 16],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>(
            [1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [1.0; 16],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| f.sqrt(),
            |f| f.sqrt_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0; 2], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.0; 16], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| f.sqrt(),
            |f| f.sqrt_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.0; 2], |f| f.sqrt(), |f| f.sqrt_inplace());
    }
}

#[cfg(all(test, feature = "remote"))]
mod remote_tests {
    use std::{sync::OnceLock, thread};

    use crate::{
        backend::{
            remote::{client::RemoteBackend, get_backend_default, server::RemoteServer},
            Backend,
        },
        core::{
            primitives::{RemoteTensor, TensorBase},
            tensor::TensorError,
            value::TensorValue,
            MetaTensor, Shape,
        },
    };

    // Lazy static backend shared across all tests
    static BACKEND: OnceLock<RemoteBackend> = OnceLock::new();

    fn get_backend() -> RemoteBackend {
        BACKEND
            .get_or_init(|| {
                // Start the server
                let mut server = RemoteServer::new("127.0.0.1".parse().unwrap(), 7878);
                thread::spawn(move || {
                    let _ = server.serve();
                });
                thread::sleep(std::time::Duration::from_millis(10));

                // Create and connect the backend
                let backend = get_backend_default().unwrap();

                backend
            })
            .clone()
    }

    fn make_remote_tensor<T: TensorValue>(
        buf: Vec<T>,
        shape: impl Into<Shape>,
    ) -> Result<RemoteTensor<T>, TensorError> {
        let shape: Shape = shape.into();
        let buf_len = buf.len();
        let expected_len: usize = shape.iter().product();

        if buf_len != expected_len {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                expected_len, buf_len
            )));
        }

        let backend = get_backend();
        let buffer = backend.alloc_from_slice(buf.into())?;
        let stride = crate::core::shape_to_stride(&shape);

        // Clone the backend for this tensor
        let tensor_backend = backend.clone();
        drop(backend); // Release the lock

        Ok(TensorBase::from_parts(
            tensor_backend,
            buffer,
            MetaTensor::new(shape, stride, 0),
        ))
    }

    #[test]
    fn test_remote_negate() {
        let tensor: TensorBase<f32, RemoteBackend> =
            make_remote_tensor(vec![1.0f32, -2.0, 3.0], (3,)).unwrap();
        let negated = -tensor;

        let expected = make_remote_tensor(vec![-1.0f32, 2.0, -3.0], (3,)).unwrap();
        assert_eq!(negated.cpu().unwrap(), expected.cpu().unwrap());
    }
}
