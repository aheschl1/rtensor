//! A high-performance tensor library with CPU and CUDA backends.
//!
//! This library provides multidimensional array (tensor) operations with support for:
//! - CPU backend using OpenBLAS for optimized linear algebra
//! - CUDA backend for GPU acceleration (when `cuda` feature is enabled)
//! # Quick Start
//!
//! ```ignore
//! use tensors::core::{Tensor, TensorAccess};
//!
//! // Create a tensor
//! let a = Tensor::<f32>::zeros((3, 4));
//! let b = Tensor::<f32>::ones((3, 4));
//!
//! // Arithmetic operations with broadcasting
//! let c = a + b;
//!
//! // Matrix multiplication
//! use tensors::ops::linalg::MatMul;
//! let x = Tensor::<f32>::ones((3, 2));
//! let y = Tensor::<f32>::ones((2, 4));
//! let z = x.matmul(&y).unwrap();
//! ```


pub mod core;
pub mod ops;
pub mod backend;
pub mod macros;
#[cfg(feature = "grad")]
pub mod grad;

pub(crate) mod testing;

pub(crate) mod openblas {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/openblas_bindings.rs"));
}
