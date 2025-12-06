pub mod core;
pub mod ops;
pub mod backend;
pub mod macros;

pub(crate) mod openblas {
    //! OpenBLAS FFI bindings
    //! 
    //! This module contains automatically generated bindings to OpenBLAS functions.
    //! The bindings are generated at build time using bindgen.
    
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    
    include!(concat!(env!("OUT_DIR"), "/openblas_bindings.rs"));
}
