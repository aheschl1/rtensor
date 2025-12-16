#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;


#[cfg(feature = "cuda")]
/// Trait for types that can be stored in tensors (with CUDA support).
/// 
/// Implemented for standard numeric types: f32, f64, i8-i128, u8-u128, isize, usize, bool.
pub trait TensorValue: 
    Copy + 
    Default +
    TypeConstants +
    DeviceRepr +
    Send + Sync + 
    PartialEq + PartialOrd +
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> +
    std::ops::AddAssign +
    std::ops::SubAssign +
    std::ops::MulAssign +
    'static
{
    const DTYPE: crate::core::value::DType;
}

#[cfg(not(feature = "cuda"))]
/// Trait for types that can be stored in tensors.
/// 
/// Implemented for standard numeric types: f32, f64, i8-i128, u8-u128, isize, usize, bool.
pub trait TensorValue: 
    Copy + 
    Default +
    TypeConstants +
    Send + Sync +
    PartialEq + PartialOrd +
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> +
    std::ops::AddAssign +
    std::ops::SubAssign +
    std::ops::MulAssign +
    'static
{
    const DTYPE: crate::core::value::DType;
}

/// Provides default constant values for tensor element types.
pub trait TypeConstants {
    const ZERO: Self;
    const ONE: Self;
    const MIN: Self;
    const MAX: Self;
    const SUPPORTS_NEG: bool;
}


macro_rules! impl_tensor_values {
    ($(($type:ty, $dtype:expr)),+ $(,)?) => {
        $(
            impl TensorValue for $type {
                const DTYPE: crate::core::value::DType = $dtype;
            }
        )+
    };
}

macro_rules! impl_default {
    ($type:ty, $zero:expr, $one:expr, $min:expr, $max:expr, $supports_neg:expr) => {
        impl TypeConstants for $type {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const MIN: Self = $min;
            const MAX: Self = $max;
            const SUPPORTS_NEG: bool = $supports_neg;
        }
    };
}

impl_tensor_values!(
    (f32, DType::F32), 
    (f64, DType::F64), 
    (i8, DType::I8), 
    (i16, DType::I16), 
    (i32, DType::I32), 
    (i64, DType::I64), 
    (i128, DType::I128), 
    // (isize, DType::I64), 
    (u8, DType::U8), 
    (u16, DType::U16), 
    (u32, DType::U32), 
    (u64, DType::U64), 
    (u128, DType::U128), 
    // (usize, DType::U64)
);

impl_default!(f32, 0.0f32, 1.0f32, f32::MIN, f32::MAX, true);
impl_default!(f64, 0.0f64, 1.0f64, f64::MIN, f64::MAX, true);
impl_default!(i8, 0i8, 1i8, i8::MIN, i8::MAX, true);
impl_default!(i16, 0i16, 1i16, i16::MIN, i16::MAX, true);
impl_default!(i32, 0i32, 1i32, i32::MIN, i32::MAX, true);
impl_default!(i64, 0i64, 1i64, i64::MIN, i64::MAX, true);
impl_default!(i128, 0i128, 1i128, i128::MIN, i128::MAX, true);
// impl_default!(isize, 0isize, 1isize, isize::MIN, isize::MAX);
impl_default!(u8, 0u8, 1u8, u8::MIN, u8::MAX, false);
impl_default!(u16, 0u16, 1u16, u16::MIN, u16::MAX, false);
impl_default!(u32, 0u32, 1u32, u32::MIN, u32::MAX, false);
impl_default!(u64, 0u64, 1u64, u64::MIN, u64::MAX, false);
impl_default!(u128, 0u128, 1u128, u128::MIN, u128::MAX, false);
// impl_default!(usize, 0usize, 1usize, usize::MIN, usize::MAX);

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
#[repr(C)]
pub enum DType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    U128 = 5,
    I32 = 6,
    U64 = 7,
    I64 = 8,
    I128 = 9,
    F32 = 10,
    F64 = 11,
    BOOL = 12,
}

#[allow(non_camel_case_types)]
pub mod types {
    use std::ops::{AddAssign, Deref, DerefMut};
    #[cfg(feature = "cuda")]
    use cudarc::driver::DeviceRepr;

    use crate::core::value::{DType, TypeConstants, TensorValue};

    #[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd)]
    #[repr(C)]
    pub struct boolean(pub bool);

    impl boolean {
        pub const FALSE: Self = Self(false);
        pub const TRUE: Self = Self(true);
    }

    // the poiunters and refs are for inner
    impl AsRef<bool> for boolean {
        #[inline(always)]
        fn as_ref(&self) -> &bool {
            &self.0
        }
    }

    impl AsMut<bool> for boolean {
        #[inline(always)]
        fn as_mut(&mut self) -> &mut bool {
            &mut self.0
        }
    }

    impl Deref for boolean {
        type Target = bool;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for boolean {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl From<u8> for boolean {
        #[inline(always)]
        fn from(value: u8) -> Self {
            Self(value != 0)
        }
    }

    impl TypeConstants for boolean {
        const ZERO: Self = Self::FALSE;
        const ONE: Self = Self::TRUE;
        const MIN: Self = Self::FALSE;
        const MAX: Self = Self::TRUE;
        const SUPPORTS_NEG: bool = false;
    }

    impl std::ops::Add for boolean {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 || rhs.0)
        }
    }

    impl std::ops::Sub for boolean {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 && !rhs.0)
        }
    }

    impl std::ops::Mul for boolean {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 && rhs.0)
        }
    }

    impl AddAssign for boolean {
        fn add_assign(&mut self, rhs: Self) {
            self.0 = self.0 || rhs.0;
        }
    }

    impl std::ops::SubAssign for boolean {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = self.0 && !rhs.0;
        }
    }

    impl std::ops::MulAssign for boolean {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = self.0 && rhs.0;
        }
    }

    impl From<boolean> for bool {
        fn from(value: boolean) -> Self {
            value.0
        }
    }

    impl From<bool> for boolean {
        fn from(value: bool) -> Self {
            Self(value)
        }
    }

    impl TensorValue for boolean {
        const DTYPE: DType = DType::BOOL;
    }

    #[cfg(feature = "cuda")]
    unsafe impl DeviceRepr for boolean {}
}

#[cfg(test)]
mod tests {
    use crate::core::{value::types::boolean, Tensor};

    #[test]
    fn boolean_tensor() {
        let mut tensor = Tensor::<boolean>::zeros((2, 3));
        tensor += boolean(true);
        tensor += boolean(true);
        let expected = Tensor::<boolean>::ones((2, 3));
        assert_eq!(tensor, expected);
    }   

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda() {
        use crate::core::primitives::CudaTensor;

        let mut tensor = CudaTensor::<boolean>::zeros((2, 3));
        tensor += boolean(true);
        tensor += boolean(true);
        let expected = CudaTensor::<boolean>::ones((2, 3));
        assert_eq!(tensor.cpu().unwrap(), expected.cpu().unwrap());
    }   
}