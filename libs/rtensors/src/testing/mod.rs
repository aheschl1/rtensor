use std::fmt::Debug;

use crate::{backend::Backend, core::{MetaTensorView, Slice, TensorViewMut, primitives::TensorBase, tensor::{AsTensor, TensorAccess, TensorAccessMut}, value::{TensorValue, WeightValue}}, ops::unary::Abs};


pub fn test_with_contiguous_2_elem_tensor<T, F, B>(definition: [T; 2], functor: F)
where 
    T: TensorValue,
    B: Backend,
    F: FnOnce(TensorBase<T, B>)
{
    functor(TensorBase::<T, B>::from_buf(definition.to_vec(), (2, 1)).unwrap())
}

// pub fn unary_assert_with_contiguous_2_elem_tensor<T, F, B>(definition: [T; 2], expected: [T; 2], functor: F)
// where 
//     T: TensorValue,
//     B: Backend,
//     F: FnOnce(&mut TensorBase<T, B>),
//     TensorBase<T, B>: PartialEq<TensorBase<T, B>> + Debug
// {
//     let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (2, 1)).unwrap();
//     functor(&mut tensor);
//     assert_eq!(tensor, TensorBase::<T, B>::from_buf(expected.to_vec(), (2, 1)).unwrap());
// }


pub fn test_with_1d_strided_tensor<T, F, B>(definition: [T; 3], functor: F)
where 
    T: TensorValue,
    B: Backend,
    for<'a> F: FnOnce(TensorViewMut<'a, T, B>)
{
    let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (3, 1)).unwrap();
    let slice = tensor.slice_mut(0, Slice::full().step(2)).unwrap();
    
    functor(slice);
}

// pub fn assert_with_1d_strided_tensor<T, F, B>(definition: [T; 3], expected: [T; 3], functor: F)
// where 
//     T: TensorValue,
//     B: Backend,
//     for<'a> F: FnOnce(&mut TensorViewMut<'a, T, B>),
//     TensorBase<T, B>: PartialEq<TensorBase<T, B>> + Debug
// {
//     let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (3, 1)).unwrap();
//     let mut slice = tensor.slice_mut(0, Slice::full().step(2)).unwrap();
    
//     functor(&mut slice);

//     assert_eq!(tensor, TensorBase::<T, B>::from_buf(expected.to_vec(), (3, 1)).unwrap());
// }


fn eps_eq<T>(a: &T, b: &T)
where 
    T: TensorValue + WeightValue + Debug
{
    if(a.eps_eq(b, 0.001)) {

    } else {
        panic!("Epsilon comparison failed!\n\tleft: {a:?}\n\tright: {b:?}");
    }
}

pub fn test_with_nd_strided_tensor<T, F, B>(definition: [T; 16], functor: F)
where 
    T: TensorValue,
    B: Backend,
    for<'a> F: FnOnce(TensorViewMut<'a, T, B>)
{
    let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (4, 4, 1)).unwrap();

    let mut lifetime = tensor.slice_mut(0, Slice::full().step(2)).unwrap();
    let l2 = lifetime.slice_mut(1, Slice::full().step(2)).unwrap();
    functor(l2);
}

// pub fn assert_with_nd_strided_tensor<T, F, B>(definition: [T; 16], expected: [T; 16], functor: F)
// where 
//     T: TensorValue,
//     B: Backend,
//     for<'a> F: FnOnce(&mut TensorViewMut<'a, T, B>),
//     TensorBase<T, B>: PartialEq<TensorBase<T, B>> + Debug
// {

//     // let yo = TensorBase::<T, B>::from_buf(definition, (4, 4, 1));
//     let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (4, 4, 1)).unwrap();

//     let mut lifetime = tensor.slice_mut(0, Slice::full().step(2)).unwrap();
//     let mut l2 = lifetime.slice_mut(1, Slice::full().step(2)).unwrap();
//     functor(&mut l2);

//     assert_eq!(tensor, TensorBase::<T, B>::from_buf(expected.to_vec(), (4, 4, 1)).unwrap());
// }


pub fn unary_assert_contiguous<T, F, FTr, B>(definition: [T; 2], truth: FTr, functor: F)
where 
    T: TensorValue,
    B: Backend,
    F: FnOnce(&mut TensorBase<T, B>),
    FTr: Fn(T) -> T,
    T: PartialEq<T> + Debug + WeightValue
{
    let mut base = TensorBase::<T, B>::from_buf(definition.to_vec(), (2, 1)).unwrap();
    let original = base.owned();
    functor(&mut base);
    for pos in original.meta().iter_coords() {
        let elem_original = original.get(pos.clone()).unwrap();
        let elem_modified = base.get(pos.clone()).unwrap();

        eps_eq(&truth(elem_original), &elem_modified);
        // assert_eq!(truth(elem_original), elem_modified);
    }
}

fn slice_1d_strided<T, B>(base: &mut TensorBase<T, B>) -> TensorViewMut<'_, T, B>
where 
    T: TensorValue,
    B: Backend
{
    base.slice_mut(0, Slice::full().step(2)).unwrap()
}

pub fn unary_assert_1d_strided<T, F, FTr, B>(definition: [T; 3], truth: FTr, functor: F)
where 
    T: TensorValue,
    B: Backend,
    F: FnOnce(&mut TensorViewMut<'_, T, B>),
    FTr: Fn(T) -> T,
    T: PartialEq<T> + Debug + WeightValue,
    // TensorBase<T, B>: Debug,
    // for<'a> TensorViewMut<'a, T, B>: Debug
{
    let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (3, 1)).unwrap();
    let mut original = tensor.clone().owned();
    let mut slice_modified = slice_1d_strided(&mut tensor);

    functor(&mut slice_modified);

    // println!("Slice: {:?}", slice);

    // let sus = tensor.owned();
    // println!("SUs: {:?}", sus);

    let slice_original = slice_1d_strided(&mut original);

    let slice = slice_modified.meta().iter_coords().collect::<Vec<_>>();

    for pos in slice {
        let elem_original = slice_original.get(pos.clone()).unwrap();
        let elem_modified = slice_modified.get(pos.clone()).unwrap();


        eps_eq(&truth(elem_original), &elem_modified);
        // assert_eq!(truth(elem_original), elem_modified);
    }
}



pub fn unary_assert_nd_strided<T, F, FTr, B>(definition: [T; 16], truth: FTr, functor: F)
where 
    T: TensorValue,
    B: Backend,
    F: FnOnce(&mut TensorViewMut<'_, T, B>),
    FTr: Fn(T) -> T,
    T: PartialEq<T> + Debug + WeightValue
{
     let mut tensor = TensorBase::<T, B>::from_buf(definition.to_vec(), (4, 4, 1)).unwrap();
    let mut original = tensor.clone();
    let mut lifetime = tensor.slice_mut(0, Slice::full().step(2)).unwrap();
    let mut l2 = lifetime.slice_mut(1, Slice::full().step(2)).unwrap();

    

    functor(&mut l2);


    let mut original_l1 = original.slice_mut(0, Slice::full().step(2)).unwrap();
    let original_l2 = original_l1.slice_mut(1, Slice::full().step(2)).unwrap();

    let slice = l2.meta().iter_coords().collect::<Vec<_>>();

    for pos in slice {

        

        let elem_original = original_l2.get(pos.clone()).unwrap();
        let elem_modified = l2.get(pos.clone()).unwrap();


        
        eps_eq(&truth(elem_original), &elem_modified);
        // assert_eq!(truth(elem_original), elem_modified);
    }
}





#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::MetaTensorView, testing::{test_with_1d_strided_tensor, test_with_contiguous_2_elem_tensor, test_with_nd_strided_tensor}};


    #[test]
    pub fn test_make_contiguous_tensor() {
        test_with_contiguous_2_elem_tensor::<_, _, Cpu>([1.0; 2],  |f| assert!(f.is_contiguous()));
    }


    #[test]
    #[cfg(feature = "cuda")]
    pub fn test_make_contiguous_tensor_cuda() {
        use crate::backend::cuda::Cuda;

      
        test_with_contiguous_2_elem_tensor::<_, _, Cuda>([1.0; 2],  |f| assert!(f.is_contiguous()));
    }

    #[test]
    pub fn test_make_1d_strided_tensor() {
        test_with_1d_strided_tensor::<_, _, Cpu>([1.0; 3],  |f| {
            assert!(!f.is_contiguous(), "The 1D strided tensor should NOT be contiguous.");
            assert_eq!(f.non_singleton_dims().len(), 1, "The non singleton dimensions should be equal to 1.");
        });
    }

    #[test]
    #[cfg(feature = "cuda")]
    pub fn test_make_1d_strided_tensor_cuda() {
        // use crate::testing::test_with_1d_strided_tensor_cuda;

        use crate::backend::cuda::Cuda;

        test_with_1d_strided_tensor::<_, _, Cuda>([1.0; 3],  |f| {
            assert!(!f.is_contiguous(), "The 1D strided tensor should NOT be contiguous.");
            assert_eq!(f.non_singleton_dims().len(), 1, "The non singleton dimensions should be equal to 1.");
        });
    }

    #[test]
    pub fn test_make_nd_tensor() {
        test_with_nd_strided_tensor::<_, _, Cpu>([0f64; 16], |f| {
            assert!(!f.is_contiguous(), "The ND strided tensor should NOT be contiguous.");
            assert_ne!(f.non_singleton_dims().len(), 1, "The non singleton dimensions should NOT be equal to 1.");
        });
    }

    #[test]
    #[cfg(feature = "cuda")]
    pub fn test_make_nd_tensor_cuda() {
        use crate::backend::cuda::Cuda;

        test_with_nd_strided_tensor::<_, _, Cuda>([0f64; 16], |f| {
            assert!(!f.is_contiguous(), "The ND strided tensor should NOT be contiguous.");
            assert_ne!(f.non_singleton_dims().len(), 1, "The non singleton dimensions should NOT be equal to 1.");
        });
    }
}