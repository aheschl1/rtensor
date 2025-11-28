
macro_rules! tget {
    ($tensor_view:expr $(, $idx:expr)+ $(,)?) => {{
        $tensor_view
            .get(Idx::Coord(&[$($idx),*]))
    }};
}

macro_rules! tset {
    ($tensor_view:expr, v:$value:expr $(, $idx:expr)+ $(,)?) => {{
        $tensor_view
            .set(Idx::Coord(&[$($idx),*]), $value)
    }};
}

// macro_rules! tslice {
//     ($tensor_view:expr $(, $range:expr)+ $(,)?) => {{
//         let base = $tensor_view;

//         let mut _idx = 0;

//         // start the chain with a binding
//         let mut _view = base.slice(_idx, $range).unwrap();
//         _idx += 1;

//         $(
//             _view = _view.slice(_idx, $range).unwrap();
//             _idx += 1;
//         )*

//         _view
//     }};
// }



#[cfg(test)]
mod tests {
    use crate::core::{tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}, value::TensorValue, CpuTensor, MetaTensorView, Shape};
    use crate::core::idx::Idx;


    fn make_tensor<T: TensorValue>(buf: Vec<T>, shape: Shape) -> CpuTensor<T> {
        CpuTensor::from_buf(buf, shape).unwrap()
    }

    #[test]
    fn test_tget_macro() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        assert_eq!(tget!(tensor.view(), 0, 0).unwrap(), 1);
        assert_eq!(tget!(tensor.view(), 0, 2).unwrap(), 3);
        assert_eq!(tget!(tensor.view(), 1, 1).unwrap(), 5);
    }

    // expect failure for too many indices
    #[test]
    #[should_panic]
    fn test_tget_macro_too_many_indices() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let _ = tget!(tensor.view(), 0, 0, 0).unwrap();
    }

    // test tset macro
    #[test]
    fn test_tset_macro() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tset!(tensor.view_mut(), v: 99, 1, 2), Ok(())));
        assert_eq!(tget!(tensor.view(), 1, 2).unwrap(), 99);
    }

    // #[test]
    // fn test_tslice_macro() {
    //     let tensor = make_tensor(vec![
    //         1, 2, 3,
    //         4, 5, 6,
    //         7, 8, 9
    //     ], vec![3, 3]);

    //     // Slice rows 1..3 and columns 0..2
    //     let slice = tslice!(tensor.view(), 1..3, 0..2);
    //     assert_eq!(*slice.shape(), vec![2, 2]);
    //     assert_eq!(tget!(slice, 0, 0).unwrap(), 4);
    //     assert_eq!(tget!(slice, 0, 1).unwrap(), 5);
    //     assert_eq!(tget!(slice, 1, 0).unwrap(), 7);
    //     assert_eq!(tget!(slice, 1, 1).unwrap(), 8);
    // }
}