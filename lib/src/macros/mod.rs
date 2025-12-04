
#[macro_export]
macro_rules! coord {
    ($($idx:expr),* $(,)?) => {
        Idx::Coord(vec![$($idx),*])
    };
}

#[macro_export]
macro_rules! get {
    ($tensor_view:expr $(, $idx:expr)+ $(,)?) => {{
        $tensor_view
        .get(Idx::Coord(vec![$($idx),*]))
    }};
}

#[macro_export]
macro_rules! set {
    ($tensor_view:expr, v:$value:expr $(, $idx:expr)+ $(,)?) => {{
        $tensor_view
            .set(Idx::Coord(vec![$($idx),*]), $value)
    }};
}


#[cfg(test)]
mod tests {
    use crate::core::{value::TensorValue, Tensor, Shape};
    use crate::core::tensor::{TensorAccess, TensorAccessMut};
    use crate::core::idx::Idx;


    fn make_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Tensor<T> {
        Tensor::from_buf(buf, shape).unwrap()
    }

    #[test]
    fn test_tcoord_macro() {
        let idx = coord![0, 1, 2];
        match idx {
            Idx::Coord(coords) => {
                assert_eq!(coords, vec!(0, 1, 2));
            },
            _ => panic!("Expected Idx::Coord variant"),
        }
    }

    #[test]
    fn test_tget_macro() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        assert_eq!(get!(tensor, 0, 0).unwrap(), 1);
        assert_eq!(get!(tensor, 0, 2).unwrap(), 3);
        assert_eq!(get!(tensor, 1, 1).unwrap(), 5);
    }

    // expect failure for too many indices
    #[test]
    #[should_panic]
    fn test_tget_macro_too_many_indices() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let _ = get!(tensor, 0, 0, 0).unwrap();
    }

    // test tset macro
    #[test]
    fn test_tset_macro() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(set!(tensor, v: 99, 1, 2), Ok(())));
        assert_eq!(get!(tensor, 1, 2).unwrap(), 99);
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