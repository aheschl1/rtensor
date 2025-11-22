use std::ops::{Index, IndexMut};


#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    IdxOutOfBounds,
    WrongDims,
    InvalidShape,
    InvalidDim
}

type Dim = usize;
type Stride = Vec<usize>;
type Shape = Vec<Dim>;

/// indices are most seperate first. so: [column] for 1d, [row, column] for 2d, [depth, row, column] for 3d 
type Idx = [usize];

fn shape_to_stride(shape: &Shape) -> Stride {
    let mut stride = vec![1; shape.len()];
        
    for i in (0..shape.len()).rev(){
        if i < shape.len() - 1 {
            stride[i] = stride[i+1] * shape[i+1];
        }
    }

    stride
}

#[derive(Debug)]
pub struct TensorView<'a, T: Sized>{
    raw: &'a [T], // row major order
    stride: Stride,
    pub shape: Shape,
    offset: usize,
}

impl<T: Sized> TensorView<'_, T> {
    pub fn get(&self, idx: &Idx) -> Result<&T, TensorError> {
        if idx.len() != self.stride.len() {
            return Err(TensorError::WrongDims)
        }
        let bidx = idx
            .iter()
            .zip(&self.stride)
            .fold(self.offset, |acc, (a, b)| acc + *a*b);
        self.raw.get(bidx).ok_or(TensorError::IdxOutOfBounds)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Tensor<T: Sized>{
    raw: Box<[T]>, // row major order
    stride: Stride,
    pub shape: Shape
}

impl<T: Sized> Tensor<T> {
    pub fn from_buf(raw: impl Into<Box<[T]>>, shape: Shape) -> Result<Self, TensorError>{
        let raw = raw.into();
        if shape.iter().fold(1, |p, x| p*x) != raw.len() {
            return Err(TensorError::InvalidShape);
        }
        Ok(Self{
            raw,
            stride: shape_to_stride(&shape),
            shape: shape,
        })
    }

    pub fn scalar(value: T) -> Self {
        Self{
            shape: vec![],
            raw: vec![value].into(),
            stride: vec![],
        }
    }

    pub fn column(column: impl Into<Box<[T]>>) -> Self {
        let column = column.into();
        Self{
            shape: vec![column.len()],
            raw: column,
            stride: vec![1],
        }
    }

    pub fn row(row: impl Into<Box<[T]>>) -> Self {
        let row = row.into();
        Self{
            stride: vec![row.len(), 1],
            shape: vec![1, row.len()],
            raw: row,
        }
    }

    pub fn empty() -> Tensor<()> {
        Tensor::<()>{
            raw: vec![].into(),
            stride: vec![],
            shape: vec![],
        }
    }

    pub fn get(&self, idx: &Idx) -> Result<&T, TensorError> {
        if idx.len() != self.stride.len() {
            return Err(TensorError::WrongDims)
        }
        let bidx = idx
            .iter()
            .zip(&self.stride)
            .fold(0, |acc, (a, b)| acc + *a*b);
        self.raw.get(bidx).ok_or(TensorError::IdxOutOfBounds)
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn get_mut(&mut self, idx: &Idx) -> Result<&mut T, TensorError> {
        if idx.len() != self.stride.len() {
            return Err(TensorError::WrongDims)
        }
        let bidx = idx
            .iter()
            .zip(&self.stride)
            .fold(0, |acc, (a, b)| acc + *a*b);
        self.raw.get_mut(bidx).ok_or(TensorError::IdxOutOfBounds)
    }

    /// Slices the tensor along the given dimension, at the given index.
    /// Returns a TensorView with the sliced data.
    pub fn slice<'a>(&'a self, dim: Dim, idx: Dim) -> Result<TensorView<'a, T>, TensorError> {
        if dim >= self.shape.len() {
            return Err(TensorError::InvalidDim);
        }
        if idx >= self.shape[dim] {
            return Err(TensorError::IdxOutOfBounds);
        }
        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);
        let mut new_stride = self.stride.clone();
        new_stride.remove(dim);
        Ok(TensorView{
            raw: &self.raw,
            stride: new_stride,
            shape: new_shape,
            offset: self.stride[dim] * idx,
        })
    }
}

impl<'a, T> Into<TensorView<'a, T>> for &'a Tensor<T> {
    fn into(self) -> TensorView<'a, T> {
        TensorView{
            raw: &self.raw,
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            offset: 0,
        }
    }
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    /// Panics if index is out of bounds or shape is invalid
    /// Use get() for a non-panicking version
    fn index(&self, index: &[usize]) -> &Self::Output {
        match self.get(index){
            Ok(v) => v,
            Err(TensorError::IdxOutOfBounds) => panic!("Index out of bounds"),
            Err(TensorError::InvalidShape) => panic!("Invalid shape"),
            _ => panic!("Indexing error"),
        }
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    /// Panics if index is out of bounds or shape is invalid
    /// Use get_mut() for a non-panicking version
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        match self.get_mut(index){
            Ok(v) => v,
            Err(TensorError::IdxOutOfBounds) => panic!("Index out of bounds"),
            Err(TensorError::InvalidShape) => panic!("Invalid shape"),
            _ => panic!("Indexing error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::slice;

    use crate::tensor::{Shape, Stride, Tensor};

    fn make_tensor<T>(buf: Vec<T>, shape: Shape) -> Tensor<T>{
        Tensor::from_buf(buf, shape).unwrap()
    }

    #[test]
    fn test_slice_matrix(){
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);

        let slice = tensor.slice(0, 0).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(slice.shape, vec![3]);
        assert_eq!(slice.stride, vec![1]);
        assert_eq!(*slice.get(&[0]).unwrap(), 1);
        assert_eq!(*slice.get(&[1]).unwrap(), 2);
        assert_eq!(*slice.get(&[2]).unwrap(), 3);

        let slice2 = tensor.slice(1, 0).unwrap(); // slice along columns, should give a view of shape [2]
        assert_eq!(slice2.shape, vec![2]);
        assert_eq!(slice2.stride, vec![3]);
        assert_eq!(*slice2.get(&[0]).unwrap(), 1);
        assert_eq!(*slice2.get(&[1]).unwrap(), 4);
    }

    #[test]
    fn test_slice_cube(){
        let buf = vec![
            1, 2,
            4, 5,

            6, 7,
            8, 9
        ];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);

        let slice = tensor.slice(0, 0).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(slice.shape, vec![2, 2]);
        assert_eq!(slice.stride, vec![2, 1]);
        assert_eq!(*slice.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*slice.get(&[0, 1]).unwrap(), 2);
        assert_eq!(*slice.get(&[1, 0]).unwrap(), 4);
        assert_eq!(*slice.get(&[1, 1]).unwrap(), 5);

        // second depth
        let slice_second_depth = tensor.slice(0, 1).unwrap();
        assert_eq!(slice_second_depth.shape, vec![2, 2]);
        assert_eq!(slice_second_depth.stride, vec![2, 1]);
        assert_eq!(*slice_second_depth.get(&[0, 0]).unwrap(), 6);
        assert_eq!(*slice_second_depth.get(&[0, 1]).unwrap(), 7);
        assert_eq!(*slice_second_depth.get(&[1, 0]).unwrap(), 8);
        assert_eq!(*slice_second_depth.get(&[1, 1]).unwrap(), 9);

        let slice2 = tensor.slice(1, 0).unwrap(); // slice along row, should give a view of shape [2, 2]
        assert_eq!(slice2.shape, vec![2, 2]);
        assert_eq!(slice2.stride, vec![4, 1]);
        assert_eq!(*slice2.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*slice2.get(&[0, 1]).unwrap(), 2);
        assert_eq!(*slice2.get(&[1, 0]).unwrap(), 6);
        assert_eq!(*slice2.get(&[1, 1]).unwrap(), 7);

        // column slice
        let slice3 = tensor.slice(2, 0).unwrap(); // slice along column
        assert_eq!(slice3.shape, vec![2, 2]);
        assert_eq!(slice3.stride, vec![4, 2]);
        assert_eq!(*slice3.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*slice3.get(&[0, 1]).unwrap(), 4);
        assert_eq!(*slice3.get(&[1, 0]).unwrap(), 6);
        assert_eq!(*slice3.get(&[1, 1]).unwrap(), 8);
    }

    #[test]
    fn test_column() {
        let tensor = Tensor::column(vec![1, 2, 3]);
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(*tensor.get(&[0]).unwrap(), 1);
        assert_eq!(*tensor.get(&[1]).unwrap(), 2);
        assert_eq!(*tensor.get(&[2]).unwrap(), 3);
    }

    #[test]
    fn test_row() {
        let tensor = Tensor::row(vec![1, 2, 3]);
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*tensor.get(&[0, 1]).unwrap(), 2);
        assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3);

        assert_eq!(tensor[&[0, 1]], 2);
    }

    #[test]
    fn test_empty() {
        let tensor = Tensor::<()>::empty();
        assert_eq!(tensor.shape, vec![]);
        assert!(tensor.raw.is_empty());
        assert!(tensor.stride.is_empty());
    }

    #[test]
    fn test_scalar(){
        let buf = vec![42];
        let shape = vec![];
        let tensor = make_tensor(buf, shape);

        assert_eq!(*tensor.get(&[]).unwrap(), 42);
        assert!(tensor.is_scalar());
        assert_eq!(Tensor::scalar(42), tensor);

    }

    #[test]
    fn test_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(*tensor.get(&[0]).unwrap(), 1);
        assert_eq!(*tensor.get(&[1]).unwrap(), 2);
        assert_eq!(*tensor.get(&[2]).unwrap(), 3);
        
        *tensor.get_mut(&[1]).unwrap() = 1;
        assert_eq!(*tensor.get(&[1]).unwrap(), 1);
    }

    #[test]
    fn test_matrix() {
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
        ];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*tensor.get(&[0, 1]).unwrap(), 2);
        assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3);
        assert_eq!(*tensor.get(&[1, 0]).unwrap(), 4);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5);
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6);
        
        *tensor.get_mut(&[1, 2]).unwrap() = 100;
        
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 100);
    }

    #[test]
    fn test_cube() {
        //
        let buf = vec![
            1, 2,
            4, 5,

            6, 7,
            8, 9
        ];
        let shape = vec![2, 2, 2];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), 1); // depth, row, column
        assert_eq!(*tensor.get(&[0, 0, 1]).unwrap(), 2);
        assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 4);
        assert_eq!(*tensor.get(&[0, 1, 1]).unwrap(), 5);
        assert_eq!(*tensor.get(&[1, 0, 0]).unwrap(), 6);
        assert_eq!(*tensor.get(&[1, 0, 1]).unwrap(), 7);
        assert_eq!(*tensor.get(&[1, 1, 0]).unwrap(), 8);
        assert_eq!(*tensor.get(&[1, 1, 1]).unwrap(), 9);
        
        // modify
        *tensor.get_mut(&[1, 0, 0]).unwrap() = 67;
        assert_eq!(*tensor.get(&[1, 0, 0]).unwrap(), 67);
    }

    #[test]
    fn test_shape_to_stride() {
        let shape = vec![2, 2, 3];
    let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![6, 3, 1]);
    }

    #[test]
    fn test_shape_to_stride_single_dim() {
        let shape = vec![4];
    let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_shape_to_stride_empty() {
        let shape: Shape = vec![];
    let stride: Stride = super::shape_to_stride(&shape);

        assert!(stride.is_empty());
    }

    #[test]
    fn test_shape_to_stride_ones() {
        let shape = vec![1, 1, 1];
    let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![1, 1, 1]);
    }

    #[test]
    fn test_shape_to_stride_mixed() {
        let shape = vec![5, 1, 2];
    let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![2, 2, 1]);
    }

    #[test]
    fn test_shape_to_stride_larger() {
        let shape = vec![3, 4, 5];
    let stride: Stride = super::shape_to_stride(&shape);

        assert_eq!(stride, vec![20, 5, 1]);
    }
}