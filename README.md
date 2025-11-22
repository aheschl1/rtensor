# tensors and such

This crate provides a 0 dependency row-major N-dimensional tensor abstraction with owned and view types.

TODO:

- [ ] Slicing with ranges
- [ ] Broadcasting elemtwise
- [ ] Lin alg
- [ ] accelerated backends
- [ ] x86 SIMD
- [ ] Nicer indexing syntax

## Core Types

- `TensorOwned<T>`: Owns a contiguous buffer (`Box<[T]>`) with a `shape: Vec<usize>` and derived `stride: Vec<usize>`.
- `TensorView<'a, T>` / `TensorViewMut<'a, T>`: Non-owning views (mutable or immutable) into an existing tensor slice; store an `offset`, `shape`, and `stride`.
- Shapes are length-per-dimension in row-major order. Strides are element steps per dimension (last dim stride = 1).

## Creating Tensors

```rust
use tengine::ndarray::TensorOwned;

// Scalar (shape = [])
let s = TensorOwned::scalar(99);

// Column vector (shape = [3])
let col = TensorOwned::column(vec![1, 2, 3]);

// Row vector (shape = [1, 3])
let row = TensorOwned::row(vec![1, 2, 3]);

// Matrix from buffer (shape = [2, 3])
let m = TensorOwned::from_buf(vec![1,2,3,4,5,6].into_boxed_slice(), vec![2,3]).unwrap();
```

## Indexing

```rust
assert_eq!(m[vec![0, 1]], 2); // row 0, col 1
```

Programmatic access uses the `Idx` enum (`At`, `Coord`, `Item`) via `get` / `item`:

```rust
use tengine::ndarray::tensor::{Idx, Tensor};
let v = m.get(&Idx::Coord(vec![1, 2])).unwrap();
```

Invalid shapes or out-of-bounds coordinates return a `TensorError` (or panic via `[]`).

## Mutation

```rust
use tengine::ndarray::tensor::TensorMut;
let mut mm = m; // from earlier
*mm.get_mut(&Idx::Coord(vec![1,2])).unwrap() = 42;
assert_eq!(mm[vec![1,2]], 42);

// Convenience with IndexMut
mm[vec![0,0]] = 10;
```

`set(&Idx, value)` returns `Result<(), TensorError>` when you prefer error handling over panics.

## Slicing Views

`slice(dim, idx)` produces a view with one fewer dimension (selecting a fixed index along `dim`).

```rust
let row_view = mm.slice(0, 1).unwrap(); // shape [3]
assert_eq!(row_view[vec![2]], 6);

let mut row_view_mut = mm.slice_mut(0, 0).unwrap();
row_view_mut[vec![1]] = 50; // reflects in original tensor
assert_eq!(mm[vec![0,1]], 50);
```

Slicing a 1-D tensor to a scalar yields shape `[]`.

## Reshaping Views

`view_as(new_shape)` changes shape/stride if total element count matches.

```rust
let view = mm.view();            // shape [2,3]
let flat = view.view_as(vec![6]).unwrap(); // shape [6], stride [1]
assert_eq!(flat[vec![4]], 5);

let reshaped = view.view_as(vec![3,2]).unwrap();
assert_eq!(reshaped[vec![2,1]], 6);
```

Fails with `TensorError::InvalidShape` if element counts differ.

## Scalars, Rows, Columns Helpers

```rust
assert!(TensorOwned::scalar(1).is_scalar());
assert!(TensorOwned::row(vec![1,2,3]).is_row());
assert!(TensorOwned::column(vec![1,2,3]).is_column());
```

## Error Summary

- `InvalidShape`: Buffer length does not match requested shape or reshape mismatch.
- `IdxOutOfBounds`: Coordinate exceeds dimension length.
- `WrongDims`: Provided index rank mismatches tensor rank (including using `Item` on non-scalar).
- `InvalidDim`: Slice dimension out of range.
