# tensors and such

Zero-dependency row‑major N‑dimensional tensor primitives.

Goal to add CUDA and optimized operations.

### todo

- [ ] Nicer syntax. macro time
- [ ] Slicing with ranges
- [ ] Elementwise broadcasting
- [ ] Basic linear algebra helpers
- [ ] Accelerated backends (GPU / parallel)
- [ ] x86 SIMD paths
- [ ] Nicer indexing syntax sugar

### example

```rust
use tengine::core::{Tensor, idx::Idx, MetaTensorView, tensor::{TensorAccess, TensorMut}};
// Matrix 2x3 (row-major):
let m = Tensor::from_buf(vec![1,2,3,4,5,6], vec![2,3]).unwrap();
assert_eq!(m.shape(), &[2,3]);

// Indexing (panic on failure):
assert_eq!(m[vec![0, 1]], 2);

// Safe access via trait:
let v = m.view();
assert_eq!(*v.get(&Idx::Coord(&[1,2])).unwrap(), 6);

// Slice first dimension (row 1): shape becomes [3]
let row1 = v.slice(0, 1).unwrap();
assert_eq!(row1.shape(), &[3]);
assert_eq!(row1[vec![2]], 6);

// Mutation through a mutable view:
let mut m2 = Tensor::row(vec![10, 20, 30]);
let mut mv = m2.view_mut();
mv.set(&Idx::At(1), 99).unwrap();
assert_eq!(m2[vec![0,1]], 99);
```