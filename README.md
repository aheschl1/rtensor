# tensors and such

Tensor primitives with arbitrary view, shape, stride. Cuda and CPU backends

Goal is high performance ML stack with minimal dependencies and maximal flexibility

## todo

- [X] Backend abstraction
- [X] Nicer syntax. macro time `tset!(tensor.view_mut(), v: 99, 1, 2)` and `tget!(tensor.view(), 1, 2)`
- [X] Basic GPU backend
- [ ] Slicing with ranges
- [X] Elementwise broadcasting
- [ ] Basic linear algebra helpers
- [X] Accelerated backends (GPU / parallel) in progress
- [ ] x86 SIMD paths (currently relying on llvm auto-vectorization for CPU which only works for contiguous memory)
- [ ] Multiple gpu devices allowed
- [ ] CUDA scheduler for syncing and fusing ops
- [X] Do not lock thread on GPU dispatch

## to optimize

- [ ] Elementwise ops require computing all offsets every time. perhaps cache them
- [ ] `view_to_owned` can probably be optimized to copy larger chunks at once
- [ ] Restrict tensor values to require basic operations
- [X] perf sucks for contiguous memory in unary CPU - fix
- [X] perf sucks for contiguous memory in unary CUDA - fix
- [ ] perf sucks for non-contiguous memory in unary CPU - fix
- [X] perf sucks for non-contiguous memory in unary CUDA - fix
