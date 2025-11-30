# tensors and such

Tensor primitives with arbitrary view, shape, stride. Cuda and CPU backends

Goal is high performance ML stack with minimal dependencies and maximal flexibility

## todo

- [X] Backend abstraction
- [X] Nicer syntax. macro time `tset!(tensor.view_mut(), v: 99, 1, 2)` and `tget!(tensor.view(), 1, 2)`
- [X] Basic GPU backend
- [X] Slicing with ranges `tensor.view().slice(0, 1..3)` etc.
- [X] Test more slicing syntaxes
- [ ] Slicing macro
- [X] Elementwise broadcasting
- [ ] Basic linear algebra helpers
- [X] Accelerated backends (GPU / parallel) in progress
- [ ] x86 SIMD paths (currently relying on llvm auto-vectorization for CPU which only works for contiguous memory)
- [ ] Multiple gpu devices allowed
- [X] Do not lock thread on GPU dispatch
- [ ] strides and offset as bytes 

## to optimize

- [ ] collapse dims to allow backend to access contiguous fast paths
- [ ] `view_to_owned` can probably be optimized to copy larger chunks at once
- [X] Restrict tensor values to require basic operations
- [X] perf sucks for contiguous memory in unary CPU - fix
- [X] perf sucks for contiguous memory in unary CUDA - fix
- [X] perf sucks for non-contiguous memory in unary CPU - fix
- [X] perf sucks for non-contiguous memory in unary CUDA - fix


## Slicing Syntax

```rust
// Basic range slicing
tensor.slice(0, 2..5)           // Select elements at indices 2, 3, 4
tensor.slice(0, 0..3)           // First 3 elements
tensor.slice(0, ..5)            // Elements from start to index 5 (exclusive)
tensor.slice(0, 2..)            // Elements from index 2 to end
tensor.slice(0, ..)             // All elements (full dimension)

// Inclusive ranges
tensor.slice(0, 2..=5)          // Select elements 2, 3, 4, 5 (inclusive end)
tensor.slice(0, 0..=2)          // First 3 elements (indices 0, 1, 2)

// Single index slicing
tensor.slice(0, 3)              // Select only element at index 3 (reduces dimension)

// Auto-reverse: reversed ranges automatically use negative step
tensor.slice(0, 5..2)           // Elements 5, 4, 3 (auto step=-1)
tensor.slice(0, 9..=0)          // Elements 9, 8, 7, ..., 1, 0 (auto step=-1)

// Explicit negative step using Slice builder
tensor.slice(0, Slice::from(..).step(-1))     // Reverse entire dimension
tensor.slice(0, Slice::from(8..).step(-1))    // From index 8 to start, reversed
tensor.slice(0, Slice::from(..5).step(-1))    // First 5 elements, reversed

// Custom step values
tensor.slice(0, Slice::from(..).step(2))      // Every other element (0, 2, 4, 6, ...)
tensor.slice(0, Slice::from(..).step(-2))     // Every other element, reversed
tensor.slice(0, Slice::from(1..8).step(3))    // Elements at 1, 4, 7

// manual slice construction
tensor.slice(0, Slice::new(Some(8), Some(2), Some(-2)))  // From 8 to 2, step -2: [8, 6, 4]
tensor.slice(0, Slice::new(None, None, Some(-1)))        // Full reverse
tensor.slice(0, Slice::new(Some(5), None, Some(1)))      // From 5 to end

// Edge cases
tensor.slice(0, Slice::from(1..3).step(-1))   // Empty slice (start < end with negative step)
tensor.slice(0, Slice::from(5..5))            // Empty slice (start == end)
```
