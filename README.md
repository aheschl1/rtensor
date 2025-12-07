# tensors and such

Tensor primitives with arbitrary view, shape, stride. Cuda and CPU backends

Goal is high performance ML stack with minimal dependencies and maximal flexibility

## todo

- [X] Backend abstraction
- [X] Nicer syntax. macro time `set!(tensor, v: 99, 1, 2)` and `get!(tensor, 1, 2)` and `coord![1, 2, ..]`
- [X] Basic GPU backend
- [X] Slicing with ranges `tensor.view().slice(0, 1..3)` etc.
- [X] Test more slicing syntaxes
- [ ] Slicing macro
- [X] Elementwise broadcasting
- [X] Basic linear algebra helpers
- [X] Accelerated backends (GPU / parallel) in progress
- [ ] x86 SIMD paths (currently relying on llvm auto-vectorization for CPU which only works for contiguous memory)
- [ ] Multiple gpu devices allowed
- [X] Do not lock thread on GPU dispatch
- [ ] strides and offset as bytes  
- [X] Broadcasting
- [X] Idx should not be ref. makes it less ergonomic
- [ ] Pull out ops into crate defined traits, which return Result, and call that from Add and AddAssign impls (panic there)
- [ ] Figure outt bool types, and in general those without Add, Sub, and Mul impls
- [ ] Allow step_by for slicing iterator
- [X] OpenBlas for more targets, investigate build system
- [X] cuBLAS

## to optimize

- [ ] collapse dims to allow backend to access contiguous fast paths
- [ ] `view_to_owned` can probably be optimized to copy larger chunks at once
- [X] Restrict tensor values to require basic operations
- [X] perf sucks for contiguous memory in unary CPU - fix
- [X] perf sucks for contiguous memory in unary CUDA - fix
- [X] perf sucks for non-contiguous memory in unary CPU - fix
- [X] perf sucks for non-contiguous memory in unary CUDA - fix
- [ ] O(rank * size) instead of O(size) broadcasting ops is bad
- [ ] Broadcast cuda kernel puts a cap on tensor dim size - fix
- [ ] Allow F contiguous for matmul, as well as C.
- [ ] Matmul not using openblas for other than T=f32/f64

## Some examples

Note that they may not reflect most up to date API and syntax (assume reality can only be better).

### Creating Tensors

```rust
let zeros = TensorBase::<f32, Cpu>::zeros((3, 4)); // 3x4 tensor of zeros

let cpu_ones = Tensor::<f32>::ones((2, 2)); // 2x2 tensor of ones
let gpu_max = CudaTensor::<f32>::max((5, 5)); // 5x5 tensor of maximum f32 values on GPU
let gpu_min = CudaTensor::<i32>::min((10, 1)); // 10x1 tensor of minimum i32 values on GPU

let cpu_tensor = Tensor::<f64>::from_buf(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
let gpu_tensor = cpu_tensor.cuda();
let cpu2 = gpu_tensor.cpu();

let buf = vec![0.0f32; 16];
let tensor = TensorBase::<f32, Cuda>::from_buf(buf, (4, 4));

// read

let value = tensor.get(coord![2, 3]).unwrap(); // get value at (2, 3)
// or
let value = get!(tensor, 2, 3).unwrap();
let value = tensor.get(5).unwrap(); // single-dimensional index
let value = tensor.get((0, 0)).unwrap(); // tuple index
// or explicit
let value = tensor.get(&Idx::Coord(vec![2, 3])).unwrap();

// write
tensor.set(&Idx::Coord(vec![1, 1]), 42.0).unwrap();
// or
set!(tensor, v: 42.0, 1, 1);

```

### Arithmetic Operations

```rust
let mut a = Tensor::<f32>::ones((2, 2));

a*= 3.0; 
a+= 2.0;
a-= 1.0;

let b: Tensor::<f32> = a * 2.0; 
```

### Slicing Syntax

```rust
// Basic range slicing
// slice(dimension, range), so modifications happen on the dimension specified
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
tensor.slice(0, Slice::from(..5).step(-1))    // last element to index 5, reversed

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

### Broadcasting

#### Broadcasting Rules

1. If tensors have different ranks, prepend 1s to the shape of the smaller rank tensor
2. For each dimension, the sizes must either:
   - Be equal, OR
   - One of them must be 1 (which gets broadcasted)

If inplace, then the tensor being modified must be the same shape the result shape after broadcasting.

#### Basic Broadcasting Examples

```rust
// Scalar to any shape - broadcasts everywhere
let scalar = Tensor::<f32>::from_buf(vec![5.0], vec![]).unwrap();
let matrix = Tensor::<f32>::ones((3, 4));
let result = scalar + matrix;  // Shape: (3, 4), all values are 6.0

// Vector to matrix - broadcasts along rows
let vector = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
let matrix = Tensor::<f32>::ones((4, 3));
let result = vector + matrix;  // Shape: (4, 3), each row is [2, 3, 4]

// Row vector + Column vector = Matrix (also, this works with views - for example, slices)
let row = Tensor::<f32>::ones((1, 4));    // Shape: (1, 4)
let col = Tensor::<f32>::ones((3, 1));    // Shape: (3, 1)
let result = row.view() + col.view();         // Shape: (3, 4), all 2.0
```

```rust
// Matrix broadcasts to 3D tensor
let matrix = Tensor::<f32>::ones((3, 4));      // Shape: (3, 4)
let tensor_3d = Tensor::<f32>::ones((2, 3, 4)); // Shape: (2, 3, 4)
let result = matrix + tensor_3d;     // Shape: (2, 3, 4)

// Singleton dimensions broadcast
let a = Tensor::<f32>::ones((1, 3, 1));    // Shape: (1, 3, 1)
let b = Tensor::<f32>::ones((2, 3, 4));    // Shape: (2, 3, 4)
let result = a.view() + b;                   // Shape: (2, 3, 4)
```

```rust
// 4D Broadcasting
let vector = Tensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap();
let tensor_4d = Tensor::<f32>::ones((3, 4, 5, 2));
let result = vector + tensor_4d.view();  // Shape: (3, 4, 5, 2)

// Complex singleton pattern
let a = Tensor::<f32>::ones((1, 1, 5, 1));  // Shape: (1, 1, 5, 1)
let b = Tensor::<f32>::ones((2, 3, 5, 4));  // Shape: (2, 3, 5, 4)
let result = a + b;               // Shape: (2, 3, 5, 4)
```

```rust
// Classic: row vector + column vector
let row = Tensor::<f32>::ones((1, 4));      // Shape: (1, 4)
let col = Tensor::<f32>::ones((3, 1));      // Shape: (3, 1)
let result = row + col;           // Shape: (3, 4)

// 3D mutual broadcasting
let a = Tensor::<f32>::ones((1, 4, 5));     // Shape: (1, 4, 5)
let b = Tensor::<f32>::ones((3, 1, 5));     // Shape: (3, 1, 5)
let result = a + b;               // Shape: (3, 4, 5)

// Complex 4D mutual broadcasting
let a = Tensor::<f32>::ones((1, 3, 1, 5));  // Shape: (1, 3, 1, 5)
let b = Tensor::<f32>::ones((2, 1, 4, 1));  // Shape: (2, 1, 4, 1)
let result = a + b;               // Shape: (2, 3, 4, 5)
```

#### Inplace Broadcasting Operations

```rust
let mut a = Tensor::<f32>::zeros((3, 4));
let b = Tensor::<f32>::ones((4,));  // Vector of shape (4)
a += b ;  // b broadcasts along rows of a
```

### Permutations and Transposes

```rust
// Permutation example
let tensor = Tensor::<i32>::ones((1, 2, 3)); // Shape: (1, 2, 3)
let permuted = tensor.permute(vec![2, 0, 1]).unwrap(); // Shape: (3, 1, 2)
```

```rust
// Transpose example
let tensor = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
let transposed = tensor.transpose().unwrap(); // Shape: (3, 2) - permutes all dimensions
```

### Matmul

With f32 and f64, OpenBlas is used for CPU backend, cuBLAS for CUDA backend.

```rust
let a = Tensor::<f32>::from_buf(
    vec![1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0],
    (3, 3),
).unwrap();


let b = Tensor::<f32>::from_buf(
    vec![1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0],
    (3, 2),
).unwrap();

let result = a.matmul(&b).unwrap(); // Shape: (3, 2)

// CUDA

let a_gpu = a.cuda();
let b_gpu = b.cuda();
let result_gpu = a_gpu.matmul(&b_gpu).unwrap(); // Shape: (3, 2)
```
