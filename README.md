# rtensors

Tensor primitives with CUDA and CPU backends. Remote execution protocol built in at backend level.
Uses BLAS, cuBLAS, and custom kernels.

Goal is high performance ML stack with minimal dependencies, and the flexibility of numpy.

## Creating Tensors

```rust
// Fill constructors
let zeros = Tensor::<f32>::zeros((3, 4));
let ones = Tensor::<f32>::ones((2, 2));
let min_vals = Tensor::<i32>::min((10, 5));
let max_vals = Tensor::<f32>::max((5, 5));

// From data
let tensor = Tensor::<f64>::from_buf(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).unwrap();

// Special shapes
let scalar = Tensor::<f32>::scalar(42.0);
let column = Tensor::<f32>::column(vec![1.0, 2.0, 3.0]);
let row = Tensor::<f32>::row(vec![1.0, 2.0, 3.0]);

// Without type alias
let tensor = TensorBase::<f32, Cpu>::from_buf(vec![1.0, 2.0, 3.0], (3,)).unwrap();
```

## CPU to and from GPU

```rust
let cpu_tensor = Tensor::<f32>::ones((2, 2));
let gpu_tensor = cpu_tensor.cuda().unwrap();
let back_to_cpu = gpu_tensor.cpu().unwrap();
```

## Indexing

```rust
let tensor = Tensor::<f32>::zeros((4, 4));

// Read - many ways, all equivalent
let value = tensor.get((2, 3)).unwrap();
let value = get!(tensor, 2, 3).unwrap();
let value = tensor.get(coord![2, 3]).unwrap();

// Write
let mut tensor = Tensor::<f32>::zeros((4, 4));
tensor.set((1, 1), 42.0).unwrap();
set!(tensor, v: 42.0, 1, 1).unwrap();
```

## Arithmetic

```rust
let mut a = Tensor::<f32>::ones((2, 2));

a *= 3.0; 
a += 2.0;
a -= 1.0;

let b = a * 2.0; 
let c = a + b;
```

## Slicing

```rust
let tensor = Tensor::<f32>::ones((10, 10));

// Range slicing
let slice = tensor.slice(0, 2..5).unwrap();      // rows 2-4
let slice = tensor.slice(0, 0..3).unwrap();      // first 3 rows
let slice = tensor.slice(0, ..5).unwrap();       // up to row 5
let slice = tensor.slice(0, 2..).unwrap();       // from row 2 to end
let slice = tensor.slice(0, ..).unwrap();        // all rows

// Inclusive ranges
let slice = tensor.slice(0, 2..=5).unwrap();     // rows 2-5 inclusive

// Single index
let row = tensor.slice(0, 3).unwrap();           // just row 3

// Reverse with auto-step
let slice = tensor.slice(0, 5..2).unwrap();      // rows 5, 4, 3

// Custom steps
let slice = tensor.slice(0, Slice::full().step(-1)).unwrap();  // reverse all
let slice = tensor.slice(0, Slice::from(..).step(2)).unwrap(); // every other
let slice = tensor.slice(0, Slice::new(Some(8), Some(2), -2)).unwrap(); // 8, 6, 4
```

## Broadcasting

Broadcasting rules:

1. Align shapes from the right
2. Dimensions must match or one must be 1
3. Missing dimensions are treated as 1

```rust
// Scalar broadcasts to any shape
let scalar = Tensor::<f32>::scalar(5.0);
let matrix = Tensor::<f32>::ones((3, 4));
let result = scalar + matrix;  // (3, 4)

// Vector to matrix
let vector = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], (3,)).unwrap();
let matrix = Tensor::<f32>::ones((4, 3));
let result = vector + matrix;  // (4, 3)

// Row + column = matrix
let row = Tensor::<f32>::ones((1, 4));
let col = Tensor::<f32>::ones((3, 1));
let result = row + col;  // (3, 4)

// Higher dimensions
let a = Tensor::<f32>::ones((1, 3, 1));
let b = Tensor::<f32>::ones((2, 3, 4));
let result = a + b;  // (2, 3, 4)

// Works with views too
let result = row.view() + col.view();

// In-place
let mut a = Tensor::<f32>::zeros((3, 4));
let b = Tensor::<f32>::ones((4,));
a += b;  // b broadcasts to each row
```

## Shape Manipulation

```rust
let tensor = Tensor::<f32>::ones((2, 3, 4));

// Permute dimensions
let permuted = tensor.permute(vec![2, 0, 1]).unwrap();  // (4, 2, 3)

// Transpose (reverse all dimensions)
let transposed = tensor.transpose();  // (4, 3, 2)

// Add dimensions
let unsqueezed = tensor.unsqueeze();  // (1, 2, 3, 4)
let unsqueezed = tensor.unsqueeze_at(1).unwrap();  // (2, 1, 3, 4)

// Remove size-1 dimensions
let squeezed = unsqueezed.squeeze();  // removes all 1s
let squeezed = unsqueezed.squeeze_at(1).unwrap();  // remove specific dim
```

## Matrix Multiplication

f32/f64 use OpenBLAS (CPU) or cuBLAS (CUDA).

```rust
let a = Tensor::<f32>::from_buf(
    vec![1.0, 2.0, 3.0,
         4.0, 5.0, 6.0],
    (2, 3),
).unwrap();

let b = Tensor::<f32>::from_buf(
    vec![1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0],
    (3, 2),
).unwrap();

let result = a.matmul(&b).unwrap();  // (2, 2)

// On GPU
let a_gpu = a.cuda().unwrap();
let b_gpu = b.cuda().unwrap();
let result = a_gpu.matmul(&b_gpu).unwrap();

// Dot product (1D)
let a = Tensor::<f32>::ones((3,));
let b = Tensor::<f32>::from_buf(vec![4.0, 5.0, 6.0], (3,)).unwrap();
let result = a.dot(&b).unwrap();  // scalar: 15.0
```

## Remote Backend

Still early stage in terms of ergonomics of use.

The protocol includes asynchronous operations, which are transparently handled.
When a long running operation which acts inplace of its operators only, an Ack is returned immediately,
and the user process can continue. When a read is needed, there is a sync point.

For example:

```rust
let remote_tensor = RemoteTensor::<f32>::ones((1000, 1000));
remote_tensor += 1; // async, will continue immediately, though the computation is long
let value = remote_tensor.get((0, 0)).unwrap(); // sync point, waits for prior ops to finish
```

### Start a server

```rust
use crate::backend::remote::server::launch_server;
launch_server("127.0.0.1", 7878);
```

```rust
// Initialize remote backend
remote_backend_init("127.0.0.1", 7878);
let remote_tensor = RemoteTensor::<f32>::ones((3, 4));
// Or, specify a backend directly
let backend = RemoteBackend::new("127.0.0.1", 7878);
backend.connect().unwrap();
let remote_tensor = RemoteTensor::from_parts(backend, vec![1, 2, 3, 4, 5, 6], Shape::from((2, 2))).unwrap();
```
