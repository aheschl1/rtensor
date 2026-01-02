use rtensors::{core::Tensor, ops::linalg::MatMul};


macro_rules! timed {
    ($block:block) => {{
        let start = std::time::Instant::now();
        let result = { $block };
        let elapsed = start.elapsed();
        (result, elapsed)
    }};
}

fn main() {
    let (result, time) = timed!({
        let a = Tensor::ones((100, 100));
        let b = Tensor::ones((100, 100));
        a.matmul(&b).unwrap()
    });
    println!("{time:?}");
}
