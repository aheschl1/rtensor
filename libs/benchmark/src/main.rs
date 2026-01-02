use rtensors::core::primitives::CudaTensor;
use rtensors::{core::Tensor, ops::linalg::MatMul};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyDict};
use std::fs::{OpenOptions, File};
use std::io::Write;
use std::path::Path;

/// Manages a benchmarking session with CSV output and visualizations
struct BenchmarkSession {
    csv_path: String,
    output_prefix: String,
}

impl BenchmarkSession {
    /// Start a new benchmark session with the given name
    fn new(session_name: &str) -> Self {
        let csv_path = format!("{}_benchmarks.csv", session_name);
        let output_prefix = format!("{}_results", session_name);
        
        // Remove old CSV file if it exists (start fresh)
        if Path::new(&csv_path).exists() {
            std::fs::remove_file(&csv_path).ok();
        }
        
        // Create new CSV with header
        let mut file = File::create(&csv_path).expect("Failed to create CSV file");
        writeln!(file, "benchmark_name,rtensors_time_us,pytorch_time_us,speedup")
            .expect("Failed to write CSV header");
        
        Self {
            csv_path,
            output_prefix,
        }
    }
    
    /// Append a benchmark result to the CSV
    fn record_result(&self, name: &str, rtensors_time_us: u128, pytorch_time_us: u128, speedup: f64) {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.csv_path)
            .expect("Failed to open CSV file");
        
        writeln!(file, "{},{},{},{:.4}", name, rtensors_time_us, pytorch_time_us, speedup)
            .expect("Failed to write to CSV");
    }
    
    /// Generate visualizations from the benchmark results
    fn generate_visualizations(&self) -> PyResult<()> {
        generate_visualizations(&self.csv_path, &self.output_prefix)
    }
}

macro_rules! timed {
    ($block:block) => {{
        let start = std::time::Instant::now();
        let result = { $block };
        let elapsed = start.elapsed();
        (result, elapsed)
    }};
}

macro_rules! timed_python_torch {
    (|$py:ident, $torch:ident|, $timed_block:block) => {{
        Python::attach(|$py| -> PyResult<((), std::time::Duration)> {
            // Run imports outside of timing
            let $torch = PyModule::import($py, "torch")?;
            // Start timing after imports
            let start = std::time::Instant::now();
            let result = { $timed_block };
            let elapsed = start.elapsed();
            Ok((result, elapsed))
        })
    }};
}

macro_rules! run_compare {
    ($session:expr, $name:expr, $rtensors_block:block, |$py:ident, $torch:ident|, $pytorch_block:block) => {{
        // Run rtensors benchmark
        let (_, rtensors_time) = timed!($rtensors_block);
        
        // Run PyTorch benchmark
        let (_, pytorch_time) = timed_python_torch!(|$py, $torch|, $pytorch_block)?;
        
        // Calculate speedup
        let speedup = pytorch_time.as_secs_f64() / rtensors_time.as_secs_f64();
        
        // Print results
        println!("{}: rtensors={:?}, PyTorch={:?}, speedup={:.2}x", 
                 $name, rtensors_time, pytorch_time, speedup);
        
        // Record to session
        $session.record_result(
            $name,
            rtensors_time.as_micros(),
            pytorch_time.as_micros(),
            speedup
        );
        
        Ok::<(), PyErr>(())
    }};
}

fn generate_visualizations(csv_path: &str, output_prefix: &str) -> PyResult<()> {
    Python::attach(|py| -> PyResult<()> {
        let plt = PyModule::import(py, "matplotlib.pyplot")?;
        let pd = PyModule::import(py, "pandas")?;
        
        // Read the CSV file
        let df = pd.call_method1("read_csv", (csv_path,))?;
        
        // Extract data
        let benchmark_names = df.call_method1("__getitem__", ("benchmark_name",))?;
        let rtensors_times = df.call_method1("__getitem__", ("rtensors_time_us",))?;
        let pytorch_times = df.call_method1("__getitem__", ("pytorch_time_us",))?;
        let speedups = df.call_method1("__getitem__", ("speedup",))?;
        
        // Create figure with subplots
        let fig_axes = plt.call_method1("subplots", (2, 2))?;
        let fig = fig_axes.get_item(0)?;
        let axes = fig_axes.get_item(1)?;
        
        fig.call_method1("set_size_inches", (12, 10))?;
        
        // Plot 1: Comparison bar chart (times in microseconds)
        let ax1 = axes.get_item(0)?.get_item(0)?;
        let numpy = PyModule::import(py, "numpy")?;
        let x = numpy.call_method1("arange", (benchmark_names.len()?,))?;
        let width = 0.35;
        
        let kwargs1 = PyDict::new(py);
        kwargs1.set_item("width", width)?;
        kwargs1.set_item("label", "rtensors")?;
        kwargs1.set_item("color", "blue")?;
        ax1.call_method("bar", (x.clone(), rtensors_times.clone()), Some(&kwargs1))?;
        
        let x_offset = x.call_method1("__add__", (width,))?;
        let kwargs2 = PyDict::new(py);
        kwargs2.set_item("width", width)?;
        kwargs2.set_item("label", "PyTorch")?;
        kwargs2.set_item("color", "orange")?;
        ax1.call_method("bar", (x_offset, pytorch_times.clone()), Some(&kwargs2))?;
        
        ax1.call_method1("set_xlabel", ("Benchmark",))?;
        ax1.call_method1("set_ylabel", ("Time (microseconds)",))?;
        ax1.call_method1("set_title", ("Execution Time Comparison",))?;
        ax1.call_method1("set_xticks", (x.call_method1("__add__", (width / 2.0,))?,))?;
        ax1.call_method1("set_xticklabels", (benchmark_names.clone(),))?;
        ax1.call_method0("legend")?;
        let grid_kwargs = PyDict::new(py);
        grid_kwargs.set_item("alpha", 0.3)?;
        ax1.call_method("grid", (), Some(&grid_kwargs))?;
        
        // Plot 2: Speedup bar chart
        let ax2 = axes.get_item(0)?.get_item(1)?;
        let kwargs3 = PyDict::new(py);
        kwargs3.set_item("color", "green")?;
        kwargs3.set_item("alpha", 0.7)?;
        ax2.call_method("bar", (benchmark_names.clone(), speedups.clone()), Some(&kwargs3))?;
        ax2.call_method1("set_xlabel", ("Benchmark",))?;
        ax2.call_method1("set_ylabel", ("Speedup (x)",))?;
        ax2.call_method1("set_title", ("rtensors Speedup over PyTorch",))?;
        let grid_kwargs2 = PyDict::new(py);
        grid_kwargs2.set_item("alpha", 0.3)?;
        ax2.call_method("grid", (), Some(&grid_kwargs2))?;
        
        // Plot 3: Log scale comparison
        let ax3 = axes.get_item(1)?.get_item(0)?;
        let kwargs4 = PyDict::new(py);
        kwargs4.set_item("marker", "o")?;
        kwargs4.set_item("label", "rtensors")?;
        kwargs4.set_item("linewidth", 2)?;
        ax3.call_method("plot", (benchmark_names.clone(), rtensors_times), Some(&kwargs4))?;
        
        let kwargs5 = PyDict::new(py);
        kwargs5.set_item("marker", "s")?;
        kwargs5.set_item("label", "PyTorch")?;
        kwargs5.set_item("linewidth", 2)?;
        ax3.call_method("plot", (benchmark_names, pytorch_times), Some(&kwargs5))?;
        
        ax3.call_method1("set_yscale", ("log",))?;
        ax3.call_method1("set_xlabel", ("Benchmark",))?;
        ax3.call_method1("set_ylabel", ("Time (microseconds, log scale)",))?;
        ax3.call_method1("set_title", ("Execution Time (Log Scale)",))?;
        ax3.call_method0("legend")?;
        let grid_kwargs3 = PyDict::new(py);
        grid_kwargs3.set_item("alpha", 0.3)?;
        grid_kwargs3.set_item("which", "both")?;
        ax3.call_method("grid", (), Some(&grid_kwargs3))?;
        
        // Plot 4: Performance summary table
        let ax4 = axes.get_item(1)?.get_item(1)?;
        ax4.call_method1("axis", ("off",))?;
        
        // Calculate statistics
        let mean_speedup = speedups.call_method0("mean")?;
        let max_speedup = speedups.call_method0("max")?;
        let min_speedup = speedups.call_method0("min")?;
        
        let summary_text = format!(
            "Performance Summary\n\n\
             Mean Speedup: {:.2}x\n\
             Max Speedup: {:.2}x\n\
             Min Speedup: {:.2}x\n",
            mean_speedup.extract::<f64>()?,
            max_speedup.extract::<f64>()?,
            min_speedup.extract::<f64>()?
        );
        
        let bbox_dict = PyDict::new(py);
        bbox_dict.set_item("boxstyle", "round")?;
        bbox_dict.set_item("facecolor", "wheat")?;
        bbox_dict.set_item("alpha", 0.5)?;
        
        let text_kwargs = PyDict::new(py);
        text_kwargs.set_item("s", summary_text)?;
        text_kwargs.set_item("ha", "center")?;
        text_kwargs.set_item("va", "center")?;
        text_kwargs.set_item("fontsize", 14)?;
        text_kwargs.set_item("bbox", bbox_dict)?;
        ax4.call_method("text", (0.5, 0.5), Some(&text_kwargs))?;
        
        // Adjust layout and save
        plt.call_method0("tight_layout")?;
        
        let png_path = format!("{}.png", output_prefix);
        
        plt.call_method1("savefig", (png_path.as_str(),))?;
        let save_kwargs = PyDict::new(py);
        save_kwargs.set_item("dpi", 300)?;

        println!("Visualizations saved to {}", png_path);
        
        Ok(())
    })
}

fn cpu_matmul_benchmark() -> Result<(), pyo3::PyErr> {
    let session = BenchmarkSession::new("matmul_comparison");
    
    println!("Running benchmarks...\n");
    
    // Benchmark 1: MatMul 100x100
    run_compare!(
        &session,
        "MatMul_100x100",
        {
            let a = Tensor::<f32>::ones((100, 100));
            let b = Tensor::<f32>::ones((100, 100));
            a.matmul(&b).unwrap()
        },
        |py, torch|,
        {
            let a = torch.call_method1("ones", ((100, 100),))?;
            let b = torch.call_method1("ones", ((100, 100),))?;
            let _result = torch.call_method1("matmul", (a, b))?;
        }
    )?;
    
    // Benchmark 2: MatMul 500x500
    run_compare!(
        &session,
        "MatMul_500x500",
        {
            let a = Tensor::<f32>::ones((500, 500));
            let b = Tensor::<f32>::ones((500, 500));
            a.matmul(&b).unwrap()
        },
        |py, torch|,
        {
            let a = torch.call_method1("ones", ((500, 500),))?;
            let b = torch.call_method1("ones", ((500, 500),))?;
            let _result = torch.call_method1("matmul", (a, b))?;
        }
    )?;
    
    // Benchmark 3: MatMul 1000x1000
    run_compare!(
        &session,
        "MatMul_1000x1000",
        {
            let a = Tensor::<f32>::ones((1000, 1000));
            let b = Tensor::<f32>::ones((1000, 1000));
            a.matmul(&b).unwrap()
        },
        |py, torch|,
        {
            let a = torch.call_method1("ones", ((1000, 1000),))?;
            let b = torch.call_method1("ones", ((1000, 1000),))?;
            let _result = torch.call_method1("matmul", (a, b))?;
        }
    )?;
    
    println!("\nAll benchmarks completed! Results saved to {}", session.csv_path);
    println!("Generating visualizations...");
    
    // Generate visualizations from the session
    session.generate_visualizations()?;

    Ok(())
}

fn cuda_matmul_benchmark() -> Result<(), pyo3::PyErr> {
    let session = BenchmarkSession::new("matmul_comparison_cuda");
    
    println!("Running benchmarks...\n");
    
    // Benchmark 1: MatMul 100x100
    run_compare!(
        &session,
        "MatMul_100x100",
        {
            let a = CudaTensor::<f32>::ones((100, 100));
            let b = CudaTensor::<f32>::ones((100, 100));
            a.matmul(&b).unwrap()
        },
        |py, torch|,
        {
            let kwargs = PyDict::new(py);
            kwargs.set_item("device", "cuda")?;
            let a = torch.call_method("ones", ((100, 100),), Some(&kwargs))?;
            let b = torch.call_method("ones", ((100, 100),), Some(&kwargs))?;
            let _result = torch.call_method1("matmul", (a, b))?;
        }
    )?;
    
    // Benchmark 2: MatMul 500x500
    run_compare!(
        &session,
        "MatMul_500x500",
        {
            let a = CudaTensor::<f32>::ones((500, 500));
            let b = CudaTensor::<f32>::ones((500, 500));
            a.matmul(&b).unwrap()
        },
        |py, torch|,
        {
            let kwargs = PyDict::new(py);
            kwargs.set_item("device", "cuda")?;
            let a = torch.call_method("ones", ((500, 500),), Some(&kwargs))?;
            let b = torch.call_method("ones", ((500, 500),), Some(&kwargs))?;
            let _result = torch.call_method1("matmul", (a, b))?;
        }
    )?;
    
    // Benchmark 3: MatMul 1000x1000
    run_compare!(
        &session,
        "MatMul_1000x1000",
        {
            let a = CudaTensor::<f32>::ones((1000, 1000));
            let b = CudaTensor::<f32>::ones((1000, 1000));
            a.matmul(&b).unwrap()
        },
        |py, torch|,
        {
            let kwargs = PyDict::new(py);
            kwargs.set_item("device", "cuda")?;
            let a = torch.call_method("ones", ((1000, 1000),), Some(&kwargs))?;
            let b = torch.call_method("ones", ((1000, 1000),), Some(&kwargs))?;
            let _result = torch.call_method1("matmul", (a, b))?;
        }
    )?;
    
    println!("\nAll benchmarks completed! Results saved to {}", session.csv_path);
    println!("Generating visualizations...");
    
    // Generate visualizations from the session
    session.generate_visualizations()?;

    Ok(())
}

fn main() -> PyResult<()> {
    // Start a new benchmark session with a unique name
    // Each session creates its own CSV file and visualization outputs
    // Example: To run multiple benchmark sessions, create different BenchmarkSession instances:
    //   let session1 = BenchmarkSession::new("matmul_comparison");
    //   let session2 = BenchmarkSession::new("elementwise_ops");
    // This will generate separate files for each session
    
    cpu_matmul_benchmark()?;
    cuda_matmul_benchmark()?;
    
    Ok(())
}
