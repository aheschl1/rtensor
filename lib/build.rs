
#[cfg(feature = "cuda")]
use std::path::PathBuf;

fn main() {
    #[cfg(feature = "cuda")]
    {
        build_cuda_kernels();
    }
    
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn find_nvcc() -> Option<PathBuf> {
    use std::path::PathBuf;
    use std::process::Command;
    
    // Try to find nvcc in PATH first
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
    }
    
    // Common CUDA installation paths
    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/local/cuda-11.0/bin/nvcc",
    ];
    
    for path in &common_paths {
        let nvcc_path = PathBuf::from(path);
        if nvcc_path.exists() {
            return Some(nvcc_path);
        }
    }
    
    None
}

#[cfg(feature = "cuda")]
fn find_cuda_lib_path() -> Option<PathBuf> {
    use std::path::PathBuf;
    
    // Common CUDA library paths
    let common_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/opt/cuda/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-11.0/lib64",
    ];
    
    for path in &common_paths {
        let lib_path = PathBuf::from(path);
        // Check if libcudart.so exists in this path
        if lib_path.join("libcudart.so").exists() {
            return Some(lib_path);
        }
    }
    
    None
}

#[cfg(feature = "cuda")]
fn find_cuda_kernel_files() -> Vec<PathBuf> {
    use std::path::PathBuf;
    use std::fs;
    
    let kernels_dir = PathBuf::from("src/cuda/kernels");
    let mut cu_files = Vec::new();
    
    // Recursively find all .cu files, excluding legacy folder
    fn find_cu_files_recursive(dir: &PathBuf, cu_files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                // Skip legacy directory
                if path.is_dir() {
                    if path.file_name().and_then(|s| s.to_str()) != Some("legacy") {
                        find_cu_files_recursive(&path, cu_files);
                    }
                } else if path.extension().and_then(|s| s.to_str()) == Some("cu") {
                    cu_files.push(path);
                }
            }
        }
    }
    
    find_cu_files_recursive(&kernels_dir, &mut cu_files);
    cu_files.sort();
    cu_files
}

#[cfg(feature = "cuda")]
fn build_cuda_kernels() {
    use std::process::Command;
    use std::env;
    
    // Find nvcc
    let nvcc = find_nvcc().expect(
        "Could not find nvcc. Please install CUDA toolkit or add nvcc to PATH."
    );
    
    println!("Found nvcc at: {}", nvcc.display());
    
    let cuda_lib_path = find_cuda_lib_path().expect(
        "Could not find CUDA libraries. Please install CUDA toolkit."
    );
    
    println!("Found CUDA libraries at: {}", cuda_lib_path.display());

    let kernel_files = find_cuda_kernel_files();
    
    if kernel_files.is_empty() {
        panic!("No CUDA kernel files (.cu) found in src/cuda/kernels/");
    }
    
    println!("Found {} CUDA kernel file(s)", kernel_files.len());
    
    println!("cargo:rerun-if-changed=src/cuda/include/kernels.h");
    println!("cargo:rerun-if-changed=src/cuda/include/common.h");
    println!("cargo:rerun-if-changed=src/cuda/include/unary.h");
    println!("cargo:rerun-if-changed=src/cuda/include/binary.h");
    // println!("cargo:rerun-if-changed=src/cuda/include/elementwise.h");
    for kernel_file in &kernel_files {
        println!("cargo:rerun-if-changed={}", kernel_file.display());
    }
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Specify the desired architecture version.
    let arch = "compute_86"; 
    let code = "sm_86";

    // Build add.cu to PTX (for runtime JIT compilation in tests)
    let add_cu = PathBuf::from("src/cuda/kernels/legacy/add.cu");
    if add_cu.exists() {
        let ptx_file = out_dir.join("add.ptx");

        let nvcc_status = Command::new(&nvcc)
            .arg("-ptx")
            .arg("-o")
            .arg(&ptx_file)
            .arg(&add_cu)
            .arg(format!("-arch={}", arch))
            .arg(format!("-code={}", code))
            .arg("-I")
            .arg("src/cuda/include")
            .status()
            .unwrap();

        assert!(
            nvcc_status.success(),
            "Failed to compile add.cu to PTX."
        );
    }

    // Build all other .cu files to object files (for static linking)
    // Note: add.cu in legacy/ is excluded as it's only used for PTX generation
    let mut object_files = Vec::new();
    
    for kernel_file in &kernel_files {
        
        let file_stem = kernel_file.file_stem()
            .and_then(|s| s.to_str())
            .expect("Invalid kernel filename");
        
        let obj_file = out_dir.join(format!("{}.o", file_stem));
        
        println!("Compiling {} to object file...", kernel_file.display());

        let nvcc_compile_status = Command::new(&nvcc)
            .arg("-c")  // Compile only (create object file)
            .arg("-o")
            .arg(&obj_file)
            .arg(kernel_file)
            .arg(format!("-arch={}", arch))
            .arg("-I")
            .arg("src/cuda/include") 
            .arg("--compiler-options")
            .arg("-fPIC")
            .status()
            .unwrap();

        assert!(
            nvcc_compile_status.success(),
            "Failed to compile {} to object file.", 
            kernel_file.display()
        );
        
        object_files.push(obj_file);
    }

    if !object_files.is_empty() {
        let kernels_lib = out_dir.join("libcudakernels.a");
        
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("crus").arg(&kernels_lib);
        
        for obj in &object_files {
            ar_cmd.arg(obj);
        }
        
        let ar_status = ar_cmd.status().unwrap();

        assert!(
            ar_status.success(),
            "Failed to create static library from CUDA object files"
        );
        
        println!("Created CUDA kernels library at: {}", kernels_lib.display());
    }

    // Tell cargo where to find our library and link it
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cudakernels");
    
    // Add CUDA library search path and link against CUDA libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    
    // Link against C++ standard library (needed for CUDA C++ code)
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/cuda/include/kernels.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // we use "no_copy" and "no_debug" here because we don't know if we can safely generate them for our structs in C code (they may contain raw pointers)
        .no_copy("*")
        .no_debug("*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // we need to make modifications to the generated code
    let generated_bindings = bindings.to_string();

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    // let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    // let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(out_path.join("bindings.rs"), generated_bindings.as_bytes())
        .expect("Failed to write bindings");
}