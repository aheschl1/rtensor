
use std::path::PathBuf;

fn main() {
    #[cfg(feature = "cuda")]
    {
        build_cuda_kernels();
    }
    setup_openblas();
    println!("cargo:rerun-if-changed=build.rs");
}

fn setup_openblas() {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let openblas_install_dir = out_dir.join("openblas-install");
    let openblas_lib_dir = openblas_install_dir.join("lib");
    let openblas_include_dir = openblas_install_dir.join("include");
    
    // Check if OpenBLAS is already built and installed
    let openblas_lib = openblas_lib_dir.join("libopenblas.a");
    let openblas_header = openblas_include_dir.join("cblas.h");
    
    if openblas_lib.exists() && openblas_header.exists() {
        println!("cargo:warning=OpenBLAS already built at: {}", openblas_install_dir.display());
    } else {
        println!("cargo:warning=Building OpenBLAS from source...");
        
        let download_url = "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.zip";
        let zip_path = out_dir.join("openblas-v0.3.30.zip");
        let openblas_source_dir = out_dir.join("OpenBLAS-0.3.30");
        
        // Download the source archive if not already present
        if !zip_path.exists() {
            println!("Downloading OpenBLAS source from {}...", download_url);
            let curl_status = Command::new("curl")
                .arg("-L")
                .arg("-o")
                .arg(&zip_path)
                .arg(download_url)
                .status()
                .expect("Failed to run curl. Please install curl.");
            
            assert!(
                curl_status.success(),
                "Failed to download OpenBLAS from {}", 
                download_url
            );
            
            println!("Downloaded OpenBLAS source to: {}", zip_path.display());
        }
        
        // Extract the zip archive if not already extracted
        if !openblas_source_dir.exists() {
            println!("Extracting OpenBLAS source...");
            let unzip_status = Command::new("unzip")
                .arg("-q")
                .arg(&zip_path)
                .arg("-d")
                .arg(&out_dir)
                .status()
                .expect("Failed to run unzip. Please install unzip.");
            
            assert!(
                unzip_status.success(),
                "Failed to extract OpenBLAS zip archive"
            );
            
            println!("Extracted OpenBLAS to: {}", openblas_source_dir.display());
            
            // Run make clean to ensure a fresh build
            println!("Cleaning OpenBLAS build directory...");
            let _ = Command::new("make")
                .current_dir(&openblas_source_dir)
                .arg("clean")
                .output();
        }
        
        // Build OpenBLAS
        println!("Compiling OpenBLAS (this may take a few minutes)...");
        
        // Get number of processors for parallel build
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "4".to_string());
        
        // Let OpenBLAS auto-detect the target, but provide a fallback
        // If auto-detection fails, use GENERIC target for maximum compatibility
        let make_output = Command::new("make")
            .current_dir(&openblas_source_dir)
            .arg("NO_SHARED=1")
            .arg("NO_LAPACK=1")
            .arg("USE_OPENMP=0")
            .arg("libs")  // Only build libraries, skip tests
            .arg("netlib")  // Build reference BLAS
            .arg(format!("-j{}", nproc))
            .output()
            .expect("Failed to run make. Please install make and a Fortran compiler (gfortran).");
        
        if !make_output.status.success() {
            eprintln!("cargo:warning=First build attempt failed, retrying with TARGET=GENERIC...");
            let make_output2 = Command::new("make")
                .current_dir(&openblas_source_dir)
                .arg("NO_SHARED=1")
                .arg("NO_LAPACK=1")
                .arg("USE_OPENMP=0")
                .arg("TARGET=GENERIC")
                .arg("libs")
                .arg("netlib")
                .arg(format!("-j{}", nproc))
                .output()
                .expect("Failed to run make. Please install make and a Fortran compiler (gfortran).");
            
            if !make_output2.status.success() {
                eprintln!("Make stdout: {}", String::from_utf8_lossy(&make_output2.stdout));
                eprintln!("Make stderr: {}", String::from_utf8_lossy(&make_output2.stderr));
                panic!("Failed to compile OpenBLAS even with TARGET=GENERIC. Make sure you have gfortran installed (apt install gfortran or yum install gcc-gfortran).");
            }
        }
        
        println!("OpenBLAS compiled successfully");
        
        // Install OpenBLAS to our custom directory
        println!("Installing OpenBLAS to: {}", openblas_install_dir.display());
        
        // Since we built with NO_SHARED=1, manually copy the static library and headers
        // instead of using `make install` which expects both static and shared libs
        std::fs::create_dir_all(&openblas_lib_dir).expect("Failed to create lib directory");
        std::fs::create_dir_all(&openblas_include_dir).expect("Failed to create include directory");
        
        // Find and copy the static library (lib*.a)
        let lib_files: Vec<_> = std::fs::read_dir(&openblas_source_dir)
            .expect("Failed to read OpenBLAS source directory")
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("lib") && name_str.ends_with(".a")
            })
            .collect();
        
        assert!(!lib_files.is_empty(), "No static library (lib*.a) found in OpenBLAS build directory");
        
        let first_lib_name = lib_files.first().map(|e| e.file_name());
        
        for lib_file in lib_files {
            let dest = openblas_lib_dir.join(lib_file.file_name());
            std::fs::copy(lib_file.path(), &dest)
                .unwrap_or_else(|_| panic!("Failed to copy {}", lib_file.file_name().to_string_lossy()));
            println!("Copied {} to {}", lib_file.file_name().to_string_lossy(), dest.display());
        }
        
        // Copy all header files from the source directory
        for entry in std::fs::read_dir(&openblas_source_dir).expect("Failed to read OpenBLAS source directory").flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("h") {
                let dest = openblas_include_dir.join(entry.file_name());
                std::fs::copy(&path, &dest)
                    .unwrap_or_else(|_| panic!("Failed to copy {}", entry.file_name().to_string_lossy()));
            }
        }
        println!("Copied all header files to include directory");
        
        // Create a symlink from libopenblas.a to whatever the actual library name is
        // This ensures consistent linking
        if !openblas_lib_dir.join("libopenblas.a").exists() {
            if let Some(lib_name) = first_lib_name {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::symlink;
                    symlink(
                        &lib_name,
                        openblas_lib_dir.join("libopenblas.a")
                    ).ok(); // Ignore errors if symlink already exists
                }
                #[cfg(not(unix))]
                {
                    // On non-Unix, just copy the file
                    std::fs::copy(
                        openblas_lib_dir.join(&lib_name),
                        openblas_lib_dir.join("libopenblas.a")
                    ).ok();
                }
            }
        }
        
        println!("OpenBLAS installed successfully");
        
        // Clean up the zip file to save space (optional)
        fs::remove_file(&zip_path).ok();
    }
    
    // Tell cargo where to find the OpenBLAS library
    println!("cargo:rustc-link-search=native={}", openblas_lib_dir.display());
    println!("cargo:rustc-link-lib=static=openblas");
    
    // Also need to link against libgfortran for OpenBLAS
    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    
    // Tell cargo to rerun if OpenBLAS directory changes
    println!("cargo:rerun-if-changed={}", openblas_install_dir.display());
    
    // Generate bindings for OpenBLAS headers
    generate_openblas_bindings(&openblas_include_dir, &out_dir);
}

fn generate_openblas_bindings(include_dir: &std::path::PathBuf, out_dir: &std::path::PathBuf) {
    let cblas_header = include_dir.join("cblas.h");
    
    println!("Generating OpenBLAS bindings from: {}", cblas_header.display());
    
    // Generate bindings for cblas.h (which is the main C interface)
    let bindings = bindgen::Builder::default()
        .header(cblas_header.to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        // Parse cblas.h and include related headers
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allowlist the functions and types we want
        .allowlist_function("openblas_.*")
        .allowlist_function("goto_.*")
        .allowlist_function("cblas_.*")
        .allowlist_type("CBLAS_.*")
        .allowlist_type("blasint")
        .allowlist_var("OPENBLAS_.*")
        // Generate types as simple as possible
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_partialeq(true)
        // Use core instead of std for better portability
        .use_core()
        // Finish the builder and generate the bindings
        .generate()
        .expect("Unable to generate OpenBLAS bindings");

    // Write the bindings to the $OUT_DIR/openblas_bindings.rs file
    let out_path = PathBuf::from(out_dir);
    bindings
        .write_to_file(out_path.join("openblas_bindings.rs"))
        .expect("Couldn't write OpenBLAS bindings!");
    
    println!("Generated OpenBLAS bindings at: {}", out_path.join("openblas_bindings.rs").display());
}

#[cfg(feature = "cuda")]
fn find_nvcc() -> Option<PathBuf> {
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