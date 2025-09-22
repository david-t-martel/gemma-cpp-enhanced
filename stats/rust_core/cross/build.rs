//! Cross-compilation build script for Gemma Rust components
//!
//! This build script handles platform-specific optimizations, feature detection,
//! and dependency configuration for cross-compilation targets.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");

    let target = env::var("TARGET").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    println!("cargo:rustc-env=BUILD_TARGET={}", target);
    println!("cargo:rustc-env=BUILD_TARGET_ARCH={}", target_arch);
    println!("cargo:rustc-env=BUILD_TARGET_OS={}", target_os);

    configure_target_optimizations(&target, &target_arch, &target_os);
    configure_simd_support(&target_arch);
    configure_math_libraries(&target_os);
    configure_gpu_support(&target_os);

    // Platform-specific configurations
    match target_os.as_str() {
        "windows" => configure_windows(),
        "linux" => configure_linux(),
        "macos" => configure_macos(),
        _ => {}
    }

    // Architecture-specific configurations
    match target_arch.as_str() {
        "x86_64" => configure_x86_64(),
        "aarch64" => configure_aarch64(),
        "wasm32" => configure_wasm32(),
        _ => {}
    }

    println!("Build configuration completed for target: {}", target);
}

/// Configure target-specific compiler optimizations
fn configure_target_optimizations(target: &str, target_arch: &str, target_os: &str) {
    // Enable target CPU optimizations
    match target_arch {
        "x86_64" => {
            // Enable modern x86_64 features
            println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=x86-64-v2");

            // Additional optimization flags for x86_64
            if cfg!(feature = "x86_64-optimizations") {
                println!("cargo:rustc-link-arg=-march=x86-64-v2");
                println!("cargo:rustc-link-arg=-mtune=generic");
            }
        }
        "aarch64" => {
            // ARM64 optimizations
            println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=generic");

            if cfg!(feature = "aarch64-optimizations") {
                println!("cargo:rustc-link-arg=-mcpu=generic");
                println!("cargo:rustc-link-arg=-mtune=generic");
            }
        }
        "wasm32" => {
            if cfg!(feature = "wasm-optimizations") {
                // WASM-specific optimizations
                println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+simd128");
            }
        }
        _ => {}
    }

    // OS-specific linking flags
    match target_os {
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
            println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        }
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-dead_strip");
        }
        "windows" => {
            println!("cargo:rustc-link-arg=/OPT:REF");
            println!("cargo:rustc-link-arg=/OPT:ICF");
        }
        _ => {}
    }
}

/// Configure SIMD instruction set support
fn configure_simd_support(target_arch: &str) {
    match target_arch {
        "x86_64" => {
            // Detect and enable x86_64 SIMD features
            println!("cargo:rustc-cfg=has_simd");
            println!("cargo:rustc-cfg=has_sse2");

            // Check for newer SIMD instruction sets
            if is_feature_available("avx2") {
                println!("cargo:rustc-cfg=has_avx2");
                println!("cargo:rustc-env=SIMD_LEVEL=avx2");
            } else if is_feature_available("avx") {
                println!("cargo:rustc-cfg=has_avx");
                println!("cargo:rustc-env=SIMD_LEVEL=avx");
            } else {
                println!("cargo:rustc-env=SIMD_LEVEL=sse2");
            }
        }
        "aarch64" => {
            // ARM NEON support
            println!("cargo:rustc-cfg=has_simd");
            println!("cargo:rustc-cfg=has_neon");
            println!("cargo:rustc-env=SIMD_LEVEL=neon");
        }
        "wasm32" => {
            // WASM SIMD support
            println!("cargo:rustc-cfg=has_wasm_simd");
            println!("cargo:rustc-env=SIMD_LEVEL=wasm128");
        }
        _ => {
            println!("cargo:rustc-env=SIMD_LEVEL=scalar");
        }
    }
}

/// Configure optimized math libraries
fn configure_math_libraries(target_os: &str) {
    match target_os {
        "linux" => {
            // Try to link with optimized BLAS/LAPACK
            if pkg_config::probe("openblas").is_ok() {
                println!("cargo:rustc-cfg=has_openblas");
                println!("cargo:rustc-link-lib=openblas");
            } else if pkg_config::probe("blas").is_ok() && pkg_config::probe("lapack").is_ok() {
                println!("cargo:rustc-cfg=has_blas_lapack");
                println!("cargo:rustc-link-lib=blas");
                println!("cargo:rustc-link-lib=lapack");
            }
        }
        "macos" => {
            // Use Accelerate framework on macOS
            println!("cargo:rustc-cfg=has_accelerate");
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        "windows" => {
            // Windows may use Intel MKL or other optimized libraries
            if let Ok(_) = env::var("MKL_ROOT") {
                println!("cargo:rustc-cfg=has_mkl");
                configure_mkl();
            }
        }
        _ => {}
    }
}

/// Configure GPU acceleration support
fn configure_gpu_support(target_os: &str) {
    // CUDA support detection
    if let Ok(cuda_root) = env::var("CUDA_ROOT") {
        println!("cargo:rustc-cfg=has_cuda");
        println!("cargo:rustc-env=CUDA_ROOT={}", cuda_root);

        let cuda_lib_path = PathBuf::from(&cuda_root).join("lib64");
        if cuda_lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=curand");
        }
    }

    // Platform-specific GPU support
    match target_os {
        "linux" => {
            // Check for ROCm (AMD GPU) support
            if PathBuf::from("/opt/rocm").exists() {
                println!("cargo:rustc-cfg=has_rocm");
                println!("cargo:rustc-link-search=native=/opt/rocm/lib");
                println!("cargo:rustc-link-lib=hip");
            }
        }
        "macos" => {
            // Metal performance shaders on macOS
            println!("cargo:rustc-cfg=has_metal");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        }
        _ => {}
    }
}

/// Windows-specific configuration
fn configure_windows() {
    println!("cargo:rustc-cfg=target_windows");

    // Link with Windows-specific libraries
    println!("cargo:rustc-link-lib=kernel32");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=advapi32");

    // Configure Windows threading
    println!("cargo:rustc-link-lib=synchronization");
}

/// Linux-specific configuration
fn configure_linux() {
    println!("cargo:rustc-cfg=target_linux");

    // Link with common Linux libraries
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");

    // Enable GNU extensions if available
    if std::process::Command::new("gcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_gnu_extensions");
    }
}

/// macOS-specific configuration
fn configure_macos() {
    println!("cargo:rustc-cfg=target_macos");

    // Link with macOS system frameworks
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=Security");

    // Configure for universal binaries if building for multiple architectures
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "aarch64" {
        println!("cargo:rustc-cfg=target_macos_arm64");
    } else {
        println!("cargo:rustc-cfg=target_macos_x86_64");
    }
}

/// x86_64-specific configuration
fn configure_x86_64() {
    println!("cargo:rustc-cfg=target_x86_64");

    // Enable x86_64-specific optimizations
    println!("cargo:rustc-env=TARGET_FEATURES=sse,sse2,sse3,ssse3,sse4.1,sse4.2");

    // Additional features detection
    if is_feature_available("avx") {
        println!("cargo:rustc-env=TARGET_FEATURES_EXTENDED=avx,avx2,fma");
    }
}

/// AArch64-specific configuration
fn configure_aarch64() {
    println!("cargo:rustc-cfg=target_aarch64");

    // Enable ARM64-specific optimizations
    println!("cargo:rustc-env=TARGET_FEATURES=neon,fp,asimd");

    // Check for additional ARM features
    if is_feature_available("sve") {
        println!("cargo:rustc-cfg=has_sve");
    }
}

/// WASM32-specific configuration
fn configure_wasm32() {
    println!("cargo:rustc-cfg=target_wasm32");

    // Configure WASM-specific features
    println!("cargo:rustc-env=TARGET_FEATURES=simd128");

    // Set memory and stack size for WASM
    println!("cargo:rustc-link-arg=-z");
    println!("cargo:rustc-link-arg=stack-size=1048576");  // 1MB stack
    println!("cargo:rustc-link-arg=--initial-memory=67108864");  // 64MB initial memory
    println!("cargo:rustc-link-arg=--max-memory=2147483648");  // 2GB max memory
}

/// Configure Intel MKL
fn configure_mkl() {
    if let Ok(mkl_root) = env::var("MKL_ROOT") {
        let mkl_lib_path = PathBuf::from(&mkl_root).join("lib").join("intel64");

        if mkl_lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", mkl_lib_path.display());

            // Link MKL libraries in the correct order
            println!("cargo:rustc-link-lib=mkl_intel_lp64");
            println!("cargo:rustc-link-lib=mkl_intel_thread");
            println!("cargo:rustc-link-lib=mkl_core");
            println!("cargo:rustc-link-lib=iomp5");
        }
    }
}

/// Check if a CPU feature is available (simplified detection)
fn is_feature_available(feature: &str) -> bool {
    // This is a simplified implementation
    // In practice, you'd want more sophisticated feature detection
    match feature {
        "avx2" => {
            // Check if we're building for a target that supports AVX2
            env::var("CARGO_CFG_TARGET_FEATURE")
                .map(|features| features.contains("avx2"))
                .unwrap_or(false)
        }
        "avx" => {
            env::var("CARGO_CFG_TARGET_FEATURE")
                .map(|features| features.contains("avx"))
                .unwrap_or(false)
        }
        "sve" => {
            // ARM SVE detection
            false  // Conservative default
        }
        _ => false,
    }
}

/// Print build information for debugging
fn print_build_info() {
    println!("cargo:warning=Build target: {}", env::var("TARGET").unwrap_or_default());
    println!("cargo:warning=Build profile: {}", env::var("PROFILE").unwrap_or_default());
    println!("cargo:warning=Host: {}", env::var("HOST").unwrap_or_default());

    if let Ok(features) = env::var("CARGO_CFG_TARGET_FEATURE") {
        println!("cargo:warning=Target features: {}", features);
    }
}
