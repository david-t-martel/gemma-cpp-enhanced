use std::env;

fn main() {
    // Re-run build script if these environment variables change
    println!("cargo:rerun-if-env-changed=GEMMA_CPP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GEMMA_CPP_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    // Link to Python if needed
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        // Dynamically detect Python version from pyo3
        let python_version = get_python_version();
        println!("cargo:rustc-link-lib=python{}", python_version);

        // Set library search path for UV Python installations
        if let Some(python_lib_path) = get_uv_python_lib_path() {
            println!("cargo:rustc-link-search=native={}", python_lib_path);
        }
    }

    // Configure gemma.cpp linking when the feature is enabled
    #[cfg(feature = "gemma-cpp")]
    configure_gemma_cpp_linking();

    // Enable SIMD features based on target architecture (compile-time detection)
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let _target_feature = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    if target_arch == "x86_64" {
        println!("cargo:rustc-cfg=feature=\"simd_x86\"");
        // Most modern x86_64 processors have SSE4.2
        println!("cargo:rustc-cfg=feature=\"simd_sse42\"");
        // Enable AVX2 for release builds
        if env::var("PROFILE").unwrap() == "release" {
            println!("cargo:rustc-cfg=feature=\"simd_avx2\"");
        }
    } else if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=feature=\"simd_aarch64\"");
        // NEON is standard on aarch64
        println!("cargo:rustc-cfg=feature=\"simd_neon\"");
    }

    // Note: Removed aggressive target-cpu=native optimization as it can cause memory issues

    // Generate build timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", timestamp);

    // Set target information
    println!("cargo:rustc-env=TARGET={}", env::var("TARGET").unwrap());

    // Generate version info
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let git_hash = get_git_hash().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    println!(
        "cargo:rustc-env=VERSION_WITH_HASH={}-{}",
        version,
        &git_hash[..8.min(git_hash.len())]
    );
}

fn get_git_hash() -> Option<String> {
    use std::process::Command;

    let output = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;

    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .to_string()
            .into()
    } else {
        None
    }
}

fn get_python_version() -> String {
    use std::process::Command;

    // Try to get Python version from pyo3-build-config first
    if let Ok(version) = env::var("PYO3_PYTHON_VERSION") {
        return version.replace(".", "");
    }

    // Try PYO3_PYTHON environment variable first (UV Python)
    let python_cmd = if let Ok(pyo3_python) = env::var("PYO3_PYTHON") {
        pyo3_python
    } else {
        env::var("PYTHON").unwrap_or_else(|_| "python".to_string())
    };

    if let Ok(output) = Command::new(&python_cmd)
        .args(&[
            "-c",
            "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')",
        ])
        .output()
    {
        if output.status.success() {
            if let Ok(version_str) = String::from_utf8(output.stdout) {
                return version_str.trim().to_string();
            }
        }
    }

    // Default fallback for Python 3.13
    "313".to_string()
}

fn get_uv_python_lib_path() -> Option<String> {
    use std::path::PathBuf;

    // Check environment variable first
    if let Ok(path) = env::var("PYTHON_LIB_PATH") {
        return Some(path);
    }

    // Try to derive from PYO3_PYTHON path if set
    if let Ok(pyo3_python) = env::var("PYO3_PYTHON") {
        let python_path = PathBuf::from(&pyo3_python);
        if let Some(parent) = python_path.parent() {
            let libs_path = parent.join("libs");
            if libs_path.exists() {
                return Some(libs_path.to_string_lossy().to_string());
            }
        }
    }

    // Check for UV Python installation paths
    let home_dir = env::var("USERPROFILE").or_else(|_| env::var("HOME")).ok()?;

    // Common UV Python installation paths
    let uv_paths = [
        format!("{}\\AppData\\Roaming\\uv\\python", home_dir),
        format!("{}/.local/share/uv/python", home_dir),
        "/opt/uv/python".to_string(),
    ];

    for base_path in &uv_paths {
        let path = PathBuf::from(base_path);
        if path.exists() {
            // Find the specific Python version directory
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let entry_path = entry.path();
                        if entry_path.is_dir() {
                            let libs_path = entry_path.join("libs");
                            if libs_path.exists() {
                                return Some(libs_path.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

#[cfg(feature = "gemma-cpp")]
fn configure_gemma_cpp_linking() {
    use std::path::PathBuf;

    // Check for gemma.cpp library directory
    let lib_dir = env::var("GEMMA_CPP_LIB_DIR").unwrap_or_else(|_| {
        // Default paths to check
        let candidates = [
            "gemma.cpp/build",
            "../gemma.cpp/build",
            "./lib",
            "./gemma.cpp/lib",
        ];

        for candidate in &candidates {
            let path = PathBuf::from(candidate);
            if path.exists() {
                return path.to_string_lossy().to_string();
            }
        }

        // If no library found, print helpful instructions
        eprintln!("Warning: gemma.cpp library not found!");
        eprintln!("To build with gemma.cpp support:");
        eprintln!("1. Clone and build gemma.cpp:");
        eprintln!("   git clone https://github.com/google/gemma.cpp.git");
        eprintln!("   cd gemma.cpp");
        eprintln!("   cmake -B build -DCMAKE_BUILD_TYPE=Release");
        eprintln!("   cmake --build build");
        eprintln!();
        eprintln!("2. Set environment variables:");
        eprintln!("   set GEMMA_CPP_LIB_DIR=C:\\path\\to\\gemma.cpp\\build");
        eprintln!("   set GEMMA_CPP_INCLUDE_DIR=C:\\path\\to\\gemma.cpp\\include");
        eprintln!();
        eprintln!("3. Build with the feature:");
        eprintln!("   cargo build --features gemma-cpp");

        "".to_string()
    });

    let include_dir = env::var("GEMMA_CPP_INCLUDE_DIR").unwrap_or_else(|_| "./include".to_string());

    if !lib_dir.is_empty() {
        println!("cargo:rustc-link-search=native={}", lib_dir);

        // On Windows, link against the DLL or static library
        if cfg!(target_os = "windows") {
            // Try both static and dynamic linking options
            println!("cargo:rustc-link-lib=dylib=gemma");
            println!("cargo:rustc-link-lib=static=gemma");
        } else {
            // On Unix systems, prefer shared library
            println!("cargo:rustc-link-lib=dylib=gemma");
        }

        println!("cargo:include={}", include_dir);
        eprintln!("Configured gemma.cpp linking:");
        eprintln!("  Library directory: {}", lib_dir);
        eprintln!("  Include directory: {}", include_dir);
    } else {
        eprintln!("Skipping gemma.cpp linking - library not found");
        eprintln!("Build will fail if gemma-cpp feature is enabled without the library");
    }

    // Additional Windows-specific linking requirements
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=kernel32");
        println!("cargo:rustc-link-lib=user32");
        println!("cargo:rustc-link-lib=gdi32");
        println!("cargo:rustc-link-lib=winspool");
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=ole32");
        println!("cargo:rustc-link-lib=oleaut32");
        println!("cargo:rustc-link-lib=uuid");
        println!("cargo:rustc-link-lib=comdlg32");
        println!("cargo:rustc-link-lib=advapi32");
    }
}
