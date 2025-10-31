# Gemma C++ Build Guide: oneAPI + CUDA Hybrid Configuration

**Last Updated:** 2025-10-13  
**Configuration:** Intel oneAPI 2025.1.1 + NVIDIA CUDA 13.0  
**Build Type:** FastDebug (O1 optimization + debug symbols)

## Overview

This guide provides instructions for building Gemma C++ with **hybrid acceleration**:
- **SYCL (Intel oneAPI)** for CPU-only inference
- **CUDA** for GPU acceleration on NVIDIA hardware

The configuration uses:
- Intel ICX compiler for C/C++ code
- NVIDIA nvcc for CUDA kernels  
- vcpkg for dependency management (no FetchContent network issues)
- ccache for build caching and acceleration
- Ninja for fast parallel builds

---

## Prerequisites Verified

Your system has:
- ✅ Intel oneAPI 2025.1.1 at `C:\Program Files (x86)\Intel\oneAPI`
- ✅ NVIDIA CUDA 13.0 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- ✅ NVIDIA GeForce RTX 5060 Ti (Driver 576.88)
- ✅ vcpkg at `C:\codedev\vcpkg`
- ✅ Visual Studio 2022 Build Tools (for nvcc host compilation)
- ✅ Ninja build system
- ✅ ccache for compilation caching
- ✅ uv for Python tooling

---

## Quick Start

### Option 1: Automated Build Script (Recommended)

```powershell
# Navigate to project
cd C:\codedev\llm\gemma

# Run the build script
.\scripts\build-oneapi-cuda-fastdebug.ps1

# Or with options:
.\scripts\build-oneapi-cuda-fastdebug.ps1 -CleanBuild  # Clean build from scratch
.\scripts\build-oneapi-cuda-fastdebug.ps1 -ConfigureOnly  # Just configure, don't build
.\scripts\build-oneapi-cuda-fastdebug.ps1 -SkipVcpkgInstall  # Skip vcpkg if already installed
.\scripts\build-oneapi-cuda-fastdebug.ps1 -Jobs 8  # Override parallel jobs count
```

### Option 2: Manual Step-by-Step

```powershell
# 1. Set environment
$env:VCPKG_ROOT = "C:\codedev\vcpkg"
$env:VCPKG_FEATURE_FLAGS = "manifests,versions,registries,binarycaching"
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

# 2. Install vcpkg dependencies (first time only, ~20-30 minutes)
C:\codedev\vcpkg\vcpkg.exe install highway sentencepiece nlohmann-json benchmark --triplet x64-windows

# 3. Configure
cmake -S . -B build/oneapi-cuda-fastdebug `
  -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="C:/codedev/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=FastDebug `
  -DCMAKE_C_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" `
  -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" `
  -DGEMMA_ENABLE_SYCL=ON `
  -DGEMMA_ENABLE_CUDA=ON `
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON `
  -DGEMMA_PREFER_SYSTEM_DEPS=ON

# 4. Build (adjust -j value based on your system)
cmake --build build/oneapi-cuda-fastdebug -- -j 8 -k 0
```

---

## Build Configuration Details

### FastDebug Build Type

The FastDebug configuration provides:
- **-O1 optimization**: Fast compilation, reasonable performance
- **Debug symbols**: Full debugging capability
- **AVX2 + FMA**: SIMD optimizations enabled
- **Frame pointers**: Better profiling and debugging

Compiler flags:
```cmake
CMAKE_C_FLAGS_FASTDEBUG = "-O1 -g -fno-omit-frame-pointer"
CMAKE_CXX_FLAGS_FASTDEBUG = "-O1 -g -fno-omit-frame-pointer -mavx2 -mfma"
CMAKE_CUDA_FLAGS_FASTDEBUG = "-O1 -g -lineinfo"
```

### Dependency Management (vcpkg)

Dependencies are now managed via `vcpkg.json` manifest:
- **highway**: SIMD library for cross-platform vectorization
- **sentencepiece**: Tokenization library
- **nlohmann-json**: JSON parsing
- **benchmark**: Performance benchmarking (Google Benchmark)

Baseline pinned to: `d5ec528843d29e3a52d745a64b469f810b2cedbf`

This eliminates the network failures you were experiencing with FetchContent.

### Compiler Selection

- **C/C++**: Intel ICX (oneAPI 2025.1.1)
  - Provides excellent SIMD support
  - Compatible with SYCL for CPU execution
  - Clang-based, modern C++20 support

- **CUDA**: NVIDIA nvcc 13.0
  - Latest CUDA toolkit
  - SM 89 architecture for RTX 5060 Ti
  - Uses MSVC as host compiler on Windows

### Build Acceleration

- **ccache**: Caches compiler outputs
  - Location: `C:\Users\david\.cache\ccache`
  - Max size: 20GB
  - Speeds up rebuilds significantly

- **Ninja**: Fast parallel build system
  - Auto-detects optimal job count based on CPU and RAM
  - Resource-aware to prevent system thrashing

---

## Build Outputs

### Logs and Reports

All build artifacts are organized under `build/oneapi-cuda-fastdebug/`:

```
build/oneapi-cuda-fastdebug/
├── logs/
│   ├── configure.log    # CMake configuration output
│   ├── build.log        # Build output
│   └── ccache.log       # ccache activity log
├── reports/
│   └── (build reports will be generated here)
├── bin/
│   └── gemma.exe        # Main executable
└── lib/
    └── (libraries)
```

### Executable Size

Expected executable size: **~1.8-2.5 MB** (FastDebug build)

---

## Running the Build

### CPU-only Execution (SYCL)

```powershell
# Set SYCL to use CPU
$env:SYCL_DEVICE_FILTER = "cpu"

# Run
.\build\oneapi-cuda-fastdebug\bin\gemma.exe --help
```

### GPU Execution (CUDA)

```powershell
# Ensure SYCL_DEVICE_FILTER is not set
Remove-Item Env:\SYCL_DEVICE_FILTER -ErrorAction SilentlyContinue

# Run
.\build\oneapi-cuda-fastdebug\bin\gemma.exe --help
```

### Verify Available Devices

```powershell
# Check SYCL devices
sycl-ls

# Check CUDA devices
nvidia-smi
```

---

## Troubleshooting

### Problem: vcpkg dependencies fail to install

**Solution:**
```powershell
# Update vcpkg
cd C:\codedev\vcpkg
git pull
.\bootstrap-vcpkg.bat

# Retry installation
vcpkg install highway sentencepiece nlohmann-json benchmark --triplet x64-windows
```

### Problem: CMake can't find Intel compiler

**Solution:**
```powershell
# Initialize oneAPI environment
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"

# Verify compiler is accessible
& "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe" --version
```

### Problem: CUDA architecture mismatch

Your RTX 5060 Ti uses SM 89 architecture. If you see warnings about unsupported architecture:

**Solution:**
```powershell
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Update CMake configuration
cmake ... -DCMAKE_CUDA_ARCHITECTURES=89  # or whatever nvidia-smi reports
```

### Problem: Build runs out of memory

**Solution:**
```powershell
# Reduce parallel jobs
cmake --build build/oneapi-cuda-fastdebug -- -j 4  # Use fewer jobs

# Or use the script with explicit jobs
.\scripts\build-oneapi-cuda-fastdebug.ps1 -Jobs 4
```

### Problem: nvcc can't find MSVC

**Solution:**
```powershell
# Ensure Visual Studio 2022 Build Tools are in PATH
where cl

# If not found, initialize VS environment
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
```

### Problem: ccache not working

**Solution:**
```powershell
# Check ccache stats
C:\Users\david\.local\bin\ccache.exe -s

# Clear cache if corrupted
C:\Users\david\.local\bin\ccache.exe -C

# Verify environment
Write-Host $env:CCACHE_DIR  # Should be C:\Users\david\.cache\ccache
```

---

## Performance Notes

### First Build
- **Time**: 20-40 minutes
- **Reasons**: 
  - vcpkg compiling dependencies from source
  - No ccache hits
  - Full compilation of Gemma codebase

### Subsequent Builds (after ccache warm-up)
- **Time**: 2-10 minutes
- **Reasons**:
  - vcpkg dependencies cached
  - ccache hits on unchanged files
  - Only modified files recompile

### Resource Usage
- **RAM**: ~8-12GB during peak linking
- **CPU**: Will use 70% of available cores by default
- **Disk**: ~5-10GB for build outputs + ~10GB for ccache

---

## Alternative Build Configurations

If the hybrid oneAPI+CUDA build proves problematic, you have these alternatives:

### 1. MSVC-only Build (Fastest to get working)
```powershell
cmake -S . -B build-msvc-only -G "Visual Studio 17 2022" -DGEMMA_ENABLE_SYCL=OFF -DGEMMA_ENABLE_CUDA=OFF
cmake --build build-msvc-only --config FastDebug
```

### 2. oneAPI-only Build (SYCL for CPU)
```powershell
cmake -S . -B build-oneapi-only -G Ninja `
  -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" `
  -DGEMMA_ENABLE_SYCL=ON -DGEMMA_ENABLE_CUDA=OFF
cmake --build build-oneapi-only
```

### 3. CUDA-only Build (GPU acceleration)
```powershell
cmake -S . -B build-cuda-only -G Ninja `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" `
  -DGEMMA_ENABLE_SYCL=OFF -DGEMMA_ENABLE_CUDA=ON
cmake --build build-cuda-only
```

### 4. MinGW Build (Alternative toolchain)
```powershell
cmake -S . -B build-mingw -G Ninja `
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ `
  -DGEMMA_ENABLE_SYCL=OFF -DGEMMA_ENABLE_CUDA=OFF
cmake --build build-mingw
```

---

## Files Modified

The following canonical files have been updated:

1. `vcpkg.json` - Enabled dependencies, pinned baseline
2. `vcpkg-configuration.json` - Reproducible build configuration
3. `scripts/build-oneapi-cuda-fastdebug.ps1` - Automated build script

---

## Next Steps After Successful Build

1. **Test the executable**:
   ```powershell
   .\build\oneapi-cuda-fastdebug\bin\gemma.exe --version
   .\build\oneapi-cuda-fastdebug\bin\gemma.exe --help
   ```

2. **Run inference** with your models:
   ```powershell
   .\build\oneapi-cuda-fastdebug\bin\gemma.exe `
     --tokenizer "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm" `
     --weights "C:/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs" `
     --prompt "Hello, how are you?"
   ```

3. **Integrate with your Python CLI**:
   ```python
   # In gemma-cli.py
   GEMMA_EXECUTABLE_PATH = r"C:\codedev\llm\gemma\build\oneapi-cuda-fastdebug\bin\gemma.exe"
   ```

4. **Profile and optimize**:
   - Use Intel VTune for CPU profiling (with SYCL)
   - Use NVIDIA Nsight for GPU profiling (with CUDA)

---

## Support and Further Help

If you encounter issues not covered here:

1. **Check the logs**:
   - `build/oneapi-cuda-fastdebug/logs/configure.log`
   - `build/oneapi-cuda-fastdebug/logs/build.log`

2. **Search for specific errors** in the build log:
   ```powershell
   Select-String -Path "build/oneapi-cuda-fastdebug/logs/build.log" -Pattern "error:" | Select-Object -First 20
   ```

3. **Verify toolchain versions**:
   ```powershell
   icx --version
   nvcc --version
   cmake --version
   ninja --version
   ```

4. **Check environment**:
   ```powershell
   Write-Host "VCPKG_ROOT: $env:VCPKG_ROOT"
   Write-Host "CUDA_PATH: $env:CUDA_PATH"
   Write-Host "CCACHE_DIR: $env:CCACHE_DIR"
   ```

---

## References

- **Intel oneAPI**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-toolkit
- **vcpkg**: https://vcpkg.io/
- **Gemma.cpp**: Original repository documentation
- **ccache**: https://ccache.dev/

---

**Good luck with your build!** 🚀
