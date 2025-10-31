# CLAUDE.md - Gemma.cpp Development Guide

This file provides guidance to Claude Code (claude.ai/code) when working with the Gemma.cpp project.

## Project Overview

Gemma.cpp is a lightweight C++ inference engine for Google's Gemma foundation models. Currently focused on CPU inference with SIMD optimization, with GPU acceleration planned as future enhancement.

## Build Instructions

### ⚠️ CRITICAL: Use Versioned Build System

**ALWAYS use Just or Make for builds** - Manual CMake commands bypass version embedding!

See `../BUILD_AUTOMATION_SYSTEM.md` for complete documentation.

### Quick Start

**Using Just (Recommended)**:
```bash
just                 # Show all available recipes
just build           # Build with default settings (max 10 jobs)
just build-msvc Release      # MSVC build
just build-oneapi perfpack   # Intel oneAPI with all optimizations
just test            # Run all tests
just deploy          # Deploy to deploy/
just release         # Create versioned release package
```

**Using Make (Traditional)**:
```bash
make help            # Show all targets
make build           # Build project (max 10 jobs)
make build-oneapi    # Intel oneAPI optimized
make test            # Run tests
make package         # Create release package
```

**Direct CMake (⚠️ Not Recommended)**:
```batch
:: Only if Just/Make unavailable - bypasses versioning!
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 10  # NEVER exceed 10 jobs
```

### Version Management

Every build is automatically versioned:
- **Format**: `MAJOR.MINOR.PATCH[+commits][-prerelease].HASH[-dirty]`
- **Example**: `1.2.3+5.a1b2c3d4-dev-release-icx-e4d3b2a1`
- **Embedded**: Version info included in binary via `cmake/Version.cmake`

### Running Inference

**Working Model Paths**:
- **2B Model**: `C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs`
  - Tokenizer: `C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm`
- **4B Model**: `C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs`
  - Tokenizer: `C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\tokenizer.spm`

**Example Commands**:
```batch
:: 2B model (fastest, good for testing)
.\build\Release\gemma.exe ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs

:: 4B model (better quality, SFP format)
.\build\Release\gemma.exe ^
  --tokenizer C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\tokenizer.spm ^
  --weights C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs
```

## Dependency Management

### Current Strategy
- **Highway SIMD**: Fetched from GitHub via FetchContent (most reliable)
- **SentencePiece**: Auto-fetched via FetchContent
- **vcpkg**: Optional, can provide some dependencies but not required

### Why This Approach?
- vcpkg versions sometimes incompatible with the project
- GitHub sources ensure correct versions
- Simpler build process with fewer configuration steps

## Project Structure

```
gemma/
├── build/              # Single build directory (keep it simple!)
├── gemma/              # Core inference engine
│   ├── gemma.h/.cc     # Main API
│   ├── configs.h/.cc   # Model configurations
│   └── kv_cache.h/.cc  # KV cache for context
├── compression/        # Weight compression (SFP, NUQ, etc.)
├── ops/               # SIMD operations via Highway
├── io/                # File I/O and model loading
└── util/              # Utilities and helpers
```

## Common Issues & Solutions

### Build Issues

**CMake can't find compiler**:
```batch
:: Explicitly specify generator and toolset
cmake -B build -G "Visual Studio 17 2022" -T v143
```

**Highway library errors**:
- Let CMake fetch it from GitHub (don't use vcpkg version)
- Check CMakeLists.txt has FetchContent fallback

### Runtime Issues

**Model loading error (3221226356)**:
- Install Visual C++ Redistributables
- Build in Release mode (Debug may have issues)

**Out of memory**:
- Use 2B model instead of 4B
- Reduce context: `--max_seq_len 2048`
- Close other applications

**Model files not found**:
- Use absolute Windows paths
- Check file exists: `dir C:\codedev\llm\.models\`

## Development Tips

### Performance
1. SFP format models run ~2x faster than standard formats
2. First query is slower (model loading), subsequent queries are faster
3. CPU inference benefits from Highway SIMD optimizations

### Testing
```batch
:: Run benchmarks
.\build\Release\single_benchmark.exe ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm

:: Interactive testing
.\build\Release\gemma.exe --weights [model] --tokenizer [tokenizer]
```

## Intel OneAPI Quick Reference

### Build Variants

**Three tested build configurations**:

1. **`build-simple.ps1`** - Fastest iteration (Ninja + sccache):
   ```powershell
   .\build-simple.ps1 -Config Release -Jobs 12 -Clean
   # Compiler: MSVC v143
   # Caching: sccache
   # SIMD: Highway (runtime dispatch)
   # Time: ~12-15 minutes first build, ~2-3 minutes cached
   ```

2. **`build_oneapi.ps1`** - Intel optimized:
   ```powershell
   # Standard (ICX compiler only)
   .\build_oneapi.ps1 -Config std

   # With TBB + IPP
   .\build_oneapi.ps1 -Config tbb-ipp -Clean

   # Full performance pack (TBB + IPP + DNNL + DPL)
   .\build_oneapi.ps1 -Config perfpack
   # Compiler: Intel ICX 2025.1
   # Libraries: MKL (parallel), IPP (parallel), TBB, DNNL
   # SIMD: AVX2/AVX-512 (-xHost -march=native)
   # Time: ~20-30 minutes first build
   ```

3. **`build-oneapi-cuda-fastdebug.ps1`** - Hybrid CPU+GPU:
   ```powershell
   .\scripts\build-oneapi-cuda-fastdebug.ps1 -Jobs 8 -ConfigureOnly
   # Compiler: Intel ICX (CPU) + NVCC (GPU)
   # Caching: ccache
   # Backends: SYCL (Intel) + CUDA (NVIDIA)
   # Time: ~25-35 minutes first build
   ```

### Intel Compiler Flags Explained

From `cmake/IntelToolchain.cmake`:

```cmake
# Base optimization
-O3              # Maximum optimization level
-xHost           # Auto-detect CPU and optimize for it
-march=native    # Use all instructions available on this CPU
-mtune=native    # Tune scheduling for this CPU

# Math optimizations
-ffast-math      # Aggressive math (breaks IEEE 754 strict compliance)
-fno-alias       # Assume pointers don't alias

# Inlining and unrolling
-finline-functions  # Inline aggressively
-funroll-loops      # Unroll loops for speed

# SIMD (auto-detected)
-msse4.2 -mavx2 -mfma              # Always enabled on x86_64
-mavx512f -mavx512cd -mavx512bw    # If CPU supports AVX-512

# Parallelization
-fopenmp         # OpenMP threading
-parallel        # Intel auto-parallelization
-mkl=parallel    # Parallel Math Kernel Library
-ipp=parallel    # Parallel Performance Primitives

# Link-time optimization
-flto            # Whole-program optimization (Release only)
```

**Performance Impact**:
- ICX vs MSVC: **10-25% faster** on matrix operations
- With MKL: **30-50% faster** on linear algebra
- With AVX-512: **40-60% faster** on SIMD-heavy code

### Environment Initialization

**Manual setup** (if build scripts fail):
```powershell
# Initialize oneAPI environment
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"

# Verify environment
$env:ONEAPI_ROOT
$env:MKLROOT
$env:IPPROOT
$env:TBBROOT

# Verify compilers
& "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe" --version
& "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icpx.exe" --version
```

**CMake with Intel toolchain**:
```powershell
cmake -B build -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="C:\codedev\llm\gemma\cmake\IntelToolchain.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGEMMA_USE_ONEAPI_LIBS=ON `
  -DGEMMA_USE_TBB=ON `
  -DGEMMA_USE_IPP=ON

cmake --build build --parallel 12
```

### Troubleshooting Intel Builds

**Issue: ICX compiler not found**
```powershell
# Solution: Check installation path
ls "C:\Program Files (x86)\Intel\oneAPI\compiler"
# If 2025.1 directory exists, update IntelToolchain.cmake:
# set(INTEL_COMPILER_ROOT "${INTEL_ONEAPI_ROOT}/compiler/2025.1")
```

**Issue: setvars.ps1 fails with execution policy**
```powershell
# Solution: Bypass execution policy temporarily
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"
```

**Issue: MKL/IPP libraries not found**
```cmake
# Solution: Verify paths in CMake output
# CMake should show:
#   oneAPI Root: C:/Program Files (x86)/Intel/oneAPI
#   TBB: ✅ ENABLED
#   IPP: ✅ ENABLED
#   DNNL: ✅ ENABLED (or ⚠️ if missing)

# If library missing, install from oneAPI installer:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html
```

**Issue: Build fails with AVX-512 errors**
```cmake
# Solution: Disable AVX-512 requirement
cmake -B build -DGEMMA_REQUIRE_AVX2=OFF
# Or force AVX2-only:
cmake -B build -DGEMMA_FORCE_AVX2=ON
```

### Benchmarking Intel vs MSVC

**Quick comparison**:
```powershell
# Build both variants
.\build-simple.ps1 -Config Release          # MSVC
.\build_oneapi.ps1 -Config perfpack         # Intel

# Run same benchmark on both
$model = "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs"
$tokenizer = "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm"

# MSVC build
Measure-Command {
  .\build-ninja\Release\gemma.exe --weights $model --tokenizer $tokenizer --verbosity 0
}

# Intel build
Measure-Command {
  .\build_perfpack\bin\gemma.exe --weights $model --tokenizer $tokenizer --verbosity 0
}

# Compare tokens/second in output
```

**Expected improvements with Intel oneAPI**:
- Matrix multiplication: **1.3-1.5x faster**
- Attention mechanism: **1.2-1.4x faster**
- Overall inference: **1.15-1.35x faster**
- Memory bandwidth: **5-10% improvement**

## Current Status

### Working ✅
- **Versioned build automation system** with hash-linked releases
- **Just/Make/CMake** build frontends with automatic versioning
- **GitHub Actions CI/CD** pipeline for automated releases
- **Session management** with persistence and interactive commands
- CPU inference with Highway SIMD (runtime dispatch)
- 2B and 4B model support (tested on Windows)
- Windows native compilation (MSVC 2022 v143)
- Intel oneAPI compilation (ICX 2025.1)
- Interactive prompt mode and file-based inference
- Comprehensive test framework (8-phase validation)
- Build acceleration (sccache for MSVC, ccache for Intel)
- Multiple build configurations (std, tbb-ipp, perfpack)

### Session Management Features
- **Command-line arguments**: `--session <id>`, `--load_session`, `--save_on_exit`
- **Interactive commands**:
  - `%q` - Quit session
  - `%c` - Clear/reset session
  - `%s [filename]` - Save session to JSON (default: session_<id>.json)
  - `%l [filename]` - Load session from JSON
  - `%h [N]` - Show last N conversation messages (default: 10)
  - `%i` - Show session statistics (turns, tokens, KV cache size)
  - `%m` - List all managed sessions
- **Features**:
  - Context-aware conversation history with intelligent trimming
  - Automatic KV cache management and reuse
  - JSON serialization for persistence
  - Multi-session management via SessionManager

**Example usage**:
```batch
:: Start new session with auto-save
.\build\Release\gemma.exe ^
  --weights model.sbs --tokenizer tokenizer.spm ^
  --session my_chat --save_on_exit

:: Resume existing session
.\build\Release\gemma.exe ^
  --weights model.sbs --tokenizer tokenizer.spm ^
  --session my_chat --load_session
```

### In Progress 🚧
- SYCL backend for Intel Arc/Xe GPUs
- CUDA backend for NVIDIA GPUs (RTX 40-series tested)
- Hybrid CPU+GPU inference pipelines
- VTune profiling integration

### Planned 📋
- Vulkan backend (cross-platform GPU)
- MCP server integration for AI workflows
- Config file support (TOML format)
- Context window extensions (current: 4K, target: 32K)
- Distributed inference across multiple GPUs
- Model quantization improvements (4-bit NUQ)
- Rust CLI wrapper with enhanced UX

## Key Files to Know

- `CMakeLists.txt` - Main build configuration
- `gemma/gemma.cc` - Core inference implementation
- `gemma/configs.cc` - Model configuration definitions
- `ops/ops.h` - SIMD operations interface
- `io/file_io.cc` - Model loading logic

## Build Requirements

- **Compiler**: Visual Studio 2022 (v143 toolset) or GCC 11+
- **CMake**: 3.14+ (available at `C:\Program Files\CMake\bin\`)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~10GB for model files

## Quick Reference

### Essential Commands

**Build (Use Just/Make, NOT direct CMake)**:
```bash
# Just (recommended)
just build-msvc Release         # MSVC build
just build-oneapi perfpack      # Intel oneAPI with all optimizations
just test                       # Run all tests

# Make (traditional)
make build                      # Default build
make build-oneapi              # Intel oneAPI
make test                      # Run tests
```

**Run Inference**:
```batch
:: 2B model
.\build\Release\gemma.exe ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm

:: With session management
.\build\Release\gemma.exe ^
  --weights [model] --tokenizer [tokenizer] ^
  --session my_chat --save_on_exit
```

**Testing**:
```bash
just test-smoke              # Quick validation
just test-session            # Session persistence tests
just benchmark               # Performance benchmark
```

### Model Paths
- 2B: `C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\`
- 4B: `C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\`

### Key Documentation
- `BUILD_AUTOMATION_SYSTEM.md` - Complete build system guide
- `CLAUDE.md` - Full project documentation
- `deploy/DEPLOYMENT_GUIDE.md` - End-user deployment guide