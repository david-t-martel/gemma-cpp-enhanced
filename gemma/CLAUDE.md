# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gemma.cpp is a lightweight C++ inference engine for Google's Gemma foundation models, designed for research and experimentation. It provides minimalist implementations of Gemma-2, Gemma-3, Griffin/RecurrentGemma, and PaliGemma-2 (vision-language) models, prioritizing simplicity and directness over full generality.

## Build Strategy

The project uses a **versioned build automation system** with hash-linked releases for reproducibility. Three build frontends are available: **Just** (modern, recommended), **Make** (traditional), and **CMake** (direct). All builds are version-tagged with Git-based identifiers.

See `BUILD_AUTOMATION_SYSTEM.md` for complete documentation.

## Versioned Build System

### Quick Start

**Using Just (Recommended)**:
```bash
just                 # Show all available recipes
just build           # Build with default settings (max 10 jobs)
just test            # Run all tests
just deploy          # Deploy to deploy/
just release         # Create versioned release package
```

**Using Make (Traditional)**:
```bash
make help            # Show all targets
make build           # Build project (max 10 jobs)
make test            # Run tests
make package         # Create release package
```

**Using CMake (Direct)** - ⚠️ Not recommended, use Just/Make instead:
```bash
# Only if Just/Make unavailable
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 10  # NEVER exceed 10 jobs
```

### Version Management

Every build is automatically versioned from Git tags:

**Version Format**: `MAJOR.MINOR.PATCH[+commits][-prerelease].HASH[-dirty]`

**Examples**:
- `v1.2.3.a1b2c3d4` - Release build on tag v1.2.3
- `1.2.3+5.a1b2c3d4-dev` - 5 commits after v1.2.3
- `1.2.3.a1b2c3d4-dirty` - Uncommitted changes

**Build Identifier**: `VERSION_FULL-BUILD_VARIANT-COMPILER_NAME-BUILD_HASH`

Example: `1.2.3+5.a1b2c3d4-dev-release-icx-e4d3b2a1`

### Build Variants

**Using Just**:
```bash
just build-msvc Release      # MSVC build (fastest iteration)
just build-oneapi perfpack   # Intel oneAPI with all optimizations
just build-all               # Build all variants
```

**Using Make**:
```bash
make build                   # Default Release build
make build-debug             # Debug build
make build-oneapi            # Intel oneAPI optimized
```

### Testing

**Using Just**:
```bash
just test-smoke              # Quick validation
just test-inference          # Model inference test
just test-session            # Session management tests
just benchmark               # Performance benchmark
```

**Using Make**:
```bash
make test                    # All tests
make test-smoke              # Quick validation
make benchmark               # Performance benchmark
```

### Running the Application

**Working Model Paths**:
- **2B Model**: `C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs`
- **4B Model**: `C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs`
- **Tokenizers**: Located in respective model directories

**Basic execution**:
```batch
:: 2B model (tested and working)
.\build\Release\gemma.exe ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs

:: 4B model (SFP format, better performance)
.\build\Release\gemma.exe ^
  --tokenizer C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\tokenizer.spm ^
  --weights C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs

:: Interactive mode (default)
.\build\Release\gemma.exe --weights [model.sbs] --tokenizer [tokenizer.spm]
```

### Development Commands

**Run benchmarks**:
```batch
.\build\Release\single_benchmark.exe ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm
```

**Convert weights** (if needed):
```batch
.\build\Release\migrate_weights.exe ^
  --tokenizer [tokenizer.spm] ^
  --weights [input.sbs] ^
  --output_weights [output-single.sbs]
```

## High-Level Architecture

### Core Components

- **gemma/** - Main inference engine with model implementations
  - `gemma.h/.cc` - Primary Gemma class and generation API
  - `configs.h/.cc` - Model configurations for all variants
  - `attention.h/.cc` - Multi-head attention implementation
  - `griffin.h/.cc` - RecurrentGemma architecture
  - `vit.h/.cc` - Vision Transformer for multimodal models
  - `kv_cache.h/.cc` - Key-value cache management

- **compression/** - Weight compression algorithms
  - SFP (Scaled Float Point) - 8-bit format for faster inference
  - NUQ (Non-Uniform Quantization) - 4-bit compression
  - BF16/F32 - Standard formats

- **ops/** - Optimized mathematical operations
  - Matrix multiplication with auto-tuning
  - SIMD-optimized operations via Highway library
  - Specialized kernels for different data types

- **io/** - I/O and storage layer
  - Memory-mapped file support
  - Parallel reading for fast weight loading
  - Cross-platform abstractions

### Model Support

The project supports multiple model architectures defined by the `Model` enum:
- **Gemma 2**: 2B, 9B, 27B parameter variants
- **Gemma 3**: 270M, 1B, 4B, 12B, 27B variants
- **Griffin/RecurrentGemma**: 2B recurrent architecture
- **PaliGemma 2**: 3B/10B vision-language models (224/448 resolution)

### Key Design Patterns

1. **Template-based SIMD optimization** - Highway library integration for portable vectorization
2. **Batch processing** - Efficient multi-query inference with `AllQueries`/`QBatch` system
3. **Memory efficiency** - NUMA-aware allocation, memory mapping, streaming support
4. **Type safety** - Strong typing with compile-time tensor shape checking

## Model Weights Setup

Model weights must be obtained from Kaggle or Hugging Face and placed in `/c/codedev/llm/.models/`:

1. Download from [Kaggle Gemma-2](https://www.kaggle.com/models/google/gemma-2/gemmaCpp)
2. Extract: `tar -xf archive.tar.gz`
3. Place files in `/c/codedev/llm/.models/`:
   - `gemma2-2b-it-sfp.sbs` (recommended starter model)
   - `tokenizer.spm`
   - Other model variants as needed

**Recommended models for testing**:
- `gemma2-2b-it-sfp` - Best balance of speed and quality
- `gemma3-1b-it-sfp` - Fastest for quick iteration
- `paligemma2-3b-mix-224-sfp` - For vision-language tasks

## API Usage

### Primary Generation API
```cpp
Gemma gemma(loader, inference, ctx);
KVCache kv_cache(gemma.Config(), inference, ctx.allocator);

// Token streaming callback
auto stream_token = [&](int token, float) {
  std::string token_text;
  gemma.Tokenizer().Decode({token}, &token_text);
  std::cout << token_text << std::flush;
  return true;
};

// Generate response
gemma.Generate(runtime_config, prompt_tokens, pos, kv_cache, env, timing_info);
```

### Batch Processing
```cpp
AllQueries queries;
// Add multiple queries...
gemma.GenerateBatch(runtime_config, queries, env, timing_info);
```

### Vision-Language Processing
```cpp
Image image;
ImageTokens image_tokens;
gemma.GenerateImageTokens(runtime_config, seq_len, image, image_tokens, env);
```

## Important Configuration

### Runtime Configuration
- `max_seq_len` - Maximum sequence length (default 4096)
- `temperature` - Sampling temperature (0.7 typical)
- `top_k` - Top-K sampling parameter (40 typical)

### Build Options
- Single `build/` directory for simplicity
- `CMAKE_BUILD_TYPE` - Release (default), Debug, RelWithDebInfo
- Highway library fetched from GitHub (more reliable than vcpkg)

### Performance Tips
1. Use SFP format models for better performance
2. Second/third queries are faster due to auto-tuning
3. Set Windows to performance mode (not battery saving)
4. Close CPU-intensive applications

## Directory-Specific Notes

- **compression/python/** - Python bindings and SafeTensors conversion tools
- **examples/hello_world/** - Minimal template for library usage
- **evals/** - Benchmarking and evaluation tools
- **paligemma/** - Vision-language model specific code
- **util/** - Threading, memory allocation, and platform utilities

The project follows data-oriented design principles, prioritizes small batch latency, and maintains a portable CPU baseline while supporting research experimentation.

## Troubleshooting

### Common Windows Build Issues

#### CMake Configuration Errors
**Issue**: CMake cannot find compiler or toolchain
**Solution**:
```batch
:: Ensure Visual Studio 2022 Build Tools are installed
:: Use explicit generator and toolset
cmake -B build -G "Visual Studio 17 2022" -T v143
```

#### Model Loading Errors
**Issue**: Error code 3221226356 when loading models
**Solution**: Install Visual C++ Redistributables or build in Release mode

#### Highway Library Issues
**Issue**: vcpkg Highway version incompatible
**Solution**: Let CMake fetch from GitHub (FetchContent fallback)

#### Path Issues
**Issue**: Model files not found
**Solution**: Use absolute Windows paths with backslashes or escape them

### Memory Issues
- Reduce context length with `--max_seq_len 2048`
- Use smaller models (2B instead of 4B)
- Close other applications to free RAM

## Dependency Strategy

### Core Dependencies
- **Highway**: SIMD library - fetched from GitHub via FetchContent (most reliable)
- **SentencePiece**: Tokenization - auto-fetched via FetchContent
- **nlohmann/json**: Optional, for future MCP support

### vcpkg Integration (Optional)
While vcpkg can be used, the project works best with FetchContent for critical dependencies:
```cmake
# In CMakeLists.txt
FetchContent_Declare(highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.2.0
)
```

### Future Enhancement: GPU Backends
GPU acceleration is planned but not required for basic functionality:
- CUDA backend for NVIDIA GPUs
- SYCL backend for Intel GPUs/NPUs
- Vulkan for cross-platform GPU support

## Requirements

### Minimum Requirements
- **C++20 compiler**: Visual Studio 2022 with v143 toolset
- **CMake 3.14+**: Available at `C:\Program Files\CMake\bin\cmake.exe`
- **RAM**: 8GB minimum (16GB recommended for 4B models)
- **Storage**: 10GB for model files

### Build Tools
- **Windows**: Visual Studio 2022 Build Tools or full IDE
- **Linux/macOS**: GCC 11+ or Clang 14+

## Intel OneAPI Integration

### Build with Intel OneAPI Compiler

**Available Build Scripts**:
```bash
# Standard build with Intel ICX compiler
.\build_oneapi.ps1 -Config std

# With TBB + IPP optimization libraries
.\build_oneapi.ps1 -Config tbb-ipp

# Full performance pack (TBB, IPP, DNNL, DPL)
.\build_oneapi.ps1 -Config perfpack -Clean

# Hybrid oneAPI (CPU) + CUDA (GPU) build
.\scripts\build-oneapi-cuda-fastdebug.ps1
```

**Key oneAPI Features**:
- **ICX Compiler**: Intel's LLVM-based compiler with aggressive optimizations
- **MKL**: Math Kernel Library for linear algebra (parallel mode)
- **IPP**: Integrated Performance Primitives for signal processing
- **TBB**: Threading Building Blocks for parallel execution
- **DNNL**: Deep Neural Network Library for inference acceleration
- **DPL**: Data Parallel C++ Library (header-only)

### oneAPI Compiler Optimizations

The Intel toolchain (`cmake/IntelToolchain.cmake`) applies:
- `-O3 -xHost -march=native`: Maximum optimization for your CPU
- `-mavx2 -mfma` or `-mavx512f`: SIMD instruction sets
- `-mkl=parallel -ipp=parallel`: Parallel Intel library modes
- `-fopenmp -parallel`: Multi-threading and parallel optimization
- `-flto`: Link-time optimization for Release builds

**Environment Setup**:
```powershell
# oneAPI environment initialization (automatic in build scripts)
& "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"

# Verify compilers
icx --version    # Intel C compiler
icpx --version   # Intel C++ compiler
```

### Hardware Backend Architecture

**Backend Selection**:
- **CPU (default)**: Highway SIMD library with multi-target dispatch
- **SYCL (Intel)**: Via DPC++ compiler for Intel GPUs/CPUs
- **CUDA**: NVIDIA GPU acceleration
- **Vulkan**: Cross-platform GPU compute (planned)

**CMake Options**:
```bash
# Enable specific backends
-DGEMMA_ENABLE_SYCL=ON           # Intel SYCL backend
-DGEMMA_ENABLE_CUDA=ON           # NVIDIA CUDA backend
-DGEMMA_ENABLE_VULKAN=ON         # Vulkan compute (future)

# oneAPI library integration
-DGEMMA_USE_ONEAPI_LIBS=ON       # Enable oneAPI framework libraries
-DGEMMA_USE_TBB=ON               # Threading Building Blocks
-DGEMMA_USE_IPP=ON               # Integrated Performance Primitives
-DGEMMA_USE_DNNL=ON              # Deep Neural Network Library
-DGEMMA_USE_DPL=ON               # Data Parallel C++ Library

# Compiler selection
-DCMAKE_TOOLCHAIN_FILE=cmake/IntelToolchain.cmake  # Intel ICX compiler
```

## Build System Best Practices

### CRITICAL BUILD REQUIREMENTS

**ALWAYS use Just or Make, never manual CMake commands**:
- ✅ **Recommended**: `just build-msvc Release` or `make build`
- ✅ **Alternative**: Use existing PowerShell scripts (`.\build-simple.ps1`, `.\build_oneapi.ps1`)
- ✅ Scripts/frontends handle: Versioning, Ninja generator, compiler caching, optimized flags
- ✅ Automatic version embedding in binary
- ❌ **DO NOT** use raw `cmake -B build` commands (bypasses versioning and optimizations)

**MANDATORY: Limit parallel jobs to maximum 10 cores**:
```bash
# Just recipes (auto-limited to 10)
just build-msvc Release          # Uses max_jobs=10
just build-oneapi perfpack       # Uses max_jobs=10

# Make targets (auto-limited to 10)
make build JOBS=10               # Default: 10 jobs
make build-oneapi JOBS=8         # Safe for large builds

# PowerShell scripts (if Just/Make unavailable)
.\build-simple.ps1 -Config Release -Jobs 10
.\build_oneapi.ps1 -Config perfpack -Jobs 8
```

**Cache management requirements**:
- Reuse build artifacts across builds (sccache/ccache automatic)
- Preserve incremental compilation state
- Never use `-Clean` flag unless absolutely necessary
- Monitor cache hit rates: `sccache --show-stats` or `ccache -s`

### Successful Build Patterns

**1. Simple CPU-only build** (fastest, most reliable):
```bash
.\build-simple.ps1 -Config Release -Jobs 10
# Uses: Ninja, sccache, Highway SIMD
# Output: .\build-ninja\Release\gemma.exe
# Build time: ~2-3 minutes with warm cache
```

**2. oneAPI optimized build**:
```bash
.\build_oneapi.ps1 -Config perfpack -Jobs 8
# Uses: Intel ICX, MKL, IPP, TBB, DNNL
# Output: .\build_perfpack\bin\gemma.exe
# Build time: ~5-8 minutes with warm cache
```

**3. Hybrid SYCL + CUDA build**:
```bash
.\scripts\build-oneapi-cuda-fastdebug.ps1 -Jobs 10
# Uses: Intel ICX (CPU), NVCC (GPU), ccache
# Output: .\build\oneapi-cuda-fastdebug\bin\gemma.exe
# Build time: ~8-12 minutes with warm cache
```

### Build Acceleration

**Compiler Caching**:
```powershell
# sccache for simple builds (build-simple.ps1)
$env:CMAKE_C_COMPILER_LAUNCHER = "sccache"
$env:CMAKE_CXX_COMPILER_LAUNCHER = "sccache"
sccache --show-stats  # View cache hit rate

# ccache for oneAPI builds
$env:CCACHE_DIR = "C:\Users\david\.cache\ccache"
$env:CCACHE_MAXSIZE = "20G"
ccache -s  # View cache statistics
```

**Parallel Jobs Calculation**:
```powershell
# From build-oneapi-cuda-fastdebug.ps1
$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$memGB = [math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 2)
$jobCount = [math]::Min([math]::Floor($cores * 0.7), [math]::Floor($memGB / 2.5))
# Typically: 70% of cores, with 2.5GB RAM per job
```

### Benchmarking Best Practices

**Quick Benchmark**:
```batch
.\build\Release\single_benchmark.exe ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm
```

**Performance Comparison**:
```powershell
# Compare different builds
.\compare_builds.ps1

# Baseline benchmark
.\benchmark_baseline.ps1
```

**Optimization Tips**:
1. **First run is always slower** - model loading and auto-tuning
2. **Second/third runs show real performance** - caches warmed up
3. **Use SFP models** - 2x faster than standard BF16 format
4. **Close background apps** - especially during benchmarking
5. **Windows performance mode** - disable battery saving

## Project Status

### Working ✅
- **Versioned build automation system** with hash-linked releases
- **Session management** with persistence, commands (%s, %l, %h, %i, %m, %c)
- CPU inference with Highway SIMD optimization
- 2B, 4B model support (tested and verified)
- Interactive prompt mode and batch processing
- Comprehensive test framework (8-phase validation)
- Windows native compilation (Visual Studio 2022)
- Intel oneAPI compiler integration (ICX)
- Build acceleration (sccache, ccache)
- Multiple backend architecture (CPU, SYCL, CUDA)
- Just/Make/CMake build frontends
- GitHub Actions CI/CD pipeline

### In Progress 🚧
- SYCL backend stabilization for Intel GPUs
- CUDA backend optimization for NVIDIA GPUs
- Advanced sampling methods (Min-P, Dynatemp, Mirostat)
- Performance profiling with VTune
- Hybrid CPU+GPU inference pipelines

### Future Plans 📋
- Vulkan backend for cross-platform GPU support
- MCP server integration for AI workflow automation
- Context window extensions (8K → 16K → 32K)
- Model quantization improvements (4-bit NUQ)
- Distributed inference across multiple GPUs
- Signed releases (GPG) with SBOM/SLSA provenance

## TODO: Build System Improvements

### High Priority
- [ ] Add CMake preset for oneAPI with all performance libraries
- [ ] Create unified build script that auto-detects available backends
- [ ] Implement automated benchmark comparison across build configs
- [ ] Add VTune profiling integration to build system

### Medium Priority
- [ ] Create Docker images for reproducible builds
- [ ] Add cross-compilation support for Linux targets
- [ ] Implement continuous benchmarking in CI/CD
- [ ] Document optimal CMake cache variables for each build type

### Low Priority
- [ ] Create vcpkg overlay ports for gemma-specific dependencies
- [ ] Add Conan package manager support as alternative to vcpkg
- [ ] Implement build artifact versioning with git commit hashes
- [ ] Create automated build matrix testing (MSVC, ICX, GCC, Clang)