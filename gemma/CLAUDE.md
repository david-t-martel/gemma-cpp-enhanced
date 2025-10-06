# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gemma.cpp is a lightweight C++ inference engine for Google's Gemma foundation models, designed for research and experimentation. It provides minimalist implementations of Gemma-2, Gemma-3, Griffin/RecurrentGemma, and PaliGemma-2 (vision-language) models, prioritizing simplicity and directness over full generality.

## Build Strategy

The project uses a simplified single-directory build approach (`build/`) with dependencies managed through a combination of vcpkg and GitHub submodules. CPU-first development with GPU backends as future enhancements.

## Build and Run Commands

### Building the Project

**Windows (Simplified Build)**:
```batch
:: Set up environment (if using vcpkg)
set VCPKG_ROOT=C:\codedev\vcpkg

:: Basic CPU build (currently working)
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4

:: With vcpkg for some dependencies (optional)
cmake -B build -G "Visual Studio 17 2022" -T v143 ^
  -DCMAKE_TOOLCHAIN_FILE=C:\codedev\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release -j 4
```

**Unix/Linux/macOS**:
```bash
# Standard build
cmake -B build
cmake --build build -j $(nproc)

# Or using presets
cmake --preset make
cmake --build --preset make -j $(nproc)
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

## Project Status

### Working
- ✅ CPU inference with Highway SIMD optimization
- ✅ 2B and 4B model support
- ✅ Interactive prompt mode
- ✅ Basic benchmarking tools
- ✅ Windows native compilation

### In Progress
- 🚧 Simplified dependency management
- 🚧 Advanced sampling methods
- 🚧 Performance optimizations

### Future Plans
- 📋 GPU acceleration (CUDA, SYCL, Vulkan)
- 📋 MCP server integration
- 📋 Session management
- 📋 Context window extensions