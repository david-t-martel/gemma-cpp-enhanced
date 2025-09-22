# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gemma.cpp is a lightweight C++ inference engine for Google's Gemma foundation models, designed for research and experimentation. It provides minimalist implementations of Gemma-2, Gemma-3, Griffin/RecurrentGemma, and PaliGemma-2 (vision-language) models, prioritizing simplicity and directness over full generality.

## Build and Run Commands

### Building the Project

**Windows (CMake 4.1.1 available at /c/Program Files/CMake/bin/cmake)**:
```bash
# Set PATH for CMake (if not already in PATH)
export PATH="/c/Program Files/CMake/bin:$PATH"

# Configure and build (Visual Studio 2022 generator recommended)
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4

# Alternative: Try presets (may fail with ClangCL)
cmake --preset windows
cmake --build --preset windows -j 4
```

**Unix/Linux/macOS**:
```bash
# Configure and build
cmake --preset make
cmake --build --preset make -j $(nproc)

# Alternative: Direct CMake approach
cmake -B build
cd build && make -j$(nproc)
```

**Bazel (alternative)**:
```bash
bazel build -c opt --cxxopt=-std=c++20 :gemma
```

### Running the Application

**Basic execution** (requires model weights from /c/codedev/llm/.models):
```bash
./build/gemma \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --weights /c/codedev/llm/.models/gemma2-2b-it-sfp.sbs
```

**With single-file format** (newer weights with embedded tokenizer):
```bash
./build/gemma --weights /c/codedev/llm/.models/gemma2-2b-it-sfp-single.sbs
```

**PaliGemma (vision-language model)**:
```bash
./build/gemma \
  --tokenizer /c/codedev/llm/.models/paligemma_tokenizer.model \
  --weights /c/codedev/llm/.models/paligemma2-3b-mix-224-sfp.sbs \
  --image_file image.ppm
```

### Development Commands

**Run benchmarks**:
```bash
./build/single_benchmark --weights [model.sbs] --tokenizer [tokenizer.spm]
./build/benchmarks --weights [model.sbs] --tokenizer [tokenizer.spm]
```

**Convert weights to single-file format**:
```bash
./build/migrate_weights \
  --tokenizer [tokenizer.spm] \
  --weights [input.sbs] \
  --output_weights [output-single.sbs]
```

**Debug prompt interaction**:
```bash
./build/debug_prompt --weights [model.sbs] --tokenizer [tokenizer.spm]
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
- `max_seq_len` - Maximum sequence length (32K typical, 128K possible)
- `decode_qbatch_size` - Batch size for decoding
- `prefill_tbatch_size` - Batch size for prefill
- `temperature` - Sampling temperature
- `top_k` - Top-K sampling parameter

### Build Options
- `BUILD_GEMMA_DLL` - Build shared library for C# interop
- `GEMMA_ENABLE_TESTS` - Enable test suite compilation
- `CMAKE_BUILD_TYPE` - Release (default), Debug, RelWithDebInfo

### Performance Tips
1. Use `-sfp` models for 2x speed improvement over BF16
2. Second/third queries are faster due to auto-tuning
3. Set laptop to performance mode (not battery saving)
4. Close CPU-intensive applications
5. Warm-up period expected on macOS

## Directory-Specific Notes

- **compression/python/** - Python bindings and SafeTensors conversion tools
- **examples/hello_world/** - Minimal template for library usage
- **evals/** - Benchmarking and evaluation tools
- **paligemma/** - Vision-language model specific code
- **util/** - Threading, memory allocation, and platform utilities

The project follows data-oriented design principles, prioritizes small batch latency, and maintains a portable CPU baseline while supporting research experimentation.

## Enhancement Roadmap

### Stage 1: Foundation & Quick Wins (Weeks 1-3)
- Fix model loading error (3221226356) with VC++ runtime
- Implement advanced sampling (Min-P, Dynatemp, DRY, Typical)
- Integrate vcpkg for dependency management
- Create comprehensive testing framework

### Stage 2: MCP Integration (Weeks 4-6)
- Integrate cpp-mcp SDK for Model Context Protocol
- Implement core tools (generate_text, count_tokens, model_info)
- Add HTTP/WebSocket transport via Crow
- Create plugin architecture for extensibility

### Stage 3: Hardware Acceleration (Weeks 7-10)
- Intel GPU/NPU via SYCL and oneAPI
- NVIDIA CUDA with cuBLAS/cuDNN
- Cross-platform Vulkan support
- Expected speedup: 5-50x

### Stage 4: Advanced Features (Weeks 11-14)
- Mirostat sampling (v1 & v2)
- Grammar-constrained generation (GBNF)
- Speculative decoding (2-3x speedup)
- Context window extensions (RoPE scaling)

### Stage 5: Performance Optimization (Weeks 15-18)
- Advanced quantization (AWQ/GPTQ, 50-75% memory reduction)
- Kernel fusion (15-25% speedup)
- Continuous batching (2-3x throughput)
- Flash Attention implementation

## Testing Strategy

### Unit Tests
- Sampling algorithms validation
- MCP protocol compliance
- Hardware backend verification
- Quantization accuracy tests

### Integration Tests
- End-to-end inference validation
- Multi-backend compatibility
- MCP server functionality
- Performance benchmarks

### Performance Targets
- Latency: <100ms first token on RTX 4060
- Throughput: 100+ tokens/second on modern GPUs
- Memory: 50% reduction via quantization
- Quality: Maintained or improved generation

## Build Configuration

### New CMake Options
```cmake
# MCP Server support
option(BUILD_GEMMA_MCP "Build with MCP server support" ON)

# Hardware acceleration
option(GEMMA_CUDA "Build with CUDA support" OFF)
option(GEMMA_SYCL "Build with SYCL/oneAPI support" OFF)
option(GEMMA_VULKAN "Build with Vulkan support" OFF)

# Advanced features
option(GEMMA_FLASH_ATTENTION "Enable Flash Attention" OFF)
option(GEMMA_SPECULATIVE "Enable speculative decoding" OFF)
```

## Dependencies

### Required
- C++20 compiler (MSVC 2022, GCC 11+, Clang 14+)
- CMake 4.1.1+ (available at `/c/Program Files/CMake/bin/cmake`)
- Highway SIMD library
- nlohmann/json (for MCP)

### Optional
- Intel oneAPI (for SYCL backend)
- CUDA Toolkit 12+ (for NVIDIA GPUs)
- Vulkan SDK (for cross-platform GPU)
- Crow framework (for HTTP/WebSocket MCP)

## External References

### Key Repositories
- [Intel IPEX-LLM](https://github.com/intel/ipex-llm) - Intel GPU/NPU acceleration
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Reference implementation
- [cpp-mcp](https://github.com/hkr04/cpp-mcp) - C++ MCP SDK
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA optimization