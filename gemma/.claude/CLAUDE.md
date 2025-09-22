# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Gemma.cpp project is a sophisticated C++ inference engine for Google's Gemma models with extensive enhancements beyond the original implementation. The codebase consists of three layers:
1. **Original gemma.cpp**: Core inference engine with SIMD optimizations
2. **Enhanced infrastructure**: MCP server, hardware acceleration backends (SYCL, CUDA, Vulkan)
3. **Comprehensive testing**: 18 test files across unit, integration, and performance categories

**Current Status**: The project is in active development with the original gemma.cpp functional on Linux/WSL but requiring patches for Windows compilation. Enhanced features are structurally complete but need compilation verification.

## Build and Run Commands

### Windows Native Build (Requires Patches)

```bash
# CMake 4.1.1 is available at /c/Program Files/CMake/bin/cmake
export PATH="/c/Program Files/CMake/bin:$PATH"

# Use Visual Studio 2022 generator (ClangCL preset doesn't work)
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4

# For enhanced features (after fixing compilation)
cmake -B build-enhanced \
  -DGEMMA_BUILD_MCP_SERVER=ON \
  -DGEMMA_BUILD_ENHANCED_TESTS=ON \
  -DGEMMA_AUTO_DETECT_BACKENDS=ON
cmake --build build-enhanced --config Release
```

**Known Issues**:
- `ops/ops-inl.h:1239`: Replace `recent_tokens.empty()` with `recent_tokens.size() > 0`
- `gemma/gemma.cc:464`: Change `[&, &recent_tokens]` to `[&]`

### WSL/Linux Build (Recommended)

```bash
# Original gemma.cpp (working)
cd /mnt/c/codedev/llm/gemma
cmake --preset make
cmake --build --preset make -j $(nproc)

# With enhanced features
cmake -B build \
  -DGEMMA_BUILD_MCP_SERVER=ON \
  -DGEMMA_BUILD_BACKENDS=ON \
  -DGEMMA_BUILD_ENHANCED_TESTS=ON
cmake --build build -j $(nproc)
```

### Running the Application

```bash
# Basic inference (model files in /c/codedev/llm/.models/)
./build/gemma \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --weights /c/codedev/llm/.models/gemma2-2b-it-sfp.sbs \
  --prompt "Hello world"

# MCP Server (stdio transport ready)
./build/gemma_mcp_stdio_server \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --weights /c/codedev/llm/.models/gemma2-2b-it-sfp.sbs

# Run comprehensive tests
./tests/run_tests.sh all              # All test categories
python run_tests.py unit               # Unit tests only
python run_tests.py performance        # Benchmarks only
```

### Model Files Setup

Available models in `/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/`:
- `2b-it.sbs` - Main model weights
- `tokenizer.spm` - Tokenizer model

Copy to expected location:
```bash
cp /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/* /c/codedev/llm/.models/
```

## High-Level Architecture

### Three-Layer Architecture

1. **Core Layer (gemma/)**:
   - Template-heavy C++ with Highway SIMD library
   - Attention mechanisms with KV cache
   - Griffin/RecurrentGemma support
   - Vision Transformer (ViT) for multimodal

2. **Enhancement Layer**:
   - **MCP Server** (mcp/): Full JSON-RPC 2.0 implementation with stdio transport
   - **Hardware Backends** (backends/): SYCL, CUDA, Vulkan implementations
   - **Advanced Sampling** (ops/): Min-P, Dynatemp, DRY, Typical algorithms

3. **Testing Layer**:
   - **Unit Tests**: Model loading, tokenization, memory, sampling
   - **Integration Tests**: End-to-end inference, backend compatibility
   - **Performance Tests**: Benchmarks with Google Benchmark framework

### MCP Server Architecture

The MCP server provides tool-calling capabilities via JSON-RPC 2.0:
- **Tools**: `generate_text`, `count_tokens`, `get_model_info`
- **Transport**: Stdio (production-ready), WebSocket (planned)
- **Integration**: Direct C++ API usage, no subprocess overhead

### Hardware Backend System

Modular backend architecture with auto-detection:
- **SYCL Backend**: Intel GPUs/NPUs with USM memory management
- **CUDA Backend**: Multi-GPU support with Flash Attention v2
- **Vulkan Backend**: Cross-platform GPU with compute shaders
- **Plugin System**: Extensible for future backends

## Critical Build Information

### Dependency Management

CMake 4.1.1 available at: `/c/Program Files/CMake/bin/cmake`

All dependencies use FetchContent (automatic download):
- Highway (SIMD operations)
- SentencePiece (tokenization)
- nlohmann/json (JSON parsing)
- Google Test/Benchmark (testing)

### Build Targets

```bash
# Core targets
gemma                  # Main CLI executable
libgemma              # Static library
gemma_mcp_stdio_server # MCP stdio server

# Backend targets (when enabled)
gemma_sycl            # Intel GPU backend
gemma_cuda            # NVIDIA GPU backend
gemma_vulkan          # Cross-platform GPU

# Test targets
test_unit             # Unit tests
test_integration      # Integration tests
test_performance      # Performance benchmarks
test_all             # Complete suite
```

### CMake Options

```cmake
# Core options
GEMMA_ENABLE_TESTS=ON              # Original test suite
CMAKE_BUILD_TYPE=Release           # Build configuration

# Enhanced options
GEMMA_BUILD_MCP_SERVER=ON          # MCP server component
GEMMA_BUILD_BACKENDS=ON            # Hardware acceleration
GEMMA_BUILD_ENHANCED_TESTS=ON      # Comprehensive tests
GEMMA_AUTO_DETECT_BACKENDS=ON      # Auto-detect available SDKs

# Backend-specific (auto-detected if AUTO_DETECT=ON)
GEMMA_BUILD_SYCL_BACKEND=OFF       # Intel GPU/NPU
GEMMA_BUILD_CUDA_BACKEND=OFF       # NVIDIA GPU
GEMMA_BUILD_VULKAN_BACKEND=OFF     # Cross-platform GPU
```

## Testing Infrastructure

### Test Organization

```
tests/
├── unit/           # 6 test files - No model dependencies
├── integration/    # 3 test files - Full pipeline tests
├── performance/    # 1 benchmark file - Performance metrics
├── backends/       # 4 test files - Hardware-specific
├── functional/     # 2 test files - Cross-backend validation
└── mcp/           # 1 test file - MCP protocol compliance
```

### Running Tests

```bash
# Quick validation (no models required)
./build/test_unit

# Full test suite (requires models)
python run_tests.py all

# Specific categories
./tests/run_tests.sh unit
./tests/run_tests.sh benchmarks

# Backend validation
python scripts/validate_backend.py cuda
python scripts/validate_backend.py sycl
```

### Performance Targets

- **Tokenization**: > 10,000 tokens/second
- **Generation**: > 50 tokens/second (CPU), > 500 tokens/second (GPU)
- **First Token**: < 100ms on modern hardware
- **Memory**: < 4GB for 2B model

## Current Development Status

### ✅ Complete and Working
- Original gemma.cpp inference engine (Linux/WSL)
- MCP server with stdio transport
- Comprehensive test framework structure
- Advanced sampling algorithms implementation
- Backend architecture and abstractions

### ⚠️ Requires Verification
- Windows native build (needs source patches)
- Hardware backend compilation (SDK dependencies)
- WebSocket MCP transport
- Full integration test execution

### ❌ Known Issues
- `hwy::Span::empty()` incompatibility (line 1239 in ops-inl.h)
- Lambda capture syntax error (line 464 in gemma.cc)
- Model file path mismatches between tests and actual locations

## Quick Fixes for Windows Compilation

```cpp
// Fix 1: ops/ops-inl.h line 1239
// Change: if (dry_multiplier > 0.0f && !recent_tokens.empty())
// To:     if (dry_multiplier > 0.0f && recent_tokens.size() > 0)

// Fix 2: gemma/gemma.cc line 464
// Change: return [&, &recent_tokens](float* logits, size_t vocab_size)
// To:     return [&](float* logits, size_t vocab_size)
```

## Development Workflow

1. **Primary Development**: Use WSL/Linux for active development
2. **Windows Testing**: Apply patches and test Windows-specific features
3. **Backend Development**: Ensure SDK installation before enabling backends
4. **MCP Development**: Use stdio server for immediate testing
5. **Test-Driven**: Run unit tests frequently, integration tests before commits

## Performance Optimization Notes

- **Auto-tuning**: Second/third inference runs are faster due to kernel auto-tuning
- **Model Format**: Use `-sfp` models for 2x speed improvement over BF16
- **Batch Processing**: Use `GenerateBatch()` for multiple queries
- **Memory Mapping**: Models are memory-mapped for efficient loading
- **SIMD Usage**: Highway library provides portable vectorization

## External Dependencies and References

### Model Sources
- [Kaggle Gemma-2](https://www.kaggle.com/models/google/gemma-2/gemmaCpp) - Official model weights
- [Hugging Face Gemma](https://huggingface.co/google/gemma-2b) - Alternative source

### Key Technologies
- [Highway](https://github.com/google/highway) - SIMD library
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization
- [cpp-mcp](https://github.com/hkr04/cpp-mcp) - MCP SDK reference
- [Intel IPEX-LLM](https://github.com/intel/ipex-llm) - Intel optimization reference

## Important Implementation Details

### Template-Heavy Design
The codebase uses extensive C++ templates for compile-time optimization. Key patterns:
- CRTP (Curiously Recurring Template Pattern) for static polymorphism
- Template metaprogramming for tensor operations
- Compile-time shape checking for type safety

### Memory Management
- NUMA-aware allocation for multi-socket systems
- Memory-mapped file I/O for efficient model loading
- Custom allocators with alignment guarantees
- Thread-local storage for token history

### SIMD Optimization
- Highway library for portable vectorization
- Platform-specific optimizations (AVX2, AVX-512, NEON)
- Auto-vectorization hints for compilers
- Explicit SIMD kernels for critical paths