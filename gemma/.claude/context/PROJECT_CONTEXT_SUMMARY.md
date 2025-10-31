# Gemma.cpp Project Context Summary
*Generated: 2025-01-19*

## Project Overview
**Gemma.cpp Enhanced** - Production-ready C++ inference engine for Google's Gemma models with extensive hardware acceleration and MCP server integration.

- **Repository**: https://github.com/david-t-martel/dtm-gemma.git
- **Root Path**: `C:\codedev\llm\gemma`
- **Latest Commits**:
  - ea2dca7: Major enhancement implementation
  - 3e2c3f9: CI/CD optimization

## Architecture Summary

### Three-Layer Design
1. **Core Layer** (`gemma/`): Template-heavy C++ with Highway SIMD
2. **Enhancement Layer** (`backends/`, `mcp/`): Hardware acceleration & MCP server
3. **Testing Layer** (`tests/`): 18 test files across unit/integration/performance

### Key Components
- **CUDA Backend**: Complete with memory pooling and stream management
- **Intel SYCL Backend**: GPU/NPU acceleration via oneAPI 2025.1
- **Vulkan Backend**: Cross-platform GPU support
- **MCP Server**: JSON-RPC 2.0 with stdio transport (WebSocket planned)

## Current Status

### ✅ Completed
- Hardware acceleration backends (CUDA, SYCL, Vulkan)
- MCP server with stdio transport
- Build optimizations (PCH, ccache, unity builds)
- CI/CD pipeline with self-hosted runners
- Pre-commit framework with auto-claude integration
- Comprehensive test framework (18 test files)

### 🚧 In Progress
- Windows compilation fixes (ApplyDRYPenalty linker issues)
- WebSocket MCP transport
- Flash Attention v3 implementation

### ⚠️ Known Issues
1. **ops/ops-inl.h:1239**: Replace `recent_tokens.empty()` with `recent_tokens.size() > 0`
2. **gemma/gemma.cc:464**: Change lambda capture from `[&, &recent_tokens]` to `[&]`
3. Long compilation times without ccache (10+ minutes vs 3-5 with optimizations)

## Quick Start Commands

### Optimized Build
```bash
# Quick optimized build script
./optimized_build.sh

# CMake presets
cmake --preset windows-fast-debug
cmake --preset make && cmake --build --preset make -j $(nproc)

# With all backends
cmake -B build -DGEMMA_AUTO_DETECT_BACKENDS=ON
```

### Running
```bash
# Basic inference
./build/gemma \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --weights /c/codedev/llm/.models/gemma2-2b-it-sfp.sbs

# MCP Server
./build/gemma_mcp_stdio_server \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --weights /c/codedev/llm/.models/gemma2-2b-it-sfp.sbs
```

### Testing
```bash
# Quick unit tests (no models required)
./build/test_unit

# Full test suite
python run_tests.py all

# Specific categories
./tests/run_tests.sh unit
./tests/run_tests.sh benchmarks
```

## Performance Metrics

- **Inference**: 50-500 tokens/second on GPUs
- **Build Time**: 3-5 minutes with optimizations
- **First Token**: < 100ms on modern hardware
- **Memory**: < 4GB for 2B model
- **Speedup**: 5-50x with hardware acceleration

## File Locations

- **Project Root**: `C:\codedev\llm\gemma`
- **Build Tools**: `C:\users\david\.local\bin`
- **Model Files**: `C:\codedev\llm\.models`
- **Auto-claude**: `C:\users\david\.claude`
- **CMake**: `C:\Program Files\CMake\bin\cmake`

## Agent Coordination History

### Successful Agents
- **cpp-pro**: Created CUDA backend implementation
- **backend-architect**: Configured Intel SYCL, created backend manager
- **deployment-engineer**: Created CI/CD pipelines and Docker configs

### Successful Combinations
- cpp-pro + backend-architect → Hardware acceleration
- deployment-engineer + quality-engineer → CI/CD setup

## Future Roadmap

### Immediate Priorities
1. Fix remaining Windows compilation issues
2. Complete WebSocket MCP transport
3. Validate all hardware backends

### Planned Features
- Flash Attention v3 (memory efficiency)
- Speculative decoding (2-3x speedup)
- Metal backend (macOS support)
- Model quantization (INT8/INT4)

### Performance Opportunities
- Kernel fusion for attention layers
- Graph optimization for inference
- Dynamic batching for throughput

### Technical Debt
- Clean up build directories from git
- Refactor ops-inl.h for modularity
- Update documentation with latest changes

## Key Design Patterns

- **CRTP**: Curiously Recurring Template Pattern for static polymorphism
- **Template Metaprogramming**: Compile-time tensor operations
- **Factory Pattern**: Backend instantiation
- **RAII**: Resource management
- **Memory-Mapped I/O**: Efficient model loading

## Testing Strategy

### Categories
- **Unit Tests**: 6 files, no model dependencies
- **Integration Tests**: 3 files, full pipeline validation
- **Performance Tests**: 1 benchmark file
- **Backend Tests**: 4 files, hardware-specific
- **Functional Tests**: 2 files, cross-backend validation
- **MCP Tests**: 1 file, protocol compliance

## Critical Notes

1. **Always use optimized builds** - Standard builds take 10+ minutes
2. **Windows requires patches** - Apply fixes to ops-inl.h and gemma.cc
3. **Use SFP models** - 2x speed improvement over BF16
4. **Enable ccache** - Critical for development iteration
5. **Test on WSL first** - More stable than Windows native

## Contact for Context
This context was captured for the Gemma.cpp enhanced project with comprehensive hardware acceleration, CI/CD, and pre-commit framework integration. The project is actively developed with a focus on production readiness and performance optimization.