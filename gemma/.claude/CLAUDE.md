# CLAUDE.md - Gemma.cpp Development Guide

This file provides guidance to Claude Code (claude.ai/code) when working with the Gemma.cpp project.

## Project Overview

Gemma.cpp is a lightweight C++ inference engine for Google's Gemma foundation models. Currently focused on CPU inference with SIMD optimization, with GPU acceleration planned as future enhancement.

## Build Instructions

### Quick Build (Windows)
```batch
:: Navigate to project root
cd C:\codedev\llm\gemma

:: Simple build (single 'build' directory)
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4

:: The executable will be in: build\Release\gemma.exe
```

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

## Current Status

### Working ✅
- CPU inference with Highway SIMD
- 2B and 4B model support
- Windows native compilation
- Interactive prompt mode
- Basic benchmarking

### Planned 📋
- GPU backends (CUDA, SYCL, Vulkan)
- Advanced sampling (Min-P, Dynatemp, Mirostat)
- MCP server integration
- Session management with context preservation

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
```batch
:: Build
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release

:: Run 2B model
.\build\Release\gemma.exe ^
  --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs ^
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm

:: Benchmark
.\build\Release\single_benchmark.exe --weights [model] --tokenizer [tokenizer]
```

### Model Paths
- 2B: `C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\`
- 4B: `C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\`

Remember: Keep it simple, use single `build/` directory, let CMake fetch dependencies from GitHub.