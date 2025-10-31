# Gemma.cpp Build and Run Task - Onboarding Documentation

## Task Overview
Build and run the gemma.cpp application with model weights located in `/c/codedev/llm/.models`.

## Environment Analysis

### System Information
- **OS**: Windows 10/11 (MINGW64_NT-10.0-26100)
- **Architecture**: x86_64
- **Shell**: Git Bash (MSYS)
- **Working Directory**: `c:\codedev\llm\gemma\gemma.cpp`

### Build Tools
- **CMake**: Version 4.1.1 (installed at `C:\Program Files\CMake`)
- **Visual Studio**: 2022 Build Tools (v17)
- **MSVC Compiler**: v143 (14.44.35207)
- **C++ Standard**: C++20 (required for designated initializers)
- **vcpkg**: Not installed (optional, can be added for package management)

### Model Weights
- **Location**: `/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/`
- **Files Available**:
  - `2b-it.sbs` (5.0 GB) - Compressed model weights
  - `tokenizer.spm` (4.2 MB) - SentencePiece tokenizer

## Project Architecture

### Core Components
1. **gemma/** - Main inference engine
   - `gemma.h/cc` - Primary API
   - `configs.h/cc` - Model configurations
   - `attention.h/cc` - Multi-head attention
   - `griffin.h/cc` - RecurrentGemma support
   - `vit.h/cc` - Vision Transformer for multimodal

2. **compression/** - Weight compression algorithms
   - SFP (Scaled Float Point) - 8-bit format
   - NUQ (Non-Uniform Quantization) - 4-bit
   - BF16/F32 standard formats

3. **ops/** - Mathematical operations
   - SIMD-optimized via Highway library
   - Auto-tuning matrix multiplication
   - Specialized kernels per data type

4. **io/** - I/O and storage
   - Memory-mapped file support
   - Cross-platform abstractions
   - Parallel reading for fast loading

### Supported Models
- **Gemma 2**: 2B, 9B, 27B variants
- **Gemma 3**: 270M, 1B, 4B, 12B, 27B variants
- **Griffin/RecurrentGemma**: 2B recurrent architecture
- **PaliGemma 2**: 3B/10B vision-language models

## Build Process

### Dependencies (via CMake FetchContent)
1. **Highway** - SIMD operations library
2. **SentencePiece** - Tokenization
3. **nlohmann/json** - JSON parsing
4. **Google Benchmark** - Performance testing

### Build Configuration
- **Generator**: Visual Studio 17 2022
- **Toolset**: MSVC v143 (ClangCL not available)
- **Build Type**: Release
- **Architecture**: x64

### Current Build Status
- ✅ CMake configuration completed
- ✅ Dependencies built successfully (Highway, SentencePiece, Benchmark, JSON)
- ❌ Main library build failing (libgemma.lib)
- ❌ Main executable not built (gemma.exe)
- 🔧 **Critical Issue**: griffin.obj file locked/corrupted (0 bytes)
- ⚠️ Type conversion warnings present (safe to ignore)

### Build Commands
```bash
# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64 -T v143 -DCMAKE_CXX_STANDARD=20

# Build (single-threaded to avoid file locking)
cmake --build build --config Release -j 1

# Run
build\Release\gemma.exe --weights C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm
```

## Created Resources

### 1. Environment Configuration (`.env`)
- Kaggle API credentials placeholders
- GCP service account configuration
- Model paths and runtime settings
- Build configuration variables

### 2. Kaggle Model Downloader Utilities
- **Shell Script**: `scripts/download-kaggle-model.sh`
  - Flexible parameter handling
  - Automatic extraction support
  - Progress indication
- **Batch Script**: `scripts/download-kaggle-model.bat`
  - Windows-native version
  - Environment validation

### 3. Build Scripts
- **PowerShell**: `build-gemma.ps1`
  - MSVC environment setup
  - Optional vcpkg integration
  - Parallel/single-threaded fallback
- **Batch**: `build-msvc.bat`
  - Direct MSVC invocation
  - Visual Studio detection

## API Usage Patterns

### Basic Generation
```cpp
Gemma gemma(loader, inference, ctx);
KVCache kv_cache(gemma.Config(), inference, ctx.allocator);

auto stream_token = [&](int token, float) {
  std::string token_text;
  gemma.Tokenizer().Decode({token}, &token_text);
  std::cout << token_text << std::flush;
  return true;
};

gemma.Generate(runtime_config, prompt_tokens, pos, kv_cache, env, timing_info);
```

### Batch Processing
```cpp
AllQueries queries;
// Add multiple queries
gemma.GenerateBatch(runtime_config, queries, env, timing_info);
```

## Performance Optimization Tips

1. **Model Selection**: Use `-sfp` models for 2x speed improvement
2. **Auto-tuning**: Second/third queries are faster
3. **Power Settings**: Ensure performance mode (not battery saving)
4. **Resource Management**: Close CPU-intensive applications
5. **Platform-specific**: macOS may have warm-up period

## Common Issues and Solutions

### Issue 1: CMake Configuration Incomplete
- **Cause**: Process interrupted during dependency fetch
- **Solution**: Re-run CMake configuration

### Issue 2: ClangCL Not Available
- **Cause**: LLVM not installed with ClangCL support
- **Solution**: Use MSVC v143 toolset or install LLVM

### Issue 3: C++20 Features Required
- **Cause**: Project uses designated initializers
- **Solution**: Set CMAKE_CXX_STANDARD to 20

### Issue 4: Permission Denied on Object Files
- **Cause**: Antivirus or parallel build conflicts
- **Solution**: Use single-threaded build (-j 1)

### Issue 5: griffin.obj File Locked (LNK1104)
- **Cause**: File locked by antivirus/Windows Defender, or compilation failure
- **Symptoms**:
  - griffin.obj is 0 bytes
  - LINK : fatal error LNK1104: cannot open file 'libgemma.dir\Release\griffin.obj'
- **Solutions Attempted**:
  - Kill all build processes: `taskkill //F //IM MSBuild.exe`
  - Remove and rebuild: File remains locked
  - Single-threaded build: Still fails
- **Potential Workarounds**:
  - Temporarily exclude build directory from Windows Defender
  - Restart system to release file locks
  - Build in WSL2 instead of native Windows
  - Comment out griffin.cc from CMakeLists.txt temporarily

## Next Steps

1. ✅ Model weights verified
2. ✅ Build environment configured
3. 🔄 Build in progress
4. ⏳ Test execution with models
5. ⏳ Performance benchmarking
6. ⏳ Integration with other tools

## Quick Reference Commands

```bash
# Check model files
ls -la /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/

# Monitor build progress
cmake --build build --config Release -j 1

# Run with default model
build/Release/gemma.exe \
  --weights C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --tokenizer C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm \
  --verbosity 1

# Run benchmarks
build/Release/single_benchmark.exe --weights [model] --tokenizer [tokenizer]
```

## Session Recovery
If returning to this task in a new session:
1. Navigate to `c:\codedev\llm\gemma\gemma.cpp`
2. Check build status: `ls -la build/Release/`
3. If incomplete, resume build: `cmake --build build --config Release -j 1`
4. Test with available model in `.models` directory

---
*Last Updated: 2025-09-16*
*Task ID: gemma-cpp-build*