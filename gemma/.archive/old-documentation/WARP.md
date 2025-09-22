# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Quick Start (Windows)

### Prerequisites
```bash
# Install required tools
winget install --id Kitware.CMake
winget install --id Microsoft.VisualStudio.2022.BuildTools --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;installRecommended"
```

### Build
```bash
# WSL (recommended due to Windows build issues)
cd /mnt/c/codedev/llm/gemma/gemma.cpp
cmake --preset make
cmake --build --preset make -j $(nproc)

# Windows native (has issues with griffin.obj)
cd C:\codedev\llm\gemma\gemma.cpp
cmake --preset windows
cmake --build --preset windows -j 4
```

### Run with Models
```bash
# Using Gemma 2B model from C:\codedev\llm\.models
./build/gemma \
  --weights /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --tokenizer /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm
```

## Common Commands

### Build Commands
```bash
# Debug build
cmake --preset make-debug
cmake --build --preset make-debug -j

# Single benchmark
./build/single_benchmark --weights [model.sbs] --tokenizer [tokenizer.spm]

# Comprehensive benchmarks
./build/benchmarks --weights [model.sbs] --tokenizer [tokenizer.spm]

# Run tests (requires model files)
./build/gemma_test --model gemma2-2b-it --tokenizer [path] --weights [path]
```

### Model Operations
```bash
# Convert to single-file format (tokenizer embedded)
./build/migrate_weights \
  --tokenizer [tokenizer.spm] \
  --weights [input.sbs] \
  --output_weights [output-single.sbs]

# PaliGemma vision-language model
./build/gemma \
  --tokenizer /c/codedev/llm/.models/paligemma_tokenizer.model \
  --weights /c/codedev/llm/.models/paligemma2-3b-mix-224-sfp.sbs \
  --image_file image.ppm
```

### Python CLI Interface
```bash
# Interactive chat mode
python gemma-cli.py \
  --model C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs \
  --tokenizer C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm \
  --temperature 0.7

# Enhanced launcher batch script
run_gemma.bat chat --model 2B --temp 0.7
run_gemma.bat prompt "Explain quantum computing" --model 4B
run_gemma.bat bench --model 2B --iterations 100
```

### Download Models
```bash
# From Kaggle (requires authentication)
scripts/download-kaggle-model.sh \
  google/gemma-2/gemmaCpp/gemma2-2b-it-sfp \
  C:/codedev/llm/.models

# Alternative: Use Ollama for quick testing
ollama pull gemma3:1b
ollama run gemma3:1b
```

## Architecture Overview

### Core Components

**gemma/** - Main inference engine
- `gemma.h/cc` - Primary Gemma class and generation API
- `configs.h/cc` - Model configurations (Gemma 2/3, Griffin, PaliGemma)
- `attention.h/cc` - Multi-head attention with FlashAttention-style optimizations
- `griffin.h/cc` - RecurrentGemma/Griffin architecture (SSM)
- `vit.h/cc` - Vision Transformer for multimodal models
- `kv_cache.h/cc` - Key-value cache with NUMA awareness

**compression/** - Weight compression
- SFP (Scaled Float Point) - 8-bit format, 2x speed improvement
- NUQ (Non-Uniform Quantization) - 4-bit compression
- Python SafeTensors conversion in `compression/python/`

**ops/** - Optimized operations
- Highway library for portable SIMD (AVX2/AVX-512/NEON)
- Auto-tuning matrix multiplication with 7 parameters per shape
- Specialized kernels for different data types

**io/** - I/O layer
- Memory-mapped file support with parallel reading
- Custom `.sbs` format with forward/backward compatibility
- Cross-platform abstractions (Windows/Linux/macOS)

### Key Design Patterns

1. **Template-based SIMD** - Highway library for multi-target compilation
2. **Batch processing** - Efficient multi-query with `AllQueries`/`QBatch`
3. **Memory efficiency** - NUMA-aware allocation, streaming support
4. **Type safety** - Compile-time tensor shape checking

### Model Support Matrix

| Model | Parameters | Use Case | Recommended Format |
|-------|-----------|----------|-------------------|
| Gemma2-2B | 2B | Quick testing | `gemma2-2b-it-sfp` |
| Gemma2-9B | 9B | Balanced performance | `gemma2-9b-it-sfp` |
| Gemma2-27B | 27B | Best quality | `gemma2-27b-it` |
| Griffin-2B | 2B | Long sequences | `griffin-2b-it` |
| PaliGemma2-3B | 3B | Vision-language | `paligemma2-3b-mix-224-sfp` |

## Development Workflows

### Adding New Model Support
1. Add configuration to `gemma/configs.cc` `ModelConfig::Create()`
2. Update `Model` enum in `gemma/configs.h`
3. Add tokenizer handling if needed in `gemma/tokenizer.cc`
4. Test with `debug_prompt` utility

### Performance Optimization
1. Use `-sfp` models for 50% size reduction, 2x speed
2. Second/third queries faster due to auto-tuning
3. Set `OMP_NUM_THREADS` to CPU core count
4. Use `prefill_tbatch_size` and `decode_qbatch_size` for batching

### Debugging Issues
```bash
# Debug build with symbols
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

# Verbose output
./build/gemma --weights [model] --tokenizer [tok] --verbosity 3

# Layer-by-layer debugging
./build/debug_prompt --weights [model] --tokenizer [tok] --prompt "test"

# Memory debugging (WSL/Linux)
valgrind --leak-check=full ./build/gemma [args]
```

## Common Issues and Solutions

### Windows Build Issues

**griffin.obj Locking (LNK1104)**
- **Cause**: Windows Defender or antivirus locks file
- **Solution**: Build in WSL or add build directory to antivirus exclusions
```bash
# WSL workaround (recommended)
cd /mnt/c/codedev/llm/gemma/gemma.cpp
cmake --preset make && cmake --build --preset make
```

**ClangCL Not Found**
- **Solution**: Use MSVC toolset instead
```bash
cmake -B build -G "Visual Studio 17 2022" -A x64 -T v143
```

**C++20 Required**
- **Solution**: Set standard explicitly
```bash
cmake -B build -DCMAKE_CXX_STANDARD=20
```

### Runtime Issues

**Model Loading Fails**
- Verify file paths are absolute
- Check file permissions
- Ensure complete download (validate checksums)

**Slow Inference**
- Use SFP models instead of BF16
- Set power plan to High Performance
- Close other CPU-intensive applications
- Allow warm-up (second run is faster)

## Testing

### Unit Tests
```bash
# Run all tests (requires Google Test)
ctest --test-dir build

# Specific test with model
./build/gemma_test --model gemma2-2b-it --weights [path] --tokenizer [path]
```

### Benchmarks
```bash
# Cross-entropy evaluation
./build/single_benchmark --cross_entropy [text_file] --weights [model] --tokenizer [tok]

# TriviaQA evaluation  
./build/single_benchmark --trivia_qa [json_file] --weights [model] --tokenizer [tok]

# Performance benchmarks
./build/benchmarks --weights [model] --tokenizer [tok]
```

## API Usage Examples

### C++ Generation API
```cpp
#include "gemma/gemma.h"

// Basic generation
Gemma gemma(loader, inference, ctx);
KVCache kv_cache(gemma.Config(), inference, ctx.allocator);

auto stream_token = [](int token, float) {
  std::string token_text;
  gemma.Tokenizer().Decode({token}, &token_text);
  std::cout << token_text << std::flush;
  return true;
};

RuntimeConfig config = {
  .max_generated_tokens = 2048,
  .temperature = 0.7f,
  .stream_token = stream_token
};

gemma.Generate(config, prompt_tokens, pos, kv_cache, env, timing_info);
```

### Batch Processing
```cpp
AllQueries queries;
queries.AddQuery(tokens1, kv_cache1);
queries.AddQuery(tokens2, kv_cache2);
gemma.GenerateBatch(runtime_config, queries, env, timing_info);
```

## Configuration Reference

### Runtime Configuration
- `max_seq_len`: Maximum sequence length (32K typical, 128K possible)
- `decode_qbatch_size`: Batch size for decoding (default: 3)
- `prefill_tbatch_size`: Batch size for prefill (default: 64)
- `temperature`: Sampling temperature (0.0-2.0)
- `top_k`: Top-K sampling parameter

### Build Options
- `BUILD_GEMMA_DLL`: Build shared library for C# interop
- `GEMMA_ENABLE_TESTS`: Enable test suite
- `CMAKE_BUILD_TYPE`: Release (default), Debug, RelWithDebInfo

### Environment Variables
- `OMP_NUM_THREADS`: Thread count for OpenMP
- `KMP_AFFINITY`: Thread affinity (e.g., `granularity=fine,compact`)

## References

- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) - Comprehensive build guide
- [CLAUDE.md](CLAUDE.md) - Claude-specific configuration
- [GEMMA_CLI_USAGE.md](GEMMA_CLI_USAGE.md) - Python CLI wrapper documentation
- [GitHub Issues](https://github.com/google/gemma.cpp/issues) - Bug reports and discussions
- [Kaggle Models](https://www.kaggle.com/models/google/gemma-2/gemmaCpp) - Download model weights