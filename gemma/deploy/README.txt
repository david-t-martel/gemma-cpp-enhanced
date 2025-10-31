# Gemma.exe Standalone Deployment

This directory contains gemma.exe with Intel oneAPI optimizations and all required
runtime libraries for standalone execution.

## Package Contents

- gemma.exe: Main inference engine executable
- gemma.config.toml: Configuration file (optional but recommended)
- DEPLOYMENT_GUIDE.md: Comprehensive usage documentation
- examples/: Example session files
- *.dll: Intel oneAPI runtime libraries

## Quick Start

### 1. Download Model Files

Download Gemma model weights from Kaggle:
https://www.kaggle.com/models/google/gemma-2/gemmaCpp

Recommended starter model:
- File: gemma2-2b-it-sfp.sbs (~2.5GB)
- Tokenizer: tokenizer.spm (~4MB)

### 2. Configure Model Paths

Edit `gemma.config.toml` (or create at `%USERPROFILE%\.gemma\config.toml`):

```toml
[model]
weights = "C:/path/to/your/gemma2-2b-it-sfp.sbs"
tokenizer = "C:/path/to/your/tokenizer.spm"
```

### 3. Run Gemma

**Basic usage (uses config file)**:
```
.\gemma.exe
```

**With session management**:
```
.\gemma.exe --session my_chat --save_on_exit
```

**Override config settings**:
```
.\gemma.exe --weights C:\models\model.sbs --tokenizer C:\models\tokenizer.spm
```

## Session Management Features ✨ NEW

Start interactive sessions with persistent conversation history:

```batch
:: Start new session with auto-save
.\gemma.exe --session work_chat --save_on_exit

:: Resume existing session
.\gemma.exe --session work_chat --load_session
```

**Interactive Commands** (type in the prompt):
- `%q` - Quit session
- `%c` - Clear/reset session history
- `%s [filename]` - Save session (default: session_<id>.json)
- `%l [filename]` - Load session from file
- `%h [N]` - Show last N conversation messages (default: 10)
- `%i` - Show session statistics (turns, tokens, cache size)
- `%m` - List all managed sessions

Session files are JSON format, human-readable and editable.

## Configuration Profiles

Use pre-defined profiles for common use cases:

```batch
.\gemma.exe --profile fast      # Quick responses (lower quality)
.\gemma.exe --profile balanced  # Default balanced mode
.\gemma.exe --profile quality   # Best quality (slower)
.\gemma.exe --profile creative  # Creative writing
.\gemma.exe --profile coding    # Code generation
```

See `gemma.config.toml` for profile details and customization.

## Intel oneAPI Runtime Libraries

Required DLLs (included in this package):
- libiomp5md.dll - OpenMP runtime
- svml_dispmd.dll - Short Vector Math Library
- libmmd.dll - Math library
- sycl7.dll - SYCL runtime (GPU acceleration)
- pi_level_zero.dll - Intel Level Zero plugin
- pi_opencl.dll - OpenCL plugin

## System Requirements

**Minimum**:
- Windows 10/11 (64-bit)
- Intel/AMD CPU with AVX2 support
- 8GB RAM
- 10GB disk space (for models)

**Recommended**:
- Windows 11
- Intel CPU with AVX-512 support
- 16GB+ RAM
- Intel GPU (Arc, Iris Xe, UHD) for SYCL acceleration
- SSD storage

**Performance Notes**:
- SFP format models run ~2x faster than standard formats
- First query is slower (model loading + auto-tuning)
- Subsequent queries benefit from optimizations
- Intel oneAPI build is 15-35% faster than MSVC

## Documentation

See `DEPLOYMENT_GUIDE.md` for:
- Detailed configuration options
- Advanced sampling parameters
- Performance tuning guide
- Troubleshooting common issues
- Complete command-line reference

## Troubleshooting

**"DLL not found" errors**:
- Ensure Intel GPU drivers are installed (for SYCL acceleration)
- Install Visual C++ Redistributables 2022 (x64)
- The executable will fall back to CPU if GPU is unavailable

**"Failed to load model"**:
- Verify file paths in config are correct
- Check file exists: `dir C:\path\to\model.sbs`
- Ensure you have read permissions

**Out of memory**:
- Reduce `seq_len` in config (try 2048 instead of 4096)
- Use smaller model (2B instead of 4B)
- Close other applications

## Support Resources

- Deployment Guide: See DEPLOYMENT_GUIDE.md
- Model Downloads: https://www.kaggle.com/models/google/gemma-2/gemmaCpp
- Source Repository: https://github.com/google/gemma.cpp

## Build Information

Built with Intel oneAPI 2025.1
- Compiler: Intel ICX with AVX2/AVX-512 optimizations
- Math Libraries: MKL (parallel), IPP (parallel)
- Threading: TBB, OpenMP
- GPU Acceleration: SYCL for Intel Arc/Iris/UHD GPUs
