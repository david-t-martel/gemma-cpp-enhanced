# Gemma.cpp Build Guide

## Quick Start

### Recommended: WSL Build (Windows users)
```bash
# From WSL Ubuntu
cd gemma/gemma.cpp
cmake --preset make && cmake --build --preset make -j $(nproc)
```

### Windows Native Build
```powershell
# Use the unified PowerShell build script
.\build_intel_oneapi.ps1 -BuildType Release
```

### Linux/macOS Build
```bash
cmake -B build -G "Unix Makefiles"
cmake --build build -j $(nproc)
```

## Build System Options

The consolidated CMakeLists.txt supports these options:

### Performance Options
- `GEMMA_ENABLE_LTO=ON` - Link Time Optimization (default: ON)
- `GEMMA_ENABLE_PCH=OFF` - Precompiled Headers (default: OFF)
- `GEMMA_USE_SCALAR_FALLBACK=ON` - SIMD fallbacks (default: ON)

### Intel OneAPI Options
- `GEMMA_USE_INTEL_ONEAPI=OFF` - Full Intel OneAPI suite
- `GEMMA_USE_INTEL_MKL=OFF` - Intel MKL for BLAS
- `GEMMA_USE_INTEL_TBB=OFF` - Intel Threading Building Blocks

### Build Variants
- `GEMMA_MINIMAL_BUILD=OFF` - Minimal build with fewer dependencies
- `GEMMA_DISABLE_GRIFFIN=OFF` - Disable Griffin support
- `GEMMA_OPTIMIZED_BUILD=OFF` - Enable all optimizations

## Prerequisites

### Windows
- Visual Studio 2022 with C++ toolchain
- CMake 3.16+
- Git
- Optional: Intel OneAPI toolkit

### Linux/WSL
- GCC 9+ or Clang 12+
- CMake 3.16+
- Git
- Development packages: `build-essential cmake git`

## Build Scripts

### Primary Build Scripts
- `build_intel_oneapi.ps1` - Intel OneAPI optimized builds
- `run_gemma_wsl.ps1` - WSL execution wrapper
- `test_intel_oneapi.ps1` - Intel OneAPI testing

### Legacy Scripts (archived)
Multiple `.bat` variants have been consolidated into PowerShell equivalents.

## Troubleshooting

### Common Issues

**Griffin compilation errors on Windows:**
- Use `-DGEMMA_DISABLE_GRIFFIN=ON`
- Or switch to WSL build

**Highway SIMD issues:**
- Use `-DGEMMA_USE_SCALAR_FALLBACK=ON`
- Check CPU feature support

**Intel OneAPI issues:**
- Ensure OneAPI environment is loaded
- Use `load-intel-env.ps1` script

### Build Directory Cleanup

Build artifacts are stored in:
- `build_wsl/` - Working WSL build (preserved)
- `build/` - Default build directory
- Archived: Multiple experimental build directories moved to `/archive/`

## Performance Notes

### Benchmark Results
- Intel OneAPI: ~2x performance improvement on Intel CPUs
- WSL performance: 90-95% of native Linux
- Scalar fallback: ~30% performance reduction but universal compatibility

### Memory Usage
- Release builds: ~500MB-1GB depending on model
- Debug builds: 2-3x memory usage

## Integration

### Python Integration
```python
# Use the consolidated Python wrapper
from gemma_cli import GemmaInterface
gemma = GemmaInterface(model_path="path/to/model")
```

### MCP Server
The Gemma MCP server provides standardized model access.

## Architecture Support

- **Windows x64**: Full support with Visual Studio
- **Linux x64**: Full support with GCC/Clang
- **WSL**: Recommended for Windows development
- **Intel CPUs**: Optimized with OneAPI
- **AMD CPUs**: Standard builds work well

---

*This document consolidates information from BUILD_INSTRUCTIONS.md, BUILD_ENVIRONMENT.md, BUILD_SYSTEM_SUMMARY.md, and BUILD_OPTIMIZATION_GUIDE.md*