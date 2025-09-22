# Gemma.cpp Enhanced Build System Documentation

## Overview

This document describes the modernized build system for the Gemma.cpp project, featuring comprehensive CMake 3.25+ integration, vcpkg dependency management, and Intel oneAPI hardware acceleration support.

## Build System Architecture

### Core Technologies

- **CMake 3.25+**: Modern CMake with target-based approach and generator expressions
- **vcpkg**: C++ dependency management with manifest mode and version constraints
- **Intel oneAPI 2024**: Hardware acceleration via MKL, TBB, IPP, and DPC++
- **Static Linking**: x64-windows-static triplet for Windows compatibility

## Prerequisites

### Required Software

1. **CMake 4.1.1+**
   - Windows: `C:\Program Files\CMake\bin\cmake.exe`
   - Add to PATH: `export PATH="/c/Program Files/CMake/bin:$PATH"`

2. **vcpkg**
   - Installation: `C:\codedev\vcpkg`
   - Bootstrap: `.\bootstrap-vcpkg.bat`
   - Integrate: `.\vcpkg integrate install`

3. **Intel oneAPI 2024** (Optional, for acceleration)
   - Location: `C:\Program Files (x86)\Intel\oneAPI`
   - Components: MKL, TBB, IPP, DPC++ Compiler

4. **Visual Studio 2022**
   - Workload: Desktop development with C++
   - Components: MSVC v143, Windows SDK, CMake tools

## Quick Start

### Windows Build (Recommended Configuration)

```bash
# Set up environment
export PATH="/c/Program Files/CMake/bin:$PATH"

# Configure with modern development preset
cmake --preset dev-modern

# Build
cmake --build --preset dev-modern -j 4

# Run tests
ctest --preset test-default
```

### Windows Build with Intel oneAPI

```bash
# Configure with Intel optimization
cmake --preset windows-intel

# Build with Intel compiler
cmake --build --preset windows-intel -j 4
```

### Alternative: Manual Configuration

```bash
# Create build directory
cmake -B build-custom \
  -G "Visual Studio 17 2022" \
  -T v143 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMMA_ENABLE_INTEL_OPTIMIZATIONS=ON \
  -DGEMMA_WARNINGS_AS_ERRORS=ON

# Build
cmake --build build-custom --config Release -j 4
```

## Build Configuration Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build configuration (Debug/Release/RelWithDebInfo) |
| `GEMMA_ENABLE_PCH` | ON | Enable precompiled headers for faster builds |
| `GEMMA_ENABLE_CCACHE` | ON | Use ccache if available |
| `GEMMA_ENABLE_LTO` | OFF | Link-time optimization (slower build, faster runtime) |
| `GEMMA_ENABLE_UNITY_BUILDS` | OFF | Unity builds for faster compilation |
| `GEMMA_WARNINGS_AS_ERRORS` | ON | Treat compiler warnings as errors |

### Feature Options

| Option | Default | Description |
|--------|---------|-------------|
| `GEMMA_BUILD_TESTS` | ON | Build test suite |
| `GEMMA_BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `GEMMA_BUILD_MCP_SERVER` | OFF | Build Model Context Protocol server |
| `GEMMA_BUILD_BACKENDS` | OFF | Build hardware acceleration backends |
| `GEMMA_AUTO_DETECT_BACKENDS` | ON | Auto-detect available SDKs |

### Intel Optimization Options

| Option | Default | Description |
|--------|---------|-------------|
| `GEMMA_ENABLE_INTEL_OPTIMIZATIONS` | OFF | Enable Intel oneAPI optimizations |
| `GEMMA_USE_INTEL_COMPILER` | OFF | Use Intel DPC++ compiler |
| `GEMMA_USE_INTEL_MKL` | AUTO | Use Intel Math Kernel Library |
| `GEMMA_USE_INTEL_TBB` | AUTO | Use Intel Threading Building Blocks |
| `GEMMA_USE_INTEL_IPP` | AUTO | Use Intel Integrated Performance Primitives |

## Dependency Management

### vcpkg Dependencies

The project uses vcpkg manifest mode (`vcpkg.json`) for automatic dependency management:

```json
{
  "name": "gemma-enhanced",
  "version": "1.0.0",
  "dependencies": [
    {"name": "highway", "version>=": "1.2.0"},
    {"name": "sentencepiece", "version>=": "0.2.0"},
    {"name": "nlohmann-json", "version>=": "3.11.3"},
    {"name": "gtest", "version>=": "1.15.0"},
    {"name": "benchmark", "version>=": "1.9.0"},
    {"name": "tbb", "version>=": "2021.11.0"}
  ]
}
```

### Manual Dependency Installation

If vcpkg integration fails, install dependencies manually:

```bash
cd C:\codedev\vcpkg
.\vcpkg install highway:x64-windows-static
.\vcpkg install sentencepiece:x64-windows-static
.\vcpkg install nlohmann-json:x64-windows-static
.\vcpkg install gtest:x64-windows-static
.\vcpkg install benchmark:x64-windows-static
```

## Build Presets

### Development Presets

- **dev-modern**: Balanced development build with debugging symbols
  - RelWithDebInfo configuration
  - PCH and ccache enabled
  - Tests and benchmarks included
  - Intel optimizations enabled

- **windows-debug**: Full debug build
  - Debug configuration
  - All optimizations disabled
  - Full test suite enabled

- **windows-release**: Production build
  - Release configuration
  - LTO and unity builds enabled
  - Tests disabled for smaller binary

### Specialized Presets

- **windows-intel**: Intel oneAPI optimized build
  - Uses Intel DPC++ compiler
  - MKL, TBB, IPP integration
  - Maximum performance optimizations

- **ci-windows**: Continuous integration build
  - Consistent reproducible builds
  - All tests enabled
  - PCH disabled for compatibility

## Compiler Configuration

### MSVC Compiler Flags

```cmake
/W4           # Warning level 4
/WX           # Warnings as errors
/permissive-  # Standards conformance
/Zc:__cplusplus # Correct __cplusplus macro
/Zc:preprocessor # Standards-conforming preprocessor
/MP           # Multi-processor compilation
/bigobj       # Large object files
```

### Warning Suppressions (Minimal)

```cmake
/wd4100  # Unreferenced formal parameter
/wd4127  # Conditional expression is constant
/wd4244  # Conversion with possible loss of data
/wd4267  # size_t to smaller type conversion
/wd4996  # Deprecated functions
```

### Intel Compiler Flags

```cmake
/Qstd=c++20   # C++20 standard
/QxHost       # Optimize for host architecture
/Qopt-matmul  # Matrix multiplication optimizations
/Qparallel    # Auto-parallelization
/Qvec         # Auto-vectorization
```

## Troubleshooting

### Common Issues and Solutions

#### 1. vcpkg Integration Failure

**Error**: "Could not find a package configuration file provided by..."

**Solution**:
```bash
# Ensure vcpkg is integrated
C:\codedev\vcpkg\vcpkg integrate install

# Set triplet for static linking
set VCPKG_DEFAULT_TRIPLET=x64-windows-static
```

#### 2. Intel oneAPI Not Detected

**Error**: "Intel oneAPI not found at expected location"

**Solution**:
```bash
# Verify installation
dir "C:\Program Files (x86)\Intel\oneAPI"

# Set environment variables
set ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
set MKL_ROOT=%ONEAPI_ROOT%\mkl\latest
```

#### 3. Compilation Warnings as Errors

**Error**: "warning C4267: conversion from 'size_t' to 'int'"

**Solution**: Fix the code to eliminate warnings, or if necessary:
```cmake
# Temporarily disable specific warning
target_compile_options(target PRIVATE /wd4267)
```

#### 4. Static Linking Issues

**Error**: "sentencepiece::SentencePieceProcessor::LoadFromSerializedProto unresolved"

**Solution**:
```bash
# Force static triplet
cmake -B build -DVCPKG_TARGET_TRIPLET=x64-windows-static
```

## Performance Optimization

### Build-Time Optimizations

1. **Precompiled Headers**: 30-40% faster compilation
2. **Unity Builds**: 20-30% faster for large projects
3. **ccache**: 50-80% faster incremental builds
4. **Parallel Compilation**: Use all CPU cores

### Runtime Optimizations

1. **Link-Time Optimization**: 10-20% performance improvement
2. **Intel MKL**: 2-5x faster matrix operations
3. **Intel TBB**: Better thread scalability
4. **Profile-Guided Optimization**: Additional 10-15% improvement

## Testing

### Running Tests

```bash
# All tests
ctest --test-dir build

# Specific test suite
ctest --test-dir build -R unit

# Verbose output
ctest --test-dir build -V

# Parallel execution
ctest --test-dir build -j 8
```

### Test Categories

- **Unit Tests**: Core functionality testing
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Benchmark suite
- **Backend Tests**: Hardware acceleration validation

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: lukka/get-cmake@latest
      - uses: lukka/run-vcpkg@v11
      - name: Configure
        run: cmake --preset ci-windows
      - name: Build
        run: cmake --build --preset ci-windows
      - name: Test
        run: ctest --preset test-windows
```

## Package Installation

### Creating Installation Package

```bash
# Configure with installation prefix
cmake -B build -DCMAKE_INSTALL_PREFIX=C:/gemma-install

# Build and install
cmake --build build --target install

# Create package
cmake --build build --target package
```

### Using Gemma as a Library

```cmake
# In consumer CMakeLists.txt
find_package(Gemma REQUIRED)
target_link_libraries(myapp PRIVATE Gemma::Core)
```

## Migration from Legacy Build

### Key Changes from Original gemma.cpp

1. **Modern CMake Patterns**
   - Target-based approach instead of directory-based
   - Generator expressions for conditional compilation
   - FetchContent replaced ExternalProject

2. **vcpkg Integration**
   - Automatic dependency management
   - Version constraints and features
   - Static linking support

3. **Intel oneAPI Support**
   - Direct integration with installed SDK
   - Compiler and library optimizations
   - Hardware acceleration backends

4. **Enhanced Testing**
   - CTest integration
   - Preset-based test configurations
   - Performance benchmarking

## Best Practices

1. **Always use presets** for consistent builds
2. **Enable warnings as errors** during development
3. **Use static analysis** tools (clang-tidy, PVS-Studio)
4. **Profile before optimizing** with Intel VTune
5. **Document build requirements** in code comments

## Support and Resources

- **Project Repository**: [GitHub - gemma.cpp](https://github.com/google/gemma.cpp)
- **vcpkg Documentation**: [vcpkg.io](https://vcpkg.io)
- **Intel oneAPI**: [intel.com/content/www/us/en/developer/tools/oneapi](https://intel.com/content/www/us/en/developer/tools/oneapi)
- **CMake Documentation**: [cmake.org/documentation](https://cmake.org/documentation)

## Appendix: Environment Variables

```bash
# Core paths
VCPKG_ROOT=C:\codedev\vcpkg
ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI

# Build configuration
VCPKG_DEFAULT_TRIPLET=x64-windows-static
CMAKE_GENERATOR="Visual Studio 17 2022"
CMAKE_GENERATOR_TOOLSET=v143

# Intel oneAPI
MKL_ROOT=%ONEAPI_ROOT%\mkl\latest
TBB_ROOT=%ONEAPI_ROOT%\tbb\latest
IPP_ROOT=%ONEAPI_ROOT%\ipp\latest
DPCPP_ROOT=%ONEAPI_ROOT%\compiler\latest
```

---

*Last Updated: January 2025*
*Build System Version: 2.0.0*
*Compatible with: gemma.cpp enhanced edition*