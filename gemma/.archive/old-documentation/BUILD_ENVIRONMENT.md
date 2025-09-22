# Gemma.cpp Build Environment Documentation

## Discovered Build Tools and Frameworks

### Primary Build Tools Location
- **Path**: `C:\Users\david\.local\bin`
- **Contents**: Complete build toolchain with 200+ tools

### C++ Compilers Available

#### Microsoft Visual Studio (MSVC)
- **Compiler**: `cl.exe`, `link.exe`, `lib.exe`, `dumpbin.exe`
- **Build System**: `msbuild.exe`, `nmake.exe`
- **Location**: Available in PATH via `.local\bin`
- **Usage**: Primary compiler for Windows builds

#### Intel oneAPI Compilers
- **DPC++ Compiler**: `dpcpp.exe` - SYCL compiler for Intel GPUs/NPUs
- **Intel C++**: `icpx.exe`, `icx.exe`
- **Intel Clang**: `intel-clang++.exe`, `intel-clang.exe`
- **Location**: `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin`
- **Version**: oneAPI 2025.1
- **Features**: SYCL support, Intel GPU/NPU acceleration

#### LLVM/Clang
- **Compilers**: `clang++.exe`, `clang.exe`
- **Android**: `android-clang++.exe`, `android-clang.exe`
- **Location**: Available in `.local\bin`

#### MinGW GCC
- **Compilers**: `mingw-g++.exe`, `mingw-gcc.exe`
- **Tools**: `mingw-ar.exe`, `mingw-ld.exe`, `mingw-make.exe`
- **Location**: Available in `.local\bin`

#### ARM Cross-Compiler
- **Toolchain**: `arm-none-eabi-gcc.exe`, `arm-none-eabi-g++.exe`
- **Debugger**: `arm-none-eabi-gdb.exe`
- **Location**: Available in `.local\bin`

### Build Systems

#### CMake
- **Version**: 4.1.1
- **Location**: `C:\Program Files\CMake\bin\cmake.exe`
- **Tools**: `cmake.exe`, `cmake-gui.exe`, `ctest.exe`, `cpack.exe`

#### Ninja
- **Location**: `C:\Users\david\.local\bin\ninja.exe`
- **Usage**: Fast incremental builds

#### Meson
- **Location**: `C:\Users\david\.local\bin\meson.exe`
- **Usage**: Alternative build system

#### Cargo (Rust)
- **Location**: `C:\Users\david\.local\bin\cargo.exe`
- **Extensions**: cargo-make, cargo-nextest, cargo-watch

### SDK and Framework Locations

#### NVIDIA CUDA Toolkit
- **Version**: 13.0 (also 12.1, 12.6, 12.8, 12.9 available)
- **Location**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- **Components**:
  - NVCC compiler
  - cuBLAS, cuDNN libraries
  - CUDA runtime and driver APIs
  - Nsight development tools

#### Intel oneAPI
- **Version**: 2025.1 (also 2025.0 available)
- **Location**: `C:\Program Files (x86)\Intel\oneAPI`
- **Components**:
  - DPC++ Compiler (SYCL)
  - Intel MKL (Math Kernel Library)
  - Intel VTune Profiler
  - Intel Advisor
  - Intel DNN Library
  - Intel MPI
  - Intel TBB (Threading Building Blocks)
  - Intel IPP (Integrated Performance Primitives)

#### Intel NPU Software
- **Location**: `C:\Program Files\Intel\Intel(R) NPU Software & Drivers`
- **Purpose**: Neural Processing Unit acceleration

#### OpenVINO
- **Version**: 2024.6.0
- **Location**: `C:\Program Files (x86)\Intel\openvino_2024.6.0`
- **Purpose**: Deep learning inference optimization

### Package Managers

#### vcpkg
- **Location**: `C:\codedev\vcpkg`
- **Executable**: `C:\Users\david\.local\bin\vcpkg.exe`
- **Usage**: C++ library management

#### Python Package Manager
- **Python**: 3.11
- **Location**: `C:\Users\david\.local\bin\python3.11.exe`
- **UV**: `uv.exe` - Fast Python package installer

#### Node.js
- **Executables**: `node.exe`, `npm.cmd`, `npx.cmd`
- **Package Manager**: Bun also available (`bun.exe`)

### Development Tools

#### Code Quality
- **Formatters**: `clang-format`, `rustfmt.exe`, `biome.exe`
- **Linters**: `ruff.exe`, `clippy-driver.exe`
- **Documentation**: `doxygen.exe`

#### Performance Tools
- **Intel VTune**: `vtune.exe`
- **Intel Advisor**: `advisor.exe`
- **Profiling**: Available through oneAPI

#### Debugging
- **GDB**: `gdb.exe`, `gdbserver.exe`
- **Rust GDB**: `rust-gdb.exe`, `rust-lldb.exe`
- **ARM Debug**: `arm-none-eabi-gdb.exe`

### MCP Servers and Tools
- **Desktop Commander**: `desktop-commander-mcp.exe`
- **Rust Sequential Thinking**: `rust-sequential-thinking.exe`
- **Rust Link**: `rust-link.exe`
- **Python FileOps**: `python-fileops-mcp.exe`
- **Other MCP servers in**: `C:\codedev\mcp_servers`

### Optimization Tools
- **ccache**: `ccache.exe` - Compiler cache
- **sccache**: `sccache.exe` - Shared compilation cache
- **fast-watcher**: `fast-watcher.exe` - File watching

### Environment Configuration Scripts
- **Load All Environments**: `load-all-envs.cmd`
- **Intel Environment**: `load-intel-env.cmd`
- **MSVC Environment**: `load-msvc-env.cmd`
- **MSYS2 Environment**: `load-msys2-env.cmd`

## Build Configurations

### Windows Native Build (Recommended)
```bash
# Set up environment
export PATH="/c/Program Files/CMake/bin:$PATH"
export PATH="/c/Users/david/.local/bin:$PATH"

# Configure with all backends
cmake -B build -G "Visual Studio 17 2022" -T v143 \
  -DGEMMA_BUILD_MCP_SERVER=ON \
  -DGEMMA_BUILD_BACKENDS=ON \
  -DGEMMA_BUILD_CUDA_BACKEND=ON \
  -DGEMMA_BUILD_SYCL_BACKEND=ON \
  -DCMAKE_CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"

# Build
cmake --build build --config Release -j 8
```

### Intel SYCL Build
```bash
# Source Intel oneAPI environment
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

# Configure with SYCL
cmake -B build-sycl -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGEMMA_BUILD_SYCL_BACKEND=ON

# Build
ninja -C build-sycl
```

### CUDA Build
```bash
# Ensure CUDA is in PATH
export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
export PATH="$CUDA_PATH/bin:$PATH"

# Configure with CUDA
cmake -B build-cuda -G Ninja \
  -DGEMMA_BUILD_CUDA_BACKEND=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"

# Build
ninja -C build-cuda
```

### Cross-Platform Build with vcpkg
```bash
# Install dependencies via vcpkg
cd C:/codedev/vcpkg
./vcpkg install nlohmann-json sentencepiece highway

# Configure with vcpkg toolchain
cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=C:/codedev/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
ninja -C build
```

## Environment Variables

### Essential Paths
```batch
set PATH=C:\Users\david\.local\bin;%PATH%
set PATH=C:\Program Files\CMake\bin;%PATH%
set CMAKE_PREFIX_PATH=C:\codedev\vcpkg\installed\x64-windows
```

### Intel oneAPI
```batch
set ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
set SETVARS_CONFIG=C:\Program Files (x86)\Intel\oneAPI\setvars-config.txt
call "%ONEAPI_ROOT%\setvars.bat"
```

### CUDA
```batch
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATH_V13_0=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%
```

## Testing Infrastructure

### Available Test Runners
- **CTest**: CMake's test runner
- **Google Test**: Available via CMake FetchContent
- **Cargo Test**: For Rust components
- **pytest**: For Python components

### Performance Testing
- **Google Benchmark**: Via CMake FetchContent
- **Intel VTune**: For profiling
- **NVIDIA Nsight**: For CUDA profiling

## Docker Support
While no Docker executable was found in `.local\bin`, Docker Desktop can be installed separately for containerization needs.

## Notes
- All tools in `C:\Users\david\.local\bin` are immediately available in PATH
- Multiple compiler toolchains allow for cross-platform development
- Intel oneAPI provides comprehensive SYCL support for Intel GPUs/NPUs
- CUDA 13.0 is the recommended version for NVIDIA GPU acceleration
- vcpkg simplifies dependency management for C++ libraries