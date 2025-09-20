# Gemma.cpp Hardware Acceleration Backends

This directory contains hardware acceleration backends for Gemma.cpp, providing GPU and specialized hardware support for high-performance inference.

## Available Backends

### CUDA Backend
- **Location**: `cuda/`
- **Requirements**: NVIDIA GPU with CUDA 11.0+
- **Features**:
  - Multi-GPU support
  - Tensor Core utilization
  - cuBLAS and cuDNN integration
  - Flash Attention v2
  - Memory pooling with buddy allocator
  - INT8/INT4 quantization support

### Intel SYCL Backend
- **Location**: `sycl/`
- **Requirements**: Intel oneAPI 2023.0+
- **Supported Hardware**: Intel GPUs (Arc, Flex, Max), NPUs (Core Ultra)
- **Features**:
  - oneAPI DPC++ compiler support
  - oneMKL optimized linear algebra
  - USM (Unified Shared Memory)
  - Intel GPU/NPU acceleration

### Vulkan Backend
- **Location**: `vulkan/`
- **Requirements**: Vulkan 1.2+
- **Features**:
  - Cross-platform GPU acceleration
  - Compute shaders for matrix operations
  - Descriptor sets for efficient memory management
  - Pipeline caching

### OpenCL Backend
- **Location**: `opencl/`
- **Requirements**: OpenCL 2.0+
- **Features**:
  - Cross-platform GPU acceleration
  - Kernel compilation caching
  - Multi-device support

### Metal Backend (macOS)
- **Location**: `metal/`
- **Requirements**: macOS 10.15+, Metal Performance Shaders
- **Features**:
  - Apple Silicon optimization
  - Metal Performance Shaders integration
  - Unified memory architecture support

## Architecture

### Backend Interface
All backends implement the `BackendInterface` abstract class which provides:
- Memory management (allocate, free, copy operations)
- Matrix operations (GEMM, matrix-vector multiply)
- Activation functions (ReLU, GELU, Softmax)
- Attention computation
- Performance monitoring

### Backend Registry
The `BackendRegistry` manages backend discovery, creation, and selection:
- Automatic backend detection based on hardware availability
- Priority-based backend selection
- Capability-based backend matching
- Runtime backend switching

### Backend Manager
The `BackendManager` provides high-level backend management:
- Automatic initialization with optimal backend selection
- Performance benchmarking
- Configuration management
- Error handling and fallback

## Usage

### Basic Usage

```cpp
#include "backends/backend_manager.h"

using namespace gemma::backends;

// Initialize with auto-detection
BackendConfig config;
config.preferred_backend = "auto";  // or "CUDA", "SYCL", etc.
config.enable_fallback = true;

BackendManager manager(config);
if (!manager.Initialize()) {
    std::cerr << "Failed to initialize backends" << std::endl;
    return -1;
}

// Get active backend
auto* backend = manager.GetActiveBackend();
if (!backend) {
    std::cerr << "No backend available" << std::endl;
    return -1;
}

// Use backend for operations
auto buffer_a = backend->AllocateBuffer(1024 * sizeof(float));
auto buffer_b = backend->AllocateBuffer(1024 * sizeof(float));
auto buffer_c = backend->AllocateBuffer(1024 * sizeof(float));

// Perform matrix multiplication
backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, 32, 32, 32);
backend->Synchronize();

// Cleanup
backend->FreeBuffer(buffer_a);
backend->FreeBuffer(buffer_b);
backend->FreeBuffer(buffer_c);
manager.Shutdown();
```

### Advanced Configuration

```cpp
BackendConfig config;
config.preferred_backend = "CUDA";
config.enable_fallback = true;
config.disabled_backends = {"OpenCL"};  // Disable specific backends
config.enable_benchmarking = true;       // Run initial benchmarks
config.verbose_logging = true;           // Enable detailed logging
config.prefer_gpu = true;                // Prefer GPU over CPU
config.min_memory_gb = 4;                // Minimum memory requirement

BackendManager manager(config);
manager.Initialize();

// Get performance information
auto performance = manager.GetBackendPerformance();
for (const auto& perf : performance) {
    std::cout << "Backend: " << perf.name
              << ", Matrix Multiply: " << perf.matrix_multiply_gflops << " GFLOPS"
              << std::endl;
}
```

### Backend Switching

```cpp
BackendManager manager;
manager.Initialize();

// Switch to specific backend
if (manager.SwitchBackend("SYCL")) {
    std::cout << "Switched to SYCL backend" << std::endl;
}

// Check capabilities
std::vector<BackendCapability> required = {
    BackendCapability::MATRIX_MULTIPLICATION,
    BackendCapability::ATTENTION_COMPUTATION
};

if (manager.SupportsCapabilities(required)) {
    std::cout << "Backend supports required capabilities" << std::endl;
}
```

## Building

### Prerequisites

1. **CMake 3.16+**
2. **C++20 compiler** (MSVC 2022, GCC 11+, Clang 14+)

### Backend-Specific Requirements

#### CUDA
- CUDA Toolkit 11.0+ (tested with 12.0+)
- cuBLAS, cuDNN (optional)

#### SYCL
- Intel oneAPI Base Toolkit 2023.0+
- Intel oneAPI DPC++ Compiler
- oneMKL (included with oneAPI)

#### Vulkan
- Vulkan SDK 1.2+
- SPIRV-Tools (for shader compilation)

#### OpenCL
- OpenCL SDK 2.0+
- Platform-specific drivers

### Build Options

```bash
# Configure with auto-detection (recommended)
cmake -B build -DGEMMA_AUTO_DETECT_BACKENDS=ON -DGEMMA_BUILD_BACKENDS=ON

# Or enable specific backends
cmake -B build \
  -DGEMMA_BUILD_CUDA_BACKEND=ON \
  -DGEMMA_BUILD_SYCL_BACKEND=ON \
  -DGEMMA_BUILD_VULKAN_BACKEND=ON

# Build
cmake --build build --config Release -j 8
```

### Environment Setup

#### Windows with Intel oneAPI
```cmd
# Source oneAPI environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

#### Linux with Intel oneAPI
```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh
```

## Testing

### Unit Tests
```bash
# Run backend-specific tests
./build/test_backend_system
./build/test_cuda_backend      # If CUDA enabled
./build/test_sycl_backend      # If SYCL enabled
./build/test_vulkan_backend    # If Vulkan enabled
```

### Benchmarking
```bash
# Run comprehensive benchmarks
./build/benchmark_backends

# Run with custom parameters
./build/benchmark_backends --iterations 20 --output results.json
```

### Integration Tests
```bash
# Test full inference pipeline with backends
./build/test_inference_backends
```

## Performance Expectations

### Matrix Multiplication (2048x2048)
- **CUDA (RTX 4060)**: ~8,000 GFLOPS
- **SYCL (Intel Arc A770)**: ~4,000 GFLOPS
- **Vulkan (RTX 4060)**: ~6,000 GFLOPS
- **CPU (i7-12700K)**: ~400 GFLOPS

### Memory Bandwidth
- **CUDA (RTX 4060)**: ~350 GB/s
- **SYCL (Intel Arc A770)**: ~200 GB/s
- **Vulkan**: ~300 GB/s
- **CPU DDR4-3200**: ~50 GB/s

*Performance varies significantly based on hardware, drivers, and problem size.*

## Troubleshooting

### Common Issues

#### Backend Not Detected
- Verify SDK installation and environment variables
- Check hardware compatibility
- Ensure drivers are up to date

#### CUDA Issues
- Verify CUDA_PATH environment variable
- Check NVIDIA driver version compatibility
- Test with `nvidia-smi` command

#### SYCL Issues
- Source oneAPI environment with `setvars.bat/sh`
- Check Intel GPU driver installation
- Verify with `sycl-ls` command

#### Vulkan Issues
- Install latest Vulkan SDK
- Update graphics drivers
- Test with `vulkaninfo` command

### Debug Mode
```bash
# Build with debug information
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGEMMA_BUILD_BACKENDS=ON
cmake --build build

# Run with verbose logging
./build/gemma --backend CUDA --verbose
```

### Environment Validation
```bash
# Check backend availability
./build/backend_test --list-backends

# Validate specific backend
./build/backend_test --test CUDA --verbose
```

## Integration with Gemma.cpp

### Automatic Integration
The backend system automatically integrates with Gemma.cpp's inference pipeline:

```cpp
// In gemma.cc - backends are automatically used for operations
void Gemma::Generate(...) {
    // Matrix operations automatically use active backend
    MatMul(weights, activations, outputs);  // -> Backend::MatrixMultiply

    // Attention computation uses backend if available
    ComputeAttention(q, k, v, output);      // -> Backend::ComputeAttention

    // Activation functions use backend acceleration
    ApplyGELU(hidden, hidden);              // -> Backend::ApplyGELU
}
```

### Manual Backend Selection
```cpp
// Force specific backend
BackendConfig config;
config.preferred_backend = "CUDA";
InitializeBackends(config);

// Create Gemma instance (will use active backend)
Gemma gemma(model_config);
```

## Contributing

### Adding New Backends
1. Create backend directory: `backends/new_backend/`
2. Implement `BackendInterface`
3. Add CMake configuration
4. Register in `BackendManager::RegisterAllBackends()`
5. Add tests and documentation

### Performance Optimization
- Profile with backend-specific tools (nsys for CUDA, VTune for SYCL)
- Implement backend-specific optimizations
- Add benchmarks for new operations
- Consider memory access patterns and cache usage

### Testing Guidelines
- Add unit tests for new features
- Include integration tests with Gemma.cpp
- Benchmark performance against reference implementations
- Test error handling and edge cases

## License

Same as Gemma.cpp project license.