# Vulkan Backend for Gemma.cpp

A high-performance, cross-platform GPU acceleration backend for Gemma.cpp using Vulkan compute shaders.

## Features

- **Cross-platform GPU support**: Works on Windows, Linux, macOS with any Vulkan-compatible GPU (NVIDIA, AMD, Intel)
- **High-performance compute shaders**: Optimized GLSL shaders for key operations
- **Memory management**: Efficient buffer allocation and management with optional VMA integration
- **Pipeline caching**: Automatic caching of compiled compute pipelines for improved startup times
- **Validation layers**: Comprehensive debugging support during development
- **Multi-device support**: Automatic device selection with capability scoring

## Supported Operations

### Matrix Operations
- **Matrix Multiplication**: Tiled matrix multiplication with shared memory optimization
- **Matrix-Vector Multiplication**: Optimized for common neural network patterns

### Neural Network Operations
- **Attention Computation**: Multi-head attention with scaled dot-product
- **Activation Functions**: ReLU, GELU, Softmax, Swish/SiLU, Tanh

### Memory Operations
- **Buffer Management**: Device-local and host-visible buffer allocation
- **Memory Transfer**: Efficient host-to-device and device-to-host transfers
- **Synchronization**: Command buffer submission and synchronization

## Architecture

```
backends/vulkan/
├── vulkan_backend.h/.cpp     # Main backend implementation
├── vulkan_utils.h/.cpp       # Utility functions and helpers
├── vulkan_test.cpp          # Comprehensive test suite
├── CMakeLists.txt           # Build configuration
├── shaders/                 # GLSL compute shaders
│   ├── matmul.comp         # Matrix multiplication
│   ├── attention.comp      # Multi-head attention
│   ├── activation.comp     # Generic activation functions
│   ├── relu.comp           # Optimized ReLU
│   ├── gelu.comp           # Optimized GELU
│   └── softmax.comp        # Numerically stable softmax
└── README.md               # This file
```

### Core Classes

- **VulkanBackend**: Main backend interface implementation
- **VulkanDevice**: Logical device abstraction with queue management
- **VulkanBuffer**: Memory buffer wrapper with mapping support
- **VulkanPipeline**: Compute pipeline management
- **VulkanCommandPool**: Command buffer allocation and management

## Build Requirements

### Prerequisites
- **Vulkan SDK**: Version 1.2 or later
- **CMake**: Version 3.16 or later
- **C++ Compiler**: C++20 support required
- **glslangValidator**: For shader compilation (included with Vulkan SDK)

### Optional Dependencies
- **Vulkan Memory Allocator (VMA)**: For advanced memory management
- **Validation layers**: For debugging (included with Vulkan SDK)

### Build Configuration

```bash
# Enable Vulkan backend
cmake -DGEMMA_BUILD_VULKAN_BACKEND=ON ..

# With validation layers (Debug builds)
cmake -DCMAKE_BUILD_TYPE=Debug -DGEMMA_BUILD_VULKAN_BACKEND=ON ..

# With VMA support
cmake -DGEMMA_BUILD_VULKAN_BACKEND=ON -DVMA_INCLUDE_DIR=/path/to/vma ..
```

## Usage

### Basic Initialization

```cpp
#include "backends/vulkan/vulkan_backend.h"

using namespace gemma::backends;

// Create and initialize backend
auto backend = std::make_unique<vulkan::VulkanBackend>();
if (!backend->Initialize()) {
    std::cerr << "Failed to initialize Vulkan backend" << std::endl;
    return false;
}

std::cout << "Backend: " << backend->GetName() << " v" << backend->GetVersion() << std::endl;
std::cout << "Devices: " << backend->GetDeviceCount() << std::endl;
```

### Buffer Operations

```cpp
// Allocate device buffer
size_t buffer_size = 1024 * sizeof(float);
auto device_buffer = backend->AllocateBuffer(buffer_size);

// Prepare host data
std::vector<float> host_data(1024, 1.0f);

// Copy to device
backend->CopyToDevice(device_buffer, host_data.data(), buffer_size);

// Copy from device
std::vector<float> result(1024);
backend->CopyFromDevice(result.data(), device_buffer, buffer_size);

// Synchronize and cleanup
backend->Synchronize();
backend->FreeBuffer(device_buffer);
```

### Matrix Multiplication

```cpp
const int M = 256, N = 256, K = 256;
size_t size_a = M * K * sizeof(float);
size_t size_b = K * N * sizeof(float);
size_t size_c = M * N * sizeof(float);

// Allocate matrices
auto buffer_a = backend->AllocateBuffer(size_a);
auto buffer_b = backend->AllocateBuffer(size_b);
auto buffer_c = backend->AllocateBuffer(size_c);

// ... copy data to buffers ...

// Perform C = A * B
bool success = backend->MatrixMultiply(
    buffer_a, buffer_b, buffer_c,
    M, N, K,
    1.0f, 0.0f  // alpha=1.0, beta=0.0
);

if (success) {
    std::cout << "Matrix multiplication completed successfully" << std::endl;
}
```

### Activation Functions

```cpp
size_t num_elements = 1024;
size_t buffer_size = num_elements * sizeof(float);

auto input_buffer = backend->AllocateBuffer(buffer_size);
auto output_buffer = backend->AllocateBuffer(buffer_size);

// ... copy input data ...

// Apply ReLU activation
backend->ApplyReLU(input_buffer, output_buffer, num_elements);

// Apply GELU activation
backend->ApplyGELU(input_buffer, output_buffer, num_elements);

// Apply Softmax
backend->ApplySoftmax(input_buffer, output_buffer, num_elements);
```

## Performance Optimization

### Workgroup Sizes
- **Matrix multiplication**: 16x16 workgroups with shared memory tiling
- **Activation functions**: 256 threads per workgroup for memory bandwidth
- **Attention**: 16x4 workgroups optimized for attention patterns

### Memory Layout
- **Row-major storage**: Standard layout for compatibility
- **Buffer alignment**: 32-byte alignment for optimal performance
- **Memory coalescing**: Workgroup access patterns optimized for GPU memory

### Pipeline Optimization
- **Specialization constants**: Compile-time optimization for different operation types
- **Descriptor caching**: Reuse descriptor sets across similar operations
- **Command buffering**: Batch multiple operations for reduced CPU overhead

## Compute Shaders

### Matrix Multiplication (`matmul.comp`)
- **Algorithm**: Tiled matrix multiplication with 16x16 tiles
- **Shared memory**: 512 floats per workgroup (16x16x2)
- **Memory access**: Coalesced reads with bank conflict avoidance
- **Optimizations**: Loop unrolling and register blocking

### Attention Computation (`attention.comp`)
- **Implementation**: Scaled dot-product attention
- **Phases**: QK^T computation, softmax normalization, weighted value sum
- **Memory**: Temporary buffer for attention scores
- **Numerical stability**: Max subtraction for softmax

### Activation Functions
- **ReLU**: Simple max(0, x) operation
- **GELU**: Tanh-based approximation for efficiency
- **Softmax**: Numerically stable with max normalization
- **Vectorization**: SIMD-friendly implementations

## Testing

### Test Suite
Run the comprehensive test suite:

```bash
# Build and run tests
cd build/backends/vulkan
./vulkan_test
```

### Test Coverage
- **Device enumeration**: Multi-device support and selection
- **Buffer operations**: Allocation, transfer, synchronization
- **Matrix operations**: Correctness and performance validation
- **Activation functions**: Mathematical accuracy verification
- **Performance benchmarks**: Throughput and latency measurements

### Expected Output
```
=== Vulkan Backend Test Suite ===

--- Initializing Vulkan Backend ---
Backend: Vulkan v1.0.0
Device count: 1
Current device: 0

--- Testing Device Information ---
Found 1 Vulkan devices
Successfully switched to device 0
Capability 0: Supported
...
Device information test: PASSED

--- Testing Buffer Operations ---
Allocated buffer: 4096 bytes
Data copied to device successfully
...
Buffer operations test: PASSED

--- Testing Matrix Multiplication ---
Matrix multiplication (128x128) * (128x128)
Execution time: 245 μs
Performance: 13.2 GFLOPS
...
Matrix multiplication test: PASSED

=== Test Summary ===
Overall result: PASSED
```

## Troubleshooting

### Common Issues

1. **Vulkan not available**
   - Install latest GPU drivers
   - Install Vulkan SDK
   - Check GPU compatibility

2. **Validation layer errors**
   - Disable validation layers for release builds
   - Update to latest Vulkan SDK
   - Check descriptor set bindings

3. **Shader compilation failures**
   - Verify glslangValidator is in PATH
   - Check GLSL syntax
   - Ensure compute capability support

4. **Performance issues**
   - Check GPU memory bandwidth
   - Verify optimal workgroup sizes
   - Profile with GPU tools (NSight, RenderDoc)

### Debug Mode
Enable detailed debugging:
```cpp
// Set environment variable for validation layers
export VK_LAYER_PATH=/path/to/vulkan/layers

// Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug -DGEMMA_BUILD_VULKAN_BACKEND=ON ..
```

## Platform-Specific Notes

### Windows
- **Driver requirements**: Latest NVIDIA/AMD/Intel drivers with Vulkan support
- **Visual Studio**: 2019 or later recommended
- **Runtime libraries**: Vulkan-1.dll must be available

### Linux
- **Packages**: vulkan-tools, vulkan-validationlayers-dev
- **Display**: X11 or Wayland support for device enumeration
- **Permissions**: GPU device access permissions

### macOS
- **MoltenVK**: Required for Vulkan-on-Metal translation
- **SDK**: Install via Vulkan SDK installer
- **Compatibility**: macOS 10.14+ required

## Future Enhancements

### Planned Features
- **Half-precision (FP16)** support for memory bandwidth optimization
- **Tensor operations** with optimized memory layouts
- **Multi-GPU** support with work distribution
- **Async compute** for overlapped execution
- **Memory pooling** for reduced allocation overhead

### Optimization Opportunities
- **Vulkan Memory Allocator** integration for advanced memory management
- **Pipeline state objects** for reduced driver overhead
- **Bindless descriptors** for flexible resource management
- **Subgroup operations** for intra-warp communication

## Contributing

When contributing to the Vulkan backend:

1. **Follow coding standards**: Use the existing code style
2. **Add tests**: Include test cases for new functionality
3. **Document changes**: Update this README and inline documentation
4. **Validate on multiple platforms**: Test Windows, Linux, and macOS
5. **Profile performance**: Ensure changes don't regress performance

### Code Structure
- Keep platform-specific code in utilities
- Use RAII for resource management
- Follow Vulkan best practices
- Minimize state changes in hot paths

## References

- [Vulkan Specification](https://www.khronos.org/registry/vulkan/)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)
- [llama.cpp Vulkan Implementation](https://github.com/ggerganov/llama.cpp/tree/master/ggml-vulkan.cpp)
- [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)