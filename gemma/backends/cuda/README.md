# NVIDIA CUDA Backend for Gemma.cpp

A production-ready CUDA backend implementation for Gemma.cpp providing GPU acceleration for inference operations with advanced optimization features.

## Features

### Core Functionality
- **Multi-GPU Support**: Automatic device detection and management
- **Mixed Precision**: FP32, FP16, BF16, INT8, and INT4 quantization
- **Memory Management**: Advanced memory pooling with buddy allocation
- **Stream Management**: Asynchronous execution with optimal load balancing
- **Kernel Optimization**: Auto-tuning and performance profiling

### Advanced Optimizations
- **Flash Attention v2**: Memory-efficient attention computation
- **Tensor Core Utilization**: Automatic use of Tensor Cores on supported hardware
- **Kernel Fusion**: Optimized fused operations for reduced memory overhead
- **Dynamic Kernel Selection**: Runtime optimization based on input characteristics
- **Cooperative Kernel Launch**: Advanced GPU utilization features

### Memory Management
- **Pooled Allocation**: Buddy system and slab allocators
- **Unified Memory**: CUDA unified memory support
- **Memory Prefetching**: Async data movement optimization
- **NUMA Awareness**: Multi-GPU memory locality optimization

## Architecture

```
backends/cuda/
├── cuda_backend.h/.cpp          # Main backend interface implementation
├── cuda_kernels.h/.cu           # Optimized CUDA kernels
├── cuda_attention.h/.cu         # Flash Attention v2 implementation
├── cuda_memory.h/.cpp           # Advanced memory management
├── cuda_stream_manager.h/.cpp   # Stream management and load balancing
├── cuda_kernel_launcher.h/.cpp  # Kernel optimization and auto-tuning
└── CMakeLists.txt              # Build configuration
```

## Requirements

### Hardware Requirements
- **NVIDIA GPU**: Compute capability 7.0+ (Volta architecture or newer)
- **Memory**: Minimum 4GB GPU memory, 8GB+ recommended
- **Multi-GPU**: Optional, automatic detection and utilization

### Software Requirements
- **CUDA Toolkit**: 11.0 or newer (12.0+ recommended)
- **cuBLAS**: Included with CUDA Toolkit
- **cuDNN**: 8.0+ (optional but recommended)
- **NCCL**: For multi-GPU communication (optional)

### Supported CUDA Architectures
- **Volta (7.0, 7.2)**: V100, Titan V
- **Turing (7.5)**: RTX 20 series, GTX 16 series, T4
- **Ampere (8.0, 8.6, 8.9)**: RTX 30/40 series, A100, A10
- **Hopper (9.0)**: H100
- **Auto-detection**: Automatically detects available architectures

## Build Configuration

### Basic Build
```bash
mkdir build && cd build
cmake .. -DGEMMA_ENABLE_CUDA=ON
cmake --build . -j$(nproc)
```

### Advanced Build Options
```bash
cmake .. \
  -DGEMMA_ENABLE_CUDA=ON \
  -DGEMMA_ENABLE_CUDA_MEMORY_DEBUG=ON \
  -DGEMMA_ENABLE_CUDA_PROFILING=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
```

### Build Options
- `GEMMA_ENABLE_CUDA_MEMORY_DEBUG`: Enable memory debugging and leak detection
- `GEMMA_ENABLE_CUDA_PROFILING`: Enable kernel profiling and performance analysis
- `CMAKE_CUDA_ARCHITECTURES`: Specify target GPU architectures

## Usage

### Basic Usage
```cpp
#include "backends/cuda/cuda_backend.h"

using namespace gemma::backends::cuda;

// Create CUDA backend with default configuration
auto backend = CreateCudaBackend();

// Initialize the backend
if (!backend->Initialize()) {
    // Handle initialization failure
}

// Allocate GPU memory
auto buffer = backend->AllocateBuffer(size);

// Perform matrix multiplication
backend->MatrixMultiply(a, b, c, m, n, k);

// Apply activation function
backend->ApplyGELU(input, output, size);
```

### Advanced Configuration
```cpp
// Custom CUDA configuration
CudaConfig config;
config.device_ids = {0, 1, 2, 3};  // Use 4 GPUs
config.memory_fraction = 0.9;       // Use 90% of GPU memory
config.default_precision = CudaPrecision::FP16;
config.enable_tensor_cores = true;
config.enable_flash_attention = true;

auto backend = CreateCudaBackend(config);
```

### Multi-GPU Setup
```cpp
CudaConfig config;
config.device_ids = {0, 1};         // Use GPUs 0 and 1
config.enable_peer_access = true;   // Enable P2P memory access
config.enable_memory_pool = true;   // Use memory pooling

auto backend = CreateCudaBackend(config);
backend->Initialize();

// Check active devices
auto active_devices = backend->GetActiveDevices();
```

## Performance Features

### Flash Attention v2
Memory-efficient attention computation that scales linearly with sequence length:

```cpp
// Enable Flash Attention (automatic for seq_len > 512)
bool success = backend->ComputeFlashAttention(
    queries, keys, values, output,
    batch_size, seq_len, head_dim, num_heads,
    scale, causal_mask
);
```

### Kernel Auto-Tuning
Automatic kernel parameter optimization for your specific hardware:

```cpp
// Enable auto-tuning (default: enabled)
CudaConfig config;
config.enable_kernel_timing = true;

// The backend will automatically optimize kernel parameters
// based on your GPU architecture and input characteristics
```

### Mixed Precision
Automatic precision selection for optimal performance:

```cpp
// Allocate buffers with specific precision
auto fp16_buffer = backend->AllocateCudaBuffer(
    size, CudaPrecision::FP16, device_id
);

// Automatic Tensor Core utilization for supported operations
backend->MatrixMultiply(a_fp16, b_fp16, c_fp16, m, n, k);
```

## Performance Optimization

### Memory Optimization
- **Memory Pooling**: Reduces allocation overhead
- **Unified Memory**: Simplifies multi-GPU programming
- **Prefetching**: Overlaps data movement with computation

### Compute Optimization
- **Tensor Cores**: Automatic utilization on Volta+ GPUs
- **Kernel Fusion**: Reduces memory bandwidth requirements
- **Cooperative Kernels**: Improved GPU utilization

### Multi-GPU Optimization
- **Load Balancing**: Automatic work distribution
- **P2P Communication**: Direct GPU-to-GPU transfers
- **NCCL Integration**: Optimized collective operations

## Debugging and Profiling

### Memory Debugging
```cpp
// Enable memory tracking
#define GEMMA_ENABLE_MEMORY_TRACKING
#include "backends/cuda/cuda_backend.h"

// Check for memory leaks
MemoryTracker::Instance().CheckForLeaks();
```

### Performance Profiling
```cpp
// Enable profiling
CudaConfig config;
config.enable_kernel_timing = true;

auto backend = CreateCudaBackend(config);

// Get performance metrics
auto metrics = backend->GetMetrics();
std::cout << "Memory bandwidth: " << metrics.memory_bandwidth_gbps << " GB/s" << std::endl;
```

### Debugging Tools
- **CUDA Memory Checker**: `cuda-memcheck ./your_program`
- **NVIDIA Nsight**: Visual profiler for detailed analysis
- **nvprof**: Command-line profiler

## Error Handling

The CUDA backend provides comprehensive error handling:

```cpp
if (!backend->Initialize()) {
    std::cerr << "Failed to initialize CUDA backend" << std::endl;
    // Check CUDA installation and GPU availability
}

if (!backend->MatrixMultiply(a, b, c, m, n, k)) {
    std::cerr << "Matrix multiplication failed" << std::endl;
    // Check buffer sizes and precision compatibility
}
```

## Common Issues and Solutions

### Out of Memory
```cpp
// Reduce memory usage
CudaConfig config;
config.memory_fraction = 0.7;  // Use only 70% of GPU memory
config.enable_unified_memory = true;  // Use system memory as fallback
```

### Poor Performance
```cpp
// Enable optimizations
CudaConfig config;
config.enable_tensor_cores = true;
config.enable_flash_attention = true;
config.default_precision = CudaPrecision::FP16;  // Use half precision
```

### Multi-GPU Issues
```cpp
// Verify P2P support
bool p2p_available = backend->EnablePeerAccess();
if (!p2p_available) {
    std::cout << "P2P not available, using slower CPU transfers" << std::endl;
}
```

## Benchmarking

### Performance Comparison
```cpp
// Benchmark different attention implementations
auto benchmark = attention::benchmark_attention_kernels(
    batch_size, num_heads, seq_len, head_dim
);

std::cout << "Flash Attention: " << benchmark.flash_attention_ms << "ms" << std::endl;
std::cout << "Standard Attention: " << benchmark.standard_attention_ms << "ms" << std::endl;
```

### Memory Usage Analysis
```cpp
auto estimate = attention::estimate_attention_memory(
    batch_size, num_heads, seq_len, head_dim
);

std::cout << "Flash Attention memory: "
          << estimate.flash_attention_bytes / 1024 / 1024 << " MB" << std::endl;
```

## Contributing

### Adding New Kernels
1. Add kernel implementation to `cuda_kernels.cu`
2. Add launcher function to `cuda_kernel_launcher.cpp`
3. Update the backend interface as needed
4. Add comprehensive tests

### Performance Optimization
1. Use NVIDIA Nsight Compute for kernel analysis
2. Optimize memory access patterns
3. Leverage shared memory and registers
4. Consider warp-level optimizations

## License

This CUDA backend implementation is part of the Gemma.cpp project and follows the same licensing terms.

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [TensorRT-LLM Optimizations](https://github.com/NVIDIA/TensorRT-LLM)