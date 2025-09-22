#pragma once

/**
 * @file cuda_kernels.h
 * @brief Header for custom CUDA kernels used in Gemma.cpp
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stddef.h>

namespace gemma {
namespace backends {
namespace cuda {
namespace kernels {

/**
 * @brief Activation function types
 */
enum class ActivationType {
    RELU,
    GELU,
    SILU,
    SWISH = SILU,  // Alias for SiLU
    SOFTMAX
};

/**
 * @brief Normalization types
 */
enum class NormalizationType {
    LAYER_NORM,
    RMS_NORM,
    BATCH_NORM
};

/**
 * @brief Quantization types
 */
enum class QuantizationType {
    INT8,
    INT4,
    BINARY
};

// Activation function kernel launchers
template<typename T>
cudaError_t launch_activation_kernel(const T* input, T* output, size_t n,
                                   ActivationType type, cudaStream_t stream = nullptr);

// Normalization kernel launchers
template<typename T>
cudaError_t launch_layer_norm_kernel(const T* input, T* output, const T* gamma, const T* beta,
                                    int batch_size, int hidden_size, float epsilon = 1e-5f,
                                    cudaStream_t stream = nullptr);

template<typename T>
cudaError_t launch_rms_norm_kernel(const T* input, T* output, const T* weight,
                                 int batch_size, int hidden_size, float epsilon = 1e-5f,
                                 cudaStream_t stream = nullptr);

// Softmax kernel launcher
template<typename T>
cudaError_t launch_softmax_kernel(const T* input, T* output, int batch_size, int feature_size,
                                 cudaStream_t stream = nullptr);

// Quantization kernel launchers
cudaError_t launch_quantize_int8_kernel(const float* input, int8_t* output, const float* scale,
                                       size_t n, cudaStream_t stream = nullptr);

cudaError_t launch_dequantize_int8_kernel(const int8_t* input, float* output, const float* scale,
                                         size_t n, cudaStream_t stream = nullptr);

cudaError_t launch_quantize_int4_kernel(const float* input, int8_t* output, const float* scale,
                                       const float* zero_point, size_t n, cudaStream_t stream = nullptr);

cudaError_t launch_dequantize_int4_kernel(const int8_t* input, float* output, const float* scale,
                                         const float* zero_point, size_t n, cudaStream_t stream = nullptr);

// Fused operation kernel launchers
template<typename T>
cudaError_t launch_fused_linear_gelu_kernel(const T* input, const T* weight, const T* bias, T* output,
                                           int batch_size, int input_size, int output_size,
                                           cudaStream_t stream = nullptr);

template<typename T>
cudaError_t launch_fused_multiply_add_kernel(const T* a, const T* b, const T* c, T* output,
                                           size_t n, T alpha = T(1), T beta = T(1),
                                           cudaStream_t stream = nullptr);

// Memory utility kernel launchers
template<typename T>
cudaError_t launch_memset_kernel(T* ptr, T value, size_t n, cudaStream_t stream = nullptr);

template<typename T>
cudaError_t launch_copy_kernel(const T* src, T* dst, size_t n, cudaStream_t stream = nullptr);

// Type conversion kernel launchers
cudaError_t launch_fp32_to_fp16_kernel(const float* input, __half* output, size_t n,
                                      cudaStream_t stream = nullptr);

cudaError_t launch_fp16_to_fp32_kernel(const __half* input, float* output, size_t n,
                                      cudaStream_t stream = nullptr);

// Utility functions for kernel configuration
struct KernelConfig {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_memory_size;
    cudaStream_t stream;

    KernelConfig() : grid_size(1), block_size(1), shared_memory_size(0), stream(nullptr) {}
};

/**
 * @brief Calculate optimal kernel configuration for given parameters
 */
KernelConfig calculate_kernel_config(size_t n, int min_block_size = 32, int max_block_size = 1024);

/**
 * @brief Calculate optimal 2D kernel configuration for matrix operations
 */
KernelConfig calculate_2d_kernel_config(int rows, int cols, int tile_size = 16);

/**
 * @brief Get device properties for optimization
 */
struct DeviceProperties {
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int warp_size;
    int multiprocessor_count;
    int max_threads_per_multiprocessor;
    bool supports_cooperative_launch;
};

DeviceProperties get_device_properties(int device_id = -1);

/**
 * @brief Check if a kernel configuration is valid for the current device
 */
bool is_kernel_config_valid(const KernelConfig& config, int device_id = -1);

/**
 * @brief Optimize kernel configuration based on device properties
 */
KernelConfig optimize_kernel_config(const KernelConfig& config, const DeviceProperties& props);

} // namespace kernels
} // namespace cuda
} // namespace backends
} // namespace gemma