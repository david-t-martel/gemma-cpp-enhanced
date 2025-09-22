/**
 * @file cuda_kernels.cu
 * @brief Custom CUDA kernels for Gemma.cpp inference operations
 *
 * This file contains optimized CUDA kernels for:
 * - Activation functions (ReLU, GELU, SiLU, Softmax)
 * - Normalization layers (LayerNorm, RMSNorm)
 * - Quantization operations (INT8, INT4)
 * - Fused operations for improved performance
 * - Memory utilities and optimized data movement
 */

#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <math_constants.h>

namespace gemma {
namespace backends {
namespace cuda {
namespace kernels {

namespace cg = cooperative_groups;

// Constants
constexpr float GELU_COEFF = 0.79788456080287f; // sqrt(2/pi)
constexpr float GELU_TANH_COEFF = 0.044715f;
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Utility functions
__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(GELU_COEFF * (x + GELU_TANH_COEFF * x * x * x)));
}

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ __half fast_gelu(__half x) {
    float fx = __half2float(x);
    return __float2half(fast_gelu(fx));
}

__device__ __forceinline__ __half fast_silu(__half x) {
    float fx = __half2float(x);
    return __float2half(fast_silu(fx));
}

// Warp-level reduction operations
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Specialization for half precision
template<>
__device__ __forceinline__ __half warp_reduce_sum(__half val) {
    float fval = __half2float(val);
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        fval += __half2float(__shfl_down_sync(0xFFFFFFFF, __float2half(fval), offset));
    }
    return __float2half(fval);
}

template<>
__device__ __forceinline__ __half warp_reduce_max(__half val) {
    float fval = __half2float(val);
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        fval = fmaxf(fval, __half2float(__shfl_down_sync(0xFFFFFFFF, __float2half(fval), offset)));
    }
    return __float2half(fval);
}

// Activation function kernels
template<typename T>
__global__ void relu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        output[idx] = fmaxf(val, T(0));
    }
}

template<typename T>
__global__ void gelu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fast_gelu(input[idx]);
    }
}

template<typename T>
__global__ void silu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fast_silu(input[idx]);
    }
}

// Vectorized activation kernels for improved memory throughput
template<typename T>
__global__ void relu_kernel_vectorized(const T* input, T* output, size_t n) {
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(T);
    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t global_idx = vec_idx * VEC_SIZE;

    if (global_idx + VEC_SIZE <= n) {
        // Vectorized load
        T vals[VEC_SIZE];
        *reinterpret_cast<float4*>(vals) = *reinterpret_cast<const float4*>(&input[global_idx]);

        // Apply ReLU
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            vals[i] = fmaxf(vals[i], T(0));
        }

        // Vectorized store
        *reinterpret_cast<float4*>(&output[global_idx]) = *reinterpret_cast<float4*>(vals);
    } else {
        // Handle remaining elements
        for (size_t i = global_idx; i < n && i < global_idx + VEC_SIZE; ++i) {
            output[i] = fmaxf(input[i], T(0));
        }
    }
}

// Softmax kernel with numerically stable implementation
template<typename T>
__global__ void softmax_kernel(const T* input, T* output, int batch_size, int feature_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const T* input_row = input + batch_idx * feature_size;
    T* output_row = output + batch_idx * feature_size;

    // Shared memory for reduction
    extern __shared__ T shmem[];

    // Find maximum for numerical stability
    T thread_max = T(-INFINITY);
    for (int i = tid; i < feature_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input_row[i]);
    }

    // Reduce to find global max
    shmem[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] = fmaxf(shmem[tid], shmem[tid + stride]);
        }
        __syncthreads();
    }

    T max_val = shmem[0];
    __syncthreads();

    // Compute exp and sum
    T thread_sum = T(0);
    for (int i = tid; i < feature_size; i += blockDim.x) {
        T exp_val = expf(input_row[i] - max_val);
        output_row[i] = exp_val;
        thread_sum += exp_val;
    }

    // Reduce to find global sum
    shmem[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    T sum_val = shmem[0];
    __syncthreads();

    // Normalize
    T inv_sum = T(1) / sum_val;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        output_row[i] *= inv_sum;
    }
}

// Layer normalization kernel
template<typename T>
__global__ void layer_norm_kernel(const T* input, T* output, const T* gamma, const T* beta,
                                 int batch_size, int hidden_size, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const T* input_row = input + batch_idx * hidden_size;
    T* output_row = output + batch_idx * hidden_size;

    extern __shared__ T shmem[];

    // Compute mean
    T thread_sum = T(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        thread_sum += input_row[i];
    }

    shmem[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    T mean = shmem[0] / T(hidden_size);
    __syncthreads();

    // Compute variance
    T thread_var = T(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        T diff = input_row[i] - mean;
        thread_var += diff * diff;
    }

    shmem[tid] = thread_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    T variance = shmem[0] / T(hidden_size);
    T inv_std = rsqrtf(variance + epsilon);
    __syncthreads();

    // Apply normalization and scaling
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        T normalized = (input_row[i] - mean) * inv_std;
        output_row[i] = normalized * gamma[i] + beta[i];
    }
}

// RMS normalization kernel
template<typename T>
__global__ void rms_norm_kernel(const T* input, T* output, const T* weight,
                               int batch_size, int hidden_size, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const T* input_row = input + batch_idx * hidden_size;
    T* output_row = output + batch_idx * hidden_size;

    extern __shared__ T shmem[];

    // Compute sum of squares
    T thread_sum_sq = T(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        T val = input_row[i];
        thread_sum_sq += val * val;
    }

    shmem[tid] = thread_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    T mean_sq = shmem[0] / T(hidden_size);
    T rms = rsqrtf(mean_sq + epsilon);
    __syncthreads();

    // Apply RMS normalization and scaling
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        output_row[i] = input_row[i] * rms * weight[i];
    }
}

// Quantization kernels
__global__ void quantize_int8_kernel(const float* input, int8_t* output, const float* scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale[0];
        val = fmaxf(-128.0f, fminf(127.0f, roundf(val)));
        output[idx] = static_cast<int8_t>(val);
    }
}

__global__ void dequantize_int8_kernel(const int8_t* input, float* output, const float* scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(input[idx]) * scale[0];
    }
}

// INT4 quantization (packed format)
__global__ void quantize_int4_kernel(const float* input, int8_t* output, const float* scale,
                                    const float* zero_point, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t element_idx = idx * 2;

    if (element_idx + 1 < n) {
        // Quantize two values and pack into one byte
        float val1 = input[element_idx] / scale[0] + zero_point[0];
        float val2 = input[element_idx + 1] / scale[0] + zero_point[0];

        val1 = fmaxf(0.0f, fminf(15.0f, roundf(val1)));
        val2 = fmaxf(0.0f, fminf(15.0f, roundf(val2)));

        int8_t packed = (static_cast<int8_t>(val1) & 0x0F) | ((static_cast<int8_t>(val2) & 0x0F) << 4);
        output[idx] = packed;
    } else if (element_idx < n) {
        // Handle last element if odd number of elements
        float val = input[element_idx] / scale[0] + zero_point[0];
        val = fmaxf(0.0f, fminf(15.0f, roundf(val)));
        output[idx] = static_cast<int8_t>(val) & 0x0F;
    }
}

__global__ void dequantize_int4_kernel(const int8_t* input, float* output, const float* scale,
                                     const float* zero_point, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t element_idx = idx * 2;

    if (element_idx + 1 < n) {
        int8_t packed = input[idx];

        float val1 = (static_cast<float>(packed & 0x0F) - zero_point[0]) * scale[0];
        float val2 = (static_cast<float>((packed >> 4) & 0x0F) - zero_point[0]) * scale[0];

        output[element_idx] = val1;
        output[element_idx + 1] = val2;
    } else if (element_idx < n) {
        int8_t packed = input[idx];
        float val = (static_cast<float>(packed & 0x0F) - zero_point[0]) * scale[0];
        output[element_idx] = val;
    }
}

// Fused operations
template<typename T>
__global__ void fused_linear_gelu_kernel(const T* input, const T* weight, const T* bias, T* output,
                                        int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= output_size) return;

    extern __shared__ T shmem[];

    // Compute dot product
    T sum = T(0);
    for (int i = tid; i < input_size; i += blockDim.x) {
        sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
    }

    // Reduce within warp
    sum = warp_reduce_sum(sum);

    // Store to shared memory for inter-warp reduction
    if (tid % WARP_SIZE == 0) {
        shmem[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    // Final reduction
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        sum = (tid < blockDim.x / WARP_SIZE) ? shmem[tid] : T(0);
        sum = warp_reduce_sum(sum);
    }

    // Apply bias and GELU
    if (tid == 0) {
        T result = sum + bias[out_idx];
        output[batch_idx * output_size + out_idx] = fast_gelu(result);
    }
}

template<typename T>
__global__ void fused_multiply_add_kernel(const T* a, const T* b, const T* c, T* output,
                                         size_t n, T alpha, T beta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = alpha * a[idx] * b[idx] + beta * c[idx];
    }
}

// Memory utilities
template<typename T>
__global__ void memset_kernel(T* ptr, T value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}

template<typename T>
__global__ void copy_kernel(const T* src, T* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Type conversion kernels
__global__ void fp32_to_fp16_kernel(const float* input, __half* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void fp16_to_fp32_kernel(const __half* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// Optimized block reduce for large reductions
template<typename T, int BLOCK_SIZE>
__device__ void block_reduce_sum(T* shmem, T val, int tid) {
    shmem[tid] = val;
    __syncthreads();

    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256) shmem[tid] += shmem[tid + 256];
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128) shmem[tid] += shmem[tid + 128];
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64) shmem[tid] += shmem[tid + 64];
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        if constexpr (BLOCK_SIZE >= 64) shmem[tid] += shmem[tid + 32];
        if constexpr (BLOCK_SIZE >= 32) shmem[tid] += shmem[tid + 16];
        if constexpr (BLOCK_SIZE >= 16) shmem[tid] += shmem[tid + 8];
        if constexpr (BLOCK_SIZE >= 8) shmem[tid] += shmem[tid + 4];
        if constexpr (BLOCK_SIZE >= 4) shmem[tid] += shmem[tid + 2];
        if constexpr (BLOCK_SIZE >= 2) shmem[tid] += shmem[tid + 1];
    }
}

// Kernel launch utilities
template<typename T>
cudaError_t launch_activation_kernel(const T* input, T* output, size_t n, ActivationType type, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (type) {
        case ActivationType::RELU:
            if (n >= 1024 && sizeof(T) <= 4) {
                // Use vectorized version for large tensors
                constexpr int VEC_SIZE = sizeof(float4) / sizeof(T);
                int vec_grid_size = ((n + VEC_SIZE - 1) / VEC_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
                relu_kernel_vectorized<<<vec_grid_size, BLOCK_SIZE, 0, stream>>>(input, output, n);
            } else {
                relu_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(input, output, n);
            }
            break;

        case ActivationType::GELU:
            gelu_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(input, output, n);
            break;

        case ActivationType::SILU:
            silu_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(input, output, n);
            break;

        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

// Explicit template instantiations
template cudaError_t launch_activation_kernel<float>(const float*, float*, size_t, ActivationType, cudaStream_t);
template cudaError_t launch_activation_kernel<__half>(const __half*, __half*, size_t, ActivationType, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace backends
} // namespace gemma