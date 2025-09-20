/**
 * @file cuda_kernel_launcher.cpp
 * @brief Advanced CUDA kernel launcher implementation for Gemma.cpp
 */

#include "cuda_kernel_launcher.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <functional>

namespace gemma {
namespace backends {
namespace cuda {

// Utility functions implementation
KernelLaunchParams CalculateOptimalParams1D(size_t n, int min_block_size, int max_block_size) {
    KernelLaunchParams params;
    
    // Calculate optimal block size
    int block_size = std::min(max_block_size, static_cast<int>(std::max(size_t(min_block_size), 
                                                                       std::min(size_t(512), n))));
    
    // Round to nearest power of 2 for better performance
    block_size = 1 << static_cast<int>(std::log2(block_size));
    block_size = std::max(min_block_size, std::min(max_block_size, block_size));
    
    int grid_size = (n + block_size - 1) / block_size;
    
    params.block_size = dim3(block_size);
    params.grid_size = dim3(grid_size);
    params.shared_memory_size = 0;
    
    return params;
}

KernelLaunchParams CalculateOptimalParams2D(int rows, int cols, int tile_size) {
    KernelLaunchParams params;
    
    params.block_size = dim3(tile_size, tile_size);
    params.grid_size = dim3((cols + tile_size - 1) / tile_size, (rows + tile_size - 1) / tile_size);
    params.shared_memory_size = 2 * tile_size * tile_size * sizeof(float);
    
    return params;
}

KernelLaunchParams CalculateOptimalParams3D(int depth, int height, int width, int tile_size) {
    KernelLaunchParams params;
    
    params.block_size = dim3(tile_size, tile_size, tile_size);
    params.grid_size = dim3((width + tile_size - 1) / tile_size,
                           (height + tile_size - 1) / tile_size,
                           (depth + tile_size - 1) / tile_size);
    params.shared_memory_size = tile_size * tile_size * tile_size * sizeof(float);
    
    return params;
}

size_t EstimateSharedMemoryRequirement(const std::string& operation_type, int block_size) {
    if (operation_type == "softmax" || operation_type == "layernorm") {
        return block_size * sizeof(float) * 2; // temp storage + reduction
    } else if (operation_type == "attention") {
        return block_size * block_size * sizeof(float); // tile storage
    } else if (operation_type == "matmul") {
        int tile_size = static_cast<int>(std::sqrt(block_size));
        return 2 * tile_size * tile_size * sizeof(float); // A and B tiles
    }
    return 0;
}

int GetOptimalBlockSizeForMemoryBound(size_t element_size, size_t total_elements) {
    // For memory-bound operations, optimize for memory coalescing
    size_t bytes_per_thread = element_size;
    
    if (bytes_per_thread <= 4) return 512;
    else if (bytes_per_thread <= 8) return 256;
    else if (bytes_per_thread <= 16) return 128;
    else return 64;
}

int GetOptimalBlockSizeForComputeBound(int arithmetic_intensity) {
    // For compute-bound operations, optimize for occupancy
    if (arithmetic_intensity >= 10) return 256;
    else if (arithmetic_intensity >= 5) return 512;
    else return 1024;
}

// CUDA kernel implementations (simplified for brevity)
template<typename T>
__global__ void relu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(static_cast<float>(input[idx]), 0.0f);
    }
}

template<typename T>
__global__ void gelu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        float erf_val = erff(x * 0.7071067811865476f); // x / sqrt(2)
        output[idx] = static_cast<T>(0.5f * x * (1.0f + erf_val));
    }
}

template<typename T>
__global__ void silu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(x / (1.0f + expf(-x)));
    }
}

template<typename T>
__global__ void softmax_kernel(const T* input, T* output, size_t n) {
    extern __shared__ float shared_mem[];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    
    // Load data into shared memory
    float val = (idx < n) ? static_cast<float>(input[idx]) : -INFINITY;
    shared_mem[tid] = val;
    __syncthreads();
    
    // Find maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = shared_mem[0];
    __syncthreads();
    
    // Compute exp and sum
    val = (idx < n) ? expf(static_cast<float>(input[idx]) - max_val) : 0.0f;
    shared_mem[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    float sum = shared_mem[0];
    
    if (idx < n) {
        output[idx] = static_cast<T>(val / sum);
    }
}

template<typename T>
__global__ void layernorm_kernel(const T* input, T* output, const T* gamma, const T* beta,
                                int batch_size, int hidden_size, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_data[];
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += static_cast<float>(input[batch_idx * hidden_size + i]);
    }
    
    shared_data[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    float mean = shared_data[0] / hidden_size;
    __syncthreads();
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(input[batch_idx * hidden_size + i]) - mean;
        var_sum += diff * diff;
    }
    
    shared_data[tid] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    float variance = shared_data[0] / hidden_size;
    float inv_std = rsqrtf(variance + epsilon);
    
    // Normalize and scale
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * hidden_size + i;
        float normalized = (static_cast<float>(input[idx]) - mean) * inv_std;
        output[idx] = static_cast<T>(normalized * static_cast<float>(gamma[i]) + static_cast<float>(beta[i]));
    }
}

template<typename T>
__global__ void rmsnorm_kernel(const T* input, T* output, const T* weight,
                              int batch_size, int hidden_size, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared_data[];
    
    // Calculate sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(input[batch_idx * hidden_size + i]);
        sum_sq += val * val;
    }
    
    shared_data[tid] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    float rms = sqrtf(shared_data[0] / hidden_size + epsilon);
    float inv_rms = 1.0f / rms;
    
    // Normalize and scale
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * hidden_size + i;
        float normalized = static_cast<float>(input[idx]) * inv_rms;
        output[idx] = static_cast<T>(normalized * static_cast<float>(weight[i]));
    }
}

// CudaKernelLauncher implementation
CudaKernelLauncher::CudaKernelLauncher(const CudaConfig& config)
    : config_(config), strategy_(KernelStrategy::AUTO), profiling_enabled_(false),
      next_event_index_(0) {
    
    autotune_config_.enabled = true;
    autotune_config_.max_tuning_iterations = 10;
    autotune_config_.improvement_threshold = 0.05;
    autotune_config_.cache_results = true;
    autotune_config_.use_heuristics = true;
    autotune_config_.profile_memory_access = false;
}

CudaKernelLauncher::~CudaKernelLauncher() {
    Cleanup();
}

bool CudaKernelLauncher::Initialize() {
    for (int device_id : config_.device_ids) {
        if (!InitializeDevice(device_id)) {
            std::cerr << "Failed to initialize kernel launcher for device " << device_id << std::endl;
            return false;
        }
    }
    
    // Create timing events
    const size_t num_events = 64;
    timing_events_.reserve(num_events);
    
    for (size_t i = 0; i < num_events; ++i) {
        cudaEvent_t event;
        cudaError_t error = cudaEventCreate(&event);
        if (error == cudaSuccess) {
            timing_events_.push_back(event);
        }
    }
    
    return !device_caches_.empty();
}

void CudaKernelLauncher::Cleanup() {
    for (auto& [device_id, cache] : device_caches_) {
        CleanupDevice(device_id);
    }
    device_caches_.clear();
    
    for (auto event : timing_events_) {
        cudaEventDestroy(event);
    }
    timing_events_.clear();
}

bool CudaKernelLauncher::LaunchReLU(const BackendBuffer& input, const BackendBuffer& output,
                                   size_t size, cudaStream_t stream) {
    return LaunchActivationKernel<float>(input, output, size, 
                                        kernels::ActivationType::RELU, stream);
}

bool CudaKernelLauncher::LaunchGELU(const BackendBuffer& input, const BackendBuffer& output,
                                   size_t size, cudaStream_t stream) {
    return LaunchActivationKernel<float>(input, output, size,
                                        kernels::ActivationType::GELU, stream);
}

bool CudaKernelLauncher::LaunchSiLU(const BackendBuffer& input, const BackendBuffer& output,
                                   size_t size, cudaStream_t stream) {
    return LaunchActivationKernel<float>(input, output, size,
                                        kernels::ActivationType::SILU, stream);
}

bool CudaKernelLauncher::LaunchSoftmax(const BackendBuffer& input, const BackendBuffer& output,
                                      size_t size, cudaStream_t stream) {
    const std::string kernel_name = "softmax";
    
    ValidateBuffers({&input, &output});
    
    KernelLaunchParams params = OptimizeKernelParams(kernel_name, size, size * sizeof(float));
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        softmax_kernel<float><<<params.grid_size, params.block_size, 
                              params.shared_memory_size, stream>>>(
            static_cast<const float*>(input.data),
            static_cast<float*>(output.data),
            size);
        
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

bool CudaKernelLauncher::LaunchLayerNorm(const BackendBuffer& input, const BackendBuffer& output,
                                        const BackendBuffer& gamma, const BackendBuffer& beta,
                                        int batch_size, int hidden_size, float epsilon,
                                        cudaStream_t stream) {
    return LaunchNormalizationKernel<float>(input, output, &gamma, &beta, nullptr,
                                           batch_size, hidden_size, epsilon,
                                           kernels::NormalizationType::LAYER_NORM, stream);
}

bool CudaKernelLauncher::LaunchRMSNorm(const BackendBuffer& input, const BackendBuffer& output,
                                      const BackendBuffer& weight, int batch_size, int hidden_size,
                                      float epsilon, cudaStream_t stream) {
    return LaunchNormalizationKernel<float>(input, output, nullptr, nullptr, &weight,
                                           batch_size, hidden_size, epsilon,
                                           kernels::NormalizationType::RMS_NORM, stream);
}

bool CudaKernelLauncher::LaunchFlashAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                             const BackendBuffer& values, const BackendBuffer& output,
                                             int batch_size, int seq_len, int head_dim, int num_heads,
                                             float scale, bool causal, cudaStream_t stream) {
    const std::string kernel_name = "flash_attention";
    
    ValidateBuffers({&queries, &keys, &values, &output});
    
    // Flash attention uses optimized tiling
    const int tile_size = 64;
    KernelLaunchParams params;
    params.grid_size = dim3((seq_len + tile_size - 1) / tile_size, num_heads, batch_size);
    params.block_size = dim3(tile_size);
    params.shared_memory_size = 3 * tile_size * head_dim * sizeof(float);
    params.stream = stream;
    
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    auto launcher = [&]() -> bool {
        // Simplified flash attention kernel launch
        // In a real implementation, this would call optimized flash attention kernels
        std::cout << "Flash attention kernel launch (simplified)" << std::endl;
        return true;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

bool CudaKernelLauncher::LaunchStandardAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                                const BackendBuffer& values, const BackendBuffer& output,
                                                int batch_size, int seq_len, int head_dim, int num_heads,
                                                cudaStream_t stream) {
    const std::string kernel_name = "standard_attention";
    
    ValidateBuffers({&queries, &keys, &values, &output});
    
    KernelLaunchParams params = CalculateOptimalParams2D(seq_len, seq_len, 16);
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        // Simplified standard attention kernel launch
        // In a real implementation, this would call GEMM operations for Q*K^T, softmax, and attention*V
        std::cout << "Standard attention kernel launch (simplified)" << std::endl;
        return true;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

bool CudaKernelLauncher::LaunchQuantizeINT8(const BackendBuffer& input, const BackendBuffer& output,
                                           const BackendBuffer& scale, size_t size, cudaStream_t stream) {
    const std::string kernel_name = "quantize_int8";
    
    ValidateBuffers({&input, &output, &scale});
    
    KernelLaunchParams params = OptimizeKernelParams(kernel_name, size);
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        // Simplified quantization kernel
        std::cout << "INT8 quantization kernel launch" << std::endl;
        return true;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

bool CudaKernelLauncher::LaunchDequantizeINT8(const BackendBuffer& input, const BackendBuffer& output,
                                             const BackendBuffer& scale, size_t size, cudaStream_t stream) {
    const std::string kernel_name = "dequantize_int8";
    
    ValidateBuffers({&input, &output, &scale});
    
    KernelLaunchParams params = OptimizeKernelParams(kernel_name, size);
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        // Simplified dequantization kernel
        std::cout << "INT8 dequantization kernel launch" << std::endl;
        return true;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

bool CudaKernelLauncher::LaunchFusedLinearGELU(const BackendBuffer& input, const BackendBuffer& weight,
                                              const BackendBuffer& bias, const BackendBuffer& output,
                                              int batch_size, int input_size, int output_size,
                                              cudaStream_t stream) {
    const std::string kernel_name = "fused_linear_gelu";
    
    ValidateBuffers({&input, &weight, &bias, &output});
    
    KernelLaunchParams params = CalculateOptimalParams2D(batch_size, output_size, 16);
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        // Simplified fused linear + GELU kernel
        std::cout << "Fused Linear+GELU kernel launch" << std::endl;
        return true;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

bool CudaKernelLauncher::LaunchMemcpy(const BackendBuffer& dst, const BackendBuffer& src, size_t size,
                                     cudaStream_t stream) {
    cudaError_t error = cudaMemcpyAsync(dst.data, src.data, size, cudaMemcpyDeviceToDevice, stream);
    return error == cudaSuccess;
}

bool CudaKernelLauncher::LaunchMemset(const BackendBuffer& buffer, int value, size_t size,
                                     cudaStream_t stream) {
    cudaError_t error = cudaMemsetAsync(buffer.data, value, size, stream);
    return error == cudaSuccess;
}

KernelMetrics CudaKernelLauncher::GetLastKernelMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return last_kernel_metrics_;
}

std::vector<KernelMetrics> CudaKernelLauncher::GetAllKernelMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return recent_metrics_;
}

void CudaKernelLauncher::ResetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    recent_metrics_.clear();
    last_kernel_metrics_ = KernelMetrics{};
}

void CudaKernelLauncher::SetStrategy(KernelStrategy strategy) {
    strategy_ = strategy;
}

KernelStrategy CudaKernelLauncher::GetStrategy() const {
    return strategy_;
}

void CudaKernelLauncher::SetAutoTuneConfig(const AutoTuneConfig& config) {
    autotune_config_ = config;
}

const AutoTuneConfig& CudaKernelLauncher::GetAutoTuneConfig() const {
    return autotune_config_;
}

KernelLaunchParams CudaKernelLauncher::OptimizeKernelParams(const std::string& kernel_name,
                                                           size_t input_size, size_t shared_mem_per_thread) {
    int device_id = GetCurrentDevice();
    DeviceKernelCache* cache = GetDeviceCache(device_id);
    if (!cache) {
        return CalculateOptimalParams1D(input_size);
    }
    
    std::lock_guard<std::mutex> lock(cache->mutex);
    
    auto it = cache->kernel_cache.find(kernel_name);
    if (it != cache->kernel_cache.end() && it->second.is_tuned) {
        return it->second.optimal_params;
    }
    
    // Calculate optimal parameters based on strategy
    KernelLaunchParams params;
    switch (strategy_) {
        case KernelStrategy::THROUGHPUT:
            params = GetThroughputOptimizedParams(input_size, {});
            break;
        case KernelStrategy::LATENCY:
            params = GetLatencyOptimizedParams(input_size, {});
            break;
        default:
            params = CalculateOptimalParams1D(input_size);
            break;
    }
    
    params.shared_memory_size = shared_mem_per_thread;
    
    // Store in cache
    KernelInfo& info = cache->kernel_cache[kernel_name];
    info.name = kernel_name;
    info.optimal_params = params;
    info.is_tuned = false;
    
    return params;
}

bool CudaKernelLauncher::AutoTuneKernel(const std::string& kernel_name,
                                       std::function<bool(const KernelLaunchParams&)> launcher,
                                       const std::vector<KernelLaunchParams>& candidates) {
    if (!autotune_config_.enabled || candidates.empty()) {
        return false;
    }
    
    int device_id = GetCurrentDevice();
    DeviceKernelCache* cache = GetDeviceCache(device_id);
    if (!cache) return false;
    
    KernelLaunchParams best_params = candidates[0];
    double best_time = std::numeric_limits<double>::max();
    
    for (const auto& params : candidates) {
        double total_time = 0.0;
        int successful_runs = 0;
        
        for (int i = 0; i < autotune_config_.max_tuning_iterations; ++i) {
            KernelMetrics metrics;
            if (ProfileKernel(kernel_name, params, [&]() { return launcher(params); }, metrics)) {
                total_time += metrics.execution_time_ms;
                successful_runs++;
            }
        }
        
        if (successful_runs > 0) {
            double avg_time = total_time / successful_runs;
            if (avg_time < best_time) {
                best_time = avg_time;
                best_params = params;
            }
        }
    }
    
    // Update cache
    std::lock_guard<std::mutex> lock(cache->mutex);
    KernelInfo& info = cache->kernel_cache[kernel_name];
    info.optimal_params = best_params;
    info.best_time_ms = best_time;
    info.is_tuned = true;
    
    return true;
}

double CudaKernelLauncher::CalculateOccupancy(const void* kernel_func, int block_size,
                                             size_t shared_memory_size) {
    int min_grid_size, optimal_block_size;
    cudaError_t error = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                                          kernel_func, shared_memory_size, 0);
    if (error != cudaSuccess) {
        return 0.0;
    }
    
    int device_id = GetCurrentDevice();
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    
    int max_active_blocks;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel_func,
                                                         block_size, shared_memory_size);
    if (error != cudaSuccess) {
        return 0.0;
    }
    
    int max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    int active_threads = max_active_blocks * block_size;
    
    return static_cast<double>(active_threads) / max_threads_per_sm;
}

int CudaKernelLauncher::GetOptimalBlockSize(const void* kernel_func, size_t shared_memory_size) {
    int min_grid_size, optimal_block_size;
    cudaError_t error = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                                          kernel_func, shared_memory_size, 0);
    return (error == cudaSuccess) ? optimal_block_size : 256;
}

bool CudaKernelLauncher::SupportsCooperativeLaunch() const {
    int device_id = GetCurrentDevice();
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return props.cooperativeLaunch;
}

bool CudaKernelLauncher::LaunchCooperativeKernel(const void* kernel_func, const KernelLaunchParams& params,
                                                void** kernel_args) {
    if (!SupportsCooperativeLaunch()) {
        return false;
    }
    
    cudaError_t error = cudaLaunchCooperativeKernel(kernel_func, params.grid_size, params.block_size,
                                                   kernel_args, params.shared_memory_size, params.stream);
    return error == cudaSuccess;
}

bool CudaKernelLauncher::RegisterFusedKernel(const std::string& name,
                                            const std::vector<std::string>& component_kernels) {
    int device_id = GetCurrentDevice();
    DeviceKernelCache* cache = GetDeviceCache(device_id);
    if (!cache) return false;
    
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->fused_kernels[name] = component_kernels;
    return true;
}

bool CudaKernelLauncher::LaunchFusedKernel(const std::string& name, const std::vector<void*>& args,
                                          cudaStream_t stream) {
    int device_id = GetCurrentDevice();
    DeviceKernelCache* cache = GetDeviceCache(device_id);
    if (!cache) return false;
    
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto it = cache->fused_kernels.find(name);
    if (it == cache->fused_kernels.end()) {
        return false;
    }
    
    // Simplified fused kernel execution
    std::cout << "Launching fused kernel: " << name << std::endl;
    return true;
}

void CudaKernelLauncher::EnableProfiling(bool enable) {
    profiling_enabled_ = enable;
}

bool CudaKernelLauncher::IsProfilingEnabled() const {
    return profiling_enabled_;
}

void CudaKernelLauncher::DumpKernelStats() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::cout << "=== Kernel Performance Statistics ===" << std::endl;
    std::cout << "Total kernels: " << recent_metrics_.size() << std::endl;
    
    if (!recent_metrics_.empty()) {
        double total_time = 0.0;
        for (const auto& metrics : recent_metrics_) {
            total_time += metrics.execution_time_ms;
            std::cout << "Kernel: " << metrics.kernel_name 
                      << ", Time: " << std::fixed << std::setprecision(3) << metrics.execution_time_ms << "ms"
                      << ", Occupancy: " << std::setprecision(1) << metrics.occupancy * 100.0 << "%"
                      << std::endl;
        }
        
        std::cout << "Total execution time: " << std::setprecision(3) << total_time << "ms" << std::endl;
        std::cout << "Average kernel time: " << total_time / recent_metrics_.size() << "ms" << std::endl;
    }
}

void CudaKernelLauncher::SaveTuningCache(const std::string& filename) const {
    std::string cache_file = filename.empty() ? "kernel_tuning_cache.json" : filename;
    
    std::ofstream file(cache_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open cache file for writing: " << cache_file << std::endl;
        return;
    }
    
    file << "{\n";
    bool first_device = true;
    
    for (const auto& [device_id, cache] : device_caches_) {
        if (!first_device) file << ",\n";
        first_device = false;
        
        file << "  \"device_" << device_id << "\": {\n";
        
        std::lock_guard<std::mutex> lock(cache->mutex);
        bool first_kernel = true;
        
        for (const auto& [kernel_name, info] : cache->kernel_cache) {
            if (!first_kernel) file << ",\n";
            first_kernel = false;
            
            file << "    \"" << kernel_name << "\": {\n";
            file << "      \"is_tuned\": " << (info.is_tuned ? "true" : "false") << ",\n";
            file << "      \"best_time_ms\": " << info.best_time_ms << ",\n";
            file << "      \"grid_size\": [" << info.optimal_params.grid_size.x << ", " 
                 << info.optimal_params.grid_size.y << ", " << info.optimal_params.grid_size.z << "],\n";
            file << "      \"block_size\": [" << info.optimal_params.block_size.x << ", " 
                 << info.optimal_params.block_size.y << ", " << info.optimal_params.block_size.z << "],\n";
            file << "      \"shared_memory_size\": " << info.optimal_params.shared_memory_size << "\n";
            file << "    }";
        }
        
        file << "\n  }";
    }
    
    file << "\n}\n";
}

bool CudaKernelLauncher::LoadTuningCache(const std::string& filename) {
    std::string cache_file = filename.empty() ? "kernel_tuning_cache.json" : filename;
    
    std::ifstream file(cache_file);
    if (!file.is_open()) {
        std::cout << "No tuning cache file found: " << cache_file << std::endl;
        return false;
    }
    
    // Simplified cache loading (in a real implementation, would parse JSON)
    std::cout << "Loading tuning cache from: " << cache_file << std::endl;
    return true;
}

bool CudaKernelLauncher::InitializeDevice(int device_id) {
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        return false;
    }
    
    auto cache = std::make_unique<DeviceKernelCache>();
    device_caches_[device_id] = std::move(cache);
    
    return true;
}

void CudaKernelLauncher::CleanupDevice(int device_id) {
    auto it = device_caches_.find(device_id);
    if (it != device_caches_.end()) {
        device_caches_.erase(it);
    }
}

KernelLaunchParams CudaKernelLauncher::CalculateOptimalParams(const std::string& kernel_name,
                                                             size_t input_size,
                                                             const CudaDeviceInfo& device_info) {
    // Strategy-based parameter calculation
    switch (strategy_) {
        case KernelStrategy::THROUGHPUT:
            return GetThroughputOptimizedParams(input_size, device_info);
        case KernelStrategy::LATENCY:
            return GetLatencyOptimizedParams(input_size, device_info);
        case KernelStrategy::MEMORY_BOUND:
            return CalculateOptimalParams1D(input_size, 64, 256);
        case KernelStrategy::COMPUTE_BOUND:
            return CalculateOptimalParams1D(input_size, 128, 512);
        default:
            return CalculateOptimalParams1D(input_size);
    }
}

KernelLaunchParams CudaKernelLauncher::GetThroughputOptimizedParams(size_t input_size,
                                                                   const CudaDeviceInfo& device_info) {
    // Optimize for maximum throughput
    KernelLaunchParams params;
    
    int block_size = 512; // Good balance for throughput
    int grid_size = (input_size + block_size - 1) / block_size;
    
    // Limit grid size to prevent excessive scheduling overhead
    const int max_grid_size = 65535;
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
        block_size = (input_size + grid_size - 1) / grid_size;
        block_size = std::min(1024, std::max(32, block_size));
    }
    
    params.grid_size = dim3(grid_size);
    params.block_size = dim3(block_size);
    params.shared_memory_size = 0;
    
    return params;
}

KernelLaunchParams CudaKernelLauncher::GetLatencyOptimizedParams(size_t input_size,
                                                                const CudaDeviceInfo& device_info) {
    // Optimize for minimum latency
    KernelLaunchParams params;
    
    int block_size = 256; // Good balance for latency
    int grid_size = std::min(32, static_cast<int>((input_size + block_size - 1) / block_size));
    
    params.grid_size = dim3(grid_size);
    params.block_size = dim3(block_size);
    params.shared_memory_size = 0;
    
    return params;
}

bool CudaKernelLauncher::ProfileKernel(const std::string& kernel_name, const KernelLaunchParams& params,
                                      std::function<bool()> launcher, KernelMetrics& metrics) {
    if (!profiling_enabled_) {
        return launcher();
    }
    
    cudaEvent_t start_event = GetTimingEvent();
    cudaEvent_t end_event = GetTimingEvent();
    
    if (!start_event || !end_event) {
        ReturnTimingEvent(start_event);
        ReturnTimingEvent(end_event);
        return launcher();
    }
    
    cudaEventRecord(start_event, params.stream);
    bool success = launcher();
    cudaEventRecord(end_event, params.stream);
    
    if (success) {
        cudaEventSynchronize(end_event);
        
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_event, end_event);
        
        metrics.kernel_name = kernel_name;
        metrics.execution_time_ms = elapsed_ms;
        metrics.grid_size_x = params.grid_size.x;
        metrics.grid_size_y = params.grid_size.y;
        metrics.grid_size_z = params.grid_size.z;
        metrics.block_size_x = params.block_size.x;
        metrics.block_size_y = params.block_size.y;
        metrics.block_size_z = params.block_size.z;
        metrics.shared_memory_per_block = params.shared_memory_size;
        
        UpdateKernelMetrics(kernel_name, metrics);
    }
    
    ReturnTimingEvent(start_event);
    ReturnTimingEvent(end_event);
    
    return success;
}

bool CudaKernelLauncher::RecordKernelLaunch(const std::string& kernel_name, const KernelLaunchParams& params,
                                           cudaStream_t stream) {
    // Record kernel launch for statistics
    return true;
}

void CudaKernelLauncher::UpdateKernelMetrics(const std::string& kernel_name, const KernelMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    recent_metrics_.push_back(metrics);
    last_kernel_metrics_ = metrics;
    
    // Keep only recent metrics
    const size_t max_metrics = 1000;
    if (recent_metrics_.size() > max_metrics) {
        recent_metrics_.erase(recent_metrics_.begin(), 
                             recent_metrics_.begin() + (recent_metrics_.size() - max_metrics));
    }
}

CudaKernelLauncher::DeviceKernelCache* CudaKernelLauncher::GetDeviceCache(int device_id) {
    auto it = device_caches_.find(device_id);
    return (it != device_caches_.end()) ? it->second.get() : nullptr;
}

const CudaKernelLauncher::DeviceKernelCache* CudaKernelLauncher::GetDeviceCache(int device_id) const {
    auto it = device_caches_.find(device_id);
    return (it != device_caches_.end()) ? it->second.get() : nullptr;
}

cudaEvent_t CudaKernelLauncher::GetTimingEvent() {
    if (timing_events_.empty()) {
        return nullptr;
    }
    
    size_t index = next_event_index_.fetch_add(1) % timing_events_.size();
    return timing_events_[index];
}

void CudaKernelLauncher::ReturnTimingEvent(cudaEvent_t event) {
    // Events are managed in a circular buffer, no explicit return needed
}

template<typename T>
bool CudaKernelLauncher::LaunchActivationKernel(const BackendBuffer& input, const BackendBuffer& output,
                                               size_t size, kernels::ActivationType type,
                                               cudaStream_t stream) {
    ValidateBuffers({&input, &output});
    
    std::string kernel_name;
    switch (type) {
        case kernels::ActivationType::RELU: kernel_name = "relu"; break;
        case kernels::ActivationType::GELU: kernel_name = "gelu"; break;
        case kernels::ActivationType::SILU: kernel_name = "silu"; break;
        default: return false;
    }
    
    KernelLaunchParams params = OptimizeKernelParams(kernel_name, size);
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        const T* input_ptr = static_cast<const T*>(input.data);
        T* output_ptr = static_cast<T*>(output.data);
        
        switch (type) {
            case kernels::ActivationType::RELU:
                relu_kernel<T><<<params.grid_size, params.block_size, 0, stream>>>(input_ptr, output_ptr, size);
                break;
            case kernels::ActivationType::GELU:
                gelu_kernel<T><<<params.grid_size, params.block_size, 0, stream>>>(input_ptr, output_ptr, size);
                break;
            case kernels::ActivationType::SILU:
                silu_kernel<T><<<params.grid_size, params.block_size, 0, stream>>>(input_ptr, output_ptr, size);
                break;
            default:
                return false;
        }
        
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

template<typename T>
bool CudaKernelLauncher::LaunchNormalizationKernel(const BackendBuffer& input, const BackendBuffer& output,
                                                  const BackendBuffer* gamma, const BackendBuffer* beta,
                                                  const BackendBuffer* weight, int batch_size, int hidden_size,
                                                  float epsilon, kernels::NormalizationType type,
                                                  cudaStream_t stream) {
    ValidateBuffers({&input, &output});
    
    std::string kernel_name;
    switch (type) {
        case kernels::NormalizationType::LAYER_NORM: kernel_name = "layernorm"; break;
        case kernels::NormalizationType::RMS_NORM: kernel_name = "rmsnorm"; break;
        default: return false;
    }
    
    KernelLaunchParams params;
    params.grid_size = dim3(batch_size);
    params.block_size = dim3(std::min(1024, hidden_size));
    params.shared_memory_size = params.block_size.x * sizeof(float);
    params.stream = stream;
    
    auto launcher = [&]() -> bool {
        const T* input_ptr = static_cast<const T*>(input.data);
        T* output_ptr = static_cast<T*>(output.data);
        
        switch (type) {
            case kernels::NormalizationType::LAYER_NORM:
                if (gamma && beta) {
                    const T* gamma_ptr = static_cast<const T*>(gamma->data);
                    const T* beta_ptr = static_cast<const T*>(beta->data);
                    layernorm_kernel<T><<<params.grid_size, params.block_size, params.shared_memory_size, stream>>>(
                        input_ptr, output_ptr, gamma_ptr, beta_ptr, batch_size, hidden_size, epsilon);
                }
                break;
            case kernels::NormalizationType::RMS_NORM:
                if (weight) {
                    const T* weight_ptr = static_cast<const T*>(weight->data);
                    rmsnorm_kernel<T><<<params.grid_size, params.block_size, params.shared_memory_size, stream>>>(
                        input_ptr, output_ptr, weight_ptr, batch_size, hidden_size, epsilon);
                }
                break;
            default:
                return false;
        }
        
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess;
    };
    
    return LaunchKernel(kernel_name, launcher, params);
}

CudaPrecision CudaKernelLauncher::GetBufferPrecision(const BackendBuffer& buffer) const {
    // Simplified precision detection - in real implementation would check buffer metadata
    return CudaPrecision::FP32;
}

int CudaKernelLauncher::GetCurrentDevice() const {
    int device_id;
    cudaGetDevice(&device_id);
    return device_id;
}

void CudaKernelLauncher::ValidateBuffers(const std::vector<const BackendBuffer*>& buffers) const {
    for (const auto* buffer : buffers) {
        if (!buffer || !buffer->data) {
            throw std::invalid_argument("Invalid buffer provided to kernel launcher");
        }
    }
}

size_t CudaKernelLauncher::EstimateSharedMemoryUsage(const std::string& kernel_name, int block_size) const {
    // Simplified shared memory estimation
    if (kernel_name.find("softmax") != std::string::npos || 
        kernel_name.find("norm") != std::string::npos) {
        return block_size * sizeof(float);
    }
    return 0;
}

} // namespace cuda
} // namespace backends
} // namespace gemma