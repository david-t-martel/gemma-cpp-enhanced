#pragma once

/**
 * @file cuda_kernel_launcher.h
 * @brief Advanced CUDA kernel launching and management system
 *
 * Provides optimized kernel launching with:
 * - Automatic kernel configuration optimization
 * - Dynamic kernel selection based on input characteristics
 * - Kernel performance profiling and auto-tuning
 * - Cooperative kernel launching
 * - Kernel fusion and optimization
 */

#include "cuda_backend.h"
#include "cuda_kernels.h"
#include "cuda_attention.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>

namespace gemma {
namespace backends {
namespace cuda {

/**
 * @brief Kernel launch parameters
 */
struct KernelLaunchParams {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_memory_size = 0;
    cudaStream_t stream = nullptr;

    KernelLaunchParams() = default;
    KernelLaunchParams(dim3 grid, dim3 block, size_t shared_mem = 0, cudaStream_t s = nullptr)
        : grid_size(grid), block_size(block), shared_memory_size(shared_mem), stream(s) {}
};

/**
 * @brief Kernel performance metrics
 */
struct KernelMetrics {
    std::string kernel_name;
    double execution_time_ms = 0.0;
    double occupancy = 0.0;
    size_t registers_per_thread = 0;
    size_t shared_memory_per_block = 0;
    double bandwidth_utilization = 0.0;
    double compute_utilization = 0.0;
    size_t grid_size_x = 0;
    size_t grid_size_y = 0;
    size_t grid_size_z = 0;
    size_t block_size_x = 0;
    size_t block_size_y = 0;
    size_t block_size_z = 0;
};

/**
 * @brief Kernel configuration strategy
 */
enum class KernelStrategy {
    AUTO,               // Automatic selection based on input size
    THROUGHPUT,         // Optimize for maximum throughput
    LATENCY,           // Optimize for minimum latency
    MEMORY_BOUND,      // Optimize for memory-bound kernels
    COMPUTE_BOUND,     // Optimize for compute-bound kernels
    MIXED_PRECISION,   // Optimize for mixed precision operations
    COOPERATIVE        // Use cooperative kernel launching
};

/**
 * @brief Kernel auto-tuning configuration
 */
struct AutoTuneConfig {
    bool enabled = true;
    int max_tuning_iterations = 10;
    double improvement_threshold = 0.05; // 5% improvement required
    bool cache_results = true;
    std::string cache_file = "";
    bool use_heuristics = true;
    bool profile_memory_access = false;
};

/**
 * @brief Kernel launcher interface
 */
class KernelLauncher {
public:
    virtual ~KernelLauncher() = default;

    // Activation functions
    virtual bool LaunchReLU(const BackendBuffer& input, const BackendBuffer& output,
                           size_t size, cudaStream_t stream = nullptr) = 0;
    virtual bool LaunchGELU(const BackendBuffer& input, const BackendBuffer& output,
                           size_t size, cudaStream_t stream = nullptr) = 0;
    virtual bool LaunchSiLU(const BackendBuffer& input, const BackendBuffer& output,
                           size_t size, cudaStream_t stream = nullptr) = 0;
    virtual bool LaunchSoftmax(const BackendBuffer& input, const BackendBuffer& output,
                              size_t size, cudaStream_t stream = nullptr) = 0;

    // Normalization
    virtual bool LaunchLayerNorm(const BackendBuffer& input, const BackendBuffer& output,
                                const BackendBuffer& gamma, const BackendBuffer& beta,
                                int batch_size, int hidden_size, float epsilon = 1e-5f,
                                cudaStream_t stream = nullptr) = 0;
    virtual bool LaunchRMSNorm(const BackendBuffer& input, const BackendBuffer& output,
                              const BackendBuffer& weight, int batch_size, int hidden_size,
                              float epsilon = 1e-5f, cudaStream_t stream = nullptr) = 0;

    // Attention
    virtual bool LaunchFlashAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                     const BackendBuffer& values, const BackendBuffer& output,
                                     int batch_size, int seq_len, int head_dim, int num_heads,
                                     float scale = 0.0f, bool causal = false,
                                     cudaStream_t stream = nullptr) = 0;
    virtual bool LaunchStandardAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                        const BackendBuffer& values, const BackendBuffer& output,
                                        int batch_size, int seq_len, int head_dim, int num_heads,
                                        cudaStream_t stream = nullptr) = 0;

    // Performance monitoring
    virtual KernelMetrics GetLastKernelMetrics() const = 0;
    virtual std::vector<KernelMetrics> GetAllKernelMetrics() const = 0;
    virtual void ResetMetrics() = 0;

    // Configuration
    virtual void SetStrategy(KernelStrategy strategy) = 0;
    virtual KernelStrategy GetStrategy() const = 0;
    virtual void SetAutoTuneConfig(const AutoTuneConfig& config) = 0;
    virtual const AutoTuneConfig& GetAutoTuneConfig() const = 0;
};

/**
 * @brief Advanced CUDA kernel launcher implementation
 */
class CudaKernelLauncher : public KernelLauncher {
public:
    explicit CudaKernelLauncher(const CudaConfig& config);
    ~CudaKernelLauncher() override;

    // Initialization
    bool Initialize();
    void Cleanup();

    // Activation functions
    bool LaunchReLU(const BackendBuffer& input, const BackendBuffer& output,
                   size_t size, cudaStream_t stream = nullptr) override;
    bool LaunchGELU(const BackendBuffer& input, const BackendBuffer& output,
                   size_t size, cudaStream_t stream = nullptr) override;
    bool LaunchSiLU(const BackendBuffer& input, const BackendBuffer& output,
                   size_t size, cudaStream_t stream = nullptr) override;
    bool LaunchSoftmax(const BackendBuffer& input, const BackendBuffer& output,
                      size_t size, cudaStream_t stream = nullptr) override;

    // Normalization
    bool LaunchLayerNorm(const BackendBuffer& input, const BackendBuffer& output,
                        const BackendBuffer& gamma, const BackendBuffer& beta,
                        int batch_size, int hidden_size, float epsilon = 1e-5f,
                        cudaStream_t stream = nullptr) override;
    bool LaunchRMSNorm(const BackendBuffer& input, const BackendBuffer& output,
                      const BackendBuffer& weight, int batch_size, int hidden_size,
                      float epsilon = 1e-5f, cudaStream_t stream = nullptr) override;

    // Attention
    bool LaunchFlashAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                             const BackendBuffer& values, const BackendBuffer& output,
                             int batch_size, int seq_len, int head_dim, int num_heads,
                             float scale = 0.0f, bool causal = false,
                             cudaStream_t stream = nullptr) override;
    bool LaunchStandardAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                                const BackendBuffer& values, const BackendBuffer& output,
                                int batch_size, int seq_len, int head_dim, int num_heads,
                                cudaStream_t stream = nullptr) override;

    // Advanced operations
    bool LaunchQuantizeINT8(const BackendBuffer& input, const BackendBuffer& output,
                           const BackendBuffer& scale, size_t size, cudaStream_t stream = nullptr);
    bool LaunchDequantizeINT8(const BackendBuffer& input, const BackendBuffer& output,
                             const BackendBuffer& scale, size_t size, cudaStream_t stream = nullptr);

    bool LaunchFusedLinearGELU(const BackendBuffer& input, const BackendBuffer& weight,
                              const BackendBuffer& bias, const BackendBuffer& output,
                              int batch_size, int input_size, int output_size,
                              cudaStream_t stream = nullptr);

    // Memory operations
    bool LaunchMemcpy(const BackendBuffer& dst, const BackendBuffer& src, size_t size,
                     cudaStream_t stream = nullptr);
    bool LaunchMemset(const BackendBuffer& buffer, int value, size_t size,
                     cudaStream_t stream = nullptr);

    // Generic kernel launcher
    template<typename KernelFunc, typename... Args>
    bool LaunchKernel(const std::string& kernel_name, KernelFunc kernel,
                     const KernelLaunchParams& params, Args&&... args);

    // Performance monitoring
    KernelMetrics GetLastKernelMetrics() const override;
    std::vector<KernelMetrics> GetAllKernelMetrics() const override;
    void ResetMetrics() override;

    // Configuration
    void SetStrategy(KernelStrategy strategy) override;
    KernelStrategy GetStrategy() const override;
    void SetAutoTuneConfig(const AutoTuneConfig& config) override;
    const AutoTuneConfig& GetAutoTuneConfig() const override;

    // Kernel optimization
    KernelLaunchParams OptimizeKernelParams(const std::string& kernel_name,
                                           size_t input_size, size_t shared_mem_per_thread = 0);
    bool AutoTuneKernel(const std::string& kernel_name,
                       std::function<bool(const KernelLaunchParams&)> launcher,
                       const std::vector<KernelLaunchParams>& candidates);

    // Occupancy calculation
    double CalculateOccupancy(const void* kernel_func, int block_size,
                             size_t shared_memory_size = 0);
    int GetOptimalBlockSize(const void* kernel_func, size_t shared_memory_size = 0);

    // Cooperative kernels
    bool SupportsCooperativeLaunch() const;
    bool LaunchCooperativeKernel(const void* kernel_func, const KernelLaunchParams& params,
                                void** kernel_args);

    // Kernel fusion
    bool RegisterFusedKernel(const std::string& name,
                           const std::vector<std::string>& component_kernels);
    bool LaunchFusedKernel(const std::string& name, const std::vector<void*>& args,
                          cudaStream_t stream = nullptr);

    // Profiling and debugging
    void EnableProfiling(bool enable);
    bool IsProfilingEnabled() const;
    void DumpKernelStats() const;
    void SaveTuningCache(const std::string& filename = "") const;
    bool LoadTuningCache(const std::string& filename = "");

private:
    struct KernelInfo {
        std::string name;
        KernelLaunchParams optimal_params;
        std::vector<KernelMetrics> metrics_history;
        bool is_tuned = false;
        double best_time_ms = std::numeric_limits<double>::max();
    };

    struct DeviceKernelCache {
        std::unordered_map<std::string, KernelInfo> kernel_cache;
        std::unordered_map<std::string, std::vector<std::string>> fused_kernels;
        mutable std::mutex mutex;
    };

    CudaConfig config_;
    KernelStrategy strategy_;
    AutoTuneConfig autotune_config_;

    // Per-device kernel management
    std::unordered_map<int, std::unique_ptr<DeviceKernelCache>> device_caches_;

    // Performance monitoring
    bool profiling_enabled_;
    mutable std::mutex metrics_mutex_;
    std::vector<KernelMetrics> recent_metrics_;
    KernelMetrics last_kernel_metrics_;

    // CUDA events for timing
    std::vector<cudaEvent_t> timing_events_;
    size_t next_event_index_;

    // Helper methods
    bool InitializeDevice(int device_id);
    void CleanupDevice(int device_id);

    KernelLaunchParams CalculateOptimalParams(const std::string& kernel_name,
                                             size_t input_size,
                                             const CudaDeviceInfo& device_info);

    KernelLaunchParams GetThroughputOptimizedParams(size_t input_size,
                                                   const CudaDeviceInfo& device_info);
    KernelLaunchParams GetLatencyOptimizedParams(size_t input_size,
                                                const CudaDeviceInfo& device_info);

    bool ProfileKernel(const std::string& kernel_name, const KernelLaunchParams& params,
                      std::function<bool()> launcher, KernelMetrics& metrics);

    bool RecordKernelLaunch(const std::string& kernel_name, const KernelLaunchParams& params,
                           cudaStream_t stream);
    void UpdateKernelMetrics(const std::string& kernel_name, const KernelMetrics& metrics);

    DeviceKernelCache* GetDeviceCache(int device_id);
    const DeviceKernelCache* GetDeviceCache(int device_id) const;

    cudaEvent_t GetTimingEvent();
    void ReturnTimingEvent(cudaEvent_t event);

    // Template implementations for type safety
    template<typename T>
    bool LaunchActivationKernel(const BackendBuffer& input, const BackendBuffer& output,
                               size_t size, kernels::ActivationType type,
                               cudaStream_t stream);

    template<typename T>
    bool LaunchNormalizationKernel(const BackendBuffer& input, const BackendBuffer& output,
                                  const BackendBuffer* gamma, const BackendBuffer* beta,
                                  const BackendBuffer* weight, int batch_size, int hidden_size,
                                  float epsilon, kernels::NormalizationType type,
                                  cudaStream_t stream);

    // Utility functions
    CudaPrecision GetBufferPrecision(const BackendBuffer& buffer) const;
    int GetCurrentDevice() const;
    void ValidateBuffers(const std::vector<const BackendBuffer*>& buffers) const;
    size_t EstimateSharedMemoryUsage(const std::string& kernel_name, int block_size) const;
};

// Template implementation for generic kernel launcher
template<typename KernelFunc, typename... Args>
bool CudaKernelLauncher::LaunchKernel(const std::string& kernel_name, KernelFunc kernel,
                                     const KernelLaunchParams& params, Args&&... args) {
    if (!RecordKernelLaunch(kernel_name, params, params.stream)) {
        return false;
    }

    cudaEvent_t start_event = nullptr, end_event = nullptr;
    if (profiling_enabled_) {
        start_event = GetTimingEvent();
        end_event = GetTimingEvent();
        cudaEventRecord(start_event, params.stream);
    }

    // Launch the kernel
    kernel<<<params.grid_size, params.block_size, params.shared_memory_size, params.stream>>>(
        std::forward<Args>(args)...);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        if (profiling_enabled_) {
            ReturnTimingEvent(start_event);
            ReturnTimingEvent(end_event);
        }
        return false;
    }

    if (profiling_enabled_) {
        cudaEventRecord(end_event, params.stream);
        cudaEventSynchronize(end_event);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_event, end_event);

        KernelMetrics metrics;
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

        ReturnTimingEvent(start_event);
        ReturnTimingEvent(end_event);
    }

    return true;
}

// Utility functions
KernelLaunchParams CalculateOptimalParams1D(size_t n, int min_block_size = 32, int max_block_size = 1024);
KernelLaunchParams CalculateOptimalParams2D(int rows, int cols, int tile_size = 16);
KernelLaunchParams CalculateOptimalParams3D(int depth, int height, int width, int tile_size = 8);

size_t EstimateSharedMemoryRequirement(const std::string& operation_type, int block_size);
int GetOptimalBlockSizeForMemoryBound(size_t element_size, size_t total_elements);
int GetOptimalBlockSizeForComputeBound(int arithmetic_intensity);

} // namespace cuda
} // namespace backends
} // namespace gemma