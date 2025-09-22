#pragma once

/**
 * @file cuda_backend.h
 * @brief NVIDIA CUDA backend implementation for Gemma.cpp
 *
 * This file provides a production-ready CUDA backend with:
 * - Multi-GPU support with device management
 * - cuBLAS and cuDNN integration
 * - Tensor Core utilization for FP16/INT8
 * - Memory pooling with optimized allocation
 * - Stream management for async operations
 * - Flash Attention v2 implementation
 * - Quantization support (INT8, INT4)
 */

#include "../backend_interface.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <queue>

namespace gemma {
namespace backends {
namespace cuda {

// Forward declarations
class CudaMemoryPool;
class CudaStreamManager;
class CudaKernelLauncher;

/**
 * @brief CUDA precision modes
 */
enum class CudaPrecision {
    FP32,           // 32-bit floating point
    FP16,           // 16-bit floating point (half precision)
    BF16,           // Brain floating point 16
    INT8,           // 8-bit integer quantization
    INT4            // 4-bit integer quantization
};

/**
 * @brief CUDA device properties and capabilities
 */
struct CudaDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    bool supports_tensor_cores;
    bool supports_fp16;
    bool supports_bf16;
    bool supports_int8;
    bool supports_cooperative_launch;
    size_t shared_memory_per_block;
    size_t l2_cache_size;
    int memory_bus_width;
    int memory_clock_rate;
};

/**
 * @brief CUDA buffer with advanced memory management
 */
struct CudaBuffer : public BackendBuffer {
    cudaStream_t stream = nullptr;
    CudaPrecision precision = CudaPrecision::FP32;
    int device_id = 0;
    bool is_managed_memory = false;
    bool is_pinned_memory = false;
    void* host_ptr = nullptr;  // For unified memory access

    CudaBuffer() = default;
    CudaBuffer(void* ptr, size_t sz, int dev_id, CudaPrecision prec = CudaPrecision::FP32)
        : BackendBuffer(ptr, sz, true), device_id(dev_id), precision(prec) {}
};

/**
 * @brief Configuration for CUDA backend
 */
struct CudaConfig {
    // Device configuration
    std::vector<int> device_ids;        // GPUs to use (-1 for all available)
    int primary_device = 0;             // Primary GPU for single-GPU operations
    bool enable_peer_access = true;     // Enable P2P memory access
    bool enable_unified_memory = false; // Use CUDA unified memory

    // Memory configuration
    size_t memory_pool_size = 0;        // Memory pool size (0 = auto)
    double memory_fraction = 0.9;       // Fraction of GPU memory to use
    bool enable_memory_pool = true;     // Use memory pooling
    size_t host_memory_pool_size = 0;   // Host pinned memory pool (0 = auto)

    // Stream configuration
    int num_streams = 4;                // Number of CUDA streams per device
    bool enable_async_execution = true; // Enable asynchronous operations

    // Compute configuration
    CudaPrecision default_precision = CudaPrecision::FP16;
    bool enable_tensor_cores = true;    // Use Tensor Cores when available
    bool enable_flash_attention = true; // Use Flash Attention kernels
    bool enable_cooperative_launch = true; // Use cooperative kernel launch

    // cuBLAS configuration
    bool enable_cublas = true;
    cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_16F;
    cublasGemmAlgo_t cublas_gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    // cuDNN configuration
    bool enable_cudnn = true;
    bool cudnn_benchmark_mode = true;   // Enable cuDNN benchmark mode

    // Performance tuning
    bool enable_kernel_timing = false;  // Profile kernel execution times
    bool enable_memory_debugging = false; // Track memory allocations
    int max_batch_size = 32;            // Maximum batch size for optimization
};

/**
 * @brief Performance monitoring for CUDA operations
 */
struct CudaPerformanceMonitor {
    std::atomic<double> total_compute_time_ms{0.0};
    std::atomic<double> total_memory_transfer_time_ms{0.0};
    std::atomic<size_t> memory_allocations{0};
    std::atomic<size_t> peak_memory_usage{0};
    std::atomic<size_t> current_memory_usage{0};
    std::atomic<size_t> total_kernel_launches{0};

    // Operation-specific timings
    std::unordered_map<std::string, double> operation_timings;
    std::mutex timing_mutex;

    void RecordKernelTime(const std::string& kernel_name, double time_ms);
    void RecordMemoryOp(size_t bytes, double time_ms);
    BackendMetrics GetMetrics() const;
    void Reset();
};

/**
 * @brief Main CUDA backend implementation
 */
class CudaBackend : public BackendInterface {
public:
    explicit CudaBackend(const CudaConfig& config = CudaConfig{});
    ~CudaBackend() override;

    // Backend interface implementation
    std::string GetName() const override { return "CUDA"; }
    std::string GetVersion() const override;
    bool Initialize() override;
    void Shutdown() override;
    bool IsAvailable() const override;
    bool SupportsCapability(BackendCapability capability) const override;

    // Device management
    int GetDeviceCount() const override;
    bool SetDevice(int device_id) override;
    int GetCurrentDevice() const override;

    // Multi-GPU support
    bool SetMultiDevice(const std::vector<int>& device_ids);
    std::vector<int> GetActiveDevices() const;
    bool EnablePeerAccess();

    // Memory management
    BackendBuffer AllocateBuffer(size_t size, size_t alignment = 32) override;
    CudaBuffer AllocateCudaBuffer(size_t size, CudaPrecision precision,
                                 int device_id = -1, size_t alignment = 32);
    void FreeBuffer(const BackendBuffer& buffer) override;
    bool CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) override;
    bool CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) override;
    bool CopyDeviceToDevice(const BackendBuffer& dst, const BackendBuffer& src, size_t size);
    void Synchronize() override;
    void SynchronizeDevice(int device_id);

    // Advanced memory operations
    bool CopyAsync(const BackendBuffer& dst, const BackendBuffer& src, size_t size, cudaStream_t stream = nullptr);
    bool PrefetchToDevice(const BackendBuffer& buffer, int device_id);
    bool MemsetAsync(const BackendBuffer& buffer, int value, size_t size, cudaStream_t stream = nullptr);

    // Performance monitoring
    BackendMetrics GetMetrics() const override;
    void ResetMetrics() override;
    CudaPerformanceMonitor& GetPerformanceMonitor() { return performance_monitor_; }

    // Matrix operations
    bool MatrixMultiply(const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
                       int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) override;
    bool MatrixVectorMultiply(const BackendBuffer& a, const BackendBuffer& x, const BackendBuffer& y,
                             int m, int n) override;

    // Advanced GEMM operations
    bool BatchedMatrixMultiply(const std::vector<BackendBuffer>& a_batch,
                              const std::vector<BackendBuffer>& b_batch,
                              const std::vector<BackendBuffer>& c_batch,
                              int m, int n, int k, float alpha = 1.0f, float beta = 0.0f);

    bool StridedBatchedMatrixMultiply(const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
                                     int batch_count, int m, int n, int k,
                                     long long stride_a, long long stride_b, long long stride_c,
                                     float alpha = 1.0f, float beta = 0.0f);

    // Attention operations
    bool ComputeAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                         const BackendBuffer& values, const BackendBuffer& output,
                         int batch_size, int seq_len, int head_dim, int num_heads) override;

    // Flash Attention implementation
    bool ComputeFlashAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                              const BackendBuffer& values, const BackendBuffer& output,
                              int batch_size, int seq_len, int head_dim, int num_heads,
                              float scale = 0.0f, bool causal = false);

    // Multi-head attention with KV cache
    bool ComputeMultiHeadAttentionWithCache(const BackendBuffer& queries,
                                           const BackendBuffer& key_cache,
                                           const BackendBuffer& value_cache,
                                           const BackendBuffer& output,
                                           int batch_size, int seq_len, int kv_seq_len,
                                           int head_dim, int num_heads, int num_kv_heads);

    // Activation functions
    bool ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;
    bool ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;
    bool ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;

    // Advanced activation functions
    bool ApplySiLU(const BackendBuffer& input, const BackendBuffer& output, size_t size);
    bool ApplyLayerNorm(const BackendBuffer& input, const BackendBuffer& output,
                       const BackendBuffer& gamma, const BackendBuffer& beta,
                       int batch_size, int hidden_size, float epsilon = 1e-5f);
    bool ApplyRMSNorm(const BackendBuffer& input, const BackendBuffer& output,
                     const BackendBuffer& weight, int batch_size, int hidden_size, float epsilon = 1e-5f);

    // Quantization operations
    bool QuantizeINT8(const BackendBuffer& input, const BackendBuffer& output,
                     const BackendBuffer& scale, size_t size);
    bool DequantizeINT8(const BackendBuffer& input, const BackendBuffer& output,
                       const BackendBuffer& scale, size_t size);
    bool QuantizeINT4(const BackendBuffer& input, const BackendBuffer& output,
                     const BackendBuffer& scale, const BackendBuffer& zero_point, size_t size);
    bool DequantizeINT4(const BackendBuffer& input, const BackendBuffer& output,
                       const BackendBuffer& scale, const BackendBuffer& zero_point, size_t size);

    // Fused operations
    bool FusedLinearGELU(const BackendBuffer& input, const BackendBuffer& weight,
                        const BackendBuffer& bias, const BackendBuffer& output,
                        int batch_size, int input_size, int output_size);

    bool FusedMultiplyAdd(const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
                         const BackendBuffer& output, size_t size, float alpha = 1.0f, float beta = 1.0f);

    // Utility functions
    CudaDeviceInfo GetDeviceInfo(int device_id) const;
    std::vector<CudaDeviceInfo> GetAllDeviceInfo() const;
    bool IsTensorCoreAvailable(int device_id) const;
    size_t GetAvailableMemory(int device_id) const;

    // Stream management
    cudaStream_t GetStream(int device_id = -1, int stream_id = 0);
    bool CreateEvent(cudaEvent_t* event);
    bool RecordEvent(cudaEvent_t event, cudaStream_t stream = nullptr);
    bool WaitEvent(cudaEvent_t event, cudaStream_t stream = nullptr);

    // Advanced configuration
    bool SetTensorCoreUsage(bool enabled);
    bool SetMemoryPoolFraction(double fraction);
    CudaConfig& GetConfig() { return config_; }
    const CudaConfig& GetConfig() const { return config_; }

private:
    CudaConfig config_;
    std::vector<CudaDeviceInfo> device_info_;
    std::vector<int> active_devices_;
    int current_device_;
    bool initialized_;

    // cuBLAS handles (one per device)
    std::unordered_map<int, cublasHandle_t> cublas_handles_;

    // cuDNN handles (one per device)
    std::unordered_map<int, cudnnHandle_t> cudnn_handles_;

    // Memory management
    std::unique_ptr<CudaMemoryPool> memory_pool_;

    // Stream management
    std::unique_ptr<CudaStreamManager> stream_manager_;

    // Kernel launcher
    std::unique_ptr<CudaKernelLauncher> kernel_launcher_;

    // Performance monitoring
    CudaPerformanceMonitor performance_monitor_;

    // Thread safety
    mutable std::mutex backend_mutex_;

    // Internal helper methods
    bool InitializeDevice(int device_id);
    bool InitializeCuBLAS(int device_id);
    bool InitializeCuDNN(int device_id);
    void CleanupDevice(int device_id);

    bool CheckCudaError(cudaError_t error, const char* operation) const;
    bool CheckCublasError(cublasStatus_t status, const char* operation) const;
    bool CheckCudnnError(cudnnStatus_t status, const char* operation) const;

    size_t GetElementSize(CudaPrecision precision) const;
    bool IsDeviceCompatible(int device_id, CudaPrecision precision) const;

    // Precision conversion utilities
    bool ConvertPrecision(const BackendBuffer& input, const BackendBuffer& output,
                         CudaPrecision src_precision, CudaPrecision dst_precision, size_t count);
};

/**
 * @brief Factory function for creating CUDA backend instances
 */
std::unique_ptr<BackendInterface> CreateCudaBackend(const CudaConfig& config = CudaConfig{});

/**
 * @brief Check if CUDA is available on the system
 */
bool IsCudaAvailable();

/**
 * @brief Get CUDA runtime version
 */
std::string GetCudaVersion();

/**
 * @brief Get optimal CUDA configuration for the current system
 */
CudaConfig GetOptimalCudaConfig();

} // namespace cuda
} // namespace backends
} // namespace gemma