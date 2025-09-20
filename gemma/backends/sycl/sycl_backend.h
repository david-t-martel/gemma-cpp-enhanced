#pragma once

/**
 * @file sycl_backend.h
 * @brief Intel SYCL backend for Gemma.cpp inference
 *
 * Provides hardware acceleration using Intel oneAPI with support for:
 * - Intel GPUs (Arc, Flex, Max series)
 * - Intel NPUs (Core Ultra processors)
 * - CPU fallback with SYCL
 * - oneMKL optimized linear algebra
 */

#include "../backend_interface.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <mutex>

namespace gemma {
namespace backends {
namespace sycl {

/**
 * @brief SYCL device types supported by the backend
 */
enum class SyclDeviceType {
    GPU,      // Intel Arc, Flex, Max GPUs
    NPU,      // Intel NPU (Core Ultra)
    CPU,      // CPU fallback
    UNKNOWN
};

/**
 * @brief SYCL device information
 */
struct SyclDeviceInfo {
    int device_id;
    SyclDeviceType type;
    std::string name;
    std::string vendor;
    size_t max_memory_bytes;
    size_t max_work_group_size;
    bool supports_fp16;
    bool supports_dp4a;  // DP4A for integer optimizations
    bool supports_unified_memory;
    std::string driver_version;
    ::sycl::device device;
};

/**
 * @brief SYCL memory allocation tracking
 */
struct SyclMemoryInfo {
    void* device_ptr;
    size_t size;
    size_t alignment;
    ::sycl::usm::alloc alloc_type;
    int device_id;
    std::chrono::high_resolution_clock::time_point alloc_time;
};

/**
 * @brief Performance profiling data for SYCL operations
 */
struct SyclProfileData {
    std::string operation_name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    size_t memory_transferred;
    size_t flops_performed;
    int device_id;
};

/**
 * @brief Intel SYCL Backend Implementation
 */
class SyclBackend : public BackendInterface {
public:
    SyclBackend();
    ~SyclBackend() override;

    // Backend interface implementation
    std::string GetName() const override { return "Intel SYCL"; }
    std::string GetVersion() const override;
    bool Initialize() override;
    void Shutdown() override;
    bool IsAvailable() const override;
    bool SupportsCapability(BackendCapability capability) const override;

    // Device management
    int GetDeviceCount() const override;
    bool SetDevice(int device_id) override;
    int GetCurrentDevice() const override;

    // Memory management
    BackendBuffer AllocateBuffer(size_t size, size_t alignment = 32) override;
    void FreeBuffer(const BackendBuffer& buffer) override;
    bool CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) override;
    bool CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) override;
    void Synchronize() override;

    // Performance monitoring
    BackendMetrics GetMetrics() const override;
    void ResetMetrics() override;

    // Matrix operations
    bool MatrixMultiply(
        const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
        int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) override;

    bool MatrixVectorMultiply(
        const BackendBuffer& a, const BackendBuffer& x, const BackendBuffer& y,
        int m, int n) override;

    // Attention operations
    bool ComputeAttention(
        const BackendBuffer& queries, const BackendBuffer& keys,
        const BackendBuffer& values, const BackendBuffer& output,
        int batch_size, int seq_len, int head_dim, int num_heads) override;

    // Activation functions
    bool ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;
    bool ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;
    bool ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) override;

    // SYCL-specific methods

    /**
     * @brief Get available SYCL devices
     * @return Vector of device information
     */
    std::vector<SyclDeviceInfo> GetAvailableDevices() const;

    /**
     * @brief Get current SYCL queue
     * @return Reference to active queue
     */
    ::sycl::queue& GetQueue() { return *current_queue_; }

    /**
     * @brief Get device info for current device
     * @return Device information
     */
    const SyclDeviceInfo& GetCurrentDeviceInfo() const;

    /**
     * @brief Set memory allocation strategy
     * @param use_usm_device Use USM device allocations (vs shared)
     */
    void SetMemoryStrategy(bool use_usm_device);

    /**
     * @brief Enable/disable performance profiling
     * @param enable Enable profiling
     */
    void EnableProfiling(bool enable) { profiling_enabled_ = enable; }

    /**
     * @brief Get detailed profiling data
     * @return Vector of profiling records
     */
    std::vector<SyclProfileData> GetProfilingData() const;

    /**
     * @brief Clear profiling data
     */
    void ClearProfilingData();

    /**
     * @brief Force device selection by type preference
     * @param prefer_gpu Prefer GPU over NPU
     * @param fallback_cpu Allow CPU fallback
     * @return true if suitable device found
     */
    bool SelectDeviceByType(bool prefer_gpu = true, bool fallback_cpu = true);

    /**
     * @brief Get memory usage statistics
     * @return Map of memory statistics
     */
    std::map<std::string, size_t> GetMemoryStats() const;

private:
    // Internal initialization
    bool InitializeDevices();
    bool InitializeQueues();
    bool TestDeviceCapabilities(const ::sycl::device& device);

    // Device management
    bool SelectBestDevice();
    SyclDeviceType GetDeviceType(const ::sycl::device& device) const;
    std::string GetDeviceName(const ::sycl::device& device) const;

    // Memory management helpers
    void* AllocateDeviceMemory(size_t size, size_t alignment);
    void FreeDeviceMemory(void* ptr);
    void TrackMemoryAllocation(void* ptr, size_t size, size_t alignment);
    void UntrackMemoryAllocation(void* ptr);

    // Performance tracking
    void BeginProfiling(const std::string& operation_name);
    void EndProfiling(const std::string& operation_name, size_t memory_transferred = 0, size_t flops = 0);
    void UpdateMetrics(double execution_time_ms, size_t memory_transferred, size_t flops);

    // Error handling
    void HandleSyclException(const ::sycl::exception& e, const std::string& operation) const;
    bool CheckDeviceError(const ::sycl::queue& queue, const std::string& operation) const;

    // Backend state
    bool initialized_;
    bool profiling_enabled_;
    bool use_usm_device_;
    int current_device_id_;

    // SYCL objects
    std::vector<SyclDeviceInfo> available_devices_;
    std::unique_ptr<::sycl::queue> current_queue_;
    std::map<int, std::unique_ptr<::sycl::queue>> device_queues_;

    // Memory tracking
    std::map<void*, SyclMemoryInfo> memory_allocations_;
    size_t total_allocated_memory_;
    size_t peak_memory_usage_;
    mutable std::mutex memory_mutex_;

    // Performance metrics
    mutable BackendMetrics metrics_;
    std::vector<SyclProfileData> profiling_data_;
    mutable std::mutex metrics_mutex_;

    // oneMKL handle for optimized operations
    oneapi::mkl::transpose trans_n_;
    oneapi::mkl::transpose trans_t_;

    // Static device detection
    static std::vector<SyclDeviceInfo> DetectDevices();
    static bool IsIntelDevice(const ::sycl::device& device);
    static bool SupportsRequiredExtensions(const ::sycl::device& device);
};

/**
 * @brief Factory function for creating SYCL backend
 * @return Unique pointer to SYCL backend instance
 */
std::unique_ptr<BackendInterface> CreateSyclBackend();

/**
 * @brief Check if SYCL backend is available on this system
 * @return true if SYCL runtime and compatible devices found
 */
bool IsSyclBackendAvailable();

/**
 * @brief Get Intel oneAPI toolkit version
 * @return Version string or empty if not available
 */
std::string GetOneAPIVersion();

} // namespace sycl
} // namespace backends
} // namespace gemma