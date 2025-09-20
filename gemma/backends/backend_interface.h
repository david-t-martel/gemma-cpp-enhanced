#pragma once

/**
 * @file backend_interface.h
 * @brief Abstract interface for hardware acceleration backends
 */

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <cstddef>

namespace gemma {
namespace backends {

/**
 * @brief Memory buffer for backend operations
 */
struct BackendBuffer {
    void* data = nullptr;
    size_t size = 0;
    size_t alignment = 32;
    bool is_device_memory = false;

    BackendBuffer() = default;
    BackendBuffer(void* ptr, size_t sz, bool device = false)
        : data(ptr), size(sz), is_device_memory(device) {}
    virtual ~BackendBuffer();

    bool IsValid() const;
    void Reset();
};

/**
 * @brief Backend capability flags
 */
enum class BackendCapability {
    MATRIX_MULTIPLICATION,
    ATTENTION_COMPUTATION,
    ACTIVATION_FUNCTIONS,
    MEMORY_POOLING,
    ASYNC_EXECUTION,
    MULTI_PRECISION
};

/**
 * @brief Backend performance metrics
 */
struct BackendMetrics {
    double compute_throughput_gflops = 0.0;
    double memory_bandwidth_gbps = 0.0;
    double latency_ms = 0.0;
    size_t memory_usage_bytes = 0;
    size_t peak_memory_bytes = 0;
    size_t num_operations = 0;
    double total_execution_time_ms = 0.0;
    double cache_hit_rate = 0.0;
    double power_consumption_watts = 0.0;

    void Reset();
    BackendMetrics operator+(const BackendMetrics& other) const;
};

/**
 * @brief Abstract base class for hardware acceleration backends
 */
class BackendInterface {
public:
    BackendInterface();
    virtual ~BackendInterface();

    /**
     * @brief Get backend name
     * @return Backend name (e.g., "CUDA", "OpenCL", "Metal")
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Get backend version
     * @return Version string
     */
    virtual std::string GetVersion() const = 0;

    /**
     * @brief Initialize the backend
     * @return true if initialization successful
     */
    virtual bool Initialize() = 0;

    /**
     * @brief Shutdown the backend
     */
    virtual void Shutdown() = 0;

    /**
     * @brief Check if backend is available
     * @return true if backend can be used
     */
    virtual bool IsAvailable() const = 0;

    /**
     * @brief Check if backend supports a capability
     * @param capability Capability to check
     * @return true if supported
     */
    virtual bool SupportsCapability(BackendCapability capability) const = 0;

    /**
     * @brief Get device count
     * @return Number of available devices
     */
    virtual int GetDeviceCount() const = 0;

    /**
     * @brief Set active device
     * @param device_id Device ID to use
     * @return true if successful
     */
    virtual bool SetDevice(int device_id) = 0;

    /**
     * @brief Get current device ID
     * @return Active device ID
     */
    virtual int GetCurrentDevice() const = 0;

    /**
     * @brief Allocate memory buffer
     * @param size Size in bytes
     * @param alignment Memory alignment requirement
     * @return Allocated buffer, or empty buffer on failure
     */
    virtual BackendBuffer AllocateBuffer(size_t size, size_t alignment = 32) = 0;

    /**
     * @brief Free memory buffer
     * @param buffer Buffer to free
     */
    virtual void FreeBuffer(const BackendBuffer& buffer) = 0;

    /**
     * @brief Copy data to device
     * @param dst Destination buffer (device)
     * @param src Source buffer (host)
     * @param size Number of bytes to copy
     * @return true if successful
     */
    virtual bool CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) = 0;

    /**
     * @brief Copy data from device
     * @param dst Destination buffer (host)
     * @param src Source buffer (device)
     * @param size Number of bytes to copy
     * @return true if successful
     */
    virtual bool CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) = 0;

    /**
     * @brief Synchronize device operations
     */
    virtual void Synchronize() = 0;

    /**
     * @brief Get performance metrics
     * @return Current performance metrics
     */
    virtual BackendMetrics GetMetrics() const = 0;

    /**
     * @brief Reset performance counters
     */
    virtual void ResetMetrics() = 0;

    // Matrix operations
    /**
     * @brief Perform matrix multiplication: C = A * B
     * @param a Matrix A buffer
     * @param b Matrix B buffer
     * @param c Matrix C buffer (output)
     * @param m Rows of A
     * @param n Columns of B
     * @param k Columns of A / Rows of B
     * @param alpha Scaling factor for A*B
     * @param beta Scaling factor for existing C
     * @return true if successful
     */
    virtual bool MatrixMultiply(
        const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
        int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) = 0;

    /**
     * @brief Perform matrix-vector multiplication: y = A * x
     * @param a Matrix A buffer
     * @param x Vector x buffer
     * @param y Vector y buffer (output)
     * @param m Rows of A
     * @param n Columns of A
     * @return true if successful
     */
    virtual bool MatrixVectorMultiply(
        const BackendBuffer& a, const BackendBuffer& x, const BackendBuffer& y,
        int m, int n) = 0;

    // Attention operations
    /**
     * @brief Compute attention weights
     * @param queries Query tensor
     * @param keys Key tensor
     * @param values Value tensor
     * @param output Output tensor
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param head_dim Head dimension
     * @param num_heads Number of attention heads
     * @return true if successful
     */
    virtual bool ComputeAttention(
        const BackendBuffer& queries, const BackendBuffer& keys, 
        const BackendBuffer& values, const BackendBuffer& output,
        int batch_size, int seq_len, int head_dim, int num_heads) = 0;

    // Activation functions
    /**
     * @brief Apply ReLU activation
     * @param input Input buffer
     * @param output Output buffer
     * @param size Number of elements
     * @return true if successful
     */
    virtual bool ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) = 0;

    /**
     * @brief Apply GELU activation
     * @param input Input buffer
     * @param output Output buffer
     * @param size Number of elements
     * @return true if successful
     */
    virtual bool ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) = 0;

    /**
     * @brief Apply Softmax activation
     * @param input Input buffer
     * @param output Output buffer
     * @param size Number of elements
     * @return true if successful
     */
    virtual bool ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) = 0;

    // Non-virtual helper methods
    bool IsInitialized() const;
    void EnableProfiling(bool enable);
    bool IsProfilingEnabled() const;
    std::string GetErrorString() const;

protected:
    void SetError(const std::string& error);
    void ClearError();

private:
    bool initialized_;
    bool profiling_enabled_;
    std::string last_error_;
};

/**
 * @brief Factory function type for creating backend instances
 */
using BackendFactory = std::function<std::unique_ptr<BackendInterface>()>;

// Utility functions
std::string BackendCapabilityToString(BackendCapability capability);
bool ValidateBufferAlignment(const BackendBuffer& buffer, size_t required_alignment);
size_t AlignSize(size_t size, size_t alignment);
bool CopyBuffer(const BackendBuffer& dst, const BackendBuffer& src, size_t size = 0);

} // namespace backends
} // namespace gemma