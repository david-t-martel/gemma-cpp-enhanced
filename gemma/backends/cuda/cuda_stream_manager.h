#pragma once

/**
 * @file cuda_stream_manager.h
 * @brief Advanced CUDA stream management for asynchronous operations
 *
 * Provides optimized stream management for:
 * - Multi-stream execution with automatic load balancing
 * - Stream priority management
 * - Dependency tracking and synchronization
 * - Event-based synchronization
 * - Stream pool management and reuse
 */

#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>

namespace gemma {
namespace backends {
namespace cuda {

/**
 * @brief Stream priority levels
 */
enum class StreamPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Stream types for different operations
 */
enum class StreamType {
    COMPUTE,        // General compute operations
    MEMORY_COPY,    // Memory transfer operations
    KERNEL_LAUNCH,  // Dedicated kernel launches
    ATTENTION,      // Attention-specific operations
    ACTIVATION,     // Activation function operations
    COMMUNICATION   // Multi-GPU communication
};

/**
 * @brief Stream configuration
 */
struct StreamConfig {
    StreamType type = StreamType::COMPUTE;
    StreamPriority priority = StreamPriority::NORMAL;
    bool blocking = false;           // Whether stream should block other streams
    int device_id = -1;             // Device for the stream
    size_t memory_pool_size = 0;    // Stream-specific memory pool size
    bool enable_timing = false;     // Enable timing events
};

/**
 * @brief Stream statistics
 */
struct StreamStats {
    size_t total_operations = 0;
    size_t active_operations = 0;
    double total_execution_time_ms = 0.0;
    double average_execution_time_ms = 0.0;
    size_t memory_operations = 0;
    size_t kernel_launches = 0;
    std::chrono::steady_clock::time_point last_operation_time;

    // Load balancing metrics
    double load_factor = 0.0;       // 0.0 = idle, 1.0 = fully utilized
    size_t queue_depth = 0;         // Number of pending operations
};

/**
 * @brief Managed CUDA stream wrapper
 */
class ManagedStream {
public:
    explicit ManagedStream(const StreamConfig& config);
    ~ManagedStream();

    // Stream access
    cudaStream_t GetStream() const { return stream_; }
    int GetDeviceId() const { return config_.device_id; }
    StreamType GetType() const { return config_.type; }
    StreamPriority GetPriority() const { return config_.priority; }

    // Stream operations
    bool Initialize();
    void Destroy();
    bool IsValid() const { return stream_ != nullptr; }

    // Synchronization
    bool Synchronize();
    bool Query() const;  // Check if all operations are complete
    bool WaitForEvent(cudaEvent_t event);
    bool RecordEvent(cudaEvent_t event);

    // Statistics and monitoring
    const StreamStats& GetStats() const { return stats_; }
    void ResetStats();
    void RecordOperation(double execution_time_ms = 0.0);
    void RecordMemoryOperation();
    void RecordKernelLaunch();

    // Load balancing
    void UpdateLoadFactor();
    bool IsIdle() const;
    bool IsOverloaded() const;
    size_t GetQueueDepth() const { return stats_.queue_depth; }

    // Priority and configuration
    bool SetPriority(StreamPriority priority);
    bool UpdateConfig(const StreamConfig& config);

private:
    StreamConfig config_;
    cudaStream_t stream_;
    mutable std::mutex stats_mutex_;
    StreamStats stats_;
    std::atomic<bool> initialized_{false};

    bool CreateStream();
    int GetCudaPriority(StreamPriority priority) const;
};

/**
 * @brief Stream dependency graph for managing execution order
 */
class StreamDependencyGraph {
public:
    StreamDependencyGraph() = default;
    ~StreamDependencyGraph() = default;

    // Dependency management
    bool AddDependency(cudaStream_t dependent, cudaStream_t prerequisite);
    bool RemoveDependency(cudaStream_t dependent, cudaStream_t prerequisite);
    void ClearDependencies(cudaStream_t stream);

    // Execution ordering
    std::vector<cudaStream_t> GetExecutionOrder() const;
    bool HasCircularDependency() const;
    bool CanExecute(cudaStream_t stream) const;

    // Synchronization helpers
    bool SynchronizeDependencies(cudaStream_t stream);
    bool WaitForPrerequisites(cudaStream_t stream);

private:
    mutable std::mutex graph_mutex_;
    std::unordered_map<cudaStream_t, std::vector<cudaStream_t>> dependencies_;
    std::unordered_map<cudaStream_t, std::vector<cudaStream_t>> dependents_;

    bool HasCircularDependencyDFS(cudaStream_t stream,
                                 std::unordered_set<cudaStream_t>& visited,
                                 std::unordered_set<cudaStream_t>& recursion_stack) const;
};

/**
 * @brief Main CUDA stream manager
 */
class CudaStreamManager {
public:
    explicit CudaStreamManager(const CudaConfig& config, const std::vector<int>& devices);
    ~CudaStreamManager();

    // Initialization and cleanup
    bool Initialize();
    void Cleanup();

    // Stream creation and management
    cudaStream_t GetStream(StreamType type = StreamType::COMPUTE, int device_id = -1);
    cudaStream_t CreateStream(const StreamConfig& config);
    bool DestroyStream(cudaStream_t stream);

    // Specialized stream getters
    cudaStream_t GetComputeStream(int device_id = -1);
    cudaStream_t GetMemoryStream(int device_id = -1);
    cudaStream_t GetAttentionStream(int device_id = -1);
    cudaStream_t GetActivationStream(int device_id = -1);
    cudaStream_t GetCommunicationStream(int device_id = -1);

    // Stream pool management
    bool ReturnStreamToPool(cudaStream_t stream);
    cudaStream_t GetStreamFromPool(StreamType type, int device_id);
    void FlushStreamPool(int device_id = -1);

    // Load balancing
    cudaStream_t GetLeastLoadedStream(StreamType type, int device_id = -1);
    void UpdateStreamLoadFactors();
    void BalanceStreams(int device_id = -1);

    // Synchronization
    bool SynchronizeAllStreams(int device_id = -1);
    bool SynchronizeStreamType(StreamType type, int device_id = -1);
    bool WaitForAllStreams(cudaStream_t waiting_stream, int device_id = -1);

    // Event management
    cudaEvent_t CreateEvent(bool enable_timing = false, bool blocking = false);
    bool DestroyEvent(cudaEvent_t event);
    bool RecordEvent(cudaEvent_t event, cudaStream_t stream);
    bool WaitEvent(cudaEvent_t event, cudaStream_t stream);
    bool EventElapsedTime(cudaEvent_t start, cudaEvent_t end, float* time_ms);

    // Dependency management
    bool AddStreamDependency(cudaStream_t dependent, cudaStream_t prerequisite);
    bool RemoveStreamDependency(cudaStream_t dependent, cudaStream_t prerequisite);
    bool SynchronizeWithDependencies(cudaStream_t stream);

    // Stream priorities
    bool SetStreamPriority(cudaStream_t stream, StreamPriority priority);
    StreamPriority GetStreamPriority(cudaStream_t stream) const;

    // Statistics and monitoring
    StreamStats GetStreamStats(cudaStream_t stream) const;
    std::vector<StreamStats> GetAllStreamStats(int device_id = -1) const;
    void ResetAllStats();

    // Configuration
    void SetMaxStreamsPerDevice(int max_streams);
    int GetMaxStreamsPerDevice() const;
    void SetDefaultStreamConfig(const StreamConfig& config);
    const StreamConfig& GetDefaultStreamConfig() const;

    // Advanced features
    bool EnableStreamCapture(cudaStream_t stream);
    bool EndStreamCapture(cudaStream_t stream, cudaGraph_t* graph);
    bool LaunchGraph(cudaGraphExec_t graph_exec, cudaStream_t stream);

    // Device management
    std::vector<int> GetActiveDevices() const;
    bool IsDeviceActive(int device_id) const;
    size_t GetStreamCount(int device_id = -1) const;

    // Debugging and diagnostics
    void PrintStreamInfo(int device_id = -1) const;
    bool ValidateStreams() const;
    void DumpStreamGraph() const;

private:
    struct DeviceStreams {
        std::vector<std::unique_ptr<ManagedStream>> streams;
        std::unordered_map<StreamType, std::queue<cudaStream_t>> stream_pools;
        std::unordered_map<cudaStream_t, std::unique_ptr<ManagedStream>> stream_map;
        mutable std::mutex mutex;

        // Load balancing
        std::atomic<size_t> round_robin_counter{0};
        std::chrono::steady_clock::time_point last_balance_time;
    };

    CudaConfig config_;
    std::vector<int> devices_;
    int max_streams_per_device_;
    StreamConfig default_config_;

    // Per-device stream management
    std::unordered_map<int, std::unique_ptr<DeviceStreams>> device_streams_;

    // Global stream tracking
    std::unordered_map<cudaStream_t, int> stream_to_device_;
    std::unordered_map<cudaStream_t, StreamConfig> stream_configs_;
    mutable std::mutex global_mutex_;

    // Event pool
    std::queue<cudaEvent_t> event_pool_;
    std::unordered_set<cudaEvent_t> active_events_;
    mutable std::mutex event_mutex_;

    // Dependency management
    std::unique_ptr<StreamDependencyGraph> dependency_graph_;

    // CUDA graphs
    std::unordered_map<cudaStream_t, cudaGraph_t> stream_graphs_;
    std::unordered_map<cudaGraph_t, cudaGraphExec_t> graph_executables_;

    // Helper methods
    bool InitializeDevice(int device_id);
    void CleanupDevice(int device_id);
    DeviceStreams* GetDeviceStreams(int device_id);
    const DeviceStreams* GetDeviceStreams(int device_id) const;
    ManagedStream* FindManagedStream(cudaStream_t stream);
    const ManagedStream* FindManagedStream(cudaStream_t stream) const;

    cudaStream_t CreateDeviceStream(const StreamConfig& config, int device_id);
    bool AddStreamToDevice(std::unique_ptr<ManagedStream> stream, int device_id);
    bool RemoveStreamFromDevice(cudaStream_t stream, int device_id);

    void UpdateGlobalStats();
    int SelectBestDevice(StreamType type, int requested_device) const;
    StreamConfig GetOptimalStreamConfig(StreamType type, int device_id) const;
};

/**
 * @brief Stream RAII wrapper for automatic cleanup
 */
class ScopedStream {
public:
    ScopedStream(CudaStreamManager* manager, StreamType type, int device_id = -1);
    ScopedStream(CudaStreamManager* manager, const StreamConfig& config);
    ~ScopedStream();

    // Non-copyable, movable
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;
    ScopedStream(ScopedStream&& other) noexcept;
    ScopedStream& operator=(ScopedStream&& other) noexcept;

    cudaStream_t Get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    bool IsValid() const { return stream_ != nullptr; }

    bool Synchronize();
    bool Query() const;

private:
    CudaStreamManager* manager_;
    cudaStream_t stream_;
    bool owns_stream_;
};

// Utility functions
StreamConfig GetDefaultStreamConfig(StreamType type);
StreamConfig GetOptimalStreamConfig(StreamType type, int device_id, const CudaDeviceInfo& device_info);
size_t EstimateOptimalStreamCount(const CudaDeviceInfo& device_info, StreamType type);

} // namespace cuda
} // namespace backends
} // namespace gemma