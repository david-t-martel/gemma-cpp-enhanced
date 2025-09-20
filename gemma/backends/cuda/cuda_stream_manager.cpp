/**
 * @file cuda_stream_manager.cpp
 * @brief Advanced CUDA stream management implementation for Gemma.cpp
 */

#include "cuda_stream_manager.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace gemma {
namespace backends {
namespace cuda {

// Utility functions implementation
StreamConfig GetDefaultStreamConfig(StreamType type) {
    StreamConfig config;
    config.type = type;
    config.priority = StreamPriority::NORMAL;
    config.blocking = false;
    config.device_id = -1;
    config.memory_pool_size = 0;
    config.enable_timing = false;
    return config;
}

StreamConfig GetOptimalStreamConfig(StreamType type, int device_id, const CudaDeviceInfo& device_info) {
    StreamConfig config = GetDefaultStreamConfig(type);
    config.device_id = device_id;
    
    // Optimize based on stream type
    switch (type) {
        case StreamType::COMPUTE:
            config.priority = StreamPriority::HIGH;
            config.memory_pool_size = device_info.total_memory / 8;
            config.enable_timing = true;
            break;
        case StreamType::MEMORY_COPY:
            config.priority = StreamPriority::NORMAL;
            config.blocking = false;
            break;
        case StreamType::ATTENTION:
            config.priority = StreamPriority::HIGH;
            config.memory_pool_size = device_info.total_memory / 4;
            config.enable_timing = true;
            break;
        case StreamType::COMMUNICATION:
            config.priority = StreamPriority::CRITICAL;
            config.blocking = true;
            break;
        default:
            break;
    }
    
    return config;
}

size_t EstimateOptimalStreamCount(const CudaDeviceInfo& device_info, StreamType type) {
    // Base stream count on multiprocessor count and stream type
    size_t base_count = device_info.multiprocessor_count / 8;
    
    switch (type) {
        case StreamType::COMPUTE:
            return std::max(size_t(2), base_count);
        case StreamType::MEMORY_COPY:
            return std::max(size_t(1), base_count / 2);
        case StreamType::ATTENTION:
            return std::max(size_t(1), base_count / 4);
        default:
            return std::max(size_t(1), base_count / 4);
    }
}

// ManagedStream implementation
ManagedStream::ManagedStream(const StreamConfig& config)
    : config_(config), stream_(nullptr) {
}

ManagedStream::~ManagedStream() {
    Destroy();
}

bool ManagedStream::Initialize() {
    if (initialized_.load()) {
        return true;
    }

    if (!CreateStream()) {
        return false;
    }

    initialized_.store(true);
    return true;
}

void ManagedStream::Destroy() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    initialized_.store(false);
}

bool ManagedStream::CreateStream() {
    if (config_.device_id >= 0) {
        cudaError_t error = cudaSetDevice(config_.device_id);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set device " << config_.device_id 
                      << ": " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }

    unsigned int flags = cudaStreamDefault;
    if (!config_.blocking) {
        flags = cudaStreamNonBlocking;
    }

    cudaError_t error;
    if (config_.priority != StreamPriority::NORMAL) {
        int cuda_priority = GetCudaPriority(config_.priority);
        error = cudaStreamCreateWithPriority(&stream_, flags, cuda_priority);
    } else {
        error = cudaStreamCreateWithFlags(&stream_, flags);
    }

    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

bool ManagedStream::Synchronize() {
    if (!stream_) return false;
    
    auto start_time = std::chrono::steady_clock::now();
    cudaError_t error = cudaStreamSynchronize(stream_);
    auto end_time = std::chrono::steady_clock::now();
    
    if (error == cudaSuccess) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        RecordOperation(duration.count() / 1000.0);
        return true;
    }
    
    return false;
}

bool ManagedStream::Query() const {
    if (!stream_) return false;
    
    cudaError_t error = cudaStreamQuery(stream_);
    return error == cudaSuccess;
}

bool ManagedStream::WaitForEvent(cudaEvent_t event) {
    if (!stream_ || !event) return false;
    
    cudaError_t error = cudaStreamWaitEvent(stream_, event, 0);
    return error == cudaSuccess;
}

bool ManagedStream::RecordEvent(cudaEvent_t event) {
    if (!stream_ || !event) return false;
    
    cudaError_t error = cudaEventRecord(event, stream_);
    return error == cudaSuccess;
}

void ManagedStream::RecordOperation(double execution_time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_operations++;
    stats_.total_execution_time_ms += execution_time_ms;
    stats_.average_execution_time_ms = stats_.total_execution_time_ms / stats_.total_operations;
    stats_.last_operation_time = std::chrono::steady_clock::now();
    
    UpdateLoadFactor();
}

void ManagedStream::RecordMemoryOperation() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.memory_operations++;
    RecordOperation();
}

void ManagedStream::RecordKernelLaunch() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.kernel_launches++;
    RecordOperation();
}

void ManagedStream::UpdateLoadFactor() {
    // Simple load factor calculation based on recent activity
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - stats_.last_operation_time).count();
    
    // Load factor decreases over time since last operation
    if (time_since_last < 100) {
        stats_.load_factor = 1.0;
    } else if (time_since_last < 1000) {
        stats_.load_factor = 1.0 - (time_since_last - 100) / 900.0;
    } else {
        stats_.load_factor = 0.0;
    }
    
    stats_.load_factor = std::max(0.0, std::min(1.0, stats_.load_factor));
}

bool ManagedStream::IsIdle() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.load_factor < 0.1 && Query();
}

bool ManagedStream::IsOverloaded() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.load_factor > 0.9 || stats_.queue_depth > 10;
}

void ManagedStream::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = StreamStats{};
}

bool ManagedStream::SetPriority(StreamPriority priority) {
    if (priority == config_.priority) {
        return true;
    }
    
    // Destroy and recreate stream with new priority
    config_.priority = priority;
    Destroy();
    return Initialize();
}

bool ManagedStream::UpdateConfig(const StreamConfig& config) {
    if (config.device_id != config_.device_id || 
        config.priority != config_.priority ||
        config.blocking != config_.blocking) {
        // Need to recreate stream
        config_ = config;
        Destroy();
        return Initialize();
    }
    
    config_ = config;
    return true;
}

int ManagedStream::GetCudaPriority(StreamPriority priority) const {
    switch (priority) {
        case StreamPriority::LOW: return 1;
        case StreamPriority::NORMAL: return 0;
        case StreamPriority::HIGH: return -1;
        case StreamPriority::CRITICAL: return -2;
        default: return 0;
    }
}

// StreamDependencyGraph implementation
bool StreamDependencyGraph::AddDependency(cudaStream_t dependent, cudaStream_t prerequisite) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    dependencies_[dependent].push_back(prerequisite);
    dependents_[prerequisite].push_back(dependent);
    
    return !HasCircularDependency();
}

bool StreamDependencyGraph::RemoveDependency(cudaStream_t dependent, cudaStream_t prerequisite) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    auto& deps = dependencies_[dependent];
    deps.erase(std::remove(deps.begin(), deps.end(), prerequisite), deps.end());
    
    auto& dependents = dependents_[prerequisite];
    dependents.erase(std::remove(dependents.begin(), dependents.end(), dependent), dependents.end());
    
    return true;
}

void StreamDependencyGraph::ClearDependencies(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    // Remove as dependent
    for (auto prerequisite : dependencies_[stream]) {
        auto& deps = dependents_[prerequisite];
        deps.erase(std::remove(deps.begin(), deps.end(), stream), deps.end());
    }
    dependencies_[stream].clear();
    
    // Remove as prerequisite
    for (auto dependent : dependents_[stream]) {
        auto& deps = dependencies_[dependent];
        deps.erase(std::remove(deps.begin(), deps.end(), stream), deps.end());
    }
    dependents_[stream].clear();
}

std::vector<cudaStream_t> StreamDependencyGraph::GetExecutionOrder() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    std::vector<cudaStream_t> result;
    std::unordered_set<cudaStream_t> visited;
    std::unordered_set<cudaStream_t> temp_visited;
    
    // Topological sort using DFS
    std::function<bool(cudaStream_t)> visit = [&](cudaStream_t stream) -> bool {
        if (temp_visited.count(stream)) {
            return false; // Circular dependency
        }
        if (visited.count(stream)) {
            return true;
        }
        
        temp_visited.insert(stream);
        
        auto it = dependencies_.find(stream);
        if (it != dependencies_.end()) {
            for (auto prerequisite : it->second) {
                if (!visit(prerequisite)) {
                    return false;
                }
            }
        }
        
        temp_visited.erase(stream);
        visited.insert(stream);
        result.push_back(stream);
        
        return true;
    };
    
    // Visit all streams
    for (const auto& [stream, _] : dependencies_) {
        if (!visited.count(stream)) {
            if (!visit(stream)) {
                return {}; // Circular dependency detected
            }
        }
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}

bool StreamDependencyGraph::HasCircularDependency() const {
    std::unordered_set<cudaStream_t> visited;
    std::unordered_set<cudaStream_t> recursion_stack;
    
    for (const auto& [stream, _] : dependencies_) {
        if (!visited.count(stream)) {
            if (HasCircularDependencyDFS(stream, visited, recursion_stack)) {
                return true;
            }
        }
    }
    
    return false;
}

bool StreamDependencyGraph::CanExecute(cudaStream_t stream) const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    auto it = dependencies_.find(stream);
    if (it == dependencies_.end()) {
        return true; // No dependencies
    }
    
    // Check if all prerequisites are complete
    for (auto prerequisite : it->second) {
        if (cudaStreamQuery(prerequisite) != cudaSuccess) {
            return false; // Prerequisite still running
        }
    }
    
    return true;
}

bool StreamDependencyGraph::SynchronizeDependencies(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    auto it = dependencies_.find(stream);
    if (it == dependencies_.end()) {
        return true;
    }
    
    for (auto prerequisite : it->second) {
        cudaError_t error = cudaStreamSynchronize(prerequisite);
        if (error != cudaSuccess) {
            return false;
        }
    }
    
    return true;
}

bool StreamDependencyGraph::WaitForPrerequisites(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    auto it = dependencies_.find(stream);
    if (it == dependencies_.end()) {
        return true;
    }
    
    // Create events for each prerequisite and wait
    for (auto prerequisite : it->second) {
        cudaEvent_t event;
        cudaError_t error = cudaEventCreate(&event);
        if (error != cudaSuccess) continue;
        
        error = cudaEventRecord(event, prerequisite);
        if (error == cudaSuccess) {
            cudaStreamWaitEvent(stream, event, 0);
        }
        
        cudaEventDestroy(event);
    }
    
    return true;
}

bool StreamDependencyGraph::HasCircularDependencyDFS(cudaStream_t stream,
                                                    std::unordered_set<cudaStream_t>& visited,
                                                    std::unordered_set<cudaStream_t>& recursion_stack) const {
    visited.insert(stream);
    recursion_stack.insert(stream);
    
    auto it = dependencies_.find(stream);
    if (it != dependencies_.end()) {
        for (auto prerequisite : it->second) {
            if (!visited.count(prerequisite)) {
                if (HasCircularDependencyDFS(prerequisite, visited, recursion_stack)) {
                    return true;
                }
            } else if (recursion_stack.count(prerequisite)) {
                return true; // Back edge found
            }
        }
    }
    
    recursion_stack.erase(stream);
    return false;
}

// CudaStreamManager implementation
CudaStreamManager::CudaStreamManager(const CudaConfig& config, const std::vector<int>& devices)
    : config_(config), devices_(devices), max_streams_per_device_(config.num_streams),
      default_config_(GetDefaultStreamConfig(StreamType::COMPUTE)) {
    dependency_graph_ = std::make_unique<StreamDependencyGraph>();
}

CudaStreamManager::~CudaStreamManager() {
    Cleanup();
}

bool CudaStreamManager::Initialize() {
    for (int device_id : devices_) {
        if (!InitializeDevice(device_id)) {
            std::cerr << "Failed to initialize streams for device " << device_id << std::endl;
            return false;
        }
    }
    
    return !device_streams_.empty();
}

void CudaStreamManager::Cleanup() {
    for (int device_id : devices_) {
        CleanupDevice(device_id);
    }
    
    device_streams_.clear();
    
    // Cleanup events
    std::lock_guard<std::mutex> event_lock(event_mutex_);
    while (!event_pool_.empty()) {
        cudaEventDestroy(event_pool_.front());
        event_pool_.pop();
    }
    
    for (auto event : active_events_) {
        cudaEventDestroy(event);
    }
    active_events_.clear();
}

cudaStream_t CudaStreamManager::GetStream(StreamType type, int device_id) {
    device_id = SelectBestDevice(type, device_id);
    
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    
    // Try to get from pool first
    auto& pool = device_streams->stream_pools[type];
    if (!pool.empty()) {
        cudaStream_t stream = pool.front();
        pool.pop();
        return stream;
    }
    
    // Create new stream if under limit
    if (device_streams->streams.size() < static_cast<size_t>(max_streams_per_device_)) {
        StreamConfig config = GetOptimalStreamConfig(type, device_id, {});
        return CreateDeviceStream(config, device_id);
    }
    
    // Return least loaded stream of this type
    return GetLeastLoadedStream(type, device_id);
}

cudaStream_t CudaStreamManager::CreateStream(const StreamConfig& config) {
    int device_id = config.device_id;
    if (device_id == -1) {
        device_id = devices_.empty() ? 0 : devices_[0];
    }
    
    return CreateDeviceStream(config, device_id);
}

bool CudaStreamManager::DestroyStream(cudaStream_t stream) {
    if (!stream) return true;
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto it = stream_to_device_.find(stream);
    if (it == stream_to_device_.end()) {
        return false;
    }
    
    int device_id = it->second;
    stream_to_device_.erase(it);
    stream_configs_.erase(stream);
    
    dependency_graph_->ClearDependencies(stream);
    
    return RemoveStreamFromDevice(stream, device_id);
}

cudaStream_t CudaStreamManager::GetComputeStream(int device_id) {
    return GetStream(StreamType::COMPUTE, device_id);
}

cudaStream_t CudaStreamManager::GetMemoryStream(int device_id) {
    return GetStream(StreamType::MEMORY_COPY, device_id);
}

cudaStream_t CudaStreamManager::GetAttentionStream(int device_id) {
    return GetStream(StreamType::ATTENTION, device_id);
}

cudaStream_t CudaStreamManager::GetActivationStream(int device_id) {
    return GetStream(StreamType::ACTIVATION, device_id);
}

cudaStream_t CudaStreamManager::GetCommunicationStream(int device_id) {
    return GetStream(StreamType::COMMUNICATION, device_id);
}

bool CudaStreamManager::ReturnStreamToPool(cudaStream_t stream) {
    if (!stream) return false;
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto config_it = stream_configs_.find(stream);
    auto device_it = stream_to_device_.find(stream);
    
    if (config_it == stream_configs_.end() || device_it == stream_to_device_.end()) {
        return false;
    }
    
    StreamType type = config_it->second.type;
    int device_id = device_it->second;
    
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) {
        return false;
    }
    
    std::lock_guard<std::mutex> device_lock(device_streams->mutex);
    device_streams->stream_pools[type].push(stream);
    
    return true;
}

cudaStream_t CudaStreamManager::GetStreamFromPool(StreamType type, int device_id) {
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    auto& pool = device_streams->stream_pools[type];
    
    if (pool.empty()) {
        return nullptr;
    }
    
    cudaStream_t stream = pool.front();
    pool.pop();
    return stream;
}

void CudaStreamManager::FlushStreamPool(int device_id) {
    if (device_id == -1) {
        for (int dev_id : devices_) {
            FlushStreamPool(dev_id);
        }
        return;
    }
    
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    for (auto& [type, pool] : device_streams->stream_pools) {
        while (!pool.empty()) {
            pool.pop();
        }
    }
}

cudaStream_t CudaStreamManager::GetLeastLoadedStream(StreamType type, int device_id) {
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    
    ManagedStream* best_stream = nullptr;
    double min_load = 2.0; // Higher than max possible load
    
    for (auto& stream : device_streams->streams) {
        if (stream->GetType() == type) {
            double load = stream->GetStats().load_factor;
            if (load < min_load) {
                min_load = load;
                best_stream = stream.get();
            }
        }
    }
    
    return best_stream ? best_stream->GetStream() : nullptr;
}

void CudaStreamManager::UpdateStreamLoadFactors() {
    for (auto& [device_id, device_streams] : device_streams_) {
        std::lock_guard<std::mutex> lock(device_streams->mutex);
        for (auto& stream : device_streams->streams) {
            stream->UpdateLoadFactor();
        }
    }
}

void CudaStreamManager::BalanceStreams(int device_id) {
    if (device_id == -1) {
        for (int dev_id : devices_) {
            BalanceStreams(dev_id);
        }
        return;
    }
    
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    device_streams->last_balance_time = std::chrono::steady_clock::now();
    
    // Simple load balancing: ensure we have streams for each type
    for (int type_int = 0; type_int < 6; ++type_int) {
        StreamType type = static_cast<StreamType>(type_int);
        
        bool found = false;
        for (auto& stream : device_streams->streams) {
            if (stream->GetType() == type) {
                found = true;
                break;
            }
        }
        
        if (!found && device_streams->streams.size() < static_cast<size_t>(max_streams_per_device_)) {
            StreamConfig config = GetOptimalStreamConfig(type, device_id, {});
            CreateDeviceStream(config, device_id);
        }
    }
}

bool CudaStreamManager::SynchronizeAllStreams(int device_id) {
    if (device_id == -1) {
        bool success = true;
        for (int dev_id : devices_) {
            success &= SynchronizeAllStreams(dev_id);
        }
        return success;
    }
    
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return false;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    for (auto& stream : device_streams->streams) {
        if (!stream->Synchronize()) {
            return false;
        }
    }
    
    return true;
}

bool CudaStreamManager::SynchronizeStreamType(StreamType type, int device_id) {
    if (device_id == -1) {
        bool success = true;
        for (int dev_id : devices_) {
            success &= SynchronizeStreamType(type, dev_id);
        }
        return success;
    }
    
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return false;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    for (auto& stream : device_streams->streams) {
        if (stream->GetType() == type) {
            if (!stream->Synchronize()) {
                return false;
            }
        }
    }
    
    return true;
}

cudaEvent_t CudaStreamManager::CreateEvent(bool enable_timing, bool blocking) {
    unsigned int flags = cudaEventDefault;
    if (!enable_timing) {
        flags |= cudaEventDisableTiming;
    }
    if (blocking) {
        flags |= cudaEventBlockingSync;
    }
    
    cudaEvent_t event;
    cudaError_t error = cudaEventCreateWithFlags(&event, flags);
    if (error != cudaSuccess) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(event_mutex_);
    active_events_.insert(event);
    
    return event;
}

bool CudaStreamManager::DestroyEvent(cudaEvent_t event) {
    if (!event) return true;
    
    std::lock_guard<std::mutex> lock(event_mutex_);
    active_events_.erase(event);
    
    cudaError_t error = cudaEventDestroy(event);
    return error == cudaSuccess;
}

bool CudaStreamManager::RecordEvent(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return false;
    
    cudaError_t error = cudaEventRecord(event, stream);
    return error == cudaSuccess;
}

bool CudaStreamManager::WaitEvent(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return false;
    
    cudaError_t error = cudaStreamWaitEvent(stream, event, 0);
    return error == cudaSuccess;
}

bool CudaStreamManager::EventElapsedTime(cudaEvent_t start, cudaEvent_t end, float* time_ms) {
    if (!start || !end || !time_ms) return false;
    
    cudaError_t error = cudaEventElapsedTime(time_ms, start, end);
    return error == cudaSuccess;
}

bool CudaStreamManager::AddStreamDependency(cudaStream_t dependent, cudaStream_t prerequisite) {
    return dependency_graph_->AddDependency(dependent, prerequisite);
}

bool CudaStreamManager::RemoveStreamDependency(cudaStream_t dependent, cudaStream_t prerequisite) {
    return dependency_graph_->RemoveDependency(dependent, prerequisite);
}

bool CudaStreamManager::SynchronizeWithDependencies(cudaStream_t stream) {
    return dependency_graph_->SynchronizeDependencies(stream);
}

StreamStats CudaStreamManager::GetStreamStats(cudaStream_t stream) const {
    ManagedStream* managed_stream = FindManagedStream(stream);
    if (managed_stream) {
        return managed_stream->GetStats();
    }
    return StreamStats{};
}

std::vector<StreamStats> CudaStreamManager::GetAllStreamStats(int device_id) const {
    std::vector<StreamStats> stats;
    
    if (device_id == -1) {
        for (int dev_id : devices_) {
            auto device_stats = GetAllStreamStats(dev_id);
            stats.insert(stats.end(), device_stats.begin(), device_stats.end());
        }
        return stats;
    }
    
    const DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return stats;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    for (const auto& stream : device_streams->streams) {
        stats.push_back(stream->GetStats());
    }
    
    return stats;
}

void CudaStreamManager::ResetAllStats() {
    for (auto& [device_id, device_streams] : device_streams_) {
        std::lock_guard<std::mutex> lock(device_streams->mutex);
        for (auto& stream : device_streams->streams) {
            stream->ResetStats();
        }
    }
}

bool CudaStreamManager::InitializeDevice(int device_id) {
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        return false;
    }
    
    auto device_streams = std::make_unique<DeviceStreams>();
    
    // Create initial streams for each type
    for (int type_int = 0; type_int < 6; ++type_int) {
        StreamType type = static_cast<StreamType>(type_int);
        StreamConfig config = GetOptimalStreamConfig(type, device_id, {});
        
        auto managed_stream = std::make_unique<ManagedStream>(config);
        if (!managed_stream->Initialize()) {
            continue;
        }
        
        cudaStream_t stream = managed_stream->GetStream();
        device_streams->stream_map[stream] = std::move(managed_stream);
    }
    
    device_streams_[device_id] = std::move(device_streams);
    return true;
}

void CudaStreamManager::CleanupDevice(int device_id) {
    auto it = device_streams_.find(device_id);
    if (it == device_streams_.end()) return;
    
    DeviceStreams* device_streams = it->second.get();
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    
    // Clear stream pools
    for (auto& [type, pool] : device_streams->stream_pools) {
        while (!pool.empty()) {
            pool.pop();
        }
    }
    
    // Destroy managed streams
    device_streams->streams.clear();
    device_streams->stream_map.clear();
    
    device_streams_.erase(it);
}

CudaStreamManager::DeviceStreams* CudaStreamManager::GetDeviceStreams(int device_id) {
    auto it = device_streams_.find(device_id);
    return (it != device_streams_.end()) ? it->second.get() : nullptr;
}

const CudaStreamManager::DeviceStreams* CudaStreamManager::GetDeviceStreams(int device_id) const {
    auto it = device_streams_.find(device_id);
    return (it != device_streams_.end()) ? it->second.get() : nullptr;
}

ManagedStream* CudaStreamManager::FindManagedStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto device_it = stream_to_device_.find(stream);
    if (device_it == stream_to_device_.end()) {
        return nullptr;
    }
    
    DeviceStreams* device_streams = GetDeviceStreams(device_it->second);
    if (!device_streams) return nullptr;
    
    std::lock_guard<std::mutex> device_lock(device_streams->mutex);
    auto stream_it = device_streams->stream_map.find(stream);
    return (stream_it != device_streams->stream_map.end()) ? stream_it->second.get() : nullptr;
}

const ManagedStream* CudaStreamManager::FindManagedStream(cudaStream_t stream) const {
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto device_it = stream_to_device_.find(stream);
    if (device_it == stream_to_device_.end()) {
        return nullptr;
    }
    
    const DeviceStreams* device_streams = GetDeviceStreams(device_it->second);
    if (!device_streams) return nullptr;
    
    std::lock_guard<std::mutex> device_lock(device_streams->mutex);
    auto stream_it = device_streams->stream_map.find(stream);
    return (stream_it != device_streams->stream_map.end()) ? stream_it->second.get() : nullptr;
}

cudaStream_t CudaStreamManager::CreateDeviceStream(const StreamConfig& config, int device_id) {
    auto managed_stream = std::make_unique<ManagedStream>(config);
    if (!managed_stream->Initialize()) {
        return nullptr;
    }
    
    cudaStream_t stream = managed_stream->GetStream();
    
    if (!AddStreamToDevice(std::move(managed_stream), device_id)) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    stream_to_device_[stream] = device_id;
    stream_configs_[stream] = config;
    
    return stream;
}

bool CudaStreamManager::AddStreamToDevice(std::unique_ptr<ManagedStream> stream, int device_id) {
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return false;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    
    cudaStream_t stream_handle = stream->GetStream();
    device_streams->stream_map[stream_handle] = std::move(stream);
    
    return true;
}

bool CudaStreamManager::RemoveStreamFromDevice(cudaStream_t stream, int device_id) {
    DeviceStreams* device_streams = GetDeviceStreams(device_id);
    if (!device_streams) return false;
    
    std::lock_guard<std::mutex> lock(device_streams->mutex);
    
    auto it = device_streams->stream_map.find(stream);
    if (it != device_streams->stream_map.end()) {
        device_streams->stream_map.erase(it);
        return true;
    }
    
    return false;
}

int CudaStreamManager::SelectBestDevice(StreamType type, int requested_device) const {
    if (requested_device != -1 && 
        std::find(devices_.begin(), devices_.end(), requested_device) != devices_.end()) {
        return requested_device;
    }
    
    // Simple device selection - return first available device
    return devices_.empty() ? 0 : devices_[0];
}

// ScopedStream implementation
ScopedStream::ScopedStream(CudaStreamManager* manager, StreamType type, int device_id)
    : manager_(manager), stream_(nullptr), owns_stream_(true) {
    if (manager_) {
        stream_ = manager_->GetStream(type, device_id);
    }
}

ScopedStream::ScopedStream(CudaStreamManager* manager, const StreamConfig& config)
    : manager_(manager), stream_(nullptr), owns_stream_(true) {
    if (manager_) {
        stream_ = manager_->CreateStream(config);
    }
}

ScopedStream::~ScopedStream() {
    if (owns_stream_ && manager_ && stream_) {
        manager_->ReturnStreamToPool(stream_);
    }
}

ScopedStream::ScopedStream(ScopedStream&& other) noexcept
    : manager_(other.manager_), stream_(other.stream_), owns_stream_(other.owns_stream_) {
    other.manager_ = nullptr;
    other.stream_ = nullptr;
    other.owns_stream_ = false;
}

ScopedStream& ScopedStream::operator=(ScopedStream&& other) noexcept {
    if (this != &other) {
        if (owns_stream_ && manager_ && stream_) {
            manager_->ReturnStreamToPool(stream_);
        }
        
        manager_ = other.manager_;
        stream_ = other.stream_;
        owns_stream_ = other.owns_stream_;
        
        other.manager_ = nullptr;
        other.stream_ = nullptr;
        other.owns_stream_ = false;
    }
    return *this;
}

bool ScopedStream::Synchronize() {
    if (!stream_) return false;
    
    cudaError_t error = cudaStreamSynchronize(stream_);
    return error == cudaSuccess;
}

bool ScopedStream::Query() const {
    if (!stream_) return false;
    
    cudaError_t error = cudaStreamQuery(stream_);
    return error == cudaSuccess;
}

} // namespace cuda
} // namespace backends
} // namespace gemma