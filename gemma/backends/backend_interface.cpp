/**
 * @file backend_interface.cpp
 * @brief Implementation of base backend interface functionality
 */

#include "backend_interface.h"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace gemma {
namespace backends {

// BackendBuffer implementation
BackendBuffer::~BackendBuffer() {
    // Base destructor - specific backends handle cleanup
}

bool BackendBuffer::IsValid() const {
    return data != nullptr && size > 0;
}

void BackendBuffer::Reset() {
    data = nullptr;
    size = 0;
    alignment = 32;
    is_device_memory = false;
}

// BackendMetrics implementation
void BackendMetrics::Reset() {
    compute_throughput_gflops = 0.0;
    memory_bandwidth_gbps = 0.0;
    latency_ms = 0.0;
    memory_usage_bytes = 0;
    peak_memory_bytes = 0;
    num_operations = 0;
    total_execution_time_ms = 0.0;
    cache_hit_rate = 0.0;
    power_consumption_watts = 0.0;
}

BackendMetrics BackendMetrics::operator+(const BackendMetrics& other) const {
    BackendMetrics result = *this;
    result.compute_throughput_gflops += other.compute_throughput_gflops;
    result.memory_bandwidth_gbps += other.memory_bandwidth_gbps;
    result.latency_ms += other.latency_ms;
    result.memory_usage_bytes += other.memory_usage_bytes;
    result.peak_memory_bytes = std::max(result.peak_memory_bytes, other.peak_memory_bytes);
    result.num_operations += other.num_operations;
    result.total_execution_time_ms += other.total_execution_time_ms;
    result.cache_hit_rate = (result.cache_hit_rate + other.cache_hit_rate) / 2.0;
    result.power_consumption_watts += other.power_consumption_watts;
    return result;
}

// BackendInterface implementation
BackendInterface::BackendInterface()
    : initialized_(false), profiling_enabled_(false) {
}

BackendInterface::~BackendInterface() {
    if (initialized_) {
        Shutdown();
    }
}

bool BackendInterface::IsInitialized() const {
    return initialized_;
}

void BackendInterface::EnableProfiling(bool enable) {
    profiling_enabled_ = enable;
}

bool BackendInterface::IsProfilingEnabled() const {
    return profiling_enabled_;
}

std::string BackendInterface::GetErrorString() const {
    return last_error_;
}

void BackendInterface::SetError(const std::string& error) {
    last_error_ = error;
}

void BackendInterface::ClearError() {
    last_error_.clear();
}

// Default implementations for optional operations
bool BackendInterface::SupportsCapability(BackendCapability capability) const {
    // Base implementation - backends should override
    switch (capability) {
        case BackendCapability::MATRIX_MULTIPLICATION:
        case BackendCapability::ACTIVATION_FUNCTIONS:
            return true;
        default:
            return false;
    }
}

bool BackendInterface::SetDevice(int device_id) {
    // Base implementation for single-device backends
    return device_id == 0;
}

int BackendInterface::GetCurrentDevice() const {
    // Base implementation
    return 0;
}

void BackendInterface::ResetMetrics() {
    // Base implementation - backends should override if they maintain metrics
}

// Utility functions
std::string BackendCapabilityToString(BackendCapability capability) {
    switch (capability) {
        case BackendCapability::MATRIX_MULTIPLICATION:
            return "Matrix Multiplication";
        case BackendCapability::ATTENTION_COMPUTATION:
            return "Attention Computation";
        case BackendCapability::ACTIVATION_FUNCTIONS:
            return "Activation Functions";
        case BackendCapability::MEMORY_POOLING:
            return "Memory Pooling";
        case BackendCapability::ASYNC_EXECUTION:
            return "Async Execution";
        case BackendCapability::MULTI_PRECISION:
            return "Multi-Precision";
        default:
            return "Unknown";
    }
}

bool ValidateBufferAlignment(const BackendBuffer& buffer, size_t required_alignment) {
    if (!buffer.IsValid()) {
        return false;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer.data);
    return (addr % required_alignment) == 0;
}

size_t AlignSize(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

bool CopyBuffer(const BackendBuffer& dst, const BackendBuffer& src, size_t size) {
    if (!dst.IsValid() || !src.IsValid()) {
        return false;
    }

    if (size == 0) {
        size = std::min(dst.size, src.size);
    }

    if (size > dst.size || size > src.size) {
        return false;
    }

    // For device memory, backends should override this
    if (dst.is_device_memory || src.is_device_memory) {
        return false;
    }

    std::memcpy(dst.data, src.data, size);
    return true;
}

} // namespace backends
} // namespace gemma