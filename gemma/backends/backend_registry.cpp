/**
 * @file backend_registry.cpp
 * @brief Implementation of backend registry for hardware acceleration
 */

#include "backend_registry.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <chrono>

namespace gemma {
namespace backends {

BackendRegistry& BackendRegistry::Instance() {
    static BackendRegistry instance;
    return instance;
}

bool BackendRegistry::RegisterBackend(const BackendInfo& info) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    if (info.name.empty() || !info.factory) {
        return false;
    }

    // Check if backend is already registered
    if (backends_.find(info.name) != backends_.end()) {
        std::cerr << "Backend '" << info.name << "' is already registered" << std::endl;
        return false;
    }

    backends_[info.name] = info;

    std::cout << "Registered backend: " << info.name
              << " (priority: " << static_cast<int>(info.priority) << ")" << std::endl;

    return true;
}

bool BackendRegistry::UnregisterBackend(const std::string& name) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    // Remove from active backends
    auto active_it = active_backends_.find(name);
    if (active_it != active_backends_.end()) {
        if (active_it->second) {
            active_it->second->Shutdown();
        }
        active_backends_.erase(active_it);
    }

    // Remove from registered backends
    auto backend_it = backends_.find(name);
    if (backend_it != backends_.end()) {
        backends_.erase(backend_it);
        return true;
    }

    return false;
}

std::vector<std::string> BackendRegistry::GetAvailableBackends() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::string> available;

    for (const auto& [name, info] : backends_) {
        if (info.is_available) {
            try {
                auto backend = info.factory();
                if (backend && backend->IsAvailable()) {
                    available.push_back(name);
                }
            } catch (const std::exception& e) {
                // Skip backends that throw exceptions during availability check
            }
        }
    }

    return available;
}

const BackendInfo* BackendRegistry::GetBackendInfo(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = backends_.find(name);
    if (it != backends_.end()) {
        return &it->second;
    }

    return nullptr;
}

std::unique_ptr<BackendInterface> BackendRegistry::CreateBackend(const std::string& name) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = backends_.find(name);
    if (it == backends_.end()) {
        std::cerr << "Backend '" << name << "' not found in registry" << std::endl;
        return nullptr;
    }

    const BackendInfo& info = it->second;

    try {
        auto backend = info.factory();
        if (backend && backend->IsAvailable()) {
            if (backend->Initialize()) {
                std::cout << "Successfully created and initialized backend: " << name << std::endl;
                return backend;
            } else {
                std::cerr << "Failed to initialize backend: " << name << std::endl;
            }
        } else {
            std::cerr << "Backend not available: " << name << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception creating backend '" << name << "': " << e.what() << std::endl;
    }

    return nullptr;
}

std::string BackendRegistry::FindBestBackend(
    const std::vector<BackendCapability>& required_capabilities,
    BackendPriority preferred_priority) const {

    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::pair<std::string, BackendPriority>> candidates;

    for (const auto& [name, info] : backends_) {
        if (!info.is_available || static_cast<int>(info.priority) < static_cast<int>(preferred_priority)) {
            continue;
        }

        // Check if backend supports all required capabilities
        bool supports_all = true;
        for (BackendCapability required : required_capabilities) {
            bool found = std::find(info.capabilities.begin(), info.capabilities.end(), required)
                        != info.capabilities.end();
            if (!found) {
                supports_all = false;
                break;
            }
        }

        if (supports_all) {
            candidates.emplace_back(name, info.priority);
        }
    }

    if (candidates.empty()) {
        return "";
    }

    // Sort by priority (highest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return static_cast<int>(a.second) > static_cast<int>(b.second);
              });

    return candidates[0].first;
}

std::string BackendRegistry::AutoSelectBackend() const {
    auto available = GetAvailableBackends();

    if (available.empty()) {
        return "";
    }

    // Priority order for auto-selection
    std::vector<std::string> priority_order = {
        "CUDA", "SYCL", "Vulkan", "OpenCL", "Metal", "CPU"
    };

    for (const std::string& preferred : priority_order) {
        if (std::find(available.begin(), available.end(), preferred) != available.end()) {
            return preferred;
        }
    }

    // Return first available if no preferred backend found
    return available[0];
}

bool BackendRegistry::SupportsCapabilities(const std::string& name,
                                          const std::vector<BackendCapability>& capabilities) const {
    const BackendInfo* info = GetBackendInfo(name);
    if (!info) {
        return false;
    }

    for (BackendCapability required : capabilities) {
        bool found = std::find(info->capabilities.begin(), info->capabilities.end(), required)
                    != info->capabilities.end();
        if (!found) {
            return false;
        }
    }

    return true;
}

std::vector<std::string> BackendRegistry::GetBackendsWithCapability(BackendCapability capability) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::vector<std::string> matching_backends;

    for (const auto& [name, info] : backends_) {
        if (info.is_available) {
            bool found = std::find(info.capabilities.begin(), info.capabilities.end(), capability)
                        != info.capabilities.end();
            if (found) {
                matching_backends.push_back(name);
            }
        }
    }

    return matching_backends;
}

bool BackendRegistry::InitializeBackends() {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    bool any_initialized = false;

    for (const auto& [name, info] : backends_) {
        if (!info.is_available) {
            continue;
        }

        try {
            auto backend = info.factory();
            if (backend && backend->IsAvailable()) {
                if (backend->Initialize()) {
                    active_backends_[name] = std::move(backend);
                    any_initialized = true;
                    std::cout << "Initialized backend: " << name << std::endl;
                } else {
                    std::cerr << "Failed to initialize backend: " << name << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception initializing backend '" << name << "': " << e.what() << std::endl;
        }
    }

    return any_initialized;
}

void BackendRegistry::ShutdownBackends() {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    for (auto& [name, backend] : active_backends_) {
        if (backend) {
            try {
                backend->Shutdown();
                std::cout << "Shut down backend: " << name << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error shutting down backend '" << name << "': " << e.what() << std::endl;
            }
        }
    }

    active_backends_.clear();
}

bool BackendRegistry::HasAvailableBackends() const {
    auto available = GetAvailableBackends();
    return !available.empty();
}

std::map<std::string, double> BackendRegistry::BenchmarkBackends(const std::string& operation_type) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    std::map<std::string, double> results;

    for (const auto& [name, backend] : active_backends_) {
        if (backend) {
            try {
                double score = BenchmarkBackend(backend.get(), operation_type);
                results[name] = score;
            } catch (const std::exception& e) {
                std::cerr << "Error benchmarking backend '" << name << "': " << e.what() << std::endl;
                results[name] = 0.0;
            }
        }
    }

    return results;
}

double BackendRegistry::BenchmarkBackend(BackendInterface* backend, const std::string& operation_type) const {
    if (!backend || !backend->IsInitialized()) {
        return 0.0;
    }

    // Simple benchmark - measure basic matrix multiplication performance
    const size_t size = 1024;
    const size_t bytes = size * size * sizeof(float);

    auto buffer_a = backend->AllocateBuffer(bytes);
    auto buffer_b = backend->AllocateBuffer(bytes);
    auto buffer_c = backend->AllocateBuffer(bytes);

    if (!buffer_a.IsValid() || !buffer_b.IsValid() || !buffer_c.IsValid()) {
        backend->FreeBuffer(buffer_a);
        backend->FreeBuffer(buffer_b);
        backend->FreeBuffer(buffer_c);
        return 0.0;
    }

    // Warm up
    backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, size, size, size);
    backend->Synchronize();

    // Benchmark
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, size, size, size);
    }

    backend->Synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    backend->FreeBuffer(buffer_a);
    backend->FreeBuffer(buffer_b);
    backend->FreeBuffer(buffer_c);

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops = 2.0 * size * size * size * iterations; // Multiply-add operations
    double gflops = (ops / 1e9) / (elapsed_ms / 1000.0);

    return gflops;
}

} // namespace backends
} // namespace gemma