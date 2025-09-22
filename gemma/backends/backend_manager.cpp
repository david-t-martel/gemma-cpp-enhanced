/**
 * @file backend_manager.cpp
 * @brief Implementation of high-level backend management
 */

#include "backend_manager.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>

// Include backend headers if available
#ifdef GEMMA_ENABLE_CUDA
#include "cuda/cuda_backend.h"
#endif

#ifdef GEMMA_ENABLE_SYCL
#include "sycl/sycl_backend.h"
#endif

#ifdef GEMMA_ENABLE_VULKAN
#include "vulkan/vulkan_backend.h"
#endif

#ifdef GEMMA_ENABLE_OPENCL
#include "opencl/opencl_backend.h"
#endif

#ifdef GEMMA_ENABLE_METAL
#include "metal/metal_backend.h"
#endif

namespace gemma {
namespace backends {

BackendManager::BackendManager(const BackendConfig& config)
    : config_(config), initialized_(false) {
}

BackendManager::~BackendManager() {
    if (initialized_) {
        Shutdown();
    }
}

bool BackendManager::Initialize() {
    if (initialized_) {
        return true;
    }

    if (config_.verbose_logging) {
        std::cout << "Initializing backend system..." << std::endl;
    }

    // Register all available backends
    RegisterAllBackends();

    // Get candidate backends
    auto candidates = GetCandidateBackends();

    if (candidates.empty()) {
        std::cerr << "No available backends found" << std::endl;
        return false;
    }

    if (config_.verbose_logging) {
        std::cout << "Available backends: ";
        for (const auto& name : candidates) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
    }

    // Select and initialize primary backend
    std::string selected_backend;

    if (config_.preferred_backend == "auto") {
        selected_backend = SelectBestCandidate(candidates);
    } else {
        // Check if preferred backend is available
        auto it = std::find(candidates.begin(), candidates.end(), config_.preferred_backend);
        if (it != candidates.end()) {
            selected_backend = config_.preferred_backend;
        } else if (config_.enable_fallback) {
            std::cerr << "Preferred backend '" << config_.preferred_backend
                      << "' not available, falling back to auto-selection" << std::endl;
            selected_backend = SelectBestCandidate(candidates);
        } else {
            std::cerr << "Preferred backend '" << config_.preferred_backend
                      << "' not available and fallback disabled" << std::endl;
            return false;
        }
    }

    if (selected_backend.empty()) {
        std::cerr << "Failed to select a backend" << std::endl;
        return false;
    }

    // Initialize the selected backend
    if (InitializeBackend(selected_backend)) {
        active_backend_name_ = selected_backend;
        initialized_ = true;

        if (config_.verbose_logging) {
            std::cout << "Successfully initialized backend: " << selected_backend << std::endl;
        }

        // Run benchmarks if requested
        if (config_.enable_benchmarking) {
            if (config_.verbose_logging) {
                std::cout << "Running benchmarks..." << std::endl;
            }
            RunBenchmarks();
        }

        return true;
    }

    std::cerr << "Failed to initialize selected backend: " << selected_backend << std::endl;
    return false;
}

void BackendManager::Shutdown() {
    if (!initialized_) {
        return;
    }

    for (auto& [name, backend] : active_backends_) {
        if (backend) {
            try {
                backend->Shutdown();
                if (config_.verbose_logging) {
                    std::cout << "Shut down backend: " << name << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error shutting down backend '" << name << "': " << e.what() << std::endl;
            }
        }
    }

    active_backends_.clear();
    performance_cache_.clear();
    active_backend_name_.clear();
    initialized_ = false;
}

BackendInterface* BackendManager::GetActiveBackend() const {
    if (!initialized_ || active_backend_name_.empty()) {
        return nullptr;
    }

    auto it = active_backends_.find(active_backend_name_);
    return (it != active_backends_.end()) ? it->second.get() : nullptr;
}

BackendInterface* BackendManager::GetBackend(const std::string& name) const {
    auto it = active_backends_.find(name);
    return (it != active_backends_.end()) ? it->second.get() : nullptr;
}

bool BackendManager::SwitchBackend(const std::string& name) {
    if (!initialized_) {
        return false;
    }

    // Check if backend is already active
    if (active_backend_name_ == name) {
        return true;
    }

    // Check if backend is already initialized
    auto it = active_backends_.find(name);
    if (it != active_backends_.end()) {
        active_backend_name_ = name;
        if (config_.verbose_logging) {
            std::cout << "Switched to backend: " << name << std::endl;
        }
        return true;
    }

    // Try to initialize the backend
    if (InitializeBackend(name)) {
        active_backend_name_ = name;
        if (config_.verbose_logging) {
            std::cout << "Initialized and switched to backend: " << name << std::endl;
        }
        return true;
    }

    std::cerr << "Failed to switch to backend: " << name << std::endl;
    return false;
}

std::vector<std::string> BackendManager::GetAvailableBackends() const {
    return BackendRegistry::Instance().GetAvailableBackends();
}

std::vector<BackendPerformance> BackendManager::GetBackendPerformance() const {
    std::vector<BackendPerformance> performance_list;

    for (const auto& [name, perf] : performance_cache_) {
        performance_list.push_back(perf);
    }

    return performance_list;
}

bool BackendManager::SupportsCapabilities(const std::vector<BackendCapability>& capabilities) const {
    auto* backend = GetActiveBackend();
    if (!backend) {
        return false;
    }

    for (auto capability : capabilities) {
        if (!backend->SupportsCapability(capability)) {
            return false;
        }
    }

    return true;
}

std::string BackendManager::FindBestBackend(const std::vector<BackendCapability>& capabilities) const {
    return BackendRegistry::Instance().FindBestBackend(capabilities);
}

std::map<std::string, BackendPerformance> BackendManager::RunBenchmarks() {
    std::map<std::string, BackendPerformance> results;

    for (auto& [name, backend] : active_backends_) {
        if (backend) {
            auto perf = BenchmarkBackend(backend.get(), name);
            results[name] = perf;
            performance_cache_[name] = perf;
        }
    }

    return results;
}

std::string BackendManager::GetStatusReport() const {
    std::ostringstream oss;
    oss << "=== Backend Manager Status ===" << std::endl;
    oss << "Initialized: " << (initialized_ ? "Yes" : "No") << std::endl;
    oss << "Active Backend: " << (active_backend_name_.empty() ? "None" : active_backend_name_) << std::endl;
    oss << "Loaded Backends: " << active_backends_.size() << std::endl;
    oss << std::endl;

    auto* active = GetActiveBackend();
    if (active) {
        oss << "Active Backend Details:" << std::endl;
        oss << "  Name: " << active->GetName() << std::endl;
        oss << "  Version: " << active->GetVersion() << std::endl;
        oss << "  Devices: " << active->GetDeviceCount() << std::endl;

        auto metrics = active->GetMetrics();
        oss << "  Memory Usage: " << (metrics.memory_usage_bytes / 1024 / 1024) << " MB" << std::endl;
        oss << "  Operations: " << metrics.num_operations << std::endl;
        oss << std::endl;
    }

    oss << "Registry Status:" << std::endl;
    oss << BackendRegistry::Instance().GetStatusReport();

    return oss.str();
}

void BackendManager::RegisterAllBackends() {
    auto& registry = BackendRegistry::Instance();

    // Register CUDA backend if available
#ifdef GEMMA_ENABLE_CUDA
    BackendInfo cuda_info;
    cuda_info.name = "CUDA";
    cuda_info.version = "1.0";
    cuda_info.priority = BackendPriority::HIGH;
    cuda_info.capabilities = {
        BackendCapability::MATRIX_MULTIPLICATION,
        BackendCapability::ATTENTION_COMPUTATION,
        BackendCapability::ACTIVATION_FUNCTIONS,
        BackendCapability::MEMORY_POOLING,
        BackendCapability::ASYNC_EXECUTION,
        BackendCapability::MULTI_PRECISION
    };
    cuda_info.is_available = cuda::IsCudaAvailable();
    cuda_info.description = "NVIDIA CUDA GPU acceleration";
    cuda_info.factory = []() { return cuda::CreateCudaBackend(); };
    registry.RegisterBackend(cuda_info);
#endif

    // Register SYCL backend if available
#ifdef GEMMA_ENABLE_SYCL
    BackendInfo sycl_info;
    sycl_info.name = "SYCL";
    sycl_info.version = "1.0";
    sycl_info.priority = BackendPriority::HIGH;
    sycl_info.capabilities = {
        BackendCapability::MATRIX_MULTIPLICATION,
        BackendCapability::ATTENTION_COMPUTATION,
        BackendCapability::ACTIVATION_FUNCTIONS,
        BackendCapability::MEMORY_POOLING,
        BackendCapability::ASYNC_EXECUTION
    };
    sycl_info.is_available = sycl::IsSyclBackendAvailable();
    sycl_info.description = "Intel SYCL/oneAPI acceleration";
    sycl_info.factory = []() { return sycl::CreateSyclBackend(); };
    registry.RegisterBackend(sycl_info);
#endif

    // Register Vulkan backend if available
#ifdef GEMMA_ENABLE_VULKAN
    BackendInfo vulkan_info;
    vulkan_info.name = "Vulkan";
    vulkan_info.version = "1.0";
    vulkan_info.priority = BackendPriority::MEDIUM;
    vulkan_info.capabilities = {
        BackendCapability::MATRIX_MULTIPLICATION,
        BackendCapability::ACTIVATION_FUNCTIONS,
        BackendCapability::ASYNC_EXECUTION
    };
    vulkan_info.is_available = vulkan::IsVulkanAvailable();
    vulkan_info.description = "Cross-platform Vulkan compute";
    vulkan_info.factory = []() { return vulkan::CreateVulkanBackend(); };
    registry.RegisterBackend(vulkan_info);
#endif

    // Register OpenCL backend if available
#ifdef GEMMA_ENABLE_OPENCL
    BackendInfo opencl_info;
    opencl_info.name = "OpenCL";
    opencl_info.version = "1.0";
    opencl_info.priority = BackendPriority::MEDIUM;
    opencl_info.capabilities = {
        BackendCapability::MATRIX_MULTIPLICATION,
        BackendCapability::ACTIVATION_FUNCTIONS
    };
    opencl_info.is_available = opencl::IsOpenCLAvailable();
    opencl_info.description = "Cross-platform OpenCL compute";
    opencl_info.factory = []() { return opencl::CreateOpenCLBackend(); };
    registry.RegisterBackend(opencl_info);
#endif

    // Register Metal backend if available (macOS only)
#ifdef GEMMA_ENABLE_METAL
    BackendInfo metal_info;
    metal_info.name = "Metal";
    metal_info.version = "1.0";
    metal_info.priority = BackendPriority::HIGH;
    metal_info.capabilities = {
        BackendCapability::MATRIX_MULTIPLICATION,
        BackendCapability::ATTENTION_COMPUTATION,
        BackendCapability::ACTIVATION_FUNCTIONS,
        BackendCapability::ASYNC_EXECUTION
    };
    metal_info.is_available = metal::IsMetalAvailable();
    metal_info.description = "Apple Metal GPU acceleration";
    metal_info.factory = []() { return metal::CreateMetalBackend(); };
    registry.RegisterBackend(metal_info);
#endif
}

std::string BackendManager::AutoSelectBackend(const BackendConfig& config) {
    auto& registry = BackendRegistry::Instance();
    return registry.AutoSelectBackend();
}

bool BackendManager::InitializeBackend(const std::string& name) {
    if (IsBackendDisabled(name)) {
        return false;
    }

    auto backend = BackendRegistry::Instance().CreateBackend(name);
    if (backend) {
        active_backends_[name] = std::move(backend);
        return true;
    }

    return false;
}

void BackendManager::ShutdownBackend(const std::string& name) {
    auto it = active_backends_.find(name);
    if (it != active_backends_.end()) {
        if (it->second) {
            it->second->Shutdown();
        }
        active_backends_.erase(it);
    }
}

BackendPerformance BackendManager::BenchmarkBackend(BackendInterface* backend, const std::string& name) {
    BackendPerformance perf;
    perf.name = name;

    if (!backend || !backend->IsInitialized()) {
        return perf;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Matrix multiplication benchmark
        const size_t size = 512;
        const size_t bytes = size * size * sizeof(float);

        auto buffer_a = backend->AllocateBuffer(bytes);
        auto buffer_b = backend->AllocateBuffer(bytes);
        auto buffer_c = backend->AllocateBuffer(bytes);

        if (buffer_a.IsValid() && buffer_b.IsValid() && buffer_c.IsValid()) {
            // Warm up
            backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, size, size, size);
            backend->Synchronize();

            // Benchmark matrix multiplication
            const int iterations = 5;
            auto mm_start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; ++i) {
                backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, size, size, size);
            }

            backend->Synchronize();
            auto mm_end = std::chrono::high_resolution_clock::now();

            double mm_time_ms = std::chrono::duration<double, std::milli>(mm_end - mm_start).count();
            double ops = 2.0 * size * size * size * iterations;
            perf.matrix_multiply_gflops = (ops / 1e9) / (mm_time_ms / 1000.0);

            // Cleanup
            backend->FreeBuffer(buffer_a);
            backend->FreeBuffer(buffer_b);
            backend->FreeBuffer(buffer_c);
        }

        // Get memory information
        auto metrics = backend->GetMetrics();
        perf.memory_bandwidth_gbps = metrics.memory_bandwidth_gbps;
        perf.available_memory_bytes = metrics.peak_memory_bytes;

    } catch (const std::exception& e) {
        std::cerr << "Error benchmarking backend '" << name << "': " << e.what() << std::endl;
        perf.is_stable = false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    perf.initialization_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return perf;
}

bool BackendManager::IsBackendDisabled(const std::string& name) const {
    auto it = std::find(config_.disabled_backends.begin(), config_.disabled_backends.end(), name);
    return it != config_.disabled_backends.end();
}

std::vector<std::string> BackendManager::GetCandidateBackends() const {
    auto available = BackendRegistry::Instance().GetAvailableBackends();
    std::vector<std::string> candidates;

    for (const auto& name : available) {
        if (!IsBackendDisabled(name)) {
            candidates.push_back(name);
        }
    }

    return candidates;
}

std::string BackendManager::SelectBestCandidate(const std::vector<std::string>& candidates) {
    if (candidates.empty()) {
        return "";
    }

    // Priority order based on configuration
    std::vector<std::string> priority_order;

    if (config_.prefer_gpu) {
        priority_order = {"CUDA", "Metal", "SYCL", "Vulkan", "OpenCL", "CPU"};
    } else {
        priority_order = {"CPU", "SYCL", "CUDA", "Metal", "Vulkan", "OpenCL"};
    }

    for (const std::string& preferred : priority_order) {
        if (std::find(candidates.begin(), candidates.end(), preferred) != candidates.end()) {
            return preferred;
        }
    }

    // Return first available if no preferred backend found
    return candidates[0];
}

// Global functions
BackendManager& GetBackendManager() {
    static BackendManager instance;
    return instance;
}

bool InitializeBackends(const BackendConfig& config) {
    return GetBackendManager().Initialize();
}

void ShutdownBackends() {
    GetBackendManager().Shutdown();
}

BackendInterface* GetActiveBackend() {
    return GetBackendManager().GetActiveBackend();
}

} // namespace backends
} // namespace gemma