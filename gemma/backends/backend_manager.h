/**
 * @file backend_manager.h
 * @brief High-level backend management for Gemma.cpp
 */

#pragma once

#include "backend_interface.h"
#include "backend_registry.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace gemma {
namespace backends {

/**
 * @brief Backend configuration options
 */
struct BackendConfig {
    std::string preferred_backend = "auto";  // "auto", "CUDA", "SYCL", etc.
    bool enable_fallback = true;             // Allow fallback to other backends
    std::vector<std::string> disabled_backends; // Explicitly disabled backends
    bool enable_benchmarking = false;       // Run initial benchmarks
    bool verbose_logging = false;           // Enable detailed logging

    // Performance hints
    bool prefer_gpu = true;                 // Prefer GPU over CPU
    bool prefer_high_memory = false;        // Prefer backends with more memory
    size_t min_memory_gb = 0;               // Minimum memory requirement
};

/**
 * @brief Backend performance information
 */
struct BackendPerformance {
    std::string name;
    double matrix_multiply_gflops = 0.0;
    double attention_gflops = 0.0;
    double memory_bandwidth_gbps = 0.0;
    size_t available_memory_bytes = 0;
    double initialization_time_ms = 0.0;
    bool is_stable = true;
};

/**
 * @brief High-level backend manager for Gemma.cpp
 */
class BackendManager {
public:
    /**
     * @brief Constructor with configuration
     * @param config Backend configuration
     */
    explicit BackendManager(const BackendConfig& config = BackendConfig{});

    /**
     * @brief Destructor
     */
    ~BackendManager();

    /**
     * @brief Initialize backend system
     * @return true if at least one backend is available
     */
    bool Initialize();

    /**
     * @brief Shutdown all backends
     */
    void Shutdown();

    /**
     * @brief Get the active backend
     * @return Pointer to active backend, or nullptr if none
     */
    BackendInterface* GetActiveBackend() const;

    /**
     * @brief Get backend by name
     * @param name Backend name
     * @return Pointer to backend, or nullptr if not found
     */
    BackendInterface* GetBackend(const std::string& name) const;

    /**
     * @brief Switch to a different backend
     * @param name Backend name
     * @return true if switch successful
     */
    bool SwitchBackend(const std::string& name);

    /**
     * @brief Get list of available backends
     * @return Vector of backend names
     */
    std::vector<std::string> GetAvailableBackends() const;

    /**
     * @brief Get backend performance information
     * @return Vector of performance data for all backends
     */
    std::vector<BackendPerformance> GetBackendPerformance() const;

    /**
     * @brief Check if a backend supports required capabilities
     * @param capabilities Required capabilities
     * @return true if active backend supports all capabilities
     */
    bool SupportsCapabilities(const std::vector<BackendCapability>& capabilities) const;

    /**
     * @brief Find best backend for specific capabilities
     * @param capabilities Required capabilities
     * @return Best backend name, or empty string if none suitable
     */
    std::string FindBestBackend(const std::vector<BackendCapability>& capabilities) const;

    /**
     * @brief Run benchmarks on all available backends
     * @return Map of backend names to performance scores
     */
    std::map<std::string, BackendPerformance> RunBenchmarks();

    /**
     * @brief Get detailed status report
     * @return Status report as string
     */
    std::string GetStatusReport() const;

    /**
     * @brief Register all available backends
     * This function registers CUDA, SYCL, Vulkan, etc. based on compile-time availability
     */
    static void RegisterAllBackends();

    /**
     * @brief Auto-detect and select best backend
     * @param config Configuration for selection
     * @return Selected backend name
     */
    static std::string AutoSelectBackend(const BackendConfig& config = BackendConfig{});

private:
    BackendConfig config_;
    std::string active_backend_name_;
    std::map<std::string, std::unique_ptr<BackendInterface>> active_backends_;
    std::map<std::string, BackendPerformance> performance_cache_;
    bool initialized_;

    // Internal methods
    bool InitializeBackend(const std::string& name);
    void ShutdownBackend(const std::string& name);
    BackendPerformance BenchmarkBackend(BackendInterface* backend, const std::string& name);
    bool IsBackendDisabled(const std::string& name) const;
    std::vector<std::string> GetCandidateBackends() const;
    std::string SelectBestCandidate(const std::vector<std::string>& candidates);
};

/**
 * @brief Global backend manager instance
 * @return Reference to singleton instance
 */
BackendManager& GetBackendManager();

/**
 * @brief Initialize global backend system with configuration
 * @param config Backend configuration
 * @return true if initialization successful
 */
bool InitializeBackends(const BackendConfig& config = BackendConfig{});

/**
 * @brief Shutdown global backend system
 */
void ShutdownBackends();

/**
 * @brief Get the currently active backend
 * @return Pointer to active backend
 */
BackendInterface* GetActiveBackend();

} // namespace backends
} // namespace gemma