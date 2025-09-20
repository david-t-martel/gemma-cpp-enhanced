#pragma once

/**
 * @file backend_registry.h
 * @brief Registry for managing available hardware backends
 */

#include "backend_interface.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace gemma {
namespace backends {

/**
 * @brief Backend priority levels
 */
enum class BackendPriority {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Backend information structure
 */
struct BackendInfo {
    std::string name;
    std::string version;
    BackendPriority priority;
    std::vector<BackendCapability> capabilities;
    bool is_available;
    std::string description;
    BackendFactory factory;
};

/**
 * @brief Registry for managing hardware acceleration backends
 */
class BackendRegistry {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to registry instance
     */
    static BackendRegistry& Instance();

    /**
     * @brief Register a backend
     * @param info Backend information
     * @return true if registration successful
     */
    bool RegisterBackend(const BackendInfo& info);

    /**
     * @brief Unregister a backend
     * @param name Backend name
     * @return true if unregistration successful
     */
    bool UnregisterBackend(const std::string& name);

    /**
     * @brief Get list of available backend names
     * @return Vector of backend names
     */
    std::vector<std::string> GetAvailableBackends() const;

    /**
     * @brief Get backend information
     * @param name Backend name
     * @return Backend info, or nullptr if not found
     */
    const BackendInfo* GetBackendInfo(const std::string& name) const;

    /**
     * @brief Create backend instance
     * @param name Backend name
     * @return Backend instance, or nullptr if creation failed
     */
    std::unique_ptr<BackendInterface> CreateBackend(const std::string& name) const;

    /**
     * @brief Find best backend for given capabilities
     * @param required_capabilities Required capabilities
     * @param preferred_priority Minimum priority level
     * @return Best backend name, or empty string if none found
     */
    std::string FindBestBackend(
        const std::vector<BackendCapability>& required_capabilities,
        BackendPriority preferred_priority = BackendPriority::MEDIUM) const;

    /**
     * @brief Auto-select backend based on system configuration
     * @return Selected backend name, or empty string if none suitable
     */
    std::string AutoSelectBackend() const;

    /**
     * @brief Check if a backend supports all required capabilities
     * @param name Backend name
     * @param capabilities Required capabilities
     * @return true if all capabilities are supported
     */
    bool SupportsCapabilities(const std::string& name, 
                             const std::vector<BackendCapability>& capabilities) const;

    /**
     * @brief Get all backends supporting a specific capability
     * @param capability Required capability
     * @return Vector of backend names
     */
    std::vector<std::string> GetBackendsWithCapability(BackendCapability capability) const;

    /**
     * @brief Initialize all registered backends
     * @return true if at least one backend initialized successfully
     */
    bool InitializeBackends();

    /**
     * @brief Shutdown all backends
     */
    void ShutdownBackends();

    /**
     * @brief Check if any backends are initialized
     * @return true if at least one backend is available
     */
    bool HasAvailableBackends() const;

    /**
     * @brief Get performance comparison of backends
     * @param operation_type Type of operation to benchmark
     * @return Map of backend names to performance scores
     */
    std::map<std::string, double> BenchmarkBackends(const std::string& operation_type) const;

private:
    BackendRegistry() = default;
    ~BackendRegistry() = default;
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    std::map<std::string, BackendInfo> backends_;
    std::map<std::string, std::unique_ptr<BackendInterface>> active_backends_;
    mutable std::mutex registry_mutex_;

    double BenchmarkBackend(BackendInterface* backend, const std::string& operation_type) const;
};

/**
 * @brief Helper macro for registering backends
 */
#define REGISTER_BACKEND(name, factory_func, priority, capabilities) \
    namespace { \
        struct BackendRegistrar_##name { \
            BackendRegistrar_##name() { \
                BackendInfo info; \
                info.name = #name; \
                info.factory = factory_func; \
                info.priority = priority; \
                info.capabilities = capabilities; \
                info.is_available = true; \
                BackendRegistry::Instance().RegisterBackend(info); \
            } \
        }; \
        static BackendRegistrar_##name g_backend_registrar_##name; \
    }

} // namespace backends
} // namespace gemma