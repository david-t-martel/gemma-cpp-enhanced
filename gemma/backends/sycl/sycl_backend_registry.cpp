/**
 * @file sycl_backend_registry.cpp
 * @brief Backend registration for Intel SYCL backend
 *
 * Registers the SYCL backend with the Gemma backend registry system
 * for automatic discovery and selection.
 */

#include "sycl_backend.h"
#include "../backend_registry.h"
#include <vector>
#include <iostream>

namespace gemma {
namespace backends {
namespace sycl {

/**
 * @brief Register SYCL backend with the backend registry
 */
void RegisterSyclBackend() {
    if (!IsSyclBackendAvailable()) {
        std::cout << "SYCL Backend: Hardware not available, skipping registration" << std::endl;
        return;
    }

    BackendInfo info;
    info.name = "Intel SYCL";
    info.version = GetOneAPIVersion();
    info.priority = BackendPriority::HIGH;  // High priority for Intel hardware
    info.description = "Intel SYCL backend with GPU, NPU, and CPU support using oneAPI";
    info.factory = CreateSyclBackend;
    info.is_available = true;

    // Define supported capabilities
    info.capabilities = {
        BackendCapability::MATRIX_MULTIPLICATION,
        BackendCapability::ATTENTION_COMPUTATION,
        BackendCapability::ACTIVATION_FUNCTIONS,
        BackendCapability::MEMORY_POOLING,
        BackendCapability::ASYNC_EXECUTION,
        BackendCapability::MULTI_PRECISION
    };

    bool registered = BackendRegistry::Instance().RegisterBackend(info);

    if (registered) {
        std::cout << "SYCL Backend: Successfully registered with backend registry" << std::endl;

        // Print detected devices
        auto backend = CreateSyclBackend();
        if (backend && backend->Initialize()) {
            auto sycl_backend = static_cast<SyclBackend*>(backend.get());
            auto devices = sycl_backend->GetAvailableDevices();

            std::cout << "SYCL Backend: Detected " << devices.size() << " compatible devices:" << std::endl;
            for (const auto& device : devices) {
                std::string type_str;
                switch (device.type) {
                    case SyclDeviceType::GPU: type_str = "GPU"; break;
                    case SyclDeviceType::NPU: type_str = "NPU"; break;
                    case SyclDeviceType::CPU: type_str = "CPU"; break;
                    default: type_str = "Unknown"; break;
                }

                std::cout << "  [" << device.device_id << "] " << device.name
                          << " (" << type_str << ", " << device.vendor << ")" << std::endl;
                std::cout << "      Memory: " << (device.max_memory_bytes / (1024*1024)) << " MB, "
                          << "FP16: " << (device.supports_fp16 ? "Yes" : "No") << ", "
                          << "USM: " << (device.supports_unified_memory ? "Yes" : "No") << std::endl;
            }

            backend->Shutdown();
        }
    } else {
        std::cerr << "SYCL Backend: Failed to register with backend registry" << std::endl;
    }
}

/**
 * @brief Unregister SYCL backend from the registry
 */
void UnregisterSyclBackend() {
    BackendRegistry::Instance().UnregisterBackend("Intel SYCL");
}

} // namespace sycl
} // namespace backends
} // namespace gemma

// Automatic registration using static initialization
namespace {
    struct SyclBackendRegistrar {
        SyclBackendRegistrar() {
            gemma::backends::sycl::RegisterSyclBackend();
        }

        ~SyclBackendRegistrar() {
            gemma::backends::sycl::UnregisterSyclBackend();
        }
    };

    // Static instance to trigger registration
    static SyclBackendRegistrar g_sycl_registrar;
}