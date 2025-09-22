#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <nlohmann/json.hpp>

// Base backend interface (assuming it exists)
// #include "../../src/backends/base/Backend.h"

namespace gemma {
namespace test {

/**
 * @brief Backend performance metrics
 */
struct BackendMetrics {
    std::chrono::nanoseconds initialization_time{0};
    std::chrono::nanoseconds inference_time{0};
    size_t memory_usage_bytes = 0;
    size_t total_inferences = 0;
    std::atomic<bool> is_busy{false};
    std::string device_name = "mock_device";
    std::string backend_type = "mock";
};

/**
 * @brief Backend status enumeration
 */
enum class BackendStatus {
    Uninitialized,
    Initializing,
    Ready,
    Busy,
    Error,
    Shutdown
};

/**
 * @brief Mock backend configuration
 */
struct MockBackendConfig {
    std::string device_id = "mock_device_0";
    size_t memory_limit_mb = 1024;
    int max_batch_size = 1;
    bool simulate_errors = false;
    double error_probability = 0.0;
    std::chrono::milliseconds simulated_inference_time{100};
    std::chrono::milliseconds simulated_init_time{500};
};

/**
 * @brief Mock backend base class
 */
class MockBackendBase {
public:
    MockBackendBase(const MockBackendConfig& config = MockBackendConfig{})
        : config_(config), status_(BackendStatus::Uninitialized) {}
    
    virtual ~MockBackendBase() = default;
    
    // Core interface methods
    MOCK_METHOD(bool, initialize, (), ());
    MOCK_METHOD(void, shutdown, (), ());
    MOCK_METHOD(bool, is_available, (), (const));
    MOCK_METHOD(BackendStatus, get_status, (), (const));
    MOCK_METHOD(std::string, get_device_name, (), (const));
    MOCK_METHOD(std::string, get_backend_type, (), (const));
    
    // Model operations
    MOCK_METHOD(bool, load_model, (const std::string& model_path), ());
    MOCK_METHOD(bool, unload_model, (), ());
    MOCK_METHOD(bool, is_model_loaded, (), (const));
    
    // Inference operations
    MOCK_METHOD(std::vector<float>, forward, (const std::vector<int>& input_tokens), ());
    MOCK_METHOD(std::vector<std::vector<float>>, forward_batch, (const std::vector<std::vector<int>>& batch_inputs), ());
    
    // Memory management
    MOCK_METHOD(size_t, get_memory_usage, (), (const));
    MOCK_METHOD(size_t, get_available_memory, (), (const));
    MOCK_METHOD(bool, clear_cache, (), ());
    
    // Performance metrics
    MOCK_METHOD(BackendMetrics, get_metrics, (), (const));
    MOCK_METHOD(void, reset_metrics, (), ());
    
    // Configuration
    const MockBackendConfig& get_config() const { return config_; }
    void set_config(const MockBackendConfig& config) { config_ = config; }

protected:
    MockBackendConfig config_;
    std::atomic<BackendStatus> status_;
    BackendMetrics metrics_;
    mutable std::mutex backend_mutex_;
};

/**
 * @brief Mock CPU backend implementation
 */
class MockCPUBackend : public MockBackendBase {
public:
    MockCPUBackend() : MockBackendBase() {
        config_.device_id = "cpu_mock";
        config_.memory_limit_mb = 8192;
        metrics_.backend_type = "cpu";
        metrics_.device_name = "Mock CPU Device";
    }
    
    // CPU-specific methods
    MOCK_METHOD(int, get_thread_count, (), (const));
    MOCK_METHOD(void, set_thread_count, (int threads), ());
    MOCK_METHOD(bool, supports_simd, (), (const));
    MOCK_METHOD(std::vector<std::string>, get_simd_features, (), (const));
};

/**
 * @brief Mock Intel backend implementation
 */
class MockIntelBackend : public MockBackendBase {
public:
    MockIntelBackend() : MockBackendBase() {
        config_.device_id = "intel_gpu_mock";
        config_.memory_limit_mb = 4096;
        metrics_.backend_type = "intel";
        metrics_.device_name = "Mock Intel GPU";
    }
    
    // Intel-specific methods
    MOCK_METHOD(bool, initialize_sycl, (), ());
    MOCK_METHOD(std::string, get_sycl_device_info, (), (const));
    MOCK_METHOD(bool, supports_usm, (), (const));
    MOCK_METHOD(size_t, get_gpu_memory, (), (const));
    MOCK_METHOD(bool, enable_profiling, (bool enable), ());
    MOCK_METHOD(nlohmann::json, get_profiling_data, (), (const));
};

/**
 * @brief Mock CUDA backend implementation
 */
class MockCUDABackend : public MockBackendBase {
public:
    MockCUDABackend() : MockBackendBase() {
        config_.device_id = "cuda_mock";
        config_.memory_limit_mb = 8192;
        metrics_.backend_type = "cuda";
        metrics_.device_name = "Mock CUDA Device";
    }
    
    // CUDA-specific methods
    MOCK_METHOD(int, get_device_id, (), (const));
    MOCK_METHOD(std::string, get_cuda_version, (), (const));
    MOCK_METHOD(size_t, get_vram_total, (), (const));
    MOCK_METHOD(size_t, get_vram_free, (), (const));
    MOCK_METHOD(bool, supports_tensor_cores, (), (const));
    MOCK_METHOD(void, synchronize, (), ());
};

/**
 * @brief Mock Vulkan backend implementation
 */
class MockVulkanBackend : public MockBackendBase {
public:
    MockVulkanBackend() : MockBackendBase() {
        config_.device_id = "vulkan_mock";
        config_.memory_limit_mb = 4096;
        metrics_.backend_type = "vulkan";
        metrics_.device_name = "Mock Vulkan Device";
    }
    
    // Vulkan-specific methods
    MOCK_METHOD(bool, initialize_vulkan, (), ());
    MOCK_METHOD(std::string, get_vulkan_version, (), (const));
    MOCK_METHOD(std::vector<std::string>, get_available_devices, (), (const));
    MOCK_METHOD(bool, supports_compute_shaders, (), (const));
    MOCK_METHOD(uint32_t, get_queue_family_index, (), (const));
};

/**
 * @brief Backend factory for creating mock backends
 */
class MockBackendFactory {
public:
    enum class BackendType {
        CPU,
        Intel,
        CUDA,
        Vulkan
    };
    
    static std::unique_ptr<MockBackendBase> create_backend(BackendType type, 
                                                          const MockBackendConfig& config = MockBackendConfig{}) {
        switch (type) {
            case BackendType::CPU:
                return std::make_unique<MockCPUBackend>();
            case BackendType::Intel:
                return std::make_unique<MockIntelBackend>();
            case BackendType::CUDA:
                return std::make_unique<MockCUDABackend>();
            case BackendType::Vulkan:
                return std::make_unique<MockVulkanBackend>();
            default:
                return nullptr;
        }
    }
    
    static std::vector<std::unique_ptr<MockBackendBase>> create_all_backends() {
        std::vector<std::unique_ptr<MockBackendBase>> backends;
        backends.push_back(create_backend(BackendType::CPU));
        backends.push_back(create_backend(BackendType::Intel));
        backends.push_back(create_backend(BackendType::CUDA));
        backends.push_back(create_backend(BackendType::Vulkan));
        return backends;
    }
};

/**
 * @brief Mock backend manager for testing backend switching
 */
class MockBackendManager {
public:
    MockBackendManager() = default;
    
    MOCK_METHOD(bool, register_backend, (const std::string& name, std::unique_ptr<MockBackendBase> backend), ());
    MOCK_METHOD(bool, set_active_backend, (const std::string& name), ());
    MOCK_METHOD(MockBackendBase*, get_active_backend, (), ());
    MOCK_METHOD(MockBackendBase*, get_backend, (const std::string& name), ());
    MOCK_METHOD(std::vector<std::string>, list_available_backends, (), (const));
    MOCK_METHOD(std::vector<std::string>, list_initialized_backends, (), (const));
    MOCK_METHOD(bool, remove_backend, (const std::string& name), ());
    MOCK_METHOD(void, shutdown_all_backends, (), ());
    MOCK_METHOD(nlohmann::json, get_backend_status, (), (const));
    
    // Test utilities
    void setup_default_backends() {
        auto backends = MockBackendFactory::create_all_backends();
        for (size_t i = 0; i < backends.size(); ++i) {
            std::string name = "backend_" + std::to_string(i);
            register_backend(name, std::move(backends[i]));
        }
    }
};

/**
 * @brief Performance benchmark mock for backend testing
 */
class MockBackendBenchmark {
public:
    struct BenchmarkResult {
        std::string backend_name;
        double avg_inference_time_ms;
        double min_inference_time_ms;
        double max_inference_time_ms;
        double throughput_tokens_per_second;
        size_t peak_memory_usage_mb;
        bool completed_successfully;
        std::string error_message;
    };
    
    MOCK_METHOD(BenchmarkResult, run_inference_benchmark, 
                (MockBackendBase* backend, const std::vector<std::vector<int>>& test_inputs, int iterations), ());
    MOCK_METHOD(std::vector<BenchmarkResult>, compare_backends, 
                (const std::vector<MockBackendBase*>& backends, const std::vector<std::vector<int>>& test_inputs), ());
    MOCK_METHOD(bool, run_stress_test, (MockBackendBase* backend, std::chrono::minutes duration), ());
    MOCK_METHOD(nlohmann::json, generate_benchmark_report, (const std::vector<BenchmarkResult>& results), ());
};

/**
 * @brief Error simulation utilities for testing error handling
 */
class BackendErrorSimulator {
public:
    enum class ErrorType {
        InitializationFailure,
        ModelLoadFailure,
        InferenceFailure,
        MemoryError,
        DeviceDisconnected,
        TimeoutError
    };
    
    static void inject_error(MockBackendBase* backend, ErrorType error_type) {
        switch (error_type) {
            case ErrorType::InitializationFailure:
                ON_CALL(*backend, initialize()).WillByDefault(::testing::Return(false));
                break;
            case ErrorType::ModelLoadFailure:
                ON_CALL(*backend, load_model(::testing::_)).WillByDefault(::testing::Return(false));
                break;
            case ErrorType::InferenceFailure:
                ON_CALL(*backend, forward(::testing::_)).WillByDefault(::testing::Return(std::vector<float>{}));
                break;
            case ErrorType::MemoryError:
                ON_CALL(*backend, get_available_memory()).WillByDefault(::testing::Return(0));
                break;
            case ErrorType::DeviceDisconnected:
                ON_CALL(*backend, is_available()).WillByDefault(::testing::Return(false));
                break;
            case ErrorType::TimeoutError:
                ON_CALL(*backend, forward(::testing::_)).WillByDefault(::testing::InvokeWithoutArgs([]() {
                    std::this_thread::sleep_for(std::chrono::seconds(10));
                    return std::vector<float>{};
                }));
                break;
        }
    }
    
    static void setup_random_failures(MockBackendBase* backend, double failure_rate = 0.1) {
        ON_CALL(*backend, forward(::testing::_))
            .WillByDefault(::testing::InvokeWithoutArgs([failure_rate]() {
                static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
                static std::uniform_real_distribution<double> dist(0.0, 1.0);
                
                if (dist(rng) < failure_rate) {
                    return std::vector<float>{}; // Simulate failure
                }
                return std::vector<float>{1.0f, 2.0f, 3.0f}; // Simulate success
            }));
    }
};

/**
 * @brief Backend test fixture base class
 */
class BackendTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock backends
        cpu_backend_ = std::make_unique<MockCPUBackend>();
        intel_backend_ = std::make_unique<MockIntelBackend>();
        cuda_backend_ = std::make_unique<MockCUDABackend>();
        vulkan_backend_ = std::make_unique<MockVulkanBackend>();
        
        // Setup default expectations
        setup_default_expectations();
    }
    
    void TearDown() override {
        // Clean up backends
        cpu_backend_.reset();
        intel_backend_.reset();
        cuda_backend_.reset();
        vulkan_backend_.reset();
    }
    
    void setup_default_expectations() {
        // Setup common successful operations
        for (auto* backend : {
            static_cast<MockBackendBase*>(cpu_backend_.get()),
            static_cast<MockBackendBase*>(intel_backend_.get()),
            static_cast<MockBackendBase*>(cuda_backend_.get()),
            static_cast<MockBackendBase*>(vulkan_backend_.get())
        }) {
            ON_CALL(*backend, initialize()).WillByDefault(::testing::Return(true));
            ON_CALL(*backend, is_available()).WillByDefault(::testing::Return(true));
            ON_CALL(*backend, get_status()).WillByDefault(::testing::Return(BackendStatus::Ready));
            ON_CALL(*backend, load_model(::testing::_)).WillByDefault(::testing::Return(true));
            ON_CALL(*backend, is_model_loaded()).WillByDefault(::testing::Return(true));
            ON_CALL(*backend, forward(::testing::_)).WillByDefault(::testing::Return(std::vector<float>{1.0f, 2.0f, 3.0f}));
            ON_CALL(*backend, get_memory_usage()).WillByDefault(::testing::Return(1024 * 1024)); // 1MB
            ON_CALL(*backend, get_available_memory()).WillByDefault(::testing::Return(1024 * 1024 * 1024)); // 1GB
        }
    }
    
    std::unique_ptr<MockCPUBackend> cpu_backend_;
    std::unique_ptr<MockIntelBackend> intel_backend_;
    std::unique_ptr<MockCUDABackend> cuda_backend_;
    std::unique_ptr<MockVulkanBackend> vulkan_backend_;
};

} // namespace test
} // namespace gemma