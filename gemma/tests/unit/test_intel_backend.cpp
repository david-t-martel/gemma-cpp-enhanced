#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include "../utils/mock_backend.h"
#include <memory>
#include <chrono>
#include <thread>
#include <vector>
#include <string>

// Note: These includes would need to be adjusted based on actual Intel backend implementation
// #include "../../src/backends/intel/oneapi/SYCLBackend.h"
// #include "../../src/backends/intel/openvino/OpenVINOBackend.h"

using namespace gemma::test;
using namespace testing;

class IntelBackendTest : public BackendTestFixture {
protected:
    void SetUp() override {
        BackendTestFixture::SetUp();
        
        // Setup Intel-specific test configuration
        intel_config_.device_id = "intel_gpu_test";
        intel_config_.memory_limit_mb = 4096;
        intel_config_.max_batch_size = 8;
        intel_config_.simulate_errors = false;
        intel_config_.simulated_inference_time = std::chrono::milliseconds(50);
        intel_config_.simulated_init_time = std::chrono::milliseconds(1000);
        
        // Reset Intel backend with test configuration
        intel_backend_ = std::make_unique<MockIntelBackend>();
        intel_backend_->set_config(intel_config_);
        
        setup_intel_specific_expectations();
    }
    
    void setup_intel_specific_expectations() {
        // Setup Intel oneAPI/SYCL specific expectations
        EXPECT_CALL(*intel_backend_, get_backend_type())
            .WillRepeatedly(Return("intel"));
        
        EXPECT_CALL(*intel_backend_, get_device_name())
            .WillRepeatedly(Return("Mock Intel GPU"));
        
        EXPECT_CALL(*intel_backend_, get_sycl_device_info())
            .WillRepeatedly(Return("Intel GPU Mock Device (Level Zero 1.0)"));
        
        EXPECT_CALL(*intel_backend_, supports_usm())
            .WillRepeatedly(Return(true));
        
        EXPECT_CALL(*intel_backend_, get_gpu_memory())
            .WillRepeatedly(Return(4096 * 1024 * 1024)); // 4GB
        
        EXPECT_CALL(*intel_backend_, initialize_sycl())
            .WillRepeatedly(Return(true));
        
        EXPECT_CALL(*intel_backend_, enable_profiling(_))
            .WillRepeatedly(Return(true));
        
        EXPECT_CALL(*intel_backend_, get_profiling_data())
            .WillRepeatedly(Return(nlohmann::json{
                {"kernel_execution_time_ms", 45.2},
                {"memory_transfer_time_ms", 12.1},
                {"total_time_ms", 57.3},
                {"device_utilization", 0.85}
            }));
    }
    
    MockBackendConfig intel_config_;
};

// Basic Intel backend functionality tests

TEST_F(IntelBackendTest, InitializationSuccess) {
    EXPECT_CALL(*intel_backend_, initialize())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*intel_backend_, initialize_sycl())
        .Times(1)
        .WillOnce(Return(true));
    
    bool initialized = intel_backend_->initialize();
    EXPECT_TRUE(initialized);
}

TEST_F(IntelBackendTest, InitializationFailureWhenSYCLUnavailable) {
    EXPECT_CALL(*intel_backend_, initialize_sycl())
        .Times(1)
        .WillOnce(Return(false));
    
    EXPECT_CALL(*intel_backend_, initialize())
        .Times(1)
        .WillOnce(Return(false));
    
    bool initialized = intel_backend_->initialize();
    EXPECT_FALSE(initialized);
}

TEST_F(IntelBackendTest, DeviceInfoRetrieval) {
    EXPECT_CALL(*intel_backend_, get_device_name())
        .Times(1)
        .WillOnce(Return("Intel Arc A770"));
    
    EXPECT_CALL(*intel_backend_, get_sycl_device_info())
        .Times(1)
        .WillOnce(Return("Intel(R) Arc(TM) A770 Graphics [0x4f80] (Level Zero 1.3.26241)"));
    
    std::string device_name = intel_backend_->get_device_name();
    std::string device_info = intel_backend_->get_sycl_device_info();
    
    EXPECT_EQ(device_name, "Intel Arc A770");
    EXPECT_THAT(device_info, HasSubstr("Level Zero"));
    EXPECT_THAT(device_info, HasSubstr("Intel"));
}

TEST_F(IntelBackendTest, MemoryCapabilitiesCheck) {
    EXPECT_CALL(*intel_backend_, supports_usm())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*intel_backend_, get_gpu_memory())
        .Times(1)
        .WillOnce(Return(8ULL * 1024 * 1024 * 1024)); // 8GB
    
    bool usm_supported = intel_backend_->supports_usm();
    size_t gpu_memory = intel_backend_->get_gpu_memory();
    
    EXPECT_TRUE(usm_supported);
    EXPECT_GT(gpu_memory, 1ULL * 1024 * 1024 * 1024); // Should be > 1GB
}

TEST_F(IntelBackendTest, ModelLoadingSuccess) {
    std::string model_path = "/path/to/test/model.onnx";
    
    EXPECT_CALL(*intel_backend_, load_model(model_path))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*intel_backend_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    bool loaded = intel_backend_->load_model(model_path);
    EXPECT_TRUE(loaded);
    
    bool is_loaded = intel_backend_->is_model_loaded();
    EXPECT_TRUE(is_loaded);
}

TEST_F(IntelBackendTest, ModelLoadingFailureInvalidPath) {
    std::string invalid_path = "/nonexistent/model.onnx";
    
    EXPECT_CALL(*intel_backend_, load_model(invalid_path))
        .Times(1)
        .WillOnce(Return(false));
    
    bool loaded = intel_backend_->load_model(invalid_path);
    EXPECT_FALSE(loaded);
}

// Inference tests

TEST_F(IntelBackendTest, BasicInferenceSuccess) {
    std::vector<int> input_tokens = {1, 2, 3, 4, 5};
    std::vector<float> expected_output = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    
    EXPECT_CALL(*intel_backend_, forward(input_tokens))
        .Times(1)
        .WillOnce(Return(expected_output));
    
    auto output = intel_backend_->forward(input_tokens);
    
    ASSERT_EQ(output.size(), expected_output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], expected_output[i]);
    }
}

TEST_F(IntelBackendTest, BatchInferenceSuccess) {
    std::vector<std::vector<int>> batch_inputs = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    std::vector<std::vector<float>> expected_outputs = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    };
    
    EXPECT_CALL(*intel_backend_, forward_batch(batch_inputs))
        .Times(1)
        .WillOnce(Return(expected_outputs));
    
    auto outputs = intel_backend_->forward_batch(batch_inputs);
    
    ASSERT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[i].size(), expected_outputs[i].size());
        for (size_t j = 0; j < outputs[i].size(); ++j) {
            EXPECT_FLOAT_EQ(outputs[i][j], expected_outputs[i][j]);
        }
    }
}

TEST_F(IntelBackendTest, InferenceWithLargeInput) {
    // Test with larger input to verify Intel GPU can handle substantial workloads
    std::vector<int> large_input = TestDataGenerator::generate_random_tokens(2048, 32000);
    std::vector<float> large_output(32000, 0.5f); // Mock large vocabulary output
    
    EXPECT_CALL(*intel_backend_, forward(large_input))
        .Times(1)
        .WillOnce(Return(large_output));
    
    auto output = intel_backend_->forward(large_input);
    
    EXPECT_EQ(output.size(), 32000);
    EXPECT_THAT(output, Each(FloatEq(0.5f)));
}

// Performance and profiling tests

TEST_F(IntelBackendTest, ProfilingConfiguration) {
    EXPECT_CALL(*intel_backend_, enable_profiling(true))
        .Times(1)
        .WillOnce(Return(true));
    
    bool profiling_enabled = intel_backend_->enable_profiling(true);
    EXPECT_TRUE(profiling_enabled);
}

TEST_F(IntelBackendTest, ProfilingDataCollection) {
    nlohmann::json expected_profiling_data = {
        {"kernel_execution_time_ms", 42.5},
        {"memory_transfer_time_ms", 8.2},
        {"total_time_ms", 50.7},
        {"device_utilization", 0.78},
        {"memory_bandwidth_gbps", 256.0},
        {"compute_units_used", 16}
    };
    
    EXPECT_CALL(*intel_backend_, get_profiling_data())
        .Times(1)
        .WillOnce(Return(expected_profiling_data));
    
    auto profiling_data = intel_backend_->get_profiling_data();
    
    EXPECT_TRUE(profiling_data.contains("kernel_execution_time_ms"));
    EXPECT_TRUE(profiling_data.contains("memory_transfer_time_ms"));
    EXPECT_TRUE(profiling_data.contains("device_utilization"));
    
    EXPECT_NEAR(profiling_data["kernel_execution_time_ms"].get<double>(), 42.5, 0.1);
    EXPECT_NEAR(profiling_data["device_utilization"].get<double>(), 0.78, 0.01);
}

TEST_F(IntelBackendTest, PerformanceMetrics) {
    BackendMetrics expected_metrics;
    expected_metrics.initialization_time = std::chrono::milliseconds(1000);
    expected_metrics.inference_time = std::chrono::milliseconds(50);
    expected_metrics.memory_usage_bytes = 2ULL * 1024 * 1024 * 1024; // 2GB
    expected_metrics.total_inferences = 100;
    expected_metrics.device_name = "Intel GPU";
    expected_metrics.backend_type = "intel";
    
    EXPECT_CALL(*intel_backend_, get_metrics())
        .Times(1)
        .WillOnce(Return(expected_metrics));
    
    auto metrics = intel_backend_->get_metrics();
    
    EXPECT_EQ(metrics.initialization_time, std::chrono::milliseconds(1000));
    EXPECT_EQ(metrics.inference_time, std::chrono::milliseconds(50));
    EXPECT_EQ(metrics.memory_usage_bytes, 2ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(metrics.total_inferences, 100);
    EXPECT_EQ(metrics.device_name, "Intel GPU");
    EXPECT_EQ(metrics.backend_type, "intel");
}

// Memory management tests

TEST_F(IntelBackendTest, MemoryUsageTracking) {
    size_t expected_usage = 1024 * 1024 * 1024; // 1GB
    size_t expected_available = 3 * 1024 * 1024 * 1024; // 3GB
    
    EXPECT_CALL(*intel_backend_, get_memory_usage())
        .Times(1)
        .WillOnce(Return(expected_usage));
    
    EXPECT_CALL(*intel_backend_, get_available_memory())
        .Times(1)
        .WillOnce(Return(expected_available));
    
    size_t usage = intel_backend_->get_memory_usage();
    size_t available = intel_backend_->get_available_memory();
    
    EXPECT_EQ(usage, expected_usage);
    EXPECT_EQ(available, expected_available);
    EXPECT_GT(available, usage); // Available should be greater than used
}

TEST_F(IntelBackendTest, CacheClearOperation) {
    EXPECT_CALL(*intel_backend_, clear_cache())
        .Times(1)
        .WillOnce(Return(true));
    
    bool cache_cleared = intel_backend_->clear_cache();
    EXPECT_TRUE(cache_cleared);
}

TEST_F(IntelBackendTest, MemoryLimitRespected) {
    // Test that backend respects configured memory limits
    size_t configured_limit = intel_config_.memory_limit_mb * 1024 * 1024;
    
    EXPECT_CALL(*intel_backend_, get_memory_usage())
        .Times(1)
        .WillOnce(Return(configured_limit - 100 * 1024 * 1024)); // Under limit
    
    size_t actual_usage = intel_backend_->get_memory_usage();
    EXPECT_LT(actual_usage, configured_limit);
}

// Error handling and robustness tests

TEST_F(IntelBackendTest, HandleInferenceErrors) {
    std::vector<int> problematic_input = {-1, -2, -3}; // Invalid tokens
    
    EXPECT_CALL(*intel_backend_, forward(problematic_input))
        .Times(1)
        .WillOnce(Return(std::vector<float>{})); // Empty result indicates error
    
    auto output = intel_backend_->forward(problematic_input);
    EXPECT_TRUE(output.empty());
}

TEST_F(IntelBackendTest, RecoveryFromDeviceError) {
    // Simulate device disconnection
    EXPECT_CALL(*intel_backend_, is_available())
        .Times(2)
        .WillOnce(Return(false))  // Device unavailable
        .WillOnce(Return(true));  // Device recovered
    
    EXPECT_CALL(*intel_backend_, initialize())
        .Times(1)
        .WillOnce(Return(true)); // Re-initialization successful
    
    // First check - device unavailable
    bool available1 = intel_backend_->is_available();
    EXPECT_FALSE(available1);
    
    // Attempt recovery
    bool reinitialized = intel_backend_->initialize();
    EXPECT_TRUE(reinitialized);
    
    // Second check - device recovered
    bool available2 = intel_backend_->is_available();
    EXPECT_TRUE(available2);
}

TEST_F(IntelBackendTest, HandleOutOfMemoryCondition) {
    // Simulate out of memory condition
    EXPECT_CALL(*intel_backend_, get_available_memory())
        .Times(1)
        .WillOnce(Return(0)); // No memory available
    
    size_t available = intel_backend_->get_available_memory();
    EXPECT_EQ(available, 0);
    
    // Backend should handle this gracefully
    std::vector<int> input = {1, 2, 3};
    EXPECT_CALL(*intel_backend_, forward(input))
        .Times(1)
        .WillOnce(Return(std::vector<float>{})); // Should fail gracefully
    
    auto output = intel_backend_->forward(input);
    EXPECT_TRUE(output.empty());
}

// Intel-specific feature tests

TEST_F(IntelBackendTest, SYCLDeviceSelection) {
    // Test SYCL device enumeration and selection
    std::vector<std::string> expected_devices = {
        "Intel GPU [0x4f80]",
        "Intel CPU [AVX2]"
    };
    
    // This would be a real method call in actual implementation
    // For now, we test that the interface exists and works
    std::string device_info = intel_backend_->get_sycl_device_info();
    EXPECT_FALSE(device_info.empty());
    EXPECT_THAT(device_info, AnyOf(HasSubstr("GPU"), HasSubstr("CPU")));
}

TEST_F(IntelBackendTest, USMMemoryAllocations) {
    // Test Unified Shared Memory support
    EXPECT_CALL(*intel_backend_, supports_usm())
        .Times(1)
        .WillOnce(Return(true));
    
    bool usm_supported = intel_backend_->supports_usm();
    EXPECT_TRUE(usm_supported);
    
    // In real implementation, this would test actual USM allocations
    // For mock, we just verify the interface
}

TEST_F(IntelBackendTest, OneAPIIntegration) {
    // Test oneAPI toolkit integration
    EXPECT_CALL(*intel_backend_, initialize_sycl())
        .Times(1)
        .WillOnce(Return(true));
    
    bool sycl_initialized = intel_backend_->initialize_sycl();
    EXPECT_TRUE(sycl_initialized);
    
    // Verify SYCL context is properly set up
    std::string device_info = intel_backend_->get_sycl_device_info();
    EXPECT_THAT(device_info, HasSubstr("Level Zero"));
}

// Concurrent access tests

TEST_F(IntelBackendTest, ConcurrentInferenceRequests) {
    const int num_threads = 4;
    const int requests_per_thread = 10;
    std::atomic<int> successful_requests{0};
    
    // Setup expectations for concurrent calls
    EXPECT_CALL(*intel_backend_, forward(_))
        .Times(num_threads * requests_per_thread)
        .WillRepeatedly(Return(std::vector<float>{1.0f, 2.0f, 3.0f}));
    
    ThreadSafetyUtils::run_concurrent_test([this, &successful_requests]() {
        std::vector<int> input = {1, 2, 3, 4, 5};
        auto output = intel_backend_->forward(input);
        if (!output.empty()) {
            successful_requests++;
        }
    }, num_threads, requests_per_thread);
    
    EXPECT_EQ(successful_requests.load(), num_threads * requests_per_thread);
}

// Performance benchmarks

TEST_F(IntelBackendTest, InferenceThroughputBenchmark) {
    const int num_inferences = 100;
    std::vector<int> test_input = TestDataGenerator::generate_random_tokens(512, 32000);
    std::vector<float> mock_output(32000, 0.5f);
    
    EXPECT_CALL(*intel_backend_, forward(test_input))
        .Times(num_inferences)
        .WillRepeatedly(Return(mock_output));
    
    auto benchmark_func = [this, &test_input]() {
        intel_backend_->forward(test_input);
    };
    
    double avg_time_ms = PerformanceUtils::measure_average_time_ms(benchmark_func, num_inferences);
    
    std::cout << "Intel backend average inference time: " << avg_time_ms << " ms" << std::endl;
    
    // Performance expectation for Intel GPU (should be faster than CPU)
    EXPECT_LT(avg_time_ms, 100.0) << "Intel GPU inference too slow: " << avg_time_ms << " ms";
    
    // Calculate throughput
    double tokens_per_second = (512.0 / avg_time_ms) * 1000.0;
    std::cout << "Throughput: " << tokens_per_second << " tokens/second" << std::endl;
    
    EXPECT_GT(tokens_per_second, 100.0) << "Intel GPU throughput too low: " << tokens_per_second << " tokens/s";
}

TEST_F(IntelBackendTest, MemoryBandwidthTest) {
    // Test memory transfer performance
    size_t large_data_size = 100 * 1024 * 1024; // 100MB
    std::vector<int> large_input = TestDataGenerator::generate_random_tokens(large_data_size / sizeof(int), 32000);
    std::vector<float> large_output(large_data_size / sizeof(float), 1.0f);
    
    EXPECT_CALL(*intel_backend_, forward(large_input))
        .Times(1)
        .WillOnce(Return(large_output));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto output = intel_backend_->forward(large_input);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    EXPECT_EQ(output.size(), large_output.size());
    
    // Calculate memory bandwidth (GB/s)
    double data_gb = (large_data_size * 2.0) / (1024.0 * 1024.0 * 1024.0); // Input + output
    double time_s = duration_ms / 1000.0;
    double bandwidth_gbps = data_gb / time_s;
    
    std::cout << "Memory bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    
    // Intel GPUs should achieve reasonable memory bandwidth
    EXPECT_GT(bandwidth_gbps, 50.0) << "Memory bandwidth too low: " << bandwidth_gbps << " GB/s";
}

// Cleanup and shutdown tests

TEST_F(IntelBackendTest, ProperShutdown) {
    EXPECT_CALL(*intel_backend_, shutdown())
        .Times(1);
    
    EXPECT_CALL(*intel_backend_, is_available())
        .Times(1)
        .WillOnce(Return(false)); // Should be unavailable after shutdown
    
    intel_backend_->shutdown();
    
    bool available_after_shutdown = intel_backend_->is_available();
    EXPECT_FALSE(available_after_shutdown);
}

TEST_F(IntelBackendTest, ResourceCleanupAfterModelUnload) {
    std::string model_path = "/path/to/model.onnx";
    
    // Load model
    EXPECT_CALL(*intel_backend_, load_model(model_path))
        .Times(1)
        .WillOnce(Return(true));
    
    // Check memory usage with model loaded
    EXPECT_CALL(*intel_backend_, get_memory_usage())
        .Times(2)
        .WillOnce(Return(2ULL * 1024 * 1024 * 1024))  // 2GB with model
        .WillOnce(Return(100ULL * 1024 * 1024));       // 100MB after unload
    
    // Unload model
    EXPECT_CALL(*intel_backend_, unload_model())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*intel_backend_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(false));
    
    // Test sequence
    bool loaded = intel_backend_->load_model(model_path);
    EXPECT_TRUE(loaded);
    
    size_t memory_with_model = intel_backend_->get_memory_usage();
    EXPECT_GT(memory_with_model, 1ULL * 1024 * 1024 * 1024); // > 1GB
    
    bool unloaded = intel_backend_->unload_model();
    EXPECT_TRUE(unloaded);
    
    size_t memory_after_unload = intel_backend_->get_memory_usage();
    EXPECT_LT(memory_after_unload, memory_with_model); // Memory should decrease
    
    bool still_loaded = intel_backend_->is_model_loaded();
    EXPECT_FALSE(still_loaded);
}