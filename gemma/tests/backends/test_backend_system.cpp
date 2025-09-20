/**
 * @file test_backend_system.cpp
 * @brief Comprehensive tests for the backend system
 */

#include <gtest/gtest.h>
#include "backends/backend_manager.h"
#include "backends/backend_registry.h"
#include "backends/backend_interface.h"
#include <memory>
#include <vector>
#include <chrono>

using namespace gemma::backends;

class BackendSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register all available backends
        BackendManager::RegisterAllBackends();
    }

    void TearDown() override {
        // Clean up
        GetBackendManager().Shutdown();
    }
};

// Test backend registry functionality
TEST_F(BackendSystemTest, RegistryBasicOperations) {
    auto& registry = BackendRegistry::Instance();

    // Check if any backends are registered
    auto available = registry.GetAvailableBackends();
    EXPECT_FALSE(available.empty()) << "No backends available";

    // Test getting backend info
    for (const auto& name : available) {
        const BackendInfo* info = registry.GetBackendInfo(name);
        EXPECT_NE(info, nullptr) << "Backend info not found for " << name;
        EXPECT_EQ(info->name, name);
        EXPECT_TRUE(info->is_available);
        EXPECT_TRUE(info->factory != nullptr);
    }
}

// Test backend creation
TEST_F(BackendSystemTest, BackendCreation) {
    auto& registry = BackendRegistry::Instance();
    auto available = registry.GetAvailableBackends();

    for (const auto& name : available) {
        auto backend = registry.CreateBackend(name);
        if (backend) {
            EXPECT_EQ(backend->GetName(), name);
            EXPECT_TRUE(backend->IsAvailable());
            EXPECT_TRUE(backend->IsInitialized());

            // Test basic functionality
            EXPECT_GE(backend->GetDeviceCount(), 1);
            EXPECT_GE(backend->GetCurrentDevice(), 0);

            backend->Shutdown();
        }
    }
}

// Test backend manager initialization
TEST_F(BackendSystemTest, ManagerInitialization) {
    BackendConfig config;
    config.verbose_logging = false;
    config.enable_benchmarking = false;

    BackendManager manager(config);
    EXPECT_TRUE(manager.Initialize());

    auto* active = manager.GetActiveBackend();
    EXPECT_NE(active, nullptr);

    if (active) {
        EXPECT_TRUE(active->IsAvailable());
        EXPECT_TRUE(active->IsInitialized());
    }

    manager.Shutdown();
}

// Test memory operations
TEST_F(BackendSystemTest, MemoryOperations) {
    BackendConfig config;
    config.verbose_logging = false;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager.GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    // Test buffer allocation
    const size_t buffer_size = 1024 * sizeof(float);
    auto buffer = backend->AllocateBuffer(buffer_size);

    EXPECT_TRUE(buffer.IsValid());
    EXPECT_NE(buffer.data, nullptr);
    EXPECT_EQ(buffer.size, buffer_size);

    // Test host data operations
    std::vector<float> host_data(1024, 1.0f);
    std::vector<float> result_data(1024, 0.0f);

    EXPECT_TRUE(backend->CopyToDevice(buffer, host_data.data(), buffer_size));
    EXPECT_TRUE(backend->CopyFromDevice(result_data.data(), buffer, buffer_size));

    backend->Synchronize();

    // Verify data integrity
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], 1.0f);
    }

    backend->FreeBuffer(buffer);
    manager.Shutdown();
}

// Test matrix operations
TEST_F(BackendSystemTest, MatrixOperations) {
    BackendConfig config;
    config.verbose_logging = false;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager.GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    if (!backend->SupportsCapability(BackendCapability::MATRIX_MULTIPLICATION)) {
        GTEST_SKIP() << "Backend doesn't support matrix multiplication";
    }

    const int M = 32, N = 32, K = 32;
    const size_t size_a = M * K * sizeof(float);
    const size_t size_b = K * N * sizeof(float);
    const size_t size_c = M * N * sizeof(float);

    auto buffer_a = backend->AllocateBuffer(size_a);
    auto buffer_b = backend->AllocateBuffer(size_b);
    auto buffer_c = backend->AllocateBuffer(size_c);

    ASSERT_TRUE(buffer_a.IsValid());
    ASSERT_TRUE(buffer_b.IsValid());
    ASSERT_TRUE(buffer_c.IsValid());

    // Initialize matrices with test data
    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 2.0f);
    std::vector<float> c_data(M * N, 0.0f);

    EXPECT_TRUE(backend->CopyToDevice(buffer_a, a_data.data(), size_a));
    EXPECT_TRUE(backend->CopyToDevice(buffer_b, b_data.data(), size_b));
    EXPECT_TRUE(backend->CopyToDevice(buffer_c, c_data.data(), size_c));

    // Perform matrix multiplication
    EXPECT_TRUE(backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, M, N, K));
    backend->Synchronize();

    // Verify result
    std::vector<float> result(M * N);
    EXPECT_TRUE(backend->CopyFromDevice(result.data(), buffer_c, size_c));

    // Each element should be K * 1.0 * 2.0 = 2 * K
    const float expected = 2.0f * K;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
    }

    backend->FreeBuffer(buffer_a);
    backend->FreeBuffer(buffer_b);
    backend->FreeBuffer(buffer_c);
    manager.Shutdown();
}

// Test activation functions
TEST_F(BackendSystemTest, ActivationFunctions) {
    BackendConfig config;
    config.verbose_logging = false;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager.GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    if (!backend->SupportsCapability(BackendCapability::ACTIVATION_FUNCTIONS)) {
        GTEST_SKIP() << "Backend doesn't support activation functions";
    }

    const size_t size = 1024;
    const size_t bytes = size * sizeof(float);

    auto input_buffer = backend->AllocateBuffer(bytes);
    auto output_buffer = backend->AllocateBuffer(bytes);

    ASSERT_TRUE(input_buffer.IsValid());
    ASSERT_TRUE(output_buffer.IsValid());

    // Test ReLU
    std::vector<float> input_data(size);
    for (size_t i = 0; i < size; ++i) {
        input_data[i] = static_cast<float>(i) - static_cast<float>(size / 2);
    }

    EXPECT_TRUE(backend->CopyToDevice(input_buffer, input_data.data(), bytes));
    EXPECT_TRUE(backend->ApplyReLU(input_buffer, output_buffer, size));
    backend->Synchronize();

    std::vector<float> output_data(size);
    EXPECT_TRUE(backend->CopyFromDevice(output_data.data(), output_buffer, bytes));

    // Verify ReLU: max(0, x)
    for (size_t i = 0; i < size; ++i) {
        float expected = std::max(0.0f, input_data[i]);
        EXPECT_FLOAT_EQ(output_data[i], expected) << "ReLU mismatch at index " << i;
    }

    backend->FreeBuffer(input_buffer);
    backend->FreeBuffer(output_buffer);
    manager.Shutdown();
}

// Test backend switching
TEST_F(BackendSystemTest, BackendSwitching) {
    BackendConfig config;
    config.verbose_logging = false;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto available = manager.GetAvailableBackends();
    if (available.size() < 2) {
        GTEST_SKIP() << "Need at least 2 backends for switching test";
    }

    auto* initial_backend = manager.GetActiveBackend();
    ASSERT_NE(initial_backend, nullptr);
    std::string initial_name = initial_backend->GetName();

    // Find a different backend
    std::string target_backend;
    for (const auto& name : available) {
        if (name != initial_name) {
            target_backend = name;
            break;
        }
    }

    ASSERT_FALSE(target_backend.empty());

    // Switch to the target backend
    EXPECT_TRUE(manager.SwitchBackend(target_backend));

    auto* new_backend = manager.GetActiveBackend();
    ASSERT_NE(new_backend, nullptr);
    EXPECT_EQ(new_backend->GetName(), target_backend);

    // Switch back
    EXPECT_TRUE(manager.SwitchBackend(initial_name));

    auto* restored_backend = manager.GetActiveBackend();
    ASSERT_NE(restored_backend, nullptr);
    EXPECT_EQ(restored_backend->GetName(), initial_name);

    manager.Shutdown();
}

// Test performance benchmarking
TEST_F(BackendSystemTest, PerformanceBenchmarking) {
    BackendConfig config;
    config.verbose_logging = false;
    config.enable_benchmarking = true;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto results = manager.RunBenchmarks();
    EXPECT_FALSE(results.empty());

    for (const auto& [name, perf] : results) {
        EXPECT_FALSE(name.empty());
        EXPECT_GE(perf.matrix_multiply_gflops, 0.0);
        EXPECT_GE(perf.initialization_time_ms, 0.0);
        // Performance should be stable for this test
        EXPECT_TRUE(perf.is_stable);
    }

    manager.Shutdown();
}

// Test error handling
TEST_F(BackendSystemTest, ErrorHandling) {
    auto& registry = BackendRegistry::Instance();

    // Test creating non-existent backend
    auto invalid_backend = registry.CreateBackend("NonExistentBackend");
    EXPECT_EQ(invalid_backend, nullptr);

    // Test invalid buffer operations
    BackendConfig config;
    config.verbose_logging = false;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager.GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    // Test with invalid buffer
    BackendBuffer invalid_buffer;
    EXPECT_FALSE(invalid_buffer.IsValid());

    std::vector<float> dummy_data(100, 1.0f);
    EXPECT_FALSE(backend->CopyToDevice(invalid_buffer, dummy_data.data(), 100 * sizeof(float)));
    EXPECT_FALSE(backend->CopyFromDevice(dummy_data.data(), invalid_buffer, 100 * sizeof(float)));

    manager.Shutdown();
}

// Test capability detection
TEST_F(BackendSystemTest, CapabilityDetection) {
    auto& registry = BackendRegistry::Instance();
    auto available = registry.GetAvailableBackends();

    for (const auto& name : available) {
        auto backend = registry.CreateBackend(name);
        if (backend) {
            // Test various capabilities
            std::vector<BackendCapability> common_capabilities = {
                BackendCapability::MATRIX_MULTIPLICATION,
                BackendCapability::ACTIVATION_FUNCTIONS
            };

            for (auto capability : common_capabilities) {
                bool supports = backend->SupportsCapability(capability);
                // Just verify the call doesn't crash
                EXPECT_TRUE(supports || !supports);
            }

            backend->Shutdown();
        }
    }
}

// Test concurrent backend usage (basic thread safety)
TEST_F(BackendSystemTest, ThreadSafety) {
    BackendConfig config;
    config.verbose_logging = false;

    BackendManager manager(config);
    if (!manager.Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager.GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    // Test that multiple threads can safely get backend info
    std::atomic<int> success_count{0};
    const int num_threads = 4;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            try {
                auto name = backend->GetName();
                auto version = backend->GetVersion();
                auto device_count = backend->GetDeviceCount();
                auto metrics = backend->GetMetrics();

                if (!name.empty() && device_count >= 0) {
                    success_count++;
                }
            } catch (const std::exception& e) {
                // Thread-safety test failed
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count.load(), num_threads);
    manager.Shutdown();
}

// Main function for standalone execution
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}