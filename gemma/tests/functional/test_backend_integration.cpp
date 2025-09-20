/**
 * @file test_backend_integration.cpp
 * @brief Functional integration tests for hardware backend registration and management
 *
 * This test suite verifies:
 * - Backend registration and discovery
 * - Fallback mechanisms between backends
 * - Memory transfer operations
 * - Basic performance benchmarks
 * - Error handling and recovery
 */

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>
#include <random>
#include <thread>

// Include backend interface and registry
#include "../../backends/backend_interface.h"
#include "../../backends/backend_registry.h"

// Mock backend implementations for testing
namespace gemma {
namespace backends {
namespace test {

/**
 * @brief Mock CPU backend for testing fallback mechanisms
 */
class MockCPUBackend : public BackendInterface {
private:
    bool initialized_ = false;
    BackendMetrics metrics_;
    std::vector<std::unique_ptr<float[]>> allocated_buffers_;

public:
    std::string GetName() const override { return "MockCPU"; }
    std::string GetVersion() const override { return "1.0.0-test"; }

    bool Initialize() override {
        initialized_ = true;
        return true;
    }

    void Shutdown() override {
        initialized_ = false;
        allocated_buffers_.clear();
    }

    bool IsAvailable() const override { return true; }

    bool SupportsCapability(BackendCapability capability) const override {
        // Mock CPU supports all basic operations
        return capability != BackendCapability::ASYNC_EXECUTION;
    }

    int GetDeviceCount() const override { return 1; }
    bool SetDevice(int device_id) override { return device_id == 0; }
    int GetCurrentDevice() const override { return 0; }

    BackendBuffer AllocateBuffer(size_t size, size_t alignment = 32) override {
        auto buffer = std::make_unique<float[]>(size / sizeof(float));
        float* ptr = buffer.get();
        allocated_buffers_.push_back(std::move(buffer));

        return BackendBuffer(ptr, size, false);
    }

    void FreeBuffer(const BackendBuffer& buffer) override {
        // In real implementation, would free the specific buffer
        // For testing, we'll just clear all on shutdown
    }

    bool CopyToDevice(const BackendBuffer& dst, const void* src, size_t size) override {
        if (!dst.data || !src) return false;
        std::memcpy(dst.data, src, size);
        return true;
    }

    bool CopyFromDevice(void* dst, const BackendBuffer& src, size_t size) override {
        if (!dst || !src.data) return false;
        std::memcpy(dst, src.data, size);
        return true;
    }

    void Synchronize() override {
        // CPU operations are synchronous
    }

    BackendMetrics GetMetrics() const override {
        return metrics_;
    }

    void ResetMetrics() override {
        metrics_ = BackendMetrics{};
    }

    bool MatrixMultiply(const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
                       int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) override {
        if (!a.data || !b.data || !c.data) return false;

        // Simple CPU matrix multiplication for testing
        float* A = static_cast<float*>(a.data);
        float* B = static_cast<float*>(b.data);
        float* C = static_cast<float*>(c.data);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] = alpha * sum + beta * C[i * n + j];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        metrics_.latency_ms = duration.count() / 1000.0;

        return true;
    }

    bool MatrixVectorMultiply(const BackendBuffer& a, const BackendBuffer& x, const BackendBuffer& y,
                             int m, int n) override {
        if (!a.data || !x.data || !y.data) return false;

        float* A = static_cast<float*>(a.data);
        float* X = static_cast<float*>(x.data);
        float* Y = static_cast<float*>(y.data);

        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += A[i * n + j] * X[j];
            }
            Y[i] = sum;
        }

        return true;
    }

    bool ComputeAttention(const BackendBuffer& queries, const BackendBuffer& keys,
                         const BackendBuffer& values, const BackendBuffer& output,
                         int batch_size, int seq_len, int head_dim, int num_heads) override {
        // Simplified attention computation for testing
        return queries.data && keys.data && values.data && output.data;
    }

    bool ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override {
        if (!input.data || !output.data) return false;

        float* in = static_cast<float*>(input.data);
        float* out = static_cast<float*>(output.data);

        for (size_t i = 0; i < size / sizeof(float); ++i) {
            out[i] = std::max(0.0f, in[i]);
        }

        return true;
    }

    bool ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) override {
        if (!input.data || !output.data) return false;

        float* in = static_cast<float*>(input.data);
        float* out = static_cast<float*>(output.data);

        for (size_t i = 0; i < size / sizeof(float); ++i) {
            float x = in[i];
            out[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        }

        return true;
    }

    bool ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) override {
        if (!input.data || !output.data) return false;

        float* in = static_cast<float*>(input.data);
        float* out = static_cast<float*>(output.data);
        size_t elements = size / sizeof(float);

        // Find maximum for numerical stability
        float max_val = in[0];
        for (size_t i = 1; i < elements; ++i) {
            max_val = std::max(max_val, in[i]);
        }

        // Compute exponentials and sum
        float sum = 0.0f;
        for (size_t i = 0; i < elements; ++i) {
            out[i] = std::exp(in[i] - max_val);
            sum += out[i];
        }

        // Normalize
        for (size_t i = 0; i < elements; ++i) {
            out[i] /= sum;
        }

        return true;
    }
};

/**
 * @brief Mock GPU backend that can fail to simulate fallback scenarios
 */
class MockGPUBackend : public MockCPUBackend {
private:
    bool should_fail_init_;
    bool should_fail_operations_;

public:
    MockGPUBackend(bool fail_init = false, bool fail_ops = false)
        : should_fail_init_(fail_init), should_fail_operations_(fail_ops) {}

    std::string GetName() const override { return "MockGPU"; }
    std::string GetVersion() const override { return "1.0.0-test-gpu"; }

    bool Initialize() override {
        if (should_fail_init_) return false;
        return MockCPUBackend::Initialize();
    }

    bool IsAvailable() const override {
        return !should_fail_init_;
    }

    bool SupportsCapability(BackendCapability capability) const override {
        return true; // GPU supports all capabilities
    }

    bool MatrixMultiply(const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
                       int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) override {
        if (should_fail_operations_) return false;

        // Simulate faster GPU computation
        auto result = MockCPUBackend::MatrixMultiply(a, b, c, m, n, k, alpha, beta);

        // Simulate 10x speedup
        auto metrics = GetMetrics();
        const_cast<BackendMetrics&>(metrics).latency_ms /= 10.0;

        return result;
    }
};

} // namespace test
} // namespace backends
} // namespace gemma

/**
 * @brief Test fixture for backend integration tests
 */
class BackendIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_unique<gemma::backends::BackendRegistry>();

        // Register mock backends
        registry_->RegisterBackend("MockCPU", []() {
            return std::make_unique<gemma::backends::test::MockCPUBackend>();
        });

        registry_->RegisterBackend("MockGPU", []() {
            return std::make_unique<gemma::backends::test::MockGPUBackend>();
        });

        registry_->RegisterBackend("MockGPU_Failing", []() {
            return std::make_unique<gemma::backends::test::MockGPUBackend>(true, false);
        });
    }

    void TearDown() override {
        registry_.reset();
    }

    std::unique_ptr<gemma::backends::BackendRegistry> registry_;
};

/**
 * @brief Test backend registration and discovery
 */
TEST_F(BackendIntegrationTest, BackendRegistration) {
    // Test getting available backends
    auto backends = registry_->GetAvailableBackends();
    EXPECT_GE(backends.size(), 2);

    bool found_cpu = false, found_gpu = false;
    for (const auto& name : backends) {
        if (name == "MockCPU") found_cpu = true;
        if (name == "MockGPU") found_gpu = true;
    }
    EXPECT_TRUE(found_cpu);
    EXPECT_TRUE(found_gpu);
}

/**
 * @brief Test backend creation and initialization
 */
TEST_F(BackendIntegrationTest, BackendCreation) {
    // Test successful backend creation
    auto cpu_backend = registry_->CreateBackend("MockCPU");
    ASSERT_NE(cpu_backend, nullptr);
    EXPECT_EQ(cpu_backend->GetName(), "MockCPU");
    EXPECT_TRUE(cpu_backend->IsAvailable());
    EXPECT_TRUE(cpu_backend->Initialize());

    // Test failed backend creation
    auto invalid_backend = registry_->CreateBackend("NonExistent");
    EXPECT_EQ(invalid_backend, nullptr);
}

/**
 * @brief Test fallback mechanism when preferred backend fails
 */
TEST_F(BackendIntegrationTest, FallbackMechanism) {
    // Try to get failing GPU backend first, should fallback to CPU
    std::vector<std::string> preference_order = {"MockGPU_Failing", "MockCPU"};

    std::unique_ptr<gemma::backends::BackendInterface> backend;
    for (const auto& backend_name : preference_order) {
        auto candidate = registry_->CreateBackend(backend_name);
        if (candidate && candidate->IsAvailable() && candidate->Initialize()) {
            backend = std::move(candidate);
            break;
        }
    }

    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->GetName(), "MockCPU"); // Should have fallen back to CPU
}

/**
 * @brief Test memory operations between host and device
 */
TEST_F(BackendIntegrationTest, MemoryOperations) {
    auto backend = registry_->CreateBackend("MockCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    // Test buffer allocation
    const size_t buffer_size = 1024 * sizeof(float);
    auto device_buffer = backend->AllocateBuffer(buffer_size);
    EXPECT_NE(device_buffer.data, nullptr);
    EXPECT_EQ(device_buffer.size, buffer_size);

    // Test host-to-device copy
    std::vector<float> host_data(1024);
    std::iota(host_data.begin(), host_data.end(), 1.0f); // Fill with 1, 2, 3, ...

    EXPECT_TRUE(backend->CopyToDevice(device_buffer, host_data.data(), buffer_size));

    // Test device-to-host copy
    std::vector<float> result_data(1024);
    EXPECT_TRUE(backend->CopyFromDevice(result_data.data(), device_buffer, buffer_size));

    // Verify data integrity
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_data[i], result_data[i]);
    }

    backend->FreeBuffer(device_buffer);
    backend->Shutdown();
}

/**
 * @brief Test matrix operations
 */
TEST_F(BackendIntegrationTest, MatrixOperations) {
    auto backend = registry_->CreateBackend("MockCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    // Test matrix multiplication: C = A * B
    const int M = 4, N = 4, K = 4;
    const size_t matrix_size = M * K * sizeof(float);

    // Allocate buffers
    auto buffer_a = backend->AllocateBuffer(matrix_size);
    auto buffer_b = backend->AllocateBuffer(K * N * sizeof(float));
    auto buffer_c = backend->AllocateBuffer(M * N * sizeof(float));

    ASSERT_NE(buffer_a.data, nullptr);
    ASSERT_NE(buffer_b.data, nullptr);
    ASSERT_NE(buffer_c.data, nullptr);

    // Initialize test data
    std::vector<float> A(M * K, 1.0f); // All ones
    std::vector<float> B(K * N, 2.0f); // All twos
    std::vector<float> C(M * N, 0.0f); // Zeros

    // Copy data to device
    backend->CopyToDevice(buffer_a, A.data(), matrix_size);
    backend->CopyToDevice(buffer_b, B.data(), K * N * sizeof(float));
    backend->CopyToDevice(buffer_c, C.data(), M * N * sizeof(float));

    // Perform matrix multiplication
    EXPECT_TRUE(backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, M, N, K));

    // Copy result back
    std::vector<float> result(M * N);
    backend->CopyFromDevice(result.data(), buffer_c, M * N * sizeof(float));

    // Verify result (should be all 8.0 = 4 * 1.0 * 2.0)
    for (float val : result) {
        EXPECT_FLOAT_EQ(val, 8.0f);
    }

    // Cleanup
    backend->FreeBuffer(buffer_a);
    backend->FreeBuffer(buffer_b);
    backend->FreeBuffer(buffer_c);
    backend->Shutdown();
}

/**
 * @brief Test activation functions
 */
TEST_F(BackendIntegrationTest, ActivationFunctions) {
    auto backend = registry_->CreateBackend("MockCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    const size_t num_elements = 8;
    const size_t buffer_size = num_elements * sizeof(float);

    auto input_buffer = backend->AllocateBuffer(buffer_size);
    auto output_buffer = backend->AllocateBuffer(buffer_size);

    ASSERT_NE(input_buffer.data, nullptr);
    ASSERT_NE(output_buffer.data, nullptr);

    // Test ReLU
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -0.5f, 1.5f};
    backend->CopyToDevice(input_buffer, input_data.data(), buffer_size);

    EXPECT_TRUE(backend->ApplyReLU(input_buffer, output_buffer, buffer_size));

    std::vector<float> relu_result(num_elements);
    backend->CopyFromDevice(relu_result.data(), output_buffer, buffer_size);

    std::vector<float> expected_relu = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.5f};
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(relu_result[i], expected_relu[i]);
    }

    // Test Softmax (simple case)
    std::vector<float> softmax_input = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    backend->CopyToDevice(input_buffer, softmax_input.data(), buffer_size);

    EXPECT_TRUE(backend->ApplySoftmax(input_buffer, output_buffer, buffer_size));

    std::vector<float> softmax_result(num_elements);
    backend->CopyFromDevice(softmax_result.data(), output_buffer, buffer_size);

    // Verify softmax properties: all positive, sum to 1
    float sum = 0.0f;
    for (float val : softmax_result) {
        EXPECT_GT(val, 0.0f);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);

    // Cleanup
    backend->FreeBuffer(input_buffer);
    backend->FreeBuffer(output_buffer);
    backend->Shutdown();
}

/**
 * @brief Test performance comparison between backends
 */
TEST_F(BackendIntegrationTest, PerformanceComparison) {
    auto cpu_backend = registry_->CreateBackend("MockCPU");
    auto gpu_backend = registry_->CreateBackend("MockGPU");

    ASSERT_NE(cpu_backend, nullptr);
    ASSERT_NE(gpu_backend, nullptr);
    ASSERT_TRUE(cpu_backend->Initialize());
    ASSERT_TRUE(gpu_backend->Initialize());

    const int M = 64, N = 64, K = 64;
    const size_t matrix_size = M * K * sizeof(float);

    // Setup test data
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    // Time CPU backend
    auto cpu_buffer_a = cpu_backend->AllocateBuffer(matrix_size);
    auto cpu_buffer_b = cpu_backend->AllocateBuffer(K * N * sizeof(float));
    auto cpu_buffer_c = cpu_backend->AllocateBuffer(M * N * sizeof(float));

    cpu_backend->CopyToDevice(cpu_buffer_a, A.data(), matrix_size);
    cpu_backend->CopyToDevice(cpu_buffer_b, B.data(), K * N * sizeof(float));
    cpu_backend->CopyToDevice(cpu_buffer_c, C.data(), M * N * sizeof(float));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_backend->MatrixMultiply(cpu_buffer_a, cpu_buffer_b, cpu_buffer_c, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_metrics = cpu_backend->GetMetrics();

    // Time GPU backend
    auto gpu_buffer_a = gpu_backend->AllocateBuffer(matrix_size);
    auto gpu_buffer_b = gpu_backend->AllocateBuffer(K * N * sizeof(float));
    auto gpu_buffer_c = gpu_backend->AllocateBuffer(M * N * sizeof(float));

    gpu_backend->CopyToDevice(gpu_buffer_a, A.data(), matrix_size);
    gpu_backend->CopyToDevice(gpu_buffer_b, B.data(), K * N * sizeof(float));
    gpu_backend->CopyToDevice(gpu_buffer_c, C.data(), M * N * sizeof(float));

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_backend->MatrixMultiply(gpu_buffer_a, gpu_buffer_b, gpu_buffer_c, M, N, K);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    auto gpu_metrics = gpu_backend->GetMetrics();

    // GPU should be faster (simulated 10x speedup)
    EXPECT_LT(gpu_metrics.latency_ms, cpu_metrics.latency_ms);

    std::cout << "CPU latency: " << cpu_metrics.latency_ms << "ms" << std::endl;
    std::cout << "GPU latency: " << gpu_metrics.latency_ms << "ms" << std::endl;

    // Cleanup
    cpu_backend->FreeBuffer(cpu_buffer_a);
    cpu_backend->FreeBuffer(cpu_buffer_b);
    cpu_backend->FreeBuffer(cpu_buffer_c);
    gpu_backend->FreeBuffer(gpu_buffer_a);
    gpu_backend->FreeBuffer(gpu_buffer_b);
    gpu_backend->FreeBuffer(gpu_buffer_c);

    cpu_backend->Shutdown();
    gpu_backend->Shutdown();
}

/**
 * @brief Test error handling and recovery
 */
TEST_F(BackendIntegrationTest, ErrorHandling) {
    auto backend = registry_->CreateBackend("MockCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    // Test invalid buffer operations
    BackendBuffer invalid_buffer{nullptr, 0, false};
    std::vector<float> data(10, 1.0f);

    EXPECT_FALSE(backend->CopyToDevice(invalid_buffer, data.data(), data.size() * sizeof(float)));
    EXPECT_FALSE(backend->CopyFromDevice(data.data(), invalid_buffer, data.size() * sizeof(float)));

    // Test invalid matrix operations
    auto valid_buffer = backend->AllocateBuffer(64 * sizeof(float));
    EXPECT_FALSE(backend->MatrixMultiply(invalid_buffer, valid_buffer, valid_buffer, 8, 8, 8));
    EXPECT_FALSE(backend->MatrixMultiply(valid_buffer, invalid_buffer, valid_buffer, 8, 8, 8));
    EXPECT_FALSE(backend->MatrixMultiply(valid_buffer, valid_buffer, invalid_buffer, 8, 8, 8));

    // Test invalid activation operations
    EXPECT_FALSE(backend->ApplyReLU(invalid_buffer, valid_buffer, 64));
    EXPECT_FALSE(backend->ApplyReLU(valid_buffer, invalid_buffer, 64));

    backend->FreeBuffer(valid_buffer);
    backend->Shutdown();
}

/**
 * @brief Test concurrent backend operations
 */
TEST_F(BackendIntegrationTest, ConcurrentOperations) {
    const int num_threads = 4;
    const int operations_per_thread = 10;
    std::vector<std::thread> threads;
    std::vector<bool> thread_results(num_threads, false);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, operations_per_thread, &thread_results]() {
            auto backend = registry_->CreateBackend("MockCPU");
            if (!backend || !backend->Initialize()) {
                return;
            }

            bool success = true;
            for (int op = 0; op < operations_per_thread; ++op) {
                auto buffer = backend->AllocateBuffer(256);
                if (!buffer.data) {
                    success = false;
                    break;
                }

                std::vector<float> data(64, float(t * operations_per_thread + op));
                if (!backend->CopyToDevice(buffer, data.data(), data.size() * sizeof(float))) {
                    success = false;
                    backend->FreeBuffer(buffer);
                    break;
                }

                backend->FreeBuffer(buffer);
            }

            backend->Shutdown();
            thread_results[t] = success;
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all threads succeeded
    for (bool result : thread_results) {
        EXPECT_TRUE(result);
    }
}

/**
 * @brief Main function to run all tests
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "Running Gemma.cpp Backend Integration Tests..." << std::endl;
    std::cout << "=============================================" << std::endl;

    return RUN_ALL_TESTS();
}