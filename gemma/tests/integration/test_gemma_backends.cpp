/**
 * @file test_gemma_backends.cpp
 * @brief Integration tests for Gemma.cpp with hardware acceleration backends
 */

#include <gtest/gtest.h>
#include "backends/backend_manager.h"
#include "backends/backend_registry.h"
#include <memory>
#include <vector>
#include <string>

using namespace gemma::backends;

class GemmaBackendIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize backend system
        BackendManager::RegisterAllBackends();

        config_.preferred_backend = "auto";
        config_.enable_fallback = true;
        config_.verbose_logging = false;
        config_.enable_benchmarking = false;
    }

    void TearDown() override {
        if (manager_) {
            manager_->Shutdown();
        }
    }

    BackendConfig config_;
    std::unique_ptr<BackendManager> manager_;
};

// Test basic backend initialization and availability
TEST_F(GemmaBackendIntegrationTest, BackendSystemInitialization) {
    manager_ = std::make_unique<BackendManager>(config_);

    auto available = manager_->GetAvailableBackends();

    if (available.empty()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    EXPECT_TRUE(manager_->Initialize());

    auto* backend = manager_->GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    EXPECT_TRUE(backend->IsAvailable());
    EXPECT_TRUE(backend->IsInitialized());
    EXPECT_FALSE(backend->GetName().empty());
    EXPECT_FALSE(backend->GetVersion().empty());
}

// Test matrix operations that would be used by Gemma
TEST_F(GemmaBackendIntegrationTest, GemmaMatrixOperations) {
    manager_ = std::make_unique<BackendManager>(config_);

    if (!manager_->Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager_->GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    if (!backend->SupportsCapability(BackendCapability::MATRIX_MULTIPLICATION)) {
        GTEST_SKIP() << "Backend doesn't support matrix multiplication";
    }

    // Test typical Gemma model sizes
    std::vector<std::tuple<int, int, int>> gemma_sizes = {
        {1, 2048, 256},     // Sequence length 1, hidden size 2048, output 256
        {32, 2048, 2048},   // Batch 32, hidden to hidden
        {1, 256, 2048},     // Small batch, feature projection
        {8, 2048, 8192},    // Feed-forward layer
    };

    for (const auto& [batch, hidden, output] : gemma_sizes) {
        // Allocate buffers for typical Gemma operation
        size_t input_size = batch * hidden * sizeof(float);
        size_t weight_size = hidden * output * sizeof(float);
        size_t output_size = batch * output * sizeof(float);

        auto input_buffer = backend->AllocateBuffer(input_size);
        auto weight_buffer = backend->AllocateBuffer(weight_size);
        auto output_buffer = backend->AllocateBuffer(output_size);

        ASSERT_TRUE(input_buffer.IsValid());
        ASSERT_TRUE(weight_buffer.IsValid());
        ASSERT_TRUE(output_buffer.IsValid());

        // Initialize with test data
        std::vector<float> input_data(batch * hidden, 0.1f);
        std::vector<float> weight_data(hidden * output, 0.05f);

        EXPECT_TRUE(backend->CopyToDevice(input_buffer, input_data.data(), input_size));
        EXPECT_TRUE(backend->CopyToDevice(weight_buffer, weight_data.data(), weight_size));

        // Perform matrix multiplication (input * weight = output)
        EXPECT_TRUE(backend->MatrixMultiply(input_buffer, weight_buffer, output_buffer,
                                          batch, output, hidden));
        backend->Synchronize();

        // Verify result
        std::vector<float> result(batch * output);
        EXPECT_TRUE(backend->CopyFromDevice(result.data(), output_buffer, output_size));

        // Check that computation produced reasonable results
        float expected = 0.1f * 0.05f * hidden; // Approximate expected value
        for (int i = 0; i < batch * output; ++i) {
            EXPECT_NEAR(result[i], expected, expected * 0.1f)
                << "Result mismatch at index " << i
                << " for size (" << batch << "x" << hidden << "x" << output << ")";
        }

        backend->FreeBuffer(input_buffer);
        backend->FreeBuffer(weight_buffer);
        backend->FreeBuffer(output_buffer);
    }
}

// Test activation functions used by Gemma
TEST_F(GemmaBackendIntegrationTest, GemmaActivationFunctions) {
    manager_ = std::make_unique<BackendManager>(config_);

    if (!manager_->Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager_->GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    if (!backend->SupportsCapability(BackendCapability::ACTIVATION_FUNCTIONS)) {
        GTEST_SKIP() << "Backend doesn't support activation functions";
    }

    // Test typical activation tensor sizes in Gemma
    std::vector<size_t> activation_sizes = {
        2048,      // Hidden size
        8192,      // Feed-forward intermediate
        32768,     // Large activation tensor
        65536      // Very large tensor
    };

    for (size_t size : activation_sizes) {
        size_t bytes = size * sizeof(float);
        auto input_buffer = backend->AllocateBuffer(bytes);
        auto output_buffer = backend->AllocateBuffer(bytes);

        ASSERT_TRUE(input_buffer.IsValid());
        ASSERT_TRUE(output_buffer.IsValid());

        // Test ReLU activation
        std::vector<float> input_data(size);
        for (size_t i = 0; i < size; ++i) {
            input_data[i] = static_cast<float>(i % 100) - 50.0f; // Mix of positive and negative
        }

        EXPECT_TRUE(backend->CopyToDevice(input_buffer, input_data.data(), bytes));
        EXPECT_TRUE(backend->ApplyReLU(input_buffer, output_buffer, size));
        backend->Synchronize();

        std::vector<float> relu_result(size);
        EXPECT_TRUE(backend->CopyFromDevice(relu_result.data(), output_buffer, bytes));

        // Verify ReLU: max(0, x)
        for (size_t i = 0; i < size; ++i) {
            float expected = std::max(0.0f, input_data[i]);
            EXPECT_FLOAT_EQ(relu_result[i], expected)
                << "ReLU mismatch at index " << i << " for size " << size;
        }

        // Test GELU activation (commonly used in transformers)
        EXPECT_TRUE(backend->CopyToDevice(input_buffer, input_data.data(), bytes));
        EXPECT_TRUE(backend->ApplyGELU(input_buffer, output_buffer, size));
        backend->Synchronize();

        std::vector<float> gelu_result(size);
        EXPECT_TRUE(backend->CopyFromDevice(gelu_result.data(), output_buffer, bytes));

        // Verify GELU produces reasonable values (should be smooth activation)
        for (size_t i = 0; i < size; ++i) {
            // GELU should produce values in a reasonable range
            EXPECT_TRUE(std::isfinite(gelu_result[i]))
                << "GELU produced non-finite value at index " << i;

            // For positive inputs, GELU should be close to input
            if (input_data[i] > 2.0f) {
                EXPECT_NEAR(gelu_result[i], input_data[i], input_data[i] * 0.1f);
            }

            // For negative inputs < -3, GELU should be close to 0
            if (input_data[i] < -3.0f) {
                EXPECT_NEAR(gelu_result[i], 0.0f, 0.1f);
            }
        }

        backend->FreeBuffer(input_buffer);
        backend->FreeBuffer(output_buffer);
    }
}

// Test attention computation if supported
TEST_F(GemmaBackendIntegrationTest, GemmaAttentionComputation) {
    manager_ = std::make_unique<BackendManager>(config_);

    if (!manager_->Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager_->GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    if (!backend->SupportsCapability(BackendCapability::ATTENTION_COMPUTATION)) {
        GTEST_SKIP() << "Backend doesn't support attention computation";
    }

    // Test typical Gemma attention parameters
    const int batch_size = 1;
    const int seq_len = 128;
    const int head_dim = 64;
    const int num_heads = 32;  // Typical for Gemma-2B

    size_t qkv_size = batch_size * seq_len * head_dim * num_heads * sizeof(float);
    size_t output_size = batch_size * seq_len * head_dim * num_heads * sizeof(float);

    auto queries_buffer = backend->AllocateBuffer(qkv_size);
    auto keys_buffer = backend->AllocateBuffer(qkv_size);
    auto values_buffer = backend->AllocateBuffer(qkv_size);
    auto output_buffer = backend->AllocateBuffer(output_size);

    ASSERT_TRUE(queries_buffer.IsValid());
    ASSERT_TRUE(keys_buffer.IsValid());
    ASSERT_TRUE(values_buffer.IsValid());
    ASSERT_TRUE(output_buffer.IsValid());

    // Initialize with test data
    size_t total_elements = batch_size * seq_len * head_dim * num_heads;
    std::vector<float> q_data(total_elements, 0.1f);
    std::vector<float> k_data(total_elements, 0.1f);
    std::vector<float> v_data(total_elements, 0.2f);

    EXPECT_TRUE(backend->CopyToDevice(queries_buffer, q_data.data(), qkv_size));
    EXPECT_TRUE(backend->CopyToDevice(keys_buffer, k_data.data(), qkv_size));
    EXPECT_TRUE(backend->CopyToDevice(values_buffer, v_data.data(), qkv_size));

    // Compute attention
    EXPECT_TRUE(backend->ComputeAttention(queries_buffer, keys_buffer, values_buffer,
                                        output_buffer, batch_size, seq_len, head_dim, num_heads));
    backend->Synchronize();

    // Verify result
    std::vector<float> attention_result(total_elements);
    EXPECT_TRUE(backend->CopyFromDevice(attention_result.data(), output_buffer, output_size));

    // Check that attention produced reasonable results
    for (size_t i = 0; i < total_elements; ++i) {
        EXPECT_TRUE(std::isfinite(attention_result[i]))
            << "Attention produced non-finite value at index " << i;
        EXPECT_GE(attention_result[i], -10.0f) << "Attention value too negative at index " << i;
        EXPECT_LE(attention_result[i], 10.0f) << "Attention value too positive at index " << i;
    }

    backend->FreeBuffer(queries_buffer);
    backend->FreeBuffer(keys_buffer);
    backend->FreeBuffer(values_buffer);
    backend->FreeBuffer(output_buffer);
}

// Test performance with Gemma-like workloads
TEST_F(GemmaBackendIntegrationTest, GemmaPerformanceBaseline) {
    manager_ = std::make_unique<BackendManager>(config_);

    if (!manager_->Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager_->GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    if (!backend->SupportsCapability(BackendCapability::MATRIX_MULTIPLICATION)) {
        GTEST_SKIP() << "Backend doesn't support matrix multiplication";
    }

    // Simulate Gemma inference workload
    const int seq_len = 1;
    const int hidden_size = 2048;
    const int intermediate_size = 8192;

    // Typical operations in a transformer layer:
    // 1. Query/Key/Value projections
    // 2. Feed-forward layers
    // 3. Output projections

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int layer = 0; layer < 5; ++layer) {  // Test 5 layers
        // QKV projection: hidden -> 3 * hidden
        {
            size_t input_size = seq_len * hidden_size * sizeof(float);
            size_t weight_size = hidden_size * (3 * hidden_size) * sizeof(float);
            size_t output_size = seq_len * (3 * hidden_size) * sizeof(float);

            auto input_buf = backend->AllocateBuffer(input_size);
            auto weight_buf = backend->AllocateBuffer(weight_size);
            auto output_buf = backend->AllocateBuffer(output_size);

            ASSERT_TRUE(input_buf.IsValid() && weight_buf.IsValid() && output_buf.IsValid());

            EXPECT_TRUE(backend->MatrixMultiply(input_buf, weight_buf, output_buf,
                                              seq_len, 3 * hidden_size, hidden_size));

            backend->FreeBuffer(input_buf);
            backend->FreeBuffer(weight_buf);
            backend->FreeBuffer(output_buf);
        }

        // Feed-forward layer 1: hidden -> intermediate
        {
            size_t input_size = seq_len * hidden_size * sizeof(float);
            size_t weight_size = hidden_size * intermediate_size * sizeof(float);
            size_t output_size = seq_len * intermediate_size * sizeof(float);

            auto input_buf = backend->AllocateBuffer(input_size);
            auto weight_buf = backend->AllocateBuffer(weight_size);
            auto output_buf = backend->AllocateBuffer(output_size);

            ASSERT_TRUE(input_buf.IsValid() && weight_buf.IsValid() && output_buf.IsValid());

            EXPECT_TRUE(backend->MatrixMultiply(input_buf, weight_buf, output_buf,
                                              seq_len, intermediate_size, hidden_size));

            // Apply GELU activation
            if (backend->SupportsCapability(BackendCapability::ACTIVATION_FUNCTIONS)) {
                EXPECT_TRUE(backend->ApplyGELU(output_buf, output_buf, seq_len * intermediate_size));
            }

            backend->FreeBuffer(input_buf);
            backend->FreeBuffer(weight_buf);
            backend->FreeBuffer(output_buf);
        }

        // Feed-forward layer 2: intermediate -> hidden
        {
            size_t input_size = seq_len * intermediate_size * sizeof(float);
            size_t weight_size = intermediate_size * hidden_size * sizeof(float);
            size_t output_size = seq_len * hidden_size * sizeof(float);

            auto input_buf = backend->AllocateBuffer(input_size);
            auto weight_buf = backend->AllocateBuffer(weight_size);
            auto output_buf = backend->AllocateBuffer(output_size);

            ASSERT_TRUE(input_buf.IsValid() && weight_buf.IsValid() && output_buf.IsValid());

            EXPECT_TRUE(backend->MatrixMultiply(input_buf, weight_buf, output_buf,
                                              seq_len, hidden_size, intermediate_size));

            backend->FreeBuffer(input_buf);
            backend->FreeBuffer(weight_buf);
            backend->FreeBuffer(output_buf);
        }
    }

    backend->Synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Performance should be reasonable (less than 1 second for this workload)
    EXPECT_LT(elapsed_ms, 1000.0) << "Backend performance is too slow for Gemma workload";

    std::cout << "Backend " << backend->GetName()
              << " completed Gemma-like workload in " << elapsed_ms << " ms" << std::endl;
}

// Test memory efficiency for large models
TEST_F(GemmaBackendIntegrationTest, GemmaMemoryEfficiency) {
    manager_ = std::make_unique<BackendManager>(config_);

    if (!manager_->Initialize()) {
        GTEST_SKIP() << "No backends available for testing";
    }

    auto* backend = manager_->GetActiveBackend();
    ASSERT_NE(backend, nullptr);

    // Test allocating memory for a model similar to Gemma-2B
    std::vector<BackendBuffer> model_buffers;

    try {
        // Typical Gemma-2B parameter sizes (simplified)
        std::vector<size_t> layer_sizes = {
            2048 * 2048 * sizeof(float),  // Self-attention weights
            2048 * 8192 * sizeof(float),  // Feed-forward layer 1
            8192 * 2048 * sizeof(float),  // Feed-forward layer 2
            2048 * sizeof(float),         // Layer norm
        };

        const int num_layers = 18;  // Simplified Gemma-2B has 18 layers

        for (int layer = 0; layer < num_layers; ++layer) {
            for (size_t layer_size : layer_sizes) {
                auto buffer = backend->AllocateBuffer(layer_size);
                if (buffer.IsValid()) {
                    model_buffers.push_back(std::move(buffer));
                } else {
                    // Memory exhausted - this is expected for large models
                    break;
                }
            }
        }

        EXPECT_GT(model_buffers.size(), 10)
            << "Backend should be able to allocate at least some model layers";

        // Test that we can still perform operations with allocated memory
        if (model_buffers.size() >= 3) {
            const int test_size = 512;
            const size_t test_bytes = test_size * test_size * sizeof(float);

            auto test_a = backend->AllocateBuffer(test_bytes);
            auto test_b = backend->AllocateBuffer(test_bytes);
            auto test_c = backend->AllocateBuffer(test_bytes);

            if (test_a.IsValid() && test_b.IsValid() && test_c.IsValid()) {
                EXPECT_TRUE(backend->MatrixMultiply(test_a, test_b, test_c,
                                                  test_size, test_size, test_size));
                backend->Synchronize();

                backend->FreeBuffer(test_a);
                backend->FreeBuffer(test_b);
                backend->FreeBuffer(test_c);
            }
        }

    } catch (const std::exception& e) {
        // Memory allocation may fail for large models - this is acceptable
        std::cout << "Memory allocation test completed with exception: " << e.what() << std::endl;
    }

    // Cleanup
    for (auto& buffer : model_buffers) {
        backend->FreeBuffer(buffer);
    }

    // Verify backend metrics
    auto metrics = backend->GetMetrics();
    EXPECT_GE(metrics.memory_usage_bytes, 0);

    std::cout << "Backend " << backend->GetName()
              << " allocated " << model_buffers.size() << " layer buffers, "
              << "memory usage: " << (metrics.memory_usage_bytes / 1024 / 1024) << " MB" << std::endl;
}

// Main function for standalone execution
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}