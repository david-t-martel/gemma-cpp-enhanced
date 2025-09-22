/**
 * @file test_inference_backends.cpp
 * @brief End-to-end inference tests comparing results across different backends
 *
 * This test suite verifies:
 * - Model loading on different backends
 * - Inference result consistency across backends
 * - Performance comparison between backends
 * - Numerical accuracy and stability
 * - Resource usage and memory management
 */

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>

// Include backend interface and test utilities
#include "../../backends/backend_interface.h"
#include "../../backends/backend_registry.h"

namespace gemma {
namespace backends {
namespace test {

/**
 * @brief Simple tensor structure for testing
 */
struct TestTensor {
    std::vector<float> data;
    std::vector<int> shape;

    TestTensor() = default;
    TestTensor(const std::vector<int>& s) : shape(s) {
        int total_size = 1;
        for (int dim : shape) total_size *= dim;
        data.resize(total_size);
    }

    size_t GetSizeBytes() const {
        return data.size() * sizeof(float);
    }

    void FillRandom(float min_val = -1.0f, float max_val = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);

        for (auto& val : data) {
            val = dist(gen);
        }
    }

    void FillSequential(float start = 0.0f, float step = 1.0f) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = start + i * step;
        }
    }
};

/**
 * @brief Test model configuration
 */
struct ModelConfig {
    int vocab_size = 32000;
    int hidden_size = 256;
    int intermediate_size = 512;
    int num_layers = 4;
    int num_attention_heads = 8;
    int max_seq_length = 128;
    float layer_norm_eps = 1e-6f;
};

/**
 * @brief Simplified model layer for testing
 */
class TestModelLayer {
private:
    BackendInterface* backend_;
    ModelConfig config_;

    // Model weights (simplified)
    BackendBuffer attention_weights_;
    BackendBuffer ffn_weights_;
    BackendBuffer layer_norm_weights_;

public:
    TestModelLayer(BackendInterface* backend, const ModelConfig& config)
        : backend_(backend), config_(config) {
        InitializeWeights();
    }

    ~TestModelLayer() {
        CleanupWeights();
    }

    void InitializeWeights() {
        // Allocate weight buffers (simplified)
        size_t attention_size = config_.hidden_size * config_.hidden_size * sizeof(float);
        size_t ffn_size = config_.hidden_size * config_.intermediate_size * sizeof(float);
        size_t norm_size = config_.hidden_size * sizeof(float);

        attention_weights_ = backend_->AllocateBuffer(attention_size);
        ffn_weights_ = backend_->AllocateBuffer(ffn_size);
        layer_norm_weights_ = backend_->AllocateBuffer(norm_size);

        // Initialize with test data
        TestTensor attention_data({config_.hidden_size, config_.hidden_size});
        TestTensor ffn_data({config_.hidden_size, config_.intermediate_size});
        TestTensor norm_data({config_.hidden_size});

        attention_data.FillRandom(-0.1f, 0.1f);
        ffn_data.FillRandom(-0.1f, 0.1f);
        norm_data.FillSequential(1.0f, 0.0f); // Initialize to ones

        backend_->CopyToDevice(attention_weights_, attention_data.data.data(), attention_size);
        backend_->CopyToDevice(ffn_weights_, ffn_data.data.data(), ffn_size);
        backend_->CopyToDevice(layer_norm_weights_, norm_data.data.data(), norm_size);
    }

    void CleanupWeights() {
        if (attention_weights_.data) backend_->FreeBuffer(attention_weights_);
        if (ffn_weights_.data) backend_->FreeBuffer(ffn_weights_);
        if (layer_norm_weights_.data) backend_->FreeBuffer(layer_norm_weights_);
    }

    bool Forward(const BackendBuffer& input, const BackendBuffer& output, int batch_size, int seq_len) {
        // Simplified forward pass: just matrix multiplication
        return backend_->MatrixMultiply(
            input, attention_weights_, output,
            batch_size * seq_len, config_.hidden_size, config_.hidden_size
        );
    }
};

/**
 * @brief Test model for inference comparison
 */
class TestInferenceModel {
private:
    BackendInterface* backend_;
    ModelConfig config_;
    std::vector<std::unique_ptr<TestModelLayer>> layers_;

    BackendBuffer input_buffer_;
    BackendBuffer output_buffer_;
    BackendBuffer intermediate_buffer_;

public:
    TestInferenceModel(BackendInterface* backend, const ModelConfig& config)
        : backend_(backend), config_(config) {
        InitializeModel();
    }

    ~TestInferenceModel() {
        CleanupModel();
    }

    void InitializeModel() {
        // Create layers
        for (int i = 0; i < config_.num_layers; ++i) {
            layers_.push_back(std::make_unique<TestModelLayer>(backend_, config_));
        }

        // Allocate buffers
        size_t buffer_size = config_.max_seq_length * config_.hidden_size * sizeof(float);
        input_buffer_ = backend_->AllocateBuffer(buffer_size);
        output_buffer_ = backend_->AllocateBuffer(buffer_size);
        intermediate_buffer_ = backend_->AllocateBuffer(buffer_size);
    }

    void CleanupModel() {
        layers_.clear();
        if (input_buffer_.data) backend_->FreeBuffer(input_buffer_);
        if (output_buffer_.data) backend_->FreeBuffer(output_buffer_);
        if (intermediate_buffer_.data) backend_->FreeBuffer(intermediate_buffer_);
    }

    bool RunInference(const TestTensor& input, TestTensor& output, int batch_size = 1) {
        if (input.shape.size() != 2 || input.shape[1] != config_.hidden_size) {
            return false;
        }

        int seq_len = input.shape[0];
        if (seq_len > config_.max_seq_length) {
            return false;
        }

        // Copy input to device
        backend_->CopyToDevice(input_buffer_, input.data.data(), input.GetSizeBytes());

        // Run through layers
        BackendBuffer current_input = input_buffer_;
        BackendBuffer current_output = output_buffer_;

        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i]->Forward(current_input, current_output, batch_size, seq_len)) {
                return false;
            }

            // Swap buffers for next layer
            if (i < layers_.size() - 1) {
                std::swap(current_input, current_output);
            }
        }

        // Copy result back to host
        output = TestTensor({seq_len, config_.hidden_size});
        backend_->CopyFromDevice(output.data.data(), current_output, output.GetSizeBytes());

        return true;
    }

    BackendMetrics GetPerformanceMetrics() const {
        return backend_->GetMetrics();
    }
};

/**
 * @brief Utility functions for numerical comparison
 */
class NumericalComparison {
public:
    static bool AreEqual(const TestTensor& a, const TestTensor& b, float tolerance = 1e-5f) {
        if (a.shape != b.shape || a.data.size() != b.data.size()) {
            return false;
        }

        for (size_t i = 0; i < a.data.size(); ++i) {
            if (std::abs(a.data[i] - b.data[i]) > tolerance) {
                return false;
            }
        }

        return true;
    }

    static float ComputeMAE(const TestTensor& a, const TestTensor& b) {
        if (a.data.size() != b.data.size()) return -1.0f;

        float sum_abs_diff = 0.0f;
        for (size_t i = 0; i < a.data.size(); ++i) {
            sum_abs_diff += std::abs(a.data[i] - b.data[i]);
        }

        return sum_abs_diff / a.data.size();
    }

    static float ComputeMSE(const TestTensor& a, const TestTensor& b) {
        if (a.data.size() != b.data.size()) return -1.0f;

        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < a.data.size(); ++i) {
            float diff = a.data[i] - b.data[i];
            sum_sq_diff += diff * diff;
        }

        return sum_sq_diff / a.data.size();
    }

    static void PrintStatistics(const TestTensor& a, const TestTensor& b, const std::string& label) {
        float mae = ComputeMAE(a, b);
        float mse = ComputeMSE(a, b);
        float rmse = std::sqrt(mse);

        std::cout << label << " Comparison Statistics:" << std::endl;
        std::cout << "  MAE:  " << std::scientific << std::setprecision(6) << mae << std::endl;
        std::cout << "  MSE:  " << std::scientific << std::setprecision(6) << mse << std::endl;
        std::cout << "  RMSE: " << std::scientific << std::setprecision(6) << rmse << std::endl;
    }
};

} // namespace test
} // namespace backends
} // namespace gemma

/**
 * @brief Test fixture for inference backend tests
 */
class InferenceBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_unique<gemma::backends::BackendRegistry>();

        // Register test backends
        registry_->RegisterBackend("TestCPU", []() {
            return std::make_unique<gemma::backends::test::MockCPUBackend>();
        });

        registry_->RegisterBackend("TestGPU", []() {
            return std::make_unique<gemma::backends::test::MockGPUBackend>();
        });

        // Initialize test configuration
        config_.vocab_size = 1000;
        config_.hidden_size = 128;
        config_.intermediate_size = 256;
        config_.num_layers = 2;
        config_.num_attention_heads = 4;
        config_.max_seq_length = 32;
    }

    void TearDown() override {
        registry_.reset();
    }

    std::unique_ptr<gemma::backends::BackendRegistry> registry_;
    gemma::backends::test::ModelConfig config_;
};

/**
 * @brief Test model loading on different backends
 */
TEST_F(InferenceBackendTest, ModelLoading) {
    auto cpu_backend = registry_->CreateBackend("TestCPU");
    auto gpu_backend = registry_->CreateBackend("TestGPU");

    ASSERT_NE(cpu_backend, nullptr);
    ASSERT_NE(gpu_backend, nullptr);
    ASSERT_TRUE(cpu_backend->Initialize());
    ASSERT_TRUE(gpu_backend->Initialize());

    // Test model creation on both backends
    auto cpu_model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        cpu_backend.get(), config_);
    auto gpu_model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        gpu_backend.get(), config_);

    // Models should initialize successfully
    EXPECT_NE(cpu_model, nullptr);
    EXPECT_NE(gpu_model, nullptr);

    cpu_model.reset();
    gpu_model.reset();

    cpu_backend->Shutdown();
    gpu_backend->Shutdown();
}

/**
 * @brief Test inference result consistency across backends
 */
TEST_F(InferenceBackendTest, ResultConsistency) {
    auto cpu_backend = registry_->CreateBackend("TestCPU");
    auto gpu_backend = registry_->CreateBackend("TestGPU");

    ASSERT_NE(cpu_backend, nullptr);
    ASSERT_NE(gpu_backend, nullptr);
    ASSERT_TRUE(cpu_backend->Initialize());
    ASSERT_TRUE(gpu_backend->Initialize());

    // Create models
    auto cpu_model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        cpu_backend.get(), config_);
    auto gpu_model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        gpu_backend.get(), config_);

    // Create test input
    gemma::backends::test::TestTensor input({16, config_.hidden_size});
    input.FillRandom(-1.0f, 1.0f);

    // Run inference on both backends
    gemma::backends::test::TestTensor cpu_output, gpu_output;

    EXPECT_TRUE(cpu_model->RunInference(input, cpu_output));
    EXPECT_TRUE(gpu_model->RunInference(input, gpu_output));

    // Compare results
    EXPECT_EQ(cpu_output.shape, gpu_output.shape);

    // Results should be numerically close (allowing for floating point differences)
    float tolerance = 1e-4f; // Relaxed tolerance for different implementations
    bool results_match = gemma::backends::test::NumericalComparison::AreEqual(
        cpu_output, gpu_output, tolerance);

    if (!results_match) {
        // Print detailed comparison statistics
        gemma::backends::test::NumericalComparison::PrintStatistics(
            cpu_output, gpu_output, "CPU vs GPU");

        // For debugging: print first few values
        std::cout << "First 10 CPU values: ";
        for (int i = 0; i < std::min(10, (int)cpu_output.data.size()); ++i) {
            std::cout << cpu_output.data[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "First 10 GPU values: ";
        for (int i = 0; i < std::min(10, (int)gpu_output.data.size()); ++i) {
            std::cout << gpu_output.data[i] << " ";
        }
        std::cout << std::endl;
    }

    // For this test, we expect exact matches since both use the same mock implementation
    EXPECT_TRUE(results_match);

    cpu_model.reset();
    gpu_model.reset();
    cpu_backend->Shutdown();
    gpu_backend->Shutdown();
}

/**
 * @brief Test performance comparison between backends
 */
TEST_F(InferenceBackendTest, PerformanceComparison) {
    auto cpu_backend = registry_->CreateBackend("TestCPU");
    auto gpu_backend = registry_->CreateBackend("TestGPU");

    ASSERT_NE(cpu_backend, nullptr);
    ASSERT_NE(gpu_backend, nullptr);
    ASSERT_TRUE(cpu_backend->Initialize());
    ASSERT_TRUE(gpu_backend->Initialize());

    // Create models
    auto cpu_model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        cpu_backend.get(), config_);
    auto gpu_model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        gpu_backend.get(), config_);

    // Create test input
    gemma::backends::test::TestTensor input({config_.max_seq_length, config_.hidden_size});
    input.FillRandom(-1.0f, 1.0f);

    const int num_runs = 5;
    std::vector<double> cpu_times, gpu_times;

    // Benchmark CPU backend
    for (int run = 0; run < num_runs; ++run) {
        cpu_backend->ResetMetrics();

        auto start = std::chrono::high_resolution_clock::now();

        gemma::backends::test::TestTensor output;
        EXPECT_TRUE(cpu_model->RunInference(input, output));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cpu_times.push_back(duration.count() / 1000.0); // Convert to milliseconds
    }

    // Benchmark GPU backend
    for (int run = 0; run < num_runs; ++run) {
        gpu_backend->ResetMetrics();

        auto start = std::chrono::high_resolution_clock::now();

        gemma::backends::test::TestTensor output;
        EXPECT_TRUE(gpu_model->RunInference(input, output));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        gpu_times.push_back(duration.count() / 1000.0); // Convert to milliseconds
    }

    // Calculate average times
    double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / num_runs;
    double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / num_runs;

    std::cout << "Performance Comparison Results:" << std::endl;
    std::cout << "  CPU Average: " << cpu_avg << " ms" << std::endl;
    std::cout << "  GPU Average: " << gpu_avg << " ms" << std::endl;
    std::cout << "  Speedup: " << cpu_avg / gpu_avg << "x" << std::endl;

    // GPU should be faster (mock GPU has simulated 10x speedup)
    EXPECT_LT(gpu_avg, cpu_avg);

    // Get detailed metrics
    auto cpu_metrics = cpu_model->GetPerformanceMetrics();
    auto gpu_metrics = gpu_model->GetPerformanceMetrics();

    std::cout << "Backend Metrics:" << std::endl;
    std::cout << "  CPU Latency: " << cpu_metrics.latency_ms << " ms" << std::endl;
    std::cout << "  GPU Latency: " << gpu_metrics.latency_ms << " ms" << std::endl;

    cpu_model.reset();
    gpu_model.reset();
    cpu_backend->Shutdown();
    gpu_backend->Shutdown();
}

/**
 * @brief Test numerical stability across different input ranges
 */
TEST_F(InferenceBackendTest, NumericalStability) {
    auto backend = registry_->CreateBackend("TestCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    auto model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        backend.get(), config_);

    // Test different input ranges
    std::vector<std::pair<float, float>> input_ranges = {
        {-1.0f, 1.0f},      // Normal range
        {-10.0f, 10.0f},    // Large range
        {-0.1f, 0.1f},      // Small range
        {0.0f, 1.0f},       // Positive only
        {-1.0f, 0.0f}       // Negative only
    };

    for (const auto& [min_val, max_val] : input_ranges) {
        gemma::backends::test::TestTensor input({16, config_.hidden_size});
        input.FillRandom(min_val, max_val);

        gemma::backends::test::TestTensor output;
        EXPECT_TRUE(model->RunInference(input, output));

        // Check for NaN or infinite values
        bool has_invalid = false;
        for (float val : output.data) {
            if (std::isnan(val) || std::isinf(val)) {
                has_invalid = true;
                break;
            }
        }

        EXPECT_FALSE(has_invalid) << "Invalid values found for input range ["
                                  << min_val << ", " << max_val << "]";

        std::cout << "Input range [" << min_val << ", " << max_val
                  << "] - Output range: ["
                  << *std::min_element(output.data.begin(), output.data.end())
                  << ", "
                  << *std::max_element(output.data.begin(), output.data.end())
                  << "]" << std::endl;
    }

    model.reset();
    backend->Shutdown();
}

/**
 * @brief Test resource usage and memory management
 */
TEST_F(InferenceBackendTest, ResourceManagement) {
    auto backend = registry_->CreateBackend("TestCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    // Test multiple model instances
    const int num_models = 3;
    std::vector<std::unique_ptr<gemma::backends::test::TestInferenceModel>> models;

    for (int i = 0; i < num_models; ++i) {
        models.push_back(std::make_unique<gemma::backends::test::TestInferenceModel>(
            backend.get(), config_));
    }

    // Test inference on all models
    gemma::backends::test::TestTensor input({8, config_.hidden_size});
    input.FillRandom(-1.0f, 1.0f);

    for (auto& model : models) {
        gemma::backends::test::TestTensor output;
        EXPECT_TRUE(model->RunInference(input, output));
    }

    // Cleanup models
    models.clear();

    // Backend should still be functional
    EXPECT_TRUE(backend->IsAvailable());

    backend->Shutdown();
}

/**
 * @brief Test batch processing capabilities
 */
TEST_F(InferenceBackendTest, BatchProcessing) {
    auto backend = registry_->CreateBackend("TestCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    auto model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        backend.get(), config_);

    // Test different sequence lengths
    std::vector<int> seq_lengths = {1, 4, 8, 16, config_.max_seq_length};

    for (int seq_len : seq_lengths) {
        gemma::backends::test::TestTensor input({seq_len, config_.hidden_size});
        input.FillRandom(-1.0f, 1.0f);

        gemma::backends::test::TestTensor output;
        EXPECT_TRUE(model->RunInference(input, output));

        EXPECT_EQ(output.shape[0], seq_len);
        EXPECT_EQ(output.shape[1], config_.hidden_size);

        std::cout << "Sequence length " << seq_len << " processed successfully" << std::endl;
    }

    model.reset();
    backend->Shutdown();
}

/**
 * @brief Test error handling during inference
 */
TEST_F(InferenceBackendTest, ErrorHandling) {
    auto backend = registry_->CreateBackend("TestCPU");
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->Initialize());

    auto model = std::make_unique<gemma::backends::test::TestInferenceModel>(
        backend.get(), config_);

    // Test invalid input shapes
    gemma::backends::test::TestTensor invalid_input1({16, config_.hidden_size + 1}); // Wrong hidden size
    gemma::backends::test::TestTensor invalid_input2({config_.max_seq_length + 1, config_.hidden_size}); // Too long

    gemma::backends::test::TestTensor output;

    EXPECT_FALSE(model->RunInference(invalid_input1, output));
    EXPECT_FALSE(model->RunInference(invalid_input2, output));

    // Test valid input after errors
    gemma::backends::test::TestTensor valid_input({8, config_.hidden_size});
    valid_input.FillRandom(-1.0f, 1.0f);

    EXPECT_TRUE(model->RunInference(valid_input, output));

    model.reset();
    backend->Shutdown();
}

/**
 * @brief Main function to run all tests
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "Running Gemma.cpp Inference Backend Tests..." << std::endl;
    std::cout << "===========================================" << std::endl;

    return RUN_ALL_TESTS();
}