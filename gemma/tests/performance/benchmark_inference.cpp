/**
 * @file benchmark_inference.cpp
 * @brief Performance benchmarks for Gemma inference operations
 * 
 * This file contains comprehensive benchmarks for testing inference performance
 * across different models, backends, and input configurations. It measures
 * token generation speed, latency, memory usage, and throughput metrics.
 */

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "../utils/test_helpers.h"
#include "../utils/mock_backend.h"

namespace gemma {
namespace tests {
namespace performance {

/**
 * @class InferenceBenchmarkTest
 * @brief Test fixture for inference performance benchmarks
 */
class InferenceBenchmarkTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        // Initialize mock backends
        cpu_backend_ = std::make_unique<MockCPUBackend>();
        intel_backend_ = std::make_unique<MockIntelBackend>();
        cuda_backend_ = std::make_unique<MockCUDABackend>();
        vulkan_backend_ = std::make_unique<MockVulkanBackend>();
        
        // Setup default inference parameters
        setupInferenceParameters();
        
        // Initialize performance tracking
        initializePerformanceTracking();
    }

    void TearDown() override {
        // Generate performance report
        generatePerformanceReport();
        GemmaTestBase::TearDown();
    }

    void setupInferenceParameters() {
        // Default model parameters
        model_config_.vocab_size = 32000;
        model_config_.max_sequence_length = 2048;
        model_config_.embedding_dim = 2048;
        model_config_.num_layers = 24;
        model_config_.num_heads = 16;
        
        // Inference parameters
        inference_params_.temperature = 0.7f;
        inference_params_.top_p = 0.9f;
        inference_params_.top_k = 40;
        inference_params_.max_tokens = 256;
        inference_params_.batch_size = 1;
    }

    void initializePerformanceTracking() {
        // Initialize metrics tracking
        metrics_.clear();
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void generatePerformanceReport() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time_).count();
        
        std::string report_path = (test_dir_ / "inference_performance_report.json").string();
        std::ofstream report(report_path);
        
        report << "{\n";
        report << "  \"test_duration_ms\": " << duration << ",\n";
        report << "  \"total_tests\": " << metrics_.size() << ",\n";
        report << "  \"metrics\": [\n";
        
        for (size_t i = 0; i < metrics_.size(); ++i) {
            const auto& metric = metrics_[i];
            report << "    {\n";
            report << "      \"test_name\": \"" << metric.test_name << "\",\n";
            report << "      \"tokens_per_second\": " << metric.tokens_per_second << ",\n";
            report << "      \"latency_ms\": " << metric.latency_ms << ",\n";
            report << "      \"memory_usage_mb\": " << metric.memory_usage_mb << ",\n";
            report << "      \"backend\": \"" << metric.backend << "\"\n";
            report << "    }";
            if (i < metrics_.size() - 1) report << ",";
            report << "\n";
        }
        
        report << "  ]\n";
        report << "}\n";
        report.close();
    }

    // Test data structures
    struct ModelConfig {
        size_t vocab_size;
        size_t max_sequence_length;
        size_t embedding_dim;
        size_t num_layers;
        size_t num_heads;
    };

    struct InferenceParams {
        float temperature;
        float top_p;
        int top_k;
        int max_tokens;
        int batch_size;
    };

    struct PerformanceMetric {
        std::string test_name;
        std::string backend;
        double tokens_per_second;
        double latency_ms;
        double memory_usage_mb;
        double throughput_mb_per_s;
    };

    // Mock backends
    std::unique_ptr<MockCPUBackend> cpu_backend_;
    std::unique_ptr<MockIntelBackend> intel_backend_;
    std::unique_ptr<MockCUDABackend> cuda_backend_;
    std::unique_ptr<MockVulkanBackend> vulkan_backend_;
    
    // Test configuration
    ModelConfig model_config_;
    InferenceParams inference_params_;
    
    // Performance tracking
    std::vector<PerformanceMetric> metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

/**
 * @brief Test single token generation performance
 */
TEST_F(InferenceBenchmarkTest, SingleTokenGeneration) {
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()},
        {"vulkan", vulkan_backend_.get()}
    };

    for (const auto& [backend_name, backend] : backends) {
        EXPECT_CALL(*backend, initialize())
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*backend, generate_token(::testing::_))
            .WillRepeatedly(::testing::Return(std::vector<int>{42}));

        auto start = std::chrono::high_resolution_clock::now();
        
        // Generate single token multiple times for average
        const int num_iterations = 1000;
        for (int i = 0; i < num_iterations; ++i) {
            std::vector<int> input_tokens = {1, 2, 3, 4, 5};
            auto result = backend->generate_token(input_tokens);
            EXPECT_FALSE(result.empty());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        PerformanceMetric metric;
        metric.test_name = "SingleTokenGeneration";
        metric.backend = backend_name;
        metric.tokens_per_second = (num_iterations * 1000.0) / duration_ms;
        metric.latency_ms = duration_ms / num_iterations;
        metric.memory_usage_mb = PerformanceUtils::GetMemoryUsageMB();
        
        metrics_.push_back(metric);
        
        // Performance assertions
        if (backend_name == "cpu") {
            EXPECT_GT(metric.tokens_per_second, 10.0) << "CPU backend too slow";
        } else if (backend_name == "cuda") {
            EXPECT_GT(metric.tokens_per_second, 100.0) << "CUDA backend too slow";
        }
        
        EXPECT_LT(metric.latency_ms, 100.0) << "Latency too high for " << backend_name;
    }
}

/**
 * @brief Test batch inference performance
 */
TEST_F(InferenceBenchmarkTest, BatchInference) {
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32};
    
    for (int batch_size : batch_sizes) {
        EXPECT_CALL(*cuda_backend_, initialize())
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*cuda_backend_, set_batch_size(batch_size))
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*cuda_backend_, generate_batch(::testing::_))
            .WillOnce(::testing::Return(std::vector<std::vector<int>>(
                batch_size, std::vector<int>{42, 43, 44})));

        auto start = std::chrono::high_resolution_clock::now();
        
        // Prepare batch input
        std::vector<std::vector<int>> batch_inputs(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batch_inputs[i] = {1, 2, 3, 4, 5, i}; // Different input per batch item
        }
        
        cuda_backend_->set_batch_size(batch_size);
        auto results = cuda_backend_->generate_batch(batch_inputs);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        EXPECT_EQ(results.size(), batch_size);
        
        PerformanceMetric metric;
        metric.test_name = "BatchInference_" + std::to_string(batch_size);
        metric.backend = "cuda";
        metric.tokens_per_second = (batch_size * 3 * 1000.0) / duration_ms; // 3 tokens per result
        metric.latency_ms = duration_ms;
        metric.memory_usage_mb = PerformanceUtils::GetMemoryUsageMB();
        metric.throughput_mb_per_s = (batch_size * 6 * sizeof(int)) / (duration_ms / 1000.0) / (1024 * 1024);
        
        metrics_.push_back(metric);
        
        // Batch processing should scale efficiently
        if (batch_size == 1) {
            EXPECT_GT(metric.tokens_per_second, 50.0);
        } else {
            // Expect better throughput with larger batches
            EXPECT_GT(metric.tokens_per_second, 50.0 * (batch_size * 0.7));
        }
    }
}

/**
 * @brief Test sequence length scaling performance
 */
TEST_F(InferenceBenchmarkTest, SequenceLengthScaling) {
    std::vector<int> sequence_lengths = {128, 256, 512, 1024, 2048};
    
    for (int seq_len : sequence_lengths) {
        EXPECT_CALL(*intel_backend_, initialize())
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*intel_backend_, set_max_sequence_length(seq_len))
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*intel_backend_, generate_token(::testing::_))
            .WillRepeatedly(::testing::Return(std::vector<int>{42}));

        auto start = std::chrono::high_resolution_clock::now();
        
        // Generate input sequence of specified length
        std::vector<int> input_tokens(seq_len);
        std::iota(input_tokens.begin(), input_tokens.end(), 1);
        
        intel_backend_->set_max_sequence_length(seq_len);
        
        // Generate multiple tokens
        const int num_tokens = 50;
        for (int i = 0; i < num_tokens; ++i) {
            auto result = intel_backend_->generate_token(input_tokens);
            EXPECT_FALSE(result.empty());
            if (!result.empty()) {
                input_tokens.push_back(result[0]);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        PerformanceMetric metric;
        metric.test_name = "SequenceLength_" + std::to_string(seq_len);
        metric.backend = "intel";
        metric.tokens_per_second = (num_tokens * 1000.0) / duration_ms;
        metric.latency_ms = duration_ms / num_tokens;
        metric.memory_usage_mb = PerformanceUtils::GetMemoryUsageMB();
        
        metrics_.push_back(metric);
        
        // Longer sequences should maintain reasonable performance
        EXPECT_GT(metric.tokens_per_second, 5.0) << "Too slow for sequence length " << seq_len;
        EXPECT_LT(metric.latency_ms, 200.0) << "Latency too high for sequence length " << seq_len;
    }
}

/**
 * @brief Test concurrent inference performance
 */
TEST_F(InferenceBenchmarkTest, ConcurrentInference) {
    const int num_threads = 4;
    const int iterations_per_thread = 100;
    
    EXPECT_CALL(*cuda_backend_, initialize())
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*cuda_backend_, is_thread_safe())
        .WillRepeatedly(::testing::Return(true));
    EXPECT_CALL(*cuda_backend_, generate_token(::testing::_))
        .WillRepeatedly(::testing::Return(std::vector<int>{42}));

    cuda_backend_->initialize();
    ASSERT_TRUE(cuda_backend_->is_thread_safe());
    
    std::atomic<int> completed_iterations{0};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch concurrent inference threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<int> input_tokens = {1, 2, 3, 4, 5, t}; // Thread-specific input
            
            for (int i = 0; i < iterations_per_thread; ++i) {
                auto result = cuda_backend_->generate_token(input_tokens);
                EXPECT_FALSE(result.empty());
                completed_iterations.fetch_add(1);
                
                // Small delay to simulate real workload
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    EXPECT_EQ(completed_iterations.load(), num_threads * iterations_per_thread);
    
    PerformanceMetric metric;
    metric.test_name = "ConcurrentInference";
    metric.backend = "cuda";
    metric.tokens_per_second = (completed_iterations.load() * 1000.0) / duration_ms;
    metric.latency_ms = duration_ms / completed_iterations.load();
    metric.memory_usage_mb = PerformanceUtils::GetMemoryUsageMB();
    
    metrics_.push_back(metric);
    
    // Concurrent performance should be reasonable
    EXPECT_GT(metric.tokens_per_second, 50.0) << "Concurrent performance too low";
    EXPECT_LT(metric.latency_ms, 50.0) << "Concurrent latency too high";
}

/**
 * @brief Test memory usage during inference
 */
TEST_F(InferenceBenchmarkTest, MemoryUsageDuringInference) {
    std::vector<int> sequence_lengths = {256, 512, 1024, 2048};
    
    for (int seq_len : sequence_lengths) {
        EXPECT_CALL(*cpu_backend_, initialize())
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*cpu_backend_, generate_token(::testing::_))
            .WillRepeatedly(::testing::Return(std::vector<int>{42}));

        double initial_memory = PerformanceUtils::GetMemoryUsageMB();
        
        cpu_backend_->initialize();
        
        // Generate long sequence to stress memory
        std::vector<int> input_tokens(seq_len);
        std::iota(input_tokens.begin(), input_tokens.end(), 1);
        
        double pre_inference_memory = PerformanceUtils::GetMemoryUsageMB();
        
        // Generate tokens and track memory growth
        std::vector<double> memory_samples;
        const int num_tokens = 100;
        
        for (int i = 0; i < num_tokens; ++i) {
            auto result = cpu_backend_->generate_token(input_tokens);
            if (!result.empty()) {
                input_tokens.push_back(result[0]);
            }
            
            if (i % 10 == 0) {
                memory_samples.push_back(PerformanceUtils::GetMemoryUsageMB());
            }
        }
        
        double final_memory = PerformanceUtils::GetMemoryUsageMB();
        
        // Calculate memory statistics
        double max_memory = *std::max_element(memory_samples.begin(), memory_samples.end());
        double avg_memory = std::accumulate(memory_samples.begin(), memory_samples.end(), 0.0) / memory_samples.size();
        double memory_growth = final_memory - pre_inference_memory;
        
        PerformanceMetric metric;
        metric.test_name = "MemoryUsage_" + std::to_string(seq_len);
        metric.backend = "cpu";
        metric.tokens_per_second = 0.0; // Not applicable for this test
        metric.latency_ms = 0.0; // Not applicable for this test
        metric.memory_usage_mb = max_memory;
        
        metrics_.push_back(metric);
        
        // Memory usage should be reasonable and not grow excessively
        EXPECT_LT(memory_growth, 100.0) << "Memory growth too high for sequence length " << seq_len;
        EXPECT_LT(max_memory, initial_memory + 500.0) << "Peak memory usage too high";
        
        // Memory should not increase linearly with each token
        double memory_per_token = memory_growth / num_tokens;
        EXPECT_LT(memory_per_token, 1.0) << "Memory per token too high";
    }
}

/**
 * @brief Test inference with different sampling parameters
 */
TEST_F(InferenceBenchmarkTest, SamplingParameterPerformance) {
    struct SamplingConfig {
        float temperature;
        float top_p;
        int top_k;
        std::string name;
    };
    
    std::vector<SamplingConfig> configs = {
        {0.1f, 0.9f, 10, "Conservative"},
        {0.7f, 0.9f, 40, "Balanced"},
        {1.0f, 1.0f, 100, "Creative"},
        {1.5f, 0.95f, 200, "VeryCreative"}
    };
    
    for (const auto& config : configs) {
        EXPECT_CALL(*vulkan_backend_, initialize())
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*vulkan_backend_, set_sampling_parameters(
            config.temperature, config.top_p, config.top_k))
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*vulkan_backend_, generate_token(::testing::_))
            .WillRepeatedly(::testing::Return(std::vector<int>{42}));

        vulkan_backend_->initialize();
        vulkan_backend_->set_sampling_parameters(config.temperature, config.top_p, config.top_k);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int> input_tokens = {1, 2, 3, 4, 5};
        const int num_tokens = 200;
        
        for (int i = 0; i < num_tokens; ++i) {
            auto result = vulkan_backend_->generate_token(input_tokens);
            EXPECT_FALSE(result.empty());
            if (!result.empty()) {
                input_tokens.push_back(result[0]);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        PerformanceMetric metric;
        metric.test_name = "Sampling_" + config.name;
        metric.backend = "vulkan";
        metric.tokens_per_second = (num_tokens * 1000.0) / duration_ms;
        metric.latency_ms = duration_ms / num_tokens;
        metric.memory_usage_mb = PerformanceUtils::GetMemoryUsageMB();
        
        metrics_.push_back(metric);
        
        // All sampling configurations should maintain reasonable performance
        EXPECT_GT(metric.tokens_per_second, 20.0) << "Too slow for " << config.name << " sampling";
        EXPECT_LT(metric.latency_ms, 100.0) << "Latency too high for " << config.name << " sampling";
    }
}

/**
 * @brief Benchmark for comparing inference across all backends
 */
TEST_F(InferenceBenchmarkTest, CrossBackendPerformanceComparison) {
    struct BackendResult {
        std::string name;
        double tokens_per_second;
        double avg_latency_ms;
        double peak_memory_mb;
    };
    
    std::vector<BackendResult> results;
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()},
        {"vulkan", vulkan_backend_.get()}
    };
    
    for (const auto& [backend_name, backend] : backends) {
        EXPECT_CALL(*backend, initialize())
            .WillOnce(::testing::Return(true));
        EXPECT_CALL(*backend, generate_token(::testing::_))
            .WillRepeatedly(::testing::Return(std::vector<int>{42}));

        backend->initialize();
        
        std::vector<double> latencies;
        double initial_memory = PerformanceUtils::GetMemoryUsageMB();
        double peak_memory = initial_memory;
        
        auto overall_start = std::chrono::high_resolution_clock::now();
        
        const int num_iterations = 100;
        std::vector<int> input_tokens = {1, 2, 3, 4, 5};
        
        for (int i = 0; i < num_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = backend->generate_token(input_tokens);
            auto end = std::chrono::high_resolution_clock::now();
            
            EXPECT_FALSE(result.empty());
            if (!result.empty()) {
                input_tokens.push_back(result[0]);
            }
            
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency);
            
            double current_memory = PerformanceUtils::GetMemoryUsageMB();
            peak_memory = std::max(peak_memory, current_memory);
        }
        
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration_ms = std::chrono::duration<double, std::milli>(
            overall_end - overall_start).count();
        
        BackendResult result;
        result.name = backend_name;
        result.tokens_per_second = (num_iterations * 1000.0) / total_duration_ms;
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        result.peak_memory_mb = peak_memory;
        
        results.push_back(result);
        
        PerformanceMetric metric;
        metric.test_name = "CrossBackendComparison";
        metric.backend = backend_name;
        metric.tokens_per_second = result.tokens_per_second;
        metric.latency_ms = result.avg_latency_ms;
        metric.memory_usage_mb = result.peak_memory_mb;
        
        metrics_.push_back(metric);
    }
    
    // Sort results by performance
    std::sort(results.begin(), results.end(), 
        [](const BackendResult& a, const BackendResult& b) {
            return a.tokens_per_second > b.tokens_per_second;
        });
    
    // Performance expectations (adjust based on actual hardware)
    EXPECT_GT(results[0].tokens_per_second, 50.0) << "Best backend should achieve > 50 tokens/sec";
    
    // All backends should be functional
    for (const auto& result : results) {
        EXPECT_GT(result.tokens_per_second, 1.0) << result.name << " backend not functional";
        EXPECT_LT(result.avg_latency_ms, 1000.0) << result.name << " backend too slow";
    }
    
    // Log performance ranking
    std::cout << "\nBackend Performance Ranking:\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << (i + 1) << ". " << result.name 
                  << ": " << result.tokens_per_second << " tokens/sec, "
                  << result.avg_latency_ms << "ms avg latency, "
                  << result.peak_memory_mb << "MB peak memory\n";
    }
}

} // namespace performance
} // namespace tests
} // namespace gemma