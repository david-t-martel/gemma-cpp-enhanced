#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <benchmark/benchmark.h>
#include "../utils/test_helpers.h"
#include "../utils/mock_backend.h"
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>

// Performance benchmarking for backend comparison
// Uses Google Benchmark for precise performance measurements

using namespace gemma::test;
using namespace testing;

// Benchmark configuration
struct BenchmarkConfig {
    size_t sequence_length = 512;
    size_t vocab_size = 32000;
    size_t batch_size = 1;
    int num_iterations = 100;
    bool warmup_enabled = true;
    int warmup_iterations = 10;
    bool collect_detailed_metrics = true;
    std::string output_format = "json";
};

// Benchmark results structure
struct BenchmarkResult {
    std::string backend_name;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double throughput_tokens_per_second;
    size_t memory_usage_bytes;
    double memory_bandwidth_gbps;
    size_t total_operations;
    bool completed_successfully;
    std::string error_message;
    
    nlohmann::json to_json() const {
        return nlohmann::json{
            {"backend_name", backend_name},
            {"avg_time_ms", avg_time_ms},
            {"min_time_ms", min_time_ms},
            {"max_time_ms", max_time_ms},
            {"std_dev_ms", std_dev_ms},
            {"throughput_tokens_per_second", throughput_tokens_per_second},
            {"memory_usage_bytes", memory_usage_bytes},
            {"memory_bandwidth_gbps", memory_bandwidth_gbps},
            {"total_operations", total_operations},
            {"completed_successfully", completed_successfully},
            {"error_message", error_message}
        };
    }
};

// Backend performance profiler
class BackendProfiler {
public:
    static BenchmarkResult profile_backend(MockBackendBase* backend, 
                                         const BenchmarkConfig& config) {
        BenchmarkResult result;
        result.backend_name = backend->get_backend_type();
        result.completed_successfully = false;
        
        try {
            // Initialize backend
            if (!backend->initialize()) {
                result.error_message = "Failed to initialize backend";
                return result;
            }
            
            // Load mock model
            if (!backend->load_model("test_model.sbs")) {
                result.error_message = "Failed to load model";
                return result;
            }
            
            // Generate test data
            auto test_tokens = TestDataGenerator::generate_random_tokens(
                config.sequence_length, config.vocab_size);
            
            std::vector<double> execution_times;
            execution_times.reserve(config.num_iterations);
            
            // Warmup phase
            if (config.warmup_enabled) {
                for (int i = 0; i < config.warmup_iterations; ++i) {
                    backend->forward(test_tokens);
                }
            }
            
            // Benchmark phase
            for (int i = 0; i < config.num_iterations; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();
                
                auto output = backend->forward(test_tokens);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_time - start_time);
                
                execution_times.push_back(duration.count() / 1e6); // Convert to ms
                
                if (output.empty()) {
                    result.error_message = "Backend returned empty output";
                    return result;
                }
            }
            
            // Calculate statistics
            result.avg_time_ms = std::accumulate(execution_times.begin(), 
                                               execution_times.end(), 0.0) / execution_times.size();
            result.min_time_ms = *std::min_element(execution_times.begin(), execution_times.end());
            result.max_time_ms = *std::max_element(execution_times.begin(), execution_times.end());
            
            // Calculate standard deviation
            double variance = 0.0;
            for (double time : execution_times) {
                variance += (time - result.avg_time_ms) * (time - result.avg_time_ms);
            }
            result.std_dev_ms = std::sqrt(variance / execution_times.size());
            
            // Calculate throughput
            result.throughput_tokens_per_second = (config.sequence_length * 1000.0) / result.avg_time_ms;
            
            // Get memory usage
            result.memory_usage_bytes = backend->get_memory_usage();
            
            // Estimate memory bandwidth (simplified calculation)
            size_t data_transferred = config.sequence_length * sizeof(float) * 2; // Input + output
            result.memory_bandwidth_gbps = (data_transferred / (1024.0 * 1024.0 * 1024.0)) / 
                                          (result.avg_time_ms / 1000.0);
            
            result.total_operations = config.num_iterations;
            result.completed_successfully = true;
            
        } catch (const std::exception& e) {
            result.error_message = e.what();
        }
        
        return result;
    }
};

// Test fixture for backend benchmarks
class BackendBenchmarkTest : public BackendTestFixture {
protected:
    void SetUp() override {
        BackendTestFixture::SetUp();
        
        config_.sequence_length = 512;
        config_.vocab_size = 32000;
        config_.batch_size = 1;
        config_.num_iterations = 50; // Reduced for testing
        config_.warmup_enabled = true;
        config_.warmup_iterations = 5;
        
        setup_benchmark_expectations();
    }
    
    void setup_benchmark_expectations() {
        // Setup performance expectations for each backend type
        setup_cpu_expectations();
        setup_intel_expectations();
        setup_cuda_expectations();
        setup_vulkan_expectations();
    }
    
    void setup_cpu_expectations() {
        // CPU is slower but more consistent
        ON_CALL(*cpu_backend_, forward(_))
            .WillByDefault(Invoke([](const std::vector<int>& tokens) {
                std::this_thread::sleep_for(std::chrono::microseconds(100)); // 100µs
                return std::vector<float>(32000, 0.1f);
            }));
    }
    
    void setup_intel_expectations() {
        // Intel GPU is faster, especially for larger workloads
        ON_CALL(*intel_backend_, forward(_))
            .WillByDefault(Invoke([](const std::vector<int>& tokens) {
                std::this_thread::sleep_for(std::chrono::microseconds(50)); // 50µs
                return std::vector<float>(32000, 0.2f);
            }));
    }
    
    void setup_cuda_expectations() {
        // CUDA is fastest for parallel workloads
        ON_CALL(*cuda_backend_, forward(_))
            .WillByDefault(Invoke([](const std::vector<int>& tokens) {
                std::this_thread::sleep_for(std::chrono::microseconds(20)); // 20µs
                return std::vector<float>(32000, 0.3f);
            }));
    }
    
    void setup_vulkan_expectations() {
        // Vulkan performance varies by implementation
        ON_CALL(*vulkan_backend_, forward(_))
            .WillByDefault(Invoke([](const std::vector<int>& tokens) {
                std::this_thread::sleep_for(std::chrono::microseconds(30)); // 30µs
                return std::vector<float>(32000, 0.4f);
            }));
    }
    
    BenchmarkConfig config_;
};

// Basic backend performance comparison
TEST_F(BackendBenchmarkTest, CompareAllBackends) {
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()},
        {"vulkan", vulkan_backend_.get()}
    };
    
    std::vector<BenchmarkResult> results;
    
    for (auto& [name, backend] : backends) {
        std::cout << "Benchmarking " << name << " backend..." << std::endl;
        
        auto result = BackendProfiler::profile_backend(backend, config_);
        results.push_back(result);
        
        EXPECT_TRUE(result.completed_successfully) 
            << "Benchmark failed for " << name << ": " << result.error_message;
        
        std::cout << "  Average time: " << result.avg_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << result.throughput_tokens_per_second << " tokens/s" << std::endl;
    }
    
    // Sort by performance (fastest first)
    std::sort(results.begin(), results.end(), 
              [](const BenchmarkResult& a, const BenchmarkResult& b) {
                  return a.avg_time_ms < b.avg_time_ms;
              });
    
    std::cout << "\nPerformance ranking:" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << i + 1 << ". " << results[i].backend_name 
                  << " - " << results[i].avg_time_ms << " ms" << std::endl;
    }
    
    // CUDA should generally be fastest
    auto cuda_result = std::find_if(results.begin(), results.end(),
        [](const BenchmarkResult& r) { return r.backend_name == "cuda"; });
    
    if (cuda_result != results.end()) {
        // CUDA should be in top 2 performers
        auto cuda_rank = std::distance(results.begin(), cuda_result);
        EXPECT_LT(cuda_rank, 2) << "CUDA should be among top performers";
    }
}

// Sequence length scaling benchmark
TEST_F(BackendBenchmarkTest, SequenceLengthScaling) {
    std::vector<size_t> sequence_lengths = {128, 256, 512, 1024, 2048};
    
    for (size_t seq_len : sequence_lengths) {
        config_.sequence_length = seq_len;
        
        auto cuda_result = BackendProfiler::profile_backend(cuda_backend_.get(), config_);
        auto cpu_result = BackendProfiler::profile_backend(cpu_backend_.get(), config_);
        
        EXPECT_TRUE(cuda_result.completed_successfully);
        EXPECT_TRUE(cpu_result.completed_successfully);
        
        std::cout << "Sequence length " << seq_len << ":" << std::endl;
        std::cout << "  CUDA: " << cuda_result.avg_time_ms << " ms" << std::endl;
        std::cout << "  CPU:  " << cpu_result.avg_time_ms << " ms" << std::endl;
        std::cout << "  Speedup: " << (cpu_result.avg_time_ms / cuda_result.avg_time_ms) << "x" << std::endl;
        
        // GPU advantage should increase with sequence length
        if (seq_len >= 512) {
            EXPECT_LT(cuda_result.avg_time_ms, cpu_result.avg_time_ms)
                << "GPU should be faster for longer sequences";
        }
    }
}

// Batch size scaling benchmark
TEST_F(BackendBenchmarkTest, BatchSizeScaling) {
    std::vector<size_t> batch_sizes = {1, 2, 4, 8, 16};
    
    // Test with Intel backend (good for batch processing)
    for (size_t batch_size : batch_sizes) {
        config_.batch_size = batch_size;
        
        // Generate batch input
        std::vector<std::vector<int>> batch_input;
        for (size_t i = 0; i < batch_size; ++i) {
            batch_input.push_back(TestDataGenerator::generate_random_tokens(
                config_.sequence_length, config_.vocab_size));
        }
        
        EXPECT_CALL(*intel_backend_, forward_batch(batch_input))
            .Times(config_.num_iterations)
            .WillRepeatedly(Return(std::vector<std::vector<float>>(
                batch_size, std::vector<float>(32000, 0.5f))));
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < config_.num_iterations; ++i) {
            auto results = intel_backend_->forward_batch(batch_input);
            EXPECT_EQ(results.size(), batch_size);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double avg_time_per_batch = static_cast<double>(duration.count()) / config_.num_iterations;
        double throughput = (batch_size * config_.sequence_length * 1000.0) / avg_time_per_batch;
        
        std::cout << "Batch size " << batch_size << ": " 
                  << avg_time_per_batch << " ms/batch, "
                  << throughput << " tokens/s" << std::endl;
        
        // Throughput should generally increase with batch size (up to a point)
        if (batch_size <= 8) {
            EXPECT_GT(throughput, 1000.0) << "Throughput too low for batch size " << batch_size;
        }
    }
}

// Memory usage benchmark
TEST_F(BackendBenchmarkTest, MemoryUsageBenchmark) {
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()},
        {"vulkan", vulkan_backend_.get()}
    };
    
    for (auto& [name, backend] : backends) {
        // Simulate different memory usage patterns
        size_t base_memory = 1024 * 1024 * 1024; // 1GB base
        size_t memory_multiplier = (name == "cpu") ? 2 : 1; // CPU uses more system RAM
        
        EXPECT_CALL(*backend, get_memory_usage())
            .WillRepeatedly(Return(base_memory * memory_multiplier));
        
        EXPECT_CALL(*backend, get_available_memory())
            .WillRepeatedly(Return(base_memory * memory_multiplier * 4)); // 4x available
        
        backend->initialize();
        backend->load_model("test_model.sbs");
        
        size_t memory_usage = backend->get_memory_usage();
        size_t available_memory = backend->get_available_memory();
        
        std::cout << name << " backend memory:" << std::endl;
        std::cout << "  Usage: " << (memory_usage / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Available: " << (available_memory / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Utilization: " << (100.0 * memory_usage / available_memory) << "%" << std::endl;
        
        EXPECT_GT(memory_usage, 0);
        EXPECT_GT(available_memory, memory_usage);
        
        // Memory utilization should be reasonable (not too high)
        double utilization = static_cast<double>(memory_usage) / available_memory;
        EXPECT_LT(utilization, 0.8) << name << " backend memory utilization too high";
    }
}

// Precision and accuracy benchmark
TEST_F(BackendBenchmarkTest, PrecisionAccuracyBenchmark) {
    auto test_tokens = TestDataGenerator::generate_random_tokens(256, 32000);
    
    // Get baseline results from CPU (assumed most accurate)
    EXPECT_CALL(*cpu_backend_, forward(test_tokens))
        .WillOnce(Return(std::vector<float>(32000, 1.0f))); // Reference output
    
    auto cpu_output = cpu_backend_->forward(test_tokens);
    
    // Compare other backends to CPU baseline
    std::vector<std::pair<std::string, MockBackendBase*>> other_backends = {
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()},
        {"vulkan", vulkan_backend_.get()}
    };
    
    for (auto& [name, backend] : other_backends) {
        // Simulate slight numerical differences
        std::vector<float> backend_output(32000);
        for (size_t i = 0; i < backend_output.size(); ++i) {
            backend_output[i] = 1.0f + (std::rand() % 100 - 50) / 100000.0f; // Small random error
        }
        
        EXPECT_CALL(*backend, forward(test_tokens))
            .WillOnce(Return(backend_output));
        
        auto output = backend->forward(test_tokens);
        
        // Calculate mean absolute error
        double mae = 0.0;
        for (size_t i = 0; i < cpu_output.size(); ++i) {
            mae += std::abs(cpu_output[i] - output[i]);
        }
        mae /= cpu_output.size();
        
        std::cout << name << " backend MAE vs CPU: " << mae << std::endl;
        
        // Numerical error should be small
        EXPECT_LT(mae, 1e-3) << name << " backend has too much numerical error";
    }
}

// Concurrent performance benchmark
TEST_F(BackendBenchmarkTest, ConcurrentPerformanceBenchmark) {
    const int num_threads = 4;
    const int operations_per_thread = 20;
    
    auto test_tokens = TestDataGenerator::generate_random_tokens(config_.sequence_length, config_.vocab_size);
    std::vector<float> mock_output(32000, 0.7f);
    
    // Test concurrent access for each backend
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()}
    };
    
    for (auto& [name, backend] : backends) {
        EXPECT_CALL(*backend, forward(test_tokens))
            .Times(num_threads * operations_per_thread)
            .WillRepeatedly(Return(mock_output));
        
        std::atomic<int> successful_operations{0};
        std::vector<double> thread_times(num_threads);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                auto thread_start = std::chrono::high_resolution_clock::now();
                
                for (int i = 0; i < operations_per_thread; ++i) {
                    try {
                        auto result = backend->forward(test_tokens);
                        if (!result.empty()) {
                            successful_operations++;
                        }
                    } catch (...) {
                        // Count failures
                    }
                }
                
                auto thread_end = std::chrono::high_resolution_clock::now();
                thread_times[t] = std::chrono::duration_cast<std::chrono::milliseconds>(
                    thread_end - thread_start).count();
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double total_throughput = (successful_operations * config_.sequence_length * 1000.0) / total_time.count();
        double avg_thread_time = std::accumulate(thread_times.begin(), thread_times.end(), 0.0) / num_threads;
        
        std::cout << name << " concurrent performance:" << std::endl;
        std::cout << "  Successful operations: " << successful_operations << "/" << (num_threads * operations_per_thread) << std::endl;
        std::cout << "  Total throughput: " << total_throughput << " tokens/s" << std::endl;
        std::cout << "  Average thread time: " << avg_thread_time << " ms" << std::endl;
        
        EXPECT_EQ(successful_operations, num_threads * operations_per_thread)
            << name << " backend failed some concurrent operations";
        
        EXPECT_GT(total_throughput, 1000.0)
            << name << " backend concurrent throughput too low";
    }
}

// Stability and reliability benchmark
TEST_F(BackendBenchmarkTest, StabilityReliabilityBenchmark) {
    const int long_run_iterations = 1000;
    auto test_tokens = TestDataGenerator::generate_random_tokens(config_.sequence_length, config_.vocab_size);
    
    // Test each backend for stability over many iterations
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()}
    };
    
    for (auto& [name, backend] : backends) {
        std::vector<double> execution_times;
        execution_times.reserve(long_run_iterations);
        int failed_operations = 0;
        
        EXPECT_CALL(*backend, forward(test_tokens))
            .Times(long_run_iterations)
            .WillRepeatedly(Invoke([&](const std::vector<int>&) {
                // Simulate occasional failures (1% failure rate)
                if (std::rand() % 100 == 0) {
                    return std::vector<float>{}; // Empty = failure
                }
                return std::vector<float>(32000, 0.5f);
            }));
        
        auto start_time = std::chrono::steady_clock::now();
        
        for (int i = 0; i < long_run_iterations; ++i) {
            auto op_start = std::chrono::high_resolution_clock::now();
            
            auto result = backend->forward(test_tokens);
            
            auto op_end = std::chrono::high_resolution_clock::now();
            auto op_duration = std::chrono::duration_cast<std::chrono::microseconds>(op_end - op_start);
            
            if (result.empty()) {
                failed_operations++;
            } else {
                execution_times.push_back(op_duration.count() / 1000.0); // Convert to ms
            }
            
            // Check for memory leaks every 100 iterations
            if (i % 100 == 0) {
                size_t memory_usage = backend->get_memory_usage();
                // Memory usage shouldn't grow unbounded
                EXPECT_LT(memory_usage, 10ULL * 1024 * 1024 * 1024) // 10GB limit
                    << name << " backend may have memory leak at iteration " << i;
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        // Calculate performance statistics
        double avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
        double min_time = *std::min_element(execution_times.begin(), execution_times.end());
        double max_time = *std::max_element(execution_times.begin(), execution_times.end());
        
        double failure_rate = static_cast<double>(failed_operations) / long_run_iterations;
        
        std::cout << name << " stability results (" << long_run_iterations << " iterations):" << std::endl;
        std::cout << "  Failure rate: " << (failure_rate * 100) << "%" << std::endl;
        std::cout << "  Average time: " << avg_time << " ms" << std::endl;
        std::cout << "  Min/Max time: " << min_time << "/" << max_time << " ms" << std::endl;
        std::cout << "  Time variation: " << ((max_time - min_time) / avg_time * 100) << "%" << std::endl;
        std::cout << "  Total runtime: " << total_duration.count() << " seconds" << std::endl;
        
        // Reliability requirements
        EXPECT_LT(failure_rate, 0.05) << name << " backend failure rate too high"; // < 5%
        EXPECT_LT((max_time - min_time) / avg_time, 2.0) << name << " backend timing too variable"; // < 200% variation
    }
}

// Power efficiency benchmark (simulated)
TEST_F(BackendBenchmarkTest, PowerEfficiencyBenchmark) {
    auto test_tokens = TestDataGenerator::generate_random_tokens(config_.sequence_length, config_.vocab_size);
    
    struct PowerMetrics {
        std::string backend_name;
        double estimated_power_watts;
        double performance_per_watt;
        double energy_per_token_mj;
    };
    
    std::vector<PowerMetrics> power_results;
    
    // Simulate power consumption based on backend type
    std::map<std::string, double> estimated_power = {
        {"cpu", 65.0},      // Desktop CPU
        {"intel", 75.0},    // Intel GPU
        {"cuda", 150.0},    // NVIDIA GPU
        {"vulkan", 100.0}   // Generic GPU
    };
    
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()},
        {"vulkan", vulkan_backend_.get()}
    };
    
    for (auto& [name, backend] : backends) {
        EXPECT_CALL(*backend, forward(test_tokens))
            .Times(config_.num_iterations)
            .WillRepeatedly(Return(std::vector<float>(32000, 0.6f)));
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < config_.num_iterations; ++i) {
            backend->forward(test_tokens);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double avg_time_ms = static_cast<double>(duration.count()) / config_.num_iterations;
        double tokens_per_second = (config_.sequence_length * 1000.0) / avg_time_ms;
        double power_watts = estimated_power[name];
        
        PowerMetrics metrics;
        metrics.backend_name = name;
        metrics.estimated_power_watts = power_watts;
        metrics.performance_per_watt = tokens_per_second / power_watts;
        metrics.energy_per_token_mj = (power_watts / tokens_per_second) * 1000.0; // mJ per token
        
        power_results.push_back(metrics);
        
        std::cout << name << " power efficiency:" << std::endl;
        std::cout << "  Estimated power: " << power_watts << " W" << std::endl;
        std::cout << "  Performance/Watt: " << metrics.performance_per_watt << " tokens/s/W" << std::endl;
        std::cout << "  Energy per token: " << metrics.energy_per_token_mj << " mJ" << std::endl;
    }
    
    // Sort by efficiency (higher tokens/s/W is better)
    std::sort(power_results.begin(), power_results.end(),
              [](const PowerMetrics& a, const PowerMetrics& b) {
                  return a.performance_per_watt > b.performance_per_watt;
              });
    
    std::cout << "\nPower efficiency ranking:" << std::endl;
    for (size_t i = 0; i < power_results.size(); ++i) {
        std::cout << i + 1 << ". " << power_results[i].backend_name 
                  << " - " << power_results[i].performance_per_watt << " tokens/s/W" << std::endl;
    }
}

// Generate comprehensive benchmark report
TEST_F(BackendBenchmarkTest, GenerateBenchmarkReport) {
    nlohmann::json report;
    report["benchmark_info"] = {
        {"timestamp", "2024-01-01T12:00:00Z"},
        {"version", "1.0.0"},
        {"config", {
            {"sequence_length", config_.sequence_length},
            {"vocab_size", config_.vocab_size},
            {"iterations", config_.num_iterations},
            {"warmup_enabled", config_.warmup_enabled}
        }}
    };
    
    // Run quick benchmark for report
    std::vector<std::pair<std::string, MockBackendBase*>> backends = {
        {"cpu", cpu_backend_.get()},
        {"intel", intel_backend_.get()},
        {"cuda", cuda_backend_.get()}
    };
    
    nlohmann::json backend_results = nlohmann::json::array();
    
    for (auto& [name, backend] : backends) {
        auto result = BackendProfiler::profile_backend(backend, config_);
        backend_results.push_back(result.to_json());
    }
    
    report["results"] = backend_results;
    
    // Save report to file
    std::string report_file = (test_dir_ / "benchmark_report.json").string();
    std::ofstream file(report_file);
    file << report.dump(2);
    file.close();
    
    std::cout << "Benchmark report saved to: " << report_file << std::endl;
    
    // Verify report structure
    EXPECT_TRUE(report.contains("benchmark_info"));
    EXPECT_TRUE(report.contains("results"));
    EXPECT_EQ(report["results"].size(), 3);
    
    // Verify all benchmarks completed successfully
    for (const auto& result : report["results"]) {
        EXPECT_TRUE(result["completed_successfully"]) 
            << "Benchmark failed for " << result["backend_name"];
    }
}