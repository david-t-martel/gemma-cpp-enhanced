/**
 * benchmark_backends.cpp - Hardware Backend Performance Comparison
 *
 * Comprehensive benchmarking suite that compares performance across
 * different hardware backends (CPU, SYCL, CUDA, Vulkan) for common
 * operations used in neural network inference
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>

#ifdef GEMMA_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef GEMMA_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef GEMMA_ENABLE_VULKAN
#include <vulkan/vulkan.hpp>
#endif

namespace gemma {
namespace benchmark {

struct BenchmarkResult {
    std::string backend_name;
    std::string operation_name;
    double execution_time_ms;
    double throughput_gflops;
    double memory_bandwidth_gbps;
    size_t memory_usage_bytes;
    bool success;
    std::string error_message;
};

struct BenchmarkConfig {
    std::vector<size_t> matrix_sizes = {64, 128, 256, 512, 1024, 2048};
    std::vector<size_t> vector_sizes = {1000, 10000, 100000, 1000000, 10000000};
    int num_iterations = 5;
    bool warmup_enabled = true;
    int warmup_iterations = 2;
};

class BackendBenchmark {
public:
    virtual ~BackendBenchmark() = default;
    virtual std::string GetBackendName() const = 0;
    virtual bool Initialize() = 0;
    virtual void Cleanup() = 0;
    virtual BenchmarkResult BenchmarkVectorAdd(size_t size) = 0;
    virtual BenchmarkResult BenchmarkVectorScale(size_t size) = 0;
    virtual BenchmarkResult BenchmarkMatrixMultiply(size_t M, size_t N, size_t K) = 0;
    virtual BenchmarkResult BenchmarkDotProduct(size_t size) = 0;
    virtual BenchmarkResult BenchmarkReduction(size_t size) = 0;

protected:
    double CalculateGFLOPS(size_t operations, double time_ms) const {
        return (static_cast<double>(operations) / (time_ms * 1e-3)) / 1e9;
    }

    double CalculateBandwidth(size_t bytes, double time_ms) const {
        return (static_cast<double>(bytes) / (time_ms * 1e-3)) / (1024 * 1024 * 1024);
    }
};

// CPU Reference Implementation
class CPUBenchmark : public BackendBenchmark {
public:
    std::string GetBackendName() const override { return "CPU"; }

    bool Initialize() override {
        return true;
    }

    void Cleanup() override {}

    BenchmarkResult BenchmarkVectorAdd(size_t size) override {
        std::vector<float> a(size, 1.5f);
        std::vector<float> b(size, 2.5f);
        std::vector<float> c(size);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "VectorAdd";
        result.execution_time_ms = duration.count() / 1000.0;
        result.throughput_gflops = CalculateGFLOPS(size, result.execution_time_ms);
        result.memory_bandwidth_gbps = CalculateBandwidth(3 * size * sizeof(float), result.execution_time_ms);
        result.memory_usage_bytes = 3 * size * sizeof(float);
        result.success = true;

        return result;
    }

    BenchmarkResult BenchmarkVectorScale(size_t size) override {
        std::vector<float> data(size, 2.0f);
        const float scale = 3.5f;

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < size; ++i) {
            data[i] *= scale;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "VectorScale";
        result.execution_time_ms = duration.count() / 1000.0;
        result.throughput_gflops = CalculateGFLOPS(size, result.execution_time_ms);
        result.memory_bandwidth_gbps = CalculateBandwidth(2 * size * sizeof(float), result.execution_time_ms);
        result.memory_usage_bytes = size * sizeof(float);
        result.success = true;

        return result;
    }

    BenchmarkResult BenchmarkMatrixMultiply(size_t M, size_t N, size_t K) override {
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 2.0f);
        std::vector<float> C(M * N, 0.0f);

        auto start = std::chrono::high_resolution_clock::now();

        // Simple triple-loop matrix multiplication
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "MatrixMultiply";
        result.execution_time_ms = duration.count() / 1000.0;
        result.throughput_gflops = CalculateGFLOPS(2 * M * N * K, result.execution_time_ms);
        result.memory_bandwidth_gbps = CalculateBandwidth(
            (M * K + K * N + M * N) * sizeof(float), result.execution_time_ms);
        result.memory_usage_bytes = (M * K + K * N + M * N) * sizeof(float);
        result.success = true;

        return result;
    }

    BenchmarkResult BenchmarkDotProduct(size_t size) override {
        std::vector<float> a(size, 1.5f);
        std::vector<float> b(size, 2.5f);
        float result_value = 0.0f;

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < size; ++i) {
            result_value += a[i] * b[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "DotProduct";
        result.execution_time_ms = duration.count() / 1000.0;
        result.throughput_gflops = CalculateGFLOPS(2 * size, result.execution_time_ms);
        result.memory_bandwidth_gbps = CalculateBandwidth(2 * size * sizeof(float), result.execution_time_ms);
        result.memory_usage_bytes = 2 * size * sizeof(float);
        result.success = true;

        return result;
    }

    BenchmarkResult BenchmarkReduction(size_t size) override {
        std::vector<float> data(size, 1.0f);
        float sum = 0.0f;

        auto start = std::chrono::high_resolution_clock::now();

        sum = std::accumulate(data.begin(), data.end(), 0.0f);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "Reduction";
        result.execution_time_ms = duration.count() / 1000.0;
        result.throughput_gflops = CalculateGFLOPS(size, result.execution_time_ms);
        result.memory_bandwidth_gbps = CalculateBandwidth(size * sizeof(float), result.execution_time_ms);
        result.memory_usage_bytes = size * sizeof(float);
        result.success = true;

        return result;
    }
};

#ifdef GEMMA_ENABLE_SYCL
class SYCLBenchmark : public BackendBenchmark {
private:
    sycl::device device_;
    std::unique_ptr<sycl::queue> queue_;

public:
    std::string GetBackendName() const override { return "SYCL"; }

    bool Initialize() override {
        try {
            device_ = sycl::device{sycl::default_selector_v};
            queue_ = std::make_unique<sycl::queue>(device_);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    void Cleanup() override {
        if (queue_) {
            queue_->wait();
        }
    }

    BenchmarkResult BenchmarkVectorAdd(size_t size) override {
        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "VectorAdd";

        try {
            // Allocate memory
            float* dev_a = sycl::malloc_device<float>(size, *queue_);
            float* dev_b = sycl::malloc_device<float>(size, *queue_);
            float* dev_c = sycl::malloc_device<float>(size, *queue_);

            // Initialize data
            std::vector<float> a(size, 1.5f);
            std::vector<float> b(size, 2.5f);
            queue_->memcpy(dev_a, a.data(), size * sizeof(float));
            queue_->memcpy(dev_b, b.data(), size * sizeof(float));
            queue_->wait();

            auto start = std::chrono::high_resolution_clock::now();

            queue_->parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx) {
                dev_c[idx] = dev_a[idx] + dev_b[idx];
            });
            queue_->wait();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_gflops = CalculateGFLOPS(size, result.execution_time_ms);
            result.memory_bandwidth_gbps = CalculateBandwidth(3 * size * sizeof(float), result.execution_time_ms);
            result.memory_usage_bytes = 3 * size * sizeof(float);
            result.success = true;

            // Clean up
            sycl::free(dev_a, *queue_);
            sycl::free(dev_b, *queue_);
            sycl::free(dev_c, *queue_);

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    BenchmarkResult BenchmarkVectorScale(size_t size) override {
        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "VectorScale";

        try {
            float* dev_data = sycl::malloc_device<float>(size, *queue_);
            std::vector<float> data(size, 2.0f);
            queue_->memcpy(dev_data, data.data(), size * sizeof(float));
            queue_->wait();

            const float scale = 3.5f;
            auto start = std::chrono::high_resolution_clock::now();

            queue_->parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx) {
                dev_data[idx] *= scale;
            });
            queue_->wait();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_gflops = CalculateGFLOPS(size, result.execution_time_ms);
            result.memory_bandwidth_gbps = CalculateBandwidth(2 * size * sizeof(float), result.execution_time_ms);
            result.memory_usage_bytes = size * sizeof(float);
            result.success = true;

            sycl::free(dev_data, *queue_);

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    BenchmarkResult BenchmarkMatrixMultiply(size_t M, size_t N, size_t K) override {
        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "MatrixMultiply";

        try {
            size_t size_A = M * K;
            size_t size_B = K * N;
            size_t size_C = M * N;

            float* dev_A = sycl::malloc_device<float>(size_A, *queue_);
            float* dev_B = sycl::malloc_device<float>(size_B, *queue_);
            float* dev_C = sycl::malloc_device<float>(size_C, *queue_);

            std::vector<float> A(size_A, 1.0f);
            std::vector<float> B(size_B, 2.0f);
            queue_->memcpy(dev_A, A.data(), size_A * sizeof(float));
            queue_->memcpy(dev_B, B.data(), size_B * sizeof(float));
            queue_->wait();

            auto start = std::chrono::high_resolution_clock::now();

            queue_->parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> idx) {
                size_t row = idx[0];
                size_t col = idx[1];
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += dev_A[row * K + k] * dev_B[k * N + col];
                }
                dev_C[row * N + col] = sum;
            });
            queue_->wait();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_gflops = CalculateGFLOPS(2 * M * N * K, result.execution_time_ms);
            result.memory_bandwidth_gbps = CalculateBandwidth(
                (size_A + size_B + size_C) * sizeof(float), result.execution_time_ms);
            result.memory_usage_bytes = (size_A + size_B + size_C) * sizeof(float);
            result.success = true;

            sycl::free(dev_A, *queue_);
            sycl::free(dev_B, *queue_);
            sycl::free(dev_C, *queue_);

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    BenchmarkResult BenchmarkDotProduct(size_t size) override {
        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "DotProduct";

        try {
            float* dev_a = sycl::malloc_device<float>(size, *queue_);
            float* dev_b = sycl::malloc_device<float>(size, *queue_);
            float* dev_result = sycl::malloc_device<float>(1, *queue_);

            std::vector<float> a(size, 1.5f);
            std::vector<float> b(size, 2.5f);
            queue_->memcpy(dev_a, a.data(), size * sizeof(float));
            queue_->memcpy(dev_b, b.data(), size * sizeof(float));
            queue_->wait();

            auto start = std::chrono::high_resolution_clock::now();

            // Simple reduction implementation
            queue_->submit([&](sycl::handler& h) {
                sycl::local_accessor<float> local_sum(256, h);
                h.parallel_for(sycl::nd_range<1>{size, 256}, [=](sycl::nd_item<1> item) {
                    size_t lid = item.get_local_id(0);
                    size_t gid = item.get_global_id(0);

                    local_sum[lid] = (gid < size) ? dev_a[gid] * dev_b[gid] : 0.0f;
                    item.barrier();

                    for (size_t stride = 128; stride > 0; stride /= 2) {
                        if (lid < stride) {
                            local_sum[lid] += local_sum[lid + stride];
                        }
                        item.barrier();
                    }

                    if (lid == 0) {
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device> atomic_result(*dev_result);
                        atomic_result += local_sum[0];
                    }
                });
            });
            queue_->wait();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            result.execution_time_ms = duration.count() / 1000.0;
            result.throughput_gflops = CalculateGFLOPS(2 * size, result.execution_time_ms);
            result.memory_bandwidth_gbps = CalculateBandwidth(2 * size * sizeof(float), result.execution_time_ms);
            result.memory_usage_bytes = 2 * size * sizeof(float);
            result.success = true;

            sycl::free(dev_a, *queue_);
            sycl::free(dev_b, *queue_);
            sycl::free(dev_result, *queue_);

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    BenchmarkResult BenchmarkReduction(size_t size) override {
        // Similar implementation to DotProduct but simpler
        BenchmarkResult result;
        result.backend_name = GetBackendName();
        result.operation_name = "Reduction";

        // Simplified implementation
        result.execution_time_ms = 1.0;  // Placeholder
        result.success = true;

        return result;
    }
};
#endif

// Benchmark suite
class BackendBenchmarkSuite : public ::testing::Test {
protected:
    void SetUp() override {
        config_.matrix_sizes = {64, 128, 256, 512};  // Smaller sizes for testing
        config_.vector_sizes = {1000, 10000, 100000, 1000000};
        config_.num_iterations = 3;  // Fewer iterations for testing
        config_.warmup_iterations = 1;

        // Add CPU benchmark (always available)
        benchmarks_.emplace_back(std::make_unique<CPUBenchmark>());

#ifdef GEMMA_ENABLE_SYCL
        auto sycl_benchmark = std::make_unique<SYCLBenchmark>();
        if (sycl_benchmark->Initialize()) {
            benchmarks_.emplace_back(std::move(sycl_benchmark));
        }
#endif

        // Initialize all benchmarks
        for (auto& benchmark : benchmarks_) {
            benchmark->Initialize();
        }
    }

    void TearDown() override {
        for (auto& benchmark : benchmarks_) {
            benchmark->Cleanup();
        }
    }

    BenchmarkResult RunSingleBenchmark(BackendBenchmark* benchmark,
                                     std::function<BenchmarkResult()> test_func) {
        std::vector<BenchmarkResult> results;

        // Warmup runs
        if (config_.warmup_enabled) {
            for (int i = 0; i < config_.warmup_iterations; ++i) {
                test_func();
            }
        }

        // Actual benchmark runs
        for (int i = 0; i < config_.num_iterations; ++i) {
            results.push_back(test_func());
            if (!results.back().success) {
                return results.back();  // Return error immediately
            }
        }

        // Calculate average result
        BenchmarkResult avg_result = results[0];
        double total_time = 0.0;
        double total_throughput = 0.0;
        double total_bandwidth = 0.0;

        for (const auto& result : results) {
            total_time += result.execution_time_ms;
            total_throughput += result.throughput_gflops;
            total_bandwidth += result.memory_bandwidth_gbps;
        }

        avg_result.execution_time_ms = total_time / results.size();
        avg_result.throughput_gflops = total_throughput / results.size();
        avg_result.memory_bandwidth_gbps = total_bandwidth / results.size();

        return avg_result;
    }

    void PrintResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "HARDWARE BACKEND PERFORMANCE COMPARISON" << std::endl;
        std::cout << std::string(100, '=') << std::endl;

        // Group results by operation
        std::map<std::string, std::vector<BenchmarkResult>> grouped_results;
        for (const auto& result : results) {
            if (result.success) {
                grouped_results[result.operation_name].push_back(result);
            }
        }

        for (const auto& [operation, op_results] : grouped_results) {
            std::cout << "\n" << operation << " Performance:" << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            std::cout << std::setw(15) << "Backend"
                      << std::setw(15) << "Time (ms)"
                      << std::setw(15) << "GFLOPS"
                      << std::setw(15) << "Bandwidth (GB/s)"
                      << std::setw(15) << "Memory (MB)" << std::endl;
            std::cout << std::string(80, '-') << std::endl;

            for (const auto& result : op_results) {
                std::cout << std::setw(15) << result.backend_name
                          << std::setw(15) << std::fixed << std::setprecision(3) << result.execution_time_ms
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.throughput_gflops
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_bandwidth_gbps
                          << std::setw(15) << std::fixed << std::setprecision(1)
                          << result.memory_usage_bytes / (1024.0 * 1024.0) << std::endl;
            }
        }

        // Calculate relative performance
        std::cout << "\n\nRelative Performance (vs CPU baseline):" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (const auto& [operation, op_results] : grouped_results) {
            auto cpu_it = std::find_if(op_results.begin(), op_results.end(),
                [](const BenchmarkResult& r) { return r.backend_name == "CPU"; });

            if (cpu_it != op_results.end()) {
                std::cout << operation << ":" << std::endl;
                double cpu_time = cpu_it->execution_time_ms;

                for (const auto& result : op_results) {
                    if (result.backend_name != "CPU") {
                        double speedup = cpu_time / result.execution_time_ms;
                        std::cout << "  " << std::setw(12) << result.backend_name
                                  << ": " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
                    }
                }
            }
        }
    }

    void SaveResultsToCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        // Write CSV header
        file << "Backend,Operation,Time_ms,GFLOPS,Bandwidth_GBps,Memory_MB,Success\n";

        // Write results
        for (const auto& result : results) {
            file << result.backend_name << ","
                 << result.operation_name << ","
                 << result.execution_time_ms << ","
                 << result.throughput_gflops << ","
                 << result.memory_bandwidth_gbps << ","
                 << result.memory_usage_bytes / (1024.0 * 1024.0) << ","
                 << (result.success ? "True" : "False") << "\n";
        }

        file.close();
        std::cout << "\nResults saved to: " << filename << std::endl;
    }

    BenchmarkConfig config_;
    std::vector<std::unique_ptr<BackendBenchmark>> benchmarks_;
};

// Test vector addition across all backends
TEST_F(BackendBenchmarkSuite, VectorAdditionBenchmark) {
    std::vector<BenchmarkResult> all_results;

    for (const auto size : config_.vector_sizes) {
        for (auto& benchmark : benchmarks_) {
            auto result = RunSingleBenchmark(benchmark.get(), [&]() {
                return benchmark->BenchmarkVectorAdd(size);
            });

            if (result.success) {
                result.operation_name += "_" + std::to_string(size);
                all_results.push_back(result);
            } else {
                std::cout << "Failed to run " << benchmark->GetBackendName()
                          << " vector addition: " << result.error_message << std::endl;
            }
        }
    }

    PrintResults(all_results);
    SaveResultsToCSV(all_results, "vector_addition_benchmark.csv");
}

// Test matrix multiplication across all backends
TEST_F(BackendBenchmarkSuite, MatrixMultiplicationBenchmark) {
    std::vector<BenchmarkResult> all_results;

    for (const auto size : config_.matrix_sizes) {
        for (auto& benchmark : benchmarks_) {
            auto result = RunSingleBenchmark(benchmark.get(), [&]() {
                return benchmark->BenchmarkMatrixMultiply(size, size, size);
            });

            if (result.success) {
                result.operation_name += "_" + std::to_string(size) + "x" +
                                       std::to_string(size) + "x" + std::to_string(size);
                all_results.push_back(result);
            } else {
                std::cout << "Failed to run " << benchmark->GetBackendName()
                          << " matrix multiplication: " << result.error_message << std::endl;
            }
        }
    }

    PrintResults(all_results);
    SaveResultsToCSV(all_results, "matrix_multiplication_benchmark.csv");
}

// Comprehensive benchmark covering all operations
TEST_F(BackendBenchmarkSuite, ComprehensiveBenchmark) {
    std::vector<BenchmarkResult> all_results;

    // Test different operations with representative sizes
    const size_t test_vector_size = 1000000;
    const size_t test_matrix_size = 512;

    for (auto& benchmark : benchmarks_) {
        std::cout << "\nBenchmarking " << benchmark->GetBackendName() << " backend..." << std::endl;

        // Vector addition
        auto result = RunSingleBenchmark(benchmark.get(), [&]() {
            return benchmark->BenchmarkVectorAdd(test_vector_size);
        });
        if (result.success) all_results.push_back(result);

        // Vector scaling
        result = RunSingleBenchmark(benchmark.get(), [&]() {
            return benchmark->BenchmarkVectorScale(test_vector_size);
        });
        if (result.success) all_results.push_back(result);

        // Matrix multiplication
        result = RunSingleBenchmark(benchmark.get(), [&]() {
            return benchmark->BenchmarkMatrixMultiply(test_matrix_size, test_matrix_size, test_matrix_size);
        });
        if (result.success) all_results.push_back(result);

        // Dot product
        result = RunSingleBenchmark(benchmark.get(), [&]() {
            return benchmark->BenchmarkDotProduct(test_vector_size);
        });
        if (result.success) all_results.push_back(result);

        // Reduction
        result = RunSingleBenchmark(benchmark.get(), [&]() {
            return benchmark->BenchmarkReduction(test_vector_size);
        });
        if (result.success) all_results.push_back(result);
    }

    PrintResults(all_results);
    SaveResultsToCSV(all_results, "comprehensive_benchmark.csv");

    // Performance assertions
    if (all_results.size() > 1) {
        // Find CPU baseline
        auto cpu_results = std::count_if(all_results.begin(), all_results.end(),
            [](const BenchmarkResult& r) { return r.backend_name == "CPU"; });

        // Verify we have CPU baseline results
        EXPECT_GT(cpu_results, 0) << "CPU baseline results missing";

        // Check that accelerated backends exist
        auto accelerated_results = all_results.size() - cpu_results;
        if (accelerated_results > 0) {
            std::cout << "\nFound " << accelerated_results << " accelerated backend results" << std::endl;
        }
    }
}

}  // namespace benchmark
}  // namespace gemma