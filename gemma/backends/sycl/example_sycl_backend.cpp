/**
 * @file example_sycl_backend.cpp
 * @brief Example usage of Intel SYCL backend for Gemma.cpp
 *
 * Demonstrates how to:
 * - Initialize the SYCL backend
 * - Detect and select Intel devices
 * - Perform basic matrix operations
 * - Use attention computation
 * - Monitor performance metrics
 */

#include "sycl_backend.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace gemma::backends::sycl;

void PrintDeviceInfo(const SyclDeviceInfo& device) {
    std::string type_str;
    switch (device.type) {
        case SyclDeviceType::GPU: type_str = "Intel GPU"; break;
        case SyclDeviceType::NPU: type_str = "Intel NPU"; break;
        case SyclDeviceType::CPU: type_str = "CPU"; break;
        default: type_str = "Unknown"; break;
    }

    std::cout << "Device " << device.device_id << ": " << device.name << std::endl;
    std::cout << "  Type: " << type_str << std::endl;
    std::cout << "  Vendor: " << device.vendor << std::endl;
    std::cout << "  Memory: " << (device.max_memory_bytes / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Max Work Group Size: " << device.max_work_group_size << std::endl;
    std::cout << "  FP16 Support: " << (device.supports_fp16 ? "Yes" : "No") << std::endl;
    std::cout << "  DP4A Support: " << (device.supports_dp4a ? "Yes" : "No") << std::endl;
    std::cout << "  Unified Memory: " << (device.supports_unified_memory ? "Yes" : "No") << std::endl;
    std::cout << "  Driver Version: " << device.driver_version << std::endl;
}

bool TestMatrixMultiplication(SyclBackend& backend) {
    std::cout << "\n=== Testing Matrix Multiplication ===" << std::endl;

    // Test dimensions
    const int M = 256, N = 256, K = 256;
    const size_t size_a = M * K * sizeof(float);
    const size_t size_b = K * N * sizeof(float);
    const size_t size_c = M * N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(M * K, 1.0f);
    std::vector<float> h_b(K * N, 2.0f);
    std::vector<float> h_c(M * N, 0.0f);

    // Initialize test data
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }

    // Allocate device memory
    auto buf_a = backend.AllocateBuffer(size_a);
    auto buf_b = backend.AllocateBuffer(size_b);
    auto buf_c = backend.AllocateBuffer(size_c);

    if (!buf_a.data || !buf_b.data || !buf_c.data) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return false;
    }

    // Copy data to device
    if (!backend.CopyToDevice(buf_a, h_a.data(), size_a) ||
        !backend.CopyToDevice(buf_b, h_b.data(), size_b)) {
        std::cerr << "Failed to copy data to device" << std::endl;
        return false;
    }

    // Enable profiling
    backend.EnableProfiling(true);

    // Perform matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    bool success = backend.MatrixMultiply(buf_a, buf_b, buf_c, M, N, K, 1.0f, 0.0f);
    backend.Synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        std::cerr << "Matrix multiplication failed" << std::endl;
        return false;
    }

    // Copy result back
    if (!backend.CopyFromDevice(h_c.data(), buf_c, size_c)) {
        std::cerr << "Failed to copy result from device" << std::endl;
        return false;
    }

    // Calculate performance metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double gflops = (2.0 * M * N * K) / (duration.count() * 1e3);  // GFLOPS

    std::cout << "Matrix multiplication " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "Time: " << duration.count() << " μs" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    // Verify result (simple check)
    bool correct = true;
    for (int i = 0; i < std::min(10, M * N); ++i) {
        if (std::abs(h_c[i]) < 1e-6) {  // Result should not be zero for our test data
            correct = false;
            break;
        }
    }

    std::cout << "Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Print performance metrics
    auto metrics = backend.GetMetrics();
    std::cout << "Backend metrics:" << std::endl;
    std::cout << "  Compute throughput: " << metrics.compute_throughput_gflops << " GFLOPS" << std::endl;
    std::cout << "  Memory bandwidth: " << metrics.memory_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "  Latency: " << metrics.latency_ms << " ms" << std::endl;

    // Clean up
    backend.FreeBuffer(buf_a);
    backend.FreeBuffer(buf_b);
    backend.FreeBuffer(buf_c);

    return correct;
}

bool TestAttentionComputation(SyclBackend& backend) {
    std::cout << "\n=== Testing Attention Computation ===" << std::endl;

    // Test dimensions
    const int batch_size = 1;
    const int seq_len = 128;
    const int head_dim = 64;
    const int num_heads = 8;

    const size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);

    // Allocate host memory
    std::vector<float> h_queries(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> h_keys(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> h_values(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> h_output(batch_size * num_heads * seq_len * head_dim, 0.0f);

    // Initialize with random-like data
    for (size_t i = 0; i < h_queries.size(); ++i) {
        h_queries[i] = static_cast<float>(i % 1000) / 1000.0f - 0.5f;
        h_keys[i] = static_cast<float>((i + 333) % 1000) / 1000.0f - 0.5f;
        h_values[i] = static_cast<float>((i + 666) % 1000) / 1000.0f - 0.5f;
    }

    // Allocate device memory
    auto buf_q = backend.AllocateBuffer(qkv_size);
    auto buf_k = backend.AllocateBuffer(qkv_size);
    auto buf_v = backend.AllocateBuffer(qkv_size);
    auto buf_o = backend.AllocateBuffer(qkv_size);

    if (!buf_q.data || !buf_k.data || !buf_v.data || !buf_o.data) {
        std::cerr << "Failed to allocate device memory for attention" << std::endl;
        return false;
    }

    // Copy data to device
    if (!backend.CopyToDevice(buf_q, h_queries.data(), qkv_size) ||
        !backend.CopyToDevice(buf_k, h_keys.data(), qkv_size) ||
        !backend.CopyToDevice(buf_v, h_values.data(), qkv_size)) {
        std::cerr << "Failed to copy attention data to device" << std::endl;
        return false;
    }

    // Perform attention computation
    auto start = std::chrono::high_resolution_clock::now();
    bool success = backend.ComputeAttention(buf_q, buf_k, buf_v, buf_o,
                                           batch_size, seq_len, head_dim, num_heads);
    backend.Synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        std::cerr << "Attention computation failed" << std::endl;
        return false;
    }

    // Copy result back
    if (!backend.CopyFromDevice(h_output.data(), buf_o, qkv_size)) {
        std::cerr << "Failed to copy attention result from device" << std::endl;
        return false;
    }

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Attention computation [" << batch_size << ", " << num_heads
              << ", " << seq_len << ", " << head_dim << "]" << std::endl;
    std::cout << "Time: " << duration.count() << " μs" << std::endl;

    // Verify result (basic sanity check)
    bool correct = true;
    double sum = 0.0;
    for (size_t i = 0; i < h_output.size(); ++i) {
        sum += std::abs(h_output[i]);
        if (std::isnan(h_output[i]) || std::isinf(h_output[i])) {
            correct = false;
            break;
        }
    }

    std::cout << "Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Average absolute value: " << (sum / h_output.size()) << std::endl;

    // Clean up
    backend.FreeBuffer(buf_q);
    backend.FreeBuffer(buf_k);
    backend.FreeBuffer(buf_v);
    backend.FreeBuffer(buf_o);

    return correct;
}

int main() {
    std::cout << "Intel SYCL Backend Example for Gemma.cpp" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Check if SYCL backend is available
    if (!IsSyclBackendAvailable()) {
        std::cerr << "SYCL backend is not available on this system." << std::endl;
        std::cerr << "Please install Intel oneAPI toolkit and ensure compatible hardware." << std::endl;
        return 1;
    }

    std::cout << "SYCL backend is available!" << std::endl;
    std::cout << "oneAPI Version: " << GetOneAPIVersion() << std::endl;

    // Create and initialize backend
    auto backend = std::make_unique<SyclBackend>();
    if (!backend->Initialize()) {
        std::cerr << "Failed to initialize SYCL backend" << std::endl;
        return 1;
    }

    std::cout << "\nBackend: " << backend->GetName() << " v" << backend->GetVersion() << std::endl;

    // Display available devices
    auto devices = backend->GetAvailableDevices();
    std::cout << "\nAvailable devices (" << devices.size() << "):" << std::endl;
    for (const auto& device : devices) {
        PrintDeviceInfo(device);
        std::cout << std::endl;
    }

    // Select best device (automatic)
    std::cout << "Selected device: " << backend->GetCurrentDeviceInfo().name << std::endl;

    // Test capabilities
    std::cout << "\nTesting backend capabilities:" << std::endl;
    std::cout << "  Matrix multiplication: "
              << (backend->SupportsCapability(gemma::backends::BackendCapability::MATRIX_MULTIPLICATION) ? "Yes" : "No") << std::endl;
    std::cout << "  Attention computation: "
              << (backend->SupportsCapability(gemma::backends::BackendCapability::ATTENTION_COMPUTATION) ? "Yes" : "No") << std::endl;
    std::cout << "  Multi-precision: "
              << (backend->SupportsCapability(gemma::backends::BackendCapability::MULTI_PRECISION) ? "Yes" : "No") << std::endl;

    // Run performance tests
    bool all_tests_passed = true;

    if (backend->SupportsCapability(gemma::backends::BackendCapability::MATRIX_MULTIPLICATION)) {
        if (!TestMatrixMultiplication(*backend)) {
            all_tests_passed = false;
        }
    }

    if (backend->SupportsCapability(gemma::backends::BackendCapability::ATTENTION_COMPUTATION)) {
        if (!TestAttentionComputation(*backend)) {
            all_tests_passed = false;
        }
    }

    // Display memory statistics
    auto memory_stats = backend->GetMemoryStats();
    std::cout << "\n=== Memory Statistics ===" << std::endl;
    for (const auto& [key, value] : memory_stats) {
        std::cout << "  " << key << ": " << (value / (1024*1024)) << " MB" << std::endl;
    }

    // Display profiling data if available
    auto profiling_data = backend->GetProfilingData();
    if (!profiling_data.empty()) {
        std::cout << "\n=== Profiling Data ===" << std::endl;
        for (const auto& profile : profiling_data) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                profile.end_time - profile.start_time).count();
            std::cout << "  " << profile.operation_name << ": " << duration << " μs";
            if (profile.flops_performed > 0) {
                double gflops = (profile.flops_performed / 1e9) / (duration / 1e6);
                std::cout << " (" << std::fixed << std::setprecision(2) << gflops << " GFLOPS)";
            }
            std::cout << std::endl;
        }
    }

    // Shutdown
    backend->Shutdown();

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All tests " << (all_tests_passed ? "PASSED" : "FAILED") << std::endl;

    return all_tests_passed ? 0 : 1;
}