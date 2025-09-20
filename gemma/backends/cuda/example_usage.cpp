/**
 * @file example_usage.cpp
 * @brief Example demonstrating CUDA backend usage in Gemma.cpp
 */

#include "cuda_backend.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

using namespace gemma::backends::cuda;

void PrintDeviceInfo(const CudaBackend& backend) {
    std::cout << "\n=== CUDA Device Information ===" << std::endl;

    auto devices = backend.GetAllDeviceInfo();
    for (const auto& device : devices) {
        std::cout << "Device " << device.device_id << ": " << device.name << std::endl;
        std::cout << "  Compute Capability: " << device.compute_capability_major
                  << "." << device.compute_capability_minor << std::endl;
        std::cout << "  Total Memory: " << (device.total_memory / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Free Memory: " << (device.free_memory / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Tensor Cores: " << (device.supports_tensor_cores ? "Yes" : "No") << std::endl;
        std::cout << "  FP16 Support: " << (device.supports_fp16 ? "Yes" : "No") << std::endl;
        std::cout << "  BF16 Support: " << (device.supports_bf16 ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }
}

void DemoBasicOperations() {
    std::cout << "\n=== Basic CUDA Operations Demo ===" << std::endl;

    // Create CUDA backend with default configuration
    auto backend = CreateCudaBackend();

    if (!backend->Initialize()) {
        std::cerr << "Failed to initialize CUDA backend!" << std::endl;
        return;
    }

    PrintDeviceInfo(*static_cast<CudaBackend*>(backend.get()));

    // Test parameters
    const size_t size = 1024 * 1024; // 1M elements
    const size_t bytes = size * sizeof(float);

    // Allocate GPU buffers
    auto input_buffer = backend->AllocateBuffer(bytes);
    auto output_buffer = backend->AllocateBuffer(bytes);

    if (!input_buffer.data || !output_buffer.data) {
        std::cerr << "Failed to allocate GPU memory!" << std::endl;
        return;
    }

    // Create test data on CPU
    std::vector<float> input_data(size);
    std::vector<float> output_data(size);

    for (size_t i = 0; i < size; ++i) {
        input_data[i] = static_cast<float>(i) / size - 0.5f; // Values from -0.5 to 0.5
    }

    // Copy data to GPU
    if (!backend->CopyToDevice(input_buffer, input_data.data(), bytes)) {
        std::cerr << "Failed to copy data to device!" << std::endl;
        return;
    }

    std::cout << "Testing activation functions..." << std::endl;

    // Test ReLU
    auto start = std::chrono::high_resolution_clock::now();
    bool success = backend->ApplyReLU(input_buffer, output_buffer, size);
    auto end = std::chrono::high_resolution_clock::now();

    if (success) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  ReLU: " << duration.count() << " μs" << std::endl;
    } else {
        std::cerr << "  ReLU failed!" << std::endl;
    }

    // Test GELU
    start = std::chrono::high_resolution_clock::now();
    success = backend->ApplyGELU(input_buffer, output_buffer, size);
    end = std::chrono::high_resolution_clock::now();

    if (success) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  GELU: " << duration.count() << " μs" << std::endl;
    } else {
        std::cerr << "  GELU failed!" << std::endl;
    }

    // Copy result back to CPU for verification
    if (backend->CopyFromDevice(output_data.data(), output_buffer, bytes)) {
        // Verify a few values
        std::cout << "  Sample outputs: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << output_data[i * (size / 5)] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    backend->FreeBuffer(input_buffer);
    backend->FreeBuffer(output_buffer);

    std::cout << "Basic operations demo completed!" << std::endl;
}

void DemoMatrixMultiplication() {
    std::cout << "\n=== Matrix Multiplication Demo ===" << std::endl;

    // Create CUDA backend
    auto backend = CreateCudaBackend();

    if (!backend->Initialize()) {
        std::cerr << "Failed to initialize CUDA backend!" << std::endl;
        return;
    }

    // Matrix dimensions
    const int M = 512, N = 512, K = 512;
    const size_t size_a = M * K * sizeof(float);
    const size_t size_b = K * N * sizeof(float);
    const size_t size_c = M * N * sizeof(float);

    // Allocate GPU buffers
    auto buffer_a = backend->AllocateBuffer(size_a);
    auto buffer_b = backend->AllocateBuffer(size_b);
    auto buffer_c = backend->AllocateBuffer(size_c);

    if (!buffer_a.data || !buffer_b.data || !buffer_c.data) {
        std::cerr << "Failed to allocate GPU memory!" << std::endl;
        return;
    }

    // Initialize matrices with test data
    std::vector<float> matrix_a(M * K, 1.0f);
    std::vector<float> matrix_b(K * N, 2.0f);
    std::vector<float> matrix_c(M * N, 0.0f);

    // Copy data to GPU
    backend->CopyToDevice(buffer_a, matrix_a.data(), size_a);
    backend->CopyToDevice(buffer_b, matrix_b.data(), size_b);
    backend->CopyToDevice(buffer_c, matrix_c.data(), size_c);

    // Perform matrix multiplication: C = A * B
    std::cout << "Performing " << M << "x" << K << " * " << K << "x" << N
              << " matrix multiplication..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    bool success = backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, M, N, K);
    backend->Synchronize(); // Wait for completion
    auto end = std::chrono::high_resolution_clock::now();

    if (success) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double gflops = (2.0 * M * N * K) / (duration.count() / 1000.0) / 1e9;

        std::cout << "  Execution time: " << duration.count() << " μs" << std::endl;
        std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;

        // Verify result (should be K * 1.0 * 2.0 = 2*K for each element)
        std::vector<float> result(M * N);
        backend->CopyFromDevice(result.data(), buffer_c, size_c);

        float expected = static_cast<float>(2 * K);
        bool correct = true;
        for (int i = 0; i < std::min(100, M * N); ++i) {
            if (std::abs(result[i] - expected) > 1e-5f) {
                correct = false;
                break;
            }
        }

        std::cout << "  Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
        if (correct) {
            std::cout << "  Expected value: " << expected
                      << ", Got: " << result[0] << std::endl;
        }
    } else {
        std::cerr << "  Matrix multiplication failed!" << std::endl;
    }

    // Clean up
    backend->FreeBuffer(buffer_a);
    backend->FreeBuffer(buffer_b);
    backend->FreeBuffer(buffer_c);
}

void DemoAdvancedFeatures() {
    std::cout << "\n=== Advanced Features Demo ===" << std::endl;

    // Create CUDA backend with advanced configuration
    CudaConfig config;
    config.default_precision = CudaPrecision::FP16;
    config.enable_tensor_cores = true;
    config.enable_flash_attention = true;
    config.memory_fraction = 0.8;

    auto cuda_backend = std::make_unique<CudaBackend>(config);

    if (!cuda_backend->Initialize()) {
        std::cerr << "Failed to initialize CUDA backend!" << std::endl;
        return;
    }

    std::cout << "Backend configuration:" << std::endl;
    std::cout << "  Default precision: FP16" << std::endl;
    std::cout << "  Tensor Cores: " << (config.enable_tensor_cores ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Flash Attention: " << (config.enable_flash_attention ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Memory fraction: " << config.memory_fraction << std::endl;

    // Test multi-device support if available
    int device_count = cuda_backend->GetDeviceCount();
    std::cout << "  Available devices: " << device_count << std::endl;

    if (device_count > 1) {
        std::cout << "Testing multi-GPU operations..." << std::endl;

        // Enable peer access
        bool peer_access = cuda_backend->EnablePeerAccess();
        std::cout << "  Peer access: " << (peer_access ? "Enabled" : "Not available") << std::endl;

        // Test device-to-device copy
        const size_t test_size = 1024 * 1024 * sizeof(float);
        auto buffer_gpu0 = cuda_backend->AllocateCudaBuffer(test_size, CudaPrecision::FP32, 0);
        auto buffer_gpu1 = cuda_backend->AllocateCudaBuffer(test_size, CudaPrecision::FP32, 1);

        if (buffer_gpu0.data && buffer_gpu1.data) {
            auto start = std::chrono::high_resolution_clock::now();
            bool success = cuda_backend->CopyDeviceToDevice(buffer_gpu1, buffer_gpu0, test_size);
            cuda_backend->Synchronize();
            auto end = std::chrono::high_resolution_clock::now();

            if (success) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double bandwidth = (test_size / 1024.0 / 1024.0 / 1024.0) / (duration.count() / 1e6);
                std::cout << "  GPU-to-GPU bandwidth: " << bandwidth << " GB/s" << std::endl;
            }
        }

        cuda_backend->FreeBuffer(buffer_gpu0);
        cuda_backend->FreeBuffer(buffer_gpu1);
    }

    // Show performance metrics
    auto metrics = cuda_backend->GetMetrics();
    std::cout << "\nPerformance metrics:" << std::endl;
    std::cout << "  Memory usage: " << (metrics.memory_usage_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Peak memory: " << (metrics.peak_memory_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Total latency: " << metrics.latency_ms << " ms" << std::endl;
}

int main() {
    std::cout << "=== Gemma.cpp CUDA Backend Demo ===" << std::endl;

    // Check if CUDA is available
    if (!IsCudaAvailable()) {
        std::cerr << "CUDA is not available on this system!" << std::endl;
        return 1;
    }

    std::cout << "CUDA Version: " << GetCudaVersion() << std::endl;

    try {
        // Run demonstrations
        DemoBasicOperations();
        DemoMatrixMultiplication();
        DemoAdvancedFeatures();

        std::cout << "\n=== All demos completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during demo: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}