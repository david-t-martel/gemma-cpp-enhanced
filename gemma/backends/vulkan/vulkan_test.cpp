#include "vulkan_backend.h"
#include "vulkan_utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>

using namespace gemma::backends;
using namespace gemma::backends::vulkan;

/**
 * Test program for Vulkan backend functionality
 */
class VulkanBackendTest {
public:
    VulkanBackendTest() : backend_(std::make_unique<VulkanBackend>()) {}

    bool RunAllTests() {
        std::cout << "=== Vulkan Backend Test Suite ===" << std::endl;

        if (!InitializeBackend()) {
            std::cerr << "Failed to initialize Vulkan backend" << std::endl;
            return false;
        }

        bool all_passed = true;

        all_passed &= TestDeviceInfo();
        all_passed &= TestBufferOperations();
        all_passed &= TestMatrixMultiplication();
        all_passed &= TestActivationFunctions();
        all_passed &= TestPerformance();

        std::cout << std::endl;
        std::cout << "=== Test Summary ===" << std::endl;
        std::cout << "Overall result: " << (all_passed ? "PASSED" : "FAILED") << std::endl;

        return all_passed;
    }

private:
    std::unique_ptr<VulkanBackend> backend_;

    bool InitializeBackend() {
        std::cout << "\n--- Initializing Vulkan Backend ---" << std::endl;

        if (!utils::IsVulkanAvailable()) {
            std::cerr << "Vulkan is not available on this system" << std::endl;
            return false;
        }

        if (!backend_->Initialize()) {
            std::cerr << "Failed to initialize Vulkan backend" << std::endl;
            return false;
        }

        std::cout << "Backend: " << backend_->GetName() << " v" << backend_->GetVersion() << std::endl;
        std::cout << "Device count: " << backend_->GetDeviceCount() << std::endl;
        std::cout << "Current device: " << backend_->GetCurrentDevice() << std::endl;

        return true;
    }

    bool TestDeviceInfo() {
        std::cout << "\n--- Testing Device Information ---" << std::endl;

        try {
            // Test device enumeration
            int device_count = backend_->GetDeviceCount();
            std::cout << "Found " << device_count << " Vulkan devices" << std::endl;

            if (device_count == 0) {
                std::cerr << "No Vulkan devices found" << std::endl;
                return false;
            }

            // Test device switching
            for (int i = 0; i < device_count; ++i) {
                if (backend_->SetDevice(i)) {
                    std::cout << "Successfully switched to device " << i << std::endl;
                } else {
                    std::cerr << "Failed to switch to device " << i << std::endl;
                    return false;
                }
            }

            // Reset to first device
            backend_->SetDevice(0);

            // Test capability checking
            std::vector<BackendCapability> capabilities = {
                BackendCapability::MATRIX_MULTIPLICATION,
                BackendCapability::ATTENTION_COMPUTATION,
                BackendCapability::ACTIVATION_FUNCTIONS,
                BackendCapability::MEMORY_POOLING,
                BackendCapability::ASYNC_EXECUTION
            };

            for (auto cap : capabilities) {
                bool supported = backend_->SupportsCapability(cap);
                std::cout << "Capability " << static_cast<int>(cap) << ": "
                          << (supported ? "Supported" : "Not supported") << std::endl;
            }

            std::cout << "Device information test: PASSED" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Device information test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestBufferOperations() {
        std::cout << "\n--- Testing Buffer Operations ---" << std::endl;

        try {
            const size_t buffer_size = 1024 * sizeof(float);
            const size_t num_elements = buffer_size / sizeof(float);

            // Create test data
            std::vector<float> host_data(num_elements);
            std::vector<float> result_data(num_elements);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

            for (size_t i = 0; i < num_elements; ++i) {
                host_data[i] = dis(gen);
            }

            // Test buffer allocation
            auto buffer = backend_->AllocateBuffer(buffer_size);
            if (!buffer.data) {
                std::cerr << "Failed to allocate device buffer" << std::endl;
                return false;
            }

            std::cout << "Allocated buffer: " << buffer_size << " bytes" << std::endl;

            // Test copy to device
            if (!backend_->CopyToDevice(buffer, host_data.data(), buffer_size)) {
                std::cerr << "Failed to copy data to device" << std::endl;
                backend_->FreeBuffer(buffer);
                return false;
            }

            std::cout << "Data copied to device successfully" << std::endl;

            // Test copy from device
            if (!backend_->CopyFromDevice(result_data.data(), buffer, buffer_size)) {
                std::cerr << "Failed to copy data from device" << std::endl;
                backend_->FreeBuffer(buffer);
                return false;
            }

            std::cout << "Data copied from device successfully" << std::endl;

            // Verify data integrity
            bool data_matches = true;
            for (size_t i = 0; i < num_elements; ++i) {
                if (std::abs(host_data[i] - result_data[i]) > 1e-6f) {
                    data_matches = false;
                    break;
                }
            }

            if (!data_matches) {
                std::cerr << "Data integrity check failed" << std::endl;
                backend_->FreeBuffer(buffer);
                return false;
            }

            std::cout << "Data integrity verified" << std::endl;

            // Test synchronization
            backend_->Synchronize();
            std::cout << "Device synchronization completed" << std::endl;

            // Free buffer
            backend_->FreeBuffer(buffer);
            std::cout << "Buffer freed successfully" << std::endl;

            std::cout << "Buffer operations test: PASSED" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Buffer operations test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestMatrixMultiplication() {
        std::cout << "\n--- Testing Matrix Multiplication ---" << std::endl;

        try {
            const int M = 128, N = 128, K = 128;
            const size_t size_a = M * K * sizeof(float);
            const size_t size_b = K * N * sizeof(float);
            const size_t size_c = M * N * sizeof(float);

            // Create test matrices
            std::vector<float> host_a(M * K), host_b(K * N), host_c(M * N), expected_c(M * N);

            // Initialize with random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

            for (int i = 0; i < M * K; ++i) host_a[i] = dis(gen);
            for (int i = 0; i < K * N; ++i) host_b[i] = dis(gen);
            for (int i = 0; i < M * N; ++i) host_c[i] = dis(gen);

            // Compute expected result using CPU
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += host_a[i * K + k] * host_b[k * N + j];
                    }
                    expected_c[i * N + j] = sum + host_c[i * N + j]; // alpha=1, beta=1
                }
            }

            // Allocate device buffers
            auto buffer_a = backend_->AllocateBuffer(size_a);
            auto buffer_b = backend_->AllocateBuffer(size_b);
            auto buffer_c = backend_->AllocateBuffer(size_c);

            if (!buffer_a.data || !buffer_b.data || !buffer_c.data) {
                std::cerr << "Failed to allocate matrix buffers" << std::endl;
                return false;
            }

            // Copy data to device
            backend_->CopyToDevice(buffer_a, host_a.data(), size_a);
            backend_->CopyToDevice(buffer_b, host_b.data(), size_b);
            backend_->CopyToDevice(buffer_c, host_c.data(), size_c);

            // Perform matrix multiplication
            auto start_time = std::chrono::high_resolution_clock::now();

            bool success = backend_->MatrixMultiply(buffer_a, buffer_b, buffer_c,
                                                   M, N, K, 1.0f, 1.0f);

            auto end_time = std::chrono::high_resolution_clock::now();

            if (!success) {
                std::cerr << "Matrix multiplication failed" << std::endl;
                backend_->FreeBuffer(buffer_a);
                backend_->FreeBuffer(buffer_b);
                backend_->FreeBuffer(buffer_c);
                return false;
            }

            // Copy result back
            std::vector<float> gpu_result(M * N);
            backend_->CopyFromDevice(gpu_result.data(), buffer_c, size_c);

            // Verify results
            float max_error = 0.0f;
            int error_count = 0;
            const float tolerance = 1e-3f; // Relaxed tolerance for GPU computation

            for (int i = 0; i < M * N; ++i) {
                float error = std::abs(gpu_result[i] - expected_c[i]);
                max_error = std::max(max_error, error);
                if (error > tolerance) {
                    error_count++;
                }
            }

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double gflops = (2.0 * M * N * K) / (duration.count() * 1000.0);

            std::cout << "Matrix multiplication (" << M << "x" << K << ") * (" << K << "x" << N << ")" << std::endl;
            std::cout << "Execution time: " << duration.count() << " μs" << std::endl;
            std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
            std::cout << "Max error: " << max_error << std::endl;
            std::cout << "Errors above tolerance: " << error_count << "/" << (M * N) << std::endl;

            bool test_passed = (error_count == 0 || max_error < tolerance * 10);

            // Clean up
            backend_->FreeBuffer(buffer_a);
            backend_->FreeBuffer(buffer_b);
            backend_->FreeBuffer(buffer_c);

            std::cout << "Matrix multiplication test: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
            return test_passed;

        } catch (const std::exception& e) {
            std::cerr << "Matrix multiplication test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestActivationFunctions() {
        std::cout << "\n--- Testing Activation Functions ---" << std::endl;

        try {
            const size_t num_elements = 1024;
            const size_t buffer_size = num_elements * sizeof(float);

            std::vector<float> input_data(num_elements);
            std::vector<float> output_data(num_elements);

            // Generate test data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

            for (size_t i = 0; i < num_elements; ++i) {
                input_data[i] = dis(gen);
            }

            // Allocate buffers
            auto input_buffer = backend_->AllocateBuffer(buffer_size);
            auto output_buffer = backend_->AllocateBuffer(buffer_size);

            if (!input_buffer.data || !output_buffer.data) {
                std::cerr << "Failed to allocate activation function buffers" << std::endl;
                return false;
            }

            backend_->CopyToDevice(input_buffer, input_data.data(), buffer_size);

            bool all_tests_passed = true;

            // Test ReLU
            std::cout << "Testing ReLU activation..." << std::endl;
            if (backend_->ApplyReLU(input_buffer, output_buffer, num_elements)) {
                backend_->CopyFromDevice(output_data.data(), output_buffer, buffer_size);

                // Verify ReLU: f(x) = max(0, x)
                bool relu_correct = true;
                for (size_t i = 0; i < num_elements; ++i) {
                    float expected = std::max(0.0f, input_data[i]);
                    if (std::abs(output_data[i] - expected) > 1e-6f) {
                        relu_correct = false;
                        break;
                    }
                }

                std::cout << "ReLU test: " << (relu_correct ? "PASSED" : "FAILED") << std::endl;
                all_tests_passed &= relu_correct;
            } else {
                std::cout << "ReLU test: FAILED (execution error)" << std::endl;
                all_tests_passed = false;
            }

            // Test GELU
            std::cout << "Testing GELU activation..." << std::endl;
            if (backend_->ApplyGELU(input_buffer, output_buffer, num_elements)) {
                backend_->CopyFromDevice(output_data.data(), output_buffer, buffer_size);

                // Verify GELU implementation (approximation)
                bool gelu_correct = true;
                const float sqrt_2_pi = 0.7978845608f;
                const float gelu_coeff = 0.044715f;

                for (size_t i = 0; i < num_elements; ++i) {
                    float x = input_data[i];
                    float x3 = x * x * x;
                    float inner = sqrt_2_pi * (x + gelu_coeff * x3);
                    float expected = 0.5f * x * (1.0f + tanh(inner));

                    if (std::abs(output_data[i] - expected) > 1e-4f) { // Relaxed tolerance for approximation
                        gelu_correct = false;
                        break;
                    }
                }

                std::cout << "GELU test: " << (gelu_correct ? "PASSED" : "FAILED") << std::endl;
                all_tests_passed &= gelu_correct;
            } else {
                std::cout << "GELU test: FAILED (execution error)" << std::endl;
                all_tests_passed = false;
            }

            // Test Softmax (simplified test with smaller data)
            const size_t softmax_size = 32;
            const size_t softmax_buffer_size = softmax_size * sizeof(float);

            std::vector<float> softmax_input(softmax_size);
            std::vector<float> softmax_output(softmax_size);

            for (size_t i = 0; i < softmax_size; ++i) {
                softmax_input[i] = dis(gen);
            }

            auto softmax_input_buffer = backend_->AllocateBuffer(softmax_buffer_size);
            auto softmax_output_buffer = backend_->AllocateBuffer(softmax_buffer_size);

            backend_->CopyToDevice(softmax_input_buffer, softmax_input.data(), softmax_buffer_size);

            std::cout << "Testing Softmax activation..." << std::endl;
            if (backend_->ApplySoftmax(softmax_input_buffer, softmax_output_buffer, softmax_size)) {
                backend_->CopyFromDevice(softmax_output.data(), softmax_output_buffer, softmax_buffer_size);

                // Verify softmax properties: sum = 1, all values > 0
                float sum = 0.0f;
                bool all_positive = true;

                for (size_t i = 0; i < softmax_size; ++i) {
                    sum += softmax_output[i];
                    if (softmax_output[i] <= 0.0f) {
                        all_positive = false;
                    }
                }

                bool softmax_correct = (std::abs(sum - 1.0f) < 1e-4f) && all_positive;

                std::cout << "Softmax test: " << (softmax_correct ? "PASSED" : "FAILED") << std::endl;
                std::cout << "  Sum: " << sum << " (expected: 1.0)" << std::endl;
                all_tests_passed &= softmax_correct;
            } else {
                std::cout << "Softmax test: FAILED (execution error)" << std::endl;
                all_tests_passed = false;
            }

            // Clean up
            backend_->FreeBuffer(input_buffer);
            backend_->FreeBuffer(output_buffer);
            backend_->FreeBuffer(softmax_input_buffer);
            backend_->FreeBuffer(softmax_output_buffer);

            std::cout << "Activation functions test: " << (all_tests_passed ? "PASSED" : "FAILED") << std::endl;
            return all_tests_passed;

        } catch (const std::exception& e) {
            std::cerr << "Activation functions test failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestPerformance() {
        std::cout << "\n--- Performance Benchmarks ---" << std::endl;

        try {
            // Reset metrics
            backend_->ResetMetrics();

            // Run performance tests with different matrix sizes
            std::vector<int> matrix_sizes = {64, 128, 256, 512};

            for (int size : matrix_sizes) {
                std::cout << "\nBenchmarking " << size << "x" << size << " matrix multiplication..." << std::endl;

                const size_t buffer_size = size * size * sizeof(float);
                std::vector<float> matrix_data(size * size, 1.0f);

                auto buffer_a = backend_->AllocateBuffer(buffer_size);
                auto buffer_b = backend_->AllocateBuffer(buffer_size);
                auto buffer_c = backend_->AllocateBuffer(buffer_size);

                backend_->CopyToDevice(buffer_a, matrix_data.data(), buffer_size);
                backend_->CopyToDevice(buffer_b, matrix_data.data(), buffer_size);
                backend_->CopyToDevice(buffer_c, matrix_data.data(), buffer_size);

                // Warm up
                backend_->MatrixMultiply(buffer_a, buffer_b, buffer_c, size, size, size);

                // Benchmark multiple iterations
                const int iterations = 5;
                auto start_time = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < iterations; ++i) {
                    backend_->MatrixMultiply(buffer_a, buffer_b, buffer_c, size, size, size);
                }

                backend_->Synchronize();
                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                double avg_time_ms = duration.count() / (1000.0 * iterations);
                double gflops = (2.0 * size * size * size) / (avg_time_ms * 1000.0);

                std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
                std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;

                backend_->FreeBuffer(buffer_a);
                backend_->FreeBuffer(buffer_b);
                backend_->FreeBuffer(buffer_c);
            }

            // Display overall metrics
            auto metrics = backend_->GetMetrics();
            std::cout << "\n--- Overall Metrics ---" << std::endl;
            std::cout << "Peak compute throughput: " << metrics.compute_throughput_gflops << " GFLOPS" << std::endl;
            std::cout << "Memory bandwidth: " << metrics.memory_bandwidth_gbps << " GB/s" << std::endl;
            std::cout << "Peak memory usage: " << (metrics.peak_memory_bytes / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "Current memory usage: " << (metrics.memory_usage_bytes / (1024 * 1024)) << " MB" << std::endl;

            std::cout << "Performance benchmarks: COMPLETED" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Performance test failed: " << e.what() << std::endl;
            return false;
        }
    }
};

int main() {
    try {
        VulkanBackendTest test;
        bool success = test.RunAllTests();
        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Test suite failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test suite failed with unknown exception" << std::endl;
        return 1;
    }
}