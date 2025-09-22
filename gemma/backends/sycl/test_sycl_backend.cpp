/**
 * @file test_sycl_backend.cpp
 * @brief Test program for Intel SYCL backend functionality
 *
 * This test verifies that the SYCL backend can be properly initialized,
 * detects Intel devices, and performs basic operations.
 */

#include "sycl_backend.h"
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

using namespace gemma::backends::sycl;

/**
 * @brief Test device detection and initialization
 */
bool TestDeviceDetection() {
    std::cout << "\n=== Testing Device Detection ===" << std::endl;
    
    auto backend = std::make_unique<SyclBackend>();
    
    if (!backend->IsAvailable()) {
        std::cout << "SYCL backend not available, checking why..." << std::endl;
        
        // Try manual device detection
        auto devices = SyclBackend::DetectDevices();
        if (devices.empty()) {
            std::cout << "No SYCL devices detected" << std::endl;
            return false;
        }
        
        std::cout << "Found " << devices.size() << " devices but backend not available" << std::endl;
        return false;
    }
    
    if (!backend->Initialize()) {
        std::cout << "Failed to initialize SYCL backend" << std::endl;
        return false;
    }
    
    std::cout << "SYCL backend initialized successfully" << std::endl;
    std::cout << "Backend name: " << backend->GetName() << std::endl;
    std::cout << "Backend version: " << backend->GetVersion() << std::endl;
    std::cout << "Device count: " << backend->GetDeviceCount() << std::endl;
    
    // Test device information
    if (backend->GetDeviceCount() > 0) {
        auto device_info = backend->GetCurrentDeviceInfo();
        std::cout << "Current device: " << device_info.name << std::endl;
        std::cout << "Device vendor: " << device_info.vendor << std::endl;
        std::cout << "Device memory: " << (device_info.max_memory_bytes / (1024*1024)) << " MB" << std::endl;
        std::cout << "FP16 support: " << (device_info.supports_fp16 ? "Yes" : "No") << std::endl;
        std::cout << "USM support: " << (device_info.supports_unified_memory ? "Yes" : "No") << std::endl;
    }
    
    backend->Shutdown();
    return true;
}

/**
 * @brief Test memory allocation and operations
 */
bool TestMemoryOperations() {
    std::cout << "\n=== Testing Memory Operations ===" << std::endl;
    
    auto backend = std::make_unique<SyclBackend>();
    
    if (!backend->Initialize()) {
        std::cout << "Failed to initialize SYCL backend for memory test" << std::endl;
        return false;
    }
    
    // Test buffer allocation
    const size_t buffer_size = 1024 * sizeof(float);
    auto buffer = backend->AllocateBuffer(buffer_size, 32);
    
    if (!buffer.data || buffer.size != buffer_size) {
        std::cout << "Failed to allocate device buffer" << std::endl;
        backend->Shutdown();
        return false;
    }
    
    std::cout << "Allocated " << buffer_size << " bytes on device" << std::endl;
    
    // Test host-to-device copy
    std::vector<float> host_data(1024);
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = static_cast<float>(i);
    }
    
    bool copy_success = backend->CopyToDevice(buffer, host_data.data(), buffer_size);
    if (!copy_success) {
        std::cout << "Failed to copy data to device" << std::endl;
        backend->FreeBuffer(buffer);
        backend->Shutdown();
        return false;
    }
    
    std::cout << "Successfully copied data to device" << std::endl;
    
    // Test device-to-host copy
    std::vector<float> result_data(1024);
    copy_success = backend->CopyFromDevice(result_data.data(), buffer, buffer_size);
    if (!copy_success) {
        std::cout << "Failed to copy data from device" << std::endl;
        backend->FreeBuffer(buffer);
        backend->Shutdown();
        return false;
    }
    
    // Verify data integrity
    bool data_correct = true;
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (result_data[i] != static_cast<float>(i)) {
            data_correct = false;
            break;
        }
    }
    
    if (!data_correct) {
        std::cout << "Data integrity check failed" << std::endl;
        backend->FreeBuffer(buffer);
        backend->Shutdown();
        return false;
    }
    
    std::cout << "Data integrity check passed" << std::endl;
    
    backend->FreeBuffer(buffer);
    backend->Shutdown();
    return true;
}

/**
 * @brief Test basic matrix operations
 */
bool TestMatrixOperations() {
    std::cout << "\n=== Testing Matrix Operations ===" << std::endl;
    
    auto backend = std::make_unique<SyclBackend>();
    
    if (!backend->Initialize()) {
        std::cout << "Failed to initialize SYCL backend for matrix test" << std::endl;
        return false;
    }
    
    // Test small matrix multiplication: C = A * B
    const int M = 4, N = 4, K = 4;
    const size_t matrix_size = M * N * sizeof(float);
    
    // Allocate device buffers
    auto buffer_a = backend->AllocateBuffer(M * K * sizeof(float));
    auto buffer_b = backend->AllocateBuffer(K * N * sizeof(float));
    auto buffer_c = backend->AllocateBuffer(M * N * sizeof(float));
    
    if (!buffer_a.data || !buffer_b.data || !buffer_c.data) {
        std::cout << "Failed to allocate matrices on device" << std::endl;
        backend->Shutdown();
        return false;
    }
    
    // Initialize test matrices
    std::vector<float> host_a(M * K, 1.0f);  // All ones
    std::vector<float> host_b(K * N, 2.0f);  // All twos
    std::vector<float> host_c(M * N, 0.0f);  // All zeros
    
    // Copy to device
    backend->CopyToDevice(buffer_a, host_a.data(), M * K * sizeof(float));
    backend->CopyToDevice(buffer_b, host_b.data(), K * N * sizeof(float));
    backend->CopyToDevice(buffer_c, host_c.data(), M * N * sizeof(float));
    
    // Perform matrix multiplication
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool mm_success = backend->MatrixMultiply(buffer_a, buffer_b, buffer_c, M, N, K);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (!mm_success) {
        std::cout << "Matrix multiplication failed" << std::endl;
        backend->FreeBuffer(buffer_a);
        backend->FreeBuffer(buffer_b);
        backend->FreeBuffer(buffer_c);
        backend->Shutdown();
        return false;
    }
    
    std::cout << "Matrix multiplication completed in " << duration.count() << " microseconds" << std::endl;
    
    // Copy result back and verify
    std::vector<float> result(M * N);
    backend->CopyFromDevice(result.data(), buffer_c, M * N * sizeof(float));
    
    // Expected result: each element should be K * 1.0 * 2.0 = 8.0
    bool result_correct = true;
    const float expected = static_cast<float>(K) * 1.0f * 2.0f;
    
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(result[i] - expected) > 1e-5f) {
            std::cout << "Incorrect result at position " << i 
                      << ": got " << result[i] << ", expected " << expected << std::endl;
            result_correct = false;
            break;
        }
    }
    
    if (result_correct) {
        std::cout << "Matrix multiplication result correct!" << std::endl;
    }
    
    // Clean up
    backend->FreeBuffer(buffer_a);
    backend->FreeBuffer(buffer_b);
    backend->FreeBuffer(buffer_c);
    backend->Shutdown();
    
    return result_correct;
}

/**
 * @brief Test activation functions
 */
bool TestActivationFunctions() {
    std::cout << "\n=== Testing Activation Functions ===" << std::endl;
    
    auto backend = std::make_unique<SyclBackend>();
    
    if (!backend->Initialize()) {
        std::cout << "Failed to initialize SYCL backend for activation test" << std::endl;
        return false;
    }
    
    const size_t size = 8;
    const size_t buffer_size = size * sizeof(float);
    
    auto input_buffer = backend->AllocateBuffer(buffer_size);
    auto output_buffer = backend->AllocateBuffer(buffer_size);
    
    if (!input_buffer.data || !output_buffer.data) {
        std::cout << "Failed to allocate buffers for activation test" << std::endl;
        backend->Shutdown();
        return false;
    }
    
    // Test data: [-2, -1, 0, 1, 2, 3, 4, 5]
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    backend->CopyToDevice(input_buffer, input_data.data(), buffer_size);
    
    // Test ReLU
    std::cout << "Testing ReLU activation..." << std::endl;
    bool relu_success = backend->ApplyReLU(input_buffer, output_buffer, size);
    
    if (relu_success) {
        std::vector<float> relu_result(size);
        backend->CopyFromDevice(relu_result.data(), output_buffer, buffer_size);
        
        std::cout << "ReLU results: ";
        for (size_t i = 0; i < size; ++i) {
            std::cout << relu_result[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "ReLU test failed" << std::endl;
    }
    
    // Test GELU
    std::cout << "Testing GELU activation..." << std::endl;
    bool gelu_success = backend->ApplyGELU(input_buffer, output_buffer, size);
    
    if (gelu_success) {
        std::vector<float> gelu_result(size);
        backend->CopyFromDevice(gelu_result.data(), output_buffer, buffer_size);
        
        std::cout << "GELU results: ";
        for (size_t i = 0; i < size; ++i) {
            std::cout << gelu_result[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "GELU test failed" << std::endl;
    }
    
    backend->FreeBuffer(input_buffer);
    backend->FreeBuffer(output_buffer);
    backend->Shutdown();
    
    return relu_success && gelu_success;
}

/**
 * @brief Main test function
 */
int main() {
    std::cout << "Intel SYCL Backend Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Check if SYCL backend is available at all
    if (!IsSyclBackendAvailable()) {
        std::cout << "SYCL backend is not available on this system" << std::endl;
        std::cout << "Please ensure Intel oneAPI toolkit is installed and configured" << std::endl;
        return 1;
    }
    
    std::cout << "SYCL backend is available" << std::endl;
    std::cout << "oneAPI version: " << GetOneAPIVersion() << std::endl;
    
    int failed_tests = 0;
    
    // Run tests
    if (!TestDeviceDetection()) {
        std::cout << "❌ Device detection test failed" << std::endl;
        failed_tests++;
    } else {
        std::cout << "✅ Device detection test passed" << std::endl;
    }
    
    if (!TestMemoryOperations()) {
        std::cout << "❌ Memory operations test failed" << std::endl;
        failed_tests++;
    } else {
        std::cout << "✅ Memory operations test passed" << std::endl;
    }
    
    if (!TestMatrixOperations()) {
        std::cout << "❌ Matrix operations test failed" << std::endl;
        failed_tests++;
    } else {
        std::cout << "✅ Matrix operations test passed" << std::endl;
    }
    
    if (!TestActivationFunctions()) {
        std::cout << "❌ Activation functions test failed" << std::endl;
        failed_tests++;
    } else {
        std::cout << "✅ Activation functions test passed" << std::endl;
    }
    
    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Total tests: 4" << std::endl;
    std::cout << "Passed: " << (4 - failed_tests) << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;
    
    if (failed_tests == 0) {
        std::cout << "\n🎉 All tests passed! Intel SYCL backend is working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "\n⚠️  Some tests failed. Please check the SYCL backend configuration." << std::endl;
        return 1;
    }
}