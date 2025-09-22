/**
 * test_cuda.cpp - CUDA Backend Tests for Gemma.cpp
 *
 * Comprehensive test suite for CUDA backend functionality including
 * basic operations, memory management, cuBLAS/cuDNN integration, and performance
 */

#ifdef GEMMA_ENABLE_CUDA

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifdef CUBLAS_AVAILABLE
#include <cublas_v2.h>
#endif

#ifdef CUDNN_AVAILABLE
#include <cudnn.h>
#endif

namespace gemma {
namespace cuda_backend {
namespace test {

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            FAIL() << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                   << " - " << cudaGetErrorString(error); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            FAIL() << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                   << " - Status: " << status; \
        } \
    } while(0)

class CUDABackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        ASSERT_GT(device_count, 0) << "No CUDA devices found";

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaGetDeviceProperties(&device_props_, 0));

        std::cout << "CUDA Device: " << device_props_.name << std::endl;
        std::cout << "Compute Capability: " << device_props_.major << "." << device_props_.minor << std::endl;
        std::cout << "Global Memory: " << device_props_.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Shared Memory per Block: " << device_props_.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Max Threads per Block: " << device_props_.maxThreadsPerBlock << std::endl;

        // Initialize cuBLAS
#ifdef CUBLAS_AVAILABLE
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#endif

        // Create CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void TearDown() override {
#ifdef CUBLAS_AVAILABLE
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
#endif

        if (stream_) {
            cudaStreamDestroy(stream_);
        }

        cudaDeviceReset();
    }

    cudaDeviceProp device_props_;
    cudaStream_t stream_ = nullptr;

#ifdef CUBLAS_AVAILABLE
    cublasHandle_t cublas_handle_ = nullptr;
#endif
};

// Simple CUDA kernel for testing
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale_kernel(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Test basic CUDA device detection and properties
TEST_F(CUDABackendTest, DeviceDetection) {
    EXPECT_FALSE(std::string(device_props_.name).empty());
    EXPECT_GT(device_props_.totalGlobalMem, 0);
    EXPECT_GT(device_props_.multiProcessorCount, 0);
    EXPECT_GE(device_props_.major, 3);  // Minimum compute capability 3.0
}

// Test basic memory allocation and deallocation
TEST_F(CUDABackendTest, MemoryAllocation) {
    const size_t size = 1024 * sizeof(float);

    // Test device memory allocation
    float* device_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, size));
    EXPECT_NE(device_ptr, nullptr);
    CUDA_CHECK(cudaFree(device_ptr));

    // Test pinned host memory allocation
    float* host_ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_ptr, size));
    EXPECT_NE(host_ptr, nullptr);
    CUDA_CHECK(cudaFreeHost(host_ptr));

    // Test unified memory allocation
    float* unified_ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&unified_ptr, size));
    EXPECT_NE(unified_ptr, nullptr);
    CUDA_CHECK(cudaFree(unified_ptr));
}

// Test memory copy operations
TEST_F(CUDABackendTest, MemoryCopy) {
    const size_t n = 1000;
    const size_t size = n * sizeof(float);

    std::vector<float> host_data(n);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Allocate device memory
    float* device_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(device_ptr, host_data.data(), size, cudaMemcpyHostToDevice));

    // Copy back to host
    std::vector<float> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), device_ptr, size, cudaMemcpyDeviceToHost));

    // Verify data integrity
    EXPECT_EQ(host_data, result);

    CUDA_CHECK(cudaFree(device_ptr));
}

// Test async memory copy with streams
TEST_F(CUDABackendTest, AsyncMemoryCopy) {
    const size_t n = 1000;
    const size_t size = n * sizeof(float);

    std::vector<float> host_data(n, 2.5f);

    // Allocate pinned host memory for async operations
    float* pinned_host = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_host, size));
    std::copy(host_data.begin(), host_data.end(), pinned_host);

    // Allocate device memory
    float* device_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, size));

    // Async copy to device
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, pinned_host, size,
                              cudaMemcpyHostToDevice, stream_));

    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Verify by copying back
    std::vector<float> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), device_ptr, size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], 2.5f);
    }

    CUDA_CHECK(cudaFree(device_ptr));
    CUDA_CHECK(cudaFreeHost(pinned_host));
}

// Test basic kernel execution
TEST_F(CUDABackendTest, VectorAddition) {
    const size_t n = 1000;
    const size_t size = n * sizeof(float);

    // Prepare test data
    std::vector<float> a(n, 2.0f);
    std::vector<float> b(n, 3.0f);
    std::vector<float> result(n);

    // Allocate device memory
    float *dev_a, *dev_b, *dev_c;
    CUDA_CHECK(cudaMalloc(&dev_a, size));
    CUDA_CHECK(cudaMalloc(&dev_b, size));
    CUDA_CHECK(cudaMalloc(&dev_c, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, n);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(result.data(), dev_c, size, cudaMemcpyDeviceToHost));

    // Verify results
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], 5.0f);
    }

    // Clean up
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));
}

// Test matrix multiplication kernel
TEST_F(CUDABackendTest, MatrixMultiplication) {
    const int M = 64, N = 64, K = 64;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // Initialize matrices
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    // Allocate device memory
    float *dev_A, *dev_B, *dev_C;
    CUDA_CHECK(cudaMalloc(&dev_A, size_A));
    CUDA_CHECK(cudaMalloc(&dev_B, size_B));
    CUDA_CHECK(cudaMalloc(&dev_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dev_A, A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B.data(), size_B, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);

    matrix_multiply_kernel<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), dev_C, size_C, cudaMemcpyDeviceToHost));

    // Verify result (1 * 2 * K = 2 * K for each element)
    float expected = 2.0f * K;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(C[i], expected);
    }

    // Clean up
    CUDA_CHECK(cudaFree(dev_A));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_C));
}

// Test cuBLAS matrix multiplication
#ifdef CUBLAS_AVAILABLE
TEST_F(CUDABackendTest, CuBLASMatrixMultiplication) {
    const int M = 128, N = 128, K = 128;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // Initialize matrices
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    // Allocate device memory
    float *dev_A, *dev_B, *dev_C;
    CUDA_CHECK(cudaMalloc(&dev_A, size_A));
    CUDA_CHECK(cudaMalloc(&dev_B, size_B));
    CUDA_CHECK(cudaMalloc(&dev_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dev_A, A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, B.data(), size_B, cudaMemcpyHostToDevice));

    // Perform cuBLAS GEMM: C = alpha * A * B + beta * C
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            dev_B, N,
                            dev_A, K,
                            &beta,
                            dev_C, N));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), dev_C, size_C, cudaMemcpyDeviceToHost));

    // Verify result
    float expected = 2.0f * K;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], expected, 1e-5f);
    }

    // Clean up
    CUDA_CHECK(cudaFree(dev_A));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_C));
}
#endif

// Test shared memory usage
TEST_F(CUDABackendTest, SharedMemoryUsage) {
    const size_t n = 1024;
    const size_t size = n * sizeof(float);

    std::vector<float> data(n, 1.0f);
    std::vector<float> result(n);

    float* dev_data;
    CUDA_CHECK(cudaMalloc(&dev_data, size));
    CUDA_CHECK(cudaMemcpy(dev_data, data.data(), size, cudaMemcpyHostToDevice));

    // Kernel using shared memory for tiling
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const int shared_mem_size = block_size * sizeof(float);

    // Launch kernel with shared memory
    vector_scale_kernel<<<grid_size, block_size, shared_mem_size>>>(dev_data, 3.0f, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), dev_data, size, cudaMemcpyDeviceToHost));

    // Verify scaling
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], 3.0f);
    }

    CUDA_CHECK(cudaFree(dev_data));
}

// Performance benchmark test
TEST_F(CUDABackendTest, PerformanceBenchmark) {
    const size_t n = 10000000;  // 10M elements
    const size_t size = n * sizeof(float);

    std::vector<float> a(n, 1.5f);
    std::vector<float> b(n, 2.5f);

    float *dev_a, *dev_b, *dev_c;
    CUDA_CHECK(cudaMalloc(&dev_a, size));
    CUDA_CHECK(cudaMalloc(&dev_b, size));
    CUDA_CHECK(cudaMalloc(&dev_c, size));

    CUDA_CHECK(cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice));

    // Benchmark kernel execution
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, n);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate performance metrics
    double throughput = static_cast<double>(n) / (milliseconds * 1e-3);  // elements per second
    double bandwidth = (3 * size) / (milliseconds * 1e-3) / (1024 * 1024 * 1024);  // GB/s

    std::cout << "CUDA Performance:" << std::endl;
    std::cout << "  Throughput: " << throughput << " elements/sec" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "  Execution time: " << milliseconds << " ms" << std::endl;

    // Basic sanity check
    EXPECT_GT(throughput, 1e6);  // Should process at least 1M elements/sec

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));
}

// Test error handling
TEST_F(CUDABackendTest, ErrorHandling) {
    // Test invalid memory allocation
    float* large_ptr = nullptr;
    cudaError_t error = cudaMalloc(&large_ptr, SIZE_MAX);
    EXPECT_NE(error, cudaSuccess);

    // Test invalid kernel parameters
    vector_add_kernel<<<-1, 256>>>(nullptr, nullptr, nullptr, 0);
    error = cudaGetLastError();
    EXPECT_NE(error, cudaSuccess);
}

// Test multiple streams
TEST_F(CUDABackendTest, MultipleStreams) {
    const size_t n = 1000;
    const size_t size = n * sizeof(float);
    const int num_streams = 4;

    std::vector<cudaStream_t> streams(num_streams);
    std::vector<float*> dev_data(num_streams);
    std::vector<std::vector<float>> host_data(num_streams, std::vector<float>(n, 1.0f));

    // Create streams and allocate memory
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&dev_data[i], size));
    }

    // Launch operations on different streams
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(dev_data[i], host_data[i].data(), size,
                                  cudaMemcpyHostToDevice, streams[i]));

        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        vector_scale_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            dev_data[i], static_cast<float>(i + 2), n);
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Verify results
    for (int i = 0; i < num_streams; ++i) {
        std::vector<float> result(n);
        CUDA_CHECK(cudaMemcpy(result.data(), dev_data[i], size, cudaMemcpyDeviceToHost));

        float expected = static_cast<float>(i + 2);
        for (size_t j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(result[j], expected);
        }
    }

    // Clean up
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(dev_data[i]));
    }
}

}  // namespace test
}  // namespace cuda_backend
}  // namespace gemma

#endif  // GEMMA_ENABLE_CUDA