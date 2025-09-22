/**
 * test_sycl.cpp - SYCL Backend Tests for Gemma.cpp
 *
 * Comprehensive test suite for SYCL/Intel oneAPI backend functionality
 * including basic operations, memory management, and performance verification
 */

#ifdef GEMMA_ENABLE_SYCL

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <numeric>

namespace gemma {
namespace sycl_backend {
namespace test {

class SYCLBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            // Initialize SYCL device
            device_ = sycl::device{sycl::default_selector_v};
            queue_ = std::make_unique<sycl::queue>(device_);

            device_info_.name = device_.get_info<sycl::info::device::name>();
            device_info_.vendor = device_.get_info<sycl::info::device::vendor>();
            device_info_.version = device_.get_info<sycl::info::device::version>();
            device_info_.max_compute_units = device_.get_info<sycl::info::device::max_compute_units>();
            device_info_.max_work_group_size = device_.get_info<sycl::info::device::max_work_group_size>();

            std::cout << "SYCL Device: " << device_info_.name << std::endl;
            std::cout << "Vendor: " << device_info_.vendor << std::endl;
            std::cout << "Compute Units: " << device_info_.max_compute_units << std::endl;

        } catch (const sycl::exception& e) {
            FAIL() << "Failed to initialize SYCL device: " << e.what();
        }
    }

    void TearDown() override {
        if (queue_) {
            queue_->wait();
        }
    }

    struct DeviceInfo {
        std::string name;
        std::string vendor;
        std::string version;
        size_t max_compute_units;
        size_t max_work_group_size;
    };

    sycl::device device_;
    std::unique_ptr<sycl::queue> queue_;
    DeviceInfo device_info_;
};

// Test basic SYCL device detection and initialization
TEST_F(SYCLBackendTest, DeviceDetection) {
    EXPECT_FALSE(device_info_.name.empty());
    EXPECT_FALSE(device_info_.vendor.empty());
    EXPECT_GT(device_info_.max_compute_units, 0);
    EXPECT_GT(device_info_.max_work_group_size, 0);
}

// Test basic memory allocation and deallocation
TEST_F(SYCLBackendTest, MemoryAllocation) {
    const size_t size = 1024;

    // Test USM allocation
    float* device_ptr = sycl::malloc_device<float>(size, *queue_);
    ASSERT_NE(device_ptr, nullptr);

    float* host_ptr = sycl::malloc_host<float>(size, *queue_);
    ASSERT_NE(host_ptr, nullptr);

    float* shared_ptr = sycl::malloc_shared<float>(size, *queue_);
    ASSERT_NE(shared_ptr, nullptr);

    // Clean up
    sycl::free(device_ptr, *queue_);
    sycl::free(host_ptr, *queue_);
    sycl::free(shared_ptr, *queue_);
}

// Test memory copy operations
TEST_F(SYCLBackendTest, MemoryCopy) {
    const size_t size = 1000;
    std::vector<float> host_data(size);

    // Initialize test data
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Allocate device memory
    float* device_ptr = sycl::malloc_device<float>(size, *queue_);
    ASSERT_NE(device_ptr, nullptr);

    // Copy to device
    queue_->memcpy(device_ptr, host_data.data(), size * sizeof(float));
    queue_->wait();

    // Copy back to host
    std::vector<float> result(size);
    queue_->memcpy(result.data(), device_ptr, size * sizeof(float));
    queue_->wait();

    // Verify data integrity
    EXPECT_EQ(host_data, result);

    sycl::free(device_ptr, *queue_);
}

// Test basic kernel execution (vector addition)
TEST_F(SYCLBackendTest, VectorAddition) {
    const size_t size = 1000;

    // Prepare test data
    std::vector<float> a(size, 2.0f);
    std::vector<float> b(size, 3.0f);
    std::vector<float> result(size, 0.0f);

    // Allocate USM memory
    float* dev_a = sycl::malloc_device<float>(size, *queue_);
    float* dev_b = sycl::malloc_device<float>(size, *queue_);
    float* dev_result = sycl::malloc_device<float>(size, *queue_);

    ASSERT_NE(dev_a, nullptr);
    ASSERT_NE(dev_b, nullptr);
    ASSERT_NE(dev_result, nullptr);

    // Copy data to device
    queue_->memcpy(dev_a, a.data(), size * sizeof(float));
    queue_->memcpy(dev_b, b.data(), size * sizeof(float));
    queue_->wait();

    // Execute kernel
    queue_->parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx) {
        dev_result[idx] = dev_a[idx] + dev_b[idx];
    });
    queue_->wait();

    // Copy result back
    queue_->memcpy(result.data(), dev_result, size * sizeof(float));
    queue_->wait();

    // Verify results
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], 5.0f);
    }

    // Clean up
    sycl::free(dev_a, *queue_);
    sycl::free(dev_b, *queue_);
    sycl::free(dev_result, *queue_);
}

// Test matrix multiplication kernel
TEST_F(SYCLBackendTest, MatrixMultiplication) {
    const size_t N = 64;  // Small matrix for testing
    const size_t size = N * N;

    // Initialize matrices
    std::vector<float> A(size, 1.0f);
    std::vector<float> B(size, 2.0f);
    std::vector<float> C(size, 0.0f);

    // Allocate device memory
    float* dev_A = sycl::malloc_device<float>(size, *queue_);
    float* dev_B = sycl::malloc_device<float>(size, *queue_);
    float* dev_C = sycl::malloc_device<float>(size, *queue_);

    // Copy data to device
    queue_->memcpy(dev_A, A.data(), size * sizeof(float));
    queue_->memcpy(dev_B, B.data(), size * sizeof(float));
    queue_->wait();

    // Matrix multiplication kernel
    queue_->parallel_for(sycl::range<2>{N, N}, [=](sycl::id<2> idx) {
        size_t row = idx[0];
        size_t col = idx[1];
        float sum = 0.0f;

        for (size_t k = 0; k < N; ++k) {
            sum += dev_A[row * N + k] * dev_B[k * N + col];
        }

        dev_C[row * N + col] = sum;
    });
    queue_->wait();

    // Copy result back
    queue_->memcpy(C.data(), dev_C, size * sizeof(float));
    queue_->wait();

    // Verify result (1 * 2 * N = 2 * N for each element)
    float expected = 2.0f * N;
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(C[i], expected);
    }

    // Clean up
    sycl::free(dev_A, *queue_);
    sycl::free(dev_B, *queue_);
    sycl::free(dev_C, *queue_);
}

// Test error handling
TEST_F(SYCLBackendTest, ErrorHandling) {
    // Test invalid memory access
    float* null_ptr = nullptr;

    EXPECT_THROW({
        queue_->memcpy(null_ptr, null_ptr, 100);
        queue_->wait();
    }, sycl::exception);
}

// Performance benchmark test
TEST_F(SYCLBackendTest, PerformanceBenchmark) {
    const size_t size = 1000000;  // 1M elements

    std::vector<float> a(size, 1.5f);
    std::vector<float> b(size, 2.5f);
    std::vector<float> result(size);

    // Allocate device memory
    float* dev_a = sycl::malloc_device<float>(size, *queue_);
    float* dev_b = sycl::malloc_device<float>(size, *queue_);
    float* dev_result = sycl::malloc_device<float>(size, *queue_);

    // Copy data to device
    queue_->memcpy(dev_a, a.data(), size * sizeof(float));
    queue_->memcpy(dev_b, b.data(), size * sizeof(float));
    queue_->wait();

    // Benchmark kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    queue_->parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx) {
        // Simulate more complex computation
        float val = dev_a[idx] * dev_b[idx] + dev_a[idx];
        dev_result[idx] = sycl::sqrt(val) + sycl::sin(val * 0.01f);
    });
    queue_->wait();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy result back
    queue_->memcpy(result.data(), dev_result, size * sizeof(float));
    queue_->wait();

    // Report performance
    double throughput = static_cast<double>(size) / duration.count() * 1e6;  // elements per second
    std::cout << "SYCL Performance: " << throughput << " elements/sec" << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    // Verify at least some computation occurred
    bool all_zero = std::all_of(result.begin(), result.end(), [](float val) { return val == 0.0f; });
    EXPECT_FALSE(all_zero);

    // Clean up
    sycl::free(dev_a, *queue_);
    sycl::free(dev_b, *queue_);
    sycl::free(dev_result, *queue_);
}

// Test work group optimization
TEST_F(SYCLBackendTest, WorkGroupOptimization) {
    const size_t global_size = 1024;
    const size_t local_size = 64;

    std::vector<float> data(global_size, 1.0f);
    std::vector<float> result(global_size / local_size, 0.0f);

    float* dev_data = sycl::malloc_device<float>(global_size, *queue_);
    float* dev_result = sycl::malloc_device<float>(global_size / local_size, *queue_);

    queue_->memcpy(dev_data, data.data(), global_size * sizeof(float));
    queue_->wait();

    // Reduction kernel using work groups
    queue_->parallel_for(
        sycl::nd_range<1>{global_size, local_size},
        [=](sycl::nd_item<1> item) {
            size_t global_id = item.get_global_id(0);
            size_t local_id = item.get_local_id(0);
            size_t group_id = item.get_group(0);

            // Use local memory for reduction
            auto local_mem = sycl::local_accessor<float, 1>{local_size, item};

            // Copy to local memory
            local_mem[local_id] = dev_data[global_id];
            item.barrier(sycl::access::fence_space::local_space);

            // Perform reduction in local memory
            for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
                if (local_id < stride) {
                    local_mem[local_id] += local_mem[local_id + stride];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            // Write result to global memory
            if (local_id == 0) {
                dev_result[group_id] = local_mem[0];
            }
        }
    );
    queue_->wait();

    // Copy result back
    queue_->memcpy(result.data(), dev_result, (global_size / local_size) * sizeof(float));
    queue_->wait();

    // Verify reduction results
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(local_size));
    }

    // Clean up
    sycl::free(dev_data, *queue_);
    sycl::free(dev_result, *queue_);
}

}  // namespace test
}  // namespace sycl_backend
}  // namespace gemma

#endif  // GEMMA_ENABLE_SYCL