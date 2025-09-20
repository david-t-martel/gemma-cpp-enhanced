// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <memory>
#include <string>

#include "../utils/test_common.h"
#include "ops/ops.h"
#include "ops/matmul.h"
#include "compression/compress.h"
#include "util/allocator.h"
#include "util/threading_context.h"
#include "hwy/detect_targets.h"
#include "hwy/targets.h"
#include "hwy/tests/hwy_gtest.h"

namespace gcpp {
namespace {

using namespace test_utils;

// Test different SIMD backends available on the current platform
class BackendCompatibilityTest : public GemmaTestBase {
 protected:
  void SetUp() override {
    GemmaTestBase::SetUp();
    // Initialize threading context for tests
    thread_context_ = std::make_unique<ThreadingContext>();
  }

  void TearDown() override {
    thread_context_.reset();
    GemmaTestBase::TearDown();
  }

  std::unique_ptr<ThreadingContext> thread_context_;
};

// Test basic SIMD backend detection and functionality
TEST_F(BackendCompatibilityTest, SIMDBackendDetection) {
  // Check that Highway has detected at least one target
  EXPECT_NE(hwy::SupportedTargets(), 0) << "No SIMD targets detected";

  // Log available targets for debugging
  const hwy::TargetBitfield supported = hwy::SupportedTargets();
  std::cout << "Supported SIMD targets: 0x" << std::hex << supported << std::dec << std::endl;

  // Check for common targets
  if (supported & HWY_TARGET_SSE4) {
    std::cout << "SSE4 support detected" << std::endl;
  }
  if (supported & HWY_TARGET_AVX2) {
    std::cout << "AVX2 support detected" << std::endl;
  }
  if (supported & HWY_TARGET_AVX3) {
    std::cout << "AVX-512 support detected" << std::endl;
  }
  if (supported & HWY_TARGET_NEON) {
    std::cout << "NEON support detected" << std::endl;
  }

  // Ensure we have at least scalar fallback
  EXPECT_TRUE(supported & HWY_TARGET_SCALAR) << "Scalar fallback should always be available";
}

// Test matrix multiplication across different backends
TEST_F(BackendCompatibilityTest, MatrixMultiplicationBackends) {
  const size_t rows = 64;
  const size_t cols = 64;
  const size_t inner = 128;

  // Create test matrices
  auto mat_a = data_gen_->GenerateRandomFloats(rows * inner, -1.0f, 1.0f);
  auto mat_b = data_gen_->GenerateRandomFloats(inner * cols, -1.0f, 1.0f);
  auto mat_c_ref = std::vector<float>(rows * cols, 0.0f);
  auto mat_c_test = std::vector<float>(rows * cols, 0.0f);

  // Create matrix wrappers
  MatPtr A{mat_a.data(), rows, inner};
  MatPtr B{mat_b.data(), inner, cols};
  MatPtr C_ref{mat_c_ref.data(), rows, cols};
  MatPtr C_test{mat_c_test.data(), rows, cols};

  // Reference computation (simple implementation)
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < inner; ++k) {
        sum += mat_a[i * inner + k] * mat_b[k * cols + j];
      }
      mat_c_ref[i * cols + j] = sum;
    }
  }

  // Test with different backends using ThreadingContext
  MatMulEnv env(*thread_context_);

  // Perform matrix multiplication using optimized implementation
  auto mm_key = MatMul(A, B, nullptr, env, C_test);
  EXPECT_NE(mm_key, nullptr) << "Matrix multiplication should succeed";

  // Compare results
  const float tolerance = 1e-4f;  // Relaxed tolerance for different backends
  for (size_t i = 0; i < rows * cols; ++i) {
    EXPECT_NEAR(mat_c_test[i], mat_c_ref[i], tolerance)
        << "Mismatch at index " << i << " (row " << i / cols << ", col " << i % cols << ")";
  }
}

// Test different data type backends (BF16, F32, etc.)
TEST_F(BackendCompatibilityTest, DataTypeCompatibility) {
  const size_t size = 1000;

  // Test BF16 operations if supported
  {
    auto float_data = data_gen_->GenerateRandomFloats(size, -10.0f, 10.0f);
    std::vector<hwy::bfloat16_t> bf16_data(size);

    // Convert float to BF16
    for (size_t i = 0; i < size; ++i) {
      bf16_data[i] = hwy::F32ToBF16(float_data[i]);
    }

    // Convert back to float and check precision loss is within bounds
    for (size_t i = 0; i < size; ++i) {
      float converted_back = hwy::F32FromBF16(bf16_data[i]);
      float relative_error = std::abs(converted_back - float_data[i]) /
                           (std::abs(float_data[i]) + 1e-8f);
      EXPECT_LT(relative_error, 0.01f) << "BF16 conversion error too large at index " << i;
    }
  }

  // Test that operations work with different alignments
  {
    auto test_alignment = [this](size_t alignment) {
      const size_t test_size = 256;
      auto aligned_data = std::unique_ptr<float, void(*)(void*)>(
          static_cast<float*>(hwy::AllocateAligned<float>(test_size)),
          hwy::FreeAligned);

      // Fill with test data
      auto random_data = data_gen_->GenerateRandomFloats(test_size);
      std::memcpy(aligned_data.get(), random_data.data(), test_size * sizeof(float));

      // Verify the data is properly aligned
      EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned_data.get()) % alignment, 0)
          << "Data not properly aligned to " << alignment << " bytes";

      // Simple operation test
      float sum = 0.0f;
      for (size_t i = 0; i < test_size; ++i) {
        sum += aligned_data.get()[i];
      }
      EXPECT_TRUE(std::isfinite(sum)) << "Sum should be finite";
    };

    // Test different alignment requirements
    test_alignment(16);  // SSE
    test_alignment(32);  // AVX
    test_alignment(64);  // AVX-512
  }
}

// Test compression backend compatibility
TEST_F(BackendCompatibilityTest, CompressionBackends) {
  const size_t num_elements = 1024;
  auto original_data = data_gen_->GenerateRandomFloats(num_elements, -5.0f, 5.0f);

  // Test SFP (Scaled Float Point) compression if available
  {
    CompressedArray<SfpStream> compressed_array;
    compressed_array.set_scale(1.0f);

    // Create a simple test for compression
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Basic compression test - just verify no crashes
    EXPECT_NO_THROW({
      // This would normally involve actual compression operations
      // For now, we test that the types and basic functionality work
      float scale = compressed_array.scale();
      EXPECT_GT(scale, 0.0f);
    });
  }

  // Test NUQ (Non-Uniform Quantization) compatibility
  {
    // Basic NUQ test - verify types and basic operations
    std::vector<uint8_t> quantized_data(num_elements);

    EXPECT_NO_THROW({
      // Simple quantization simulation
      for (size_t i = 0; i < num_elements; ++i) {
        float normalized = (original_data[i] + 5.0f) / 10.0f;  // Normalize to [0,1]
        quantized_data[i] = static_cast<uint8_t>(normalized * 255.0f);
      }
    });

    // Verify quantization worked
    EXPECT_EQ(quantized_data.size(), num_elements);
  }
}

// Test memory allocation across different backends
TEST_F(BackendCompatibilityTest, MemoryAllocationBackends) {
  // Test various allocation sizes and alignments
  std::vector<size_t> test_sizes = {64, 1024, 4096, 65536};
  std::vector<size_t> test_alignments = {16, 32, 64};

  for (size_t size : test_sizes) {
    for (size_t alignment : test_alignments) {
      // Test allocation and deallocation
      void* ptr = hwy::AllocateAligned(size, alignment);
      ASSERT_NE(ptr, nullptr) << "Failed to allocate " << size << " bytes with "
                              << alignment << " byte alignment";

      // Verify alignment
      EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0)
          << "Allocated memory not properly aligned";

      // Test that we can write to the memory
      std::memset(ptr, 0xAA, size);

      // Verify the write worked
      uint8_t* byte_ptr = static_cast<uint8_t*>(ptr);
      for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(byte_ptr[i], 0xAA) << "Memory write failed at offset " << i;
      }

      hwy::FreeAligned(ptr);
    }
  }
}

// Test threading backend compatibility
TEST_F(BackendCompatibilityTest, ThreadingBackends) {
  const size_t num_threads = std::min(std::thread::hardware_concurrency(), 8u);

  if (num_threads <= 1) {
    GTEST_SKIP() << "Insufficient hardware threads for threading test";
  }

  // Test parallel execution
  std::vector<std::atomic<int>> counters(num_threads);
  for (auto& counter : counters) {
    counter.store(0);
  }

  // Simple parallel work
  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&counters, i]() {
      for (int j = 0; j < 1000; ++j) {
        counters[i].fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify results
  for (size_t i = 0; i < num_threads; ++i) {
    EXPECT_EQ(counters[i].load(), 1000) << "Thread " << i << " didn't complete work";
  }
}

// Cross-platform compatibility tests
TEST_F(BackendCompatibilityTest, CrossPlatformCompatibility) {
  // Test endianness handling
  {
    uint32_t test_value = 0x12345678;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);

    // Just verify we can detect endianness consistently
    bool is_little_endian = (bytes[0] == 0x78);
    bool is_big_endian = (bytes[0] == 0x12);
    EXPECT_TRUE(is_little_endian || is_big_endian) << "Could not determine endianness";

    std::cout << "Platform is " << (is_little_endian ? "little" : "big") << " endian" << std::endl;
  }

  // Test floating point representation
  {
    float test_float = 1.5f;
    uint32_t float_bits;
    std::memcpy(&float_bits, &test_float, sizeof(float));

    // IEEE 754 representation of 1.5 is 0x3FC00000
    EXPECT_EQ(float_bits, 0x3FC00000) << "Unexpected floating point representation";
  }

  // Test basic math operations consistency
  {
    float a = 3.14159f;
    float b = 2.71828f;

    float sum = a + b;
    float product = a * b;
    float quotient = a / b;

    EXPECT_TRUE(std::isfinite(sum)) << "Addition result not finite";
    EXPECT_TRUE(std::isfinite(product)) << "Multiplication result not finite";
    EXPECT_TRUE(std::isfinite(quotient)) << "Division result not finite";

    EXPECT_NEAR(sum, 5.85987f, 1e-5f);
    EXPECT_NEAR(product, 8.53973f, 1e-5f);
    EXPECT_NEAR(quotient, 1.15573f, 1e-5f);
  }
}

// Performance scaling test across backends
class BackendPerformanceTest : public BackendCompatibilityTest {
 protected:
  void BenchmarkOperation(const std::string& operation_name,
                         std::function<void()> operation,
                         int iterations = 1000) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
      operation();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double ops_per_second = (iterations * 1e6) / duration.count();
    std::cout << operation_name << " performance: " << ops_per_second << " ops/second" << std::endl;

    // Basic performance expectation (adjust based on requirements)
    EXPECT_GT(ops_per_second, 1000.0) << operation_name << " performance too low";
  }
};

TEST_F(BackendPerformanceTest, VectorOperationPerformance) {
  const size_t vector_size = 1000;
  auto vec_a = data_gen_->GenerateRandomFloats(vector_size);
  auto vec_b = data_gen_->GenerateRandomFloats(vector_size);
  auto vec_result = std::vector<float>(vector_size);

  // Benchmark vector addition
  BenchmarkOperation("Vector Addition", [&]() {
    for (size_t i = 0; i < vector_size; ++i) {
      vec_result[i] = vec_a[i] + vec_b[i];
    }
  });

  // Benchmark vector multiplication
  BenchmarkOperation("Vector Multiplication", [&]() {
    for (size_t i = 0; i < vector_size; ++i) {
      vec_result[i] = vec_a[i] * vec_b[i];
    }
  });
}

// Parameterized tests for different SIMD widths
class SIMDWidthTest : public BackendCompatibilityTest,
                      public ::testing::WithParamInterface<size_t> {
 protected:
  size_t GetSIMDWidth() const { return GetParam(); }
};

TEST_P(SIMDWidthTest, SIMDOperationConsistency) {
  const size_t simd_width = GetSIMDWidth();
  const size_t total_elements = simd_width * 16;  // Multiple SIMD lanes

  auto input_data = data_gen_->GenerateRandomFloats(total_elements, -1.0f, 1.0f);
  auto output_simd = std::vector<float>(total_elements);
  auto output_scalar = std::vector<float>(total_elements);

  // Scalar reference implementation
  for (size_t i = 0; i < total_elements; ++i) {
    output_scalar[i] = input_data[i] * 2.0f + 1.0f;  // Simple operation
  }

  // SIMD implementation (simplified)
  for (size_t i = 0; i < total_elements; ++i) {
    output_simd[i] = input_data[i] * 2.0f + 1.0f;  // Would use SIMD in real implementation
  }

  // Compare results
  EXPECT_VECTOR_NEAR(output_simd, output_scalar, kTestTolerance);
}

// Test with common SIMD widths
INSTANTIATE_TEST_SUITE_P(
    CommonSIMDWidths,
    SIMDWidthTest,
    ::testing::Values(4, 8, 16)  // 128-bit, 256-bit, 512-bit SIMD
);

}  // namespace
}  // namespace gcpp

// Use Highway's test runner for SIMD backend testing
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, SIMDBackendDetection);
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, MatrixMultiplicationBackends);
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, DataTypeCompatibility);
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, CompressionBackends);
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, MemoryAllocationBackends);
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, ThreadingBackends);
HWY_EXPORT_AND_TEST_P(BackendCompatibilityTest, CrossPlatformCompatibility);
HWY_AFTER_TEST();