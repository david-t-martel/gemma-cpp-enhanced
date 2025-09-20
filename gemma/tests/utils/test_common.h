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

#ifndef GEMMA_TESTS_UTILS_TEST_COMMON_H_
#define GEMMA_TESTS_UTILS_TEST_COMMON_H_

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <vector>
#include <string>
#include <memory>
#include "util/allocator.h"
#include "gemma/configs.h"

namespace gcpp {
namespace test_utils {

// Test configuration constants
constexpr size_t kDefaultVocabSize = 256000;
constexpr size_t kDefaultSequenceLength = 2048;
constexpr float kTestTolerance = 1e-6f;

// Mock model configurations for testing
class MockModelConfig {
 public:
  static ModelConfig CreateTestConfig(Model model = Model::GEMMA2_2B) {
    ModelConfig config;
    // Initialize with basic test parameters
    config.model = model;
    config.vocab_size = kDefaultVocabSize;
    config.seq_len = kDefaultSequenceLength;
    config.model_dim = 2048;
    config.intermediate_size = 16384;
    config.num_layers = 18;
    config.num_heads = 8;
    config.num_kv_heads = 1;
    config.head_dim = 256;
    return config;
  }
};

// Test data generators
class TestDataGenerator {
 public:
  explicit TestDataGenerator(uint32_t seed = 42) : gen_(seed) {}

  // Generate random float vector
  std::vector<float> GenerateRandomFloats(size_t count, float min = -1.0f, float max = 1.0f) {
    std::uniform_real_distribution<float> dist(min, max);
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] = dist(gen_);
    }
    return result;
  }

  // Generate random int vector (for tokens)
  std::vector<int> GenerateRandomTokens(size_t count, int vocab_size = kDefaultVocabSize) {
    std::uniform_int_distribution<int> dist(0, vocab_size - 1);
    std::vector<int> result(count);
    for (size_t i = 0; i < count; ++i) {
      result[i] = dist(gen_);
    }
    return result;
  }

  // Generate test probabilities (normalized)
  std::vector<float> GenerateNormalizedProbabilities(size_t count) {
    auto probs = GenerateRandomFloats(count, 0.0f, 1.0f);
    float sum = 0.0f;
    for (float p : probs) sum += p;
    for (float& p : probs) p /= sum;
    return probs;
  }

 private:
  std::mt19937 gen_;
};

// Memory allocation helpers for tests
class TestAllocator {
 public:
  static std::unique_ptr<Allocator> CreateTestAllocator() {
    return std::make_unique<Allocator>();
  }

  template<typename T>
  static T* AllocateAligned(Allocator& allocator, size_t count) {
    return allocator.AllocateArray<T>(count);
  }
};

// Test fixtures
class GemmaTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = TestAllocator::CreateTestAllocator();
    data_gen_ = std::make_unique<TestDataGenerator>();
  }

  void TearDown() override {
    allocator_.reset();
    data_gen_.reset();
  }

  std::unique_ptr<Allocator> allocator_;
  std::unique_ptr<TestDataGenerator> data_gen_;
};

// Parameterized test fixture for different model types
class ModelTypeTest : public GemmaTestBase,
                      public ::testing::WithParamInterface<Model> {
 protected:
  void SetUp() override {
    GemmaTestBase::SetUp();
    model_config_ = MockModelConfig::CreateTestConfig(GetParam());
  }

  ModelConfig model_config_;
};

// Helper macros for floating point comparisons
#define EXPECT_FLOAT_NEAR(val1, val2, tolerance) \
  EXPECT_NEAR(val1, val2, tolerance)

#define EXPECT_VECTOR_NEAR(vec1, vec2, tolerance) \
  do { \
    ASSERT_EQ(vec1.size(), vec2.size()); \
    for (size_t i = 0; i < vec1.size(); ++i) { \
      EXPECT_NEAR(vec1[i], vec2[i], tolerance) << "at index " << i; \
    } \
  } while(0)

// Performance measurement utilities
class PerformanceMeasurer {
 public:
  static double MeasureExecutionTime(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Return milliseconds
  }
};

}  // namespace test_utils
}  // namespace gcpp

#endif  // GEMMA_TESTS_UTILS_TEST_COMMON_H_