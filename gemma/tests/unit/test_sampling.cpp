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
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

// Include Gemma ops for sampling functions
#include "ops/ops-inl.h"
#include "ops/ops.h"
#include "util/basics.h"
#include "hwy/tests/hwy_gtest.h"
#include "../utils/test_common.h"

namespace gcpp {
namespace {

using namespace test_utils;

class SamplingTest : public GemmaTestBase {
 protected:
  void SetUp() override {
    GemmaTestBase::SetUp();
    gen_.seed(test_seed_);
    // Initialize profiler for testing
    profiler_ = &hwy::Profiler::Get();
  }

  static constexpr uint32_t test_seed_ = 0x12345678;
  static constexpr size_t worker_id_ = 0;
  std::mt19937 gen_;
  hwy::Profiler* profiler_;
};

// Test basic softmax functionality
TEST_F(SamplingTest, SoftmaxBasic) {
  const size_t size = 10;
  auto logits = data_gen_->GenerateRandomFloats(size, -5.0f, 5.0f);

  // Apply softmax with temperature = 1.0
  float temperature = 1.0f;
  Softmax(logits.data(), size, *profiler_, worker_id_, temperature);

  // Check normalization: sum should be approximately 1.0
  float sum = 0.0f;
  for (float prob : logits) {
    EXPECT_GE(prob, 0.0f) << "Probability should be non-negative";
    sum += prob;
  }
  EXPECT_NEAR(sum, 1.0f, kTestTolerance) << "Probabilities should sum to 1.0";
}

// Test softmax with different temperatures
TEST_F(SamplingTest, SoftmaxTemperature) {
  const size_t size = 100;
  auto original_logits = data_gen_->GenerateRandomFloats(size, -2.0f, 2.0f);

  // Test with low temperature (sharper distribution)
  {
    auto logits = original_logits;
    float temperature = 0.1f;
    Softmax(logits.data(), size, *profiler_, worker_id_, temperature);

    // Find max probability
    float max_prob = *std::max_element(logits.begin(), logits.end());
    EXPECT_GT(max_prob, 0.5f) << "Low temperature should create sharper distribution";
  }

  // Test with high temperature (flatter distribution)
  {
    auto logits = original_logits;
    float temperature = 10.0f;
    Softmax(logits.data(), size, *profiler_, worker_id_, temperature);

    // Distribution should be more uniform
    float max_prob = *std::max_element(logits.begin(), logits.end());
    float min_prob = *std::min_element(logits.begin(), logits.end());
    EXPECT_LT(max_prob - min_prob, 0.1f) << "High temperature should create flatter distribution";
  }
}

// Test temperature = 0 edge case
TEST_F(SamplingTest, ZeroTemperature) {
  const size_t size = 50;
  auto logits = data_gen_->GenerateRandomFloats(size, -3.0f, 3.0f);

  // Find the argmax before softmax
  auto max_it = std::max_element(logits.begin(), logits.end());
  size_t expected_argmax = std::distance(logits.begin(), max_it);

  // Apply softmax with temperature = 0
  float temperature = 0.0f;
  Softmax(logits.data(), size, *profiler_, worker_id_, temperature);

  // The maximum element should have probability close to 1
  EXPECT_NEAR(logits[expected_argmax], 1.0f, kTestTolerance);

  // All other elements should be close to 0
  for (size_t i = 0; i < size; ++i) {
    if (i != expected_argmax) {
      EXPECT_NEAR(logits[i], 0.0f, kTestTolerance);
    }
  }
}

// Test top-K sampling
TEST_F(SamplingTest, TopKSampling) {
  const size_t vocab_size = 1000;
  const size_t k = 5;
  auto probabilities = data_gen_->GenerateNormalizedProbabilities(vocab_size);

  std::function<bool(int, float)> accept_all = [](int, float) { return true; };

  float temperature = 1.0f;
  std::vector<int> sampled_tokens;

  // Sample multiple times and verify all results are within top-K
  for (int trial = 0; trial < 100; ++trial) {
    int token = SampleTopK(probabilities.data(), k, vocab_size, gen_, temperature, accept_all);
    sampled_tokens.push_back(token);
    EXPECT_GE(token, 0) << "Token should be valid";
    EXPECT_LT(token, vocab_size) << "Token should be within vocabulary";
  }

  // Verify that we get diverse samples (not always the same token)
  std::set<int> unique_tokens(sampled_tokens.begin(), sampled_tokens.end());
  EXPECT_GT(unique_tokens.size(), 1) << "Should sample diverse tokens";
  EXPECT_LE(unique_tokens.size(), k) << "Should only sample from top-K";
}

// Test top-K sampling with acceptance filter
TEST_F(SamplingTest, TopKSamplingWithFilter) {
  const size_t vocab_size = 100;
  const size_t k = 10;
  auto probabilities = data_gen_->GenerateNormalizedProbabilities(vocab_size);

  // Only accept even tokens
  std::function<bool(int, float)> accept_even = [](int token, float) {
    return token % 2 == 0;
  };

  float temperature = 1.0f;

  // Sample multiple times and verify all results are even
  for (int trial = 0; trial < 50; ++trial) {
    int token = SampleTopK(probabilities.data(), k, vocab_size, gen_, temperature, accept_even);
    EXPECT_TRUE(token % 2 == 0) << "Should only sample even tokens";
  }
}

// Test top-1 sampling (argmax)
TEST_F(SamplingTest, Top1Sampling) {
  const size_t vocab_size = 200;
  auto probabilities = data_gen_->GenerateNormalizedProbabilities(vocab_size);

  // Find the true argmax
  auto max_it = std::max_element(probabilities.begin(), probabilities.end());
  int expected_argmax = std::distance(probabilities.begin(), max_it);

  std::function<bool(int, float)> accept_all = [](int, float) { return true; };
  float temperature = 0.0f; // Zero temperature for deterministic argmax

  // Top-1 with zero temperature should always return argmax
  for (int trial = 0; trial < 10; ++trial) {
    int token = SampleTopK(probabilities.data(), 1, vocab_size, gen_, temperature, accept_all);
    EXPECT_EQ(token, expected_argmax) << "Top-1 with zero temperature should return argmax";
  }
}

// Test fused softmax and top-K sampling
TEST_F(SamplingTest, FusedSoftmaxTopK) {
  const size_t vocab_size = 500;
  const size_t k = 3;
  auto logits = data_gen_->GenerateRandomFloats(vocab_size, -5.0f, 5.0f);

  std::function<bool(int, float)> accept_all = [](int, float) { return true; };
  float temperature = 1.0f;

  // Test fused operation
  auto result = FusedSoftmaxAndSampleTopK(logits.data(), k, vocab_size, gen_,
                                          temperature, accept_all, *profiler_, worker_id_);

  EXPECT_GE(result.token, 0) << "Token should be valid";
  EXPECT_LT(result.token, vocab_size) << "Token should be within vocabulary";
  EXPECT_GT(result.prob, 0.0f) << "Probability should be positive";
  EXPECT_LE(result.prob, 1.0f) << "Probability should not exceed 1.0";
}

// Test argmax functionality
TEST_F(SamplingTest, SampleArgmax) {
  const size_t vocab_size = 50;
  auto probabilities = data_gen_->GenerateNormalizedProbabilities(vocab_size);

  // Find expected argmax
  auto max_it = std::max_element(probabilities.begin(), probabilities.end());
  size_t expected_argmax = std::distance(probabilities.begin(), max_it);

  size_t result = SampleArgmax(probabilities.data(), vocab_size);
  EXPECT_EQ(result, expected_argmax) << "SampleArgmax should return the index of maximum element";
}

// Test temperature scaling effects on distribution entropy
TEST_F(SamplingTest, TemperatureEntropyEffect) {
  const size_t size = 100;
  auto original_logits = data_gen_->GenerateRandomFloats(size, -2.0f, 2.0f);

  auto calculate_entropy = [](const std::vector<float>& probs) {
    float entropy = 0.0f;
    for (float p : probs) {
      if (p > 0.0f) {
        entropy -= p * std::log2(p);
      }
    }
    return entropy;
  };

  // Low temperature should decrease entropy
  {
    auto logits = original_logits;
    Softmax(logits.data(), size, *profiler_, worker_id_, 0.5f);
    float low_temp_entropy = calculate_entropy(logits);

    logits = original_logits;
    Softmax(logits.data(), size, *profiler_, worker_id_, 1.0f);
    float normal_entropy = calculate_entropy(logits);

    EXPECT_LT(low_temp_entropy, normal_entropy) << "Lower temperature should reduce entropy";
  }

  // High temperature should increase entropy
  {
    auto logits = original_logits;
    Softmax(logits.data(), size, *profiler_, worker_id_, 1.0f);
    float normal_entropy = calculate_entropy(logits);

    logits = original_logits;
    Softmax(logits.data(), size, *profiler_, worker_id_, 2.0f);
    float high_temp_entropy = calculate_entropy(logits);

    EXPECT_GT(high_temp_entropy, normal_entropy) << "Higher temperature should increase entropy";
  }
}

// Test edge cases and robustness
TEST_F(SamplingTest, EdgeCases) {
  // Test with single element
  {
    std::vector<float> single_logit = {1.0f};
    Softmax(single_logit.data(), 1, *profiler_, worker_id_, 1.0f);
    EXPECT_NEAR(single_logit[0], 1.0f, kTestTolerance);
  }

  // Test with all zeros
  {
    std::vector<float> zero_logits(10, 0.0f);
    Softmax(zero_logits.data(), 10, *profiler_, worker_id_, 1.0f);
    for (float prob : zero_logits) {
      EXPECT_NEAR(prob, 0.1f, kTestTolerance); // Should be uniform
    }
  }

  // Test with very large values (numerical stability)
  {
    std::vector<float> large_logits = {100.0f, 101.0f, 99.0f};
    Softmax(large_logits.data(), 3, *profiler_, worker_id_, 1.0f);

    // Should not produce NaN or infinity
    for (float prob : large_logits) {
      EXPECT_TRUE(std::isfinite(prob)) << "Probability should be finite";
      EXPECT_GE(prob, 0.0f) << "Probability should be non-negative";
    }

    // Sum should still be 1
    float sum = std::accumulate(large_logits.begin(), large_logits.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, kTestTolerance);
  }
}

// Parameterized test for different model types
class SamplingModelTest : public ModelTypeTest {};

TEST_P(SamplingModelTest, ModelSpecificSampling) {
  const size_t vocab_size = model_config_.vocab_size;
  auto probabilities = data_gen_->GenerateNormalizedProbabilities(vocab_size);

  std::function<bool(int, float)> accept_all = [](int, float) { return true; };

  // Test sampling with model-specific vocabulary size
  int token = SampleTopK(probabilities.data(), 10, vocab_size, gen_, 1.0f, accept_all);
  EXPECT_GE(token, 0);
  EXPECT_LT(token, static_cast<int>(vocab_size));
}

// Instantiate parameterized tests for different models
INSTANTIATE_TEST_SUITE_P(
    AllModels,
    SamplingModelTest,
    ::testing::Values(
        Model::GEMMA2_2B,
        Model::GEMMA2_9B,
        Model::GEMMA2_27B,
        Model::GRIFFIN_2B
    )
);

}  // namespace
}  // namespace gcpp

// Use Highway's test framework integration
HWY_EXPORT_AND_TEST_P(SamplingTest, SoftmaxBasic);
HWY_EXPORT_AND_TEST_P(SamplingTest, SoftmaxTemperature);
HWY_EXPORT_AND_TEST_P(SamplingTest, ZeroTemperature);
HWY_EXPORT_AND_TEST_P(SamplingTest, TopKSampling);
HWY_EXPORT_AND_TEST_P(SamplingTest, TopKSamplingWithFilter);
HWY_EXPORT_AND_TEST_P(SamplingTest, Top1Sampling);
HWY_EXPORT_AND_TEST_P(SamplingTest, FusedSoftmaxTopK);
HWY_EXPORT_AND_TEST_P(SamplingTest, SampleArgmax);
HWY_EXPORT_AND_TEST_P(SamplingTest, TemperatureEntropyEffect);
HWY_EXPORT_AND_TEST_P(SamplingTest, EdgeCases);
HWY_AFTER_TEST();