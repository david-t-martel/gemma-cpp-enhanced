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
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#include "../utils/test_common.h"
#include "gemma/configs.h"
#include "gemma/weights.h"
#include "gemma/tokenizer.h"
#include "gemma/model_store.h"
#include "io/blob_store.h"
#include "io/io.h"
#include "util/allocator.h"

namespace gcpp {
namespace {

using namespace test_utils;
namespace fs = std::filesystem;

// Mock weight loader for testing without actual model files
class MockWeightLoader {
 public:
  static std::unique_ptr<BlobStore> CreateMockWeights(const ModelConfig& config,
                                                      Allocator& allocator) {
    auto blob_store = std::make_unique<BlobStore>();

    // Create mock weight tensors with correct dimensions
    const size_t vocab_embed_size = config.vocab_size * config.model_dim;
    const size_t layer_weight_size = config.model_dim * config.intermediate_size;

    // Allocate and initialize mock embeddings
    float* embeddings = allocator.AllocateArray<float>(vocab_embed_size);
    std::fill(embeddings, embeddings + vocab_embed_size, 0.1f);

    // Create mock layer weights
    for (size_t layer = 0; layer < config.num_layers; ++layer) {
      float* layer_weights = allocator.AllocateArray<float>(layer_weight_size);
      std::fill(layer_weights, layer_weights + layer_weight_size, 0.05f);
    }

    return blob_store;
  }
};

// Mock tokenizer for testing
class MockTokenizer {
 public:
  static std::unique_ptr<SentencePieceProcessor> CreateMockTokenizer() {
    // Create a minimal mock tokenizer
    auto tokenizer = std::make_unique<SentencePieceProcessor>();

    // In a real implementation, this would load actual tokenizer data
    // For testing, we'll create a minimal functional tokenizer

    return tokenizer;
  }

  static std::vector<int> MockTokenize(const std::string& text) {
    // Simple mock tokenization: split on spaces and assign sequential IDs
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    int token_id = 1000;  // Start from arbitrary ID

    while (iss >> word) {
      tokens.push_back(token_id++);
    }

    if (tokens.empty()) {
      tokens.push_back(1000);  // Always return at least one token
    }

    return tokens;
  }

  static std::string MockDetokenize(const std::vector<int>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i > 0) result += " ";
      result += "token_" + std::to_string(tokens[i]);
    }
    return result;
  }
};

class ModelLoadingTest : public GemmaTestBase {
 protected:
  void SetUp() override {
    GemmaTestBase::SetUp();

    // Create temporary directory for test files
    test_dir_ = fs::temp_directory_path() / "gemma_test";
    fs::create_directories(test_dir_);

    // Create various model configurations for testing
    configs_ = {
        MockModelConfig::CreateTestConfig(Model::GEMMA2_2B),
        MockModelConfig::CreateTestConfig(Model::GEMMA2_9B),
        MockModelConfig::CreateTestConfig(Model::GRIFFIN_2B)
    };
  }

  void TearDown() override {
    // Clean up test directory
    if (fs::exists(test_dir_)) {
      std::error_code ec;
      fs::remove_all(test_dir_, ec);
      // Ignore errors during cleanup
    }

    GemmaTestBase::TearDown();
  }

  void CreateMockWeightFile(const fs::path& file_path, size_t size_bytes) {
    std::ofstream file(file_path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to create mock weight file: " << file_path;

    // Write some dummy data
    std::vector<char> dummy_data(size_bytes, 0x42);
    file.write(dummy_data.data(), size_bytes);
    file.close();

    ASSERT_TRUE(fs::exists(file_path)) << "Mock weight file was not created: " << file_path;
  }

  void CreateMockTokenizerFile(const fs::path& file_path) {
    std::ofstream file(file_path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to create mock tokenizer file: " << file_path;

    // Write minimal SentencePiece model data (simplified)
    const char mock_sp_data[] = {
        0x08, 0x01, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74  // Minimal protobuf data
    };
    file.write(mock_sp_data, sizeof(mock_sp_data));
    file.close();

    ASSERT_TRUE(fs::exists(file_path)) << "Mock tokenizer file was not created: " << file_path;
  }

  fs::path test_dir_;
  std::vector<ModelConfig> configs_;
};

// Test basic model configuration validation
TEST_F(ModelLoadingTest, ModelConfigValidation) {
  for (const auto& config : configs_) {
    // Test required fields are set
    EXPECT_GT(config.vocab_size, 0) << "Vocabulary size should be positive";
    EXPECT_GT(config.seq_len, 0) << "Sequence length should be positive";
    EXPECT_GT(config.model_dim, 0) << "Model dimension should be positive";
    EXPECT_GT(config.num_layers, 0) << "Number of layers should be positive";
    EXPECT_GT(config.num_heads, 0) << "Number of heads should be positive";
    EXPECT_GT(config.head_dim, 0) << "Head dimension should be positive";

    // Test consistency constraints
    EXPECT_EQ(config.model_dim, config.num_heads * config.head_dim)
        << "Model dimension should equal num_heads * head_dim";

    EXPECT_GE(config.num_kv_heads, 1) << "Number of KV heads should be at least 1";
    EXPECT_LE(config.num_kv_heads, config.num_heads)
        << "Number of KV heads should not exceed total heads";

    // Test model specifier
    std::string specifier = config.Specifier();
    EXPECT_FALSE(specifier.empty()) << "Model specifier should not be empty";

    std::cout << "Model: " << specifier
              << ", Vocab: " << config.vocab_size
              << ", Seq len: " << config.seq_len
              << ", Layers: " << config.num_layers << std::endl;
  }
}

// Test weight file loading simulation
TEST_F(ModelLoadingTest, WeightFileLoading) {
  const auto& config = configs_[0];  // Use first config for testing

  // Create mock weight files
  fs::path weight_file = test_dir_ / "test_weights.sbs";
  size_t expected_weight_size = EstimateWeightSize(config);

  CreateMockWeightFile(weight_file, expected_weight_size);

  // Test file existence and size
  EXPECT_TRUE(fs::exists(weight_file)) << "Weight file should exist";
  EXPECT_GE(fs::file_size(weight_file), expected_weight_size)
      << "Weight file should have expected size";

  // Test file reading
  std::ifstream file(weight_file, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(file.is_open()) << "Should be able to open weight file";

  size_t file_size = file.tellg();
  EXPECT_EQ(file_size, expected_weight_size) << "File size should match expected size";

  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(1024);
  file.read(buffer.data(), buffer.size());

  EXPECT_EQ(file.gcount(), 1024) << "Should be able to read from file";

  // Verify file content (our dummy data should be 0x42)
  for (char byte : buffer) {
    EXPECT_EQ(byte, 0x42) << "File content should match written data";
  }
}

// Test tokenizer loading simulation
TEST_F(ModelLoadingTest, TokenizerLoading) {
  fs::path tokenizer_file = test_dir_ / "test_tokenizer.model";
  CreateMockTokenizerFile(tokenizer_file);

  // Test file existence
  EXPECT_TRUE(fs::exists(tokenizer_file)) << "Tokenizer file should exist";

  // Test basic tokenization functionality (mocked)
  std::string test_text = "Hello world test";
  auto tokens = MockTokenizer::MockTokenize(test_text);

  EXPECT_FALSE(tokens.empty()) << "Tokenization should produce tokens";
  EXPECT_GT(tokens.size(), 0) << "Should have at least one token";

  // Test detokenization
  std::string detokenized = MockTokenizer::MockDetokenize(tokens);
  EXPECT_FALSE(detokenized.empty()) << "Detokenization should produce text";

  std::cout << "Original: \"" << test_text << "\"" << std::endl;
  std::cout << "Tokens: [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << tokens[i];
  }
  std::cout << "]" << std::endl;
  std::cout << "Detokenized: \"" << detokenized << "\"" << std::endl;
}

// Test model weight allocation and initialization
TEST_F(ModelLoadingTest, WeightAllocation) {
  for (const auto& config : configs_) {
    // Calculate memory requirements
    size_t total_params = EstimateParameterCount(config);
    size_t memory_bytes = total_params * sizeof(float);

    std::cout << "Model " << config.Specifier()
              << " estimated parameters: " << total_params
              << " (" << memory_bytes / (1024 * 1024) << " MB)" << std::endl;

    // Test allocation
    ASSERT_NO_THROW({
      auto weights = MockWeightLoader::CreateMockWeights(config, *allocator_);
      EXPECT_NE(weights, nullptr) << "Weight allocation should succeed";
    }) << "Weight allocation failed for " << config.Specifier();
  }
}

// Test single-file format loading
TEST_F(ModelLoadingTest, SingleFileFormat) {
  const auto& config = configs_[0];

  // Create mock single-file format (weights + tokenizer)
  fs::path single_file = test_dir_ / "test_model_single.sbs";

  // Estimate combined size
  size_t weight_size = EstimateWeightSize(config);
  size_t tokenizer_size = 1024;  // Small tokenizer
  size_t total_size = weight_size + tokenizer_size + 256;  // Extra for headers

  CreateMockWeightFile(single_file, total_size);

  // Test file properties
  EXPECT_TRUE(fs::exists(single_file)) << "Single file should exist";
  EXPECT_GE(fs::file_size(single_file), total_size) << "File should have expected size";

  // Test that we can read the file header (mocked)
  std::ifstream file(single_file, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << "Should be able to open single file";

  // Read first few bytes as header
  std::vector<uint8_t> header(32);
  file.read(reinterpret_cast<char*>(header.data()), header.size());
  EXPECT_EQ(file.gcount(), 32) << "Should be able to read header";
}

// Test error handling for missing files
TEST_F(ModelLoadingTest, MissingFileHandling) {
  fs::path nonexistent_file = test_dir_ / "nonexistent.sbs";

  // Ensure file doesn't exist
  EXPECT_FALSE(fs::exists(nonexistent_file)) << "Test file should not exist";

  // Test error handling
  std::ifstream file(nonexistent_file);
  EXPECT_FALSE(file.is_open()) << "Opening nonexistent file should fail";
  EXPECT_TRUE(file.fail()) << "File stream should be in fail state";
}

// Test corrupted file handling
TEST_F(ModelLoadingTest, CorruptedFileHandling) {
  fs::path corrupted_file = test_dir_ / "corrupted.sbs";

  // Create file with invalid data
  std::ofstream file(corrupted_file, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << "Should be able to create test file";

  // Write invalid header or truncated data
  const char invalid_data[] = {0xFF, 0xFF, 0xFF, 0xFF};
  file.write(invalid_data, sizeof(invalid_data));
  file.close();

  EXPECT_TRUE(fs::exists(corrupted_file)) << "Corrupted file should exist";
  EXPECT_LT(fs::file_size(corrupted_file), 1024) << "Corrupted file should be small";

  // Test reading corrupted file
  std::ifstream read_file(corrupted_file, std::ios::binary);
  ASSERT_TRUE(read_file.is_open()) << "Should be able to open corrupted file";

  std::vector<char> buffer(1024);
  read_file.read(buffer.data(), buffer.size());
  EXPECT_LT(read_file.gcount(), 1024) << "Should not be able to read full buffer from small file";
}

// Test memory-mapped file loading simulation
TEST_F(ModelLoadingTest, MemoryMappedLoading) {
  const auto& config = configs_[0];
  fs::path weight_file = test_dir_ / "test_mmap.sbs";
  size_t file_size = EstimateWeightSize(config);

  CreateMockWeightFile(weight_file, file_size);

  // Test file properties for memory mapping
  EXPECT_TRUE(fs::exists(weight_file)) << "File should exist for memory mapping";
  EXPECT_GE(fs::file_size(weight_file), file_size) << "File should have correct size";

  // Test that file is readable (prerequisite for memory mapping)
  std::ifstream file(weight_file, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << "File should be openable";

  // Check file permissions (simplified test)
  auto perms = fs::status(weight_file).permissions();
  bool readable = (perms & fs::perms::owner_read) != fs::perms::none;
  EXPECT_TRUE(readable) << "File should be readable for memory mapping";
}

// Performance test for model loading
class ModelLoadingPerformanceTest : public ModelLoadingTest {
 protected:
  void BenchmarkFileOperation(const std::string& operation_name,
                             std::function<void()> operation) {
    auto start_time = std::chrono::high_resolution_clock::now();

    operation();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << operation_name << " took " << duration.count() << " ms" << std::endl;

    // Basic performance expectation (adjust based on requirements)
    EXPECT_LT(duration.count(), 5000) << operation_name << " took too long (> 5 seconds)";
  }
};

TEST_F(ModelLoadingPerformanceTest, FileCreationPerformance) {
  const auto& config = configs_[0];
  size_t file_size = EstimateWeightSize(config);

  BenchmarkFileOperation("Large file creation", [this, file_size]() {
    fs::path perf_file = test_dir_ / "perf_test.sbs";
    CreateMockWeightFile(perf_file, file_size);
  });
}

TEST_F(ModelLoadingPerformanceTest, FileReadingPerformance) {
  const auto& config = configs_[0];
  size_t file_size = EstimateWeightSize(config);
  fs::path perf_file = test_dir_ / "read_perf_test.sbs";

  CreateMockWeightFile(perf_file, file_size);

  BenchmarkFileOperation("Large file reading", [perf_file, file_size]() {
    std::ifstream file(perf_file, std::ios::binary);
    ASSERT_TRUE(file.is_open());

    const size_t buffer_size = 64 * 1024;  // 64KB buffer
    std::vector<char> buffer(buffer_size);

    size_t total_read = 0;
    while (total_read < file_size && file.good()) {
      file.read(buffer.data(), buffer.size());
      total_read += file.gcount();
    }

    EXPECT_EQ(total_read, file_size) << "Should read entire file";
  });
}

// Parameterized test for different model sizes
class ModelSizeTest : public ModelLoadingTest,
                      public ::testing::WithParamInterface<Model> {
 protected:
  ModelConfig GetTestConfig() const {
    return MockModelConfig::CreateTestConfig(GetParam());
  }
};

TEST_P(ModelSizeTest, ModelSizeConsistency) {
  auto config = GetTestConfig();

  // Test that parameter count scales appropriately
  size_t param_count = EstimateParameterCount(config);
  size_t memory_mb = (param_count * sizeof(float)) / (1024 * 1024);

  std::cout << "Model " << config.Specifier()
            << ": " << param_count << " parameters (" << memory_mb << " MB)" << std::endl;

  // Basic sanity checks
  EXPECT_GT(param_count, 1000000) << "Model should have at least 1M parameters";
  EXPECT_LT(memory_mb, 100000) << "Model should be less than 100GB";

  // Test that larger models have more parameters
  if (config.model == Model::GEMMA2_2B) {
    EXPECT_LT(memory_mb, 10000) << "2B model should be less than 10GB";
  } else if (config.model == Model::GEMMA2_9B) {
    EXPECT_GT(memory_mb, 5000) << "9B model should be more than 5GB";
    EXPECT_LT(memory_mb, 50000) << "9B model should be less than 50GB";
  }
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(
    AllModelSizes,
    ModelSizeTest,
    ::testing::Values(
        Model::GEMMA2_2B,
        Model::GEMMA2_9B,
        Model::GEMMA2_27B,
        Model::GRIFFIN_2B
    )
);

// Helper functions for estimation (simplified implementations)
size_t EstimateParameterCount(const ModelConfig& config) {
  // Rough estimate based on typical transformer architecture
  size_t embedding_params = config.vocab_size * config.model_dim;
  size_t attention_params = config.num_layers * config.model_dim * config.model_dim * 4;  // Q, K, V, O projections
  size_t ffn_params = config.num_layers * config.model_dim * config.intermediate_size * 2;  // Up and down projections
  size_t norm_params = config.num_layers * config.model_dim * 2;  // Layer norms

  return embedding_params + attention_params + ffn_params + norm_params;
}

size_t EstimateWeightSize(const ModelConfig& config) {
  return EstimateParameterCount(config) * sizeof(float);
}

}  // namespace
}  // namespace gcpp