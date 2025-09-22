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
#include <memory>
#include <string>

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "io/io.h"
#include "util/test_util.h"

namespace gcpp {
namespace {

class ModelLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test environment
        test_model_path_ = "/c/codedev/llm/.models/";
        test_tokenizer_path_ = test_model_path_ + "tokenizer.spm";
        test_weights_path_ = test_model_path_ + "gemma2-2b-it-sfp.sbs";
        test_single_weights_path_ = test_model_path_ + "gemma2-2b-it-sfp-single.sbs";
    }

    void TearDown() override {
        // Clean up after tests
    }

    std::string test_model_path_;
    std::string test_tokenizer_path_;
    std::string test_weights_path_;
    std::string test_single_weights_path_;
};

TEST_F(ModelLoadingTest, CanLoadValidConfig) {
    // Test loading various model configurations
    for (Model model : {Model::GEMMA_2B, Model::GEMMA_3_1B, Model::GEMMA_3_4B}) {
        SCOPED_TRACE("Testing model: " + std::to_string(static_cast<int>(model)));

        ModelConfig config = ModelConfig::FromModel(model);
        EXPECT_GT(config.model_dim, 0);
        EXPECT_GT(config.num_layers, 0);
        EXPECT_GT(config.vocab_size, 0);
        EXPECT_GT(config.num_heads, 0);
        EXPECT_GT(config.head_dim, 0);

        // Verify model dimensions are consistent
        EXPECT_EQ(config.model_dim, config.num_heads * config.head_dim);
    }
}

TEST_F(ModelLoadingTest, ConfigValidation) {
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    // Test specific values for Gemma 2B
    EXPECT_EQ(config.model_dim, 2048);
    EXPECT_EQ(config.num_layers, 18);
    EXPECT_EQ(config.vocab_size, 256000);
    EXPECT_EQ(config.num_heads, 8);
    EXPECT_EQ(config.head_dim, 256);
    EXPECT_EQ(config.num_kv_heads, 1);
}

TEST_F(ModelLoadingTest, InvalidModelHandling) {
    // Test handling of invalid model enum values
    EXPECT_THROW({
        ModelConfig config = ModelConfig::FromModel(static_cast<Model>(999));
    }, std::runtime_error);
}

TEST_F(ModelLoadingTest, WeightsFileValidation) {
    // Test file existence checks
    if (std::filesystem::exists(test_weights_path_)) {
        Path weights_path(test_weights_path_);
        EXPECT_TRUE(weights_path.Exists());
    } else {
        GTEST_SKIP() << "Test weights file not found: " << test_weights_path_;
    }

    if (std::filesystem::exists(test_tokenizer_path_)) {
        Path tokenizer_path(test_tokenizer_path_);
        EXPECT_TRUE(tokenizer_path.Exists());
    } else {
        GTEST_SKIP() << "Test tokenizer file not found: " << test_tokenizer_path_;
    }
}

TEST_F(ModelLoadingTest, ModelStoreInitialization) {
    if (!std::filesystem::exists(test_weights_path_) ||
        !std::filesystem::exists(test_tokenizer_path_)) {
        GTEST_SKIP() << "Required test files not found";
    }

    ModelStore store;
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    // Test model store can be initialized with valid paths
    EXPECT_NO_THROW({
        store.LoaderForTest(Path(test_weights_path_), config.weights.NumCores());
    });
}

TEST_F(ModelLoadingTest, TokenizerInitialization) {
    if (!std::filesystem::exists(test_tokenizer_path_)) {
        GTEST_SKIP() << "Tokenizer file not found: " << test_tokenizer_path_;
    }

    // Test tokenizer can be loaded
    sentencepiece::SentencePieceProcessor tokenizer;
    auto status = tokenizer.Load(test_tokenizer_path_);
    EXPECT_TRUE(status.ok()) << "Failed to load tokenizer: " << status.ToString();

    // Test basic tokenizer functionality
    if (status.ok()) {
        std::vector<int> tokens;
        auto encode_status = tokenizer.Encode("Hello world", &tokens);
        EXPECT_TRUE(encode_status.ok());
        EXPECT_GT(tokens.size(), 0);

        std::string decoded;
        auto decode_status = tokenizer.Decode(tokens, &decoded);
        EXPECT_TRUE(decode_status.ok());
        EXPECT_FALSE(decoded.empty());
    }
}

TEST_F(ModelLoadingTest, MemoryAllocationValidation) {
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    // Test memory requirements calculation
    size_t memory_required = config.weights.NumBytes();
    EXPECT_GT(memory_required, 0);

    // Verify reasonable memory bounds (should be several GB for 2B model)
    EXPECT_GT(memory_required, 1000000000UL);  // > 1GB
    EXPECT_LT(memory_required, 50000000000UL); // < 50GB
}

TEST_F(ModelLoadingTest, MultipleModelLoading) {
    // Test that multiple model configs can be created without interference
    ModelConfig config1 = ModelConfig::FromModel(Model::GEMMA_2B);
    ModelConfig config2 = ModelConfig::FromModel(Model::GEMMA_3_1B);

    EXPECT_NE(config1.model_dim, config2.model_dim);
    EXPECT_NE(config1.num_layers, config2.num_layers);

    // Verify configs remain independent
    EXPECT_EQ(config1.model_dim, 2048);
    EXPECT_EQ(config2.model_dim, 2048);  // Both are 2048 but different models
}

TEST_F(ModelLoadingTest, ConfigSpecifierGeneration) {
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
    std::string specifier = config.Specifier();

    EXPECT_FALSE(specifier.empty());
    EXPECT_NE(specifier.find("gemma"), std::string::npos);
}

TEST_F(ModelLoadingTest, WeightsTypeValidation) {
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    // Test that weights configuration is valid
    EXPECT_GT(config.weights.NumCores(), 0);
    EXPECT_GT(config.weights.NumBytes(), 0);

    // Test weight type enumeration
    auto weight_type = config.weights.Type();
    EXPECT_TRUE(weight_type == WeightT::kSFP ||
                weight_type == WeightT::kBF16 ||
                weight_type == WeightT::kF32);
}

} // namespace
} // namespace gcpp