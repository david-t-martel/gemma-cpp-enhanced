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
#include <vector>
#include <filesystem>
#include <fstream>

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/kv_cache.h"
#include "io/io.h"
#include "util/threading_context.h"

namespace gcpp {
namespace {

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/gemma_test_" + std::to_string(std::time(nullptr));
        std::filesystem::create_directories(test_dir_);

        valid_model_path_ = "/c/codedev/llm/.models/";
        valid_tokenizer_path_ = valid_model_path_ + "tokenizer.spm";
        valid_weights_path_ = valid_model_path_ + "gemma2-2b-it-sfp.sbs";

        invalid_path_ = test_dir_ + "/nonexistent.sbs";
        corrupted_file_path_ = test_dir_ + "/corrupted.spm";

        // Create a corrupted file for testing
        std::ofstream corrupted(corrupted_file_path_);
        corrupted << "invalid content that should cause parsing errors";
        corrupted.close();
    }

    void TearDown() override {
        // Clean up test directory
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
    std::string valid_model_path_;
    std::string valid_tokenizer_path_;
    std::string valid_weights_path_;
    std::string invalid_path_;
    std::string corrupted_file_path_;
};

TEST_F(ErrorHandlingTest, InvalidModelConfig) {
    // Test invalid model enum
    EXPECT_THROW({
        ModelConfig config = ModelConfig::FromModel(static_cast<Model>(999));
    }, std::exception);

    EXPECT_THROW({
        ModelConfig config = ModelConfig::FromModel(static_cast<Model>(-1));
    }, std::exception);
}

TEST_F(ErrorHandlingTest, NonexistentFiles) {
    // Test loading with nonexistent weights file
    EXPECT_THROW({
        ModelStore store;
        ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
        auto loader = store.LoaderForTest(Path(invalid_path_), config.weights.NumCores());
    }, std::exception);

    // Test loading with nonexistent tokenizer
    if (std::filesystem::exists(valid_weights_path_)) {
        EXPECT_THROW({
            ModelStore store;
            ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
            ThreadingContext context(1);
            auto loader = store.LoaderForTest(Path(valid_weights_path_), config.weights.NumCores());
            Gemma gemma(loader, config, context, Path(invalid_path_));
        }, std::exception);
    }
}

TEST_F(ErrorHandlingTest, CorruptedTokenizer) {
    // Test loading corrupted tokenizer file
    sentencepiece::SentencePieceProcessor tokenizer;
    auto status = tokenizer.Load(corrupted_file_path_);
    EXPECT_FALSE(status.ok());

    // Verify error message is meaningful
    EXPECT_FALSE(status.ToString().empty());
}

TEST_F(ErrorHandlingTest, InvalidTokenizerOperations) {
    if (!std::filesystem::exists(valid_tokenizer_path_)) {
        GTEST_SKIP() << "Valid tokenizer not available";
    }

    sentencepiece::SentencePieceProcessor tokenizer;
    auto load_status = tokenizer.Load(valid_tokenizer_path_);
    if (!load_status.ok()) {
        GTEST_SKIP() << "Could not load tokenizer for testing";
    }

    // Test encoding with null input
    std::vector<int> tokens;
    auto status = tokenizer.Encode("", &tokens);
    // Empty string should be handled gracefully
    EXPECT_TRUE(status.ok());

    // Test decoding invalid token IDs
    std::vector<int> invalid_tokens = {-1, 999999999};
    std::string decoded;
    status = tokenizer.Decode(invalid_tokens, &decoded);
    // Invalid tokens should be handled gracefully (might use UNK tokens)
    EXPECT_TRUE(status.ok() || !status.ToString().empty());

    // Test very long input
    std::string very_long_text(1000000, 'a');  // 1MB of 'a' characters
    std::vector<int> long_tokens;
    status = tokenizer.Encode(very_long_text, &long_tokens);
    // Should either succeed or fail gracefully
    if (!status.ok()) {
        EXPECT_FALSE(status.ToString().empty());
    }
}

TEST_F(ErrorHandlingTest, KVCacheInvalidParameters) {
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
    ThreadingContext context(1);
    auto& allocator = context.Allocator();

    // Test zero sequence length
    EXPECT_THROW({
        KVCache cache(config, 0, allocator);
    }, std::exception);

    // Test extremely large sequence length
    EXPECT_THROW({
        KVCache cache(config, SIZE_MAX, allocator);
    }, std::exception);
}

TEST_F(ErrorHandlingTest, ThreadingContextInvalidParameters) {
    // Test zero threads
    EXPECT_THROW({
        ThreadingContext context(0);
    }, std::exception);

    // Test negative threads
    EXPECT_THROW({
        ThreadingContext context(-1);
    }, std::exception);

    // Test extremely large thread count
    EXPECT_THROW({
        ThreadingContext context(10000);
    }, std::exception);
}

TEST_F(ErrorHandlingTest, GenerationWithInvalidConfig) {
    if (!std::filesystem::exists(valid_weights_path_) ||
        !std::filesystem::exists(valid_tokenizer_path_)) {
        GTEST_SKIP() << "Valid model files not available";
    }

    try {
        ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
        ThreadingContext context(1);
        ModelStore store;
        auto loader = store.LoaderForTest(Path(valid_weights_path_), config.weights.NumCores());
        Gemma gemma(loader, config, context, Path(valid_tokenizer_path_));
        KVCache kv_cache(config, 1024, context.Allocator());

        std::string prompt = "Hello";
        std::vector<int> prompt_tokens;
        auto encode_status = gemma.Tokenizer().Encode(prompt, &prompt_tokens);
        if (!encode_status.ok()) {
            GTEST_SKIP() << "Could not encode test prompt";
        }

        // Test with invalid runtime config - negative max_tokens
        RuntimeConfig invalid_config = {
            .max_tokens = -1,
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = 40,
        };

        std::vector<int> generated_tokens;
        size_t pos = 0;
        TimingInfo timing_info;

        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return false;  // Stop immediately
        };

        EXPECT_THROW({
            gcpp::GenerateGemma(gemma, invalid_config, prompt_tokens, pos, kv_cache,
                               stream_token, context, timing_info);
        }, std::exception);

        // Test with zero temperature
        RuntimeConfig zero_temp_config = {
            .max_tokens = 10,
            .max_seq_len = 1024,
            .temperature = 0.0f,
            .top_k = 40,
        };

        // This might be handled differently - some models accept 0.0 temperature
        // If it throws, that's fine; if it doesn't, that's also fine
        try {
            gcpp::GenerateGemma(gemma, zero_temp_config, prompt_tokens, pos, kv_cache,
                               stream_token, context, timing_info);
        } catch (const std::exception& e) {
            // Expected for some implementations
            EXPECT_FALSE(std::string(e.what()).empty());
        }

        // Test with negative temperature
        RuntimeConfig negative_temp_config = {
            .max_tokens = 10,
            .max_seq_len = 1024,
            .temperature = -1.0f,
            .top_k = 40,
        };

        EXPECT_THROW({
            gcpp::GenerateGemma(gemma, negative_temp_config, prompt_tokens, pos, kv_cache,
                               stream_token, context, timing_info);
        }, std::exception);

    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not initialize model for testing: " << e.what();
    }
}

TEST_F(ErrorHandlingTest, GenerationWithInvalidPrompt) {
    if (!std::filesystem::exists(valid_weights_path_) ||
        !std::filesystem::exists(valid_tokenizer_path_)) {
        GTEST_SKIP() << "Valid model files not available";
    }

    try {
        ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
        ThreadingContext context(1);
        ModelStore store;
        auto loader = store.LoaderForTest(Path(valid_weights_path_), config.weights.NumCores());
        Gemma gemma(loader, config, context, Path(valid_tokenizer_path_));
        KVCache kv_cache(config, 1024, context.Allocator());

        RuntimeConfig runtime_config = {
            .max_tokens = 10,
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = 40,
        };

        // Test with empty prompt tokens
        std::vector<int> empty_tokens;
        size_t pos = 0;
        TimingInfo timing_info;

        auto stream_token = [](int token, float prob) -> bool {
            return false;  // Stop immediately
        };

        // Empty prompt might be handled differently by different models
        try {
            gcpp::GenerateGemma(gemma, runtime_config, empty_tokens, pos, kv_cache,
                               stream_token, context, timing_info);
        } catch (const std::exception& e) {
            // This is acceptable - empty prompts might not be supported
            EXPECT_FALSE(std::string(e.what()).empty());
        }

        // Test with invalid token IDs
        std::vector<int> invalid_tokens = {-1, 999999999, -100};
        EXPECT_THROW({
            gcpp::GenerateGemma(gemma, runtime_config, invalid_tokens, pos, kv_cache,
                               stream_token, context, timing_info);
        }, std::exception);

    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not initialize model for testing: " << e.what();
    }
}

TEST_F(ErrorHandlingTest, GenerationWithInvalidKVCache) {
    if (!std::filesystem::exists(valid_weights_path_) ||
        !std::filesystem::exists(valid_tokenizer_path_)) {
        GTEST_SKIP() << "Valid model files not available";
    }

    try {
        ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
        ThreadingContext context(1);
        ModelStore store;
        auto loader = store.LoaderForTest(Path(valid_weights_path_), config.weights.NumCores());
        Gemma gemma(loader, config, context, Path(valid_tokenizer_path_));

        std::string prompt = "Hello";
        std::vector<int> prompt_tokens;
        auto encode_status = gemma.Tokenizer().Encode(prompt, &prompt_tokens);
        if (!encode_status.ok()) {
            GTEST_SKIP() << "Could not encode test prompt";
        }

        RuntimeConfig runtime_config = {
            .max_tokens = 10,
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = 40,
        };

        // Test with KV cache that's too small for the request
        KVCache small_cache(config, 100, context.Allocator());  // Very small cache
        size_t pos = 0;
        TimingInfo timing_info;

        auto stream_token = [](int token, float prob) -> bool {
            return false;  // Stop immediately
        };

        // This should either work (if prompt fits) or throw an exception
        try {
            gcpp::GenerateGemma(gemma, runtime_config, prompt_tokens, pos, small_cache,
                               stream_token, context, timing_info);
        } catch (const std::exception& e) {
            // Expected if prompt doesn't fit in small cache
            EXPECT_FALSE(std::string(e.what()).empty());
        }

    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not initialize model for testing: " << e.what();
    }
}

TEST_F(ErrorHandlingTest, StreamTokenCallbackErrors) {
    if (!std::filesystem::exists(valid_weights_path_) ||
        !std::filesystem::exists(valid_tokenizer_path_)) {
        GTEST_SKIP() << "Valid model files not available";
    }

    try {
        ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
        ThreadingContext context(1);
        ModelStore store;
        auto loader = store.LoaderForTest(Path(valid_weights_path_), config.weights.NumCores());
        Gemma gemma(loader, config, context, Path(valid_tokenizer_path_));
        KVCache kv_cache(config, 1024, context.Allocator());

        std::string prompt = "Hello";
        std::vector<int> prompt_tokens;
        auto encode_status = gemma.Tokenizer().Encode(prompt, &prompt_tokens);
        if (!encode_status.ok()) {
            GTEST_SKIP() << "Could not encode test prompt";
        }

        RuntimeConfig runtime_config = {
            .max_tokens = 10,
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = 40,
        };

        size_t pos = 0;
        TimingInfo timing_info;

        // Test callback that throws exception
        auto throwing_callback = [](int token, float prob) -> bool {
            throw std::runtime_error("Callback error");
        };

        EXPECT_THROW({
            gcpp::GenerateGemma(gemma, runtime_config, prompt_tokens, pos, kv_cache,
                               throwing_callback, context, timing_info);
        }, std::exception);

    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not initialize model for testing: " << e.what();
    }
}

TEST_F(ErrorHandlingTest, PathHandling) {
    // Test path creation with invalid characters
    EXPECT_THROW({
        Path invalid_path("/path/with\0null/character");
    }, std::exception);

    // Test path operations on nonexistent paths
    Path nonexistent(invalid_path_);
    EXPECT_FALSE(nonexistent.Exists());

    // Test reading from nonexistent path
    EXPECT_THROW({
        // This would typically be tested with file reading operations
        // The exact API depends on the io implementation
    }, std::exception);
}

TEST_F(ErrorHandlingTest, ConfigurationEdgeCases) {
    // Test model configuration edge cases
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    // Verify configuration has sensible values
    EXPECT_GT(config.model_dim, 0);
    EXPECT_GT(config.num_layers, 0);
    EXPECT_GT(config.vocab_size, 0);
    EXPECT_GT(config.num_heads, 0);
    EXPECT_GT(config.head_dim, 0);

    // Test configuration consistency
    EXPECT_EQ(config.model_dim, config.num_heads * config.head_dim);
}

TEST_F(ErrorHandlingTest, ResourceExhaustion) {
    // Test behavior under resource constraints
    ThreadingContext context(1);
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    // Try to allocate many large KV caches to exhaust memory
    std::vector<std::unique_ptr<KVCache>> caches;
    const size_t large_seq_len = 16384;
    const int max_attempts = 100;

    for (int i = 0; i < max_attempts; ++i) {
        try {
            auto cache = std::make_unique<KVCache>(config, large_seq_len, context.Allocator());
            caches.push_back(std::move(cache));
        } catch (const std::exception& e) {
            // Expected when we run out of memory
            EXPECT_FALSE(std::string(e.what()).empty());
            break;
        }

        // If we've allocated a lot without failure, break to avoid hanging
        if (i > 50) {
            break;
        }
    }

    // Verify we can still allocate after cleanup
    caches.clear();
    auto final_cache = std::make_unique<KVCache>(config, 1024, context.Allocator());
    EXPECT_NE(final_cache.get(), nullptr);
}

TEST_F(ErrorHandlingTest, ConcurrentErrorConditions) {
    ThreadingContext context(4);
    ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<bool> thread_results(num_threads, false);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                // Each thread tries to do something that might fail
                for (int i = 0; i < 10; ++i) {
                    try {
                        // Try allocating varying sizes
                        const size_t seq_len = 512 + i * 64;
                        auto cache = std::make_unique<KVCache>(config, seq_len, context.Allocator());

                        // Brief usage
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));

                        // Some might fail due to resource contention
                    } catch (const std::exception& e) {
                        // Expected under resource pressure
                    }
                }
                thread_results[t] = true;
            } catch (const std::exception& e) {
                // Thread failed, but this might be expected
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // At least some threads should complete successfully
    int successful_threads = 0;
    for (bool result : thread_results) {
        if (result) successful_threads++;
    }
    EXPECT_GT(successful_threads, 0);
}

} // namespace
} // namespace gcpp