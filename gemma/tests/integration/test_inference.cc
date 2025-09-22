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
#include <chrono>

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/kv_cache.h"
#include "io/io.h"
#include "util/threading_context.h"
#include "hwy/timer.h"

namespace gcpp {
namespace {

class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_model_path_ = "/c/codedev/llm/.models/";
        test_tokenizer_path_ = test_model_path_ + "tokenizer.spm";
        test_weights_path_ = test_model_path_ + "gemma2-2b-it-sfp.sbs";

        // Check if required files exist
        weights_available_ = std::filesystem::exists(test_weights_path_);
        tokenizer_available_ = std::filesystem::exists(test_tokenizer_path_);

        if (weights_available_ && tokenizer_available_) {
            try {
                InitializeGemma();
                gemma_available_ = true;
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize Gemma: " << e.what() << std::endl;
                gemma_available_ = false;
            }
        }
    }

    void TearDown() override {
        // Cleanup
        gemma_.reset();
        kv_cache_.reset();
    }

    void InitializeGemma() {
        // Initialize model configuration
        config_ = ModelConfig::FromModel(Model::GEMMA_2B);

        // Initialize threading context
        threading_context_ = std::make_unique<ThreadingContext>(1);

        // Load model weights
        model_store_ = std::make_unique<ModelStore>();
        auto loader = model_store_->LoaderForTest(Path(test_weights_path_), config_.weights.NumCores());

        // Initialize Gemma instance
        gemma_ = std::make_unique<Gemma>(loader, config_, *threading_context_, Path(test_tokenizer_path_));

        // Initialize KV cache
        const size_t kSeqLen = 2048;
        kv_cache_ = std::make_unique<KVCache>(config_, kSeqLen, threading_context_->Allocator());
    }

    std::string test_model_path_;
    std::string test_tokenizer_path_;
    std::string test_weights_path_;
    bool weights_available_ = false;
    bool tokenizer_available_ = false;
    bool gemma_available_ = false;

    ModelConfig config_;
    std::unique_ptr<ThreadingContext> threading_context_;
    std::unique_ptr<ModelStore> model_store_;
    std::unique_ptr<Gemma> gemma_;
    std::unique_ptr<KVCache> kv_cache_;
};

TEST_F(InferenceTest, BasicGeneration) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    std::vector<std::string> test_prompts = {
        "Hello",
        "What is",
        "The capital of France is",
        "2 + 2 =",
        "def fibonacci(n):"
    };

    for (const auto& prompt : test_prompts) {
        SCOPED_TRACE("Testing prompt: " + prompt);

        // Tokenize prompt
        std::vector<int> prompt_tokens;
        EXPECT_TRUE(gemma_->Tokenizer().Encode(prompt, &prompt_tokens).ok());
        EXPECT_FALSE(prompt_tokens.empty());

        // Set up generation parameters
        RuntimeConfig runtime_config = {
            .max_tokens = 20,
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = 40,
        };

        // Reset KV cache
        kv_cache_.reset();
        kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

        // Generate response
        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < 20; // Stop after 20 tokens
        };

        EXPECT_NO_THROW({
            gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                               stream_token, *threading_context_, timing_info);
        });

        // Verify generation produced tokens
        EXPECT_GT(generated_tokens.size(), 0);
        EXPECT_LE(generated_tokens.size(), 20);

        // Decode generated text
        std::string generated_text;
        EXPECT_TRUE(gemma_->Tokenizer().Decode(generated_tokens, &generated_text).ok());
        EXPECT_FALSE(generated_text.empty());

        std::cout << "Prompt: \"" << prompt << "\" -> Generated: \"" << generated_text << "\"" << std::endl;
    }
}

TEST_F(InferenceTest, TemperatureVariation) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    std::string prompt = "The weather today is";
    std::vector<int> prompt_tokens;
    EXPECT_TRUE(gemma_->Tokenizer().Encode(prompt, &prompt_tokens).ok());

    std::vector<float> temperatures = {0.1f, 0.7f, 1.0f, 1.5f};

    for (float temperature : temperatures) {
        SCOPED_TRACE("Testing temperature: " + std::to_string(temperature));

        RuntimeConfig runtime_config = {
            .max_tokens = 15,
            .max_seq_len = 1024,
            .temperature = temperature,
            .top_k = 40,
        };

        // Reset KV cache for each temperature test
        kv_cache_.reset();
        kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < 15;
        };

        EXPECT_NO_THROW({
            gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                               stream_token, *threading_context_, timing_info);
        });

        EXPECT_GT(generated_tokens.size(), 0);

        std::string generated_text;
        EXPECT_TRUE(gemma_->Tokenizer().Decode(generated_tokens, &generated_text).ok());

        std::cout << "Temperature " << temperature << ": \"" << generated_text << "\"" << std::endl;
    }
}

TEST_F(InferenceTest, TopKVariation) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    std::string prompt = "Once upon a time";
    std::vector<int> prompt_tokens;
    EXPECT_TRUE(gemma_->Tokenizer().Encode(prompt, &prompt_tokens).ok());

    std::vector<int> top_k_values = {1, 10, 40, 100};

    for (int top_k : top_k_values) {
        SCOPED_TRACE("Testing top_k: " + std::to_string(top_k));

        RuntimeConfig runtime_config = {
            .max_tokens = 15,
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = top_k,
        };

        // Reset KV cache for each top_k test
        kv_cache_.reset();
        kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < 15;
        };

        EXPECT_NO_THROW({
            gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                               stream_token, *threading_context_, timing_info);
        });

        EXPECT_GT(generated_tokens.size(), 0);

        std::string generated_text;
        EXPECT_TRUE(gemma_->Tokenizer().Decode(generated_tokens, &generated_text).ok());

        std::cout << "Top-K " << top_k << ": \"" << generated_text << "\"" << std::endl;
    }
}

TEST_F(InferenceTest, LongSequenceGeneration) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    std::string prompt = "Write a story about a robot:";
    std::vector<int> prompt_tokens;
    EXPECT_TRUE(gemma_->Tokenizer().Encode(prompt, &prompt_tokens).ok());

    RuntimeConfig runtime_config = {
        .max_tokens = 100,  // Generate longer sequence
        .max_seq_len = 2048,
        .temperature = 0.8f,
        .top_k = 40,
    };

    // Use larger KV cache for longer sequences
    kv_cache_.reset();
    kv_cache_ = std::make_unique<KVCache>(config_, 2048, threading_context_->Allocator());

    std::vector<int> generated_tokens;
    size_t pos = 0;

    TimingInfo timing_info;
    auto stream_token = [&generated_tokens](int token, float prob) -> bool {
        generated_tokens.push_back(token);
        return generated_tokens.size() < 100;
    };

    EXPECT_NO_THROW({
        gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                           stream_token, *threading_context_, timing_info);
    });

    EXPECT_GT(generated_tokens.size(), 50);  // Should generate substantial content

    std::string generated_text;
    EXPECT_TRUE(gemma_->Tokenizer().Decode(generated_tokens, &generated_text).ok());

    std::cout << "Long generation result: \"" << generated_text << "\"" << std::endl;
}

TEST_F(InferenceTest, ConversationalContext) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    // Test maintaining conversation context
    std::vector<std::string> conversation = {
        "Human: Hello, how are you?",
        "Assistant: I'm doing well, thank you for asking!",
        "Human: What's your favorite color?",
        "Assistant:"
    };

    std::string full_prompt;
    for (const auto& turn : conversation) {
        full_prompt += turn + "\n";
    }

    std::vector<int> prompt_tokens;
    EXPECT_TRUE(gemma_->Tokenizer().Encode(full_prompt, &prompt_tokens).ok());

    RuntimeConfig runtime_config = {
        .max_tokens = 30,
        .max_seq_len = 1024,
        .temperature = 0.7f,
        .top_k = 40,
    };

    kv_cache_.reset();
    kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

    std::vector<int> generated_tokens;
    size_t pos = 0;

    TimingInfo timing_info;
    auto stream_token = [&generated_tokens](int token, float prob) -> bool {
        generated_tokens.push_back(token);
        return generated_tokens.size() < 30;
    };

    EXPECT_NO_THROW({
        gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                           stream_token, *threading_context_, timing_info);
    });

    EXPECT_GT(generated_tokens.size(), 0);

    std::string generated_text;
    EXPECT_TRUE(gemma_->Tokenizer().Decode(generated_tokens, &generated_text).ok());

    std::cout << "Conversational response: \"" << generated_text << "\"" << std::endl;
}

TEST_F(InferenceTest, SpecialTokenHandling) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    // Test handling of special tokens
    std::vector<std::string> special_prompts = {
        "<bos>Hello world",
        "Generate text<eos>",
        "Test with <unk> token"
    };

    for (const auto& prompt : special_prompts) {
        SCOPED_TRACE("Testing special token prompt: " + prompt);

        std::vector<int> prompt_tokens;
        auto encode_status = gemma_->Tokenizer().Encode(prompt, &prompt_tokens);

        if (encode_status.ok() && !prompt_tokens.empty()) {
            RuntimeConfig runtime_config = {
                .max_tokens = 10,
                .max_seq_len = 1024,
                .temperature = 0.7f,
                .top_k = 40,
            };

            kv_cache_.reset();
            kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

            std::vector<int> generated_tokens;
            size_t pos = 0;

            TimingInfo timing_info;
            auto stream_token = [&generated_tokens](int token, float prob) -> bool {
                generated_tokens.push_back(token);
                return generated_tokens.size() < 10;
            };

            EXPECT_NO_THROW({
                gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                                   stream_token, *threading_context_, timing_info);
            });
        }
    }
}

TEST_F(InferenceTest, StopTokenGeneration) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    std::string prompt = "List three items:";
    std::vector<int> prompt_tokens;
    EXPECT_TRUE(gemma_->Tokenizer().Encode(prompt, &prompt_tokens).ok());

    RuntimeConfig runtime_config = {
        .max_tokens = 100,  // Allow many tokens
        .max_seq_len = 1024,
        .temperature = 0.7f,
        .top_k = 40,
    };

    kv_cache_.reset();
    kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

    std::vector<int> generated_tokens;
    size_t pos = 0;
    bool stopped_early = false;

    TimingInfo timing_info;
    auto stream_token = [&](int token, float prob) -> bool {
        generated_tokens.push_back(token);

        // Stop if we encounter EOS token or reach limit
        if (token == gemma_->Tokenizer().eos_id()) {
            stopped_early = true;
            return false;
        }
        return generated_tokens.size() < 100;
    };

    EXPECT_NO_THROW({
        gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                           stream_token, *threading_context_, timing_info);
    });

    EXPECT_GT(generated_tokens.size(), 0);

    std::string generated_text;
    EXPECT_TRUE(gemma_->Tokenizer().Decode(generated_tokens, &generated_text).ok());

    if (stopped_early) {
        std::cout << "Generation stopped early at EOS token" << std::endl;
    }

    std::cout << "Stop token test result: \"" << generated_text << "\"" << std::endl;
}

TEST_F(InferenceTest, TimingValidation) {
    if (!gemma_available_) {
        GTEST_SKIP() << "Gemma model not available for testing";
    }

    std::string prompt = "The meaning of life is";
    std::vector<int> prompt_tokens;
    EXPECT_TRUE(gemma_->Tokenizer().Encode(prompt, &prompt_tokens).ok());

    RuntimeConfig runtime_config = {
        .max_tokens = 20,
        .max_seq_len = 1024,
        .temperature = 0.7f,
        .top_k = 40,
    };

    kv_cache_.reset();
    kv_cache_ = std::make_unique<KVCache>(config_, 1024, threading_context_->Allocator());

    std::vector<int> generated_tokens;
    size_t pos = 0;

    TimingInfo timing_info;
    auto stream_token = [&generated_tokens](int token, float prob) -> bool {
        generated_tokens.push_back(token);
        return generated_tokens.size() < 20;
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    EXPECT_NO_THROW({
        gcpp::GenerateGemma(*gemma_, runtime_config, prompt_tokens, pos, *kv_cache_,
                           stream_token, *threading_context_, timing_info);
    });

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Verify timing info is populated
    EXPECT_GT(timing_info.prefill_tok_s, 0.0);
    EXPECT_GT(timing_info.gen_tok_s, 0.0);

    // Generation should complete in reasonable time (< 30 seconds)
    EXPECT_LT(duration.count(), 30000);

    std::cout << "Generation timing - Prefill: " << timing_info.prefill_tok_s
              << " tok/s, Generation: " << timing_info.gen_tok_s << " tok/s" << std::endl;
    std::cout << "Total duration: " << duration.count() << " ms" << std::endl;
}

} // namespace
} // namespace gcpp