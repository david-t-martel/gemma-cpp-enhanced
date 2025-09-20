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

#include <benchmark/benchmark.h>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "gemma/kv_cache.h"
#include "io/io.h"
#include "util/threading_context.h"
#include "hwy/timer.h"

namespace gcpp {
namespace {

// Global test resources
class BenchmarkFixture {
public:
    static BenchmarkFixture& Instance() {
        static BenchmarkFixture instance;
        return instance;
    }

    bool IsAvailable() const { return gemma_available_; }
    Gemma* GetGemma() const { return gemma_.get(); }
    const ModelConfig& GetConfig() const { return config_; }
    ThreadingContext* GetThreadingContext() const { return threading_context_.get(); }

private:
    BenchmarkFixture() {
        std::string model_path = "/c/codedev/llm/.models/";
        std::string tokenizer_path = model_path + "tokenizer.spm";
        std::string weights_path = model_path + "gemma2-2b-it-sfp.sbs";

        if (!std::filesystem::exists(weights_path) || !std::filesystem::exists(tokenizer_path)) {
            gemma_available_ = false;
            return;
        }

        try {
            // Initialize model configuration
            config_ = ModelConfig::FromModel(Model::GEMMA_2B);

            // Initialize threading context
            threading_context_ = std::make_unique<ThreadingContext>(1);

            // Load model weights
            model_store_ = std::make_unique<ModelStore>();
            auto loader = model_store_->LoaderForTest(Path(weights_path), config_.weights.NumCores());

            // Initialize Gemma instance
            gemma_ = std::make_unique<Gemma>(loader, config_, *threading_context_, Path(tokenizer_path));

            gemma_available_ = true;
        } catch (const std::exception& e) {
            gemma_available_ = false;
        }
    }

    bool gemma_available_ = false;
    ModelConfig config_;
    std::unique_ptr<ThreadingContext> threading_context_;
    std::unique_ptr<ModelStore> model_store_;
    std::unique_ptr<Gemma> gemma_;
};

} // namespace

// Tokenization benchmarks
static void BM_TokenizeShortText(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    std::string text = "Hello world, this is a test.";
    auto* gemma = fixture.GetGemma();

    for (auto _ : state) {
        std::vector<int> tokens;
        auto status = gemma->Tokenizer().Encode(text, &tokens);
        benchmark::DoNotOptimize(tokens);
        if (!status.ok()) {
            state.SkipWithError("Tokenization failed");
            return;
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TokenizeShortText);

static void BM_TokenizeLongText(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    // Create a longer text
    std::string text;
    for (int i = 0; i < 100; ++i) {
        text += "This is sentence number " + std::to_string(i) + ". ";
    }

    auto* gemma = fixture.GetGemma();

    for (auto _ : state) {
        std::vector<int> tokens;
        auto status = gemma->Tokenizer().Encode(text, &tokens);
        benchmark::DoNotOptimize(tokens);
        if (!status.ok()) {
            state.SkipWithError("Tokenization failed");
            return;
        }
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_TokenizeLongText);

static void BM_DecodeTokens(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    auto* gemma = fixture.GetGemma();
    std::string text = "The quick brown fox jumps over the lazy dog.";
    std::vector<int> tokens;
    auto encode_status = gemma->Tokenizer().Encode(text, &tokens);
    if (!encode_status.ok()) {
        state.SkipWithError("Failed to encode test text");
        return;
    }

    for (auto _ : state) {
        std::string decoded;
        auto status = gemma->Tokenizer().Decode(tokens, &decoded);
        benchmark::DoNotOptimize(decoded);
        if (!status.ok()) {
            state.SkipWithError("Decoding failed");
            return;
        }
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DecodeTokens);

// KV Cache benchmarks
static void BM_KVCacheInitialization(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    const size_t seq_len = state.range(0);
    const auto& config = fixture.GetConfig();
    auto* threading_context = fixture.GetThreadingContext();

    for (auto _ : state) {
        auto kv_cache = std::make_unique<KVCache>(config, seq_len, threading_context->Allocator());
        benchmark::DoNotOptimize(kv_cache);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_KVCacheInitialization)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

// Generation benchmarks
static void BM_ShortGeneration(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    auto* gemma = fixture.GetGemma();
    const auto& config = fixture.GetConfig();
    auto* threading_context = fixture.GetThreadingContext();

    std::string prompt = "Hello";
    std::vector<int> prompt_tokens;
    auto encode_status = gemma->Tokenizer().Encode(prompt, &prompt_tokens);
    if (!encode_status.ok()) {
        state.SkipWithError("Failed to encode prompt");
        return;
    }

    for (auto _ : state) {
        // Create fresh KV cache for each iteration
        auto kv_cache = std::make_unique<KVCache>(config, 1024, threading_context->Allocator());

        RuntimeConfig runtime_config = {
            .max_tokens = 5,  // Short generation
            .max_seq_len = 1024,
            .temperature = 0.7f,
            .top_k = 40,
        };

        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < 5;
        };

        try {
            gcpp::GenerateGemma(*gemma, runtime_config, prompt_tokens, pos, *kv_cache,
                               stream_token, *threading_context, timing_info);
        } catch (const std::exception& e) {
            state.SkipWithError("Generation failed");
            return;
        }

        benchmark::DoNotOptimize(generated_tokens);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ShortGeneration);

static void BM_MediumGeneration(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    auto* gemma = fixture.GetGemma();
    const auto& config = fixture.GetConfig();
    auto* threading_context = fixture.GetThreadingContext();

    std::string prompt = "Write a short story:";
    std::vector<int> prompt_tokens;
    auto encode_status = gemma->Tokenizer().Encode(prompt, &prompt_tokens);
    if (!encode_status.ok()) {
        state.SkipWithError("Failed to encode prompt");
        return;
    }

    for (auto _ : state) {
        auto kv_cache = std::make_unique<KVCache>(config, 2048, threading_context->Allocator());

        RuntimeConfig runtime_config = {
            .max_tokens = 50,  // Medium generation
            .max_seq_len = 2048,
            .temperature = 0.7f,
            .top_k = 40,
        };

        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < 50;
        };

        try {
            gcpp::GenerateGemma(*gemma, runtime_config, prompt_tokens, pos, *kv_cache,
                               stream_token, *threading_context, timing_info);
        } catch (const std::exception& e) {
            state.SkipWithError("Generation failed");
            return;
        }

        benchmark::DoNotOptimize(generated_tokens);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MediumGeneration);

// Memory allocation benchmarks
static void BM_ConfigCreation(benchmark::State& state) {
    for (auto _ : state) {
        ModelConfig config = ModelConfig::FromModel(Model::GEMMA_2B);
        benchmark::DoNotOptimize(config);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ConfigCreation);

static void BM_ThreadingContextCreation(benchmark::State& state) {
    const int num_threads = state.range(0);

    for (auto _ : state) {
        auto context = std::make_unique<ThreadingContext>(num_threads);
        benchmark::DoNotOptimize(context);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ThreadingContextCreation)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

// Throughput benchmarks
static void BM_TokensPerSecond(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    auto* gemma = fixture.GetGemma();
    const auto& config = fixture.GetConfig();
    auto* threading_context = fixture.GetThreadingContext();

    std::string prompt = "Calculate:";
    std::vector<int> prompt_tokens;
    auto encode_status = gemma->Tokenizer().Encode(prompt, &prompt_tokens);
    if (!encode_status.ok()) {
        state.SkipWithError("Failed to encode prompt");
        return;
    }

    const int target_tokens = state.range(0);
    size_t total_tokens_generated = 0;

    for (auto _ : state) {
        auto kv_cache = std::make_unique<KVCache>(config, 2048, threading_context->Allocator());

        RuntimeConfig runtime_config = {
            .max_tokens = target_tokens,
            .max_seq_len = 2048,
            .temperature = 0.7f,
            .top_k = 40,
        };

        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < static_cast<size_t>(target_tokens);
        };

        try {
            gcpp::GenerateGemma(*gemma, runtime_config, prompt_tokens, pos, *kv_cache,
                               stream_token, *threading_context, timing_info);
        } catch (const std::exception& e) {
            state.SkipWithError("Generation failed");
            return;
        }

        total_tokens_generated += generated_tokens.size();
        benchmark::DoNotOptimize(generated_tokens);
    }

    state.SetItemsProcessed(total_tokens_generated);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        total_tokens_generated, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_TokensPerSecond)->Arg(10)->Arg(25)->Arg(50)->Arg(100);

// Batch processing benchmarks
static void BM_BatchTokenization(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    auto* gemma = fixture.GetGemma();
    const int batch_size = state.range(0);

    std::vector<std::string> batch_texts;
    for (int i = 0; i < batch_size; ++i) {
        batch_texts.push_back("This is test sentence number " + std::to_string(i) + ".");
    }

    for (auto _ : state) {
        std::vector<std::vector<int>> batch_tokens(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            auto status = gemma->Tokenizer().Encode(batch_texts[i], &batch_tokens[i]);
            if (!status.ok()) {
                state.SkipWithError("Batch tokenization failed");
                return;
            }
        }

        benchmark::DoNotOptimize(batch_tokens);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_BatchTokenization)->Arg(1)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

// Memory usage benchmarks
static void BM_MemoryFootprint(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    const size_t seq_len = state.range(0);
    const auto& config = fixture.GetConfig();
    auto* threading_context = fixture.GetThreadingContext();

    for (auto _ : state) {
        // Measure memory allocation for different sequence lengths
        auto kv_cache = std::make_unique<KVCache>(config, seq_len, threading_context->Allocator());

        // Force memory allocation
        benchmark::DoNotOptimize(kv_cache);

        // Simulate some usage
        // (In real benchmark, we'd measure actual memory usage)
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["seq_len"] = benchmark::Counter(seq_len);
}
BENCHMARK(BM_MemoryFootprint)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// Temperature sensitivity benchmarks
static void BM_TemperatureImpact(benchmark::State& state) {
    auto& fixture = BenchmarkFixture::Instance();
    if (!fixture.IsAvailable()) {
        state.SkipWithError("Gemma model not available");
        return;
    }

    auto* gemma = fixture.GetGemma();
    const auto& config = fixture.GetConfig();
    auto* threading_context = fixture.GetThreadingContext();

    const float temperature = static_cast<float>(state.range(0)) / 10.0f;  // 0.1, 0.5, 1.0, etc.

    std::string prompt = "The answer is";
    std::vector<int> prompt_tokens;
    auto encode_status = gemma->Tokenizer().Encode(prompt, &prompt_tokens);
    if (!encode_status.ok()) {
        state.SkipWithError("Failed to encode prompt");
        return;
    }

    for (auto _ : state) {
        auto kv_cache = std::make_unique<KVCache>(config, 1024, threading_context->Allocator());

        RuntimeConfig runtime_config = {
            .max_tokens = 15,
            .max_seq_len = 1024,
            .temperature = temperature,
            .top_k = 40,
        };

        std::vector<int> generated_tokens;
        size_t pos = 0;

        TimingInfo timing_info;
        auto stream_token = [&generated_tokens](int token, float prob) -> bool {
            generated_tokens.push_back(token);
            return generated_tokens.size() < 15;
        };

        try {
            gcpp::GenerateGemma(*gemma, runtime_config, prompt_tokens, pos, *kv_cache,
                               stream_token, *threading_context, timing_info);
        } catch (const std::exception& e) {
            state.SkipWithError("Generation failed");
            return;
        }

        benchmark::DoNotOptimize(generated_tokens);
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["temperature"] = benchmark::Counter(temperature);
}
BENCHMARK(BM_TemperatureImpact)->Arg(1)->Arg(5)->Arg(10)->Arg(15)->Arg(20); // 0.1 to 2.0

} // namespace gcpp

BENCHMARK_MAIN();