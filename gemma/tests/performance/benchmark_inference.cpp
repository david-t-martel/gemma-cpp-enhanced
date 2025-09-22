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
#include <vector>
#include <string>
#include <random>

#include "../utils/test_common.h"
#include "ops/ops-inl.h"
#include "ops/matmul.h"
#include "gemma/attention.h"
#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "util/allocator.h"
#include "util/threading_context.h"
#include "hwy/tests/hwy_gtest.h"

namespace gcpp {
namespace {

using namespace test_utils;

// Global test data and state
class BenchmarkState {
 public:
  static BenchmarkState& GetInstance() {
    static BenchmarkState instance;
    return instance;
  }

  void Initialize() {
    if (initialized_) return;

    allocator_ = std::make_unique<Allocator>();
    data_gen_ = std::make_unique<TestDataGenerator>(42);
    thread_context_ = std::make_unique<ThreadingContext>();
    profiler_ = &hwy::Profiler::Get();

    // Initialize model configs for different sizes
    configs_ = {
        MockModelConfig::CreateTestConfig(Model::GEMMA2_2B),
        MockModelConfig::CreateTestConfig(Model::GEMMA2_9B),
        MockModelConfig::CreateTestConfig(Model::GRIFFIN_2B)
    };

    initialized_ = true;
  }

  Allocator& GetAllocator() { return *allocator_; }
  TestDataGenerator& GetDataGenerator() { return *data_gen_; }
  ThreadingContext& GetThreadContext() { return *thread_context_; }
  hwy::Profiler& GetProfiler() { return *profiler_; }
  const std::vector<ModelConfig>& GetConfigs() { return configs_; }

 private:
  bool initialized_ = false;
  std::unique_ptr<Allocator> allocator_;
  std::unique_ptr<TestDataGenerator> data_gen_;
  std::unique_ptr<ThreadingContext> thread_context_;
  hwy::Profiler* profiler_;
  std::vector<ModelConfig> configs_;
};

// Benchmark softmax operation
static void BM_Softmax(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t vocab_size = state.range(0);
  const float temperature = static_cast<float>(state.range(1)) / 10.0f;

  auto& data_gen = benchmark_state.GetDataGenerator();
  auto& profiler = benchmark_state.GetProfiler();

  auto logits = data_gen.GenerateRandomFloats(vocab_size, -5.0f, 5.0f);

  for (auto _ : state) {
    // Copy logits for each iteration to ensure consistent input
    auto logits_copy = logits;
    Softmax(logits_copy.data(), vocab_size, profiler, 0, temperature);
    benchmark::DoNotOptimize(logits_copy.data());
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(vocab_size * sizeof(float)));
  state.counters["vocab_size"] = vocab_size;
  state.counters["temperature"] = temperature;
}

// Register softmax benchmarks with different vocabulary sizes and temperatures
BENCHMARK(BM_Softmax)
    ->Args({1000, 10})      // 1K vocab, temp 1.0
    ->Args({32000, 10})     // 32K vocab, temp 1.0
    ->Args({256000, 10})    // 256K vocab, temp 1.0
    ->Args({256000, 5})     // 256K vocab, temp 0.5
    ->Args({256000, 20})    // 256K vocab, temp 2.0
    ->Unit(benchmark::kMicrosecond);

// Benchmark top-K sampling
static void BM_TopKSampling(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t vocab_size = state.range(0);
  const size_t k = state.range(1);

  auto& data_gen = benchmark_state.GetDataGenerator();
  auto probabilities = data_gen.GenerateNormalizedProbabilities(vocab_size);

  std::mt19937 gen(42);
  std::function<bool(int, float)> accept_all = [](int, float) { return true; };
  float temperature = 1.0f;

  for (auto _ : state) {
    int token = SampleTopK(probabilities.data(), k, vocab_size, gen, temperature, accept_all);
    benchmark::DoNotOptimize(token);
  }

  state.SetItemsProcessed(int64_t(state.iterations()));
  state.counters["vocab_size"] = vocab_size;
  state.counters["k"] = k;
}

BENCHMARK(BM_TopKSampling)
    ->Args({32000, 1})      // 32K vocab, top-1
    ->Args({32000, 10})     // 32K vocab, top-10
    ->Args({32000, 50})     // 32K vocab, top-50
    ->Args({256000, 1})     // 256K vocab, top-1
    ->Args({256000, 10})    // 256K vocab, top-10
    ->Args({256000, 50})    // 256K vocab, top-50
    ->Unit(benchmark::kMicrosecond);

// Benchmark matrix multiplication
static void BM_MatrixMultiplication(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t M = state.range(0);
  const size_t K = state.range(1);
  const size_t N = state.range(2);

  auto& data_gen = benchmark_state.GetDataGenerator();
  auto& thread_context = benchmark_state.GetThreadContext();

  auto mat_a = data_gen.GenerateRandomFloats(M * K, -1.0f, 1.0f);
  auto mat_b = data_gen.GenerateRandomFloats(K * N, -1.0f, 1.0f);
  auto mat_c = std::vector<float>(M * N, 0.0f);

  MatPtr A{mat_a.data(), M, K};
  MatPtr B{mat_b.data(), K, N};
  MatPtr C{mat_c.data(), M, N};

  MatMulEnv env(thread_context);

  for (auto _ : state) {
    std::fill(mat_c.begin(), mat_c.end(), 0.0f);
    auto mm_key = MatMul(A, B, nullptr, env, C);
    benchmark::DoNotOptimize(mat_c.data());
    benchmark::ClobberMemory();
  }

  // Calculate FLOPS (2 * M * K * N for matrix multiplication)
  int64_t flops_per_iteration = 2LL * M * K * N;
  state.SetItemsProcessed(int64_t(state.iterations()) * flops_per_iteration);
  state.counters["GFLOPS"] = benchmark::Counter(
      flops_per_iteration * state.iterations(),
      benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1000
  );
  state.counters["M"] = M;
  state.counters["K"] = K;
  state.counters["N"] = N;
}

BENCHMARK(BM_MatrixMultiplication)
    ->Args({512, 512, 512})      // Small square matrices
    ->Args({1024, 1024, 1024})   // Medium square matrices
    ->Args({2048, 2048, 2048})   // Large square matrices
    ->Args({4096, 4096, 1})      // Very wide matrix (common in transformers)
    ->Args({1, 4096, 4096})      // Very tall matrix
    ->Args({2048, 4096, 2048})   // Typical transformer FFN dimensions
    ->Unit(benchmark::kMillisecond);

// Benchmark attention computation (simplified)
static void BM_AttentionComputation(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t seq_len = state.range(0);
  const size_t model_dim = state.range(1);
  const size_t num_heads = state.range(2);

  ASSERT_EQ(model_dim % num_heads, 0) << "Model dim must be divisible by num_heads";
  const size_t head_dim = model_dim / num_heads;

  auto& data_gen = benchmark_state.GetDataGenerator();

  // Generate random Q, K, V matrices
  auto q_data = data_gen.GenerateRandomFloats(seq_len * model_dim, -1.0f, 1.0f);
  auto k_data = data_gen.GenerateRandomFloats(seq_len * model_dim, -1.0f, 1.0f);
  auto v_data = data_gen.GenerateRandomFloats(seq_len * model_dim, -1.0f, 1.0f);
  auto output_data = std::vector<float>(seq_len * model_dim, 0.0f);

  for (auto _ : state) {
    // Simplified attention computation
    // In reality, this would involve complex tensor operations
    std::fill(output_data.begin(), output_data.end(), 0.0f);

    // Mock attention computation: scaled dot-product attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (size_t head = 0; head < num_heads; ++head) {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
          float attention_score = 0.0f;
          // Compute dot product for this head
          for (size_t d = 0; d < head_dim; ++d) {
            size_t q_idx = i * model_dim + head * head_dim + d;
            size_t k_idx = j * model_dim + head * head_dim + d;
            attention_score += q_data[q_idx] * k_data[k_idx];
          }
          attention_score *= scale;

          // Apply attention to values (simplified)
          for (size_t d = 0; d < head_dim; ++d) {
            size_t v_idx = j * model_dim + head * head_dim + d;
            size_t out_idx = i * model_dim + head * head_dim + d;
            output_data[out_idx] += attention_score * v_data[v_idx];
          }
        }
      }
    }

    benchmark::DoNotOptimize(output_data.data());
    benchmark::ClobberMemory();
  }

  // Calculate FLOPS for attention
  int64_t flops_per_iteration = 4LL * seq_len * seq_len * model_dim; // Approximate
  state.SetItemsProcessed(int64_t(state.iterations()) * flops_per_iteration);
  state.counters["GFLOPS"] = benchmark::Counter(
      flops_per_iteration * state.iterations(),
      benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1000
  );
  state.counters["seq_len"] = seq_len;
  state.counters["model_dim"] = model_dim;
  state.counters["num_heads"] = num_heads;
}

BENCHMARK(BM_AttentionComputation)
    ->Args({128, 2048, 8})     // Short sequence
    ->Args({512, 2048, 8})     // Medium sequence
    ->Args({1024, 2048, 8})    // Long sequence
    ->Args({2048, 2048, 8})    // Very long sequence
    ->Args({512, 4096, 16})    // Larger model
    ->Unit(benchmark::kMillisecond);

// Benchmark memory allocation patterns
static void BM_MemoryAllocation(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t alloc_size = state.range(0);
  const size_t num_allocs = state.range(1);

  auto& allocator = benchmark_state.GetAllocator();

  for (auto _ : state) {
    std::vector<float*> ptrs;
    ptrs.reserve(num_allocs);

    // Allocate
    for (size_t i = 0; i < num_allocs; ++i) {
      float* ptr = allocator.AllocateArray<float>(alloc_size);
      ptrs.push_back(ptr);
      benchmark::DoNotOptimize(ptr);
    }

    // Use the memory (write)
    for (float* ptr : ptrs) {
      std::fill(ptr, ptr + alloc_size, 1.0f);
      benchmark::ClobberMemory();
    }

    // Deallocate happens automatically with allocator reset
    // For benchmark purposes, we'll simulate cleanup
    ptrs.clear();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(num_allocs * alloc_size * sizeof(float)));
  state.counters["alloc_size"] = alloc_size;
  state.counters["num_allocs"] = num_allocs;
}

BENCHMARK(BM_MemoryAllocation)
    ->Args({1024, 10})         // Small allocations
    ->Args({1024 * 1024, 1})   // Large single allocation
    ->Args({1024, 100})        // Many small allocations
    ->Args({64 * 1024, 10})    // Medium allocations
    ->Unit(benchmark::kMicrosecond);

// Benchmark token processing throughput
static void BM_TokenProcessingThroughput(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t batch_size = state.range(0);
  const size_t seq_len = state.range(1);
  const size_t vocab_size = 32000;  // Typical vocabulary size

  auto& data_gen = benchmark_state.GetDataGenerator();

  // Generate batches of token sequences
  std::vector<std::vector<int>> token_batches;
  for (size_t b = 0; b < batch_size; ++b) {
    token_batches.push_back(data_gen.GenerateRandomTokens(seq_len, vocab_size));
  }

  for (auto _ : state) {
    size_t total_tokens_processed = 0;

    for (const auto& tokens : token_batches) {
      // Simulate token processing (validation, lookup, etc.)
      for (int token : tokens) {
        // Simple processing: bounds checking and hash computation
        bool valid = (token >= 0 && token < static_cast<int>(vocab_size));
        size_t hash = std::hash<int>{}(token);
        benchmark::DoNotOptimize(valid);
        benchmark::DoNotOptimize(hash);
        total_tokens_processed++;
      }
    }

    benchmark::DoNotOptimize(total_tokens_processed);
  }

  int64_t tokens_per_iteration = batch_size * seq_len;
  state.SetItemsProcessed(int64_t(state.iterations()) * tokens_per_iteration);
  state.counters["tokens_per_sec"] = benchmark::Counter(
      tokens_per_iteration * state.iterations(),
      benchmark::Counter::kIsRate
  );
  state.counters["batch_size"] = batch_size;
  state.counters["seq_len"] = seq_len;
}

BENCHMARK(BM_TokenProcessingThroughput)
    ->Args({1, 512})      // Single sequence
    ->Args({4, 512})      // Small batch
    ->Args({16, 512})     // Medium batch
    ->Args({32, 512})     // Large batch
    ->Args({8, 1024})     // Longer sequences
    ->Args({8, 2048})     // Very long sequences
    ->Unit(benchmark::kMicrosecond);

// Benchmark different temperature effects on sampling
static void BM_TemperatureEffects(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  const size_t vocab_size = 32000;
  const float temperature = static_cast<float>(state.range(0)) / 100.0f;  // Scale to get reasonable temps

  auto& data_gen = benchmark_state.GetDataGenerator();
  auto& profiler = benchmark_state.GetProfiler();

  auto original_logits = data_gen.GenerateRandomFloats(vocab_size, -3.0f, 3.0f);

  for (auto _ : state) {
    auto logits = original_logits;  // Copy for each iteration
    Softmax(logits.data(), vocab_size, profiler, 0, temperature);

    // Compute entropy as a measure of distribution sharpness
    float entropy = 0.0f;
    for (float prob : logits) {
      if (prob > 0.0f) {
        entropy -= prob * std::log2(prob);
      }
    }

    benchmark::DoNotOptimize(entropy);
  }

  state.counters["temperature"] = temperature;
  state.counters["vocab_size"] = vocab_size;
}

BENCHMARK(BM_TemperatureEffects)
    ->Arg(1)      // temp = 0.01 (very sharp)
    ->Arg(10)     // temp = 0.10 (sharp)
    ->Arg(50)     // temp = 0.50 (moderate)
    ->Arg(100)    // temp = 1.00 (normal)
    ->Arg(200)    // temp = 2.00 (smooth)
    ->Arg(500)    // temp = 5.00 (very smooth)
    ->Unit(benchmark::kMicrosecond);

// Benchmark cache hit/miss patterns (simulated)
static void BM_CachePatterns(benchmark::State& state) {
  const size_t cache_size = state.range(0);
  const size_t access_pattern = state.range(1);  // 0=sequential, 1=random, 2=strided

  // Create test data
  std::vector<float> cache_data(cache_size, 1.0f);
  std::vector<size_t> access_indices;

  std::mt19937 gen(42);

  // Generate access pattern
  if (access_pattern == 0) {
    // Sequential access
    for (size_t i = 0; i < cache_size; ++i) {
      access_indices.push_back(i);
    }
  } else if (access_pattern == 1) {
    // Random access
    std::uniform_int_distribution<size_t> dist(0, cache_size - 1);
    for (size_t i = 0; i < cache_size; ++i) {
      access_indices.push_back(dist(gen));
    }
  } else {
    // Strided access
    const size_t stride = 16;
    for (size_t i = 0; i < cache_size; i += stride) {
      access_indices.push_back(i);
    }
  }

  for (auto _ : state) {
    float sum = 0.0f;
    for (size_t idx : access_indices) {
      sum += cache_data[idx];
    }
    benchmark::DoNotOptimize(sum);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(access_indices.size() * sizeof(float)));
  state.counters["cache_size_kb"] = (cache_size * sizeof(float)) / 1024;
  state.counters["access_pattern"] = access_pattern;
}

BENCHMARK(BM_CachePatterns)
    ->Args({1024, 0})        // 4KB sequential
    ->Args({1024, 1})        // 4KB random
    ->Args({1024, 2})        // 4KB strided
    ->Args({64 * 1024, 0})   // 256KB sequential
    ->Args({64 * 1024, 1})   // 256KB random
    ->Args({64 * 1024, 2})   // 256KB strided
    ->Args({1024 * 1024, 0}) // 4MB sequential
    ->Args({1024 * 1024, 1}) // 4MB random
    ->Unit(benchmark::kMicrosecond);

// Custom main function to initialize benchmark state
static void BenchmarkInitialization(benchmark::State& state) {
  auto& benchmark_state = BenchmarkState::GetInstance();
  benchmark_state.Initialize();

  for (auto _ : state) {
    // Just measure initialization overhead
    volatile bool initialized = true;
    benchmark::DoNotOptimize(initialized);
  }
}

BENCHMARK(BenchmarkInitialization);

}  // namespace
}  // namespace gcpp

// Custom main function with proper initialization
int main(int argc, char** argv) {
  // Initialize Google Benchmark
  benchmark::Initialize(&argc, argv);

  // Initialize our benchmark state
  gcpp::BenchmarkState::GetInstance().Initialize();

  // Check that benchmark will actually run
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  // Run benchmarks
  benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  benchmark::Shutdown();

  return 0;
}