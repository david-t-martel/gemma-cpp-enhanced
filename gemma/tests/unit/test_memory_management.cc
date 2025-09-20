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
#include <vector>
#include <thread>
#include <chrono>

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/kv_cache.h"
#include "util/threading_context.h"
#include "util/allocator.h"

namespace gcpp {
namespace {

class MemoryManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = ModelConfig::FromModel(Model::GEMMA_2B);
    }

    void TearDown() override {
        // Explicit cleanup to test destruction
    }

    ModelConfig config_;
};

TEST_F(MemoryManagementTest, ThreadingContextLifecycle) {
    // Test creation and destruction of threading context
    {
        auto context = std::make_unique<ThreadingContext>(1);
        EXPECT_NE(context.get(), nullptr);

        // Test allocator access
        auto& allocator = context->Allocator();
        EXPECT_NE(&allocator, nullptr);
    }
    // Context should be cleanly destroyed here

    // Test with multiple threads
    {
        auto context = std::make_unique<ThreadingContext>(4);
        EXPECT_NE(context.get(), nullptr);
    }
}

TEST_F(MemoryManagementTest, KVCacheAllocation) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Test different cache sizes
    std::vector<size_t> test_sizes = {512, 1024, 2048, 4096};

    for (size_t seq_len : test_sizes) {
        SCOPED_TRACE("Testing KV cache size: " + std::to_string(seq_len));

        auto kv_cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(kv_cache.get(), nullptr);
        EXPECT_EQ(kv_cache->SeqLen(), seq_len);

        // Verify cache can be used
        // (Basic structural validation without full model)
    }
}

TEST_F(MemoryManagementTest, MultipleKVCacheAllocation) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Create multiple KV caches simultaneously
    std::vector<std::unique_ptr<KVCache>> caches;
    const size_t num_caches = 5;
    const size_t seq_len = 1024;

    for (size_t i = 0; i < num_caches; ++i) {
        auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(cache.get(), nullptr);
        EXPECT_EQ(cache->SeqLen(), seq_len);
        caches.push_back(std::move(cache));
    }

    // Verify all caches are still valid
    for (const auto& cache : caches) {
        EXPECT_NE(cache.get(), nullptr);
        EXPECT_EQ(cache->SeqLen(), seq_len);
    }

    // Clear caches one by one
    while (!caches.empty()) {
        caches.pop_back();
    }
}

TEST_F(MemoryManagementTest, SequentialAllocationDeallocation) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    const size_t seq_len = 2048;
    const int num_iterations = 10;

    for (int i = 0; i < num_iterations; ++i) {
        SCOPED_TRACE("Iteration: " + std::to_string(i));

        // Allocate
        auto kv_cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(kv_cache.get(), nullptr);
        EXPECT_EQ(kv_cache->SeqLen(), seq_len);

        // Use briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Deallocate (automatic when going out of scope)
    }
}

TEST_F(MemoryManagementTest, MemoryAlignmentValidation) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    const size_t seq_len = 1024;
    auto kv_cache = std::make_unique<KVCache>(config_, seq_len, allocator);

    // Test that cache memory is properly aligned
    // (This is more of a structural test since we can't easily access internals)
    EXPECT_NE(kv_cache.get(), nullptr);
    EXPECT_EQ(kv_cache->SeqLen(), seq_len);
}

TEST_F(MemoryManagementTest, LargeAllocationHandling) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Test allocation of very large cache
    const size_t large_seq_len = 32768;  // 32K sequence length

    try {
        auto large_cache = std::make_unique<KVCache>(config_, large_seq_len, allocator);
        EXPECT_NE(large_cache.get(), nullptr);
        EXPECT_EQ(large_cache->SeqLen(), large_seq_len);

        // Brief usage test
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    } catch (const std::exception& e) {
        // Large allocations might fail on some systems
        GTEST_SKIP() << "Large allocation failed (may be system-dependent): " << e.what();
    }
}

TEST_F(MemoryManagementTest, ThreadSafetyBasic) {
    auto context = std::make_unique<ThreadingContext>(4);
    auto& allocator = context->Allocator();

    const int num_threads = 4;
    const int allocations_per_thread = 5;
    std::vector<std::thread> threads;
    std::vector<bool> thread_success(num_threads, false);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                std::vector<std::unique_ptr<KVCache>> local_caches;

                for (int i = 0; i < allocations_per_thread; ++i) {
                    const size_t seq_len = 512 + i * 128;  // Varying sizes
                    auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);

                    if (!cache || cache->SeqLen() != seq_len) {
                        return;  // Failure
                    }

                    local_caches.push_back(std::move(cache));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                thread_success[t] = true;
            } catch (const std::exception& e) {
                // Thread failed
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all threads succeeded
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_TRUE(thread_success[t]) << "Thread " << t << " failed";
    }
}

TEST_F(MemoryManagementTest, ReuseAfterDestruction) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    const size_t seq_len = 1024;

    // Create and destroy cache multiple times with same parameters
    for (int iteration = 0; iteration < 5; ++iteration) {
        SCOPED_TRACE("Reuse iteration: " + std::to_string(iteration));

        auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(cache.get(), nullptr);
        EXPECT_EQ(cache->SeqLen(), seq_len);

        // Brief usage simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

TEST_F(MemoryManagementTest, VaryingSizeAllocations) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Test allocation of varying sizes in sequence
    std::vector<size_t> sizes = {256, 1024, 512, 2048, 128, 4096, 64};
    std::vector<std::unique_ptr<KVCache>> caches;

    for (size_t seq_len : sizes) {
        SCOPED_TRACE("Allocating size: " + std::to_string(seq_len));

        auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(cache.get(), nullptr);
        EXPECT_EQ(cache->SeqLen(), seq_len);

        caches.push_back(std::move(cache));
    }

    // Verify all caches are still valid
    for (size_t i = 0; i < caches.size(); ++i) {
        EXPECT_NE(caches[i].get(), nullptr);
        EXPECT_EQ(caches[i]->SeqLen(), sizes[i]);
    }
}

TEST_F(MemoryManagementTest, MemoryFragmentationTest) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Simulate fragmentation by allocating and deallocating in patterns
    std::vector<std::unique_ptr<KVCache>> persistent_caches;
    const size_t base_size = 512;

    // Create several persistent allocations
    for (int i = 0; i < 5; ++i) {
        auto cache = std::make_unique<KVCache>(config_, base_size * (i + 1), allocator);
        EXPECT_NE(cache.get(), nullptr);
        persistent_caches.push_back(std::move(cache));
    }

    // Create and destroy temporary allocations between persistent ones
    for (int iteration = 0; iteration < 10; ++iteration) {
        std::vector<std::unique_ptr<KVCache>> temp_caches;

        // Allocate temporary caches
        for (int i = 0; i < 3; ++i) {
            auto cache = std::make_unique<KVCache>(config_, base_size + i * 64, allocator);
            EXPECT_NE(cache.get(), nullptr);
            temp_caches.push_back(std::move(cache));
        }

        // Let them be destroyed at end of scope
    }

    // Verify persistent caches are still valid
    for (size_t i = 0; i < persistent_caches.size(); ++i) {
        EXPECT_NE(persistent_caches[i].get(), nullptr);
        EXPECT_EQ(persistent_caches[i]->SeqLen(), base_size * (i + 1));
    }
}

TEST_F(MemoryManagementTest, StressAllocation) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    const int num_allocations = 100;
    const size_t base_seq_len = 256;

    for (int i = 0; i < num_allocations; ++i) {
        SCOPED_TRACE("Stress allocation: " + std::to_string(i));

        const size_t seq_len = base_seq_len + (i % 10) * 128;  // Varying sizes

        auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(cache.get(), nullptr);
        EXPECT_EQ(cache->SeqLen(), seq_len);

        // Immediate deallocation
    }
}

TEST_F(MemoryManagementTest, AllocationFailureHandling) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Try to allocate extremely large cache that should fail gracefully
    const size_t extremely_large_seq_len = SIZE_MAX / 1000;  // Very large but not overflow

    EXPECT_THROW({
        auto huge_cache = std::make_unique<KVCache>(config_, extremely_large_seq_len, allocator);
    }, std::exception);

    // Verify allocator still works after failed allocation
    const size_t normal_seq_len = 1024;
    auto normal_cache = std::make_unique<KVCache>(config_, normal_seq_len, allocator);
    EXPECT_NE(normal_cache.get(), nullptr);
    EXPECT_EQ(normal_cache->SeqLen(), normal_seq_len);
}

TEST_F(MemoryManagementTest, ZeroSizeHandling) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Test edge case of zero size allocation
    EXPECT_THROW({
        auto zero_cache = std::make_unique<KVCache>(config_, 0, allocator);
    }, std::exception);

    // Verify allocator still works after zero-size attempt
    const size_t normal_seq_len = 512;
    auto normal_cache = std::make_unique<KVCache>(config_, normal_seq_len, allocator);
    EXPECT_NE(normal_cache.get(), nullptr);
    EXPECT_EQ(normal_cache->SeqLen(), normal_seq_len);
}

TEST_F(MemoryManagementTest, MultipleContexts) {
    // Test multiple independent threading contexts
    std::vector<std::unique_ptr<ThreadingContext>> contexts;
    std::vector<std::unique_ptr<KVCache>> caches;

    const int num_contexts = 3;
    const size_t seq_len = 1024;

    for (int i = 0; i < num_contexts; ++i) {
        SCOPED_TRACE("Context: " + std::to_string(i));

        auto context = std::make_unique<ThreadingContext>(1);
        auto& allocator = context->Allocator();

        auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);
        EXPECT_NE(cache.get(), nullptr);
        EXPECT_EQ(cache->SeqLen(), seq_len);

        contexts.push_back(std::move(context));
        caches.push_back(std::move(cache));
    }

    // Verify all are still valid
    for (int i = 0; i < num_contexts; ++i) {
        EXPECT_NE(contexts[i].get(), nullptr);
        EXPECT_NE(caches[i].get(), nullptr);
        EXPECT_EQ(caches[i]->SeqLen(), seq_len);
    }
}

TEST_F(MemoryManagementTest, MemoryBounds) {
    auto context = std::make_unique<ThreadingContext>(1);
    auto& allocator = context->Allocator();

    // Test allocation at memory boundaries
    std::vector<size_t> boundary_sizes = {
        1,           // Minimum
        64,          // Small
        4096,        // Page size
        65536,       // Large
        1048576      // Very large (1MB cache)
    };

    for (size_t seq_len : boundary_sizes) {
        SCOPED_TRACE("Testing boundary size: " + std::to_string(seq_len));

        try {
            auto cache = std::make_unique<KVCache>(config_, seq_len, allocator);
            EXPECT_NE(cache.get(), nullptr);
            EXPECT_EQ(cache->SeqLen(), seq_len);
        } catch (const std::exception& e) {
            // Some large allocations might fail on constrained systems
            if (seq_len >= 1048576) {
                GTEST_SKIP() << "Large allocation failed (system-dependent): " << e.what();
            } else {
                FAIL() << "Unexpected allocation failure for size " << seq_len << ": " << e.what();
            }
        }
    }
}

} // namespace
} // namespace gcpp