#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <random>
#include <nlohmann/json.hpp>

// Core Gemma includes (assuming they exist)
#include "../../src/session/SessionManager.h"
#include "../../src/session/Session.h"
#include "../../src/core/Model.h"
#include "../../src/backends/base/Backend.h"

namespace gemma {
namespace test {

/**
 * @brief Test fixture base class with common setup/teardown
 */
class GemmaTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test directory
        test_dir_ = std::filesystem::temp_directory_path() / "gemma_tests" / 
                   std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        std::filesystem::create_directories(test_dir_);
        
        // Initialize random generator
        rng_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    void TearDown() override {
        // Clean up test directory
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }
    
    std::filesystem::path test_dir_;
    std::mt19937 rng_;
};

/**
 * @brief Helper class for creating test configurations
 */
class TestConfigBuilder {
public:
    static nlohmann::json create_test_config() {
        return nlohmann::json{
            {"test_mode", true},
            {"log_level", "debug"},
            {"timeout_ms", 5000},
            {"max_retries", 3}
        };
    }
    
    static session::SessionManager::Config create_session_manager_config() {
        session::SessionManager::Config config;
        config.default_max_context_tokens = 1024;
        config.enable_metrics = true;
        config.metrics_interval = std::chrono::minutes(1);
        return config;
    }
};

/**
 * @brief Mock model for testing
 */
class MockModel {
public:
    MOCK_METHOD(bool, load_weights, (const std::string& weights_path), ());
    MOCK_METHOD(bool, load_tokenizer, (const std::string& tokenizer_path), ());
    MOCK_METHOD(std::vector<int>, tokenize, (const std::string& text), (const));
    MOCK_METHOD(std::string, detokenize, (const std::vector<int>& tokens), (const));
    MOCK_METHOD(std::vector<float>, generate, (const std::vector<int>& input_tokens), ());
    MOCK_METHOD(bool, is_loaded, (), (const));
    MOCK_METHOD(size_t, get_vocab_size, (), (const));
    MOCK_METHOD(size_t, get_max_sequence_length, (), (const));
};

/**
 * @brief Test utilities for file operations
 */
class FileTestUtils {
public:
    static std::string create_temp_file(const std::filesystem::path& dir, 
                                       const std::string& content = "",
                                       const std::string& extension = ".txt") {
        auto filename = "test_" + std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count()) + extension;
        auto filepath = dir / filename;
        
        std::ofstream file(filepath);
        file << content;
        file.close();
        
        return filepath.string();
    }
    
    static nlohmann::json create_test_session_export() {
        return nlohmann::json{
            {"version", "1.0"},
            {"created_at", "2024-01-01T00:00:00Z"},
            {"sessions", {
                {
                    {"id", "test-session-1"},
                    {"created_at", "2024-01-01T00:00:00Z"},
                    {"last_activity", "2024-01-01T01:00:00Z"},
                    {"max_context_tokens", 2048},
                    {"messages", {
                        {
                            {"role", "user"},
                            {"content", "Hello, world!"},
                            {"timestamp", "2024-01-01T00:30:00Z"},
                            {"token_count", 5}
                        },
                        {
                            {"role", "assistant"},
                            {"content", "Hello! How can I help you today?"},
                            {"timestamp", "2024-01-01T00:30:01Z"},
                            {"token_count", 12}
                        }
                    }}
                }
            }}
        };
    }
};

/**
 * @brief Performance testing utilities
 */
class PerformanceUtils {
public:
    template<typename Func>
    static std::chrono::nanoseconds measure_execution_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    }
    
    template<typename Func>
    static double measure_average_time_ms(Func&& func, int iterations = 10) {
        std::vector<double> times;
        times.reserve(iterations);
        
        for (int i = 0; i < iterations; ++i) {
            auto duration = measure_execution_time(func);
            times.push_back(duration.count() / 1e6); // Convert to milliseconds
        }
        
        double sum = 0.0;
        for (double time : times) {
            sum += time;
        }
        return sum / iterations;
    }
    
    static void benchmark_with_warmup(std::function<void()> func, 
                                     int warmup_iterations = 3,
                                     int benchmark_iterations = 10) {
        // Warmup
        for (int i = 0; i < warmup_iterations; ++i) {
            func();
        }
        
        // Benchmark
        auto avg_time = measure_average_time_ms(func, benchmark_iterations);
        std::cout << "Average execution time: " << avg_time << " ms" << std::endl;
    }
};

/**
 * @brief Memory testing utilities
 */
class MemoryTestUtils {
public:
    static size_t get_peak_memory_usage() {
        // Platform-specific implementation would go here
        // For testing purposes, return a mock value
        return 1024 * 1024; // 1MB
    }
    
    static bool check_memory_leak(std::function<void()> func, 
                                 size_t acceptable_leak_bytes = 1024) {
        size_t initial_memory = get_peak_memory_usage();
        func();
        size_t final_memory = get_peak_memory_usage();
        
        return (final_memory - initial_memory) <= acceptable_leak_bytes;
    }
};

/**
 * @brief Thread safety testing utilities
 */
class ThreadSafetyUtils {
public:
    template<typename Func>
    static void run_concurrent_test(Func&& func, int num_threads = 4, int iterations_per_thread = 100) {
        std::vector<std::thread> threads;
        std::atomic<int> error_count{0};
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&func, &error_count, iterations_per_thread]() {
                for (int j = 0; j < iterations_per_thread; ++j) {
                    try {
                        func();
                    } catch (...) {
                        error_count++;
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        EXPECT_EQ(error_count, 0) << "Concurrent execution resulted in " << error_count << " errors";
    }
};

/**
 * @brief Custom matchers for Gemma-specific types
 */
MATCHER_P(HasTokenCount, expected_count, "") {
    return arg.size() == expected_count;
}

MATCHER_P(ContainsMessage, message_content, "") {
    for (const auto& msg : arg) {
        if (msg.content == message_content) {
            return true;
        }
    }
    return false;
}

MATCHER_P2(IsWithinRange, min_val, max_val, "") {
    return arg >= min_val && arg <= max_val;
}

/**
 * @brief Test data generators
 */
class TestDataGenerator {
public:
    static std::vector<std::string> generate_test_messages(size_t count = 10) {
        std::vector<std::string> messages;
        messages.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            messages.push_back("Test message " + std::to_string(i + 1) + 
                             " with some content to test tokenization and processing.");
        }
        
        return messages;
    }
    
    static std::vector<int> generate_random_tokens(size_t count, int max_token_id = 32000) {
        std::vector<int> tokens;
        tokens.reserve(count);
        
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<int> dist(1, max_token_id);
        
        for (size_t i = 0; i < count; ++i) {
            tokens.push_back(dist(rng));
        }
        
        return tokens;
    }
    
    static nlohmann::json generate_test_metadata() {
        return nlohmann::json{
            {"test_type", "unit_test"},
            {"created_by", "test_framework"},
            {"tags", {"test", "automated", "ci"}},
            {"version", "1.0.0"}
        };
    }
};

/**
 * @brief Async testing utilities
 */
class AsyncTestUtils {
public:
    template<typename Future>
    static bool wait_for_completion(Future&& future, 
                                   std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        return future.wait_for(timeout) == std::future_status::ready;
    }
    
    static void wait_for_condition(std::function<bool()> condition, 
                                  std::chrono::milliseconds timeout = std::chrono::milliseconds(5000),
                                  std::chrono::milliseconds check_interval = std::chrono::milliseconds(10)) {
        auto start_time = std::chrono::steady_clock::now();
        
        while (!condition()) {
            if (std::chrono::steady_clock::now() - start_time > timeout) {
                FAIL() << "Condition not met within timeout period";
            }
            std::this_thread::sleep_for(check_interval);
        }
    }
};

} // namespace test
} // namespace gemma