#pragma once

/**
 * @file test_utils.h
 * @brief Common utilities for Gemma.cpp testing
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include "gemma/gemma.h"
#include "gemma/tokenizer.h"
#include "gemma/configs.h"

namespace gemma {
namespace testing {

/**
 * @brief Test fixture base class with common setup
 */
class GemmaTestBase : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    // Helper methods
    std::string GetTestDataPath() const;
    std::string GetTestOutputPath() const;
    bool CreateTestFile(const std::string& filename, const std::string& content);
    bool FileExists(const std::string& path) const;
    std::vector<uint8_t> ReadBinaryFile(const std::string& path) const;

    // Common test data
    static constexpr const char* kTestPrompt = "What is the capital of France?";
    static constexpr const char* kTestResponse = "The capital of France is Paris.";
    static constexpr int kTestTokens[] = {1, 2, 3, 4, 5};

private:
    std::filesystem::path test_data_dir_;
    std::filesystem::path test_output_dir_;
};

/**
 * @brief Mock tokenizer for testing
 */
class MockTokenizer {
public:
    MockTokenizer();
    ~MockTokenizer();

    MOCK_METHOD(bool, Encode, (const std::string& text, std::vector<int>* tokens), (const));
    MOCK_METHOD(bool, Decode, (const std::vector<int>& tokens, std::string* text), (const));
    MOCK_METHOD(int, GetVocabSize, (), (const));
    MOCK_METHOD(int, GetTokenId, (const std::string& token), (const));
    MOCK_METHOD(std::string, GetToken, (int token_id), (const));
};

/**
 * @brief Test data generator
 */
class TestDataGenerator {
public:
    /**
     * @brief Generate random float data
     * @param size Number of elements
     * @param min Minimum value
     * @param max Maximum value
     * @return Vector of random floats
     */
    static std::vector<float> GenerateRandomFloats(size_t size, float min = -1.0f, float max = 1.0f);

    /**
     * @brief Generate random integer data
     * @param size Number of elements
     * @param min Minimum value
     * @param max Maximum value
     * @return Vector of random integers
     */
    static std::vector<int> GenerateRandomInts(size_t size, int min = 0, int max = 100);

    /**
     * @brief Generate test token sequence
     * @param length Sequence length
     * @param vocab_size Vocabulary size
     * @return Vector of token IDs
     */
    static std::vector<int> GenerateTokenSequence(size_t length, int vocab_size = 32000);

    /**
     * @brief Generate test matrix data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param pattern Data pattern (0=random, 1=identity, 2=zeros, 3=ones)
     * @return Flattened matrix data
     */
    static std::vector<float> GenerateMatrixData(size_t rows, size_t cols, int pattern = 0);
};

/**
 * @brief Performance measurement utilities
 */
class PerformanceMeasure {
public:
    PerformanceMeasure();
    ~PerformanceMeasure();

    void Start();
    void Stop();
    double GetElapsedMs() const;
    void Reset();

    // Static helpers
    static double MeasureFunction(std::function<void()> func);
    static void PrintStats(const std::string& name, const std::vector<double>& times);

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;
};

/**
 * @brief Memory usage tracker
 */
class MemoryTracker {
public:
    struct MemoryStats {
        size_t current_usage = 0;
        size_t peak_usage = 0;
        size_t allocation_count = 0;
        size_t deallocation_count = 0;
    };

    static void Reset();
    static MemoryStats GetStats();
    static void EnableTracking(bool enable = true);

private:
    static MemoryStats stats_;
    static bool tracking_enabled_;
    static std::mutex stats_mutex_;
};

/**
 * @brief Test environment setup
 */
class TestEnvironment : public ::testing::Environment {
public:
    void SetUp() override;
    void TearDown() override;

    static TestEnvironment* Instance();

private:
    static TestEnvironment* instance_;
};

/**
 * @brief Assertion helpers
 */
#define EXPECT_NEAR_VECTOR(vec1, vec2, tolerance) \
    do { \
        ASSERT_EQ(vec1.size(), vec2.size()); \
        for (size_t i = 0; i < vec1.size(); ++i) { \
            EXPECT_NEAR(vec1[i], vec2[i], tolerance) << "at index " << i; \
        } \
    } while(0)

#define EXPECT_EQ_VECTOR(vec1, vec2) \
    do { \
        ASSERT_EQ(vec1.size(), vec2.size()); \
        for (size_t i = 0; i < vec1.size(); ++i) { \
            EXPECT_EQ(vec1[i], vec2[i]) << "at index " << i; \
        } \
    } while(0)

/**
 * @brief Temporary file helper
 */
class TempFile {
public:
    explicit TempFile(const std::string& content = "");
    ~TempFile();

    const std::string& GetPath() const { return path_; }
    bool Write(const std::string& content);
    std::string Read() const;

private:
    std::string path_;
};

/**
 * @brief Test parameter combinations
 */
template<typename... Args>
class ParameterizedTest : public GemmaTestBase, 
                         public ::testing::WithParamInterface<std::tuple<Args...>> {
protected:
    auto GetParams() const { return this->GetParam(); }
    
    template<size_t N>
    auto GetParam() const { return std::get<N>(this->GetParam()); }
};

} // namespace testing
} // namespace gemma