#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <limits>

// Include the gemma.cpp headers we need
#include "gemma.cpp/compression/types.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "hwy/highway.h"
#include "hwy/profiler.h"

// After highway.h
#include "gemma.cpp/ops/ops-inl.h"

namespace gcpp {
namespace HWY_NAMESPACE {

// Test MinPFilter function
void TestMinPFilter() {
    std::cout << "Testing MinPFilter..." << std::endl;

    hwy::Profiler& p = hwy::Profiler::Get();
    const size_t worker = 0;
    const size_t kSize = 10;
    std::vector<float> logits = {1.0f, 0.5f, 0.3f, 0.2f, 0.1f, 0.05f, 0.02f, 0.01f, 0.005f, 0.001f};

    // Test with min_p = 0.1
    std::vector<float> filtered_logits = logits;
    MinPFilter(filtered_logits.data(), kSize, 0.1f, p, worker);

    // Check filtering results
    assert(filtered_logits[0] == 1.0f);    // Keep: 1.0 >= 0.1
    assert(filtered_logits[1] == 0.5f);    // Keep: 0.5 >= 0.1
    assert(filtered_logits[2] == 0.3f);    // Keep: 0.3 >= 0.1
    assert(filtered_logits[3] == 0.2f);    // Keep: 0.2 >= 0.1
    assert(filtered_logits[4] == 0.1f);    // Keep: 0.1 >= 0.1
    assert(filtered_logits[5] == -std::numeric_limits<float>::infinity()); // Filter: 0.05 < 0.1
    assert(filtered_logits[6] == -std::numeric_limits<float>::infinity()); // Filter: 0.02 < 0.1

    std::cout << "✓ MinPFilter tests passed!" << std::endl;
}

// Test DRY penalty function
void TestApplyDRYPenalty() {
    std::cout << "Testing ApplyDRYPenalty..." << std::endl;

    hwy::Profiler& p = hwy::Profiler::Get();
    const size_t worker = 0;
    const size_t kVocabSize = 10;

    // Create simple logits
    std::vector<float> logits(kVocabSize, 1.0f);

    // Create a sequence with repetition: [1, 2, 3, 1, 2]
    std::vector<int> recent_tokens = {1, 2, 3, 1, 2};
    hwy::Span<const int> token_span(recent_tokens.data(), recent_tokens.size());

    // Apply DRY penalty
    float dry_multiplier = 0.8f;
    float dry_base = 1.75f;
    size_t dry_allowed_length = 2;

    std::vector<float> penalized_logits = logits;
    ApplyDRYPenalty(penalized_logits.data(), kVocabSize, token_span,
                    dry_multiplier, dry_base, dry_allowed_length, p, worker);

    // Tokens 1 and 2 should be penalized (they repeat)
    assert(penalized_logits[1] < logits[1]); // Token 1 is penalized
    assert(penalized_logits[2] < logits[2]); // Token 2 is penalized
    assert(penalized_logits[3] == logits[3]); // Token 3 should not be penalized
    assert(penalized_logits[0] == logits[0]); // Token 0 should not be penalized

    std::cout << "✓ ApplyDRYPenalty tests passed!" << std::endl;
}

// Test dynamic temperature calculation
void TestCalculateDynamicTemperature() {
    std::cout << "Testing CalculateDynamicTemperature..." << std::endl;

    hwy::Profiler& p = hwy::Profiler::Get();
    const size_t worker = 0;
    const size_t kSize = 4;

    // Test with uniform distribution (high entropy)
    std::vector<float> uniform_probs = {0.25f, 0.25f, 0.25f, 0.25f};
    float temp1 = CalculateDynamicTemperature(uniform_probs.data(), kSize,
                                             1.0f, 0.5f, 1.0f, p, worker);

    // Test with peaked distribution (low entropy)
    std::vector<float> peaked_probs = {0.97f, 0.01f, 0.01f, 0.01f};
    float temp2 = CalculateDynamicTemperature(peaked_probs.data(), kSize,
                                             1.0f, 0.5f, 1.0f, p, worker);

    // High entropy should result in higher temperature than low entropy
    assert(temp1 > temp2);

    // Test with dynatemp_range = 0.0 (should return base temperature)
    float temp3 = CalculateDynamicTemperature(uniform_probs.data(), kSize,
                                             1.0f, 0.0f, 1.0f, p, worker);
    assert(std::abs(temp3 - 1.0f) < 1e-6);

    std::cout << "✓ CalculateDynamicTemperature tests passed!" << std::endl;
}

// Test the complete AdvancedSample function
void TestAdvancedSample() {
    std::cout << "Testing AdvancedSample..." << std::endl;

    hwy::Profiler& p = hwy::Profiler::Get();
    const size_t worker = 0;
    const size_t kVocabSize = 10;
    std::mt19937 gen(12345);

    // Create simple logits with a clear maximum
    std::vector<float> logits = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // Test with no advanced sampling
    std::function<bool(int, float)> accept_all = [](int, float) { return true; };
    std::vector<int> empty_tokens;
    hwy::Span<const int> empty_span(empty_tokens.data(), empty_tokens.size());

    auto result = AdvancedSample(logits.data(), kVocabSize, gen,
                                1.0f,   // temperature
                                0.0f,   // min_p (disabled)
                                1.0f,   // typical_p (disabled)
                                kVocabSize, // top_k (all tokens)
                                0.0f,   // dynatemp_range (disabled)
                                1.0f,   // dynatemp_exponent
                                empty_span, // recent_tokens
                                0.0f,   // dry_multiplier (disabled)
                                1.75f,  // dry_base
                                2,      // dry_allowed_length
                                256,    // dry_penalty_last_n
                                accept_all, p, worker);

    // Should return a valid token
    assert(result.token >= 0);
    assert(result.token < static_cast<int>(kVocabSize));
    assert(result.prob > 0.0f);
    assert(result.prob <= 1.0f);

    // Test with temperature = 0.0 (should always return argmax)
    auto result_greedy = AdvancedSample(logits.data(), kVocabSize, gen,
                                       0.0f,   // temperature (greedy)
                                       0.0f,   // min_p
                                       1.0f,   // typical_p
                                       kVocabSize, // top_k
                                       0.0f,   // dynatemp_range
                                       1.0f,   // dynatemp_exponent
                                       empty_span, // recent_tokens
                                       0.0f,   // dry_multiplier
                                       1.75f,  // dry_base
                                       2,      // dry_allowed_length
                                       256,    // dry_penalty_last_n
                                       accept_all, p, worker);

    assert(result_greedy.token == 9); // Should be the argmax (index 9)

    std::cout << "✓ AdvancedSample tests passed!" << std::endl;
}

} // namespace HWY_NAMESPACE
} // namespace gcpp

int main() {
    std::cout << "Running Advanced Sampling Tests\n" << std::endl;

    try {
        gcpp::HWY_NAMESPACE::TestMinPFilter();
        gcpp::HWY_NAMESPACE::TestApplyDRYPenalty();
        gcpp::HWY_NAMESPACE::TestCalculateDynamicTemperature();
        gcpp::HWY_NAMESPACE::TestAdvancedSample();

        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}