// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// DRY (Don't Repeat Yourself) penalty implementation

#include "ops/dry_penalty.h"
#include <algorithm>
#include <cmath>
#include "hwy/highway.h"
#include "util/basics.h"

namespace gcpp {
namespace HWY_NAMESPACE {

void ApplyDRYPenalty(
    float* HWY_RESTRICT logits, size_t vocab_size,
    const hwy::Span<const int>& recent_tokens, float dry_multiplier,
    float dry_base, size_t dry_allowed_length, size_t dry_penalty_last_n,
    const std::vector<std::string>& dry_sequence_breakers) {

  if (dry_multiplier <= 0.0f || recent_tokens.size() == 0) {
    return;  // No penalty to apply
  }

  const size_t num_recent = HWY_MIN(recent_tokens.size(), dry_penalty_last_n);
  if (num_recent <= dry_allowed_length) {
    return;  // Not enough tokens to apply penalty
  }

  // Build a map of repetition counts for each token
  std::vector<std::pair<size_t, size_t>> token_repetitions;  // (token_id, max_consecutive_length)

  for (size_t i = 0; i < vocab_size; ++i) {
    size_t max_consecutive = 0;
    size_t current_consecutive = 0;

    // Look for consecutive repetitions in recent history
    for (size_t j = recent_tokens.size() - num_recent; j < recent_tokens.size(); ++j) {
      if (static_cast<size_t>(recent_tokens[j]) == i) {
        current_consecutive++;
        max_consecutive = HWY_MAX(max_consecutive, current_consecutive);
      } else {
        current_consecutive = 0;
      }
    }

    if (max_consecutive > dry_allowed_length) {
      token_repetitions.emplace_back(i, max_consecutive);
    }
  }

  // Apply penalties to repetitive tokens
  for (const auto& [token_id, rep_length] : token_repetitions) {
    if (token_id < vocab_size) {
      // Exponential penalty: penalty = multiplier * (base ^ (length - allowed_length))
      const float penalty_exponent = static_cast<float>(rep_length - dry_allowed_length);
      const float penalty = dry_multiplier * std::pow(dry_base, penalty_exponent);
      logits[token_id] -= penalty;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp