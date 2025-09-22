// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// DRY (Don't Repeat Yourself) penalty implementation
// Separate header to avoid multiple definition issues

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_DRY_PENALTY_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_DRY_PENALTY_H_

#include <vector>
#include <string>
#include "hwy/base.h"

namespace gcpp {
namespace HWY_NAMESPACE {

// DRY (Don't Repeat Yourself) penalty implementation
// Applies exponential penalty to tokens that would extend repetitive sequences
void ApplyDRYPenalty(
    float* HWY_RESTRICT logits, size_t vocab_size,
    const hwy::Span<const int>& recent_tokens, float dry_multiplier,
    float dry_base, size_t dry_allowed_length, size_t dry_penalty_last_n,
    const std::vector<std::string>& dry_sequence_breakers = {});

}  // namespace HWY_NAMESPACE
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_OPS_DRY_PENALTY_H_