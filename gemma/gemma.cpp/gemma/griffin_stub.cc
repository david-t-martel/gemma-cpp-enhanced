// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Stub implementation for GriffinRecurrent functionality
// This file provides minimal implementations to allow compilation
// when griffin.cc is excluded from the build.

#include <stddef.h>
#include <stdint.h>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "gemma/weights.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/griffin_stub.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Stub implementation of GriffinRecurrent
// This function does nothing but provides a linkable symbol
void GriffinRecurrent(size_t num_tokens, size_t griffin_layer,
                      const LayerWeightsPtrs* layer_weights,
                      Activations& activations, QBatch& qbatch,
                      MatMulEnv& mat_mul_env) {
  // This is a stub implementation
  // In a real implementation, this would contain the Griffin/RecurrentGemma logic
  // For now, we just return to avoid linker errors

  // Log a warning that Griffin functionality is not available
  static bool warned = false;
  if (!warned) {
    fprintf(stderr, "Warning: Griffin/RecurrentGemma functionality is not available in this build\n");
    warned = true;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();