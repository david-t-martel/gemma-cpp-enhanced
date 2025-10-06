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

#ifndef GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_
#define GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_

// This header provides scalar fallback implementations for Highway SIMD
// functions that are not available in scalar mode (HWY_SCALAR).
// This ensures compatibility with Intel oneAPI, CUDA, and standard C++.

#include <algorithm>  // For std::clamp
#include <limits>     // For std::numeric_limits
#include <type_traits> // For type traits
#include "hwy/highway.h"
#include "util/basics.h"

// Only define these in scalar mode where they're missing
#if HWY_TARGET == HWY_SCALAR

namespace hwy {
namespace HWY_NAMESPACE {

// PromoteOddTo: Promotes odd-indexed elements to a wider type
// In scalar mode, we only have one element, so we promote that element
template <class D, class V, HWY_IF_T_SIZE_D(D, HWY_MAX(4, 2 * sizeof(TFromV<V>)))>
HWY_API VFromD<D> PromoteOddTo(D d, V v) {
  using TTo = TFromD<D>;
  using TFrom = TFromV<V>;
  static_assert(sizeof(TTo) >= sizeof(TFrom), "Target type must be at least as large as source");

  // In scalar mode, v only has one element at index 0
  // Since there's no index 1 (odd), we return zero (no odd elements)
  return Zero(d);
}

// PromoteEvenTo: Promotes even-indexed elements to a wider type
// In scalar mode, element at index 0 is "even"
template <class D, class V, HWY_IF_T_SIZE_D(D, HWY_MAX(4, 2 * sizeof(TFromV<V>)))>
HWY_API VFromD<D> PromoteEvenTo(D d, V v) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;
  static_assert(sizeof(TTo) >= sizeof(TFrom), "Target type must be at least as large as source");

  // In scalar mode, promote the single element (index 0, which is even)
  const TFrom val = GetLane(v);
  return Set(d, static_cast<TTo>(val));
}

// NOTE: PromoteLowerTo is now provided by Highway directly in generic_ops-inl.h
// We don't define it here to avoid redefinition conflicts

// PromoteUpperTo: Promotes upper half of vector to wider type
// In scalar mode with single element, there is no "upper half"
template <class D, class V>
HWY_API VFromD<D> PromoteUpperTo(D d, V v) {
  // In scalar mode, there's no upper half, return zero
  return Zero(d);
}

// OrderedDemote2To: Demotes two vectors to narrower type and packs them
// In scalar mode, we can only handle one element
template <class D, class V, HWY_IF_T_SIZE_V(V, 4), HWY_IF_T_SIZE_D(D, 2)>
HWY_API VFromD<D> OrderedDemote2To(D d, V a, V b) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;
  static_assert(sizeof(TFrom) == 2 * sizeof(TTo),
                "Source type must be exactly twice the size of target");

  // In scalar mode: demote the first input using proper clamping
  const TFrom a_val = GetLane(a);

  // Clamp to target type range before conversion
  constexpr TFrom min_val = static_cast<TFrom>(std::numeric_limits<TTo>::lowest());
  constexpr TFrom max_val = static_cast<TFrom>(std::numeric_limits<TTo>::max());

  const TFrom clamped = std::clamp(a_val, min_val, max_val);
  return Set(d, static_cast<TTo>(clamped));
}

// Additional overload for BF16->INT8 conversion
template <class D, class V, HWY_IF_T_SIZE_V(V, 2), HWY_IF_T_SIZE_D(D, 1)>
HWY_API VFromD<D> OrderedDemote2To(D d, V a, V b) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;

  const TFrom a_val = GetLane(a);

  // For BF16/F16 to INT8, convert to float first then to target
  if constexpr (std::is_same_v<TFrom, BF16> || std::is_same_v<TFrom, hwy::float16_t>) {
    const float f_val = static_cast<float>(a_val);
    const float clamped = std::clamp(f_val,
                                   static_cast<float>(std::numeric_limits<TTo>::lowest()),
                                   static_cast<float>(std::numeric_limits<TTo>::max()));
    return Set(d, static_cast<TTo>(clamped));
  } else {
    constexpr TFrom min_val = static_cast<TFrom>(std::numeric_limits<TTo>::lowest());
    constexpr TFrom max_val = static_cast<TFrom>(std::numeric_limits<TTo>::max());
    const TFrom clamped = std::clamp(a_val, min_val, max_val);
    return Set(d, static_cast<TTo>(clamped));
  }
}

// UpperHalf: Gets the upper half of a vector
// In scalar mode, there is no upper half
template <class D, class V>
HWY_API VFromD<D> UpperHalf(D d, V v) {
  // In scalar mode: no upper half exists, return zero
  return Zero(d);
}

// LowerHalf: Gets the lower half of a vector
// In scalar mode, the single element is the lower half
template <class D, class V>
HWY_API VFromD<D> LowerHalf(D d, V v) {
  // In scalar mode: the single element is the lower half
  return v;
}

// ConcatEven: Concatenates even-indexed elements from two vectors
// In scalar mode, each vector has one element (index 0, which is even)
template <class D, class V>
HWY_API V ConcatEven(D d, V hi, V lo) {
  // In scalar mode, we only have one element to return
  // Convention: return the low element (similar to taking the first even element)
  return lo;
}

// ConcatOdd: Concatenates odd-indexed elements from two vectors
// In scalar mode, there are no odd-indexed elements
template <class D, class V>
HWY_API V ConcatOdd(D d, V hi, V lo) {
  // In scalar mode, no odd elements exist, return zero
  return Zero(d);
}

// ZeroExtendVector: Zero-extends a vector to a wider type
// In scalar mode, promotes the single element with zero extension
template <class D, class V, HWY_IF_T_SIZE_D(D, HWY_MAX(2, sizeof(TFromV<V>)))>
HWY_API VFromD<D> ZeroExtendVector(D d, V v) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;
  static_assert(sizeof(TTo) >= sizeof(TFrom),
                "Target type must be at least as large as source");
  static_assert(std::is_unsigned_v<TFrom> && std::is_unsigned_v<TTo>,
                "ZeroExtendVector requires unsigned integer types");

  // In scalar mode: zero-extend the single element
  const TFrom val = GetLane(v);
  return Set(d, static_cast<TTo>(val));
}

// InterleaveEven: Interleaves even-indexed elements from two vectors
// In scalar mode, returns the even element (index 0) from lo
template <class D, class V>
HWY_API V InterleaveEven(D d, V hi, V lo) {
  // In scalar mode, return lo (the even element)
  return lo;
}

// InterleaveOdd: Interleaves odd-indexed elements from two vectors
// In scalar mode, there are no odd elements
template <class D, class V>
HWY_API V InterleaveOdd(D d, V hi, V lo) {
  // In scalar mode, return hi (as a convention for the "odd" position)
  return hi;
}

// InterleaveWholeLower: Interleaves whole lower halves of two vectors
// In scalar mode, the single element is the whole "lower half"
template <class D, class V>
HWY_API V InterleaveWholeLower(D d, V a, V b) {
  // In scalar mode, return b (lower comes first in interleave)
  return b;
}

// InterleaveWholeUpper: Interleaves whole upper halves of two vectors
// In scalar mode, there is no upper half
template <class D, class V>
HWY_API V InterleaveWholeUpper(D d, V a, V b) {
  // In scalar mode, return a (upper comes second in interleave)
  return a;
}

// Decompress2: Decompresses 2-element compressed vectors (for compression modules)
// In scalar mode, decompress a single element
template <class D, class V, HWY_IF_T_SIZE_D(D, 4)>
HWY_API VFromD<D> Decompress2(D d, V compressed) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;

  // In scalar mode, just return the decompressed single element
  const TFrom comp_val = GetLane(compressed);
  return Set(d, static_cast<TTo>(comp_val));
}

// Additional overloads for different type combinations
template <class D, class V, HWY_IF_T_SIZE_D(D, 2)>
HWY_API VFromD<D> Decompress2(D d, V compressed) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;

  const TFrom comp_val = GetLane(compressed);
  return Set(d, static_cast<TTo>(comp_val));
}

// PromoteTo overload with explicit type constraints for better template deduction
template <class D, class V, HWY_IF_T_SIZE_D(D, 8), HWY_IF_T_SIZE_V(V, 4)>
HWY_API VFromD<D> PromoteTo(D d, V v) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;
  static_assert(sizeof(TTo) == 2 * sizeof(TFrom), "Target must be twice the size of source");

  const TFrom val = GetLane(v);
  return Set(d, static_cast<TTo>(val));
}

// Additional PromoteTo overload for different size combinations
template <class D, class V, HWY_IF_T_SIZE_D(D, 4), HWY_IF_T_SIZE_V(V, 2)>
HWY_API VFromD<D> PromoteTo(D d, V v) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;
  static_assert(sizeof(TTo) == 2 * sizeof(TFrom), "Target must be twice the size of source");

  const TFrom val = GetLane(v);
  return Set(d, static_cast<TTo>(val));
}

// DemoteTo with proper type constraints
template <class D, class V, HWY_IF_T_SIZE_D(D, 2), HWY_IF_T_SIZE_V(V, 4)>
HWY_API VFromD<D> DemoteTo(D d, V v) {
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;
  static_assert(sizeof(TFrom) == 2 * sizeof(TTo), "Source must be twice the size of target");

  const TFrom val = GetLane(v);

  // Clamp to target range for safe conversion
  constexpr TFrom min_val = static_cast<TFrom>(std::numeric_limits<TTo>::lowest());
  constexpr TFrom max_val = static_cast<TFrom>(std::numeric_limits<TTo>::max());
  const TFrom clamped = std::clamp(val, min_val, max_val);

  return Set(d, static_cast<TTo>(clamped));
}

// NOTE: LoadN is already provided by Highway in scalar-inl.h
// We don't define it here to avoid ambiguous calls

// NOTE: PromoteLowerTo is now provided by Highway directly, no need to define

}  // namespace HWY_NAMESPACE
}  // namespace hwy

// NOTE: FastPromoteOddTo is defined by individual files (like matmul-inl.h)
// that need optimized versions. We don't provide a fallback for it here
// to avoid conflicts with their specialized implementations.

#endif  // HWY_TARGET == HWY_SCALAR

#endif  // GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_