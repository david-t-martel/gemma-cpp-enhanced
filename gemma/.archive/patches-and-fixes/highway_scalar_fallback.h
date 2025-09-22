// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Highway SIMD scalar fallback implementations for missing functions.
// This header provides scalar implementations for functions that are
// marked as "unsupported" in the Highway scalar backend but are required
// by gemma.cpp compression and ops modules.

#ifndef HIGHWAY_SCALAR_FALLBACK_H_
#define HIGHWAY_SCALAR_FALLBACK_H_

#include "hwy/highway.h"

// Only define these functions when using the scalar target
#if HWY_TARGET == HWY_SCALAR

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// ------------------------------ ConcatEven/ConcatOdd

// Concatenate even lanes from hi and lo vectors
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> ConcatEven(D d, VFromD<D> hi, VFromD<D> lo) {
  // For scalar: only one lane, so we return lo (even index 0)
  static_assert(MaxLanes(d) == 1, "Scalar should have only one lane");
  (void)hi;  // Unused in scalar case
  return lo;
}

// Concatenate odd lanes from hi and lo vectors
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> ConcatOdd(D d, VFromD<D> hi, VFromD<D> lo) {
  // For scalar: only one lane, so we return hi (odd would be index 1, which doesn't exist)
  // In practice, for scalar we return hi to match expected behavior
  static_assert(MaxLanes(d) == 1, "Scalar should have only one lane");
  (void)lo;  // Unused in scalar case
  return hi;
}

// ------------------------------ InterleaveWhole*

// Interleave lower halves of two vectors
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> InterleaveWholeLower(D d, VFromD<D> a, VFromD<D> b) {
  // For scalar: return a (lower half is the single element)
  static_assert(MaxLanes(d) == 1, "Scalar should have only one lane");
  (void)b;  // Unused in scalar case
  return a;
}

// Interleave upper halves of two vectors
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> InterleaveWholeUpper(D d, VFromD<D> a, VFromD<D> b) {
  // For scalar: return b (upper half equivalent)
  static_assert(MaxLanes(d) == 1, "Scalar should have only one lane");
  (void)a;  // Unused in scalar case
  return b;
}

// ------------------------------ InterleaveEven/InterleaveOdd

// Interleave even-indexed lanes
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> InterleaveEven(D d, VFromD<D> a, VFromD<D> b) {
  // For scalar: return a (index 0 is even)
  static_assert(MaxLanes(d) == 1, "Scalar should have only one lane");
  (void)b;  // Unused in scalar case
  return a;
}

// Interleave odd-indexed lanes
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> InterleaveOdd(D d, VFromD<D> a, VFromD<D> b) {
  // For scalar: return b (odd lanes come from second vector)
  static_assert(MaxLanes(d) == 1, "Scalar should have only one lane");
  (void)a;  // Unused in scalar case
  return b;
}

// ------------------------------ PromoteUpperTo

// Promote upper half to wider type
template <class D, class V, typename T = TFromD<D>>
HWY_API VFromD<D> PromoteUpperTo(D d, V v) {
  // For scalar: just convert the single element to the target type
  using TFrom = TFromV<V>;
  static_assert(sizeof(T) == 2 * sizeof(TFrom), "Target should be twice the width");
  return Set(d, static_cast<T>(GetLane(v)));
}

// ------------------------------ PromoteOddTo

// Promote odd-indexed lanes to wider type
template <class D, class V, typename T = TFromD<D>>
HWY_API VFromD<D> PromoteOddTo(D d, V v) {
  // For scalar: just convert the single element (treat as "odd" conceptually)
  using TFrom = TFromV<V>;
  static_assert(sizeof(T) == 2 * sizeof(TFrom), "Target should be twice the width");
  return Set(d, static_cast<T>(GetLane(v)));
}

// ------------------------------ OrderedDemote2To

// Demote two vectors to narrower type in order
template <class DN, class V, typename TN = TFromD<DN>>
HWY_API VFromD<DN> OrderedDemote2To(DN dn, V a, V b) {
  // For scalar: demote first vector element (ignore second for scalar)
  using TFrom = TFromV<V>;
  static_assert(sizeof(TN) * 2 == sizeof(TFrom), "Target should be half the width");

  // Apply saturation/clamping based on target type
  TFrom val = GetLane(a);
  TN result;

  if constexpr (std::is_signed_v<TN>) {
    // Signed saturation
    constexpr TFrom min_val = static_cast<TFrom>(std::numeric_limits<TN>::min());
    constexpr TFrom max_val = static_cast<TFrom>(std::numeric_limits<TN>::max());
    val = std::max(min_val, std::min(max_val, val));
    result = static_cast<TN>(val);
  } else {
    // Unsigned saturation
    constexpr TFrom max_val = static_cast<TFrom>(std::numeric_limits<TN>::max());
    val = std::max(TFrom{0}, std::min(max_val, val));
    result = static_cast<TN>(val);
  }

  (void)b;  // Unused in scalar case
  return Set(dn, result);
}

// ------------------------------ ZeroExtendVector

// Zero-extend smaller vector to larger vector
template <class D, class V>
HWY_API VFromD<D> ZeroExtendVector(D d, V v) {
  // For scalar: just convert the element type if needed
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;

  if constexpr (std::is_same_v<TFrom, TTo>) {
    // Same type, just return
    return BitCast(d, v);
  } else {
    // Convert element
    return Set(d, static_cast<TTo>(GetLane(v)));
  }
}

// ------------------------------ UpperHalf

// Get upper half of vector
template <class D, class V>
HWY_API VFromD<D> UpperHalf(D d, V v) {
  // For scalar: just return the single element (cast to target type if needed)
  using TFrom = TFromV<V>;
  using TTo = TFromD<D>;

  if constexpr (std::is_same_v<TFrom, TTo>) {
    return BitCast(d, v);
  } else {
    return Set(d, static_cast<TTo>(GetLane(v)));
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#endif  // HWY_TARGET == HWY_SCALAR

#endif  // HIGHWAY_SCALAR_FALLBACK_H_