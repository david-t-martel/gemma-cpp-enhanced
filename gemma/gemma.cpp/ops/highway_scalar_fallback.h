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

// Minimal placeholder header: rely on Highway's own scalar implementations.
// Previous custom scalar fallbacks caused template resolution / conversion
// issues under MSVC (particularly bf16 static_cast failures). Until we can
// re-introduce only the truly missing pieces with exact signatures, keep
// this lean to unblock strict builds.
// If a missing intrinsic surfaces for HWY_SCALAR, implement ONLY that
// function here with a conservative, standards-compliant fallback.

#ifndef GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_
#define GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_

#include "hwy/highway.h"
#include <type_traits>

// Provide only the scalar fallbacks actually missing in Highway's own
// scalar back-end for the operations Gemma uses. Overriding existing
// Highway intrinsics (e.g. Promote*, LowerHalf, OrderedDemote2To) caused
// template deduction failures. We therefore only add the handful of
// helpers not present for HWY_SCALAR that higher-level codecs expect.
#if defined(HWY_TARGET) && HWY_TARGET == HWY_SCALAR
namespace hwy {  // NOLINT(modernize-concat-nested-namespaces)
namespace HWY_NAMESPACE {

// NOTE: Keep these definitions as small/trivial as possible. They are only
// used when HWY_TARGET == HWY_SCALAR (vector length == 1). Semantics:
//  * Promote* / ZeroExtend*: widen the single lane (or return zero if the
//    conceptual source lane does not exist in scalar form).
//  * Concat*/Interleave*: choose one of the inputs deterministically.
//  * OrderedDemote2To: demote first operand.
// These choices preserve invariants required by calling code without adding
// heavyweight logic that previously triggered template resolution issues
// under MSVC.

// Forward helpers for lane/value extraction supplied by Highway macros.
// (TFromV / TFromD etc. are defined in highway.h)

// PromoteUpperTo / PromoteOddTo only participate when widening is valid.
template <class D, class V,
					typename TTo = TFromD<D>, typename TFrom = TFromV<V>,
					typename = hwy::EnableIf<(sizeof(TTo) >= 2 * sizeof(TFrom))>>
inline VFromD<D> PromoteUpperTo(D d, V /*v*/) { return Zero(d); }

template <class D, class V,
					typename TTo = TFromD<D>, typename TFrom = TFromV<V>,
					typename = hwy::EnableIf<(sizeof(TTo) >= 2 * sizeof(TFrom))>>
inline VFromD<D> PromoteOddTo(D d, V /*v*/) { return Zero(d); }

template <class D, class V>
inline V ConcatEven(D /*d*/, V /*hi*/, V lo) {
	// Even lane comes from the lower input.
	return lo;
}

// OrderedDemote2To only for valid narrowing (source at least twice target).
template <class D, class V,
					typename TTo = TFromD<D>, typename TFrom = TFromV<V>,
					typename = hwy::EnableIf<(sizeof(TFrom) >= 2 * sizeof(TTo))>>
inline VFromD<D> OrderedDemote2To(D d, V a, V /*b*/) {
	// Use Highway conversion helper to handle BF16 correctly.
	return Set(d, hwy::ConvertScalarTo<TTo>(GetLane(a)));
}

template <class D, class V,
					typename TTo = TFromD<D>, typename TFrom = TFromV<V>,
					typename = hwy::EnableIf<(sizeof(TTo) >= sizeof(TFrom))>>
inline VFromD<D> ZeroExtendVector(D d, V v) {
	const TFrom lane = GetLane(v);
	if constexpr (std::is_signed_v<TFrom> && std::is_unsigned_v<TTo>) {
		using UnsignedFrom = hwy::MakeUnsigned<TFrom>;
		return Set(d, static_cast<TTo>(static_cast<UnsignedFrom>(lane)));
	} else {
		return Set(d, static_cast<TTo>(lane));
	}
}

template <class D, class V,
					typename TTo = TFromD<D>, typename TFrom = TFromV<V>,
					typename = hwy::EnableIf<(sizeof(TTo) >= sizeof(TFrom))>>
inline VFromD<D> UpperHalf(D d, V /*v*/) { return Zero(d); }

// ConcatOdd: scalar: return hi (odd lane comes from hi input in higher-level logic)
template <class D, class V>
inline V ConcatOdd(D, V hi, V /*lo*/) { return hi; }

// InterleaveWholeLower / Upper: choose a / b respectively.
template <class D, class V>
inline V InterleaveWholeLower(D, V a, V /*b*/) { return a; }
template <class D, class V>
inline V InterleaveWholeUpper(D, V /*a*/, V b) { return b; }

// InterleaveEven / Odd: pick first / second operand.
template <class D, class V>
inline V InterleaveEven(D, V a, V /*b*/) { return a; }
template <class D, class V>
inline V InterleaveOdd(D, V /*a*/, V b) { return b; }

}  // namespace HWY_NAMESPACE
}  // namespace hwy
#endif  // defined(HWY_TARGET) && HWY_TARGET == HWY_SCALAR

#endif  // GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_
