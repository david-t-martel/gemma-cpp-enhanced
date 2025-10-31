// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Intel SIMD intrinsics implementations for missing Highway scalar operations
// Provides native Intel CPU optimizations for gemma.cpp

#ifndef THIRD_PARTY_GEMMA_CPP_OPS_INTEL_SIMD_OPS_H_
#define THIRD_PARTY_GEMMA_CPP_OPS_INTEL_SIMD_OPS_H_

#ifdef INTEL_OPTIMIZED_BUILD

#include <immintrin.h>
#include <cstdint>
#include <type_traits>

namespace gcpp {
namespace intel_simd {

// Intel intrinsics implementation of ConcatEven
template<typename T>
__m256i ConcatEven(__m256i a, __m256i b) {
  if constexpr (sizeof(T) == 1) {
    // 8-bit elements: use unpack operations
    __m256i even_a = _mm256_unpacklo_epi8(a, _mm256_setzero_si256());
    __m256i even_b = _mm256_unpacklo_epi8(b, _mm256_setzero_si256());
    return _mm256_packus_epi16(even_a, even_b);
  } else if constexpr (sizeof(T) == 2) {
    // 16-bit elements: use shuffle to extract even elements
    const __m256i mask = _mm256_set_epi8(
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i even_a = _mm256_shuffle_epi8(a, mask);
    __m256i even_b = _mm256_shuffle_epi8(b, mask);
    return _mm256_unpacklo_epi64(even_a, even_b);
  } else if constexpr (sizeof(T) == 4) {
    // 32-bit elements: use permute to extract even elements
    __m256i even_a = _mm256_permute4x64_epi64(a, 0b11011000);  // 0,2,1,3 -> 0,2,X,X
    __m256i even_b = _mm256_permute4x64_epi64(b, 0b11011000);
    return _mm256_unpacklo_epi64(even_a, even_b);
  }
}

// Intel intrinsics implementation of ConcatOdd
template<typename T>
__m256i ConcatOdd(__m256i a, __m256i b) {
  if constexpr (sizeof(T) == 1) {
    // 8-bit elements: use unpack operations for odd elements
    __m256i odd_a = _mm256_unpackhi_epi8(a, _mm256_setzero_si256());
    __m256i odd_b = _mm256_unpackhi_epi8(b, _mm256_setzero_si256());
    return _mm256_packus_epi16(odd_a, odd_b);
  } else if constexpr (sizeof(T) == 2) {
    // 16-bit elements: use shuffle to extract odd elements
    const __m256i mask = _mm256_set_epi8(
        31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1,
        31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    __m256i odd_a = _mm256_shuffle_epi8(a, mask);
    __m256i odd_b = _mm256_shuffle_epi8(b, mask);
    return _mm256_unpacklo_epi64(odd_a, odd_b);
  } else if constexpr (sizeof(T) == 4) {
    // 32-bit elements: use permute to extract odd elements
    __m256i odd_a = _mm256_permute4x64_epi64(a, 0b11110001);  // 1,3,X,X
    __m256i odd_b = _mm256_permute4x64_epi64(b, 0b11110001);
    return _mm256_unpacklo_epi64(odd_a, odd_b);
  }
}

// Intel intrinsics implementation of InterleaveWholeLower
template<typename T>
__m256i InterleaveWholeLower(__m256i a, __m256i b) {
  if constexpr (sizeof(T) == 1) {
    return _mm256_unpacklo_epi8(a, b);
  } else if constexpr (sizeof(T) == 2) {
    return _mm256_unpacklo_epi16(a, b);
  } else if constexpr (sizeof(T) == 4) {
    return _mm256_unpacklo_epi32(a, b);
  } else if constexpr (sizeof(T) == 8) {
    return _mm256_unpacklo_epi64(a, b);
  }
}

// Intel intrinsics implementation of InterleaveWholeUpper
template<typename T>
__m256i InterleaveWholeUpper(__m256i a, __m256i b) {
  if constexpr (sizeof(T) == 1) {
    return _mm256_unpackhi_epi8(a, b);
  } else if constexpr (sizeof(T) == 2) {
    return _mm256_unpackhi_epi16(a, b);
  } else if constexpr (sizeof(T) == 4) {
    return _mm256_unpackhi_epi32(a, b);
  } else if constexpr (sizeof(T) == 8) {
    return _mm256_unpackhi_epi64(a, b);
  }
}

// Intel intrinsics implementation of PromoteUpperTo
template<typename FromT, typename ToT>
auto PromoteUpperTo(__m256i v) {
  if constexpr (std::is_same_v<FromT, uint8_t> && std::is_same_v<ToT, uint16_t>) {
    // Extract upper 128 bits and zero-extend to 16-bit
    __m128i upper = _mm256_extracti128_si256(v, 1);
    return _mm256_cvtepu8_epi16(upper);
  } else if constexpr (std::is_same_v<FromT, uint16_t> && std::is_same_v<ToT, uint32_t>) {
    // Extract upper 128 bits and zero-extend to 32-bit
    __m128i upper = _mm256_extracti128_si256(v, 1);
    return _mm256_cvtepu16_epi32(upper);
  } else if constexpr (std::is_same_v<FromT, uint32_t> && std::is_same_v<ToT, uint64_t>) {
    // Extract upper 128 bits and zero-extend to 64-bit
    __m128i upper = _mm256_extracti128_si256(v, 1);
    return _mm256_cvtepu32_epi64(upper);
  } else if constexpr (std::is_same_v<FromT, int8_t> && std::is_same_v<ToT, int16_t>) {
    // Extract upper 128 bits and sign-extend to 16-bit
    __m128i upper = _mm256_extracti128_si256(v, 1);
    return _mm256_cvtepi8_epi16(upper);
  } else if constexpr (std::is_same_v<FromT, int16_t> && std::is_same_v<ToT, int32_t>) {
    // Extract upper 128 bits and sign-extend to 32-bit
    __m128i upper = _mm256_extracti128_si256(v, 1);
    return _mm256_cvtepi16_epi32(upper);
  } else if constexpr (std::is_same_v<FromT, int32_t> && std::is_same_v<ToT, int64_t>) {
    // Extract upper 128 bits and sign-extend to 64-bit
    __m128i upper = _mm256_extracti128_si256(v, 1);
    return _mm256_cvtepi32_epi64(upper);
  }
}

// Intel intrinsics implementation of PromoteOddTo
template<typename FromT, typename ToT>
auto PromoteOddTo(__m256i v) {
  if constexpr (std::is_same_v<FromT, uint8_t> && std::is_same_v<ToT, uint16_t>) {
    // Extract odd bytes and promote to 16-bit
    __m256i odd = _mm256_srli_epi16(v, 8);
    return _mm256_and_si256(odd, _mm256_set1_epi16(0xFF));
  } else if constexpr (std::is_same_v<FromT, uint16_t> && std::is_same_v<ToT, uint32_t>) {
    // Extract odd 16-bit elements and promote to 32-bit
    __m256i odd = _mm256_srli_epi32(v, 16);
    return _mm256_and_si256(odd, _mm256_set1_epi32(0xFFFF));
  } else if constexpr (std::is_same_v<FromT, uint32_t> && std::is_same_v<ToT, uint64_t>) {
    // Extract odd 32-bit elements and promote to 64-bit
    __m256i odd = _mm256_srli_epi64(v, 32);
    return odd;
  }
}

// Intel intrinsics implementation of OrderedDemote2To
template<typename FromT, typename ToT>
__m256i OrderedDemote2To(__m256i a, __m256i b) {
  if constexpr (std::is_same_v<FromT, int32_t> && std::is_same_v<ToT, int16_t>) {
    return _mm256_packs_epi32(a, b);
  } else if constexpr (std::is_same_v<FromT, int16_t> && std::is_same_v<ToT, int8_t>) {
    return _mm256_packs_epi16(a, b);
  } else if constexpr (std::is_same_v<FromT, uint32_t> && std::is_same_v<ToT, uint16_t>) {
    return _mm256_packus_epi32(a, b);
  } else if constexpr (std::is_same_v<FromT, uint16_t> && std::is_same_v<ToT, uint8_t>) {
    return _mm256_packus_epi16(a, b);
  }
}

// Intel intrinsics implementation of ZeroExtendVector
template<typename T>
__m256i ZeroExtendVector(__m128i v) {
  return _mm256_inserti128_si256(_mm256_setzero_si256(), v, 0);
}

// Intel intrinsics implementation of UpperHalf
template<typename T>
__m128i UpperHalf(__m256i v) {
  return _mm256_extracti128_si256(v, 1);
}

// Intel intrinsics implementation of LowerHalf
template<typename T>
__m128i LowerHalf(__m256i v) {
  return _mm256_extracti128_si256(v, 0);
}

// Float vector operations using Intel AVX
class IntelFloatOps {
public:
  // Vectorized multiply-add: a * b + c
  static __m256 MulAdd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
  }

  // Vectorized exponential function
  static __m256 Exp(__m256 x) {
    // Approximation using polynomial (can be replaced with Intel SVML)
    const __m256 log2e = _mm256_set1_ps(1.44269504f);
    const __m256 ln2 = _mm256_set1_ps(0.69314718f);

    __m256 fx = _mm256_mul_ps(x, log2e);
    __m256 ix = _mm256_floor_ps(fx);
    __m256 fx_fract = _mm256_sub_ps(fx, ix);

    // Use 2^ix * 2^fx_fract approximation
    // This is a simplified version; Intel SVML provides better accuracy
    return _mm256_exp_ps(x);  // Use Intel SVML if available
  }

  // Vectorized ReLU: max(0, x)
  static __m256 ReLU(__m256 x) {
    const __m256 zero = _mm256_setzero_ps();
    return _mm256_max_ps(x, zero);
  }

  // Vectorized GELU approximation
  static __m256 GELU(__m256 x) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 coeff = _mm256_set1_ps(0.7978845608f);  // sqrt(2/pi)
    const __m256 poly_coeff = _mm256_set1_ps(0.044715f);

    // x^3
    __m256 x_cubed = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    // 0.044715 * x^3
    __m256 poly_term = _mm256_mul_ps(poly_coeff, x_cubed);
    // x + 0.044715 * x^3
    __m256 inner = _mm256_add_ps(x, poly_term);
    // sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = _mm256_mul_ps(coeff, inner);

    // Approximate tanh (can use Intel SVML for better accuracy)
    __m256 tanh_val = _mm256_tanh_ps(inner);  // Use Intel SVML if available

    // 1 + tanh(...)
    __m256 one_plus_tanh = _mm256_add_ps(one, tanh_val);
    // x * (1 + tanh(...))
    __m256 result = _mm256_mul_ps(x, one_plus_tanh);
    // 0.5 * x * (1 + tanh(...))
    return _mm256_mul_ps(half, result);
  }
};

// Check Intel SIMD availability at runtime
bool IsIntelSIMDAvailable() {
  // Check for AVX2 support
  int cpu_info[4];
  __cpuid(cpu_info, 7);
  return (cpu_info[1] & (1 << 5)) != 0;  // AVX2 bit
}

// Check Intel AVX-512 availability
bool IsIntelAVX512Available() {
  int cpu_info[4];
  __cpuid(cpu_info, 7);
  return (cpu_info[1] & (1 << 16)) != 0;  // AVX-512F bit
}

}  // namespace intel_simd
}  // namespace gcpp

#endif  // INTEL_OPTIMIZED_BUILD
#endif  // THIRD_PARTY_GEMMA_CPP_OPS_INTEL_SIMD_OPS_H_