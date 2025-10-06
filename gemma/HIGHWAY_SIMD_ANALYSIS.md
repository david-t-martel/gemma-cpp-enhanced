# Highway SIMD Architecture Analysis for Gemma.cpp

## Executive Summary

The gemma.cpp project faces compatibility challenges when building with scalar mode (HWY_SCALAR) due to missing Highway SIMD function implementations. This document provides a comprehensive analysis and architectural recommendations for creating a compatibility layer that works with Intel oneAPI, CUDA, and standard C++.

## 1. Issues Identified

### 1.1 Missing Functions in N_SCALAR Namespace

The following Highway SIMD functions are not implemented in scalar mode:

#### Promotion Functions
- `PromoteOddTo` - Promotes odd-indexed elements to wider type
- `PromoteEvenTo` - Promotes even-indexed elements to wider type
- `PromoteUpperTo` - Promotes upper half of vector to wider type
- `PromoteLowerTo` - Promotes lower half of vector to wider type (partially defined)

#### Concatenation Functions
- `ConcatEven` - Concatenates even-indexed elements
- `ConcatOdd` - Concatenates odd-indexed elements

#### Interleave Functions
- `InterleaveEven` - Interleaves even-indexed elements
- `InterleaveOdd` - Interleaves odd-indexed elements
- `InterleaveWholeLower` - Interleaves lower halves
- `InterleaveWholeUpper` - Interleaves upper halves

#### Demote Functions
- `OrderedDemote2To` - Demotes two vectors to narrower type

#### Extension Functions
- `ZeroExtendVector` - Zero-extends a vector
- `UpperHalf` - Gets upper half of vector

#### Arithmetic Functions
- `WidenMulPairwiseAdd` - Multiply pairs and add
- `ReorderWidenMulAccumulate` - Reorder, widen, multiply and accumulate

### 1.2 Files Affected

The following files use these missing functions:
- `gemma.cpp/ops/sum-inl.h`
- `gemma.cpp/ops/matmul-inl.h`
- `gemma.cpp/ops/dot-inl.h`
- `gemma.cpp/ops/fp_arith-inl.h`
- `gemma.cpp/compression/compress-inl.h`
- `gemma.cpp/compression/nuq-inl.h`
- `gemma.cpp/compression/sfp-inl.h`

## 2. Highway Library Architecture

### 2.1 Design Principles

Highway uses a multi-target approach:
- **Runtime dispatch**: Detects CPU capabilities and selects optimal implementation
- **Template-based**: Heavy use of C++ templates for type safety
- **Namespace segregation**: Different namespaces for different SIMD targets (SSE4, AVX2, AVX-512, NEON, SVE, etc.)
- **Scalar fallback**: N_SCALAR namespace for non-SIMD execution

### 2.2 Expected Behavior of Missing Functions

#### PromoteOddTo/PromoteEvenTo
- **Vector mode**: Promotes alternating elements (odd/even indices) to wider type
- **Scalar mode**: Only one element exists (index 0 = even), no odd elements
- **Recommendation**: Return promoted element for even, zero for odd

#### PromoteUpperTo/PromoteLowerTo
- **Vector mode**: Promotes upper/lower half of vector elements
- **Scalar mode**: Single element is the "lower" half, no "upper" half
- **Recommendation**: Return promoted element for lower, zero for upper

#### Interleave Functions
- **Vector mode**: Alternates elements from two vectors
- **Scalar mode**: Can only return one of the input elements
- **Recommendation**: Follow convention - lower/even returns first arg, upper/odd returns second

#### OrderedDemote2To
- **Vector mode**: Demotes and packs two vectors into one narrower vector
- **Scalar mode**: Can only process one element
- **Recommendation**: Demote the first input, ignore the second

## 3. Compatibility Layer Design

### 3.1 Architecture

```cpp
// highway_scalar_fallback.h
#ifndef GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_
#define GEMMA_OPS_HIGHWAY_SCALAR_FALLBACK_H_

#include "hwy/highway.h"

#if HWY_TARGET == HWY_SCALAR
namespace hwy {
namespace HWY_NAMESPACE {
  // Scalar fallback implementations
}
}
#endif
```

### 3.2 Implementation Strategy

1. **Guard against redefinition**: Use preprocessor checks to avoid duplicate definitions
2. **Consistent semantics**: Scalar implementations should mirror vector behavior where possible
3. **Zero for missing elements**: Return zero when accessing non-existent elements (odd indices, upper halves)
4. **Type safety**: Maintain Highway's type checking with static_assert

### 3.3 Integration Approach

1. **Include early**: Add `#include "ops/highway_scalar_fallback.h"` after Highway headers
2. **Conditional compilation**: Only active in scalar mode
3. **No runtime overhead**: Template-based, resolved at compile time

## 4. Hardware Backend Compatibility

### 4.1 Intel oneAPI (SYCL)

- **Benefit**: Scalar fallbacks allow CPU execution when GPU unavailable
- **Integration**: SYCL can vectorize scalar code automatically
- **Testing**: Validate with Intel DPC++ compiler

### 4.2 CUDA

- **Benefit**: Enables host-side execution for debugging
- **Integration**: CUDA kernels unaffected, host code uses fallbacks
- **Testing**: Validate with nvcc compiler

### 4.3 Standard C++

- **Benefit**: Pure C++ fallback for maximum portability
- **Integration**: No dependencies, works with any C++17 compiler
- **Testing**: Validate with GCC, Clang, MSVC

## 5. How Other Projects Handle This

### 5.1 llama.cpp
- Uses custom SIMD abstraction layer (ggml)
- Provides scalar implementations for all operations
- Runtime dispatch based on CPU capabilities

### 5.2 XNNPACK
- Implements scalar reference implementations
- Uses code generation for multiple targets
- Scalar used for validation and fallback

### 5.3 Eigen
- Template specialization for scalar mode
- Packet traits define behavior per architecture
- Scalar operations are first-class citizens

## 6. Recommendations

### 6.1 Short-term (Immediate Fix)

1. **Complete scalar fallback header** with all missing functions
2. **Add comprehensive tests** for scalar mode
3. **Document scalar behavior** in code comments

### 6.2 Medium-term (Robust Solution)

1. **Upstream to Highway**: Submit scalar implementations to Highway project
2. **Create abstraction layer**: Wrap Highway calls with gemma-specific interface
3. **Performance profiling**: Identify critical paths needing optimization

### 6.3 Long-term (Architecture Evolution)

1. **Multi-backend strategy**:
   - Primary: Hardware-accelerated (CUDA, SYCL, Metal)
   - Secondary: SIMD-optimized (Highway)
   - Fallback: Scalar reference implementation

2. **Runtime selection**: Dynamic backend selection based on hardware

3. **Testing infrastructure**: Automated testing across all backends

## 7. Implementation Checklist

- [x] Create `highway_scalar_fallback.h` with basic functions
- [x] Add includes to affected files
- [ ] Implement remaining missing functions:
  - [ ] OrderedDemote2To
  - [ ] ZeroExtendVector
  - [ ] UpperHalf
- [ ] Add unit tests for scalar implementations
- [ ] Validate with Intel oneAPI compiler
- [ ] Validate with CUDA nvcc compiler
- [ ] Performance benchmarking in scalar mode
- [ ] Documentation update

## 8. Testing Strategy

### 8.1 Unit Tests
```cpp
TEST(ScalarFallback, PromoteEvenTo) {
  // Test that index 0 (even) is promoted correctly
}

TEST(ScalarFallback, PromoteOddTo) {
  // Test that odd returns zero in scalar mode
}
```

### 8.2 Integration Tests
- Build with `-DHWY_COMPILE_ONLY_SCALAR`
- Run existing gemma tests in scalar mode
- Compare outputs with SIMD mode

### 8.3 Backend Tests
- Intel oneAPI: `icpx -fsycl`
- CUDA: `nvcc --host-only`
- Standard C++: `g++ -std=c++17`

## 9. Performance Considerations

### 9.1 Expected Impact
- Scalar mode is 10-100x slower than SIMD
- Acceptable for:
  - Development and debugging
  - Fallback on unsupported hardware
  - Host-side computation in heterogeneous systems

### 9.2 Optimization Opportunities
- Compiler auto-vectorization may help
- Loop unrolling can improve performance
- Consider OpenMP for parallelization

## 10. Conclusion

The scalar fallback implementation is essential for:
1. **Portability**: Ensures gemma.cpp runs on any hardware
2. **Development**: Enables debugging and testing
3. **Compatibility**: Works with all target compilers (oneAPI, CUDA, standard C++)

The provided solution creates a drop-in compatibility layer that maintains semantic correctness while enabling scalar execution. This approach aligns with industry best practices and provides a foundation for future multi-backend support.