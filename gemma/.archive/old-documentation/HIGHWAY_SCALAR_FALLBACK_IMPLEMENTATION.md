# Highway SIMD Scalar Fallback Implementation Guide

## Problem Statement

The gemma.cpp codebase uses Highway SIMD functions that are marked as "unsupported" in the scalar backend. This causes compilation failures when building for scalar-only targets or when SIMD is disabled.

## Missing Functions

The following functions are used in gemma.cpp but not implemented in Highway's scalar backend:

1. **ConcatEven, ConcatOdd** - Used in compression/sfp-inl.h and compression/nuq-inl.h
2. **InterleaveWholeLower, InterleaveWholeUpper** - Used in compression/sfp-inl.h
3. **InterleaveEven, InterleaveOdd** - Used in compression/sfp-inl.h
4. **PromoteUpperTo, PromoteOddTo** - Used in ops/matmul-inl.h, compression/sfp-inl.h, compression/nuq-inl.h
5. **OrderedDemote2To** - Used in compression/compress-inl.h
6. **ZeroExtendVector** - Used in compression/nuq-inl.h
7. **UpperHalf** - Used in ops/fp_arith-inl.h, compression/nuq_test.cc

## Solution

### 1. Header File: `highway_scalar_fallback.h`

Place this file in the gemma.cpp root directory. The implementation provides scalar fallbacks that:

- Work with single scalar values (1 lane)
- Maintain semantic correctness for scalar operations
- Include proper type safety and bounds checking
- Handle signed/unsigned saturation in demote operations

### 2. Integration Points

Add the fallback header to these files:

```cpp
#if HWY_TARGET == HWY_SCALAR
#include "highway_scalar_fallback.h"
#endif
```

**Files to modify:**
- `compression/sfp-inl.h`
- `compression/nuq-inl.h`
- `compression/compress-inl.h`
- `ops/matmul-inl.h`
- `ops/fp_arith-inl.h`

### 3. CMake Integration

Add the header to the CMakeLists.txt sources list:

```cmake
set(SOURCES_LIBGEMMA
  # ... existing files ...
  highway_scalar_fallback.h
)
```

## Function Semantics

### Concatenation Functions

- **ConcatEven**: Returns `lo` vector (even index 0 in scalar)
- **ConcatOdd**: Returns `hi` vector (represents odd index behavior)

### Interleave Functions

- **InterleaveWholeLower**: Returns first vector `a`
- **InterleaveWholeUpper**: Returns second vector `b`
- **InterleaveEven**: Returns first vector `a` (even indices)
- **InterleaveOdd**: Returns second vector `b` (odd indices)

### Promotion Functions

- **PromoteUpperTo**: Converts single element to wider type
- **PromoteOddTo**: Converts single element to wider type

### Demotion Functions

- **OrderedDemote2To**: Converts first vector element to narrower type with saturation

### Extension Functions

- **ZeroExtendVector**: Type conversion of single element
- **UpperHalf**: Type conversion/extraction of single element

## Verification

### Build Test

```bash
cd gemma.cpp
cmake -B build -DHWY_FORCE_STATIC_DISPATCH=ON -DHWY_COMPILE_ONLY_SCALAR=ON
cmake --build build
```

### Runtime Test

```bash
# Run with scalar dispatch forced
HWY_COMPILE_ONLY_SCALAR=1 ./build/gemma_test
```

## Performance Considerations

- Scalar fallbacks have minimal overhead (single element operations)
- No vectorization benefits, but ensures compilation succeeds
- Maintains correctness for scalar-only builds
- Enables broader platform compatibility

## Integration Testing

After implementation, verify that:

1. All compression modules compile with scalar backend
2. Matrix operations work correctly in scalar mode
3. Numerical results match reference implementations
4. No performance regressions in SIMD builds (fallbacks only used for scalar)

## File Structure

```
gemma.cpp/
├── highway_scalar_fallback.h          # New fallback header
├── compression/
│   ├── sfp-inl.h                     # Modified: add fallback include
│   ├── nuq-inl.h                     # Modified: add fallback include
│   └── compress-inl.h                # Modified: add fallback include
├── ops/
│   ├── matmul-inl.h                  # Modified: add fallback include
│   └── fp_arith-inl.h                # Modified: add fallback include
└── CMakeLists.txt                     # Modified: add header to sources
```

This implementation ensures gemma.cpp compiles successfully on all targets while maintaining correctness and performance characteristics.