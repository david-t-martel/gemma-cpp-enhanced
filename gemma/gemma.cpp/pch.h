// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Precompiled Header for Gemma.cpp
// Contains commonly used headers to speed up compilation

#ifndef GEMMA_CPP_PCH_H_
#define GEMMA_CPP_PCH_H_

// Standard library headers (most frequently used)
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// Platform-specific headers
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // Prevent windows.h from defining min/max macros
#endif
#include <windows.h>
#endif

// Highway SIMD library (heavily templated, slow to compile)
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/detect_targets.h"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/targets.h"

// Commonly used Gemma headers
#include "util/basics.h"
#include "util/allocator.h"
#include "util/threading.h"
#include "util/threading_context.h"

// Mathematics and compression headers (template-heavy)
#include "compression/types.h"
#include "ops/matmul.h"

#endif  // GEMMA_CPP_PCH_H_