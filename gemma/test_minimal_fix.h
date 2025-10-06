#ifndef TEST_MINIMAL_FIX_H_
#define TEST_MINIMAL_FIX_H_

#include "hwy/highway.h"
#include "util/basics.h"

// Only define in scalar mode
#if HWY_TARGET == HWY_SCALAR

namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// FastPromoteOddTo: Template signature must match the original in matmul-inl.h
template <class DF, class DBF = hn::Repartition<BF16, DF>>
static hn::VFromD<DF> FastPromoteOddTo(DF df, hn::VFromD<DBF> vbf) {
  // In scalar mode, there are no odd elements, so return zero
  return hn::Zero(df);
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp

#endif  // HWY_TARGET == HWY_SCALAR

#endif  // TEST_MINIMAL_FIX_H_