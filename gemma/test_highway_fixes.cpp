#include <iostream>

// Test compilation of highway scalar fallback fixes
#define HWY_TARGET HWY_SCALAR

#include "util/basics.h"
#include "ops/highway_scalar_fallback.h"

// Test the template instantiation that was failing
namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void test_fast_promote_odd_to() {
    // Test the FastPromoteOddTo template with the same signature as matmul-inl.h
    hn::ScalableTag<float> df;
    hn::Repartition<BF16, decltype(df)> dbf;

    // Create a test BF16 vector (in scalar mode, just one element)
    auto vbf = hn::Set(dbf, static_cast<BF16>(1.5f));

    // This should compile without errors now
    auto result = FastPromoteOddTo(df, vbf);

    std::cout << "FastPromoteOddTo test passed - no compilation errors" << std::endl;
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp

int main() {
    gcpp::HWY_NAMESPACE::test_fast_promote_odd_to();
    return 0;
}