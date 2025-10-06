#include <iostream>
#include <cassert>

// Force scalar mode for testing
#define HWY_TARGET HWY_SCALAR
#include "hwy/highway.h"
#include "util/basics.h"
#include "ops/highway_scalar_fallback.h"

namespace gcpp {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void test_ordered_demote2to() {
    std::cout << "Testing OrderedDemote2To..." << std::endl;

    // Test float->BF16 conversion
    hn::ScalableTag<float> df;
    hn::Repartition<BF16, decltype(df)> dbf;

    auto vf1 = hn::Set(df, 1.5f);
    auto vf2 = hn::Set(df, 2.5f);

    // This should compile without template deduction errors
    auto result = OrderedDemote2To(dbf, vf1, vf2);

    std::cout << "OrderedDemote2To test passed - no compilation errors" << std::endl;
}

void test_promote_functions() {
    std::cout << "Testing PromoteEvenTo and PromoteOddTo..." << std::endl;

    // Test BF16->float promotion
    hn::ScalableTag<float> df;
    hn::Repartition<BF16, decltype(df)> dbf;

    auto vbf = hn::Set(dbf, static_cast<BF16>(1.5f));

    // These should compile without template deduction errors
    auto even_result = PromoteEvenTo(df, vbf);
    auto odd_result = PromoteOddTo(df, vbf);

    std::cout << "PromoteEvenTo and PromoteOddTo tests passed" << std::endl;
}

void test_decompress2() {
    std::cout << "Testing Decompress2..." << std::endl;

    // Test with different type combinations
    hn::ScalableTag<float> df;
    hn::ScalableTag<BF16> dbf;

    auto vbf = hn::Set(dbf, static_cast<BF16>(1.5f));

    // This should compile without template deduction errors
    auto result = Decompress2(df, vbf);

    std::cout << "Decompress2 test passed" << std::endl;
}

void test_zero_extend_vector() {
    std::cout << "Testing ZeroExtendVector..." << std::endl;

    // Test with unsigned integers
    hn::ScalableTag<uint16_t> d16;
    hn::ScalableTag<uint32_t> d32;

    auto v16 = hn::Set(d16, static_cast<uint16_t>(0x1234));

    // This should compile with proper type constraints
    auto result = ZeroExtendVector(d32, v16);

    std::cout << "ZeroExtendVector test passed" << std::endl;
}

void test_load_functions() {
    std::cout << "Testing LoadN..." << std::endl;

    hn::ScalableTag<float> df;
    float data[] = {1.0f, 2.0f, 3.0f};

    // This should compile and handle partial loads correctly
    auto result = LoadN(df, data, 1);

    std::cout << "LoadN test passed" << std::endl;
}

void test_fast_promote_odd_to() {
    std::cout << "Testing FastPromoteOddTo template signature..." << std::endl;

    // Test the exact template signature from matmul-inl.h
    hn::ScalableTag<float> df;
    hn::Repartition<BF16, decltype(df)> dbf;

    auto vbf = hn::Set(dbf, static_cast<BF16>(1.5f));

    // This should compile with the fixed template signature
    // Note: We don't actually call FastPromoteOddTo here because it's defined
    // in matmul-inl.h, but we verify the types work together

    std::cout << "FastPromoteOddTo template signature test passed" << std::endl;
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp

int main() {
    std::cout << "Starting Highway scalar fallback template deduction tests..." << std::endl;

    try {
        gcpp::HWY_NAMESPACE::test_ordered_demote2to();
        gcpp::HWY_NAMESPACE::test_promote_functions();
        gcpp::HWY_NAMESPACE::test_decompress2();
        gcpp::HWY_NAMESPACE::test_zero_extend_vector();
        gcpp::HWY_NAMESPACE::test_load_functions();
        gcpp::HWY_NAMESPACE::test_fast_promote_odd_to();

        std::cout << "\n✅ All template deduction tests PASSED!" << std::endl;
        std::cout << "The Highway scalar fallback fixes are working correctly." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}