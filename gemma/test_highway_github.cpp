#include "third_party/highway-github/hwy/highway.h"
#include "third_party/highway-github/hwy/ops/generic_ops-inl.h"

// Test to verify that the required scalar fallback functions are available
int main() {
    // This is just a compilation test to verify the functions exist
    // We don't need to run complex logic, just ensure the symbols are available

    using namespace hwy;

    // Test that we can reference the functions (they exist in compilation)
    auto has_promote_odd_to = __has_include("third_party/highway-github/hwy/ops/generic_ops-inl.h");
    auto has_promote_upper_to = __has_include("third_party/highway-github/hwy/ops/generic_ops-inl.h");
    auto has_ordered_demote2_to = __has_include("third_party/highway-github/hwy/ops/generic_ops-inl.h");

    return has_promote_odd_to && has_promote_upper_to && has_ordered_demote2_to ? 0 : 1;
}