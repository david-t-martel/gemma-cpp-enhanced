// Simple test file to validate vcpkg package linking
#include <iostream>

// Test includes for vcpkg packages
#ifdef HWY_HIGHWAY
#include "hwy/highway.h"
#endif

#ifdef JSON_NLOHMANN
#include <nlohmann/json.hpp>
#endif

#ifdef SENTENCEPIECE_PROCESSOR_H_
#include "sentencepiece_processor.h"
#endif

int main() {
    std::cout << "vcpkg integration test successful!" << std::endl;

#ifdef JSON_NLOHMANN
    nlohmann::json j = nlohmann::json::parse(R"({"test": "vcpkg"})");
    std::cout << "nlohmann-json: " << j.dump() << std::endl;
#endif

    return 0;
}