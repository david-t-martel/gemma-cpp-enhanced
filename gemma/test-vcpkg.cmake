# Test script to validate vcpkg integration
cmake_minimum_required(VERSION 3.20)

# Set up vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "C:/codedev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")
set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg triplet")

project(test_vcpkg_integration CXX)

set(CMAKE_CXX_STANDARD 20)

# Add module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(VcpkgHelper)
include(FindVcpkgPackages)

message(STATUS "=== Testing vcpkg package detection ===")

# Test Highway
find_package(hwy QUIET CONFIG)
if(TARGET hwy)
    message(STATUS "✓ Highway: Found via vcpkg")
else()
    message(STATUS "✗ Highway: Not found")
endif()

# Test nlohmann-json
find_package(nlohmann_json QUIET CONFIG)
if(TARGET nlohmann_json::nlohmann_json)
    message(STATUS "✓ nlohmann-json: Found via vcpkg")
else()
    message(STATUS "✗ nlohmann-json: Not found")
endif()

# Test Google Benchmark
find_package(benchmark QUIET CONFIG)
if(TARGET benchmark::benchmark)
    message(STATUS "✓ Google Benchmark: Found via vcpkg")
else()
    message(STATUS "✗ Google Benchmark: Not found")
endif()

# Test SentencePiece (manual detection)
find_vcpkg_library(SENTENCEPIECE_LIB sentencepiece)
find_vcpkg_include(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)

if(SENTENCEPIECE_LIB AND SENTENCEPIECE_INCLUDE_DIR)
    create_vcpkg_target(sentencepiece-static SENTENCEPIECE_LIB SENTENCEPIECE_INCLUDE_DIR)
    message(STATUS "✓ SentencePiece: Found via vcpkg (manual target creation)")
else()
    message(STATUS "✗ SentencePiece: Not found")
endif()

message(STATUS "=== vcpkg integration test complete ===")

# Create a simple test executable to validate linking
add_executable(test_vcpkg_simple test_simple.cpp)

if(TARGET hwy)
    target_link_libraries(test_vcpkg_simple PRIVATE hwy)
endif()

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(test_vcpkg_simple PRIVATE nlohmann_json::nlohmann_json)
endif()

if(TARGET sentencepiece-static)
    target_link_libraries(test_vcpkg_simple PRIVATE sentencepiece-static)
endif()