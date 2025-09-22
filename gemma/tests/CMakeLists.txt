# Enhanced Testing Framework for Gemma.cpp
# Comprehensive test suite covering new MCP and backend components

cmake_minimum_required(VERSION 3.16)

# Enable testing
enable_testing()

# Find required testing libraries
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Find benchmark library
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
        EXCLUDE_FROM_ALL
    )
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# Common test utilities
add_library(gemma_test_utils
    common/test_utils.cpp
    common/test_utils.h
    common/mock_backends.cpp
    common/mock_backends.h
    fixtures/model_fixtures.cpp
    fixtures/model_fixtures.h
)

target_include_directories(gemma_test_utils PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
)

target_link_libraries(gemma_test_utils PUBLIC 
    libgemma
    GTest::gtest
    GTest::gmock
)

if(GEMMA_BUILD_MCP_SERVER)
    target_link_libraries(gemma_test_utils PUBLIC gemma_mcp_lib)
endif()

if(GEMMA_BUILD_BACKENDS)
    target_link_libraries(gemma_test_utils PUBLIC gemma_backends_common)
endif()

set_property(TARGET gemma_test_utils PROPERTY CXX_STANDARD 20)

# Test configuration
set(GEMMA_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Test data directory")
set(GEMMA_TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/output" CACHE PATH "Test output directory")

# Create test output directory
file(MAKE_DIRECTORY ${GEMMA_TEST_OUTPUT_DIR})

# Configure test data paths
target_compile_definitions(gemma_test_utils PUBLIC
    GEMMA_TEST_DATA_DIR="${GEMMA_TEST_DATA_DIR}"
    GEMMA_TEST_OUTPUT_DIR="${GEMMA_TEST_OUTPUT_DIR}"
)

# Unit tests
message(STATUS "Adding unit tests")
add_subdirectory(unit)

# Integration tests
message(STATUS "Adding integration tests")
add_subdirectory(integration)

# Performance tests
message(STATUS "Adding performance tests")
add_subdirectory(performance)

# Functional tests
message(STATUS "Adding functional tests")
add_subdirectory(functional)

# Custom test targets
add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "unit"
    COMMENT "Running unit tests"
    USES_TERMINAL
)

add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "integration"
    COMMENT "Running integration tests"
    USES_TERMINAL
)

add_custom_target(test_performance
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "performance"
    COMMENT "Running performance tests"
    USES_TERMINAL
)

add_custom_target(test_functional
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "functional"
    COMMENT "Running functional tests"
    USES_TERMINAL
)

add_custom_target(test_backends
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "backend"
    COMMENT "Running backend tests"
    USES_TERMINAL
)

add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running all tests"
    USES_TERMINAL
)

# Coverage target (if coverage tools are available)
find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)

if(GCOV_PATH AND LCOV_PATH AND GENHTML_PATH AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_target(coverage
        COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
        COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' --output-file coverage.info
        COMMAND ${LCOV_PATH} --list coverage.info
        COMMAND ${GENHTML_PATH} -o coverage coverage.info
        COMMENT "Generating coverage report"
        USES_TERMINAL
    )
endif()

# Test summary
message(STATUS "=== Test Configuration ===")
message(STATUS "Test Data Directory: ${GEMMA_TEST_DATA_DIR}")
message(STATUS "Test Output Directory: ${GEMMA_TEST_OUTPUT_DIR}")
message(STATUS "Unit Tests: ON")
message(STATUS "Integration Tests: ON")
message(STATUS "Performance Tests: ON")
if(GEMMA_BUILD_MCP_SERVER)
    message(STATUS "MCP Tests: ON")
endif()
if(GEMMA_BUILD_BACKENDS)
    message(STATUS "Backend Tests: ON")
endif()
message(STATUS "Functional Tests: ON")
message(STATUS "============================")