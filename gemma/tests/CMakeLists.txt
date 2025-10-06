# CMakeLists.txt for Gemma.cpp Enhanced Test Suite
# This comprehensive test configuration supports unit, integration, and performance testing
# with Google Test, Google Mock, and Google Benchmark frameworks.

cmake_minimum_required(VERSION 3.20)
project(GemmaTests VERSION 1.0.0 LANGUAGES CXX)

# Ensure C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    # Visual Studio specific options
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Enable parallel compilation
    add_compile_options(/MP)
else()
    # GCC/Clang options
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -fsanitize=address -fsanitize=undefined)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
endif()

# Test configuration options
option(GEMMA_BUILD_UNIT_TESTS "Build unit tests" ON)
option(GEMMA_BUILD_INTEGRATION_TESTS "Build integration tests" ON)
option(GEMMA_BUILD_PERFORMANCE_TESTS "Build performance tests" ON)
option(GEMMA_BUILD_BACKEND_TESTS "Build backend-specific tests" ON)
option(GEMMA_BUILD_FUNCTIONAL_TESTS "Build functional tests" ON)
option(GEMMA_BUILD_MCP_TESTS "Build MCP protocol tests" ON)
option(GEMMA_ENABLE_TEST_COVERAGE "Enable test coverage reporting" OFF)
option(GEMMA_ENABLE_MEMORY_TESTING "Enable memory testing tools" OFF)
option(GEMMA_PRINT_BACKEND_INCLUDE_HINTS "Print helpful backend include path hints" ON)

# Find required packages
find_package(Threads REQUIRED)

# Include FetchContent for dependency management
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
    GIT_SHALLOW    TRUE
)

# nlohmann/json for JSON handling
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

# Configure Google Test
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# Configure Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

# Make dependencies available
FetchContent_MakeAvailable(googletest benchmark json)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../gemma)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

# Test utilities library
add_library(gemma_test_utils STATIC
    utils/test_helpers.h
    utils/mock_backend.h
)

target_link_libraries(gemma_test_utils
    PUBLIC
        gtest
        gmock
        nlohmann_json::nlohmann_json
        Threads::Threads
)

target_include_directories(gemma_test_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Function to create test executable
function(create_test_executable TEST_NAME TEST_FILES)
    add_executable(${TEST_NAME} ${TEST_FILES})
    
    target_link_libraries(${TEST_NAME}
        PRIVATE
            gemma_test_utils
            gtest_main
            gmock_main
            nlohmann_json::nlohmann_json
            Threads::Threads
    )
    
    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    
    # Platform-specific settings
    if(WIN32)
        set_target_properties(${TEST_NAME} PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
    endif()
endfunction()

# Function to create benchmark executable
function(create_benchmark_executable BENCHMARK_NAME BENCHMARK_FILES)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILES})
    
    target_link_libraries(${BENCHMARK_NAME}
        PRIVATE
            gemma_test_utils
            benchmark::benchmark
            benchmark::benchmark_main
            gtest
            gmock
            nlohmann_json::nlohmann_json
            Threads::Threads
    )
    
    set_target_properties(${BENCHMARK_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

# Enable testing
enable_testing()

# Unit Tests
if(GEMMA_BUILD_UNIT_TESTS)
    message(STATUS "Building unit tests")
    
    create_test_executable(test_session_manager
        unit/test_session_manager.cpp
    )
    
    create_test_executable(test_intel_backend
        unit/test_intel_backend.cpp
    )
    
    create_test_executable(test_mcp_protocol
        unit/test_mcp_protocol.cpp
    )
    
    create_test_executable(test_cli_commands
        unit/test_cli_commands.cpp
    )
    
    # Create unit test suite target
    add_custom_target(unit_tests
        DEPENDS
            test_session_manager
            test_intel_backend
            test_mcp_protocol
            test_cli_commands
    )
endif()

# Integration Tests
if(GEMMA_BUILD_INTEGRATION_TESTS)
    message(STATUS "Building integration tests")
    
    create_test_executable(test_end_to_end
        integration/test_end_to_end.cpp
    )

    # Newly extracted focused workflow tests
    create_test_executable(test_model_inference_workflow
        integration/test_model_inference_workflow.cpp
    )

    create_test_executable(test_backend_switching_workflow
        integration/test_backend_switching_workflow.cpp
    )

    create_test_executable(test_mcp_server_workflow
        integration/test_mcp_server_workflow.cpp
    )

    create_test_executable(test_batch_processing_workflow
        integration/test_batch_processing_workflow.cpp
    )

    create_test_executable(test_error_recovery_workflow
        integration/test_error_recovery_workflow.cpp
    )


    create_test_executable(test_system_metrics_workflow
        integration/test_system_metrics_workflow.cpp
    )

    create_test_executable(test_configuration_update_workflow
        integration/test_configuration_update_workflow.cpp
    )
    
    create_test_executable(test_backend_switching
        integration/test_backend_switching.cpp
    )
    
    create_test_executable(test_session_persistence
        integration/test_session_persistence.cpp
    )
    
    create_test_executable(test_mcp_server
        integration/test_mcp_server.cpp
    )
    
    # Create integration test suite target
    add_custom_target(integration_tests
        DEPENDS
            test_end_to_end
            test_model_inference_workflow
            test_backend_switching_workflow
            test_mcp_server_workflow
            test_batch_processing_workflow
            test_error_recovery_workflow
            test_system_metrics_workflow
            test_configuration_update_workflow
            test_backend_switching
            test_session_persistence
            test_mcp_server
    )
endif()

# MCP protocol validation / verification tests (schema-level, roundtrip, etc.)
if(GEMMA_BUILD_MCP_TESTS)
    message(STATUS "Building MCP validation tests")
    create_test_executable(test_mcp_validation
        mcp/test_mcp_validation.cpp
    )
    add_custom_target(mcp_tests DEPENDS test_mcp_validation)
endif()

# Performance Tests (Benchmarks)
if(GEMMA_BUILD_PERFORMANCE_TESTS)
    message(STATUS "Building performance tests")
    
    create_benchmark_executable(benchmark_backends
        performance/benchmark_backends.cpp
    )
    
    create_benchmark_executable(benchmark_session_ops
        performance/benchmark_session_ops.cpp
    )
    
    create_benchmark_executable(benchmark_inference
        performance/benchmark_inference.cpp
    )
    
    # Create performance test suite target
    add_custom_target(performance_tests
        DEPENDS
            benchmark_backends
            benchmark_session_ops
            benchmark_inference
    )
endif()

# Backend-specific tests (conditional compilation based on available SDKs)
if(GEMMA_BUILD_BACKEND_TESTS)
    message(STATUS "Building backend tests")
    
    # Check for Intel oneAPI/SYCL
    find_package(IntelSYCL QUIET)
    if(IntelSYCL_FOUND)
        message(STATUS "Intel SYCL found - enabling SYCL backend tests")
        add_compile_definitions(GEMMA_ENABLE_SYCL_TESTS)
    endif()
    
    # Check for CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found - enabling CUDA backend tests")
        add_compile_definitions(GEMMA_ENABLE_CUDA_TESTS)
    endif()
    
    # Check for Vulkan
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found - enabling Vulkan backend tests")
        add_compile_definitions(GEMMA_ENABLE_VULKAN_TESTS)
    endif()
endif()

# Test coverage (GCC/Clang only)
if(GEMMA_ENABLE_TEST_COVERAGE AND NOT MSVC)
    message(STATUS "Enabling test coverage")
    
    target_compile_options(gemma_test_utils PUBLIC --coverage)
    target_link_options(gemma_test_utils PUBLIC --coverage)
    
    # Add coverage target
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
        COMMAND lcov --capture --directory . --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info
        COMMAND lcov --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info '/usr/*' '*/test/*' '*/build/*' --output-file ${CMAKE_BINARY_DIR}/coverage/coverage_filtered.info
        COMMAND genhtml ${CMAKE_BINARY_DIR}/coverage/coverage_filtered.info --output-directory ${CMAKE_BINARY_DIR}/coverage/html
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating test coverage report"
    )
endif()

# Memory testing with Valgrind (Linux only)
if(GEMMA_ENABLE_MEMORY_TESTING AND UNIX AND NOT APPLE)
    find_program(VALGRIND_EXECUTABLE valgrind)
    if(VALGRIND_EXECUTABLE)
        message(STATUS "Valgrind found - enabling memory tests")
        
        # Add memory test target
        add_custom_target(memory_tests
            COMMAND ${VALGRIND_EXECUTABLE} --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 
                    $<TARGET_FILE:test_session_manager>
            COMMAND ${VALGRIND_EXECUTABLE} --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 
                    $<TARGET_FILE:test_intel_backend>
            DEPENDS unit_tests
            COMMENT "Running memory tests with Valgrind"
        )
    endif()
endif()

# Custom test targets for convenience
add_custom_target(test_all
    DEPENDS
        $<$<BOOL:${GEMMA_BUILD_UNIT_TESTS}>:unit_tests>
        $<$<BOOL:${GEMMA_BUILD_INTEGRATION_TESTS}>:integration_tests>
    $<$<BOOL:${GEMMA_BUILD_MCP_TESTS}>:mcp_tests>
)

add_custom_target(test_quick
    DEPENDS
        $<$<BOOL:${GEMMA_BUILD_UNIT_TESTS}>:test_session_manager>
        $<$<BOOL:${GEMMA_BUILD_UNIT_TESTS}>:test_mcp_protocol>
        $<$<BOOL:${GEMMA_BUILD_MCP_TESTS}>:test_mcp_validation>
    COMMENT "Running quick test suite"
)

# Performance benchmark target
add_custom_target(test_benchmarks
    DEPENDS
        $<$<BOOL:${GEMMA_BUILD_PERFORMANCE_TESTS}>:performance_tests>
    COMMENT "Building all test benchmarks"
)

# Test execution helpers
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4 -R "test_.*"
    DEPENDS unit_tests
    COMMENT "Running unit tests"
)

add_custom_target(run_integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 2 -R "test_.*"
    DEPENDS integration_tests
    COMMENT "Running integration tests"
)

add_custom_target(run_benchmarks
    COMMAND ${CMAKE_COMMAND} -E echo "Running backend benchmarks..."
    COMMAND $<$<TARGET_EXISTS:benchmark_backends>:$<TARGET_FILE:benchmark_backends>>
    COMMAND ${CMAKE_COMMAND} -E echo "Running session operation benchmarks..."
    COMMAND $<$<TARGET_EXISTS:benchmark_session_ops>:$<TARGET_FILE:benchmark_session_ops>>
    COMMAND ${CMAKE_COMMAND} -E echo "Running inference benchmarks..."
    COMMAND $<$<TARGET_EXISTS:benchmark_inference>:$<TARGET_FILE:benchmark_inference>>
    DEPENDS benchmarks
    COMMENT "Running all benchmarks"
)

# Test data and model setup
add_custom_target(setup_test_environment
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test_data
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test_models
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test_data ${CMAKE_BINARY_DIR}/test_data
    COMMENT "Setting up test environment"
)

# IDE support - organize tests in folders
if(MSVC OR XCODE)
    # Unit tests folder
    set_target_properties(test_session_manager test_intel_backend test_mcp_protocol test_cli_commands
        PROPERTIES FOLDER "Tests/Unit")
    
    # Integration tests folder
    set_target_properties(test_end_to_end test_backend_switching test_session_persistence test_mcp_server
        PROPERTIES FOLDER "Tests/Integration")
    set_target_properties(test_model_inference_workflow test_backend_switching_workflow test_mcp_server_workflow
        PROPERTIES FOLDER "Tests/Integration/Workflows")
    set_target_properties(test_batch_processing_workflow test_error_recovery_workflow
        test_system_metrics_workflow test_configuration_update_workflow
        PROPERTIES FOLDER "Tests/Integration/Workflows")
    
    # Performance tests folder
    set_target_properties(benchmark_backends benchmark_session_ops benchmark_inference
        PROPERTIES FOLDER "Tests/Performance")
    
    # Utilities folder
    set_target_properties(gemma_test_utils
        PROPERTIES FOLDER "Tests/Utilities")
endif()

# Summary
message(STATUS "")
message(STATUS "Gemma Test Suite Configuration Summary:")
message(STATUS "  Unit Tests:        ${GEMMA_BUILD_UNIT_TESTS}")
message(STATUS "  Integration Tests: ${GEMMA_BUILD_INTEGRATION_TESTS}")
message(STATUS "  Performance Tests: ${GEMMA_BUILD_PERFORMANCE_TESTS}")
message(STATUS "  Backend Tests:     ${GEMMA_BUILD_BACKEND_TESTS}")
message(STATUS "  MCP Tests:         ${GEMMA_BUILD_MCP_TESTS}")
message(STATUS "  Test Coverage:     ${GEMMA_ENABLE_TEST_COVERAGE}")
message(STATUS "  Memory Testing:    ${GEMMA_ENABLE_MEMORY_TESTING}")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  cmake --build . --target test_all        # Build all tests")
message(STATUS "  cmake --build . --target unit_tests      # Build unit tests only")
message(STATUS "  cmake --build . --target benchmarks      # Build benchmarks")
message(STATUS "  ctest --output-on-failure                # Run all tests")
message(STATUS "  cmake --build . --target run_benchmarks  # Run benchmarks")
message(STATUS "")

# Install configuration (optional)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

install(TARGETS gemma_test_utils
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# CPack configuration for test distribution
set(CPACK_PACKAGE_NAME "GemmaTestSuite")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Comprehensive test suite for Gemma.cpp")
set(CPACK_GENERATOR "ZIP")
include(CPack)

# -----------------------------------------------------------------------------
# Test Labeling Section
# -----------------------------------------------------------------------------
# Provide stable labels for CI filtering. Only set labels for tests that exist.
function(_label_if_exists test_name labels)
    if(TARGET ${test_name})
        set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
    endif()
endfunction()

_label_if_exists(test_end_to_end                "integration;full;smoke")
_label_if_exists(test_model_inference_workflow  "integration;workflow;inference")
_label_if_exists(test_backend_switching_workflow "integration;workflow;backend")
_label_if_exists(test_mcp_server_workflow       "integration;workflow;mcp")
_label_if_exists(test_batch_processing_workflow "integration;workflow;batch")
_label_if_exists(test_error_recovery_workflow   "integration;workflow;error")
_label_if_exists(test_system_metrics_workflow   "integration;workflow;metrics")
_label_if_exists(test_configuration_update_workflow "integration;workflow;config")
_label_if_exists(test_backend_switching         "integration;legacy;backend")
_label_if_exists(test_session_persistence       "integration;session;persistence")
_label_if_exists(test_mcp_server                "integration;legacy;mcp")
_label_if_exists(test_session_manager           "unit;session")
_label_if_exists(test_intel_backend             "unit;backend;intel")
_label_if_exists(test_mcp_protocol              "unit;mcp;protocol")
_label_if_exists(test_cli_commands              "unit;cli")
_label_if_exists(test_mcp_validation            "mcp;validation;protocol")
_label_if_exists(session_split_smoke            "session;refactor;smoke")

# Provide a meta target to list labels (developer convenience)
add_custom_target(print_test_labels
    COMMAND ${CMAKE_CTEST_COMMAND} -N | ${CMAKE_COMMAND} -DCTEST_FILE:PATH=${CMAKE_BINARY_DIR}/CTestTestfile.cmake -P ${CMAKE_CURRENT_LIST_DIR}/../tools/PrintLabels.cmake
    COMMENT "(Placeholder) print labels - implement PrintLabels.cmake if needed"
)

# -----------------------------------------------------------------------------
# Backend include path hints for developers and LLM agents when things fail.
# -----------------------------------------------------------------------------
if(GEMMA_PRINT_BACKEND_INCLUDE_HINTS)
    message(STATUS "-- Backend Include Path Hints (enable/disable via GEMMA_PRINT_BACKEND_INCLUDE_HINTS) --")
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    else()
        message(STATUS "CUDA NOT FOUND. Typical Windows paths: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include ; Linux: /usr/local/cuda/include")
    endif()
    if(IntelSYCL_FOUND)
        message(STATUS "Intel SYCL include dirs: ${IntelSYCL_INCLUDE_DIRS}")
    else()
        message(STATUS "Intel SYCL NOT FOUND. oneAPI basekit typical root (Windows): C:/Program Files (x86)/Intel/oneAPI ; Linux: /opt/intel/oneapi")
    endif()
    if(Vulkan_FOUND)
        message(STATUS "Vulkan include dirs: ${Vulkan_INCLUDE_DIRS}")
    else()
        message(STATUS "Vulkan NOT FOUND. Windows SDK path example: C:/VulkanSDK/<ver>/Include ; Linux package: /usr/include/vulkan")
    endif()
endif()
