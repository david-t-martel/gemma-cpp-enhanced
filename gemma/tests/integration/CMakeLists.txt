# Integration Tests CMakeLists.txt
# Tests for component interaction and end-to-end functionality

# Helper function to add integration tests
function(add_gemma_integration_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    # Set C++20 standard
    set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 20)

    # Link dependencies
    target_link_libraries(${TEST_NAME} PRIVATE
        libgemma
        gemma_test_utils
        GTest::gtest_main
        GTest::gmock
        hwy
        hwy_contrib
        nlohmann_json::nlohmann_json
    )

    # Include directories
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )

    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # Set test labels and timeout
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "integration"
        TIMEOUT 120
    )

    # Set environment for consistent testing
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;HWY_COMPILE_ALL_ATTAINABLE=1"
    )
endfunction()

# Backend compatibility tests
add_gemma_integration_test(test_backends_integration test_backends.cpp)

# Model loading integration tests
add_gemma_integration_test(test_model_loading_integration test_model_loading.cpp)

# Inference pipeline tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_inference.cc)
    add_gemma_integration_test(test_inference_integration test_inference.cc)

    # Inference tests may need more memory and time
    set_tests_properties(test_inference_integration PROPERTIES
        TIMEOUT 300
        LABELS "integration;inference"
    )
endif()

# Set specific properties for backend tests
set_tests_properties(test_backends_integration PROPERTIES
    LABELS "integration;backends;simd"
    TIMEOUT 90
)

# Set specific properties for model loading tests
set_tests_properties(test_model_loading_integration PROPERTIES
    LABELS "integration;model_loading;io"
    TIMEOUT 150
)

# MCP integration tests (if MCP support is enabled)
if(GEMMA_BUILD_MCP_SERVER)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp)
        add_subdirectory(mcp)
    endif()
endif()

# Backend-specific integration tests
if(GEMMA_BUILD_BACKENDS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/backends)
        add_subdirectory(backends)
    endif()
endif()

message(STATUS "Integration tests configured successfully")