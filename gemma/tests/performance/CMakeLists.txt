# Performance Tests CMakeLists.txt
# Benchmarks and performance regression tests

# Helper function to add performance benchmarks
function(add_gemma_benchmark BENCHMARK_NAME SOURCE_FILE)
    add_executable(${BENCHMARK_NAME} ${SOURCE_FILE})

    # Set C++20 standard
    set_property(TARGET ${BENCHMARK_NAME} PROPERTY CXX_STANDARD 20)

    # Link dependencies
    target_link_libraries(${BENCHMARK_NAME} PRIVATE
        libgemma
        gemma_test_utils
        benchmark::benchmark
        benchmark::benchmark_main
        hwy
        hwy_contrib
        nlohmann_json::nlohmann_json
    )

    # Include directories
    target_include_directories(${BENCHMARK_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )

    # Add benchmark as a test
    add_test(NAME ${BENCHMARK_NAME} COMMAND ${BENCHMARK_NAME} --benchmark_min_time=0.1)

    # Set test labels and properties
    set_tests_properties(${BENCHMARK_NAME} PROPERTIES
        LABELS "performance;benchmark"
        TIMEOUT 600
    )

    # Set performance-optimized environment
    set_tests_properties(${BENCHMARK_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=4;HWY_COMPILE_ALL_ATTAINABLE=1"
    )
endfunction()

# Helper function for performance regression tests (using GTest)
function(add_gemma_perf_test TEST_NAME SOURCE_FILE)
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

    # Set test labels
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "performance;regression"
        TIMEOUT 300
    )

    # Performance environment
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=4;HWY_COMPILE_ALL_ATTAINABLE=1"
    )
endfunction()

# Main inference benchmarks
add_gemma_benchmark(benchmark_inference benchmark_inference.cpp)

# Set specific properties for inference benchmarks
set_tests_properties(benchmark_inference PROPERTIES
    LABELS "performance;benchmark;inference;core"
    TIMEOUT 900
)

# Custom benchmark targets
add_custom_target(benchmark_quick
    COMMAND benchmark_inference --benchmark_filter="BM_Softmax|BM_TopK" --benchmark_min_time=0.05
    DEPENDS benchmark_inference
    COMMENT "Running quick performance benchmarks"
    USES_TERMINAL
)

add_custom_target(benchmark_full
    COMMAND benchmark_inference --benchmark_min_time=0.5
    DEPENDS benchmark_inference
    COMMENT "Running full performance benchmarks"
    USES_TERMINAL
)

add_custom_target(benchmark_memory
    COMMAND benchmark_inference --benchmark_filter="BM_.*Memory.*" --benchmark_min_time=0.1
    DEPENDS benchmark_inference
    COMMENT "Running memory allocation benchmarks"
    USES_TERMINAL
)

# Create performance comparison baseline
add_custom_target(benchmark_baseline
    COMMAND benchmark_inference --benchmark_out=baseline_results.json --benchmark_out_format=json
    DEPENDS benchmark_inference
    COMMENT "Generating performance baseline"
    USES_TERMINAL
)

# Performance regression test
add_custom_target(benchmark_compare
    COMMAND benchmark_inference --benchmark_out=current_results.json --benchmark_out_format=json
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/compare_benchmarks.py baseline_results.json current_results.json
    DEPENDS benchmark_inference
    COMMENT "Comparing current performance to baseline"
    USES_TERMINAL
)

message(STATUS "Performance benchmarks configured successfully")