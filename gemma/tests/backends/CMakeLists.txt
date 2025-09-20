# tests/backends/CMakeLists.txt - Backend-specific test configuration
#
# This file configures backend-specific tests and benchmarks for the Gemma.cpp project.
# It handles conditional compilation based on available hardware backends.

cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Backend test configuration
set(BACKEND_TESTS_ENABLED FALSE)

# Create a common test utilities library
add_library(backend_test_utils STATIC
    ../common/test_utils.cpp
    ../common/test_config.cpp
)

target_include_directories(backend_test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gemma.cpp
)

target_link_libraries(backend_test_utils
    GTest::gtest
    GTest::gtest_main
)

# SYCL Backend Tests
if(GEMMA_BUILD_SYCL_BACKEND AND IntelSYCL_FOUND)
    add_executable(test_sycl test_sycl.cpp)

    target_link_libraries(test_sycl
        backend_test_utils
        Intel::SYCL
        GTest::gtest
        GTest::gtest_main
    )

    target_compile_definitions(test_sycl PRIVATE
        GEMMA_ENABLE_SYCL
    )

    # Set SYCL-specific compile flags
    target_compile_options(test_sycl PRIVATE
        -fsycl
    )

    # Add test to CTest
    add_test(NAME SYCLBackendTests COMMAND test_sycl)
    set_tests_properties(SYCLBackendTests PROPERTIES
        TIMEOUT 300
        LABELS "backend;sycl"
    )

    set(BACKEND_TESTS_ENABLED TRUE)
    message(STATUS "SYCL backend tests enabled")
endif()

# CUDA Backend Tests
if(GEMMA_BUILD_CUDA_BACKEND AND CUDAToolkit_FOUND)
    add_executable(test_cuda test_cuda.cpp)

    target_link_libraries(test_cuda
        backend_test_utils
        CUDA::cudart
        CUDA::cublas
        GTest::gtest
        GTest::gtest_main
    )

    target_compile_definitions(test_cuda PRIVATE
        GEMMA_ENABLE_CUDA
    )

    # Enable CUDA language for this target
    set_property(TARGET test_cuda PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_property(TARGET test_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Add test to CTest
    add_test(NAME CUDABackendTests COMMAND test_cuda)
    set_tests_properties(CUDABackendTests PROPERTIES
        TIMEOUT 300
        LABELS "backend;cuda;gpu"
    )

    set(BACKEND_TESTS_ENABLED TRUE)
    message(STATUS "CUDA backend tests enabled")
endif()

# Vulkan Backend Tests
if(GEMMA_BUILD_VULKAN_BACKEND AND Vulkan_FOUND)
    add_executable(test_vulkan test_vulkan.cpp)

    target_link_libraries(test_vulkan
        backend_test_utils
        Vulkan::Vulkan
        GTest::gtest
        GTest::gtest_main
    )

    target_compile_definitions(test_vulkan PRIVATE
        GEMMA_ENABLE_VULKAN
        VK_ENABLE_BETA_EXTENSIONS
    )

    # Add test to CTest
    add_test(NAME VulkanBackendTests COMMAND test_vulkan)
    set_tests_properties(VulkanBackendTests PROPERTIES
        TIMEOUT 300
        LABELS "backend;vulkan;gpu"
    )

    set(BACKEND_TESTS_ENABLED TRUE)
    message(STATUS "Vulkan backend tests enabled")
endif()

# OpenCL Backend Tests
if(GEMMA_BUILD_OPENCL_BACKEND AND OpenCL_FOUND)
    add_executable(test_opencl test_opencl.cpp)

    target_link_libraries(test_opencl
        backend_test_utils
        OpenCL::OpenCL
        GTest::gtest
        GTest::gtest_main
    )

    target_compile_definitions(test_opencl PRIVATE
        GEMMA_ENABLE_OPENCL
    )

    # Add test to CTest
    add_test(NAME OpenCLBackendTests COMMAND test_opencl)
    set_tests_properties(OpenCLBackendTests PROPERTIES
        TIMEOUT 300
        LABELS "backend;opencl;gpu"
    )

    set(BACKEND_TESTS_ENABLED TRUE)
    message(STATUS "OpenCL backend tests enabled")
endif()

# Metal Backend Tests (macOS only)
if(GEMMA_BUILD_METAL_BACKEND AND APPLE)
    add_executable(test_metal test_metal.mm)

    target_link_libraries(test_metal
        backend_test_utils
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        GTest::gtest
        GTest::gtest_main
    )

    target_compile_definitions(test_metal PRIVATE
        GEMMA_ENABLE_METAL
    )

    # Set Objective-C++ properties
    set_target_properties(test_metal PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # Add test to CTest
    add_test(NAME MetalBackendTests COMMAND test_metal)
    set_tests_properties(MetalBackendTests PROPERTIES
        TIMEOUT 300
        LABELS "backend;metal;gpu;macos"
    )

    set(BACKEND_TESTS_ENABLED TRUE)
    message(STATUS "Metal backend tests enabled")
endif()

# Backend Performance Benchmark
if(BACKEND_TESTS_ENABLED AND GEMMA_BUILD_BENCHMARKS)
    add_executable(benchmark_backends benchmark_backends.cpp)

    target_link_libraries(benchmark_backends
        backend_test_utils
        benchmark::benchmark
        GTest::gtest
    )

    # Add backend-specific libraries based on what's available
    if(GEMMA_BUILD_SYCL_BACKEND AND IntelSYCL_FOUND)
        target_link_libraries(benchmark_backends Intel::SYCL)
        target_compile_definitions(benchmark_backends PRIVATE GEMMA_ENABLE_SYCL)
        target_compile_options(benchmark_backends PRIVATE -fsycl)
    endif()

    if(GEMMA_BUILD_CUDA_BACKEND AND CUDAToolkit_FOUND)
        target_link_libraries(benchmark_backends CUDA::cudart CUDA::cublas)
        target_compile_definitions(benchmark_backends PRIVATE GEMMA_ENABLE_CUDA)
        set_property(TARGET benchmark_backends PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    endif()

    if(GEMMA_BUILD_VULKAN_BACKEND AND Vulkan_FOUND)
        target_link_libraries(benchmark_backends Vulkan::Vulkan)
        target_compile_definitions(benchmark_backends PRIVATE GEMMA_ENABLE_VULKAN)
    endif()

    if(GEMMA_BUILD_OPENCL_BACKEND AND OpenCL_FOUND)
        target_link_libraries(benchmark_backends OpenCL::OpenCL)
        target_compile_definitions(benchmark_backends PRIVATE GEMMA_ENABLE_OPENCL)
    endif()

    if(GEMMA_BUILD_METAL_BACKEND AND APPLE)
        target_link_libraries(benchmark_backends
            ${METAL_FRAMEWORK}
            ${METALKIT_FRAMEWORK}
            ${METALPERFORMANCESHADERS_FRAMEWORK}
        )
        target_compile_definitions(benchmark_backends PRIVATE GEMMA_ENABLE_METAL)
    endif()

    # Add benchmark to CTest
    add_test(NAME BackendBenchmarks COMMAND benchmark_backends --benchmark_format=csv)
    set_tests_properties(BackendBenchmarks PROPERTIES
        TIMEOUT 600
        LABELS "benchmark;performance"
    )

    message(STATUS "Backend performance benchmarks enabled")
endif()

# Create custom test targets for convenience
if(BACKEND_TESTS_ENABLED)
    # Target to run all backend tests
    add_custom_target(test_all_backends
        COMMAND ${CMAKE_CTEST_COMMAND} --label-regex "backend" --output-on-failure
        DEPENDS ${BACKEND_TEST_TARGETS}
        COMMENT "Running all backend tests"
    )

    # Target to run GPU-specific tests
    add_custom_target(test_gpu_backends
        COMMAND ${CMAKE_CTEST_COMMAND} --label-regex "gpu" --output-on-failure
        COMMENT "Running GPU backend tests"
    )

    # Target to run performance benchmarks
    if(GEMMA_BUILD_BENCHMARKS)
        add_custom_target(benchmark_all_backends
            COMMAND ${CMAKE_CTEST_COMMAND} --label-regex "benchmark" --output-on-failure
            COMMENT "Running backend performance benchmarks"
        )
    endif()

    message(STATUS "Backend test targets created:")
    message(STATUS "  - test_all_backends: Run all backend tests")
    message(STATUS "  - test_gpu_backends: Run GPU-specific tests")
    if(GEMMA_BUILD_BENCHMARKS)
        message(STATUS "  - benchmark_all_backends: Run performance benchmarks")
    endif()
else()
    message(WARNING "No backend tests enabled. Install backend SDKs to enable testing.")
endif()

# Test discovery and reporting
if(BACKEND_TESTS_ENABLED)
    # Configure CTest
    enable_testing()

    # Set test environment variables
    set_tests_properties(${BACKEND_TEST_TARGETS} PROPERTIES
        ENVIRONMENT "GTEST_OUTPUT=xml:${CMAKE_BINARY_DIR}/test_results/"
    )

    # Create test results directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test_results)

    # Custom test configuration
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in
        ${CMAKE_BINARY_DIR}/CTestCustom.cmake
        @ONLY
    )
endif()

# Installation rules for test executables
if(BACKEND_TESTS_ENABLED)
    install(TARGETS ${BACKEND_TEST_TARGETS}
        RUNTIME DESTINATION bin/tests
        COMPONENT tests
    )

    if(GEMMA_BUILD_BENCHMARKS)
        install(TARGETS benchmark_backends
            RUNTIME DESTINATION bin/benchmarks
            COMPONENT benchmarks
        )
    endif()
endif()

# Summary
message(STATUS "Backend tests configuration:")
message(STATUS "  SYCL tests: ${GEMMA_BUILD_SYCL_BACKEND}")
message(STATUS "  CUDA tests: ${GEMMA_BUILD_CUDA_BACKEND}")
message(STATUS "  Vulkan tests: ${GEMMA_BUILD_VULKAN_BACKEND}")
message(STATUS "  OpenCL tests: ${GEMMA_BUILD_OPENCL_BACKEND}")
message(STATUS "  Metal tests: ${GEMMA_BUILD_METAL_BACKEND}")
message(STATUS "  Benchmarks: ${GEMMA_BUILD_BENCHMARKS}")
message(STATUS "  Overall enabled: ${BACKEND_TESTS_ENABLED}")