# Functional Tests for Gemma.cpp Backend Integration
# These tests verify end-to-end functionality across all backend combinations

cmake_minimum_required(VERSION 3.16)

# Backend integration tests
if(GEMMA_BUILD_BACKENDS)
    message(STATUS "Configuring backend functional tests")

    # Backend integration test
    add_executable(test_backend_integration
        test_backend_integration.cpp
    )

    target_link_libraries(test_backend_integration
        PRIVATE
            gemma_test_utils
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
    )

    # Link backend libraries if available
    if(GEMMA_BUILD_BACKENDS)
        target_link_libraries(test_backend_integration PRIVATE gemma_backends_common)
        target_include_directories(test_backend_integration PRIVATE
            ${CMAKE_SOURCE_DIR}/backends
        )
    endif()

    set_property(TARGET test_backend_integration PROPERTY CXX_STANDARD 20)

    # Add test with appropriate labels
    add_test(
        NAME BackendIntegration
        COMMAND test_backend_integration
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_property(TEST BackendIntegration PROPERTY LABELS "functional;integration;backend")

    # Inference comparison tests
    add_executable(test_inference_backends
        test_inference_backends.cpp
    )

    target_link_libraries(test_inference_backends
        PRIVATE
            gemma_test_utils
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
    )

    if(GEMMA_BUILD_BACKENDS)
        target_link_libraries(test_inference_backends PRIVATE gemma_backends_common)
        target_include_directories(test_inference_backends PRIVATE
            ${CMAKE_SOURCE_DIR}/backends
        )
    endif()

    set_property(TARGET test_inference_backends PROPERTY CXX_STANDARD 20)

    add_test(
        NAME InferenceBackends
        COMMAND test_inference_backends
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_property(TEST InferenceBackends PROPERTY LABELS "functional;inference;backend")

    # Backend-specific tests for each enabled backend
    set(BACKEND_TEST_TARGETS "")

    if(GEMMA_BUILD_CUDA_BACKEND)
        add_executable(test_cuda_backend_specific
            test_backend_integration.cpp
        )

        target_link_libraries(test_cuda_backend_specific
            PRIVATE
                gemma_test_utils
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )

        if(GEMMA_BUILD_BACKENDS)
            target_link_libraries(test_cuda_backend_specific PRIVATE gemma_backends_common)
            if(TARGET gemma_cuda_backend)
                target_link_libraries(test_cuda_backend_specific PRIVATE gemma_cuda_backend)
            endif()
        endif()

        target_compile_definitions(test_cuda_backend_specific PRIVATE
            GEMMA_ENABLE_CUDA
            GEMMA_TEST_CUDA_ONLY
        )

        set_property(TARGET test_cuda_backend_specific PROPERTY CXX_STANDARD 20)

        add_test(
            NAME CUDABackendSpecific
            COMMAND test_cuda_backend_specific
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_property(TEST CUDABackendSpecific PROPERTY LABELS "functional;cuda;backend")
        list(APPEND BACKEND_TEST_TARGETS test_cuda_backend_specific)
    endif()

    if(GEMMA_BUILD_SYCL_BACKEND)
        add_executable(test_sycl_backend_specific
            test_backend_integration.cpp
        )

        target_link_libraries(test_sycl_backend_specific
            PRIVATE
                gemma_test_utils
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )

        if(GEMMA_BUILD_BACKENDS)
            target_link_libraries(test_sycl_backend_specific PRIVATE gemma_backends_common)
            if(TARGET gemma_sycl_backend)
                target_link_libraries(test_sycl_backend_specific PRIVATE gemma_sycl_backend)
            endif()
        endif()

        target_compile_definitions(test_sycl_backend_specific PRIVATE
            GEMMA_ENABLE_SYCL
            GEMMA_TEST_SYCL_ONLY
        )

        set_property(TARGET test_sycl_backend_specific PROPERTY CXX_STANDARD 20)

        add_test(
            NAME SYCLBackendSpecific
            COMMAND test_sycl_backend_specific
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_property(TEST SYCLBackendSpecific PROPERTY LABELS "functional;sycl;backend")
        list(APPEND BACKEND_TEST_TARGETS test_sycl_backend_specific)
    endif()

    if(GEMMA_BUILD_VULKAN_BACKEND)
        add_executable(test_vulkan_backend_specific
            test_backend_integration.cpp
        )

        target_link_libraries(test_vulkan_backend_specific
            PRIVATE
                gemma_test_utils
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )

        if(GEMMA_BUILD_BACKENDS)
            target_link_libraries(test_vulkan_backend_specific PRIVATE gemma_backends_common)
            if(TARGET gemma_vulkan_backend)
                target_link_libraries(test_vulkan_backend_specific PRIVATE gemma_vulkan_backend)
            endif()
        endif()

        target_compile_definitions(test_vulkan_backend_specific PRIVATE
            GEMMA_ENABLE_VULKAN
            GEMMA_TEST_VULKAN_ONLY
        )

        set_property(TARGET test_vulkan_backend_specific PROPERTY CXX_STANDARD 20)

        add_test(
            NAME VulkanBackendSpecific
            COMMAND test_vulkan_backend_specific
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_property(TEST VulkanBackendSpecific PROPERTY LABELS "functional;vulkan;backend")
        list(APPEND BACKEND_TEST_TARGETS test_vulkan_backend_specific)
    endif()

    if(GEMMA_BUILD_OPENCL_BACKEND)
        add_executable(test_opencl_backend_specific
            test_backend_integration.cpp
        )

        target_link_libraries(test_opencl_backend_specific
            PRIVATE
                gemma_test_utils
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )

        if(GEMMA_BUILD_BACKENDS)
            target_link_libraries(test_opencl_backend_specific PRIVATE gemma_backends_common)
            if(TARGET gemma_opencl_backend)
                target_link_libraries(test_opencl_backend_specific PRIVATE gemma_opencl_backend)
            endif()
        endif()

        target_compile_definitions(test_opencl_backend_specific PRIVATE
            GEMMA_ENABLE_OPENCL
            GEMMA_TEST_OPENCL_ONLY
        )

        set_property(TARGET test_opencl_backend_specific PROPERTY CXX_STANDARD 20)

        add_test(
            NAME OpenCLBackendSpecific
            COMMAND test_opencl_backend_specific
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_property(TEST OpenCLBackendSpecific PROPERTY LABELS "functional;opencl;backend")
        list(APPEND BACKEND_TEST_TARGETS test_opencl_backend_specific)
    endif()

    if(GEMMA_BUILD_METAL_BACKEND AND APPLE)
        add_executable(test_metal_backend_specific
            test_backend_integration.cpp
        )

        target_link_libraries(test_metal_backend_specific
            PRIVATE
                gemma_test_utils
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )

        if(GEMMA_BUILD_BACKENDS)
            target_link_libraries(test_metal_backend_specific PRIVATE gemma_backends_common)
            if(TARGET gemma_metal_backend)
                target_link_libraries(test_metal_backend_specific PRIVATE gemma_metal_backend)
            endif()
        endif()

        target_compile_definitions(test_metal_backend_specific PRIVATE
            GEMMA_ENABLE_METAL
            GEMMA_TEST_METAL_ONLY
        )

        set_property(TARGET test_metal_backend_specific PROPERTY CXX_STANDARD 20)

        add_test(
            NAME MetalBackendSpecific
            COMMAND test_metal_backend_specific
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_property(TEST MetalBackendSpecific PROPERTY LABELS "functional;metal;backend")
        list(APPEND BACKEND_TEST_TARGETS test_metal_backend_specific)
    endif()

    # Create convenience targets for backend-specific tests
    if(BACKEND_TEST_TARGETS)
        add_custom_target(test_backend_specific
            DEPENDS ${BACKEND_TEST_TARGETS}
            COMMENT "Build all backend-specific tests"
        )

        add_custom_target(run_backend_specific
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "backend"
            COMMENT "Run all backend-specific tests"
            USES_TERMINAL
        )
    endif()

    # Cross-backend comparison tests
    if(BACKEND_TEST_TARGETS)
        add_executable(test_cross_backend_comparison
            test_inference_backends.cpp
        )

        target_link_libraries(test_cross_backend_comparison
            PRIVATE
                gemma_test_utils
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )

        if(GEMMA_BUILD_BACKENDS)
            target_link_libraries(test_cross_backend_comparison PRIVATE gemma_backends_common)
        endif()

        target_compile_definitions(test_cross_backend_comparison PRIVATE
            GEMMA_TEST_CROSS_BACKEND_COMPARISON
        )

        set_property(TARGET test_cross_backend_comparison PROPERTY CXX_STANDARD 20)

        add_test(
            NAME CrossBackendComparison
            COMMAND test_cross_backend_comparison
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_property(TEST CrossBackendComparison PROPERTY LABELS "functional;comparison;backend")
    endif()

else()
    message(STATUS "Backend tests disabled - GEMMA_BUILD_BACKENDS is OFF")
endif()

# CPU-only functional tests (always available)
add_executable(test_cpu_functionality
    test_backend_integration.cpp
)

target_link_libraries(test_cpu_functionality
    PRIVATE
        gemma_test_utils
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

target_compile_definitions(test_cpu_functionality PRIVATE
    GEMMA_TEST_CPU_ONLY
)

set_property(TARGET test_cpu_functionality PROPERTY CXX_STANDARD 20)

add_test(
    NAME CPUFunctionality
    COMMAND test_cpu_functionality
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_property(TEST CPUFunctionality PROPERTY LABELS "functional;cpu")

# Test configuration summary
message(STATUS "=== Functional Test Configuration ===")
message(STATUS "Backend Integration Tests: ${GEMMA_BUILD_BACKENDS}")
if(GEMMA_BUILD_BACKENDS)
    message(STATUS "CUDA Tests: ${GEMMA_BUILD_CUDA_BACKEND}")
    message(STATUS "SYCL Tests: ${GEMMA_BUILD_SYCL_BACKEND}")
    message(STATUS "Vulkan Tests: ${GEMMA_BUILD_VULKAN_BACKEND}")
    message(STATUS "OpenCL Tests: ${GEMMA_BUILD_OPENCL_BACKEND}")
    message(STATUS "Metal Tests: ${GEMMA_BUILD_METAL_BACKEND}")
endif()
message(STATUS "CPU Functionality Tests: ON")
message(STATUS "======================================")