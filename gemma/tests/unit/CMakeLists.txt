# Unit Tests for Enhanced Gemma.cpp Components

# Core unit tests (existing and new)
set(CORE_TEST_SOURCES
    test_sampling.cpp
    test_tokenization.cc
    test_model_loading.cc
    test_memory_management.cc
    test_error_handling.cc
)

# MCP unit tests
if(GEMMA_BUILD_MCP_SERVER)
    set(MCP_TEST_SOURCES
        mcp/test_mcp_server.cpp
        mcp/test_inference_handler.cpp
        mcp/test_model_manager.cpp
        mcp/test_mcp_protocol.cpp
    )
endif()

# Backend unit tests
if(GEMMA_BUILD_BACKENDS)
    set(BACKEND_TEST_SOURCES
        backends/test_backend_interface.cpp
        backends/test_backend_registry.cpp
        backends/test_memory_manager.cpp
        backends/test_kernel_dispatcher.cpp
    )
endif()

# Create test executable for core components
if(CORE_TEST_SOURCES)
    add_executable(test_gemma_core ${CORE_TEST_SOURCES})
    target_link_libraries(test_gemma_core PRIVATE
        gemma_test_utils
        libgemma
        GTest::gtest_main
    )
    set_property(TARGET test_gemma_core PROPERTY CXX_STANDARD 20)
    
    gtest_discover_tests(test_gemma_core
        PROPERTIES LABELS "unit;core"
    )
endif()

# Create test executable for MCP components
if(GEMMA_BUILD_MCP_SERVER AND MCP_TEST_SOURCES)
    add_executable(test_gemma_mcp ${MCP_TEST_SOURCES})
    target_link_libraries(test_gemma_mcp PRIVATE
        gemma_test_utils
        gemma_mcp_lib
        GTest::gtest_main
    )
    set_property(TARGET test_gemma_mcp PROPERTY CXX_STANDARD 20)
    
    gtest_discover_tests(test_gemma_mcp
        PROPERTIES LABELS "unit;mcp"
    )
endif()

# Create test executable for backend components
if(GEMMA_BUILD_BACKENDS AND BACKEND_TEST_SOURCES)
    add_executable(test_gemma_backends ${BACKEND_TEST_SOURCES})
    target_link_libraries(test_gemma_backends PRIVATE
        gemma_test_utils
        gemma_backends_common
        GTest::gtest_main
    )
    set_property(TARGET test_gemma_backends PROPERTY CXX_STANDARD 20)
    
    gtest_discover_tests(test_gemma_backends
        PROPERTIES LABELS "unit;backends"
    )
endif()

# Individual backend tests
if(GEMMA_BUILD_CUDA_BACKEND)
    add_executable(test_cuda_backend backends/test_cuda_backend.cpp)
    target_link_libraries(test_cuda_backend PRIVATE
        gemma_test_utils
        gemma_cuda_backend
        GTest::gtest_main
    )
    set_property(TARGET test_cuda_backend PROPERTY CXX_STANDARD 20)
    
    gtest_discover_tests(test_cuda_backend
        PROPERTIES LABELS "unit;backends;cuda"
    )
endif()

message(STATUS "Unit tests configured for available components")