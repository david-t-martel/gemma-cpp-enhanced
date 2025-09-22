# Enhanced Gemma.cpp Project Root CMakeLists.txt
# This file orchestrates the build of the enhanced Gemma.cpp with hardware acceleration backends

cmake_minimum_required(VERSION 3.20)

# Set policy versions to handle compatibility with dependencies
# CMP0111: An imported target with INTERFACE_SOURCES may not be built by the generator
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

# CMP0091: MSVC runtime library flags are selected by an abstraction
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# Set minimum policy version for all subdirectories to handle old dependencies
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # option() honors normal variables
set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)  # Include file check macros honor CMAKE_REQUIRED_LIBRARIES

project(gemma_enhanced
    VERSION 1.0.0
    DESCRIPTION "Enhanced Gemma.cpp with MCP server and hardware acceleration backends"
    LANGUAGES CXX C
)

# Set global C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options for new components
option(GEMMA_BUILD_MCP_SERVER "Build MCP server component" ON)
option(GEMMA_BUILD_BACKENDS "Build hardware acceleration backends" ON)

# Hardware backend options with automatic detection
option(GEMMA_BUILD_SYCL_BACKEND "Build SYCL/Intel oneAPI backend" OFF)
option(GEMMA_BUILD_CUDA_BACKEND "Build CUDA acceleration backend" OFF)
option(GEMMA_BUILD_OPENCL_BACKEND "Build OpenCL acceleration backend" OFF)
option(GEMMA_BUILD_VULKAN_BACKEND "Build Vulkan acceleration backend" OFF)
option(GEMMA_BUILD_METAL_BACKEND "Build Metal acceleration backend" OFF)

# Test and documentation options
option(GEMMA_BUILD_ENHANCED_TESTS "Build enhanced test suite" ON)
option(GEMMA_BUILD_BACKEND_TESTS "Build backend-specific tests" ON)
option(GEMMA_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(GEMMA_BUILD_DOCS "Build documentation" OFF)

# Auto-detection options
option(GEMMA_AUTO_DETECT_BACKENDS "Automatically detect available backends" ON)

# Platform detection
if(WIN32)
    set(GEMMA_PLATFORM "Windows")
elseif(APPLE)
    set(GEMMA_PLATFORM "macOS")
elseif(UNIX)
    set(GEMMA_PLATFORM "Linux")
endif()

message(STATUS "Building Enhanced Gemma.cpp for ${GEMMA_PLATFORM}")
message(STATUS "MCP Server: ${GEMMA_BUILD_MCP_SERVER}")
message(STATUS "Hardware Backends: ${GEMMA_BUILD_BACKENDS}")

# Include directories for enhanced components
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Create cmake directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Auto-detect available backends if enabled
if(GEMMA_AUTO_DETECT_BACKENDS AND GEMMA_BUILD_BACKENDS)
    message(STATUS "Auto-detecting available hardware backends...")

    # Try to find oneAPI/SYCL
    find_package(IntelSYCL QUIET)
    if(IntelSYCL_FOUND OR EXISTS "$ENV{ONEAPI_ROOT}")
        message(STATUS "Intel oneAPI/SYCL detected")
        set(GEMMA_BUILD_SYCL_BACKEND ON CACHE BOOL "Build SYCL backend" FORCE)
    endif()

    # Try to find CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit detected (version ${CUDAToolkit_VERSION})")
        set(GEMMA_BUILD_CUDA_BACKEND ON CACHE BOOL "Build CUDA backend" FORCE)
    endif()

    # Try to find Vulkan
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan SDK detected (version ${Vulkan_VERSION})")
        set(GEMMA_BUILD_VULKAN_BACKEND ON CACHE BOOL "Build Vulkan backend" FORCE)
    endif()

    # Try to find OpenCL
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL detected")
        set(GEMMA_BUILD_OPENCL_BACKEND ON CACHE BOOL "Build OpenCL backend" FORCE)
    endif()

    # Metal is available on macOS by default
    if(APPLE)
        message(STATUS "Metal backend available on macOS")
        set(GEMMA_BUILD_METAL_BACKEND ON CACHE BOOL "Build Metal backend" FORCE)
    endif()
endif()

# Find required packages for enabled backends
if(GEMMA_BUILD_SYCL_BACKEND AND GEMMA_BUILD_BACKENDS)
    # Set oneAPI root hint for Windows
    if(WIN32 AND NOT DEFINED ENV{ONEAPI_ROOT})
        set(ENV{ONEAPI_ROOT} "C:/Program Files (x86)/Intel/oneAPI")
    endif()
    find_package(IntelSYCL)
    if(IntelSYCL_FOUND)
        add_definitions(-DGEMMA_ENABLE_SYCL)
        message(STATUS "Intel SYCL backend enabled")
    else()
        message(WARNING "Intel SYCL not found, disabling SYCL backend")
        set(GEMMA_BUILD_SYCL_BACKEND OFF)
    endif()
endif()

if(GEMMA_BUILD_CUDA_BACKEND AND GEMMA_BUILD_BACKENDS)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    add_definitions(-DGEMMA_ENABLE_CUDA)
endif()

if(GEMMA_BUILD_VULKAN_BACKEND AND GEMMA_BUILD_BACKENDS)
    find_package(Vulkan REQUIRED)
    add_definitions(-DGEMMA_ENABLE_VULKAN)
endif()

if(GEMMA_BUILD_OPENCL_BACKEND AND GEMMA_BUILD_BACKENDS)
    find_package(OpenCL REQUIRED)
    add_definitions(-DGEMMA_ENABLE_OPENCL)
endif()

if(GEMMA_BUILD_METAL_BACKEND AND GEMMA_BUILD_BACKENDS AND APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    add_definitions(-DGEMMA_ENABLE_METAL)
endif()

# Set policy minimum version to handle legacy dependencies like sentencepiece
# This is required for sentencepiece which has cmake_minimum_required(VERSION 3.1)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Add the original gemma.cpp as a subdirectory
add_subdirectory(gemma.cpp)

# Add MCP server component if enabled
if(GEMMA_BUILD_MCP_SERVER)
    message(STATUS "Adding MCP server component")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp)
        add_subdirectory(mcp)
    else()
        message(WARNING "MCP directory not found, skipping MCP server")
    endif()
endif()

# Add hardware acceleration backends if enabled
if(GEMMA_BUILD_BACKENDS)
    message(STATUS "Adding hardware acceleration backends")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/backends)
        add_subdirectory(backends)
    else()
        message(WARNING "Backends directory not found, skipping backends")
    endif()
endif()

# Add enhanced testing framework
if(GEMMA_BUILD_ENHANCED_TESTS)
    message(STATUS "Adding enhanced test suite")
    enable_testing()
    add_subdirectory(tests)
endif()

# Add documentation if enabled
if(GEMMA_BUILD_DOCS)
    message(STATUS "Adding documentation build")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs)
        add_subdirectory(docs)
    else()
        message(WARNING "Docs directory not found, skipping documentation")
    endif()
endif()

# Add tools directory (CLI and other utilities)
message(STATUS "Adding tools directory")
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    add_subdirectory(tools)
else()
    message(WARNING "Tools directory not found, skipping tools")
endif()

# Create convenience targets
add_custom_target(gemma_all
    DEPENDS gemma libgemma
    COMMENT "Build all core Gemma components"
)

if(GEMMA_BUILD_MCP_SERVER AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp)
    add_dependencies(gemma_all gemma_mcp_server)
endif()

if(GEMMA_BUILD_BACKENDS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/backends)
    add_dependencies(gemma_all gemma_backends)
endif()

# Add CLI tool to main target
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    add_dependencies(gemma_all tools)
endif()

# Backend-specific convenience targets
if(GEMMA_BUILD_SYCL_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_sycl COMMENT "Build with SYCL backend")
endif()

if(GEMMA_BUILD_CUDA_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_cuda COMMENT "Build with CUDA backend")
endif()

if(GEMMA_BUILD_VULKAN_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_vulkan COMMENT "Build with Vulkan backend")
endif()

if(GEMMA_BUILD_OPENCL_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_opencl COMMENT "Build with OpenCL backend")
endif()

if(GEMMA_BUILD_METAL_BACKEND AND GEMMA_BUILD_BACKENDS)
    add_custom_target(gemma_metal COMMENT "Build with Metal backend")
endif()

# Installation configuration
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install CACHE PATH "Installation directory")

# Configure compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /W4")
endif()

# Summary
message(STATUS "=== Enhanced Gemma.cpp Build Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Platform: ${GEMMA_PLATFORM}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  - Core Gemma.cpp: ON")
message(STATUS "  - MCP Server: ${GEMMA_BUILD_MCP_SERVER}")
message(STATUS "  - Hardware Backends: ${GEMMA_BUILD_BACKENDS}")
message(STATUS "  - Enhanced Tests: ${GEMMA_BUILD_ENHANCED_TESTS}")
message(STATUS "  - Backend Tests: ${GEMMA_BUILD_BACKEND_TESTS}")
message(STATUS "  - Benchmarks: ${GEMMA_BUILD_BENCHMARKS}")
message(STATUS "  - Documentation: ${GEMMA_BUILD_DOCS}")
message(STATUS "")
message(STATUS "Hardware Backends:")
message(STATUS "  - SYCL/Intel oneAPI: ${GEMMA_BUILD_SYCL_BACKEND}")
message(STATUS "  - CUDA: ${GEMMA_BUILD_CUDA_BACKEND}")
message(STATUS "  - Vulkan: ${GEMMA_BUILD_VULKAN_BACKEND}")
message(STATUS "  - OpenCL: ${GEMMA_BUILD_OPENCL_BACKEND}")
message(STATUS "  - Metal: ${GEMMA_BUILD_METAL_BACKEND}")
message(STATUS "===============================================")