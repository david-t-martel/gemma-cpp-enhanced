# Hardware Acceleration Backends for Gemma.cpp
# Provides GPU and specialized hardware acceleration support

cmake_minimum_required(VERSION 3.16)

# Backend interface sources
set(BACKEND_COMMON_SOURCES
    backend_interface.cpp
    backend_interface.h
    backend_registry.cpp
    backend_registry.h
    backend_manager.cpp
    backend_manager.h
)

# Create common backend library
add_library(gemma_backends_common ${BACKEND_COMMON_SOURCES})
target_include_directories(gemma_backends_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemma_backends_common PUBLIC libgemma)
set_property(TARGET gemma_backends_common PROPERTY CXX_STANDARD 20)

# CUDA Backend
if(GEMMA_BUILD_CUDA_BACKEND)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        add_subdirectory(cuda)
        message(STATUS "CUDA backend enabled")
    else()
        message(WARNING "CUDA not found, disabling CUDA backend")
        set(GEMMA_BUILD_CUDA_BACKEND OFF)
    endif()
endif()

# OpenCL Backend
if(GEMMA_BUILD_OPENCL_BACKEND)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        add_subdirectory(opencl)
        message(STATUS "OpenCL backend enabled")
    else()
        message(WARNING "OpenCL not found, disabling OpenCL backend")
        set(GEMMA_BUILD_OPENCL_BACKEND OFF)
    endif()
endif()

# Vulkan Backend
if(GEMMA_BUILD_VULKAN_BACKEND)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        add_subdirectory(vulkan)
        message(STATUS "Vulkan backend enabled")
    else()
        message(WARNING "Vulkan not found, disabling Vulkan backend")
        set(GEMMA_BUILD_VULKAN_BACKEND OFF)
    endif()
endif()

# Metal Backend (macOS only)
if(GEMMA_BUILD_METAL_BACKEND AND APPLE)
    add_subdirectory(metal)
    message(STATUS "Metal backend enabled")
elseif(GEMMA_BUILD_METAL_BACKEND AND NOT APPLE)
    message(WARNING "Metal backend only available on macOS, disabling")
    set(GEMMA_BUILD_METAL_BACKEND OFF)
endif()

# Intel SYCL Backend
if(GEMMA_BUILD_SYCL_BACKEND)
    add_subdirectory(sycl)
    message(STATUS "SYCL backend configuration complete")
else()
    message(STATUS "SYCL backend disabled")
endif()

# Create unified backends library
add_library(gemma_backends INTERFACE)
target_link_libraries(gemma_backends INTERFACE gemma_backends_common)

# Link enabled backends
if(GEMMA_BUILD_CUDA_BACKEND AND CUDA_FOUND)
    target_link_libraries(gemma_backends INTERFACE gemma_cuda_backend)
endif()

if(GEMMA_BUILD_OPENCL_BACKEND AND OpenCL_FOUND)
    target_link_libraries(gemma_backends INTERFACE gemma_opencl_backend)
endif()

if(GEMMA_BUILD_VULKAN_BACKEND AND Vulkan_FOUND)
    target_link_libraries(gemma_backends INTERFACE gemma_vulkan_backend)
endif()

if(GEMMA_BUILD_METAL_BACKEND AND APPLE)
    target_link_libraries(gemma_backends INTERFACE gemma_metal_backend)
endif()

if(GEMMA_BUILD_SYCL_BACKEND AND GEMMA_SYCL_BACKEND_AVAILABLE)
    target_link_libraries(gemma_backends INTERFACE gemma_sycl_backend)
endif()

# Installation
install(TARGETS gemma_backends_common
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    backend_interface.h
    backend_registry.h
    memory_manager.h
    kernel_dispatcher.h
    DESTINATION include/gemma/backends
)

# Status summary
message(STATUS "Backend configuration summary:")
message(STATUS "  - CUDA: ${GEMMA_BUILD_CUDA_BACKEND}")
message(STATUS "  - OpenCL: ${GEMMA_BUILD_OPENCL_BACKEND}")
message(STATUS "  - Vulkan: ${GEMMA_BUILD_VULKAN_BACKEND}")
message(STATUS "  - Metal: ${GEMMA_BUILD_METAL_BACKEND}")
message(STATUS "  - Intel SYCL: ${GEMMA_SYCL_BACKEND_AVAILABLE}")