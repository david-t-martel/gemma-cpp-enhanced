# CUDA Backend for Gemma.cpp
# Provides CUDA GPU acceleration for inference operations

cmake_minimum_required(VERSION 3.18)  # Required for CUDA support

# Enable CUDA language
enable_language(CUDA)

# Find required CUDA libraries
find_package(CUDAToolkit REQUIRED)

# Check for additional CUDA features
include(CheckLanguage)
check_language(CUDA)

# CUDA sources
set(CUDA_HEADERS
    cuda_backend.h
    cuda_kernels.h
    cuda_memory.h
    cuda_attention.h
    cuda_stream_manager.h
    cuda_kernel_launcher.h
)

set(CUDA_SOURCES
    cuda_backend.cpp
    cuda_kernels.cu
    cuda_attention.cu
)

# Optional source files (will be created if needed)
set(CUDA_OPTIONAL_SOURCES
    cuda_memory.cpp
    cuda_stream_manager.cpp
    cuda_kernel_launcher.cpp
)

# Check which optional sources exist and add them
foreach(SOURCE ${CUDA_OPTIONAL_SOURCES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
        list(APPEND CUDA_SOURCES ${SOURCE})
        message(STATUS "Adding optional CUDA source: ${SOURCE}")
    else()
        message(STATUS "Optional CUDA source not found: ${SOURCE}")
    endif()
endforeach()

# Create CUDA backend library
add_library(gemma_cuda_backend ${CUDA_SOURCES} ${CUDA_HEADERS})

# Set language standards
set_property(TARGET gemma_cuda_backend PROPERTY CUDA_STANDARD 17)
set_property(TARGET gemma_cuda_backend PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET gemma_cuda_backend PROPERTY CXX_STANDARD 20)
set_property(TARGET gemma_cuda_backend PROPERTY CXX_STANDARD_REQUIRED ON)

# CUDA architecture configuration with auto-detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect GPU architectures if possible
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAPS
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(GPU_COMPUTE_CAPS)
        string(REPLACE "\n" ";" GPU_COMPUTE_CAPS_LIST ${GPU_COMPUTE_CAPS})
        list(REMOVE_DUPLICATES GPU_COMPUTE_CAPS_LIST)
        string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${GPU_COMPUTE_CAPS_LIST}")
        message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Fallback to common modern architectures
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
        message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Advanced CUDA compiler options
target_compile_options(gemma_cuda_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -use_fast_math
        --ptxas-options=-v,--warn-on-spills
        --generate-line-info
        --source-in-ptx
        --keep
        -maxrregcount=128
    >
)

# Optimization for specific architectures
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(ARCH GREATER_EQUAL 80)
        # Ampere and newer: enable Tensor Core operations
        target_compile_options(gemma_cuda_backend PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_${ARCH}>
            $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_${ARCH}>
        )
    endif()
endforeach()

# Include directories
target_include_directories(gemma_cuda_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find required CUDA libraries
find_package(CUDAToolkit REQUIRED COMPONENTS cublas curand cudnn)

# Core CUDA libraries
target_link_libraries(gemma_cuda_backend
    PUBLIC
        gemma_backends_common
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        CUDA::cufft
        CUDA::cusparse
)

# Optional libraries with fallback
find_package(CUDAToolkit QUIET COMPONENTS cudnn)
if(CUDAToolkit_cudnn_FOUND)
    target_link_libraries(gemma_cuda_backend PRIVATE CUDA::cudnn)
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_CUDNN)
    message(STATUS "cuDNN found and enabled")
else()
    # Try to find cuDNN manually
    find_library(CUDNN_LIBRARY
        NAMES cudnn libcudnn
        HINTS
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
    )

    if(CUDNN_LIBRARY)
        target_link_libraries(gemma_cuda_backend PRIVATE ${CUDNN_LIBRARY})
        target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_CUDNN)
        message(STATUS "cuDNN found manually: ${CUDNN_LIBRARY}")
    else()
        message(STATUS "cuDNN not found, using basic CUDA operations")
    endif()
endif()

# NCCL for multi-GPU communication (optional)
find_package(CUDAToolkit QUIET COMPONENTS nccl)
if(CUDAToolkit_nccl_FOUND)
    target_link_libraries(gemma_cuda_backend PRIVATE CUDA::nccl)
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_NCCL)
    message(STATUS "NCCL found and enabled for multi-GPU support")
endif()

# CUB (CUDA Unbound) - usually included with CUDA toolkit
target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_CUB)

# Cooperative Groups support
target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_COOPERATIVE_GROUPS)

# Memory debugging (optional)
option(GEMMA_ENABLE_CUDA_MEMORY_DEBUG "Enable CUDA memory debugging" OFF)
if(GEMMA_ENABLE_CUDA_MEMORY_DEBUG)
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_MEMORY_TRACKING)
    message(STATUS "CUDA memory debugging enabled")
endif()

# Performance profiling (optional)
option(GEMMA_ENABLE_CUDA_PROFILING "Enable CUDA kernel profiling" OFF)
if(GEMMA_ENABLE_CUDA_PROFILING)
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_PROFILING)
    message(STATUS "CUDA profiling enabled")
endif()

# Compiler-specific settings
if(MSVC)
    target_compile_options(gemma_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /wd4996 /wd4267 /wd4244>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-wd4996,-wd4267,-wd4244>
    )
    # Disable specific MSVC warnings that are common with CUDA
    target_compile_definitions(gemma_cuda_backend PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
else()
    target_compile_options(gemma_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    )
endif()

# Build type specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(gemma_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo -DDEBUG>
    )
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_DEBUG_MODE)
    message(STATUS "CUDA debug mode enabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(gemma_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
    )
    message(STATUS "CUDA release mode enabled")
endif()

# Set CUDA properties for optimization
set_target_properties(gemma_cuda_backend PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
)

# Enable CUDA graph support if available (CUDA 10.0+)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0")
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_CUDA_GRAPHS)
    message(STATUS "CUDA Graphs support enabled")
endif()

# Enable memory pools if available (CUDA 11.2+)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.2")
    target_compile_definitions(gemma_cuda_backend PRIVATE GEMMA_ENABLE_MEMORY_POOLS)
    message(STATUS "CUDA Memory Pools support enabled")
endif()

# Installation
install(TARGETS gemma_cuda_backend
    EXPORT gemma-cuda-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CUDA_HEADERS}
    DESTINATION include/gemma/backends/cuda
)

# Export configuration
install(EXPORT gemma-cuda-targets
    FILE gemma-cuda-targets.cmake
    NAMESPACE Gemma::
    DESTINATION lib/cmake/gemma
)

# Configuration summary
message(STATUS "=== CUDA Backend Configuration Summary ===")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "cuDNN: ${CUDAToolkit_cudnn_FOUND}")
message(STATUS "NCCL: ${CUDAToolkit_nccl_FOUND}")
message(STATUS "Memory Debug: ${GEMMA_ENABLE_CUDA_MEMORY_DEBUG}")
message(STATUS "Profiling: ${GEMMA_ENABLE_CUDA_PROFILING}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===========================================")