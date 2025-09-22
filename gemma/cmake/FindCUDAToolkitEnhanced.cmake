# FindCUDAToolkitEnhanced.cmake
# Enhanced CUDA detection with additional libraries and configurations
#
# This module extends the standard FindCUDAToolkit with additional
# libraries and utilities commonly needed for AI/ML workloads
#
# This module sets the following variables:
#
# CUDAToolkitEnhanced_FOUND      - True if CUDA Toolkit is found
# CUDAToolkit_VERSION            - Version of CUDA Toolkit
# CUDAToolkit_INCLUDE_DIRS       - Include directories
# CUDAToolkit_LIBRARY_DIR        - Library directory
# CUDAToolkit_BIN_DIR            - Binary directory
# CUDAToolkit_NVCC_EXECUTABLE    - Path to nvcc compiler
# CUDAToolkit_HAS_CUTLASS        - True if CUTLASS is available
# CUDAToolkit_HAS_CUDNN          - True if cuDNN is available
# CUDAToolkit_HAS_CUBLAS         - True if cuBLAS is available
# CUDAToolkit_HAS_CUFFT          - True if cuFFT is available
# CUDAToolkit_HAS_CUSPARSE       - True if cuSPARSE is available
#
# This module also creates enhanced targets for common libraries

cmake_minimum_required(VERSION 3.20)

# First, try to find the standard CUDA toolkit
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    set(CUDAToolkitEnhanced_FOUND TRUE)

    # Get CUDA compute capabilities
    if(CMAKE_CUDA_COMPILER_LOADED OR CUDAToolkit_NVCC_EXECUTABLE)
        # Common GPU architectures
        set(CUDA_ARCHITECTURES_COMMON "52;60;61;70;75;80;86;89;90")

        # Detect GPU if possible
        if(CUDAToolkit_NVCC_EXECUTABLE)
            execute_process(
                COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --help
                OUTPUT_VARIABLE NVCC_HELP_OUTPUT
                ERROR_QUIET
            )
        endif()
    endif()

    # Enhanced library detection
    set(CUDA_ENHANCED_LIBRARIES
        cublas
        cublasLt
        cudnn
        cufft
        curand
        cusparse
        cusolver
        cutlass
        nccl
        nvjpeg
        nvml
    )

    # Check for each enhanced library
    foreach(lib ${CUDA_ENHANCED_LIBRARIES})
        string(TOUPPER ${lib} LIB_UPPER)

        if(TARGET CUDA::${lib})
            set(CUDAToolkit_HAS_${LIB_UPPER} TRUE)
            message(STATUS "Found CUDA library: ${lib}")
        else()
            set(CUDAToolkit_HAS_${LIB_UPPER} FALSE)
        endif()
    endforeach()

    # Special handling for cuDNN (often installed separately)
    if(NOT CUDAToolkit_HAS_CUDNN)
        find_path(CUDNN_INCLUDE_DIR
            NAMES cudnn.h
            PATHS
                ${CUDAToolkit_INCLUDE_DIRS}
                /usr/local/cuda/include
                /usr/include
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/include"
                "C:/tools/cuda/include"
                $ENV{CUDNN_ROOT}/include
                $ENV{CUDA_PATH}/include
        )

        find_library(CUDNN_LIBRARY
            NAMES cudnn libcudnn
            PATHS
                ${CUDAToolkit_LIBRARY_DIR}
                /usr/local/cuda/lib64
                /usr/lib/x86_64-linux-gnu
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/lib/x64"
                "C:/tools/cuda/lib/x64"
                $ENV{CUDNN_ROOT}/lib
                $ENV{CUDA_PATH}/lib/x64
        )

        if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
            set(CUDAToolkit_HAS_CUDNN TRUE)
            message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")

            # Create cuDNN target if not exists
            if(NOT TARGET CUDA::cudnn)
                add_library(CUDA::cudnn UNKNOWN IMPORTED)
                set_target_properties(CUDA::cudnn PROPERTIES
                    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
                )
            endif()
        endif()
    endif()

    # Special handling for CUTLASS (header-only library)
    find_path(CUTLASS_INCLUDE_DIR
        NAMES cutlass/cutlass.h
        PATHS
            ${CUDAToolkit_INCLUDE_DIRS}
            /usr/local/cuda/include
            $ENV{CUTLASS_ROOT}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    )

    if(CUTLASS_INCLUDE_DIR)
        set(CUDAToolkit_HAS_CUTLASS TRUE)
        message(STATUS "Found CUTLASS: ${CUTLASS_INCLUDE_DIR}")

        if(NOT TARGET CUDA::cutlass)
            add_library(CUDA::cutlass INTERFACE IMPORTED)
            set_target_properties(CUDA::cutlass PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${CUTLASS_INCLUDE_DIR}"
                INTERFACE_COMPILE_FEATURES cxx_std_17
            )
        endif()
    endif()

    # Enhanced compilation flags
    set(CUDA_ENHANCED_COMPILE_FLAGS
        "--extended-lambda"
        "--expt-relaxed-constexpr"
        "--use_fast_math"
    )

    # Architecture-specific optimizations
    if(CMAKE_CUDA_ARCHITECTURES)
        list(APPEND CUDA_ENHANCED_COMPILE_FLAGS
            "--generate-code=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Default to common architectures
        list(APPEND CUDA_ENHANCED_COMPILE_FLAGS
            "--generate-code=arch=compute_70,code=sm_70"
            "--generate-code=arch=compute_75,code=sm_75"
            "--generate-code=arch=compute_80,code=sm_80"
            "--generate-code=arch=compute_86,code=sm_86"
        )
    endif()

    # Performance tuning flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND CUDA_ENHANCED_COMPILE_FLAGS
            "-O3"
            "--use_fast_math"
            "-DNDEBUG"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND CUDA_ENHANCED_COMPILE_FLAGS
            "-G"
            "-O0"
            "--device-debug"
        )
    endif()

    # Create enhanced CUDA target
    if(NOT TARGET CUDA::enhanced)
        add_library(CUDA::enhanced INTERFACE IMPORTED)
        set_target_properties(CUDA::enhanced PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${CUDA_ENHANCED_COMPILE_FLAGS}"
        )

        # Link common CUDA libraries
        set(ENHANCED_LINK_LIBRARIES CUDA::cudart)

        if(CUDAToolkit_HAS_CUBLAS)
            list(APPEND ENHANCED_LINK_LIBRARIES CUDA::cublas)
        endif()

        if(CUDAToolkit_HAS_CUFFT)
            list(APPEND ENHANCED_LINK_LIBRARIES CUDA::cufft)
        endif()

        if(CUDAToolkit_HAS_CURAND)
            list(APPEND ENHANCED_LINK_LIBRARIES CUDA::curand)
        endif()

        set_target_properties(CUDA::enhanced PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ENHANCED_LINK_LIBRARIES}"
        )
    endif()

    # Utility functions
    function(add_cuda_enhanced_executable target)
        set(sources ${ARGN})
        add_executable(${target} ${sources})

        target_link_libraries(${target} PRIVATE CUDA::enhanced)

        # Set CUDA separable compilation if needed
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endfunction()

    function(add_cuda_enhanced_library target)
        set(sources ${ARGN})
        add_library(${target} ${sources})

        target_link_libraries(${target} PUBLIC CUDA::enhanced)

        # Set CUDA properties
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
        )
    endfunction()

    # GPU detection utility
    function(detect_cuda_gpus)
        if(CUDAToolkit_NVCC_EXECUTABLE)
            execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
                OUTPUT_VARIABLE GPU_COMPUTE_CAPS
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(GPU_COMPUTE_CAPS)
                message(STATUS "Detected CUDA GPUs with compute capabilities: ${GPU_COMPUTE_CAPS}")
                # Convert to CMake list and set architectures
                string(REPLACE "\n" ";" GPU_CAPS_LIST ${GPU_COMPUTE_CAPS})
                string(REPLACE "." "" GPU_CAPS_LIST "${GPU_CAPS_LIST}")
                set(CMAKE_CUDA_ARCHITECTURES ${GPU_CAPS_LIST} PARENT_SCOPE)
            endif()
        endif()
    endfunction()

else()
    set(CUDAToolkitEnhanced_FOUND FALSE)

    # Try to provide helpful error messages
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        message(STATUS "NVIDIA driver found but CUDA Toolkit not detected")
        message(STATUS "Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
    else()
        message(STATUS "No NVIDIA GPU driver detected")
    endif()
endif()

# Standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDAToolkitEnhanced
    FOUND_VAR CUDAToolkitEnhanced_FOUND
    REQUIRED_VARS CUDAToolkit_FOUND
    VERSION_VAR CUDAToolkit_VERSION
)

if(CUDAToolkitEnhanced_FOUND)
    message(STATUS "CUDA Toolkit Enhanced: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "NVCC: ${CUDAToolkit_NVCC_EXECUTABLE}")

    # Show available enhanced libraries
    message(STATUS "Enhanced CUDA libraries:")
    foreach(lib ${CUDA_ENHANCED_LIBRARIES})
        string(TOUPPER ${lib} LIB_UPPER)
        if(CUDAToolkit_HAS_${LIB_UPPER})
            message(STATUS "  ✓ ${lib}")
        else()
            message(STATUS "  ✗ ${lib}")
        endif()
    endforeach()
endif()