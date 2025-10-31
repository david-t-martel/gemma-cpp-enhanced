# Intel OneAPI Configuration Module for Gemma.cpp
# This module sets up Intel OneAPI optimizations including:
# - Intel DPC++ compiler (icpx)
# - Intel MKL for optimized BLAS operations
# - Intel IPP for signal processing
# - Intel TBB for threading
# - Intel SYCL for GPU offload (optional)

# Detect Intel OneAPI installation
set(INTEL_ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")
if(NOT EXISTS "${INTEL_ONEAPI_ROOT}")
    message(WARNING "Intel OneAPI not found at default location: ${INTEL_ONEAPI_ROOT}")
    return()
endif()

message(STATUS "==========================================")
message(STATUS "Intel OneAPI Integration Enabled")
message(STATUS "Installation Root: ${INTEL_ONEAPI_ROOT}")
message(STATUS "==========================================")

# Set Intel compiler paths
set(INTEL_COMPILER_ROOT "${INTEL_ONEAPI_ROOT}/compiler/latest/windows")
set(INTEL_MKL_ROOT "${INTEL_ONEAPI_ROOT}/mkl/latest")
set(INTEL_IPP_ROOT "${INTEL_ONEAPI_ROOT}/ipp/latest")
set(INTEL_TBB_ROOT "${INTEL_ONEAPI_ROOT}/tbb/latest")
set(INTEL_DNNL_ROOT "${INTEL_ONEAPI_ROOT}/dnnl/latest")

# Function to setup Intel compiler flags
function(setup_intel_compiler_flags target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        # Intel compiler-specific optimizations
        target_compile_options(${target} PRIVATE
            # Core optimization flags
            /O3                    # Maximum optimization
            /Qipo                  # Interprocedural optimization
            /QxHOST                # Optimize for current CPU
            /Qvec                  # Enable vectorization
            /Qsimd                 # Enable SIMD optimizations
            /fp:fast               # Fast floating-point model
            /Qparallel             # Auto-parallelization
            /Qopenmp               # OpenMP support

            # Intel-specific optimizations
            /Qopt-matmul           # Optimize matrix multiplication
            /Qopt-prefetch:5       # Aggressive prefetching
            /Qopt-streaming-stores:always  # Streaming stores
            /Qopt-mem-layout-trans:3      # Memory layout transformations

            # Vectorization reports (optional, for analysis)
            # /Qvec-report:2
            # /Qopt-report:5
            # /Qopt-report-phase:vec
        )

        # Link-time optimization
        target_link_options(${target} PRIVATE
            /Qipo                  # Interprocedural optimization
            /Qparallel             # Auto-parallelization
        )
    endif()
endfunction()

# Function to setup Intel MKL
function(setup_intel_mkl target)
    if(NOT EXISTS "${INTEL_MKL_ROOT}")
        message(WARNING "Intel MKL not found at: ${INTEL_MKL_ROOT}")
        return()
    endif()

    message(STATUS "Configuring Intel MKL for ${target}")

    # MKL include directories
    target_include_directories(${target} PRIVATE
        "${INTEL_MKL_ROOT}/include"
    )

    # MKL compile definitions
    target_compile_definitions(${target} PRIVATE
        USE_INTEL_MKL
        MKL_ILP64              # Use 64-bit integers
        MKL_DIRECT_CALL        # Direct function calls
        EIGEN_USE_MKL_ALL      # If using Eigen
    )

    # MKL libraries based on compiler
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        # Use Intel compiler's built-in MKL support
        target_compile_options(${target} PRIVATE
            /Qmkl:parallel        # Use parallel MKL
        )
        target_link_options(${target} PRIVATE
            /Qmkl:parallel
        )
    else()
        # Manual MKL linking for non-Intel compilers
        target_link_directories(${target} PRIVATE
            "${INTEL_MKL_ROOT}/lib/intel64"
        )

        target_link_libraries(${target} PRIVATE
            mkl_intel_ilp64
            mkl_intel_thread
            mkl_core
            libiomp5md          # Intel OpenMP runtime
        )
    endif()

    # Add MKL DLL directory to PATH (for runtime)
    set(MKL_DLL_DIR "${INTEL_MKL_ROOT}/redist/intel64" PARENT_SCOPE)
endfunction()

# Function to setup Intel IPP
function(setup_intel_ipp target)
    if(NOT EXISTS "${INTEL_IPP_ROOT}")
        message(WARNING "Intel IPP not found at: ${INTEL_IPP_ROOT}")
        return()
    endif()

    message(STATUS "Configuring Intel IPP for ${target}")

    target_include_directories(${target} PRIVATE
        "${INTEL_IPP_ROOT}/include"
    )

    target_compile_definitions(${target} PRIVATE
        USE_INTEL_IPP
    )

    target_link_directories(${target} PRIVATE
        "${INTEL_IPP_ROOT}/lib/intel64"
    )

    # Core IPP libraries
    target_link_libraries(${target} PRIVATE
        ipps                   # Signal processing
        ippvm                  # Vector math
        ippcore                # Core functions
    )

    # Add IPP DLL directory to PATH (for runtime)
    set(IPP_DLL_DIR "${INTEL_IPP_ROOT}/redist/intel64" PARENT_SCOPE)
endfunction()

# Function to setup Intel TBB
function(setup_intel_tbb target)
    if(NOT EXISTS "${INTEL_TBB_ROOT}")
        message(WARNING "Intel TBB not found at: ${INTEL_TBB_ROOT}")
        return()
    endif()

    message(STATUS "Configuring Intel TBB for ${target}")

    target_include_directories(${target} PRIVATE
        "${INTEL_TBB_ROOT}/include"
    )

    target_compile_definitions(${target} PRIVATE
        USE_INTEL_TBB
        TBB_SUPPRESS_DEPRECATED_MESSAGES
    )

    target_link_directories(${target} PRIVATE
        "${INTEL_TBB_ROOT}/lib/intel64/vc14"
    )

    target_link_libraries(${target} PRIVATE
        tbb
        tbbmalloc              # Scalable memory allocator
        tbbmalloc_proxy        # Proxy for malloc/free
    )

    # Add TBB DLL directory to PATH (for runtime)
    set(TBB_DLL_DIR "${INTEL_TBB_ROOT}/redist/intel64/vc14" PARENT_SCOPE)
endfunction()

# Function to setup Intel DNNL (Deep Neural Network Library)
function(setup_intel_dnnl target)
    if(NOT EXISTS "${INTEL_DNNL_ROOT}")
        message(WARNING "Intel DNNL not found at: ${INTEL_DNNL_ROOT}")
        return()
    endif()

    message(STATUS "Configuring Intel DNNL for ${target}")

    target_include_directories(${target} PRIVATE
        "${INTEL_DNNL_ROOT}/include"
    )

    target_compile_definitions(${target} PRIVATE
        USE_INTEL_DNNL
    )

    target_link_directories(${target} PRIVATE
        "${INTEL_DNNL_ROOT}/lib"
    )

    target_link_libraries(${target} PRIVATE
        dnnl
    )

    # Add DNNL DLL directory to PATH (for runtime)
    set(DNNL_DLL_DIR "${INTEL_DNNL_ROOT}/bin" PARENT_SCOPE)
endfunction()

# Function to setup Intel SYCL (Data Parallel C++)
function(setup_intel_sycl target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        message(STATUS "Configuring Intel SYCL for ${target}")

        target_compile_options(${target} PRIVATE
            -fsycl                 # Enable SYCL
            -fsycl-device-code-split=per_kernel  # Split device code
        )

        target_link_options(${target} PRIVATE
            -fsycl
        )

        target_compile_definitions(${target} PRIVATE
            USE_INTEL_SYCL
            SYCL_LANGUAGE_VERSION=2020
        )
    else()
        message(WARNING "Intel SYCL requires Intel DPC++ compiler (icpx)")
    endif()
endfunction()

# Main function to apply all Intel optimizations
function(apply_intel_optimizations target)
    message(STATUS "")
    message(STATUS "Applying Intel OneAPI optimizations to ${target}")
    message(STATUS "========================================")

    # Apply compiler-specific optimizations
    setup_intel_compiler_flags(${target})

    # Apply individual component optimizations based on options
    if(GEMMA_USE_INTEL_MKL)
        setup_intel_mkl(${target})
    endif()

    if(GEMMA_USE_INTEL_IPP)
        setup_intel_ipp(${target})
    endif()

    if(GEMMA_USE_INTEL_TBB)
        setup_intel_tbb(${target})
    endif()

    if(GEMMA_USE_INTEL_DNNL)
        setup_intel_dnnl(${target})
    endif()

    if(GEMMA_USE_INTEL_SYCL)
        setup_intel_sycl(${target})
    endif()

    # Set up runtime environment variables
    if(MKL_DLL_DIR OR IPP_DLL_DIR OR TBB_DLL_DIR OR DNNL_DLL_DIR)
        message(STATUS "")
        message(STATUS "Runtime DLL Directories (add to PATH):")
        if(MKL_DLL_DIR)
            message(STATUS "  MKL: ${MKL_DLL_DIR}")
        endif()
        if(IPP_DLL_DIR)
            message(STATUS "  IPP: ${IPP_DLL_DIR}")
        endif()
        if(TBB_DLL_DIR)
            message(STATUS "  TBB: ${TBB_DLL_DIR}")
        endif()
        if(DNNL_DLL_DIR)
            message(STATUS "  DNNL: ${DNNL_DLL_DIR}")
        endif()
    endif()

    message(STATUS "========================================")
endfunction()

# Performance benchmarking helper
function(setup_intel_benchmarking target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        target_compile_options(${target} PRIVATE
            /Qopt-report:5         # Generate optimization report
            /Qopt-report-phase:vec # Focus on vectorization
            /Qvec-report:2         # Vectorization report
        )
    endif()
endfunction()