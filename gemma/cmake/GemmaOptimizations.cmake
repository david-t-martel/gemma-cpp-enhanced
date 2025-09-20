# Gemma.cpp Build Optimizations
# This module provides comprehensive build optimizations including:
# - ccache configuration
# - Precompiled headers
# - Parallel build settings
# - Template instantiation optimization
# - Incremental build strategies

# ccache Configuration
function(configure_ccache)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "Found ccache: ${CCACHE_PROGRAM}")

        # Set ccache as compiler launcher
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" PARENT_SCOPE)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" PARENT_SCOPE)

        # Configure ccache environment
        set(ENV{CCACHE_MAXSIZE} "5G")              # 5GB cache size
        set(ENV{CCACHE_COMPRESS} "true")           # Compress cached objects
        set(ENV{CCACHE_COMPRESSLEVEL} "6")         # Compression level
        set(ENV{CCACHE_SLOPPINESS} "pch_defines,time_macros")  # More permissive caching
        set(ENV{CCACHE_NOHASHDIR} "true")          # Don't hash directories
        set(ENV{CCACHE_BASEDIR} "${CMAKE_SOURCE_DIR}")  # Base directory for relative paths

        # Enable stats collection
        execute_process(COMMAND ${CCACHE_PROGRAM} --zero-stats OUTPUT_QUIET)

        message(STATUS "ccache configured successfully")
        message(STATUS "  Max size: 5GB")
        message(STATUS "  Compression: enabled (level 6)")
        message(STATUS "  Base directory: ${CMAKE_SOURCE_DIR}")
    else()
        message(WARNING "ccache not found. Install ccache for faster rebuilds.")
        message(STATUS "  Windows: scoop install ccache")
        message(STATUS "  Linux: sudo apt install ccache")
        message(STATUS "  macOS: brew install ccache")
    endif()
endfunction()

# Precompiled Headers Configuration
function(configure_precompiled_headers target_name)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
        # Use CMake's native PCH support (CMake 3.16+)
        target_precompile_headers(${target_name} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.h>"
        )
        message(STATUS "Enabled native precompiled headers for ${target_name}")
    else()
        message(WARNING "CMake 3.16+ required for native PCH support")
    endif()
endfunction()

# Template Instantiation Optimization
function(optimize_template_compilation target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Reduce template instantiation depth and improve error messages
        target_compile_options(${target_name} PRIVATE
            -ftemplate-depth=1000           # Increase template depth limit
            -ftemplate-backtrace-limit=10   # Limit template error backtraces
        )

        # Enable template instantiation caching (GCC 9+)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
            target_compile_options(${target_name} PRIVATE
                -fno-implicit-templates     # Don't instantiate templates unless used
            )
        endif()

        # Clang-specific optimizations
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${target_name} PRIVATE
                -fdelayed-template-parsing  # Delay template parsing
            )
        endif()

    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # MSVC template optimizations
        target_compile_options(${target_name} PRIVATE
            /bigobj                         # Increase object file size limit
            /constexpr:depth1000           # Increase constexpr evaluation depth
            /constexpr:backtrace10         # Limit constexpr error backtraces
        )
    endif()
endfunction()

# Parallel Build Configuration
function(configure_parallel_builds)
    # Set parallel build based on processor count
    include(ProcessorCount)
    ProcessorCount(N)
    if(NOT N EQUAL 0)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N} PARENT_SCOPE)
        message(STATUS "Configured for ${N} parallel build jobs")
    endif()

    # MSVC: Enable multiprocessor compilation
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/MP)
        message(STATUS "Enabled MSVC multiprocessor compilation")
    endif()
endfunction()

# Unity Build Configuration (for faster compilation)
function(configure_unity_builds target_name)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
        # Group source files for unity builds
        set_target_properties(${target_name} PROPERTIES
            UNITY_BUILD ON
            UNITY_BUILD_BATCH_SIZE 8        # Files per unity batch
        )
        message(STATUS "Enabled unity builds for ${target_name} (batch size: 8)")
    endif()
endfunction()

# Incremental Build Optimization
function(optimize_incremental_builds target_name)
    # Separate object files by configuration
    set_target_properties(${target_name} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )

    # Enable fast linking (Windows)
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_link_options(${target_name} PRIVATE
            $<$<CONFIG:Debug>:/DEBUG:FASTLINK>     # Faster debug linking
            $<$<CONFIG:Release>:/OPT:REF>          # Remove unused functions
            $<$<CONFIG:Release>:/OPT:ICF>          # Identical COMDAT folding
        )
    endif()
endfunction()

# SIMD Optimization Configuration
function(configure_simd_optimizations target_name)
    # Highway SIMD optimizations
    target_compile_definitions(${target_name} PRIVATE
        HWY_COMPILE_ONLY_SCALAR=0           # Enable all SIMD targets
        HWY_DISABLED_TARGETS=0              # Don't disable any targets
    )

    # Platform-specific SIMD flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target_name} PRIVATE
                -msse4.2 -mavx2             # Enable modern x86 SIMD
            )
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            target_compile_options(${target_name} PRIVATE
                -mcpu=native                # Enable ARM NEON optimizations
            )
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target_name} PRIVATE
            /arch:AVX2                      # Enable AVX2 on MSVC
        )
    endif()
endfunction()

# Memory Optimization Configuration
function(configure_memory_optimizations target_name)
    # Reduce memory usage during compilation
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE
            -fno-keep-inline-dllexport      # Reduce object file size
        )

        # Large file support
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target_name} PRIVATE
                -Wa,--compress-debug-sections   # Compress debug sections
            )
        endif()
    endif()
endfunction()

# Complete Optimization Setup
function(apply_gemma_optimizations target_name)
    message(STATUS "Applying comprehensive optimizations to ${target_name}")

    # Core optimizations
    configure_precompiled_headers(${target_name})
    optimize_template_compilation(${target_name})
    optimize_incremental_builds(${target_name})
    configure_simd_optimizations(${target_name})
    configure_memory_optimizations(${target_name})

    # Optional unity builds (can be controlled by option)
    option(GEMMA_ENABLE_UNITY_BUILDS "Enable unity builds for faster compilation" OFF)
    if(GEMMA_ENABLE_UNITY_BUILDS)
        configure_unity_builds(${target_name})
    endif()

    message(STATUS "Optimization setup complete for ${target_name}")
endfunction()

# Global setup function
function(setup_gemma_build_optimizations)
    message(STATUS "=== Setting up Gemma.cpp Build Optimizations ===")

    configure_ccache()
    configure_parallel_builds()

    # Global compiler optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Applying Release build optimizations")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            add_compile_options(-O3 -DNDEBUG -flto)
            add_link_options(-flto)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_compile_options(/O2 /DNDEBUG /GL)
            add_link_options(/LTCG)
        endif()
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Applying Debug build optimizations")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            add_compile_options(-O0 -g3)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_compile_options(/Od /Zi)
        endif()
    endif()

    message(STATUS "=== Build Optimization Setup Complete ===")
endfunction()

# Build profiles
function(setup_build_profiles)
    # RelWithSymbols profile (optimized but with debug symbols)
    set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "-O2 -g -DNDEBUG" CACHE STRING
        "Flags used by the C++ compiler during RelWithSymbols builds." FORCE)
    set(CMAKE_C_FLAGS_RELWITHSYMBOLS "-O2 -g -DNDEBUG" CACHE STRING
        "Flags used by the C compiler during RelWithSymbols builds." FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHSYMBOLS "" CACHE STRING
        "Flags used for linking binaries during RelWithSymbols builds." FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHSYMBOLS "" CACHE STRING
        "Flags used by the shared libraries linker during RelWithSymbols builds." FORCE)
    mark_as_advanced(
        CMAKE_CXX_FLAGS_RELWITHSYMBOLS
        CMAKE_C_FLAGS_RELWITHSYMBOLS
        CMAKE_EXE_LINKER_FLAGS_RELWITHSYMBOLS
        CMAKE_SHARED_LINKER_FLAGS_RELWITHSYMBOLS
    )

    # FastDebug profile (minimal optimization with debug symbols)
    set(CMAKE_CXX_FLAGS_FASTDEBUG "-O1 -g -DDEBUG" CACHE STRING
        "Flags used by the C++ compiler during FastDebug builds." FORCE)
    set(CMAKE_C_FLAGS_FASTDEBUG "-O1 -g -DDEBUG" CACHE STRING
        "Flags used by the C compiler during FastDebug builds." FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_FASTDEBUG "" CACHE STRING
        "Flags used for linking binaries during FastDebug builds." FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_FASTDEBUG "" CACHE STRING
        "Flags used by the shared libraries linker during FastDebug builds." FORCE)
    mark_as_advanced(
        CMAKE_CXX_FLAGS_FASTDEBUG
        CMAKE_C_FLAGS_FASTDEBUG
        CMAKE_EXE_LINKER_FLAGS_FASTDEBUG
        CMAKE_SHARED_LINKER_FLAGS_FASTDEBUG
    )
endfunction()

# Intel oneAPI Specific Optimizations
# These functions provide Intel compiler-specific optimizations for maximum performance

# Intel MKL Configuration
function(configure_intel_mkl target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Configuring Intel MKL for ${target_name}")

        # Find Intel MKL
        find_package(MKL QUIET)
        if(MKL_FOUND OR EXISTS "$ENV{MKLROOT}")
            # Use Intel MKL for BLAS/LAPACK operations
            target_compile_definitions(${target_name} PRIVATE
                USE_INTEL_MKL
                MKL_ILP64                    # 64-bit integer interface
                EIGEN_USE_MKL_ALL           # Use MKL for Eigen operations
            )

            # Intel MKL linking
            target_compile_options(${target_name} PRIVATE -mkl=parallel)
            target_link_options(${target_name} PRIVATE -mkl=parallel)

            # MKL threading control
            target_compile_definitions(${target_name} PRIVATE
                MKL_THREADING_INTEL         # Use Intel threading
                MKL_DOMAIN_ALL             # Use MKL for all domains
            )

            message(STATUS "Intel MKL enabled for ${target_name}")
        else()
            message(WARNING "Intel MKL not found for ${target_name}")
        endif()
    endif()
endfunction()

# Intel IPP Configuration
function(configure_intel_ipp target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Configuring Intel IPP for ${target_name}")

        # Find Intel IPP
        if(EXISTS "$ENV{IPPROOT}")
            # Use Intel IPP for signal processing operations
            target_compile_definitions(${target_name} PRIVATE
                USE_INTEL_IPP
                IPP_STATIC_LIBS             # Use static IPP libraries
            )

            # Intel IPP linking
            target_compile_options(${target_name} PRIVATE -ipp=parallel)
            target_link_options(${target_name} PRIVATE -ipp=parallel)

            message(STATUS "Intel IPP enabled for ${target_name}")
        else()
            message(WARNING "Intel IPP not found for ${target_name}")
        endif()
    endif()
endfunction()

# Intel TBB Configuration
function(configure_intel_tbb target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Configuring Intel TBB for ${target_name}")

        # Find Intel TBB
        find_package(TBB QUIET)
        if(TBB_FOUND OR EXISTS "$ENV{TBBROOT}")
            # Use Intel TBB for parallel algorithms
            target_compile_definitions(${target_name} PRIVATE
                USE_INTEL_TBB
                TBB_USE_THREADING_TOOLS     # Enable TBB threading tools
            )

            # Link with Intel TBB
            target_link_libraries(${target_name} PRIVATE TBB::tbb)

            message(STATUS "Intel TBB enabled for ${target_name}")
        else()
            message(WARNING "Intel TBB not found for ${target_name}")
        endif()
    endif()
endfunction()

# Intel VTune Profiling Configuration
function(configure_intel_vtune target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Configuring Intel VTune profiling for ${target_name}")

        # VTune profiling support
        target_compile_options(${target_name} PRIVATE
            -g                              # Debug information
            -gdwarf-4                       # DWARF 4 debug format
            -fno-omit-frame-pointer         # Keep frame pointers
            -fno-inline-functions           # Disable aggressive inlining for profiling
        )

        # VTune API integration if available
        if(EXISTS "$ENV{VTUNE_PROFILER_DIR}")
            target_include_directories(${target_name} PRIVATE
                "$ENV{VTUNE_PROFILER_DIR}/include"
            )

            target_compile_definitions(${target_name} PRIVATE
                INTEL_VTUNE_PROFILING       # Enable VTune API calls
            )

            # Link with VTune libraries
            if(WIN32)
                target_link_libraries(${target_name} PRIVATE
                    "$ENV{VTUNE_PROFILER_DIR}/lib64/libittnotify.lib"
                )
            else()
                target_link_libraries(${target_name} PRIVATE
                    "$ENV{VTUNE_PROFILER_DIR}/lib64/libittnotify.a"
                    dl
                )
            endif()

            message(STATUS "Intel VTune API integration enabled for ${target_name}")
        endif()
    endif()
endfunction()

# Intel Highway SIMD Optimizations
function(configure_intel_highway_simd target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Configuring Intel-optimized Highway SIMD for ${target_name}")

        # Intel-specific Highway configuration
        target_compile_definitions(${target_name} PRIVATE
            HWY_INTEL_OPTIMIZED=1           # Enable Intel optimizations
            HWY_COMPILE_ONLY_SCALAR=0       # Enable all SIMD targets
            HWY_DISABLED_TARGETS=0          # Don't disable any targets
            HWY_WANT_AVX512=1              # Enable AVX-512 if available
        )

        # Intel SIMD instruction sets
        target_compile_options(${target_name} PRIVATE
            -msse4.2                        # SSE 4.2
            -mavx2                          # AVX2
            -mfma                           # Fused multiply-add
        )

        # Check for AVX-512 support
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
        if(COMPILER_SUPPORTS_AVX512F)
            target_compile_options(${target_name} PRIVATE
                -mavx512f                   # AVX-512 Foundation
                -mavx512cd                  # AVX-512 Conflict Detection
                -mavx512bw                  # AVX-512 Byte and Word
                -mavx512dq                  # AVX-512 Doubleword and Quadword
                -mavx512vl                  # AVX-512 Vector Length Extensions
            )

            target_compile_definitions(${target_name} PRIVATE
                HWY_INTEL_AVX512_ENABLED=1
            )

            message(STATUS "Intel AVX-512 enabled for ${target_name}")
        else()
            message(STATUS "Intel AVX-512 not available, using AVX2 for ${target_name}")
        endif()
    endif()
endfunction()

# Intel Aggressive Optimization Configuration
function(configure_intel_aggressive_optimizations target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Applying Intel aggressive optimizations to ${target_name}")

        # Intel-specific optimization flags
        target_compile_options(${target_name} PRIVATE
            -O3                             # Maximum optimization
            -xHost                          # Auto-detect CPU and optimize
            -march=native                   # Use native instruction set
            -mtune=native                   # Tune for native CPU
            -ffast-math                     # Aggressive math optimizations
            -finline-functions              # Aggressive function inlining
            -funroll-loops                  # Loop unrolling
            -fno-alias                      # Assume no pointer aliasing
            -fopenmp                        # OpenMP parallel execution
            -parallel                       # Intel parallel optimization
        )

        # Intel-specific security flags
        target_compile_options(${target_name} PRIVATE
            -fstack-protector-strong        # Stack protection
            -D_FORTIFY_SOURCE=2             # Buffer overflow protection
        )

        # Link-time optimization
        target_compile_options(${target_name} PRIVATE -flto)
        target_link_options(${target_name} PRIVATE -flto)

        message(STATUS "Intel aggressive optimizations applied to ${target_name}")
    endif()
endfunction()

# Comprehensive Intel Optimization Setup
function(apply_intel_optimizations target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "=== Applying Intel oneAPI optimizations to ${target_name} ===")

        # Apply all Intel optimizations
        configure_intel_aggressive_optimizations(${target_name})
        configure_intel_mkl(${target_name})
        configure_intel_ipp(${target_name})
        configure_intel_tbb(${target_name})
        configure_intel_highway_simd(${target_name})
        configure_intel_vtune(${target_name})

        # Add Intel-specific definitions
        target_compile_definitions(${target_name} PRIVATE
            INTEL_OPTIMIZED_BUILD=1
            INTEL_ONEAPI_VERSION=2025
            GEMMA_INTEL_OPTIMIZED=1
        )

        message(STATUS "=== Intel oneAPI optimizations complete for ${target_name} ===")
    else()
        message(STATUS "Non-Intel compiler detected, skipping Intel optimizations for ${target_name}")
    endif()
endfunction()

# Intel Performance Monitoring Setup
function(setup_intel_performance_monitoring target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        message(STATUS "Setting up Intel performance monitoring for ${target_name}")

        # Performance monitoring compile definitions
        target_compile_definitions(${target_name} PRIVATE
            INTEL_PERFORMANCE_MONITORING=1
            ENABLE_PERFORMANCE_COUNTERS=1
            VTUNE_PROFILING_ENABLED=1
        )

        # Create performance monitoring wrapper
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/intel_perf_monitor.h.in"
            "${CMAKE_CURRENT_BINARY_DIR}/include/intel_perf_monitor.h"
            @ONLY
        )

        target_include_directories(${target_name} PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/include"
        )

        message(STATUS "Intel performance monitoring configured for ${target_name}")
    endif()
endfunction()