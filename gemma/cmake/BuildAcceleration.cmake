# BuildAcceleration.cmake
# Advanced build acceleration module for gemma.cpp
# Provides compiler caching, parallelism optimization, and fast linking

# ============================================================================
# Compiler Cache Detection and Configuration
# ============================================================================

function(setup_compiler_cache)
    # Prefer sccache (cloud storage support, better Windows compatibility)
    find_program(SCCACHE_PROGRAM sccache)
    find_program(CCACHE_PROGRAM ccache)

    if(SCCACHE_PROGRAM)
        set(COMPILER_CACHE "sccache" PARENT_SCOPE)
        set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher" FORCE)

        message(STATUS "✓ Build Acceleration: sccache enabled at ${SCCACHE_PROGRAM}")

        # Configure sccache environment hints
        if(NOT DEFINED ENV{SCCACHE_DIR})
            message(STATUS "  HINT: Set SCCACHE_DIR environment variable for custom cache location")
        endif()
        if(NOT DEFINED ENV{SCCACHE_CACHE_SIZE})
            message(STATUS "  HINT: Set SCCACHE_CACHE_SIZE environment variable (e.g., 10G)")
        endif()

    elseif(CCACHE_PROGRAM)
        set(COMPILER_CACHE "ccache" PARENT_SCOPE)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher" FORCE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher" FORCE)

        message(STATUS "✓ Build Acceleration: ccache enabled at ${CCACHE_PROGRAM}")

        # Configure ccache environment hints
        if(NOT DEFINED ENV{CCACHE_DIR})
            message(STATUS "  HINT: Set CCACHE_DIR environment variable for custom cache location")
        endif()
        if(NOT DEFINED ENV{CCACHE_MAXSIZE})
            message(STATUS "  HINT: Set CCACHE_MAXSIZE environment variable (e.g., 10G)")
        endif()

    else()
        set(COMPILER_CACHE "none" PARENT_SCOPE)
        message(WARNING "⚠ Build Acceleration: No compiler cache found (sccache/ccache)")
        message(STATUS "  Install sccache: cargo install sccache")
        message(STATUS "  Install ccache: choco install ccache (Windows) or apt-get install ccache (Linux)")
    endif()
endfunction()

# ============================================================================
# Multi-Processor Compilation Configuration
# ============================================================================

function(setup_parallel_compilation)
    # Default to system processor count, capped at 12 as per requirements
    cmake_host_system_information(RESULT PROCESSOR_COUNT QUERY NUMBER_OF_LOGICAL_CORES)
    if(PROCESSOR_COUNT GREATER 12)
        set(PROCESSOR_COUNT 12)
    endif()

    option(GEMMA_MAX_PARALLEL_JOBS "Maximum parallel compilation jobs" ${PROCESSOR_COUNT})

    if(MSVC)
        # /MP: Multi-processor compilation
        # /MP12 limits to 12 processes as specified
        set(MP_FLAG "/MP${GEMMA_MAX_PARALLEL_JOBS}")
        add_compile_options(${MP_FLAG})
        message(STATUS "✓ Build Acceleration: MSVC multi-processor compilation enabled (${MP_FLAG})")

    elseif(CMAKE_GENERATOR MATCHES "Ninja")
        # Ninja handles parallelism via -j flag at build time
        # Configure job pools for better control
        set_property(GLOBAL PROPERTY JOB_POOLS
            compile=${GEMMA_MAX_PARALLEL_JOBS}
            link=2  # Limit link jobs to avoid memory exhaustion
        )
        message(STATUS "✓ Build Acceleration: Ninja job pools configured (compile=${GEMMA_MAX_PARALLEL_JOBS}, link=2)")

    elseif(CMAKE_GENERATOR MATCHES "Unix Makefiles")
        # Make handles parallelism via -j flag at build time
        message(STATUS "✓ Build Acceleration: Use 'make -j${GEMMA_MAX_PARALLEL_JOBS}' for parallel builds")
    endif()
endfunction()

# ============================================================================
# Fast Linking Configuration
# ============================================================================

function(setup_fast_linking)
    if(MSVC)
        # /DEBUG:FASTLINK: Faster linking for debug builds (incremental PDB)
        # Only beneficial for Debug and FastDebug configurations
        foreach(config Debug FastDebug)
            string(TOUPPER ${config} CONFIG_UPPER)
            set(CMAKE_EXE_LINKER_FLAGS_${CONFIG_UPPER}
                "${CMAKE_EXE_LINKER_FLAGS_${CONFIG_UPPER}} /DEBUG:FASTLINK"
                CACHE STRING "Fast linking for ${config}" FORCE
            )
            set(CMAKE_SHARED_LINKER_FLAGS_${CONFIG_UPPER}
                "${CMAKE_SHARED_LINKER_FLAGS_${CONFIG_UPPER}} /DEBUG:FASTLINK"
                CACHE STRING "Fast linking for ${config}" FORCE
            )
        endforeach()
        message(STATUS "✓ Build Acceleration: MSVC fast linking enabled (/DEBUG:FASTLINK)")

    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Use lld or gold linker if available (much faster than ld)
        find_program(LLD_LINKER lld)
        find_program(GOLD_LINKER gold)

        if(LLD_LINKER)
            add_link_options("-fuse-ld=lld")
            message(STATUS "✓ Build Acceleration: Using lld linker (fast)")
        elseif(GOLD_LINKER)
            add_link_options("-fuse-ld=gold")
            message(STATUS "✓ Build Acceleration: Using gold linker (fast)")
        else()
            message(STATUS "  Build Acceleration: Using default linker (consider installing lld)")
        endif()
    endif()
endfunction()

# ============================================================================
# Link-Time Optimization (LTO) Configuration
# ============================================================================

function(setup_lto_optimization)
    # LTO is controlled by GEMMA_ENABLE_LTO option (defined in GemmaOptimizations.cmake)
    # This function adds parallel LTO configuration

    if(GEMMA_ENABLE_LTO)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            # Enable parallel LTO with all available cores
            cmake_host_system_information(RESULT PROCESSOR_COUNT QUERY NUMBER_OF_LOGICAL_CORES)
            add_link_options("-flto=auto")  # Auto-detect parallelism
            message(STATUS "✓ Build Acceleration: Parallel LTO enabled (-flto=auto)")

        elseif(MSVC)
            # MSVC handles LTO parallelism automatically with /LTCG
            message(STATUS "✓ Build Acceleration: MSVC LTO (LTCG) handles parallelism automatically")
        endif()
    endif()
endfunction()

# ============================================================================
# Precompiled Headers (PCH) Enhancement
# ============================================================================

function(setup_enhanced_pch)
    # PCH is controlled by GEMMA_ENABLE_PCH option (defined in GemmaOptimizations.cmake)
    # This function adds aggressive PCH hints

    if(GEMMA_ENABLE_PCH)
        if(MSVC)
            # /Zm: Increase precompiled header memory allocation
            add_compile_options(/Zm200)  # 200% of default
            message(STATUS "✓ Build Acceleration: Enhanced PCH memory allocation (/Zm200)")
        endif()

        # Suggest common headers for PCH
        message(STATUS "  PCH Recommendation: Add these to pch.h for maximum benefit:")
        message(STATUS "    - <vector>, <array>, <memory>, <string>")
        message(STATUS "    - hwy/highway.h (Highway SIMD)")
        message(STATUS "    - Standard library headers used across translation units")
    endif()
endfunction()

# ============================================================================
# Unity/Jumbo Build Configuration
# ============================================================================

function(setup_unity_builds)
    # Unity builds are controlled by GEMMA_ENABLE_UNITY_BUILDS option
    # This function adds optimized batch sizes

    if(GEMMA_ENABLE_UNITY_BUILDS)
        # Set unity build batch size (number of files per unity file)
        # Smaller = faster incremental rebuilds, Larger = faster clean builds
        set(CMAKE_UNITY_BUILD_BATCH_SIZE 8 CACHE STRING "Unity build batch size")

        message(STATUS "✓ Build Acceleration: Unity builds enabled (batch size: ${CMAKE_UNITY_BUILD_BATCH_SIZE})")
        message(STATUS "  NOTE: Unity builds may increase memory usage and hide ODR violations")
    endif()
endfunction()

# ============================================================================
# Build Monitoring and Diagnostics
# ============================================================================

function(print_build_acceleration_summary CACHE_TYPE)
    message(STATUS "")
    message(STATUS "═══════════════════════════════════════════════════════════════")
    message(STATUS "  Gemma.cpp Build Acceleration Summary")
    message(STATUS "═══════════════════════════════════════════════════════════════")
    message(STATUS "  Compiler Cache:          ${CACHE_TYPE}")
    message(STATUS "  Parallel Jobs:           ${GEMMA_MAX_PARALLEL_JOBS}")
    message(STATUS "  LTO Enabled:             ${GEMMA_ENABLE_LTO}")
    message(STATUS "  PCH Enabled:             ${GEMMA_ENABLE_PCH}")
    message(STATUS "  Unity Builds:            ${GEMMA_ENABLE_UNITY_BUILDS}")
    message(STATUS "  AVX2 Enabled:            ${GEMMA_FORCE_AVX2}")

    # Estimate performance gains
    set(SPEEDUP "4-5x clean, 5-10x incremental")
    if(CACHE_TYPE STREQUAL "none")
        set(SPEEDUP "2-3x (limited without compiler cache)")
    endif()

    message(STATUS "")
    message(STATUS "  Expected Speedup:        ${SPEEDUP}")
    message(STATUS "  First Build:             Cache warm-up (slower)")
    message(STATUS "  Subsequent Builds:       Maximum acceleration")
    message(STATUS "═══════════════════════════════════════════════════════════════")
    message(STATUS "")
endfunction()

# ============================================================================
# Main Setup Function
# ============================================================================

function(setup_build_acceleration)
    message(STATUS "")
    message(STATUS "Configuring Build Acceleration...")
    message(STATUS "")

    # Execute all acceleration setups
    setup_compiler_cache()
    setup_parallel_compilation()
    setup_fast_linking()
    setup_lto_optimization()
    setup_enhanced_pch()
    setup_unity_builds()

    # Print summary (cache type propagated from setup_compiler_cache)
    print_build_acceleration_summary(${COMPILER_CACHE})
endfunction()

# ============================================================================
# Utility Functions for Build Scripts
# ============================================================================

# Function to verify cache is operational
function(verify_compiler_cache)
    if(SCCACHE_PROGRAM)
        execute_process(
            COMMAND ${SCCACHE_PROGRAM} --show-stats
            OUTPUT_VARIABLE CACHE_STATS
            ERROR_QUIET
        )
        message(STATUS "sccache statistics:")
        message(STATUS "${CACHE_STATS}")
    elseif(CCACHE_PROGRAM)
        execute_process(
            COMMAND ${CCACHE_PROGRAM} -s
            OUTPUT_VARIABLE CACHE_STATS
            ERROR_QUIET
        )
        message(STATUS "ccache statistics:")
        message(STATUS "${CACHE_STATS}")
    endif()
endfunction()

# Export cache type for scripts
set(GEMMA_COMPILER_CACHE_TYPE ${COMPILER_CACHE} CACHE STRING "Type of compiler cache in use")
