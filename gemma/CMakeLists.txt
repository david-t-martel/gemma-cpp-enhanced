# Enhanced Gemma.cpp Project Root CMakeLists.txt
# This file orchestrates the build of the enhanced Gemma.cpp with hardware acceleration backends

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Early vcpkg toolchain integration (must occur before first project()).
# If the user did not specify -DCMAKE_TOOLCHAIN_FILE and VCPKG_ROOT is present,
# auto-wire the vcpkg toolchain so that all subsequent find_package() calls
# (in root and subdirectories) resolve to vcpkg-provided ports before any
# FetchContent fallbacks execute in subprojects.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(_VCPKG_CAND "$ENV{VCPKG_ROOT}")
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
        # Monorepo layout sibling
        get_filename_component(_VCPKG_CAND "${CMAKE_CURRENT_LIST_DIR}/../vcpkg" ABSOLUTE)
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        # In-tree vcpkg clone
        get_filename_component(_VCPKG_CAND "${CMAKE_CURRENT_LIST_DIR}/vcpkg" ABSOLUTE)
    endif()
    if(_VCPKG_CAND AND EXISTS "${_VCPKG_CAND}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${_VCPKG_CAND}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain")
        set(GEMMA_USING_VCPKG ON CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
        message(STATUS "[vcpkg] Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    else()
        set(GEMMA_USING_VCPKG OFF CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
    endif()
else()
    # Respect user-specified toolchain; detect if it is vcpkg for reporting.
    if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake$")
        set(GEMMA_USING_VCPKG ON CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
    else()
        set(GEMMA_USING_VCPKG OFF CACHE BOOL "Gemma build is using vcpkg toolchain" FORCE)
    endif()
endif()

# Set policy versions to handle compatibility with dependencies
# CMP0111: An imported target with INTERFACE_SOURCES may not be built by the generator
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

# CMP0091: MSVC runtime library flags are selected by an abstraction
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# CMP0126: set(CACHE) does not remove a normal variable of the same name
if(POLICY CMP0126)
  cmake_policy(SET CMP0126 NEW)
endif()

# CMP0169: FetchContent_Populate deprecation
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 NEW)
endif()

# Set minimum policy version for all subdirectories to handle old dependencies
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # option() honors normal variables
set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)  # Include file check macros honor CMAKE_REQUIRED_LIBRARIES

project(gemma_enhanced
    VERSION 1.0.0
    DESCRIPTION "Enhanced Gemma.cpp with MCP server and hardware acceleration backends"
    LANGUAGES CXX C
)

# Early environment validation (optional step to diagnose user setups)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnvValidation)
gemma_validate_environment()

# Set global C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------
# Optional: force-enable AVX2 (and FMA) for x86_64 builds.
# Highway will still multi-target internally, but /arch:AVX2 on MSVC
# ensures the compiler is allowed to emit AVX2 code in generated TU(s).
# Can be disabled with -DGEMMA_FORCE_AVX2=OFF or overridden by a user-
# supplied /arch flag.
# ---------------------------------------------------------------
option(GEMMA_FORCE_AVX2 "Force /arch:AVX2 (or -mavx2) for supported x86_64 builds" ON)
option(GEMMA_REQUIRE_AVX2 "Fail CMake configure if AVX2 cannot be enabled" ON)
if(GEMMA_FORCE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "[Xx]86_64|AMD64")
    # Skip if user already injected an /arch flag manually.
    string(REGEX MATCH "/arch:[A-Za-z0-9_]+" _have_arch_flag "${CMAKE_CXX_FLAGS}")
    if(NOT _have_arch_flag)
        if(MSVC)
            add_compile_options(/arch:AVX2)
            # Also append to global flags so try_compile/check_cxx_source_compiles inherit it
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2" CACHE STRING "Global CXX flags" FORCE)
            set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /arch:AVX2")
            message(STATUS "GEMMA_FORCE_AVX2: enabling /arch:AVX2")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_compile_options(-mavx2 -mfma)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma" CACHE STRING "Global CXX flags" FORCE)
            set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mavx2 -mfma")
            message(STATUS "GEMMA_FORCE_AVX2: enabling -mavx2 -mfma")
        endif()
    else()
        message(STATUS "GEMMA_FORCE_AVX2 requested but existing /arch flag present; leaving as-is: ${_have_arch_flag}")
    endif()
endif()

# If AVX2 is required, perform a compile-time probe to ensure __AVX2__ is defined
if(GEMMA_REQUIRE_AVX2)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "[Xx]86_64|AMD64")
        message(FATAL_ERROR "GEMMA_REQUIRE_AVX2=ON but target processor '${CMAKE_SYSTEM_PROCESSOR}' is not x86_64/AMD64")
    endif()
    # For MSVC, rely on presence of /arch:AVX2 flag because try_compile may drop it in some generators.
    if(MSVC)
        string(REGEX MATCH "/arch:AVX2" _msvc_has_avx2 "${CMAKE_CXX_FLAGS}")
        if(_msvc_has_avx2)
            set(GEMMA_HAS_AVX2 TRUE)
            message(STATUS "AVX2 requirement satisfied (MSVC /arch:AVX2 detected).")
        endif()
    endif()
    if(NOT GEMMA_HAS_AVX2)
        include(CheckCXXSourceCompiles)
        set(_AVX2_TEST_SRC "#include <immintrin.h>\nint main(){ __m256i a=_mm256_set1_epi32(1); __m256i b=_mm256_set1_epi32(2); __m256i c=_mm256_add_epi32(a,b); (void)c; return 0; }")
        check_cxx_source_compiles("${_AVX2_TEST_SRC}" GEMMA_HAS_AVX2)
    endif()
    if(NOT GEMMA_HAS_AVX2)
        message(FATAL_ERROR "AVX2 required but compile probe failed. Provide /arch:AVX2 (MSVC) or -mavx2 -mfma (GCC/Clang) or set -DGEMMA_REQUIRE_AVX2=OFF.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Output Directory Configuration
# Organize all build outputs into a clean hierarchy under build/
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ----------------------------------------------------------------------------
# Global Strict Mode (treat warnings as errors for project targets)
# Excludes targets whose SOURCE_DIR lives under third_party/ to avoid
# inheriting external project noise. Individual subdirectories (e.g. session)
# may already add /WX or -Werror; this provides a unified toggle.
# ----------------------------------------------------------------------------
option(GEMMA_STRICT "Treat warnings as errors for all in-repo targets (ex third_party)" ON)
if(GEMMA_STRICT)
    message(STATUS "Strict mode: ON (promoting warnings to errors for project targets)")
    # Collect all defined targets so far
    get_property(_ALL_GEMMA_TARGETS GLOBAL PROPERTY TARGETS)
    foreach(_t IN LISTS _ALL_GEMMA_TARGETS)
        # Skip imported / interface / utility targets
        get_target_property(_t_type ${_t} TYPE)
        if(_t_type STREQUAL "INTERFACE_LIBRARY" OR _t_type STREQUAL "UTILITY")
            continue()
        endif()
        # Some generated helper targets have no SOURCE_DIR (skip quietly)
        get_target_property(_t_src_dir ${_t} SOURCE_DIR)
        if(NOT _t_src_dir)
            continue()
        endif()
        # Exclude any target defined under third_party
        string(FIND "${_t_src_dir}" "${CMAKE_SOURCE_DIR}/third_party" _third_party_pos)
        if(_third_party_pos EQUAL 0)
            continue()
        endif()
        # Apply warnings-as-errors
        if(MSVC)
            target_compile_options(${_t} PRIVATE /WX)
        else()
            target_compile_options(${_t} PRIVATE -Werror)
        endif()
    endforeach()
endif()

# For multi-configuration generators (Visual Studio, Xcode), create per-config subdirectories
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin/${OUTPUTCONFIG})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib/${OUTPUTCONFIG})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib/${OUTPUTCONFIG})
endforeach()

message(STATUS "Output directories configured:")
message(STATUS "  Runtime (executables): ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  Libraries (shared): ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "  Archives (static): ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

# ---------------------------------------------------------------------------
# Normalize / support custom (non-standard) build types used by presets.
# CMake recognizes: Debug, Release, RelWithDebInfo, MinSizeRel by default.
# We introduce mapping for 'FastDebug' (O1 + debug info) and 'RelWithSymbols'
# (O2 + debug info) so users get expected optimization levels instead of
# falling back to an unrecognized build type with no flags.
# ---------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES) # Single-config generators only
    if(CMAKE_BUILD_TYPE STREQUAL "FastDebug")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(CMAKE_CXX_FLAGS_FASTDEBUG "/Od /O1 /Zi" CACHE STRING "Flags for FastDebug" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS_FASTDEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING "Linker flags FastDebug" FORCE)
        else()
            set(CMAKE_CXX_FLAGS_FASTDEBUG "-O1 -g" CACHE STRING "Flags for FastDebug" FORCE)
        endif()
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithSymbols")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "/O2 /Zi /DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS_RELWITHSYMBOLS "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}" CACHE STRING "Linker flags RelWithSymbols" FORCE)
        else()
            set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "-O2 -g -DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
        endif()
    endif()
else()
    # Multi-config generator (e.g. Visual Studio): extend configuration list if missing.
    set(_EXTRA_CONFIGS_ADDED FALSE)
    foreach(_cfg IN ITEMS FastDebug RelWithSymbols)
        list(FIND CMAKE_CONFIGURATION_TYPES ${_cfg} _idx)
        if(_idx EQUAL -1)
            list(APPEND CMAKE_CONFIGURATION_TYPES ${_cfg})
            set(_EXTRA_CONFIGS_ADDED TRUE)
        endif()
    endforeach()
    if(_EXTRA_CONFIGS_ADDED)
        list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Supported build configurations" FORCE)
        message(STATUS "Added custom build configurations: FastDebug, RelWithSymbols")
    endif()
    # Define per-config flags if not already defined by toolchain files.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS_FASTDEBUG "/Od /O1 /Zi" CACHE STRING "Flags for FastDebug" FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "/O2 /Zi /DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
    else()
        set(CMAKE_CXX_FLAGS_FASTDEBUG "-O1 -g" CACHE STRING "Flags for FastDebug" FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHSYMBOLS "-O2 -g -DNDEBUG" CACHE STRING "Flags for RelWithSymbols" FORCE)
    endif()
endif()

# ----------------------------------------------------------------------------
# oneAPI root auto-detection (user may supply -DGEMMA_ONEAPI_ROOT= or -Doneapi_root=)
# Propagate as cache variable so subproject backend detection can leverage it.
# ----------------------------------------------------------------------------
if(NOT DEFINED GEMMA_ONEAPI_ROOT OR "${GEMMA_ONEAPI_ROOT}" STREQUAL "")
    if(DEFINED oneapi_root AND EXISTS "${oneapi_root}")
        set(GEMMA_ONEAPI_ROOT "${oneapi_root}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    elseif(DEFINED ENV{GEMMA_ONEAPI_ROOT} AND EXISTS "$ENV{GEMMA_ONEAPI_ROOT}")
        set(GEMMA_ONEAPI_ROOT "$ENV{GEMMA_ONEAPI_ROOT}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}")
        set(GEMMA_ONEAPI_ROOT "$ENV{ONEAPI_ROOT}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    elseif(DEFINED ENV{INTEL_ONEAPI_ROOT} AND EXISTS "$ENV{INTEL_ONEAPI_ROOT}")
        set(GEMMA_ONEAPI_ROOT "$ENV{INTEL_ONEAPI_ROOT}" CACHE PATH "Root path to Intel oneAPI installation" FORCE)
    endif()
endif()
if(GEMMA_ONEAPI_ROOT)
    message(STATUS "Detected oneAPI root: ${GEMMA_ONEAPI_ROOT}")
endif()

# Build options for new components
option(GEMMA_BUILD_MCP_SERVER "Build MCP server component" OFF)
option(GEMMA_BUILD_BACKENDS "Build hardware acceleration backends" OFF)

# Hardware backend options with automatic detection (disabled for CPU-only build)
option(GEMMA_BUILD_SYCL_BACKEND "(Legacy root control) Build SYCL/Intel oneAPI backend" OFF)
option(GEMMA_BUILD_CUDA_BACKEND "(Legacy root control) Build CUDA acceleration backend" OFF)
option(GEMMA_BUILD_OPENCL_BACKEND "(Legacy root control) Build OpenCL acceleration backend" OFF)
option(GEMMA_BUILD_VULKAN_BACKEND "(Legacy root control) Build Vulkan acceleration backend" OFF)
option(GEMMA_BUILD_METAL_BACKEND "(Legacy root control) Build Metal acceleration backend" OFF)
option(GEMMA_USE_LEGACY_BACKEND_BUILD "Use root-level backend detection/add_subdirectory (deprecated). OFF lets gemma.cpp handle backends." OFF)

# Test and documentation options
option(GEMMA_BUILD_ENHANCED_TESTS "Build enhanced test suite" ON)
option(GEMMA_BUILD_BACKEND_TESTS "Build backend-specific tests" ON)
option(GEMMA_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(GEMMA_BUILD_DOCS "Build documentation" OFF)
option(GEMMA_ENABLE_DOXYGEN "Generate Doxygen API docs (implies GEMMA_BUILD_DOCS)" OFF)
option(GEMMA_ENABLE_CLANG_TIDY "Run clang-tidy static analysis during compile (may slow builds)" OFF)
option(GEMMA_ENABLE_SESSION "Enable experimental session management subsystem" OFF)
option(GEMMA_BUILD_SESSION_BENCH "Build session benchmarks (requires GEMMA_ENABLE_SESSION)" OFF)
option(GEMMA_BUILD_ENHANCED_CLI "Build enhanced experimental CLI (depends optionally on session)" OFF)

# Auto-detection options (disabled for CPU-only build)
option(GEMMA_AUTO_DETECT_BACKENDS "Automatically detect available backends" OFF)
option(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS "Print extended backend environment diagnostics during configure" OFF)

# ---------------------------------------------------------------------------
# Backward / cross-tree option compatibility layer
# ---------------------------------------------------------------------------
# The original gemma.cpp subproject introduced GEMMA_ENABLE_AUTO_BACKENDS and
# GEMMA_ENABLE_* backend toggles, while the enhanced root uses
# GEMMA_AUTO_DETECT_BACKENDS + GEMMA_BUILD_BACKENDS. Provide aliasing so that
# older invocation styles continue to work and we can map a single flag to the
# subproject expectations without forcing users to memorize both.

# Map legacy root flag to subproject auto backend discover flag if user passed
# -DGEMMA_AUTO_DETECT_BACKENDS but not the subproject one.
if(DEFINED GEMMA_AUTO_DETECT_BACKENDS AND NOT DEFINED GEMMA_ENABLE_AUTO_BACKENDS)
    set(GEMMA_ENABLE_AUTO_BACKENDS ${GEMMA_AUTO_DETECT_BACKENDS} CACHE BOOL "(alias) Auto-detect available GPU backends" FORCE)
endif()

# If user requested consolidated backend build (GEMMA_BUILD_BACKENDS ON) and
# is NOT forcing legacy root backend path, proactively enable individual
# backend flags so gemma.cpp subproject detection code engages. We only flip
# those not explicitly set by the user to avoid overriding deliberate OFF.
if(GEMMA_BUILD_BACKENDS AND NOT GEMMA_USE_LEGACY_BACKEND_BUILD)
    foreach(_backend IN ITEMS CUDA SYCL OPENCL VULKAN METAL)
        if(NOT DEFINED GEMMA_ENABLE_${_backend})
            # Default to ON; detection code will gracefully disable if toolchain missing.
            set(GEMMA_ENABLE_${_backend} ON CACHE BOOL "(forwarded) Enable ${_backend} backend" FORCE)
        endif()
    endforeach()
    # If user asked for auto-detect (root flag) propagate to subproject flag.
    if(GEMMA_AUTO_DETECT_BACKENDS)
        set(GEMMA_ENABLE_AUTO_BACKENDS ON CACHE BOOL "(alias) Auto-detect and enable available GPU backends" FORCE)
    endif()
    if(GEMMA_ENABLE_AUTO_BACKENDS)
        message(STATUS "Unified backend enable: AUTO_DETECT=ON (request all and let detection prune)")
    else()
        message(STATUS "Unified backend enable: static enable flags set (no auto-detect)")
    endif()
endif()

# Propagate verbose diagnostics flag to subprojects
set(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS ${GEMMA_VERBOSE_BACKEND_DIAGNOSTICS} CACHE BOOL "Verbose backend diagnostics (propagated)" FORCE)

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

# Include centralized dependency management
include(Dependencies)

# Auto-detect available backends if enabled
if(GEMMA_USE_LEGACY_BACKEND_BUILD AND GEMMA_BUILD_BACKENDS)
    message(STATUS "(LEGACY) Root-level auto-detecting hardware backends...")
    # Legacy path retained for compatibility; modern path handled in gemma.cpp/CMakeLists.txt.
    if(GEMMA_AUTO_DETECT_BACKENDS)
        find_package(IntelSYCL QUIET)
        if(IntelSYCL_FOUND OR EXISTS "$ENV{ONEAPI_ROOT}")
            message(STATUS "Intel oneAPI/SYCL detected (legacy root)")
            set(GEMMA_BUILD_SYCL_BACKEND ON CACHE BOOL "Build SYCL backend" FORCE)
        endif()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit detected (legacy root) ${CUDAToolkit_VERSION}")
            set(GEMMA_BUILD_CUDA_BACKEND ON CACHE BOOL "Build CUDA backend" FORCE)
        endif()
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            message(STATUS "Vulkan SDK detected (legacy root) ${Vulkan_VERSION}")
            set(GEMMA_BUILD_VULKAN_BACKEND ON CACHE BOOL "Build Vulkan backend" FORCE)
        endif()
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL detected (legacy root)")
            set(GEMMA_BUILD_OPENCL_BACKEND ON CACHE BOOL "Build OpenCL backend" FORCE)
        endif()
        if(APPLE)
            message(STATUS "Metal backend available on macOS (legacy root)")
            set(GEMMA_BUILD_METAL_BACKEND ON CACHE BOOL "Build Metal backend" FORCE)
        endif()
    endif()
else()
    message(STATUS "Using embedded gemma.cpp backend detection (root legacy backend detection disabled)")
endif()

# Find required packages for enabled backends
# Map legacy root backend toggles to gemma.cpp cache variables (only if legacy path intentionally used)
if(GEMMA_USE_LEGACY_BACKEND_BUILD)
    set(GEMMA_ENABLE_CUDA ${GEMMA_BUILD_CUDA_BACKEND} CACHE BOOL "Forwarded CUDA enable" FORCE)
    set(GEMMA_ENABLE_SYCL ${GEMMA_BUILD_SYCL_BACKEND} CACHE BOOL "Forwarded SYCL enable" FORCE)
    set(GEMMA_ENABLE_VULKAN ${GEMMA_BUILD_VULKAN_BACKEND} CACHE BOOL "Forwarded Vulkan enable" FORCE)
    set(GEMMA_ENABLE_OPENCL ${GEMMA_BUILD_OPENCL_BACKEND} CACHE BOOL "Forwarded OpenCL enable" FORCE)
endif()

# Set policy minimum version to handle legacy dependencies like sentencepiece
# This is required for sentencepiece which has cmake_minimum_required(VERSION 3.1)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ---------------------------------------------------------------------------
# Local third-party dependencies setup
# Detect and configure local dependencies before gemma.cpp subdirectory
# ---------------------------------------------------------------------------
message(STATUS "Configuring local third-party dependencies...")

# Check for local Highway availability and set flags for Dependencies.cmake
# Priority order: highway-github (complete GitHub version) > highway (custom version)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway-github/CMakeLists.txt")
    message(STATUS "Found GitHub Highway in third_party/highway-github (commit 1d16731233de45a365b43867f27d0a5f73925300)")
    # Set flag for Dependencies.cmake to handle the inclusion
    # Force local Highway even if system/vcpkg highway is present so that contrib headers are available
    set(GEMMA_LOCAL_HIGHWAY_PROVIDED ON CACHE BOOL "Highway provided by local third_party (GitHub version - forced)" FORCE)
    set(GEMMA_LOCAL_HIGHWAY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway-github" CACHE PATH "Local Highway source path (forced)" FORCE)
    set(GEMMA_PREFER_SYSTEM_DEPS OFF CACHE BOOL "Disable system/vcpkg preference to ensure local Highway with contrib" FORCE)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway/CMakeLists.txt")
    message(STATUS "Found local Highway in third_party/highway")
    # Set flag for Dependencies.cmake to handle the inclusion
    set(GEMMA_LOCAL_HIGHWAY_PROVIDED ON CACHE BOOL "Highway provided by local third_party" FORCE)
    set(GEMMA_LOCAL_HIGHWAY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/highway" CACHE PATH "Local Highway source path" FORCE)
else()
    message(STATUS "Local Highway not found in third_party/ directories")
    set(GEMMA_LOCAL_HIGHWAY_PROVIDED OFF CACHE BOOL "Highway provided by local third_party" FORCE)
    unset(GEMMA_LOCAL_HIGHWAY_PATH CACHE)
endif()

# Check and add SentencePiece from local third_party directory
# Note: Disabled for CPU-only build to prefer vcpkg packages
if(FALSE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece/CMakeLists.txt")
    message(STATUS "Found local SentencePiece in third_party/sentencepiece")
    # Configure sentencepiece build options before adding
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "Disable shared libraries for sentencepiece")
    set(SPM_ABSL_PROVIDER "module" CACHE STRING "Use module provider for absl")
    set(SPM_BUILD_TEST OFF CACHE BOOL "Disable sentencepiece tests")
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Disable tcmalloc")
    add_subdirectory(third_party/sentencepiece)
    set(GEMMA_LOCAL_SENTENCEPIECE_PROVIDED ON CACHE BOOL "SentencePiece provided by local third_party" FORCE)
else()
    message(STATUS "Using vcpkg SentencePiece instead of local third_party")
    set(GEMMA_LOCAL_SENTENCEPIECE_PROVIDED OFF CACHE BOOL "SentencePiece provided by local third_party" FORCE)
endif()

# Add the original gemma.cpp as a subdirectory
add_subdirectory(gemma.cpp)

# Optional session subsystem (experimental)
if(GEMMA_ENABLE_SESSION)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/session/CMakeLists.txt)
        message(STATUS "Session subsystem enabled")
        add_subdirectory(tools/session)
    else()
        message(WARNING "Session subsystem requested but tools/session not found")
    endif()
endif()

# ---------------------------------------------------------------------------
# Optional clang-tidy integration (kept AFTER subdirectory add so that
# cache variables from toolchains are visible but BEFORE targets built).
# This applies globally; for more granular control we could restrict to
# specific directories later. Keeping logic minimal to avoid side-effects.
# ---------------------------------------------------------------------------
if(GEMMA_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-18 clang-tidy-17)
    if(CLANG_TIDY_EXE)
        message(STATUS "Enabling clang-tidy: ${CLANG_TIDY_EXE}")
        # Quiet some common noise; users can override with -DCLANG_TIDY_CHECKS="..."
        if(NOT DEFINED CLANG_TIDY_CHECKS)
            set(CLANG_TIDY_CHECKS "-*,clang-analyzer-*,performance-*,readability-*,modernize-*" CACHE STRING "clang-tidy checks" FORCE)
        endif()
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=${CLANG_TIDY_CHECKS}" CACHE STRING "clang-tidy command" FORCE)
    else()
        message(WARNING "GEMMA_ENABLE_CLANG_TIDY=ON but clang-tidy not found")
    endif()
endif()

# Add MCP server component if enabled
if(GEMMA_BUILD_MCP_SERVER)
    message(STATUS "Adding MCP server component")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp/CMakeLists.txt)
        add_subdirectory(mcp)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp)
        message(WARNING "MCP directory present but missing CMakeLists.txt; skipping MCP server")
    else()
        message(WARNING "MCP directory not found, skipping MCP server")
    endif()
endif()

# Remove duplicate backend add_subdirectory to avoid target redefinition; handled inside gemma.cpp

# Add enhanced testing framework
if(GEMMA_BUILD_ENHANCED_TESTS)
    message(STATUS "Adding enhanced test suite")
    enable_testing()
    add_subdirectory(tests)
endif()

# Add documentation if enabled
if(GEMMA_BUILD_DOCS OR GEMMA_ENABLE_DOXYGEN)
    message(STATUS "Adding documentation build")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs)
        add_subdirectory(docs)
    else()
        message(WARNING "Docs directory not found, skipping documentation")
    endif()
    if(GEMMA_ENABLE_DOXYGEN)
        find_package(Doxygen QUIET)
        if(DOXYGEN_FOUND)
            set(DOXYGEN_GENERATE_HTML YES)
            set(DOXYGEN_GENERATE_LATEX NO)
            set(DOXYGEN_EXTRACT_PRIVATE NO)
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
            add_custom_target(docs_doxygen
                COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating Doxygen documentation" VERBATIM)
        else()
            message(WARNING "Doxygen requested but not found")
        endif()
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
    if(TARGET gemma_backends)
        add_dependencies(gemma_all gemma_backends)
    endif()
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

# Standard install directories
include(GNUInstallDirs)
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)

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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json/single_include/nlohmann/json.hpp")
    message(STATUS "Vendored nlohmann_json: ON")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite/sqlite3.c")
    message(STATUS "Vendored sqlite3: ON")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Platform: ${GEMMA_PLATFORM}")
if(CMAKE_BUILD_TYPE MATCHES "FastDebug|RelWithSymbols")
    message(STATUS "Using custom build type '${CMAKE_BUILD_TYPE}' (mapped flags applied)")
endif()
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  - Core Gemma.cpp: ON")
message(STATUS "  - MCP Server: ${GEMMA_BUILD_MCP_SERVER}")
message(STATUS "  - Hardware Backends: ${GEMMA_BUILD_BACKENDS}")
message(STATUS "  - Enhanced Tests: ${GEMMA_BUILD_ENHANCED_TESTS}")
message(STATUS "  - Session Subsystem: ${GEMMA_ENABLE_SESSION}")
message(STATUS "  - Session Benchmarks: ${GEMMA_BUILD_SESSION_BENCH}")
message(STATUS "  - Enhanced CLI: ${GEMMA_BUILD_ENHANCED_CLI}")
message(STATUS "  - Backend Tests: ${GEMMA_BUILD_BACKEND_TESTS}")
message(STATUS "  - Benchmarks: ${GEMMA_BUILD_BENCHMARKS}")
message(STATUS "  - Documentation: ${GEMMA_BUILD_DOCS}")
message(STATUS "  - Doxygen: ${GEMMA_ENABLE_DOXYGEN}")
message(STATUS "")
message(STATUS "Hardware Backends:")
message(STATUS "  - SYCL/Intel oneAPI: ${GEMMA_BUILD_SYCL_BACKEND}")
message(STATUS "  - CUDA: ${GEMMA_BUILD_CUDA_BACKEND}")
message(STATUS "  - Vulkan: ${GEMMA_BUILD_VULKAN_BACKEND}")
message(STATUS "  - OpenCL: ${GEMMA_BUILD_OPENCL_BACKEND}")
message(STATUS "  - Metal: ${GEMMA_BUILD_METAL_BACKEND}")
message(STATUS "===============================================")