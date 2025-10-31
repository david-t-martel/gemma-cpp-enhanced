# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Baseline minimum lowered previously; require at least 3.5 but declare
# compatibility up to 3.27 for modern policies when available.
cmake_minimum_required(VERSION 3.5...3.27)

# Handle policy compatibility for dependencies
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# Set policy defaults for FetchContent dependencies
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # option() honors normal variables
set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)  # Include file check macros honor CMAKE_REQUIRED_LIBRARIES

include(FetchContent)

# vcpkg integration handled at project root (before first project()).
# We still allow users to prefer system/vcpkg packages by default.
option(GEMMA_PREFER_SYSTEM_DEPS "Use system/vcpkg packages before FetchContent" ON)
if(GEMMA_USING_VCPKG)
  message(STATUS "[vcpkg] Root toolchain active inside gemma.cpp subproject")
endif()

# Centralized dependency management is handled by root CMakeLists.txt
# Dependencies are resolved in cmake/Dependencies.cmake before this subproject

project(gemma)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include optimization module
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(VcpkgHelper)
include(FindVcpkgPackages)
include(GemmaOptimizations)

# Note: Dependencies resolved by root CMakeLists.txt via cmake/Dependencies.cmake

# Setup global build optimizations
setup_gemma_build_optimizations()
setup_build_profiles()

# Build optimization options
option(GEMMA_ENABLE_UNITY_BUILDS "Enable unity builds for faster compilation" OFF)
option(GEMMA_ENABLE_LTO "Enable Link Time Optimization" ON)
option(GEMMA_ENABLE_PCH "Enable Precompiled Headers" ON)
option(GEMMA_REQUIRE_HWY_CONTRIB "Fail configuration if hwy_contrib (Highway contrib) is not available" OFF)

# Highway dependency should already be resolved by Dependencies.cmake
if(NOT DEFINED GEMMA_HWY_LIBS)
    message(FATAL_ERROR "GEMMA_HWY_LIBS not defined. Dependencies.cmake should have been included first.")
endif()
if(NOT TARGET hwy)
    message(FATAL_ERROR "Highway target 'hwy' not available. Check Dependencies.cmake resolution.")
endif()
message(STATUS "Using Highway libraries: ${GEMMA_HWY_LIBS}")

# SentencePiece dependency should already be resolved by Dependencies.cmake
if(NOT DEFINED GEMMA_SENTENCEPIECE_LIB)
    message(FATAL_ERROR "GEMMA_SENTENCEPIECE_LIB not defined. Dependencies.cmake should have been included first.")
endif()
if(NOT TARGET ${GEMMA_SENTENCEPIECE_LIB})
    message(FATAL_ERROR "SentencePiece target '${GEMMA_SENTENCEPIECE_LIB}' not available. Check Dependencies.cmake resolution.")
endif()
message(STATUS "Using SentencePiece library: ${GEMMA_SENTENCEPIECE_LIB}")

# nlohmann-json dependency should already be resolved by Dependencies.cmake
if(NOT DEFINED GEMMA_JSON_LIB)
    message(FATAL_ERROR "GEMMA_JSON_LIB not defined. Dependencies.cmake should have been included first.")
endif()
if(NOT TARGET ${GEMMA_JSON_LIB})
    message(FATAL_ERROR "nlohmann-json target '${GEMMA_JSON_LIB}' not available. Check Dependencies.cmake resolution.")
endif()
message(STATUS "Using nlohmann-json library: ${GEMMA_JSON_LIB}")

# Google Benchmark dependency handled by Dependencies.cmake (if GEMMA_BUILD_BENCHMARKS=ON)
if(GEMMA_BUILD_BENCHMARKS)
    if(NOT DEFINED GEMMA_BENCHMARK_LIB)
        message(FATAL_ERROR "GEMMA_BENCHMARK_LIB not defined but GEMMA_BUILD_BENCHMARKS=ON. Check Dependencies.cmake.")
    endif()
    if(NOT TARGET ${GEMMA_BENCHMARK_LIB})
        message(FATAL_ERROR "Benchmark target '${GEMMA_BENCHMARK_LIB}' not available. Check Dependencies.cmake resolution.")
    endif()
    message(STATUS "Using Google Benchmark library: ${GEMMA_BENCHMARK_LIB}")
endif()

# ============================================================================
# Package Detection Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Package Detection Summary ===")
vcpkg_package_status("Highway" "hwy")
vcpkg_package_status("SentencePiece" "sentencepiece-static")
vcpkg_package_status("nlohmann-json" "nlohmann_json::nlohmann_json")
vcpkg_package_status("Google Benchmark" "benchmark::benchmark")
message(STATUS "Backend Auto Detect (effective): ${GEMMA_ENABLE_AUTO_BACKENDS}")
message(STATUS "=================================")
message(STATUS "")

# Base source files
set(SOURCES
  compression/compress-inl.h
  compression/compress.cc
  compression/compress.h
  compression/nuq-inl.h
  compression/sfp-inl.h
  compression/types.h
  compression/test_util-inl.h
  evals/benchmark_helper.cc
  evals/benchmark_helper.h
  evals/cross_entropy.cc
  evals/cross_entropy.h
  gemma/activations.h
  gemma/attention.cc
  gemma/attention.h
  gemma/configs.cc
  gemma/configs.h
  gemma/context.cc
  gemma/context.h
  gemma/gemma_args.h
  gemma/gemma-inl.h
  gemma/gemma.cc
  gemma/gemma.h
  gemma/griffin.h
  gemma/kv_cache.cc
  gemma/kv_cache.h
  gemma/model_store.cc
  gemma/model_store.h
  gemma/session.cc
  gemma/session.h
  gemma/tensor_info.cc
  gemma/tensor_info.h
  gemma/tokenizer.cc
  gemma/tokenizer.h
  gemma/vit.cc
  gemma/vit.h
  gemma/weights.cc
  gemma/weights.h
  io/blob_store.cc
  io/blob_store.h
  io/fields.cc
  io/fields.h
  io/io_win.cc
  io/io.cc
  io/io.h
  ops/dot-inl.h
  ops/dry_penalty.cc
  ops/dry_penalty.h
  ops/matmul_static_bf16.cc
  ops/matmul_static_f32.cc
  ops/matmul_static_nuq.cc
  ops/matmul_static_sfp.cc
  ops/matmul-inl.h
  ops/matmul.cc
  ops/matmul.h
  ops/matvec-inl.h
  ops/ops-inl.h
  ops/ops.h
  ops/sum-inl.h
  paligemma/image.cc
  paligemma/image.h
  util/allocator.cc
  util/allocator.h
  util/basics.h
  util/mat.cc
  util/mat.h
  util/test_util.h
  util/threading_context.cc
  util/threading_context.h
  util/threading.cc
  util/threading.h
  util/topology.cc
  util/topology.h
)

# Optionally disable the full Griffin (recurrent) implementation to speed up
# builds or work around platform issues. When disabled, a stub providing the
# GriffinRecurrent symbol is used so other translation units still link.
option(GEMMA_DISABLE_GRIFFIN "Build without full Griffin recurrent layer (use stub)" OFF)
option(GEMMA_EXCLUDE_ARCHIVE "Exclude .archive directory from packaging/install" ON)
option(GEMMA_ENABLE_MCP "Build optional MCP server (requires mcp directory)" OFF)
# Backend options (disabled for CPU-only build)
option(GEMMA_ENABLE_CUDA "Enable CUDA backend if available" OFF)
option(GEMMA_ENABLE_OPENCL "Enable OpenCL backend if available" OFF)
option(GEMMA_ENABLE_SYCL "Enable SYCL backend if available" OFF)
option(GEMMA_ENABLE_VULKAN "Enable Vulkan backend if available" OFF)
option(GEMMA_ENABLE_AUTO_BACKENDS "Auto-detect and enable available GPU backends" OFF)

# Honor root alias GEMMA_AUTO_DETECT_BACKENDS if provided and subproject flag left default.
if(DEFINED GEMMA_AUTO_DETECT_BACKENDS AND NOT DEFINED CACHE{GEMMA_ENABLE_AUTO_BACKENDS})
  # Only adopt if user did not explicitly set GEMMA_ENABLE_AUTO_BACKENDS on command line.
  if(NOT GEMMA_ENABLE_AUTO_BACKENDS AND GEMMA_AUTO_DETECT_BACKENDS)
    set(GEMMA_ENABLE_AUTO_BACKENDS ON CACHE BOOL "(alias) Auto-detect and enable available GPU backends" FORCE)
  endif()
endif()
option(GEMMA_FORCE_INTEL_COMPILER_FOR_SYCL "Force Intel compiler (icpx) for SYCL compilation when detected" ON)

if(GEMMA_DISABLE_GRIFFIN)
  message(STATUS "GEMMA_DISABLE_GRIFFIN=ON: using gemma/griffin_stub.cc (no recurrent layer logic)")
  list(APPEND SOURCES gemma/griffin_stub.cc)
else()
  list(APPEND SOURCES gemma/griffin.cc)
endif()

# Add C API sources only when building DLL
if(BUILD_GEMMA_DLL)
  list(APPEND SOURCES
    gemma/bindings/context.h
    gemma/bindings/context.cc
    gemma/bindings/c_api.h
    gemma/bindings/c_api.cc
  )
  message(STATUS "Including C API files for DLL build")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()


## Library Target

add_library(libgemma ${SOURCES})
set_property(TARGET libgemma PROPERTY CXX_STANDARD 20)
set_target_properties(libgemma PROPERTIES PREFIX "")
set_property(TARGET libgemma PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(libgemma PUBLIC ./ ./gemma)
# GEMMA_SENTENCEPIECE_LIB should already be set by Dependencies.cmake
if(NOT DEFINED GEMMA_SENTENCEPIECE_LIB)
    message(FATAL_ERROR "GEMMA_SENTENCEPIECE_LIB not defined. Dependencies.cmake should have been included first.")
endif()

# Note: Threads already found by root CMakeLists.txt at line 70
target_link_libraries(libgemma ${GEMMA_HWY_LIBS} ${GEMMA_SENTENCEPIECE_LIB} ${GEMMA_JSON_LIB} Threads::Threads ${GEMMA_BENCHMARK_LIB} ${GEMMA_GTEST_LIBS} ${GEMMA_BACKEND_LIBS})
target_include_directories(libgemma PUBLIC ${sentencepiece_SOURCE_DIR})
target_compile_definitions(libgemma PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(libgemma PRIVATE $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:Clang>>:-Wno-deprecated-declarations>)

# ----------------------------------------------------------------------------
# Backend detection & integration (initial focus: SYCL/oneAPI)
# ----------------------------------------------------------------------------
set(GEMMA_BACKEND_LIBS "")

if(GEMMA_ENABLE_AUTO_BACKENDS)
  # If auto enabled, turn on specific backends unless explicitly OFF
  if(NOT DEFINED GEMMA_ENABLE_CUDA)
    set(GEMMA_ENABLE_CUDA ON)
  endif()
  if(NOT DEFINED GEMMA_ENABLE_SYCL)
    set(GEMMA_ENABLE_SYCL ON)
  endif()
  if(NOT DEFINED GEMMA_ENABLE_OPENCL)
    set(GEMMA_ENABLE_OPENCL ON)
  endif()
  if(NOT DEFINED GEMMA_ENABLE_VULKAN)
    set(GEMMA_ENABLE_VULKAN OFF) # heavier dependency; default off
  endif()
endif()

if(GEMMA_ENABLE_SYCL)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
  find_package(IntelSYCL QUIET)

  # Determine candidate oneAPI roots (priority order):
  set(_ONEAPI_ROOT_CANDIDATES "")
  if(GEMMA_ONEAPI_ROOT)
    list(APPEND _ONEAPI_ROOT_CANDIDATES "${GEMMA_ONEAPI_ROOT}")
  endif()
  if(DEFINED oneapi_root AND NOT oneapi_root STREQUAL "")
    list(APPEND _ONEAPI_ROOT_CANDIDATES "${oneapi_root}")
  endif()
  foreach(_env IN ITEMS GEMMA_ONEAPI_ROOT ONEAPI_ROOT INTEL_ONEAPI_ROOT)
    if(DEFINED ENV{${_env}})
      list(APPEND _ONEAPI_ROOT_CANDIDATES "$ENV{${_env}}")
    endif()
  endforeach()
  # Fallback conventional install path
  list(APPEND _ONEAPI_ROOT_CANDIDATES "C:/Program Files (x86)/Intel/oneAPI")

  list(REMOVE_DUPLICATES _ONEAPI_ROOT_CANDIDATES)

  if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
    message(STATUS "[SYCL][Diag] Candidate oneAPI roots (pre-scan):")
    foreach(_cand ${_ONEAPI_ROOT_CANDIDATES})
      message(STATUS "  - ${_cand}")
    endforeach()
    message(STATUS "[SYCL][Diag] Env snapshot: ONEAPI_ROOT='$ENV{ONEAPI_ROOT}' INTEL_ONEAPI_ROOT='$ENV{INTEL_ONEAPI_ROOT}' GEMMA_ONEAPI_ROOT='${GEMMA_ONEAPI_ROOT}'")
  endif()

  # Manual fallback detection on Windows only sets IntelSYCL_FOUND if ALL components are located
  if(NOT IntelSYCL_FOUND AND WIN32)
    foreach(_cand_root ${_ONEAPI_ROOT_CANDIDATES})
      if(NOT EXISTS "${_cand_root}")
        if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
          message(STATUS "[SYCL][Diag] Skipping non-existent root: ${_cand_root}")
        endif()
        continue()
      endif()
      set(_oneapi_compiler_root "${_cand_root}/compiler")
      if(NOT EXISTS "${_oneapi_compiler_root}")
        if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
          message(STATUS "[SYCL][Diag] Missing compiler dir under: ${_cand_root}")
        endif()
        continue()
      endif()
      file(GLOB _oneapi_version_dirs RELATIVE "${_oneapi_compiler_root}" "${_oneapi_compiler_root}/*")
      set(_version_list "")
      foreach(d ${_oneapi_version_dirs})
        if(d STREQUAL "latest")
          list(APPEND _version_list ${d})
        elseif(d MATCHES "^[0-9]+\\.[0-9]+")
          list(APPEND _version_list ${d})
        endif()
      endforeach()
      set(_sorted_versions "")
      foreach(v ${_version_list})
        if(v STREQUAL "latest")
          list(INSERT _sorted_versions 0 ${v})
        else()
          list(APPEND _sorted_versions ${v})
        endif()
      endforeach()
      foreach(v ${_sorted_versions})
        if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
          message(STATUS "[SYCL][Diag] Scanning version '${v}' under ${_cand_root}")
        endif()
        set(_bin_paths
          "${_oneapi_compiler_root}/${v}/windows/bin"
          "${_oneapi_compiler_root}/${v}/bin"
        )
        set(_include_paths
          "${_oneapi_compiler_root}/${v}/windows/include"
          "${_oneapi_compiler_root}/${v}/include"
        )
        set(_lib_paths
          "${_oneapi_compiler_root}/${v}/windows/lib"
          "${_oneapi_compiler_root}/${v}/lib"
        )
        foreach(_bp ${_bin_paths})
          if(NOT IntelSYCL_COMPILER)
            foreach(_cand icx.exe icpx.exe)
              if(EXISTS "${_bp}/${_cand}")
                set(IntelSYCL_COMPILER "${_bp}/${_cand}")
                break()
              endif()
            endforeach()
          endif()
        endforeach()
        foreach(_ip ${_include_paths})
          if(NOT IntelSYCL_INCLUDE_DIR AND EXISTS "${_ip}/sycl/sycl.hpp")
            set(IntelSYCL_INCLUDE_DIR "${_ip}")
          endif()
        endforeach()
        foreach(_lp ${_lib_paths})
          if(NOT IntelSYCL_LIBRARY)
            file(GLOB _sycl_libs LIST_DIRECTORIES FALSE "${_lp}/sycl*.lib")
            if(_sycl_libs)
              set(_chosen "")
              foreach(_lib ${_sycl_libs})
                if(_lib MATCHES "sycl-devicelib-host")
                  if(NOT _chosen)
                    set(_chosen ${_lib})
                  endif()
                else()
                  set(_chosen ${_lib})
                  break()
                endif()
              endforeach()
              if(_chosen)
                set(IntelSYCL_LIBRARY "${_chosen}")
              endif()
            endif()
          endif()
        endforeach()
        if(IntelSYCL_COMPILER AND IntelSYCL_INCLUDE_DIR AND IntelSYCL_LIBRARY)
          # Infer INTEL_ONEAPI_ROOT if not set yet
          if(NOT INTEL_ONEAPI_ROOT)
            get_filename_component(_comp_dir ${IntelSYCL_COMPILER} DIRECTORY)
            get_filename_component(INTEL_ONEAPI_ROOT ${_comp_dir}/../../.. ABSOLUTE)
          endif()
          set(IntelSYCL_FOUND TRUE)
          set(GEMMA_ONEAPI_ROOT "${INTEL_ONEAPI_ROOT}" CACHE PATH "Detected oneAPI root (propagated)" FORCE)
          message(STATUS "Intel oneAPI SYCL fallback success from root '${_cand_root}':\n  Compiler: ${IntelSYCL_COMPILER}\n  Include : ${IntelSYCL_INCLUDE_DIR}\n  Library : ${IntelSYCL_LIBRARY}")
          if(IntelSYCL_LIBRARY MATCHES "sycl-devicelib-host")
            message(WARNING "Using device library variant (${IntelSYCL_LIBRARY}); linking may require full SYCL import library in some configurations")
          endif()

          # Note: Compiler is set per-target for SYCL backend (see lines 453-458)
          # Global CMAKE_CXX_COMPILER override removed to avoid dangerous mid-configuration changes

          break()
        endif()
      endforeach()
      if(IntelSYCL_FOUND)
        break()
      endif()
    endforeach()
    if(NOT IntelSYCL_FOUND)
      message(STATUS "SYCL fallback: unable to locate complete toolchain in candidates: ${_ONEAPI_ROOT_CANDIDATES}")
      if(NOT IntelSYCL_COMPILER)
        message(STATUS "  Missing compiler (icx/icpx)")
      endif()
      if(NOT IntelSYCL_INCLUDE_DIR)
        message(STATUS "  Missing headers (sycl/sycl.hpp)")
      endif()
      if(NOT IntelSYCL_LIBRARY)
        message(STATUS "  Missing import library (sycl*.lib)")
      endif()
      if(GEMMA_VERBOSE_BACKEND_DIAGNOSTICS)
        message(STATUS "[SYCL][Diag] Fallback search exhausted without success.")
      endif()
    endif()
  endif()

  if(IntelSYCL_FOUND)
    message(STATUS "Enabling SYCL backend")

    # Pass Intel compiler information to SYCL backend
    set(GEMMA_SYCL_COMPILER "${IntelSYCL_COMPILER}" CACHE INTERNAL "Intel SYCL compiler path")
    set(GEMMA_SYCL_INCLUDE_DIR "${IntelSYCL_INCLUDE_DIR}" CACHE INTERNAL "Intel SYCL include directory")
    set(GEMMA_SYCL_LIBRARY "${IntelSYCL_LIBRARY}" CACHE INTERNAL "Intel SYCL library")

    # Save and remove /MP and other compile options before adding SYCL subdirectory
    # SYCL offloading is incompatible with /MP multi-processor compilation
    get_directory_property(_SAVED_COMPILE_OPTIONS COMPILE_OPTIONS)
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "")
    message(STATUS "SYCL build: Cleared global COMPILE_OPTIONS (including /MP, incompatible with -fsycl)")
    
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../backends/sycl ${CMAKE_BINARY_DIR}/backends_sycl)
    
    # Restore compile options after SYCL subdirectory
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_SAVED_COMPILE_OPTIONS}")
    if(TARGET gemma_sycl_backend)
      list(APPEND GEMMA_BACKEND_LIBS gemma_sycl_backend)
      target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_SYCL)

      # Force Intel compiler for SYCL backend target
      if(IntelSYCL_COMPILER AND EXISTS "${IntelSYCL_COMPILER}")
        set_target_properties(gemma_sycl_backend PROPERTIES
          CXX_COMPILER "${IntelSYCL_COMPILER}"
        )
        message(STATUS "SYCL backend configured to use Intel compiler: ${IntelSYCL_COMPILER}")
      endif()
    else()
      message(WARNING "SYCL backend sources did not produce target 'gemma_sycl_backend'")
    endif()
  else()
    message(STATUS "Intel SYCL not fully detected; disabling SYCL backend")
  endif()
endif()

if(GEMMA_ENABLE_CUDA)
  enable_language(CXX) # ensure
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found: enabling CUDA backend")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../backends/cuda ${CMAKE_BINARY_DIR}/backends_cuda)
    if(TARGET gemma_cuda_backend)
      list(APPEND GEMMA_BACKEND_LIBS gemma_cuda_backend)
      target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_CUDA)
    endif()
  else()
    message(STATUS "CUDA toolkit not found; CUDA backend disabled")
  endif()
endif()

if(GEMMA_ENABLE_VULKAN)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../backends/vulkan ${CMAKE_BINARY_DIR}/backends_vulkan)
    if(TARGET gemma_vulkan_backend)
      list(APPEND GEMMA_BACKEND_LIBS gemma_vulkan_backend)
      target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_VULKAN)
    endif()
  else()
    message(STATUS "Vulkan SDK not found; Vulkan backend disabled")
  endif()
endif()

if(GEMMA_ENABLE_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    # Placeholder: opencl backend directory currently empty; skip add_subdirectory
    message(STATUS "OpenCL found (backend sources missing or not yet implemented)")
    target_compile_definitions(libgemma PUBLIC GEMMA_ENABLE_OPENCL)
  else()
    message(STATUS "OpenCL not found; OpenCL backend disabled")
  endif()
endif()

# Apply comprehensive optimizations
apply_gemma_optimizations(libgemma)

# ============================================================================
# Intel oneAPI Framework Library Integration (Optional)
# ============================================================================
if(GEMMA_USE_ONEAPI_LIBS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OneAPILibs.cmake)
endif()

# Set proper output directories for library
set_target_properties(libgemma PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

# Install library target
install(TARGETS libgemma
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Development
)

# Optionally skip archived sources (they are not in target, but avoid packaging docs)
if(GEMMA_EXCLUDE_ARCHIVE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.archive)
  install(CODE "message(STATUS \"Skipping installation of archived sources\")")
endif()

# Shared library target for C# interop
if(BUILD_GEMMA_DLL)
    add_library(gemma_shared SHARED ${SOURCES})
set_property(TARGET gemma_shared PROPERTY CXX_STANDARD 20)
set_target_properties(gemma_shared PROPERTIES 
    PREFIX ""
    OUTPUT_NAME "gemma"
)
set_property(TARGET gemma_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(gemma_shared PUBLIC ./ ./gemma)
target_link_libraries(gemma_shared PRIVATE 
  $<LINK_LIBRARY:WHOLE_ARCHIVE,hwy>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,hwy_contrib>
  $<LINK_LIBRARY:WHOLE_ARCHIVE,${GEMMA_SENTENCEPIECE_LIB}>
  Threads::Threads
  ${GEMMA_JSON_LIB}
  ${GEMMA_BENCHMARK_LIB}
  ${GEMMA_GTEST_LIBS}
  ${GEMMA_BACKEND_LIBS}
)
target_include_directories(gemma_shared PUBLIC ${sentencepiece_SOURCE_DIR})
target_compile_definitions(gemma_shared 
    PRIVATE 
    GEMMA_EXPORTS
    $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>
)
target_compile_options(gemma_shared PRIVATE $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:Clang>>:-Wno-deprecated-declarations>)

# Set proper output directories for shared library
set_target_properties(gemma_shared PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
)

# Install shared library and headers
install(TARGETS gemma_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)
install(FILES gemma/c_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemma)
install(FILES gemma/GemmaInterop.cs DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gemma)
endif()

# Executable Target

add_executable(gemma run.cc)
target_link_libraries(gemma libgemma ${GEMMA_HWY_LIBS} ${GEMMA_SENTENCEPIECE_LIB} ${GEMMA_BACKEND_LIBS} Threads::Threads ${GEMMA_JSON_LIB} ${GEMMA_BENCHMARK_LIB} ${GEMMA_GTEST_LIBS})
apply_gemma_optimizations(gemma)

# Determine versioned output name based on enabled backends and oneAPI libs
set(_GEMMA_OUTPUT_NAME "gemma")
set(_BACKEND_SUFFIX "")
set(_ONEAPI_SUFFIX "")

# Build backend suffix
if(GEMMA_BACKEND_LIBS)
    set(_ACTIVE_BACKENDS "")
    
    if("gemma_sycl_backend" IN_LIST GEMMA_BACKEND_LIBS)
        list(APPEND _ACTIVE_BACKENDS "sycl")
    endif()
    if("gemma_cuda_backend" IN_LIST GEMMA_BACKEND_LIBS)
        list(APPEND _ACTIVE_BACKENDS "cuda")
    endif()
    if("gemma_vulkan_backend" IN_LIST GEMMA_BACKEND_LIBS)
        list(APPEND _ACTIVE_BACKENDS "vulkan")
    endif()
    if("gemma_opencl_backend" IN_LIST GEMMA_BACKEND_LIBS)
        list(APPEND _ACTIVE_BACKENDS "opencl")
    endif()
    if("gemma_metal_backend" IN_LIST GEMMA_BACKEND_LIBS)
        list(APPEND _ACTIVE_BACKENDS "metal")
    endif()
    
    if(_ACTIVE_BACKENDS)
        string(REPLACE ";" "-" _BACKEND_SUFFIX "${_ACTIVE_BACKENDS}")
    endif()
endif()

# Build oneAPI library suffix
if(GEMMA_ONEAPI_ENABLED_LIBS)
    string(REPLACE ";" "-" _ONEAPI_SUFFIX "${GEMMA_ONEAPI_ENABLED_LIBS}")
endif()

# Construct final executable name
if(_BACKEND_SUFFIX AND _ONEAPI_SUFFIX)
    set(_GEMMA_OUTPUT_NAME "gemma_hw-${_BACKEND_SUFFIX}+${_ONEAPI_SUFFIX}")
    message(STATUS "Executable: ${_GEMMA_OUTPUT_NAME}.exe (backends: ${_BACKEND_SUFFIX}, oneAPI: ${_ONEAPI_SUFFIX})")
elseif(_BACKEND_SUFFIX)
    set(_GEMMA_OUTPUT_NAME "gemma_hw-${_BACKEND_SUFFIX}")
    message(STATUS "Executable: ${_GEMMA_OUTPUT_NAME}.exe (backends: ${_BACKEND_SUFFIX})")
elseif(_ONEAPI_SUFFIX)
    set(_GEMMA_OUTPUT_NAME "gemma_std+${_ONEAPI_SUFFIX}")
    message(STATUS "Executable: ${_GEMMA_OUTPUT_NAME}.exe (oneAPI: ${_ONEAPI_SUFFIX})")
else()
    set(_GEMMA_OUTPUT_NAME "gemma_std")
    message(STATUS "Executable: ${_GEMMA_OUTPUT_NAME}.exe (standard CPU-only)")
endif()

# Ensure gemma executable goes to proper output directory with versioned name
set_target_properties(gemma PROPERTIES
    OUTPUT_NAME "${_GEMMA_OUTPUT_NAME}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Install target
install(TARGETS gemma
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)

add_executable(single_benchmark evals/benchmark.cc)
target_link_libraries(single_benchmark libgemma ${GEMMA_HWY_LIBS} ${GEMMA_JSON_LIB} ${GEMMA_BACKEND_LIBS} Threads::Threads ${GEMMA_BENCHMARK_LIB} ${GEMMA_GTEST_LIBS})
set_target_properties(single_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

add_executable(benchmarks evals/benchmarks.cc)
if(GEMMA_BUILD_BENCHMARKS AND DEFINED GEMMA_BENCHMARK_LIB)
    target_link_libraries(benchmarks libgemma ${GEMMA_HWY_LIBS} ${GEMMA_JSON_LIB} ${GEMMA_BENCHMARK_LIB} ${GEMMA_BACKEND_LIBS} Threads::Threads ${GEMMA_GTEST_LIBS})
else()
    # Fallback for when benchmarks are built without proper dependency resolution
    target_link_libraries(benchmarks libgemma ${GEMMA_HWY_LIBS} ${GEMMA_JSON_LIB} ${GEMMA_BACKEND_LIBS} Threads::Threads ${GEMMA_BENCHMARK_LIB} ${GEMMA_GTEST_LIBS})
endif()
set_target_properties(benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

add_executable(debug_prompt evals/debug_prompt.cc)
target_link_libraries(debug_prompt libgemma ${GEMMA_HWY_LIBS} ${GEMMA_JSON_LIB} ${GEMMA_BACKEND_LIBS} Threads::Threads ${GEMMA_BENCHMARK_LIB} ${GEMMA_GTEST_LIBS})
set_target_properties(debug_prompt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

## Tests
set(GEMMA_ENABLE_TESTS OFF CACHE BOOL "Enable Gemma tests")
if (GEMMA_ENABLE_TESTS)

enable_testing()
include(GoogleTest)

set(GEMMA_TEST_FILES
  compression/compress_test.cc
  compression/distortion_test.cc
  compression/nuq_test.cc
  compression/sfp_test.cc
  evals/gemma_test.cc
  gemma/tensor_info_test.cc
  io/blob_store_test.cc
  io/fields_test.cc
  ops/bench_matmul.cc
  ops/dot_test.cc
  ops/gemma_matvec_test.cc
  ops/matmul_test.cc
  ops/ops_test.cc
  paligemma/image_test.cc
  paligemma/paligemma_test.cc
  util/threading_test.cc
)

foreach (TESTFILE IN LISTS GEMMA_TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE})
  # Test all targets, not just the best/baseline. This changes the default
  # policy to all-attainable; note that setting -DHWY_COMPILE_* directly can
  # cause compile errors because only one may be set, and other CMakeLists.txt
  # that include us may set them.
  target_compile_options(${TESTNAME} PRIVATE -DHWY_IS_TEST=1)

  target_link_libraries(${TESTNAME} PRIVATE libgemma GTest::Main ${GEMMA_HWY_LIBS} hwy_test Threads::Threads ${GEMMA_JSON_LIB} ${GEMMA_BENCHMARK_LIB} ${GEMMA_GTEST_LIBS} ${GEMMA_BACKEND_LIBS})

  gtest_discover_tests(${TESTNAME})
endforeach ()

add_executable(gemma_batch_bench evals/gemma_batch_bench.cc)
if(GEMMA_ENABLE_TESTS AND DEFINED GEMMA_GTEST_LIBS)
    list(GET GEMMA_GTEST_LIBS 1 GEMMA_GTEST_MAIN)  # gtest_main is second element
    target_link_libraries(gemma_batch_bench libgemma ${GEMMA_GTEST_MAIN} ${GEMMA_HWY_LIBS} ${GEMMA_JSON_LIB} Threads::Threads ${GEMMA_BENCHMARK_LIB} ${GEMMA_BACKEND_LIBS})
else()
    # Fallback for when tests are built without proper dependency resolution
    target_link_libraries(gemma_batch_bench libgemma ${GEMMA_HWY_LIBS} ${GEMMA_JSON_LIB} Threads::Threads ${GEMMA_BENCHMARK_LIB} ${GEMMA_BACKEND_LIBS})
endif()

endif()  # GEMMA_ENABLE_TESTS

## Enhanced Testing Framework
# Option to enable the comprehensive test suite
option(GEMMA_ENABLE_ENHANCED_TESTS "Enable enhanced testing framework with unit, integration, and performance tests" OFF)

if(GEMMA_ENABLE_ENHANCED_TESTS)
    message(STATUS "Enhanced testing framework enabled")
    add_subdirectory(../tests tests_build)
endif()

## Tools

add_executable(migrate_weights io/migrate_weights.cc)
target_link_libraries(migrate_weights libgemma ${GEMMA_HWY_LIBS} Threads::Threads)
set_target_properties(migrate_weights PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Install tools
install(TARGETS migrate_weights
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Tools
)

# Optional MCP server component
if(GEMMA_ENABLE_MCP)
  if(EXISTS "${CMAKE_SOURCE_DIR}/mcp/CMakeLists.txt")
    message(STATUS "Configuring MCP server component")
    add_subdirectory(${CMAKE_SOURCE_DIR}/mcp mcp_build)
  else()
    message(WARNING "GEMMA_ENABLE_MCP=ON but mcp/ directory missing; skipping")
  endif()
endif()

# ----------------------------------------------------------------------------
# Build Summary Banner
# ----------------------------------------------------------------------------
message(STATUS "\n========== Gemma Build Configuration ==========")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System Deps Preferred: ${GEMMA_PREFER_SYSTEM_DEPS}")
message(STATUS "Griffin Disabled: ${GEMMA_DISABLE_GRIFFIN}")
message(STATUS "Archive Excluded: ${GEMMA_EXCLUDE_ARCHIVE}")
message(STATUS "Auto Backend Detect: ${GEMMA_ENABLE_AUTO_BACKENDS}")
message(STATUS "SYCL Enabled (requested): ${GEMMA_ENABLE_SYCL}")
if(IntelSYCL_FOUND)
  message(STATUS "SYCL Compiler: ${IntelSYCL_COMPILER}")
  message(STATUS "SYCL Force Intel Compiler: ${GEMMA_FORCE_INTEL_COMPILER_FOR_SYCL}")
endif()
message(STATUS "CUDA Enabled (requested): ${GEMMA_ENABLE_CUDA}")
message(STATUS "Vulkan Enabled (requested): ${GEMMA_ENABLE_VULKAN}")
message(STATUS "OpenCL Enabled (requested): ${GEMMA_ENABLE_OPENCL}")
if(GEMMA_BACKEND_LIBS)
  string(REPLACE ";" ", " _BACKENDS_LIST "${GEMMA_BACKEND_LIBS}")
  message(STATUS "Active Backend Libraries: ${_BACKENDS_LIST}")
else()
  message(STATUS "Active Backend Libraries: (none)")
endif()
message(STATUS "===============================================\n")

# Print effective compiler flags for visibility (single-config generators)
if(NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Compiler Flags (effective):")
  message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt_uc)
  message(STATUS "  CMAKE_CXX_FLAGS_${_bt_uc}: ${CMAKE_CXX_FLAGS_${_bt_uc}}")
else()
  message(STATUS "Multi-config generator detected; key config flags:")
  foreach(_cfg IN ITEMS Debug Release RelWithDebInfo RelWithSymbols FastDebug MinSizeRel)
    string(TOUPPER "${_cfg}" _ucfg)
    if(DEFINED CMAKE_CXX_FLAGS_${_ucfg})
      message(STATUS "  ${_cfg}: ${CMAKE_CXX_FLAGS_${_ucfg}}")
    endif()
  endforeach()
endif()

# Include dependency summary module (created if not present at top-level)
set(_DEP_SUMMARY_MODULE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DependencySummary.cmake")
if(EXISTS ${_DEP_SUMMARY_MODULE})
  include(${_DEP_SUMMARY_MODULE})
  gemma_print_dependency_summary()
else()
  message(STATUS "(DependencySummary.cmake missing; skipping dependency provenance block)")
endif()
