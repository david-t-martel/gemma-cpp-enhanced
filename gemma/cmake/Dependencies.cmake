# Dependencies.cmake
# Centralized dependency management for Gemma.cpp Enhanced
# Priority: GitHub Highway (local) > vcpkg > FetchContent fallback

include(FetchContent)

# Set GEMMA_PREFER_SYSTEM_DEPS to ON by default if vcpkg is active
if(GEMMA_USING_VCPKG AND NOT DEFINED GEMMA_PREFER_SYSTEM_DEPS)
    set(GEMMA_PREFER_SYSTEM_DEPS ON CACHE BOOL "Use system/vcpkg packages before FetchContent")
endif()

# =============================================================================
# Dependency Resolution Functions
# =============================================================================

function(report_dependency_status name target_name status source)
    message(STATUS "[Deps] ${name}: ${status} (${source})")
endfunction()

function(ensure_highway_dependency)
    # Handle local Highway first (if available)
    if(GEMMA_LOCAL_HIGHWAY_PROVIDED AND DEFINED GEMMA_LOCAL_HIGHWAY_PATH)
        if(NOT TARGET hwy)
            message(STATUS "Adding local Highway from ${GEMMA_LOCAL_HIGHWAY_PATH}")
            add_subdirectory("${GEMMA_LOCAL_HIGHWAY_PATH}" "${CMAKE_BINARY_DIR}/highway_local")
        endif()
        if(TARGET hwy)
            report_dependency_status("Highway" "hwy" "FOUND" "local third_party")
            set(GEMMA_HWY_LIBS hwy PARENT_SCOPE)
            if(TARGET hwy_contrib)
                set(GEMMA_HWY_LIBS hwy hwy_contrib PARENT_SCOPE)
            endif()
            return()
        endif()
    endif()

    if(GEMMA_PREFER_SYSTEM_DEPS)
        find_package(hwy QUIET CONFIG)
        if(TARGET hwy::hwy)
            # vcpkg provides hwy::hwy
            report_dependency_status("Highway" "hwy::hwy" "FOUND" "vcpkg/cmake-config")
            # Create alias for compatibility with gemma.cpp expectations
            if(NOT TARGET hwy)
                add_library(hwy ALIAS hwy::hwy)
            endif()
            set(GEMMA_HWY_LIBS hwy::hwy PARENT_SCOPE)
            return()
        elseif(TARGET hwy)
            # Local or other provides hwy
            report_dependency_status("Highway" "hwy" "FOUND" "vcpkg/system")
            set(GEMMA_HWY_LIBS hwy PARENT_SCOPE)
            if(TARGET hwy_contrib)
                set(GEMMA_HWY_LIBS hwy hwy_contrib PARENT_SCOPE)
            endif()
            return()
        endif()
    endif()

    # FetchContent fallback
    report_dependency_status("Highway" "hwy" "FETCHING" "GitHub FetchContent")
    FetchContent_Declare(
        highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG 1d16731233de45a365b43867f27d0a5f73925300
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(highway)
    set(GEMMA_HWY_LIBS hwy PARENT_SCOPE)
    if(TARGET hwy_contrib)
        set(GEMMA_HWY_LIBS hwy hwy_contrib PARENT_SCOPE)
    endif()
endfunction()

function(ensure_sentencepiece_dependency)
    # Set build options before any discovery
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "Disable shared libraries for sentencepiece")
    set(SPM_ABSL_PROVIDER "module" CACHE STRING "Use module provider for absl")
    set(SPM_BUILD_TEST OFF CACHE BOOL "Disable sentencepiece tests")
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Disable tcmalloc")

    if(GEMMA_LOCAL_SENTENCEPIECE_PROVIDED AND TARGET sentencepiece)
        report_dependency_status("SentencePiece" "sentencepiece" "FOUND" "local third_party")
        if(NOT TARGET sentencepiece-static)
            add_library(sentencepiece-static ALIAS sentencepiece)
        endif()
        set(GEMMA_SENTENCEPIECE_LIB sentencepiece-static PARENT_SCOPE)
        return()
    endif()

    if(GEMMA_PREFER_SYSTEM_DEPS)
        # Try to find SentencePiece via CMake config first (vcpkg provides this)
        find_package(PkgConfig QUIET)
        find_package(sentencepiece QUIET CONFIG)
        if(TARGET sentencepiece::sentencepiece)
            add_library(sentencepiece-static ALIAS sentencepiece::sentencepiece)
            report_dependency_status("SentencePiece" "sentencepiece-static" "FOUND" "vcpkg/cmake-config")
            set(GEMMA_SENTENCEPIECE_LIB sentencepiece-static PARENT_SCOPE)
            return()
        elseif(PkgConfig_FOUND)
            pkg_check_modules(SENTENCEPIECE QUIET sentencepiece)
            if(SENTENCEPIECE_FOUND)
                add_library(sentencepiece-vcpkg INTERFACE)
                target_link_libraries(sentencepiece-vcpkg INTERFACE ${SENTENCEPIECE_LIBRARIES})
                target_include_directories(sentencepiece-vcpkg INTERFACE ${SENTENCEPIECE_INCLUDE_DIRS})
                add_library(sentencepiece-static ALIAS sentencepiece-vcpkg)
                report_dependency_status("SentencePiece" "sentencepiece-static" "FOUND" "vcpkg/pkg-config")
                set(GEMMA_SENTENCEPIECE_LIB sentencepiece-static PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # FetchContent fallback
    report_dependency_status("SentencePiece" "sentencepiece-static" "FETCHING" "GitHub FetchContent")
    FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece
        GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        PATCH_COMMAND ${CMAKE_COMMAND} -E echo "Patching sentencepiece CMakeLists.txt for compatibility" &&
                      ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_sentencepiece.cmake" <SOURCE_DIR>
    )
    FetchContent_MakeAvailable(sentencepiece)
    set(GEMMA_SENTENCEPIECE_LIB sentencepiece-static PARENT_SCOPE)
endfunction()

function(ensure_nlohmann_json_dependency)
    # Check local third_party first
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include/nlohmann/json.hpp")
        if(NOT TARGET nlohmann_json_local)
            add_library(nlohmann_json_local INTERFACE)
            target_include_directories(nlohmann_json_local INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include")
            add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_local)
        endif()
        report_dependency_status("nlohmann-json" "nlohmann_json::nlohmann_json" "FOUND" "local third_party")
        set(GEMMA_JSON_LIB nlohmann_json::nlohmann_json PARENT_SCOPE)
        return()
    endif()
    
    if(GEMMA_PREFER_SYSTEM_DEPS)
        find_package(nlohmann_json QUIET CONFIG)
        if(TARGET nlohmann_json::nlohmann_json)
            report_dependency_status("nlohmann-json" "nlohmann_json::nlohmann_json" "FOUND" "vcpkg/system")
            set(GEMMA_JSON_LIB nlohmann_json::nlohmann_json PARENT_SCOPE)
            return()
        endif()
    endif()

    # FetchContent fallback with shallow clone for speed
    report_dependency_status("nlohmann-json" "nlohmann_json" "FETCHING" "GitHub FetchContent")
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(json)
    set(GEMMA_JSON_LIB nlohmann_json::nlohmann_json PARENT_SCOPE)
endfunction()

function(ensure_gtest_dependency)
    if(GEMMA_PREFER_SYSTEM_DEPS)
        find_package(GTest QUIET CONFIG)
        if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
            report_dependency_status("Google Test" "GTest::gtest" "FOUND" "vcpkg/system")
            set(GEMMA_GTEST_LIBS GTest::gtest GTest::gtest_main PARENT_SCOPE)
            return()
        endif()
    endif()

    # FetchContent fallback
    report_dependency_status("Google Test" "gtest" "FETCHING" "GitHub FetchContent")
    set(BUILD_GMOCK OFF CACHE BOOL "Disable GMock")
    set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest install")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
    set(GEMMA_GTEST_LIBS gtest gtest_main PARENT_SCOPE)
endfunction()

function(ensure_benchmark_dependency)
    if(GEMMA_PREFER_SYSTEM_DEPS)
        find_package(benchmark QUIET CONFIG)
        if(TARGET benchmark::benchmark)
            report_dependency_status("Google Benchmark" "benchmark::benchmark" "FOUND" "vcpkg/system")
            set(GEMMA_BENCHMARK_LIB benchmark::benchmark PARENT_SCOPE)
            return()
        endif()
    endif()

    # FetchContent fallback
    report_dependency_status("Google Benchmark" "benchmark" "FETCHING" "GitHub FetchContent")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest tests")
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install")
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.2
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(benchmark)
    set(GEMMA_BENCHMARK_LIB benchmark::benchmark PARENT_SCOPE)
endfunction()

# =============================================================================
# Main Dependency Resolution
# =============================================================================

message(STATUS "")
message(STATUS "=== Resolving Dependencies (Priority: local > vcpkg > FetchContent) ===")

# Core dependencies (always required)
ensure_highway_dependency()
ensure_sentencepiece_dependency()
ensure_nlohmann_json_dependency()

# Optional dependencies (based on build options)
if(GEMMA_ENABLE_TESTS OR GEMMA_BUILD_ENHANCED_TESTS)
    ensure_gtest_dependency()
endif()

if(GEMMA_BUILD_BENCHMARKS)
    ensure_benchmark_dependency()
endif()

# Set cache variables for use by subdirectories
set(GEMMA_HWY_LIBS ${GEMMA_HWY_LIBS} CACHE STRING "Highway libraries resolved by Dependencies.cmake")
set(GEMMA_SENTENCEPIECE_LIB ${GEMMA_SENTENCEPIECE_LIB} CACHE STRING "SentencePiece library resolved by Dependencies.cmake")
set(GEMMA_JSON_LIB ${GEMMA_JSON_LIB} CACHE STRING "nlohmann-json library resolved by Dependencies.cmake")
set(GEMMA_GTEST_LIBS ${GEMMA_GTEST_LIBS} CACHE STRING "Google Test libraries resolved by Dependencies.cmake")
set(GEMMA_BENCHMARK_LIB ${GEMMA_BENCHMARK_LIB} CACHE STRING "Google Benchmark library resolved by Dependencies.cmake")

message(STATUS "=== Dependency Resolution Complete ===")
message(STATUS "")