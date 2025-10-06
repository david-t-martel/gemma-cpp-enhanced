cmake_minimum_required(VERSION 3.20)
project(rag_redis_system VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_SIMD "Enable SIMD optimizations" ON)
option(ENABLE_GPU "Enable GPU support" OFF)
option(ENABLE_METRICS "Enable metrics collection" ON)
option(BUILD_FFI "Build FFI interface" ON)
option(BUILD_MCP_SERVER "Build MCP server" ON)
option(USE_REDIS "Use Redis backend" ON)
option(USE_DOCKER "Build Docker images" OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Platform detection
if(WIN32)
    set(PLATFORM_NAME "windows")
    set(LIB_PREFIX "")
    set(LIB_SUFFIX ".dll")
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
elseif(APPLE)
    set(PLATFORM_NAME "macos")
    set(LIB_PREFIX "lib")
    set(LIB_SUFFIX ".dylib")
elseif(UNIX)
    set(PLATFORM_NAME "linux")
    set(LIB_PREFIX "lib")
    set(LIB_SUFFIX ".so")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        $<$<CONFIG:Debug>:-g3>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
    )

    if(ENABLE_SIMD)
        add_compile_options(-mavx2 -mfma)
    endif()

    # Link-time optimization for Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
elseif(MSVC)
    add_compile_options(
        /W4
        /permissive-
        $<$<CONFIG:Debug>:/Od>
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
    )

    if(ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
    endif()

    # Link-time optimization for Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_link_options(/LTCG)
    endif()
endif()

# Find packages
find_package(Threads REQUIRED)

# Find Rust and Cargo
find_program(CARGO cargo REQUIRED)
find_program(RUSTC rustc REQUIRED)

# Get Rust target directory
execute_process(
    COMMAND ${CARGO} metadata --format-version 1 --no-deps
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CARGO_METADATA
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Default to target/release or target/debug based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUST_PROFILE "debug")
else()
    set(RUST_PROFILE "release")
endif()

set(RUST_TARGET_DIR ${CMAKE_SOURCE_DIR}/target/${RUST_PROFILE})

# Rust library build
set(RUST_LIB_NAME rag_redis_system)
set(RUST_STATIC_LIB ${RUST_TARGET_DIR}/${LIB_PREFIX}${RUST_LIB_NAME}.a)
set(RUST_SHARED_LIB ${RUST_TARGET_DIR}/${LIB_PREFIX}${RUST_LIB_NAME}${LIB_SUFFIX})

# Features list
set(RUST_FEATURES "")
if(ENABLE_SIMD)
    list(APPEND RUST_FEATURES "simsimd")
endif()
if(ENABLE_GPU)
    list(APPEND RUST_FEATURES "gpu")
endif()
if(ENABLE_METRICS)
    list(APPEND RUST_FEATURES "metrics")
endif()
if(BUILD_FFI)
    list(APPEND RUST_FEATURES "ffi")
endif()

# Convert features list to string
if(RUST_FEATURES)
    string(JOIN "," RUST_FEATURES_STR ${RUST_FEATURES})
    set(CARGO_FEATURES_ARG "--features" ${RUST_FEATURES_STR})
else()
    set(CARGO_FEATURES_ARG "")
endif()

# Custom target to build Rust library
add_custom_target(
    rust_build ALL
    COMMAND ${CARGO} build $<$<CONFIG:Release>:--release> ${CARGO_FEATURES_ARG}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Rust library with Cargo"
    VERBATIM
)

# Create imported library for Rust static lib
add_library(rag_redis_static STATIC IMPORTED)
set_target_properties(rag_redis_static PROPERTIES
    IMPORTED_LOCATION ${RUST_STATIC_LIB}
)
add_dependencies(rag_redis_static rust_build)

# Create imported library for Rust shared lib
if(BUILD_SHARED_LIBS)
    add_library(rag_redis_shared SHARED IMPORTED)
    set_target_properties(rag_redis_shared PROPERTIES
        IMPORTED_LOCATION ${RUST_SHARED_LIB}
        IMPORTED_IMPLIB ${RUST_TARGET_DIR}/${LIB_PREFIX}${RUST_LIB_NAME}.lib
    )
    add_dependencies(rag_redis_shared rust_build)
endif()

# FFI interface library
if(BUILD_FFI)
    add_library(rag_redis_ffi INTERFACE)
    target_include_directories(rag_redis_ffi INTERFACE
        ${CMAKE_SOURCE_DIR}/include
    )

    if(BUILD_SHARED_LIBS)
        target_link_libraries(rag_redis_ffi INTERFACE rag_redis_shared)
    else()
        target_link_libraries(rag_redis_ffi INTERFACE rag_redis_static)
    endif()

    target_link_libraries(rag_redis_ffi INTERFACE
        Threads::Threads
        $<$<PLATFORM_ID:Windows>:ws2_32>
        $<$<PLATFORM_ID:Windows>:userenv>
        $<$<PLATFORM_ID:Windows>:bcrypt>
        $<$<PLATFORM_ID:Linux>:dl>
        $<$<PLATFORM_ID:Linux>:m>
    )
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    # Rust tests
    add_test(
        NAME rust_tests
        COMMAND ${CARGO} test ${CARGO_FEATURES_ARG}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # C++ tests (if FFI is enabled)
    if(BUILD_FFI AND EXISTS ${CMAKE_SOURCE_DIR}/tests/cpp)
        add_subdirectory(tests/cpp)
    endif()
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_custom_target(
        benchmarks
        COMMAND ${CARGO} bench ${CARGO_FEATURES_ARG}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running Rust benchmarks"
        VERBATIM
    )
endif()

# MCP Server
if(BUILD_MCP_SERVER)
    add_custom_target(
        mcp_server
        COMMAND ${CARGO} build $<$<CONFIG:Release>:--release> --manifest-path mcp-server/Cargo.toml
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Building MCP server"
        VERBATIM
    )
endif()

# Documentation
add_custom_target(
    docs
    COMMAND ${CARGO} doc --all-features --no-deps --open
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating Rust documentation"
    VERBATIM
)

# Format code
add_custom_target(
    format
    COMMAND ${CARGO} fmt --all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting Rust code"
    VERBATIM
)

# Lint code
add_custom_target(
    lint
    COMMAND ${CARGO} clippy --all-features -- -D warnings
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Linting Rust code"
    VERBATIM
)

# Clean
add_custom_target(
    clean_all
    COMMAND ${CARGO} clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning all build artifacts"
    VERBATIM
)

# Install targets
include(GNUInstallDirs)

# Install Rust libraries
if(BUILD_FFI)
    install(FILES ${RUST_STATIC_LIB}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT libraries
    )

    if(BUILD_SHARED_LIBS)
        install(FILES ${RUST_SHARED_LIB}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT libraries
        )
    endif()

    # Install headers
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif()

# Install MCP server
if(BUILD_MCP_SERVER)
    install(PROGRAMS ${RUST_TARGET_DIR}/mcp-server
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT runtime
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)

# Create package config file
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/RagRedisConfig.cmake.in
    ${CMAKE_BINARY_DIR}/RagRedisConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RagRedis
)

# Create version file
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/RagRedisConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package files
install(FILES
    ${CMAKE_BINARY_DIR}/RagRedisConfig.cmake
    ${CMAKE_BINARY_DIR}/RagRedisConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RagRedis
    COMPONENT development
)

# CPack configuration for creating packages
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance RAG system with Redis backend")
set(CPACK_PACKAGE_VENDOR "RAG-Redis Development Team")
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/LICENSE)
set(CPACK_RESOURCE_FILE_README ${CMAKE_SOURCE_DIR}/README.md)

# Platform-specific package generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
elseif(UNIX)
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
endif()

# Component configuration
set(CPACK_COMPONENTS_ALL libraries headers runtime development)
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C/C++ Headers")
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development")

include(CPack)

# Status messages
message(STATUS "========================================")
message(STATUS "RAG-Redis System Configuration:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform:         ${PLATFORM_NAME}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Features:")
message(STATUS "    SIMD:           ${ENABLE_SIMD}")
message(STATUS "    GPU:            ${ENABLE_GPU}")
message(STATUS "    Metrics:        ${ENABLE_METRICS}")
message(STATUS "    FFI:            ${BUILD_FFI}")
message(STATUS "    MCP Server:     ${BUILD_MCP_SERVER}")
message(STATUS "    Shared Libs:    ${BUILD_SHARED_LIBS}")
message(STATUS "  Directories:")
message(STATUS "    Binary:         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "    Library:        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "    Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
