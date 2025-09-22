cmake_minimum_required(VERSION 3.20)
project(gemma-chatbot VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings via maturin" ON)
option(BUILD_WASM "Build WebAssembly target" OFF)
option(ENABLE_SIMD "Enable SIMD optimizations" ON)

# Find Python and UV
find_program(UV_EXECUTABLE uv REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_program(MATURIN_EXECUTABLE maturin)

# Rust build configuration
set(RUST_BUILD_DIR ${CMAKE_BINARY_DIR}/rust)
set(RUST_TARGET_DIR ${RUST_BUILD_DIR}/target)

# Configure Rust build based on CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(RUST_BUILD_TYPE "--release")
    set(RUST_BUILD_SUBDIR "release")
else()
    set(RUST_BUILD_TYPE "")
    set(RUST_BUILD_SUBDIR "debug")
endif()

# Platform-specific configurations
if(WIN32)
    set(RUST_TARGET "x86_64-pc-windows-msvc")
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(RUST_TARGET "aarch64-apple-darwin")
    else()
        set(RUST_TARGET "x86_64-apple-darwin")
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(RUST_TARGET "aarch64-unknown-linux-gnu")
    else()
        set(RUST_TARGET "x86_64-unknown-linux-gnu")
    endif()
endif()

# Build Rust core components
add_custom_target(rust-core ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building Rust core components..."
    COMMAND ${CARGO_EXECUTABLE} build --workspace ${RUST_BUILD_TYPE} --target-dir ${RUST_TARGET_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust_core
    COMMENT "Building Rust workspace"
)

# Build Python extensions with maturin
if(BUILD_PYTHON_BINDINGS AND MATURIN_EXECUTABLE)
    add_custom_target(python-bindings
        COMMAND ${CMAKE_COMMAND} -E echo "Building Python bindings with maturin..."
        COMMAND ${UV_EXECUTABLE} run maturin build ${RUST_BUILD_TYPE} --out ${CMAKE_BINARY_DIR}/wheels
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust_extensions
        COMMENT "Building Python bindings"
        DEPENDS rust-core
    )
else()
    message(STATUS "Maturin not found or Python bindings disabled")
endif()

# WASM build target
if(BUILD_WASM)
    add_custom_target(wasm
        COMMAND ${CMAKE_COMMAND} -E echo "Building WASM target..."
        COMMAND ${CARGO_EXECUTABLE} build --target wasm32-unknown-unknown ${RUST_BUILD_TYPE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust_core/wasm
        COMMENT "Building WebAssembly target"
    )
endif()

# Python environment setup
add_custom_target(setup-python
    COMMAND ${CMAKE_COMMAND} -E echo "Setting up Python environment with UV..."
    COMMAND ${UV_EXECUTABLE} sync --all-groups
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Installing Python dependencies"
)

# Install Python package in development mode
add_custom_target(install-dev
    COMMAND ${CMAKE_COMMAND} -E echo "Installing in development mode..."
    COMMAND ${UV_EXECUTABLE} pip install -e .
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS setup-python python-bindings
    COMMENT "Installing package in editable mode"
)

# Test targets
enable_testing()

add_test(NAME rust-tests
    COMMAND ${CARGO_EXECUTABLE} test --workspace
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust_core
)

add_test(NAME python-tests
    COMMAND ${UV_EXECUTABLE} run pytest tests/
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Clean target
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${RUST_TARGET_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/wheels
    COMMAND ${CARGO_EXECUTABLE} clean
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning all build artifacts"
)

# Install target
install(DIRECTORY ${CMAKE_BINARY_DIR}/wheels/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/site-packages
    FILES_MATCHING PATTERN "*.whl"
)

# Print configuration
message(STATUS "CMake Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Rust Target: ${RUST_TARGET}")
message(STATUS "  Python Bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  WASM Build: ${BUILD_WASM}")
message(STATUS "  SIMD Optimizations: ${ENABLE_SIMD}")
