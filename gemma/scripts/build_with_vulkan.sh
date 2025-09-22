#!/bin/bash
# build_with_vulkan.sh - Build Gemma.cpp with Vulkan compute backend
#
# This script builds Gemma.cpp with Vulkan compute acceleration support.
# It handles environment setup, dependency verification, and optimized compilation.
#
# Usage:
#   ./scripts/build_with_vulkan.sh [options]
#
# Options:
#   --debug          Build in debug mode
#   --release        Build in release mode (default)
#   --clean          Clean build directory before building
#   --install        Install after building
#   --test           Run tests after building
#   --benchmark      Run benchmarks after building
#   --validation     Enable Vulkan validation layers
#   --help           Show this help message

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-vulkan"
INSTALL_DIR="${PROJECT_ROOT}/install-vulkan"

# Default options
BUILD_TYPE="Release"
CLEAN_BUILD=false
INSTALL_AFTER_BUILD=false
RUN_TESTS=false
RUN_BENCHMARKS=false
ENABLE_VALIDATION=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Build Gemma.cpp with Vulkan compute backend

Usage: $0 [options]

Options:
    --debug          Build in debug mode
    --release        Build in release mode (default)
    --clean          Clean build directory before building
    --install        Install after building
    --test           Run tests after building
    --benchmark      Run benchmarks after building
    --validation     Enable Vulkan validation layers (debug builds only)
    --verbose        Enable verbose build output
    --help           Show this help message

Environment Variables:
    VULKAN_SDK       Path to Vulkan SDK installation
    VK_LAYER_PATH    Path to Vulkan validation layers
    CMAKE_BUILD_PARALLEL_LEVEL  Number of parallel build jobs

Examples:
    $0                           # Basic release build
    $0 --debug --validation     # Debug build with validation layers
    $0 --clean --install        # Clean release build with install
    $0 --benchmark              # Release build with benchmarks

Requirements:
    - Vulkan SDK 1.2 or later
    - CMake 3.20 or later
    - C++20 capable compiler
    - Vulkan-capable GPU with updated drivers

Vulkan GPU Support:
    NVIDIA:   GTX 10xx/RTX series or newer
    AMD:      RX Vega/RDNA/RDNA2 or newer
    Intel:    Arc Alchemist or newer (integrated graphics vary)

For more information, see: https://vulkan.lunarg.com/sdk/home
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            --release)
                BUILD_TYPE="Release"
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --install)
                INSTALL_AFTER_BUILD=true
                shift
                ;;
            --test)
                RUN_TESTS=true
                shift
                ;;
            --benchmark)
                RUN_BENCHMARKS=true
                shift
                ;;
            --validation)
                ENABLE_VALIDATION=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake 3.20 or later."
        exit 1
    fi

    local cmake_version=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    log_info "Found CMake version: $cmake_version"

    # Check for Vulkan SDK
    if [[ -z "${VULKAN_SDK:-}" ]]; then
        # Try common Vulkan SDK installation paths
        local vulkan_paths=(
            "$HOME/VulkanSDK/*/x86_64"
            "/usr/local/vulkan"
            "/opt/vulkan"
            "/usr/share/vulkan"
        )

        for path_pattern in "${vulkan_paths[@]}"; do
            for path in $path_pattern; do
                if [[ -d "$path" && -f "$path/include/vulkan/vulkan.h" ]]; then
                    export VULKAN_SDK="$path"
                    log_info "Found Vulkan SDK at: $VULKAN_SDK"
                    break 2
                fi
            done
        done

        if [[ -z "${VULKAN_SDK:-}" ]]; then
            log_error "Vulkan SDK not found. Please install Vulkan SDK."
            log_error "Download from: https://vulkan.lunarg.com/sdk/home"
            exit 1
        fi
    else
        log_info "Using Vulkan SDK from VULKAN_SDK: $VULKAN_SDK"
    fi

    # Verify Vulkan SDK installation
    if [[ ! -f "$VULKAN_SDK/include/vulkan/vulkan.h" ]]; then
        log_error "Vulkan headers not found at: $VULKAN_SDK/include/vulkan/vulkan.h"
        exit 1
    fi

    # Check for Vulkan loader library
    local vulkan_lib_paths=(
        "$VULKAN_SDK/lib/libvulkan.so"
        "$VULKAN_SDK/lib64/libvulkan.so"
        "/usr/lib/x86_64-linux-gnu/libvulkan.so"
        "/usr/lib64/libvulkan.so"
        "/usr/lib/libvulkan.so"
    )

    local vulkan_lib_found=false
    for lib_path in "${vulkan_lib_paths[@]}"; do
        if [[ -f "$lib_path" ]]; then
            vulkan_lib_found=true
            log_info "Found Vulkan library: $lib_path"
            break
        fi
    done

    if [[ "$vulkan_lib_found" == false ]]; then
        log_error "Vulkan library not found in standard locations"
        exit 1
    fi

    log_success "System requirements check passed"
}

# Check GPU and driver support
check_gpu_support() {
    log_info "Checking GPU and driver support..."

    # Check for vulkaninfo utility
    local vulkaninfo_paths=(
        "$VULKAN_SDK/bin/vulkaninfo"
        "/usr/bin/vulkaninfo"
        "/usr/local/bin/vulkaninfo"
    )

    local vulkaninfo_cmd=""
    for path in "${vulkaninfo_paths[@]}"; do
        if [[ -f "$path" ]]; then
            vulkaninfo_cmd="$path"
            break
        fi
    done

    if [[ -n "$vulkaninfo_cmd" ]]; then
        log_info "Running Vulkan device detection..."

        # Run vulkaninfo and capture output
        if vulkan_output=$($vulkaninfo_cmd --summary 2>/dev/null); then
            log_info "Vulkan devices detected:"
            echo "$vulkan_output" | grep -E "deviceName|deviceType|apiVersion" | head -10
        else
            log_warning "vulkaninfo failed to run or no Vulkan devices found"
            log_warning "This may indicate driver issues or unsupported hardware"
        fi
    else
        log_warning "vulkaninfo not found, skipping device detection"
    fi

    # Check for common GPU drivers
    if lspci | grep -i "vga\|3d\|display" &> /dev/null; then
        log_info "Detected graphics hardware:"
        lspci | grep -i "vga\|3d\|display" | head -5
    fi

    log_success "GPU support check completed"
}

# Setup Vulkan environment
setup_vulkan_environment() {
    log_info "Setting up Vulkan environment..."

    # Add Vulkan SDK to environment paths
    export PATH="$VULKAN_SDK/bin:$PATH"
    export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="$VULKAN_SDK/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

    # Set validation layer path if validation is enabled
    if [[ "$ENABLE_VALIDATION" == true ]] || [[ "$BUILD_TYPE" == "Debug" ]]; then
        if [[ -d "$VULKAN_SDK/etc/vulkan/explicit_layer.d" ]]; then
            export VK_LAYER_PATH="$VULKAN_SDK/etc/vulkan/explicit_layer.d:${VK_LAYER_PATH:-}"
            log_info "Validation layers enabled"
        fi

        if [[ -d "$VULKAN_SDK/share/vulkan/explicit_layer.d" ]]; then
            export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d:${VK_LAYER_PATH:-}"
        fi
    fi

    # Check for glslc (GLSL to SPIR-V compiler)
    if command -v glslc &> /dev/null; then
        local glslc_version=$(glslc --version | head -1)
        log_info "Found GLSL compiler: $glslc_version"
    else
        log_warning "glslc (GLSL compiler) not found. Some shader compilation features may be unavailable."
    fi

    # Check for spirv-opt (SPIR-V optimizer)
    if command -v spirv-opt &> /dev/null; then
        log_info "Found SPIR-V optimizer: spirv-opt"
    else
        log_warning "spirv-opt not found. Shader optimization may be limited."
    fi

    log_success "Vulkan environment setup complete"
}

# Setup build directory
setup_build_directory() {
    log_info "Setting up build directory: $BUILD_DIR"

    if [[ "$CLEAN_BUILD" == true ]] && [[ -d "$BUILD_DIR" ]]; then
        log_info "Cleaning existing build directory..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    log_success "Build directory ready: $BUILD_DIR"
}

# Configure CMake build
configure_cmake() {
    log_info "Configuring CMake build (Build Type: $BUILD_TYPE)..."

    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR"
        "-DGEMMA_BUILD_VULKAN_BACKEND=ON"
        "-DGEMMA_BUILD_BACKEND_TESTS=ON"
        "-DGEMMA_BUILD_BENCHMARKS=ON"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )

    # Vulkan-specific configurations
    cmake_args+=(
        "-DVulkan_INCLUDE_DIR=$VULKAN_SDK/include"
        "-DVulkan_LIBRARY=$VULKAN_SDK/lib/libvulkan.so"
        "-DVULKAN_SDK_PATH=$VULKAN_SDK"
    )

    # Add glslc path if available
    if command -v glslc &> /dev/null; then
        cmake_args+=(
            "-DGLSLC_EXECUTABLE=$(which glslc)"
        )
    fi

    # Validation layer configuration
    if [[ "$ENABLE_VALIDATION" == true ]] || [[ "$BUILD_TYPE" == "Debug" ]]; then
        cmake_args+=(
            "-DVULKAN_ENABLE_VALIDATION=ON"
            "-DVK_LAYER_PATH=$VK_LAYER_PATH"
        )
    else
        cmake_args+=(
            "-DVULKAN_ENABLE_VALIDATION=OFF"
        )
    fi

    # Debug-specific options
    if [[ "$BUILD_TYPE" == "Debug" ]]; then
        cmake_args+=(
            "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g3 -DVK_ENABLE_BETA_EXTENSIONS"
            "-DVULKAN_DEBUG=ON"
        )
    else
        cmake_args+=(
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -DVK_ENABLE_BETA_EXTENSIONS"
            "-DVULKAN_OPTIMIZATIONS=ON"
        )
    fi

    # Add verbose output if requested
    if [[ "$VERBOSE" == true ]]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi

    log_info "CMake configuration: ${cmake_args[*]}"

    if cmake "${cmake_args[@]}" "$PROJECT_ROOT"; then
        log_success "CMake configuration successful"
    else
        log_error "CMake configuration failed"
        exit 1
    fi
}

# Build the project
build_project() {
    log_info "Building project..."

    local build_args=(
        "--build" "."
        "--config" "$BUILD_TYPE"
    )

    # Set parallel build jobs
    local parallel_jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}"
    build_args+=("--parallel" "$parallel_jobs")

    if [[ "$VERBOSE" == true ]]; then
        build_args+=("--verbose")
    fi

    log_info "Build configuration: ${build_args[*]}"
    log_info "Using $parallel_jobs parallel jobs"

    local start_time=$(date +%s)

    if cmake "${build_args[@]}"; then
        local end_time=$(date +%s)
        local build_time=$((end_time - start_time))
        log_success "Build completed successfully in ${build_time} seconds"
    else
        log_error "Build failed"
        exit 1
    fi
}

# Install the project
install_project() {
    if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
        log_info "Installing project to: $INSTALL_DIR"

        if cmake --install . --config "$BUILD_TYPE"; then
            log_success "Installation completed"
            log_info "Installed files in: $INSTALL_DIR"
        else
            log_error "Installation failed"
            exit 1
        fi
    fi
}

# Run tests
run_tests() {
    if [[ "$RUN_TESTS" == true ]]; then
        log_info "Running Vulkan backend tests..."

        # Set Vulkan environment for tests
        export VK_ICD_FILENAMES=""  # Use system default
        if [[ "$ENABLE_VALIDATION" == true ]] || [[ "$BUILD_TYPE" == "Debug" ]]; then
            export VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
        fi

        # Run CTest
        if ctest --build-config "$BUILD_TYPE" --output-on-failure --parallel "$(($(nproc 2>/dev/null || echo 4)/2))"; then
            log_success "All tests passed"
        else
            log_warning "Some tests failed"
        fi

        # Run specific Vulkan tests if available
        if [[ -f "./tests/backends/test_vulkan" ]]; then
            log_info "Running specific Vulkan backend tests..."
            ./tests/backends/test_vulkan || log_warning "Vulkan specific tests failed"
        fi
    fi
}

# Run benchmarks
run_benchmarks() {
    if [[ "$RUN_BENCHMARKS" == true ]]; then
        log_info "Running Vulkan backend benchmarks..."

        # Set optimal Vulkan environment for benchmarks
        export VK_INSTANCE_LAYERS=""  # Disable validation for performance

        # Run backend comparison benchmarks
        if [[ -f "./tests/backends/benchmark_backends" ]]; then
            log_info "Running backend performance comparison..."
            ./tests/backends/benchmark_backends || log_warning "Benchmarks failed"
        fi

        # Display Vulkan device information if available
        if command -v vulkaninfo &> /dev/null; then
            log_info "Vulkan Device Information:"
            vulkaninfo --summary 2>/dev/null | grep -E "deviceName|deviceType|driverVersion|apiVersion" | head -10 || true
        fi
    fi
}

# Print build summary
print_summary() {
    log_info "Build Summary:"
    echo "  Project Root: $PROJECT_ROOT"
    echo "  Build Directory: $BUILD_DIR"
    echo "  Build Type: $BUILD_TYPE"
    echo "  Vulkan SDK: ${VULKAN_SDK:-Not set}"
    echo "  Validation Layers: $ENABLE_VALIDATION"

    if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
        echo "  Install Directory: $INSTALL_DIR"
    fi

    echo "  Tests Run: $RUN_TESTS"
    echo "  Benchmarks Run: $RUN_BENCHMARKS"

    log_success "Vulkan backend build process completed!"

    # Provide usage instructions
    echo ""
    log_info "Usage Instructions:"
    echo "  1. Run Gemma with Vulkan: $BUILD_DIR/gemma --weights <model.sbs> --tokenizer <tokenizer.spm>"
    echo "  2. Run tests: cd $BUILD_DIR && ctest"
    echo "  3. Run benchmarks: $BUILD_DIR/tests/backends/benchmark_backends"
    echo "  4. Check Vulkan devices: vulkaninfo --summary"

    # Vulkan-specific tips
    echo ""
    log_info "Vulkan Performance Tips:"
    echo "  - Use dedicated GPU over integrated graphics when available"
    echo "  - Monitor GPU memory usage and avoid over-allocation"
    echo "  - Enable validation layers only during development"
    echo "  - Update GPU drivers for optimal performance"
    echo "  - Consider using Vulkan memory allocator (VMA) for large workloads"

    # Troubleshooting
    echo ""
    log_info "Troubleshooting:"
    echo "  - If vulkaninfo fails: Check GPU drivers and Vulkan runtime"
    echo "  - For validation errors: Set VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation"
    echo "  - For performance issues: Disable validation layers in release builds"
    echo "  - For memory errors: Reduce buffer sizes or use memory-mapped buffers"
}

# Main execution function
main() {
    echo "============================================================"
    echo "  Gemma.cpp Vulkan Backend Build Script"
    echo "============================================================"

    parse_args "$@"
    check_requirements
    check_gpu_support
    setup_vulkan_environment
    setup_build_directory
    configure_cmake
    build_project
    install_project
    run_tests
    run_benchmarks
    print_summary
}

# Trap errors and cleanup
trap 'log_error "Build process interrupted or failed"' ERR INT TERM

# Execute main function
main "$@"