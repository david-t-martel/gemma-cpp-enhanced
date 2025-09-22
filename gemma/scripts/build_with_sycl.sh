#!/bin/bash
# build_with_sycl.sh - Build Gemma.cpp with Intel oneAPI SYCL backend
#
# This script builds Gemma.cpp with Intel oneAPI SYCL acceleration support.
# It handles environment setup, dependency verification, and optimized compilation.
#
# Usage:
#   ./scripts/build_with_sycl.sh [options]
#
# Options:
#   --debug          Build in debug mode
#   --release        Build in release mode (default)
#   --clean          Clean build directory before building
#   --install        Install after building
#   --test           Run tests after building
#   --benchmark      Run benchmarks after building
#   --help           Show this help message

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-sycl"
INSTALL_DIR="${PROJECT_ROOT}/install-sycl"

# Default options
BUILD_TYPE="Release"
CLEAN_BUILD=false
INSTALL_AFTER_BUILD=false
RUN_TESTS=false
RUN_BENCHMARKS=false
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
Build Gemma.cpp with Intel oneAPI SYCL backend

Usage: $0 [options]

Options:
    --debug          Build in debug mode
    --release        Build in release mode (default)
    --clean          Clean build directory before building
    --install        Install after building
    --test           Run tests after building
    --benchmark      Run benchmarks after building
    --verbose        Enable verbose build output
    --help           Show this help message

Environment Variables:
    ONEAPI_ROOT      Path to Intel oneAPI installation
    SYCL_DEVICE      Target SYCL device (cpu, gpu, fpga)
    CMAKE_BUILD_PARALLEL_LEVEL  Number of parallel build jobs

Examples:
    $0                           # Basic release build
    $0 --debug --test           # Debug build with tests
    $0 --clean --install        # Clean release build with install
    $0 --benchmark              # Release build with benchmarks

Requirements:
    - Intel oneAPI toolkit (2023.0 or later)
    - CMake 3.20 or later
    - C++20 capable compiler

For more information, see: https://software.intel.com/content/www/us/en/develop/tools/oneapi.html
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

    # Check for Intel oneAPI
    if [[ -z "${ONEAPI_ROOT:-}" ]]; then
        # Try common installation paths
        local oneapi_paths=(
            "/opt/intel/oneapi"
            "$HOME/intel/oneapi"
            "/usr/local/intel/oneapi"
        )

        for path in "${oneapi_paths[@]}"; do
            if [[ -d "$path" ]]; then
                export ONEAPI_ROOT="$path"
                log_info "Found oneAPI at: $ONEAPI_ROOT"
                break
            fi
        done

        if [[ -z "${ONEAPI_ROOT:-}" ]]; then
            log_error "Intel oneAPI not found. Please install Intel oneAPI toolkit."
            log_error "Download from: https://software.intel.com/content/www/us/en/develop/tools/oneapi.html"
            exit 1
        fi
    else
        log_info "Using oneAPI from ONEAPI_ROOT: $ONEAPI_ROOT"
    fi

    # Verify oneAPI installation
    local setvars_script="$ONEAPI_ROOT/setvars.sh"
    if [[ ! -f "$setvars_script" ]]; then
        log_error "oneAPI setvars.sh not found at: $setvars_script"
        exit 1
    fi

    log_success "System requirements check passed"
}

# Setup oneAPI environment
setup_oneapi_environment() {
    log_info "Setting up Intel oneAPI environment..."

    # Source oneAPI environment
    local setvars_script="$ONEAPI_ROOT/setvars.sh"
    if [[ -f "$setvars_script" ]]; then
        log_info "Sourcing oneAPI environment: $setvars_script"
        source "$setvars_script" --silent

        # Verify SYCL compiler is available
        if command -v icpx &> /dev/null; then
            local compiler_version=$(icpx --version | head -n1)
            log_info "SYCL compiler: $compiler_version"
        elif command -v clang++ &> /dev/null; then
            local compiler_version=$(clang++ --version | head -n1)
            log_info "SYCL compiler: $compiler_version"
        else
            log_error "SYCL compiler not found after sourcing oneAPI environment"
            exit 1
        fi

        # Set SYCL-specific environment variables
        export CXX="${CXX:-icpx}"
        export CC="${CC:-icx}"

        log_success "oneAPI environment setup complete"
    else
        log_error "oneAPI setvars.sh not found: $setvars_script"
        exit 1
    fi
}

# Configure SYCL device target
configure_sycl_device() {
    log_info "Configuring SYCL device target..."

    # Set default device if not specified
    if [[ -z "${SYCL_DEVICE:-}" ]]; then
        SYCL_DEVICE="gpu"
        log_info "Using default SYCL device: $SYCL_DEVICE"
    else
        log_info "Using specified SYCL device: $SYCL_DEVICE"
    fi

    # Validate device target
    case "$SYCL_DEVICE" in
        cpu|gpu|fpga)
            log_info "Valid SYCL device target: $SYCL_DEVICE"
            ;;
        *)
            log_warning "Unknown SYCL device: $SYCL_DEVICE, using 'gpu' as fallback"
            SYCL_DEVICE="gpu"
            ;;
    esac

    export SYCL_DEVICE_FILTER="$SYCL_DEVICE"
    log_success "SYCL device configuration complete"
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
        "-DGEMMA_BUILD_SYCL_BACKEND=ON"
        "-DGEMMA_BUILD_BACKEND_TESTS=ON"
        "-DGEMMA_BUILD_BENCHMARKS=ON"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )

    # Add SYCL-specific configurations
    cmake_args+=(
        "-DCMAKE_CXX_COMPILER=${CXX:-icpx}"
        "-DCMAKE_C_COMPILER=${CC:-icx}"
        "-DSYCL_DEVICE_FILTER=$SYCL_DEVICE"
    )

    # Debug-specific options
    if [[ "$BUILD_TYPE" == "Debug" ]]; then
        cmake_args+=(
            "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g3 -fsycl-device-debug"
            "-DSYCL_ENABLE_DEBUG=ON"
        )
    else
        cmake_args+=(
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -fsycl"
            "-DSYCL_ENABLE_OPTIMIZATIONS=ON"
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
        log_info "Running SYCL backend tests..."

        # Run CTest
        if ctest --build-config "$BUILD_TYPE" --output-on-failure --parallel "$(($(nproc 2>/dev/null || echo 4)/2))"; then
            log_success "All tests passed"
        else
            log_warning "Some tests failed"
        fi

        # Run specific SYCL tests if available
        if [[ -f "./tests/backends/test_sycl" ]]; then
            log_info "Running specific SYCL backend tests..."
            ./tests/backends/test_sycl || log_warning "SYCL specific tests failed"
        fi
    fi
}

# Run benchmarks
run_benchmarks() {
    if [[ "$RUN_BENCHMARKS" == true ]]; then
        log_info "Running SYCL backend benchmarks..."

        # Run backend comparison benchmarks
        if [[ -f "./tests/backends/benchmark_backends" ]]; then
            log_info "Running backend performance comparison..."
            ./tests/backends/benchmark_backends || log_warning "Benchmarks failed"
        fi

        # Run other relevant benchmarks
        if [[ -f "./benchmark" ]] || [[ -f "./gemma_benchmark" ]]; then
            log_info "Running additional benchmarks..."
            # Add specific benchmark commands here
        fi
    fi
}

# Print build summary
print_summary() {
    log_info "Build Summary:"
    echo "  Project Root: $PROJECT_ROOT"
    echo "  Build Directory: $BUILD_DIR"
    echo "  Build Type: $BUILD_TYPE"
    echo "  SYCL Device: $SYCL_DEVICE"
    echo "  oneAPI Root: ${ONEAPI_ROOT:-Not set}"

    if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
        echo "  Install Directory: $INSTALL_DIR"
    fi

    echo "  Tests Run: $RUN_TESTS"
    echo "  Benchmarks Run: $RUN_BENCHMARKS"

    log_success "SYCL backend build process completed!"

    # Provide usage instructions
    echo ""
    log_info "Usage Instructions:"
    echo "  1. Set up oneAPI environment: source $ONEAPI_ROOT/setvars.sh"
    echo "  2. Run Gemma with SYCL: $BUILD_DIR/gemma --weights <model.sbs> --tokenizer <tokenizer.spm>"
    echo "  3. Run tests: cd $BUILD_DIR && ctest"
    echo "  4. Run benchmarks: $BUILD_DIR/tests/backends/benchmark_backends"
}

# Main execution function
main() {
    echo "============================================================"
    echo "  Gemma.cpp SYCL Backend Build Script"
    echo "============================================================"

    parse_args "$@"
    check_requirements
    setup_oneapi_environment
    configure_sycl_device
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