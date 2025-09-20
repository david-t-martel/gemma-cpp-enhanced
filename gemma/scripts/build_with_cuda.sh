#!/bin/bash
# build_with_cuda.sh - Build Gemma.cpp with CUDA backend
#
# This script builds Gemma.cpp with NVIDIA CUDA acceleration support.
# It handles environment setup, dependency verification, and optimized compilation.
#
# Usage:
#   ./scripts/build_with_cuda.sh [options]
#
# Options:
#   --debug          Build in debug mode
#   --release        Build in release mode (default)
#   --clean          Clean build directory before building
#   --install        Install after building
#   --test           Run tests after building
#   --benchmark      Run benchmarks after building
#   --cuda-arch      Specify CUDA architectures (e.g., "75;80;86")
#   --help           Show this help message

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-cuda"
INSTALL_DIR="${PROJECT_ROOT}/install-cuda"

# Default options
BUILD_TYPE="Release"
CLEAN_BUILD=false
INSTALL_AFTER_BUILD=false
RUN_TESTS=false
RUN_BENCHMARKS=false
VERBOSE=false
CUDA_ARCHITECTURES=""

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
Build Gemma.cpp with NVIDIA CUDA backend

Usage: $0 [options]

Options:
    --debug          Build in debug mode
    --release        Build in release mode (default)
    --clean          Clean build directory before building
    --install        Install after building
    --test           Run tests after building
    --benchmark      Run benchmarks after benchmarks
    --cuda-arch      Specify CUDA architectures (e.g., "75;80;86")
    --verbose        Enable verbose build output
    --help           Show this help message

Environment Variables:
    CUDA_HOME        Path to CUDA Toolkit installation
    CUDNN_ROOT       Path to cuDNN installation (optional)
    CMAKE_BUILD_PARALLEL_LEVEL  Number of parallel build jobs

Examples:
    $0                           # Basic release build
    $0 --debug --test           # Debug build with tests
    $0 --clean --install        # Clean release build with install
    $0 --cuda-arch "75;80;86"   # Build for specific GPU architectures
    $0 --benchmark              # Release build with benchmarks

GPU Architecture Guide:
    Maxwell:     50, 52, 53
    Pascal:      60, 61, 62
    Volta:       70, 72
    Turing:      75
    Ampere:      80, 86, 87
    Ada Lovelace: 89
    Hopper:      90

Requirements:
    - NVIDIA CUDA Toolkit 11.0 or later
    - CMake 3.20 or later
    - C++20 capable compiler
    - NVIDIA GPU with compute capability 5.0+

For more information, see: https://developer.nvidia.com/cuda-downloads
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
            --cuda-arch)
                CUDA_ARCHITECTURES="$2"
                shift 2
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

    # Check for NVIDIA drivers
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA drivers not found. Please install NVIDIA drivers."
        exit 1
    fi

    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    log_info "Found NVIDIA driver version: $driver_version"

    # Check for CUDA
    if [[ -z "${CUDA_HOME:-}" ]]; then
        # Try common CUDA installation paths
        local cuda_paths=(
            "/usr/local/cuda"
            "/opt/cuda"
            "/usr/cuda"
            "/usr/local/cuda-12"
            "/usr/local/cuda-11"
        )

        for path in "${cuda_paths[@]}"; do
            if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
                export CUDA_HOME="$path"
                log_info "Found CUDA at: $CUDA_HOME"
                break
            fi
        done

        if [[ -z "${CUDA_HOME:-}" ]]; then
            log_error "CUDA Toolkit not found. Please install CUDA Toolkit."
            log_error "Download from: https://developer.nvidia.com/cuda-downloads"
            exit 1
        fi
    else
        log_info "Using CUDA from CUDA_HOME: $CUDA_HOME"
    fi

    # Verify CUDA installation
    if [[ ! -f "$CUDA_HOME/bin/nvcc" ]]; then
        log_error "nvcc not found at: $CUDA_HOME/bin/nvcc"
        exit 1
    fi

    local nvcc_version=$($CUDA_HOME/bin/nvcc --version | grep "release" | grep -oE '[0-9]+\.[0-9]+')
    log_info "Found CUDA version: $nvcc_version"

    # Check CUDA version compatibility
    local major_version=$(echo $nvcc_version | cut -d. -f1)
    if [[ $major_version -lt 11 ]]; then
        log_warning "CUDA version $nvcc_version may not be fully supported. Recommended: 11.0+"
    fi

    log_success "System requirements check passed"
}

# Detect GPU architectures
detect_gpu_architectures() {
    log_info "Detecting GPU architectures..."

    if [[ -n "$CUDA_ARCHITECTURES" ]]; then
        log_info "Using specified CUDA architectures: $CUDA_ARCHITECTURES"
        return
    fi

    # Try to detect GPU compute capabilities
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null)
        if [[ -n "$gpu_info" ]]; then
            # Convert compute capabilities to architecture numbers
            local arch_list=""
            while IFS= read -r compute_cap; do
                local arch=$(echo "$compute_cap" | tr -d '.')
                if [[ -n "$arch_list" ]]; then
                    arch_list="$arch_list;$arch"
                else
                    arch_list="$arch"
                fi
            done <<< "$gpu_info"

            CUDA_ARCHITECTURES="$arch_list"
            log_info "Detected GPU compute capabilities: $gpu_info"
            log_info "Using CUDA architectures: $CUDA_ARCHITECTURES"
        fi
    fi

    # Fallback to common architectures if detection failed
    if [[ -z "$CUDA_ARCHITECTURES" ]]; then
        CUDA_ARCHITECTURES="70;75;80;86"
        log_info "Using default CUDA architectures: $CUDA_ARCHITECTURES"
    fi
}

# Setup CUDA environment
setup_cuda_environment() {
    log_info "Setting up CUDA environment..."

    # Add CUDA to PATH
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    # Set CUDA-specific environment variables
    export NVCC_PREPEND_FLAGS='-ccbin g++'

    # Check for cuDNN
    if [[ -n "${CUDNN_ROOT:-}" ]]; then
        log_info "Using cuDNN from CUDNN_ROOT: $CUDNN_ROOT"
        export LD_LIBRARY_PATH="$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH"
        export CPLUS_INCLUDE_PATH="$CUDNN_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
    else
        # Try to find cuDNN in common locations
        local cudnn_paths=(
            "$CUDA_HOME"
            "/usr/local/cudnn"
            "/opt/cudnn"
        )

        for path in "${cudnn_paths[@]}"; do
            if [[ -f "$path/include/cudnn.h" ]]; then
                export CUDNN_ROOT="$path"
                log_info "Found cuDNN at: $CUDNN_ROOT"
                export LD_LIBRARY_PATH="$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH"
                export CPLUS_INCLUDE_PATH="$CUDNN_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
                break
            fi
        done
    fi

    log_success "CUDA environment setup complete"
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
        "-DGEMMA_BUILD_CUDA_BACKEND=ON"
        "-DGEMMA_BUILD_BACKEND_TESTS=ON"
        "-DGEMMA_BUILD_BENCHMARKS=ON"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
        "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES"
    )

    # CUDA-specific configurations
    cmake_args+=(
        "-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"
        "-DCUDAToolkit_ROOT=$CUDA_HOME"
        "-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON"
        "-DCMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS=ON"
    )

    # Add cuDNN if available
    if [[ -n "${CUDNN_ROOT:-}" ]]; then
        cmake_args+=(
            "-DCUDNN_ROOT=$CUDNN_ROOT"
            "-DCUDNN_INCLUDE_DIR=$CUDNN_ROOT/include"
            "-DCUDNN_LIBRARY=$CUDNN_ROOT/lib64"
        )
    fi

    # Debug-specific options
    if [[ "$BUILD_TYPE" == "Debug" ]]; then
        cmake_args+=(
            "-DCMAKE_CUDA_FLAGS_DEBUG=-O0 -g -G"
            "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g3"
        )
    else
        cmake_args+=(
            "-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG --use_fast_math"
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG"
        )
    fi

    # Performance optimizations
    cmake_args+=(
        "-DCMAKE_CUDA_FLAGS=--extended-lambda --expt-relaxed-constexpr"
    )

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

    # Set parallel build jobs (reduce for CUDA builds to avoid memory issues)
    local parallel_jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-$(($(nproc 2>/dev/null || echo 4)/2))}"
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
        log_info "Running CUDA backend tests..."

        # Check GPU availability for tests
        if ! nvidia-smi &> /dev/null; then
            log_warning "GPU not accessible, skipping GPU-specific tests"
            return
        fi

        # Run CTest
        if ctest --build-config "$BUILD_TYPE" --output-on-failure --parallel "$(($(nproc 2>/dev/null || echo 4)/2))"; then
            log_success "All tests passed"
        else
            log_warning "Some tests failed"
        fi

        # Run specific CUDA tests if available
        if [[ -f "./tests/backends/test_cuda" ]]; then
            log_info "Running specific CUDA backend tests..."
            ./tests/backends/test_cuda || log_warning "CUDA specific tests failed"
        fi
    fi
}

# Run benchmarks
run_benchmarks() {
    if [[ "$RUN_BENCHMARKS" == true ]]; then
        log_info "Running CUDA backend benchmarks..."

        # Check GPU availability for benchmarks
        if ! nvidia-smi &> /dev/null; then
            log_warning "GPU not accessible, skipping benchmarks"
            return
        fi

        # Run backend comparison benchmarks
        if [[ -f "./tests/backends/benchmark_backends" ]]; then
            log_info "Running backend performance comparison..."
            ./tests/backends/benchmark_backends || log_warning "Benchmarks failed"
        fi

        # Display GPU information
        log_info "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    fi
}

# Print build summary
print_summary() {
    log_info "Build Summary:"
    echo "  Project Root: $PROJECT_ROOT"
    echo "  Build Directory: $BUILD_DIR"
    echo "  Build Type: $BUILD_TYPE"
    echo "  CUDA Home: ${CUDA_HOME:-Not set}"
    echo "  CUDA Architectures: $CUDA_ARCHITECTURES"

    if [[ -n "${CUDNN_ROOT:-}" ]]; then
        echo "  cuDNN Root: $CUDNN_ROOT"
    fi

    if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
        echo "  Install Directory: $INSTALL_DIR"
    fi

    echo "  Tests Run: $RUN_TESTS"
    echo "  Benchmarks Run: $RUN_BENCHMARKS"

    log_success "CUDA backend build process completed!"

    # Provide usage instructions
    echo ""
    log_info "Usage Instructions:"
    echo "  1. Run Gemma with CUDA: $BUILD_DIR/gemma --weights <model.sbs> --tokenizer <tokenizer.spm>"
    echo "  2. Run tests: cd $BUILD_DIR && ctest"
    echo "  3. Run benchmarks: $BUILD_DIR/tests/backends/benchmark_backends"
    echo "  4. Check GPU status: nvidia-smi"

    # GPU recommendations
    echo ""
    log_info "Performance Tips:"
    echo "  - Use tensor cores on RTX/A/H series GPUs for maximum performance"
    echo "  - Monitor GPU memory usage with nvidia-smi"
    echo "  - Consider using FP16 precision for inference"
    echo "  - Use CUDA streams for overlapping computation and memory transfers"
}

# Main execution function
main() {
    echo "============================================================"
    echo "  Gemma.cpp CUDA Backend Build Script"
    echo "============================================================"

    parse_args "$@"
    check_requirements
    detect_gpu_architectures
    setup_cuda_environment
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