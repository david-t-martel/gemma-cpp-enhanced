#!/bin/bash
# build_all_backends.sh - Build Gemma.cpp with all available hardware backends
#
# This script automatically detects and builds Gemma.cpp with all available
# hardware acceleration backends (CPU, SYCL, CUDA, Vulkan, OpenCL, Metal).
# It provides a unified interface for cross-backend development and testing.
#
# Usage:
#   ./scripts/build_all_backends.sh [options]
#
# Options:
#   --debug          Build all backends in debug mode
#   --release        Build all backends in release mode (default)
#   --clean          Clean all build directories before building
#   --install        Install all backends after building
#   --test           Run tests for all backends after building
#   --benchmark      Run benchmarks for all backends after building
#   --parallel       Build backends in parallel (experimental)
#   --cpu-only       Build only CPU backend (baseline)
#   --skip-missing   Skip backends with missing dependencies
#   --help           Show this help message

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_BASE_DIR="${PROJECT_ROOT}/build-all-backends"

# Default options
BUILD_TYPE="Release"
CLEAN_BUILD=false
INSTALL_AFTER_BUILD=false
RUN_TESTS=false
RUN_BENCHMARKS=false
PARALLEL_BUILD=false
CPU_ONLY=false
SKIP_MISSING=false
VERBOSE=false

# Backend detection results
declare -A BACKEND_AVAILABLE
declare -A BACKEND_BUILD_DIR
declare -A BACKEND_SCRIPTS

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

log_backend() {
    echo -e "${CYAN}[BACKEND]${NC} $1"
}

log_benchmark() {
    echo -e "${MAGENTA}[BENCHMARK]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Build Gemma.cpp with all available hardware backends

This script automatically detects and builds Gemma.cpp with all available
hardware acceleration backends, providing a unified development environment.

Usage: $0 [options]

Options:
    --debug          Build all backends in debug mode
    --release        Build all backends in release mode (default)
    --clean          Clean all build directories before building
    --install        Install all backends after building
    --test           Run tests for all backends after building
    --benchmark      Run benchmarks for all backends after building
    --parallel       Build backends in parallel (experimental)
    --cpu-only       Build only CPU backend (baseline reference)
    --skip-missing   Skip backends with missing dependencies instead of failing
    --verbose        Enable verbose build output
    --help           Show this help message

Supported Backends:
    CPU              Always available (baseline implementation)
    SYCL             Intel oneAPI Data Parallel C++ (Intel GPUs, CPUs)
    CUDA             NVIDIA CUDA Toolkit (NVIDIA GPUs)
    Vulkan           Vulkan SDK (Cross-vendor GPU compute)
    OpenCL           OpenCL runtime (Cross-vendor GPU/CPU compute)
    Metal            Apple Metal (macOS/iOS - Apple Silicon/AMD/Intel)

Environment Variables:
    ONEAPI_ROOT      Path to Intel oneAPI installation
    CUDA_HOME        Path to CUDA Toolkit installation
    VULKAN_SDK       Path to Vulkan SDK installation
    CMAKE_BUILD_PARALLEL_LEVEL  Number of parallel build jobs

Examples:
    $0                           # Build all available backends (release)
    $0 --debug --test           # Debug build with tests for all backends
    $0 --clean --benchmark      # Clean build with performance comparison
    $0 --cpu-only --install     # Build only CPU baseline with install
    $0 --parallel --skip-missing # Fast parallel build, skip missing deps

Build Structure:
    build-all-backends/
    ├── cpu/                    # CPU-only build
    ├── sycl/                   # SYCL backend build
    ├── cuda/                   # CUDA backend build
    ├── vulkan/                 # Vulkan backend build
    ├── opencl/                 # OpenCL backend build
    ├── metal/                  # Metal backend build (macOS only)
    └── reports/                # Performance comparison reports

For more information about specific backends, run the individual build scripts
with --help option (e.g., ./scripts/build_with_cuda.sh --help)
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
            --parallel)
                PARALLEL_BUILD=true
                shift
                ;;
            --cpu-only)
                CPU_ONLY=true
                shift
                ;;
            --skip-missing)
                SKIP_MISSING=true
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

# Setup base build directory
setup_base_build_directory() {
    log_info "Setting up base build directory: $BUILD_BASE_DIR"

    if [[ "$CLEAN_BUILD" == true ]] && [[ -d "$BUILD_BASE_DIR" ]]; then
        log_info "Cleaning existing build directory..."
        rm -rf "$BUILD_BASE_DIR"
    fi

    mkdir -p "$BUILD_BASE_DIR"
    mkdir -p "$BUILD_BASE_DIR/reports"

    log_success "Base build directory ready"
}

# Detect available backends
detect_backends() {
    log_info "Detecting available hardware backends..."

    # CPU is always available
    BACKEND_AVAILABLE["cpu"]=true
    BACKEND_BUILD_DIR["cpu"]="$BUILD_BASE_DIR/cpu"
    log_backend "CPU: Always available (baseline)"

    # Skip other backends if CPU-only mode
    if [[ "$CPU_ONLY" == true ]]; then
        log_info "CPU-only mode enabled, skipping other backends"
        return
    fi

    # Detect SYCL/Intel oneAPI
    if [[ -n "${ONEAPI_ROOT:-}" ]] || [[ -d "/opt/intel/oneapi" ]] || [[ -d "$HOME/intel/oneapi" ]] || command -v icpx &> /dev/null; then
        BACKEND_AVAILABLE["sycl"]=true
        BACKEND_BUILD_DIR["sycl"]="$BUILD_BASE_DIR/sycl"
        BACKEND_SCRIPTS["sycl"]="$SCRIPT_DIR/build_with_sycl.sh"
        log_backend "SYCL: Intel oneAPI detected"
    else
        BACKEND_AVAILABLE["sycl"]=false
        log_backend "SYCL: Intel oneAPI not found"
    fi

    # Detect CUDA
    if [[ -n "${CUDA_HOME:-}" ]] || [[ -d "/usr/local/cuda" ]] || command -v nvcc &> /dev/null; then
        # Also check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            BACKEND_AVAILABLE["cuda"]=true
            BACKEND_BUILD_DIR["cuda"]="$BUILD_BASE_DIR/cuda"
            BACKEND_SCRIPTS["cuda"]="$SCRIPT_DIR/build_with_cuda.sh"
            log_backend "CUDA: NVIDIA CUDA Toolkit and GPU detected"
        else
            BACKEND_AVAILABLE["cuda"]=false
            log_backend "CUDA: CUDA Toolkit found but no NVIDIA GPU detected"
        fi
    else
        BACKEND_AVAILABLE["cuda"]=false
        log_backend "CUDA: NVIDIA CUDA Toolkit not found"
    fi

    # Detect Vulkan
    if [[ -n "${VULKAN_SDK:-}" ]] || [[ -d "$HOME/VulkanSDK" ]] || command -v vulkaninfo &> /dev/null; then
        # Check if Vulkan devices are available
        if command -v vulkaninfo &> /dev/null && vulkaninfo --summary &> /dev/null; then
            BACKEND_AVAILABLE["vulkan"]=true
            BACKEND_BUILD_DIR["vulkan"]="$BUILD_BASE_DIR/vulkan"
            BACKEND_SCRIPTS["vulkan"]="$SCRIPT_DIR/build_with_vulkan.sh"
            log_backend "Vulkan: Vulkan SDK and compatible GPU detected"
        else
            BACKEND_AVAILABLE["vulkan"]=false
            log_backend "Vulkan: Vulkan SDK found but no compatible devices"
        fi
    else
        BACKEND_AVAILABLE["vulkan"]=false
        log_backend "Vulkan: Vulkan SDK not found"
    fi

    # Detect OpenCL (basic check)
    if command -v clinfo &> /dev/null && clinfo &> /dev/null; then
        BACKEND_AVAILABLE["opencl"]=true
        BACKEND_BUILD_DIR["opencl"]="$BUILD_BASE_DIR/opencl"
        log_backend "OpenCL: OpenCL runtime and devices detected"
    else
        BACKEND_AVAILABLE["opencl"]=false
        log_backend "OpenCL: OpenCL runtime not available"
    fi

    # Detect Metal (macOS only)
    if [[ "$(uname)" == "Darwin" ]]; then
        BACKEND_AVAILABLE["metal"]=true
        BACKEND_BUILD_DIR["metal"]="$BUILD_BASE_DIR/metal"
        log_backend "Metal: macOS detected, Metal available"
    else
        BACKEND_AVAILABLE["metal"]=false
        log_backend "Metal: Not available (macOS only)"
    fi

    # Summary
    local available_count=0
    for backend in "${!BACKEND_AVAILABLE[@]}"; do
        if [[ "${BACKEND_AVAILABLE[$backend]}" == true ]]; then
            ((available_count++))
        fi
    done

    log_success "Backend detection complete: $available_count backends available"
}

# Build CPU-only baseline
build_cpu_baseline() {
    log_info "Building CPU baseline..."

    local build_dir="${BACKEND_BUILD_DIR["cpu"]}"
    mkdir -p "$build_dir"
    cd "$build_dir"

    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DGEMMA_BUILD_BACKENDS=OFF"
        "-DGEMMA_BUILD_ENHANCED_TESTS=ON"
        "-DGEMMA_BUILD_BENCHMARKS=ON"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )

    if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
        cmake_args+=("-DCMAKE_INSTALL_PREFIX=$BUILD_BASE_DIR/install/cpu")
    fi

    if [[ "$VERBOSE" == true ]]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi

    log_info "Configuring CPU build..."
    if cmake "${cmake_args[@]}" "$PROJECT_ROOT"; then
        log_info "Building CPU backend..."
        local parallel_jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}"

        if cmake --build . --config "$BUILD_TYPE" --parallel "$parallel_jobs"; then
            log_success "CPU baseline build completed"

            if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
                cmake --install . --config "$BUILD_TYPE"
                log_success "CPU baseline installed"
            fi
        else
            log_error "CPU baseline build failed"
            return 1
        fi
    else
        log_error "CPU baseline configuration failed"
        return 1
    fi
}

# Build specific backend
build_backend() {
    local backend="$1"

    if [[ "${BACKEND_AVAILABLE[$backend]}" != true ]]; then
        if [[ "$SKIP_MISSING" == true ]]; then
            log_warning "Skipping $backend backend (not available)"
            return 0
        else
            log_error "$backend backend not available but required"
            return 1
        fi
    fi

    log_info "Building $backend backend..."

    local script="${BACKEND_SCRIPTS[$backend]}"
    if [[ -z "$script" || ! -f "$script" ]]; then
        log_warning "No build script available for $backend backend"
        return 0
    fi

    # Prepare build arguments
    local build_args=()

    if [[ "$BUILD_TYPE" == "Debug" ]]; then
        build_args+=("--debug")
    else
        build_args+=("--release")
    fi

    if [[ "$CLEAN_BUILD" == true ]]; then
        build_args+=("--clean")
    fi

    if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
        build_args+=("--install")
    fi

    if [[ "$RUN_TESTS" == true ]]; then
        build_args+=("--test")
    fi

    if [[ "$RUN_BENCHMARKS" == true ]]; then
        build_args+=("--benchmark")
    fi

    if [[ "$VERBOSE" == true ]]; then
        build_args+=("--verbose")
    fi

    # Override build directory to use our unified structure
    export OVERRIDE_BUILD_DIR="${BACKEND_BUILD_DIR[$backend]}"

    log_backend "Executing: $script ${build_args[*]}"

    if "$script" "${build_args[@]}"; then
        log_success "$backend backend build completed"
        return 0
    else
        log_error "$backend backend build failed"
        return 1
    fi
}

# Build all backends
build_all_backends() {
    log_info "Starting multi-backend build process..."

    local build_start_time=$(date +%s)
    local successful_builds=()
    local failed_builds=()

    # Always build CPU baseline first
    if build_cpu_baseline; then
        successful_builds+=("cpu")
    else
        failed_builds+=("cpu")
        if [[ "$SKIP_MISSING" != true ]]; then
            log_error "CPU baseline build failed, aborting"
            exit 1
        fi
    fi

    # Build other backends
    local backends_to_build=()
    for backend in "${!BACKEND_AVAILABLE[@]}"; do
        if [[ "$backend" != "cpu" && "${BACKEND_AVAILABLE[$backend]}" == true ]]; then
            backends_to_build+=("$backend")
        fi
    done

    if [[ "$PARALLEL_BUILD" == true ]]; then
        log_info "Building backends in parallel: ${backends_to_build[*]}"

        # Build backends in parallel using background processes
        declare -A build_pids

        for backend in "${backends_to_build[@]}"; do
            log_info "Starting parallel build for $backend..."
            build_backend "$backend" &
            build_pids["$backend"]=$!
        done

        # Wait for all parallel builds to complete
        for backend in "${!build_pids[@]}"; do
            local pid=${build_pids["$backend"]}
            if wait "$pid"; then
                successful_builds+=("$backend")
                log_success "$backend backend completed (parallel)"
            else
                failed_builds+=("$backend")
                log_error "$backend backend failed (parallel)"
            fi
        done
    else
        # Build backends sequentially
        for backend in "${backends_to_build[@]}"; do
            if build_backend "$backend"; then
                successful_builds+=("$backend")
            else
                failed_builds+=("$backend")
                if [[ "$SKIP_MISSING" != true ]]; then
                    log_error "Backend build failed, aborting"
                    exit 1
                fi
            fi
        done
    fi

    local build_end_time=$(date +%s)
    local total_build_time=$((build_end_time - build_start_time))

    # Report build results
    log_info "Multi-backend build completed in ${total_build_time} seconds"
    log_success "Successful builds (${#successful_builds[@]}): ${successful_builds[*]}"

    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        log_warning "Failed builds (${#failed_builds[@]}): ${failed_builds[*]}"
    fi
}

# Run comprehensive benchmarks
run_comprehensive_benchmarks() {
    if [[ "$RUN_BENCHMARKS" != true ]]; then
        return
    fi

    log_benchmark "Running comprehensive backend comparison..."

    local report_file="$BUILD_BASE_DIR/reports/backend_comparison_$(date +%Y%m%d_%H%M%S).csv"
    echo "Backend,Operation,Size,Time_ms,GFLOPS,Bandwidth_GBps,Memory_MB" > "$report_file"

    # Collect benchmark results from all successful builds
    for backend_dir in "$BUILD_BASE_DIR"/*; do
        if [[ -d "$backend_dir" && -f "$backend_dir/tests/backends/benchmark_backends" ]]; then
            local backend_name=$(basename "$backend_dir")
            log_benchmark "Collecting results for $backend_name backend..."

            # Run benchmark and append results
            cd "$backend_dir"
            if ./tests/backends/benchmark_backends > "benchmark_${backend_name}.csv" 2>/dev/null; then
                tail -n +2 "benchmark_${backend_name}.csv" >> "$report_file" 2>/dev/null || true
            fi
        fi
    done

    # Generate summary report
    generate_benchmark_summary "$report_file"
}

# Generate benchmark summary
generate_benchmark_summary() {
    local report_file="$1"
    local summary_file="$BUILD_BASE_DIR/reports/benchmark_summary.txt"

    log_benchmark "Generating benchmark summary: $summary_file"

    cat > "$summary_file" << EOF
=================================================================
GEMMA.CPP MULTI-BACKEND PERFORMANCE COMPARISON
=================================================================
Generated: $(date)
Build Type: $BUILD_TYPE
System: $(uname -a)

EOF

    if [[ -f "$report_file" ]]; then
        echo "Detailed results available in: $report_file" >> "$summary_file"
        echo "" >> "$summary_file"

        # Extract unique operations and show best backend for each
        local operations=$(tail -n +2 "$report_file" | cut -d',' -f2 | sort -u)

        echo "PERFORMANCE WINNERS BY OPERATION:" >> "$summary_file"
        echo "=================================" >> "$summary_file"

        for operation in $operations; do
            local best_backend=$(grep "$operation" "$report_file" | sort -t',' -k5 -nr | head -1 | cut -d',' -f1)
            local best_gflops=$(grep "$operation" "$report_file" | sort -t',' -k5 -nr | head -1 | cut -d',' -f5)
            echo "$operation: $best_backend ($best_gflops GFLOPS)" >> "$summary_file"
        done
    fi

    echo "" >> "$summary_file"
    echo "BUILD SUMMARY:" >> "$summary_file"
    echo "==============" >> "$summary_file"

    for backend in "${!BACKEND_AVAILABLE[@]}"; do
        if [[ "${BACKEND_AVAILABLE[$backend]}" == true ]]; then
            echo "✓ $backend: Available and built" >> "$summary_file"
        else
            echo "✗ $backend: Not available" >> "$summary_file"
        fi
    done

    log_benchmark "Benchmark summary generated: $summary_file"
}

# Print comprehensive summary
print_comprehensive_summary() {
    log_info "==================================================================="
    log_info "               MULTI-BACKEND BUILD SUMMARY"
    log_info "==================================================================="

    echo "Build Configuration:"
    echo "  Project Root: $PROJECT_ROOT"
    echo "  Build Base Directory: $BUILD_BASE_DIR"
    echo "  Build Type: $BUILD_TYPE"
    echo "  Parallel Build: $PARALLEL_BUILD"
    echo "  Skip Missing: $SKIP_MISSING"
    echo ""

    echo "Backend Status:"
    for backend in cpu sycl cuda vulkan opencl metal; do
        if [[ "${BACKEND_AVAILABLE[$backend]:-false}" == true ]]; then
            local build_dir="${BACKEND_BUILD_DIR[$backend]}"
            if [[ -d "$build_dir" ]]; then
                echo "  ✓ $backend: Built successfully ($build_dir)"
            else
                echo "  ⚠ $backend: Available but not built"
            fi
        else
            echo "  ✗ $backend: Not available"
        fi
    done

    echo ""
    echo "Available Executables:"
    for backend_dir in "$BUILD_BASE_DIR"/*; do
        if [[ -d "$backend_dir" ]]; then
            local backend_name=$(basename "$backend_dir")
            local gemma_exe="$backend_dir/gemma"
            if [[ -f "$gemma_exe" ]]; then
                echo "  $backend_name: $gemma_exe"
            fi
        fi
    done

    if [[ "$RUN_BENCHMARKS" == true ]]; then
        echo ""
        echo "Benchmark Reports:"
        echo "  Summary: $BUILD_BASE_DIR/reports/benchmark_summary.txt"
        echo "  Detailed: $BUILD_BASE_DIR/reports/backend_comparison_*.csv"
    fi

    echo ""
    log_info "Usage Examples:"
    echo "  # Run with best available backend:"
    echo "  $BUILD_BASE_DIR/cuda/gemma --weights model.sbs  # (if CUDA available)"
    echo "  $BUILD_BASE_DIR/vulkan/gemma --weights model.sbs  # (if Vulkan available)"
    echo "  $BUILD_BASE_DIR/cpu/gemma --weights model.sbs  # (CPU baseline)"
    echo ""
    echo "  # Compare performance:"
    echo "  $BUILD_BASE_DIR/*/tests/backends/benchmark_backends"
    echo ""
    echo "  # Run tests:"
    echo "  cd $BUILD_BASE_DIR/cuda && ctest  # (for specific backend)"

    log_success "Multi-backend build process completed!"
}

# Main execution function
main() {
    echo "============================================================"
    echo "  Gemma.cpp Multi-Backend Build System"
    echo "============================================================"

    parse_args "$@"
    setup_base_build_directory
    detect_backends
    build_all_backends
    run_comprehensive_benchmarks
    print_comprehensive_summary
}

# Trap errors and cleanup
trap 'log_error "Multi-backend build process interrupted or failed"' ERR INT TERM

# Execute main function
main "$@"