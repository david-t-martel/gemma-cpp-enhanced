#!/bin/bash
# Gemma.cpp Windows Optimized Build Script
# This script configures and builds gemma.cpp with optimal settings for Windows

# Default parameters
BUILD_TYPE="Release"
BUILD_DIR="build-windows-optimized"
CLEAN=false
VERBOSE=false
TEST=false
USE_SCALAR_FALLBACK=true
JOBS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --test)
            TEST=true
            shift
            ;;
        --no-scalar-fallback)
            USE_SCALAR_FALLBACK=false
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build-type TYPE    Build type (Release, Debug, RelWithDebInfo)"
            echo "  --build-dir DIR      Build directory name"
            echo "  --clean              Clean build directory before building"
            echo "  --verbose            Enable verbose output"
            echo "  --test               Enable tests"
            echo "  --no-scalar-fallback Disable scalar fallback"
            echo "  --jobs N             Number of parallel jobs"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Color output functions
print_success() {
    echo -e "\033[32m✓ $1\033[0m"
}

print_info() {
    echo -e "\033[34mℹ $1\033[0m"
}

print_warning() {
    echo -e "\033[33m⚠ $1\033[0m"
}

print_error() {
    echo -e "\033[31m✗ $1\033[0m"
}

# Main build function
build_gemma() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"
    print_info "Working directory: $(pwd)"

    # Clean build directory if requested
    if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
        print_info "Cleaning existing build directory..."
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    fi

    # Create build directory
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        print_success "Created build directory: $BUILD_DIR"
    fi

    # Determine number of parallel jobs
    if [ "$JOBS" -eq 0 ]; then
        JOBS=$(nproc 2>/dev/null || echo 4)
        print_info "Using $JOBS parallel jobs (auto-detected)"
    else
        print_info "Using $JOBS parallel jobs (user-specified)"
    fi

    # Check for required tools
    print_info "Checking for required tools..."

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        if [ -f "/c/Program Files/CMake/bin/cmake" ]; then
            CMAKE_PATH="/c/Program Files/CMake/bin/cmake"
        else
            print_error "CMake not found. Please install CMake or add it to PATH."
            return 1
        fi
    else
        CMAKE_PATH="cmake"
    fi
    print_success "Found CMake: $CMAKE_PATH"

    # Check for ccache
    if command -v ccache &> /dev/null; then
        print_success "Found ccache: $(which ccache)"
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        export CMAKE_C_COMPILER_LAUNCHER=ccache
    else
        print_warning "ccache not found. Build will be slower without caching."
    fi

    # Copy optimized CMakeLists.txt if it exists
    if [ -f "CMakeLists_optimized.txt" ]; then
        print_info "Using optimized CMakeLists.txt..."
        cp "CMakeLists_optimized.txt" "CMakeLists.txt"
        print_success "Optimized CMakeLists.txt applied"
    fi

    # Prepare CMake arguments
    cmake_args=(
        "-B" "$BUILD_DIR"
        "-G" "Visual Studio 17 2022"
        "-T" "v143"
        "-A" "x64"
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DSPM_ENABLE_SHARED=OFF"
        "-DSPM_ABSL_PROVIDER=module"
        "-DHWY_ENABLE_TESTS=OFF"
        "-DHWY_ENABLE_EXAMPLES=OFF"
        "-DBENCHMARK_ENABLE_TESTING=OFF"
        "-DBENCHMARK_ENABLE_GTEST_TESTS=OFF"
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    )

    # Add scalar fallback option
    if [ "$USE_SCALAR_FALLBACK" = true ]; then
        cmake_args+=("-DGEMMA_USE_SCALAR_FALLBACK=ON")
        print_info "Scalar fallback enabled"
    else
        cmake_args+=("-DGEMMA_USE_SCALAR_FALLBACK=OFF")
    fi

    # Add test option
    if [ "$TEST" = true ]; then
        cmake_args+=("-DGEMMA_ENABLE_TESTS=ON")
        print_info "Tests enabled"
    fi

    # Configure the build
    print_info "Configuring build with CMake..."
    print_info "CMake command: $CMAKE_PATH ${cmake_args[*]}"

    if ! "$CMAKE_PATH" "${cmake_args[@]}"; then
        print_error "CMake configuration failed"
        return 1
    fi
    print_success "CMake configuration completed successfully"

    # Build the project
    print_info "Building project..."
    build_args=(
        "--build" "$BUILD_DIR"
        "--config" "$BUILD_TYPE"
        "--parallel" "$JOBS"
    )

    if [ "$VERBOSE" = true ]; then
        build_args+=("--verbose")
    fi

    print_info "Build command: $CMAKE_PATH ${build_args[*]}"

    if ! "$CMAKE_PATH" "${build_args[@]}"; then
        print_error "Build failed"
        return 1
    fi
    print_success "Build completed successfully"

    # Check if executable was created
    exe_path="$BUILD_DIR/$BUILD_TYPE/gemma.exe"
    if [ -f "$exe_path" ]; then
        print_success "Gemma executable created: $exe_path"

        # Get file info
        size_mb=$(du -m "$exe_path" | cut -f1)
        print_info "Executable size: ${size_mb} MB"
        print_info "Created: $(stat -c %y "$exe_path" 2>/dev/null || stat -f %Sm "$exe_path" 2>/dev/null || echo "unknown")"
    else
        print_warning "Gemma executable not found at expected location: $exe_path"

        # Try to find it in other locations
        search_paths=(
            "$BUILD_DIR/gemma.exe"
            "$BUILD_DIR/Debug/gemma.exe"
            "$BUILD_DIR/RelWithDebInfo/gemma.exe"
        )

        for path in "${search_paths[@]}"; do
            if [ -f "$path" ]; then
                print_success "Found executable at: $path"
                exe_path="$path"
                break
            fi
        done
    fi

    # Run tests if requested and executable exists
    if [ "$TEST" = true ] && [ -f "$exe_path" ]; then
        print_info "Running basic executable test..."
        if "$exe_path" --help &>/dev/null || echo "$?" | grep -q "0\|1"; then
            print_success "Executable test passed"
        else
            print_warning "Executable test may have failed"
        fi
    fi

    # Summary
    print_success "=== BUILD SUMMARY ==="
    print_info "Build Type: $BUILD_TYPE"
    print_info "Build Directory: $BUILD_DIR"
    print_info "Parallel Jobs: $JOBS"
    print_info "Scalar Fallback: $USE_SCALAR_FALLBACK"
    if [ -f "$exe_path" ]; then
        print_success "Executable: $exe_path"
    fi

    # Check for other built targets
    other_exes=("single_benchmark.exe" "benchmarks.exe" "debug_prompt.exe" "migrate_weights.exe")
    for exe in "${other_exes[@]}"; do
        full_path="$BUILD_DIR/$BUILD_TYPE/$exe"
        if [ -f "$full_path" ]; then
            print_info "Additional target: $full_path"
        fi
    done

    print_success "Build process completed successfully!"
}

# CCCache configuration function
setup_ccache() {
    print_info "Setting up ccache configuration..."

    if command -v ccache &> /dev/null; then
        # Configure ccache for optimal performance
        ccache --set-config max_size=5G
        ccache --set-config compression=true
        ccache --set-config compression_level=6
        ccache --set-config base_dir="$(pwd)"

        # Show ccache status
        print_info "CCCache configuration:"
        ccache --show-config | grep -E "max_size|compression|base_dir" || true

        print_success "CCCache configured successfully"
    else
        print_warning "CCCache not found. Install it for faster rebuilds:"
        print_info "  choco install ccache"
        print_info "  or download from: https://ccache.dev/download.html"
    fi
}

# Environment check function
test_build_environment() {
    print_info "Testing build environment..."

    local issues=()

    # Check available memory (if available)
    if command -v free &> /dev/null; then
        local memory_gb=$(($(free -m | awk '/^Mem:/{print $2}') / 1024))
        if [ "$memory_gb" -lt 8 ]; then
            issues+=("At least 8GB RAM recommended (found: ${memory_gb}GB)")
        fi
    fi

    # Check disk space
    local free_space_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$free_space_gb" -lt 5 ]; then
        issues+=("At least 5GB free disk space required (found: ${free_space_gb}GB)")
    fi

    if [ ${#issues[@]} -eq 0 ]; then
        print_success "Build environment check passed"
    else
        print_warning "Build environment issues found:"
        for issue in "${issues[@]}"; do
            print_warning "  - $issue"
        done
    fi
}

# Main execution
main() {
    print_info "=== Gemma.cpp Windows Optimized Build Script ==="
    print_info "Build Type: $BUILD_TYPE"
    print_info "Build Directory: $BUILD_DIR"
    print_info "Clean: $CLEAN"
    print_info "Verbose: $VERBOSE"
    print_info "Test: $TEST"
    print_info "Use Scalar Fallback: $USE_SCALAR_FALLBACK"

    # Run environment check
    test_build_environment

    # Setup ccache if available
    setup_ccache

    # Run the build
    if build_gemma; then
        print_success "Script completed successfully!"
        exit 0
    else
        print_error "Script failed!"
        exit 1
    fi
}

# Run main function
main "$@"