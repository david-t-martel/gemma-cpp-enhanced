#!/bin/bash

# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive test runner for Gemma.cpp test suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
TEST_MODEL_PATH="/c/codedev/llm/.models"
REQUIRED_FILES=("tokenizer.spm" "gemma2-2b-it-sfp.sbs")

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if we're in the right directory
    if [[ ! -f "CMakeLists.txt" ]]; then
        print_error "CMakeLists.txt not found. Please run from the tests directory."
        exit 1
    fi

    # Check for required tools
    local tools=("cmake" "make" "g++")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done

    # Check for model files (optional but recommended)
    local missing_files=()
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$TEST_MODEL_PATH/$file" ]]; then
            missing_files+=("$file")
        fi
    done

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_warning "Some model files are missing from $TEST_MODEL_PATH:"
        for file in "${missing_files[@]}"; do
            print_warning "  - $file"
        done
        print_warning "Integration tests may be skipped."
    else
        print_success "All model files found."
    fi

    print_success "Prerequisites check completed."
}

# Function to build tests
build_tests() {
    print_status "Building tests..."

    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure with CMake
    print_status "Configuring CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Release

    # Build tests
    print_status "Compiling tests..."
    make -j$(nproc)

    cd ..
    print_success "Tests built successfully."
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running unit tests..."

    cd "$BUILD_DIR"

    local unit_tests=("test_model_loading" "test_tokenization" "test_memory_management" "test_error_handling")
    local passed=0
    local failed=0

    for test in "${unit_tests[@]}"; do
        print_status "Running $test..."
        if ./tests/"$test"; then
            print_success "$test passed"
            ((passed++))
        else
            print_error "$test failed"
            ((failed++))
        fi
        echo
    done

    cd ..

    print_status "Unit tests summary: $passed passed, $failed failed"
    return $failed
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."

    cd "$BUILD_DIR"

    local integration_tests=("test_inference")
    local passed=0
    local failed=0

    for test in "${integration_tests[@]}"; do
        print_status "Running $test..."
        if ./tests/"$test"; then
            print_success "$test passed"
            ((passed++))
        else
            print_error "$test failed"
            ((failed++))
        fi
        echo
    done

    cd ..

    print_status "Integration tests summary: $passed passed, $failed failed"
    return $failed
}

# Function to run performance benchmarks
run_benchmarks() {
    print_status "Running performance benchmarks..."

    cd "$BUILD_DIR"

    if [[ -f "tests/test_performance" ]]; then
        print_status "Running performance benchmarks..."
        ./tests/test_performance --benchmark_format=console --benchmark_out=benchmark_results.json --benchmark_out_format=json
        print_success "Benchmarks completed. Results saved to benchmark_results.json"
    else
        print_error "Performance test binary not found"
        cd ..
        return 1
    fi

    cd ..
    return 0
}

# Function to run all tests with CTest
run_ctest() {
    print_status "Running all tests with CTest..."

    cd "$BUILD_DIR"

    if ctest --output-on-failure --verbose; then
        print_success "All CTest tests passed"
        cd ..
        return 0
    else
        print_error "Some CTest tests failed"
        cd ..
        return 1
    fi
}

# Function to generate coverage report
generate_coverage() {
    print_status "Generating coverage report..."

    cd "$BUILD_DIR"

    if make coverage 2>/dev/null; then
        print_success "Coverage report generated in coverage/"
        if command -v xdg-open &> /dev/null; then
            xdg-open coverage/index.html &
        fi
    else
        print_warning "Coverage generation not available (requires Debug build with gcov/lcov)"
    fi

    cd ..
}

# Function to clean build artifacts
clean_build() {
    print_status "Cleaning build artifacts..."

    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    else
        print_warning "Build directory does not exist"
    fi
}

# Function to show help
show_help() {
    cat << EOF
Gemma.cpp Test Runner

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    all         Run all tests (default)
    unit        Run only unit tests
    integration Run only integration tests
    benchmarks  Run performance benchmarks
    ctest       Run tests using CTest
    coverage    Generate coverage report (Debug builds only)
    clean       Clean build artifacts
    build       Build tests only
    help        Show this help message

Options:
    -v, --verbose   Enable verbose output
    -j, --jobs N    Use N parallel jobs for building (default: nproc)
    --debug         Build with debug information
    --no-check      Skip prerequisite checks

Examples:
    $0              # Run all tests
    $0 unit         # Run only unit tests
    $0 benchmarks   # Run performance benchmarks
    $0 clean build  # Clean and rebuild
    $0 --debug coverage  # Debug build with coverage

EOF
}

# Parse command line arguments
VERBOSE=false
JOBS=$(nproc)
DEBUG=false
NO_CHECK=false
COMMANDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --no-check)
            NO_CHECK=true
            shift
            ;;
        all|unit|integration|benchmarks|ctest|coverage|clean|build|help)
            COMMANDS+=("$1")
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default command if none specified
if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    COMMANDS=("all")
fi

# Set build type
if [[ "$DEBUG" == true ]]; then
    BUILD_TYPE="Debug"
else
    BUILD_TYPE="Release"
fi

# Main execution
main() {
    local overall_result=0

    print_status "Gemma.cpp Test Suite Runner"
    print_status "Build type: $BUILD_TYPE"
    print_status "Parallel jobs: $JOBS"
    echo

    # Check prerequisites unless skipped
    if [[ "$NO_CHECK" != true ]]; then
        check_prerequisites
        echo
    fi

    # Execute commands
    for cmd in "${COMMANDS[@]}"; do
        case $cmd in
            help)
                show_help
                ;;
            clean)
                clean_build
                ;;
            build)
                build_tests
                ;;
            unit)
                build_tests
                if ! run_unit_tests; then
                    overall_result=1
                fi
                ;;
            integration)
                build_tests
                if ! run_integration_tests; then
                    overall_result=1
                fi
                ;;
            benchmarks)
                build_tests
                if ! run_benchmarks; then
                    overall_result=1
                fi
                ;;
            ctest)
                build_tests
                if ! run_ctest; then
                    overall_result=1
                fi
                ;;
            coverage)
                # Force debug build for coverage
                BUILD_TYPE="Debug"
                build_tests
                run_unit_tests
                generate_coverage
                ;;
            all)
                build_tests
                print_status "Running comprehensive test suite..."
                echo

                if ! run_unit_tests; then
                    overall_result=1
                fi
                echo

                if ! run_integration_tests; then
                    overall_result=1
                fi
                ;;
        esac
        echo
    done

    # Final summary
    if [[ $overall_result -eq 0 ]]; then
        print_success "All requested operations completed successfully!"
    else
        print_error "Some operations failed. Check the output above for details."
    fi

    return $overall_result
}

# Run main function
main "$@"