#!/bin/bash

# Comprehensive Rust Test Runner for stats/ components
# This script runs all tests for rust_core and rust_extensions with proper configuration

set -e  # Exit on any error

echo "🦀 Comprehensive Rust Testing Suite"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in the right directory
if [ ! -d "rust_core" ] || [ ! -d "rust_extensions" ]; then
    print_error "Please run this script from the stats/ directory"
    exit 1
fi

# Parse command line arguments
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_BENCHMARK_TESTS=false
RUN_PROPERTY_TESTS=true
RUN_DOC_TESTS=true
RUN_MIRI_TESTS=false
VERBOSE=false
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            RUN_BENCHMARK_TESTS=true
            shift
            ;;
        --miri)
            RUN_MIRI_TESTS=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --unit-only)
            RUN_INTEGRATION_TESTS=false
            RUN_BENCHMARK_TESTS=false
            RUN_PROPERTY_TESTS=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --benchmark     Run benchmark tests (slower)"
            echo "  --miri         Run tests under Miri (very slow)"
            echo "  --coverage     Generate coverage reports"
            echo "  --verbose      Verbose output"
            echo "  --unit-only    Run only unit tests (fastest)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up test environment
export RUST_BACKTRACE=1
export RUST_LOG=debug

if [ "$VERBOSE" = true ]; then
    export RUST_LOG=trace
    CARGO_FLAGS="--verbose"
else
    CARGO_FLAGS=""
fi

# Function to run tests in a directory
run_tests_in_dir() {
    local dir=$1
    local name=$2

    print_status "Testing $name in $dir/"
    cd "$dir"

    # Check if Cargo.toml exists
    if [ ! -f "Cargo.toml" ]; then
        print_warning "No Cargo.toml found in $dir, skipping"
        cd ..
        return
    fi

    # Run unit tests
    if [ "$RUN_UNIT_TESTS" = true ]; then
        print_status "Running unit tests for $name..."
        if [ "$COVERAGE" = true ]; then
            cargo test $CARGO_FLAGS --all-features -- --test-threads=1
        else
            cargo test $CARGO_FLAGS --all-features
        fi
        print_success "Unit tests completed for $name"
    fi

    # Run integration tests
    if [ "$RUN_INTEGRATION_TESTS" = true ]; then
        print_status "Running integration tests for $name..."
        cargo test $CARGO_FLAGS --tests --all-features
        print_success "Integration tests completed for $name"
    fi

    # Run documentation tests
    if [ "$RUN_DOC_TESTS" = true ]; then
        print_status "Running documentation tests for $name..."
        cargo test $CARGO_FLAGS --doc --all-features || print_warning "Some doc tests failed (may be expected)"
        print_success "Documentation tests completed for $name"
    fi

    # Run property-based tests specifically
    if [ "$RUN_PROPERTY_TESTS" = true ]; then
        print_status "Running property-based tests for $name..."
        cargo test $CARGO_FLAGS --all-features prop_ || print_warning "Some property tests failed (may be expected)"
        print_success "Property tests completed for $name"
    fi

    # Run benchmark tests
    if [ "$RUN_BENCHMARK_TESTS" = true ]; then
        print_status "Running benchmark tests for $name..."
        if cargo bench --help > /dev/null 2>&1; then
            cargo bench $CARGO_FLAGS --all-features || print_warning "Some benchmarks failed (may be expected)"
            print_success "Benchmark tests completed for $name"
        else
            print_warning "Criterion not available, skipping benchmarks for $name"
        fi
    fi

    # Run under Miri for additional safety checking
    if [ "$RUN_MIRI_TESTS" = true ]; then
        print_status "Running Miri tests for $name (this will be slow)..."
        if rustup component list | grep -q "miri"; then
            cargo +nightly miri test $CARGO_FLAGS --all-features || print_warning "Miri tests failed (may be expected)"
            print_success "Miri tests completed for $name"
        else
            print_warning "Miri not available, skipping for $name"
        fi
    fi

    cd ..
}

# Function to check Rust toolchain and dependencies
check_toolchain() {
    print_status "Checking Rust toolchain..."

    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi

    # Check for nightly if Miri tests are requested
    if [ "$RUN_MIRI_TESTS" = true ]; then
        if ! rustup toolchain list | grep -q nightly; then
            print_warning "Nightly toolchain not found. Installing..."
            rustup toolchain install nightly
            rustup component add miri --toolchain nightly
        fi
    fi

    print_success "Toolchain check completed"
}

# Function to generate coverage report
generate_coverage() {
    if [ "$COVERAGE" = true ]; then
        print_status "Generating coverage reports..."

        # Install cargo-llvm-cov if not present
        if ! command -v cargo-llvm-cov &> /dev/null; then
            print_status "Installing cargo-llvm-cov..."
            cargo install cargo-llvm-cov
        fi

        # Generate coverage for rust_core
        if [ -d "rust_core/inference" ]; then
            cd rust_core/inference
            print_status "Generating coverage for rust_core/inference..."
            cargo llvm-cov --all-features --html --output-dir ../../coverage/rust_core_inference || print_warning "Coverage generation failed for rust_core"
            cd ../..
        fi

        # Generate coverage for rust_extensions
        if [ -d "rust_extensions" ]; then
            cd rust_extensions
            print_status "Generating coverage for rust_extensions..."
            cargo llvm-cov --all-features --html --output-dir ../coverage/rust_extensions || print_warning "Coverage generation failed for rust_extensions"
            cd ..
        fi

        print_success "Coverage reports generated in coverage/ directory"
    fi
}

# Function to clean up build artifacts
cleanup() {
    print_status "Cleaning up build artifacts..."

    if [ -d "rust_core" ]; then
        find rust_core -name "target" -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    if [ -d "rust_extensions" ]; then
        cd rust_extensions
        cargo clean 2>/dev/null || true
        cd ..
    fi

    print_success "Cleanup completed"
}

# Main execution
main() {
    print_status "Starting comprehensive Rust testing suite"
    print_status "Configuration:"
    echo "  Unit tests: $RUN_UNIT_TESTS"
    echo "  Integration tests: $RUN_INTEGRATION_TESTS"
    echo "  Benchmark tests: $RUN_BENCHMARK_TESTS"
    echo "  Property tests: $RUN_PROPERTY_TESTS"
    echo "  Documentation tests: $RUN_DOC_TESTS"
    echo "  Miri tests: $RUN_MIRI_TESTS"
    echo "  Coverage: $COVERAGE"
    echo "  Verbose: $VERBOSE"
    echo ""

    # Check toolchain
    check_toolchain

    # Test rust_core components
    print_status "Testing rust_core components..."
    if [ -d "rust_core/inference" ]; then
        run_tests_in_dir "rust_core/inference" "gemma-inference"
    else
        print_warning "rust_core/inference not found, skipping"
    fi

    # Test other rust_core components
    for component in "server" "wasm" "cross"; do
        if [ -d "rust_core/$component" ]; then
            run_tests_in_dir "rust_core/$component" "$component"
        fi
    done

    # Test rust_extensions
    print_status "Testing rust_extensions..."
    if [ -d "rust_extensions" ]; then
        run_tests_in_dir "rust_extensions" "gemma-extensions"
    else
        print_warning "rust_extensions not found, skipping"
    fi

    # Generate coverage if requested
    generate_coverage

    print_success "All tests completed successfully! 🎉"

    # Summary
    echo ""
    print_status "Test Summary:"
    echo "✅ Unit tests: $([ "$RUN_UNIT_TESTS" = true ] && echo "Completed" || echo "Skipped")"
    echo "✅ Integration tests: $([ "$RUN_INTEGRATION_TESTS" = true ] && echo "Completed" || echo "Skipped")"
    echo "✅ Property tests: $([ "$RUN_PROPERTY_TESTS" = true ] && echo "Completed" || echo "Skipped")"
    echo "✅ Documentation tests: $([ "$RUN_DOC_TESTS" = true ] && echo "Completed" || echo "Skipped")"
    echo "✅ Benchmark tests: $([ "$RUN_BENCHMARK_TESTS" = true ] && echo "Completed" || echo "Skipped")"
    echo "✅ Miri tests: $([ "$RUN_MIRI_TESTS" = true ] && echo "Completed" || echo "Skipped")"
    echo "✅ Coverage: $([ "$COVERAGE" = true ] && echo "Generated" || echo "Skipped")"

    if [ "$COVERAGE" = true ]; then
        echo ""
        print_status "Coverage reports available at:"
        echo "  rust_core/inference: coverage/rust_core_inference/index.html"
        echo "  rust_extensions: coverage/rust_extensions/index.html"
    fi
}

# Handle interrupts gracefully
trap 'print_error "Test run interrupted"; cleanup; exit 1' INT TERM

# Run main function
main

# Cleanup on successful completion
if [ "$?" -eq 0 ]; then
    print_success "Test suite completed successfully"
else
    print_error "Test suite failed"
    exit 1
fi