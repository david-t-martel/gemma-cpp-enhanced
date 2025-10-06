#!/bin/bash
# Build script for RAG-Redis system integration (Linux/WSL)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATS_DIR="$(dirname "$SCRIPT_DIR")"
RAG_REDIS_DIR="$STATS_DIR/rag-redis-system"
RUST_EXT_DIR="$STATS_DIR/rust_extensions"

# Parse command line arguments
RELEASE=false
CLEAN=false
TEST=false
FEATURES=true
FEATURE="full"

while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            RELEASE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            TEST=true
            shift
            ;;
        --feature)
            FEATURE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== RAG-Redis System Build Script ==="
echo "Stats Dir: $STATS_DIR"
echo "RAG-Redis Dir: $RAG_REDIS_DIR"
echo "Rust Extensions Dir: $RUST_EXT_DIR"

# Function to check Redis availability
check_redis() {
    echo "Checking Redis connection..."
    if redis-cli ping >/dev/null 2>&1; then
        echo "✓ Redis is running"
        return 0
    else
        echo "✗ Redis not available. Please start Redis server on localhost:6379"
        echo "  WSL/Linux: sudo service redis-server start"
        echo "  Or: redis-server"
        return 1
    fi
}

# Function to build RAG-Redis system
build_rag_redis() {
    echo "=== Building RAG-Redis System ==="
    
    cd "$RAG_REDIS_DIR"
    
    if [[ "$CLEAN" == "true" ]]; then
        echo "Cleaning previous build..."
        cargo clean
    fi
    
    BUILD_ARGS=("build")
    if [[ "$RELEASE" == "true" ]]; then
        BUILD_ARGS+=("--release")
    fi
    
    if [[ "$FEATURES" == "true" ]]; then
        BUILD_ARGS+=("--features" "$FEATURE")
    fi
    
    echo "Running: cargo ${BUILD_ARGS[*]}"
    cargo "${BUILD_ARGS[@]}"
    
    echo "✓ RAG-Redis system built successfully"
}

# Function to build Rust extensions
build_rust_extensions() {
    echo "=== Building Rust Extensions for Python ==="
    
    cd "$RUST_EXT_DIR"
    
    if [[ "$CLEAN" == "true" ]]; then
        echo "Cleaning previous build..."
        cargo clean
    fi
    
    echo "Building with maturin..."
    BUILD_ARGS=("develop")
    if [[ "$RELEASE" == "true" ]]; then
        BUILD_ARGS+=("--release")
    fi
    
    uv run maturin "${BUILD_ARGS[@]}"
    
    echo "✓ Rust extensions built successfully"
}

# Function to run tests
run_tests() {
    echo "=== Running Tests ==="
    
    # Test RAG-Redis system
    echo "Testing RAG-Redis system..."
    cd "$RAG_REDIS_DIR"
    
    TEST_ARGS=("test")
    if [[ "$FEATURES" == "true" ]]; then
        TEST_ARGS+=("--features" "$FEATURE")
    fi
    
    if cargo "${TEST_ARGS[@]}"; then
        echo "✓ RAG-Redis tests passed"
    else
        echo "⚠ Some RAG-Redis tests failed"
    fi
    
    # Test Rust extensions
    echo "Testing Rust extensions..."
    cd "$RUST_EXT_DIR"
    
    if cargo test; then
        echo "✓ Rust extension tests passed"
    else
        echo "⚠ Some Rust extension tests failed"
    fi
    
    # Test Python integration
    echo "Testing Python integration..."
    cd "$STATS_DIR"
    
    if uv run python -c "import gemma_extensions; print('✓ Python extensions imported successfully')"; then
        echo "✓ Python integration test passed"
    else
        echo "⚠ Python integration test failed"
    fi
}

# Main execution
main() {
    # Check prerequisites
    if ! command -v cargo &> /dev/null; then
        echo "Error: Rust/Cargo not found. Please install Rust toolchain."
        exit 1
    fi
    
    if ! command -v uv &> /dev/null; then
        echo "Error: uv not found. Please install uv Python package manager."
        exit 1
    fi
    
    # Check Redis if we're running tests
    if [[ "$TEST" == "true" ]] && ! check_redis; then
        echo "Warning: Redis not available - integration tests may fail"
    fi
    
    # Build components
    build_rag_redis
    build_rust_extensions
    
    # Run tests if requested
    if [[ "$TEST" == "true" ]]; then
        run_tests
    fi
    
    echo ""
    echo "=== Build Complete ==="
    echo "✓ RAG-Redis system ready for integration"
    echo "✓ Python bindings available"
    
    if [[ "$RELEASE" == "true" ]]; then
        echo "✓ Release build artifacts created"
    fi
    
    # Show next steps
    echo ""
    echo "=== Next Steps ==="
    echo "1. Start Redis server: redis-server"
    echo "2. Run demo: uv run python examples/rag_integration_demo.py"
    echo "3. Run tests: ./scripts/test-rag-integration.sh"
}

main "$@"
