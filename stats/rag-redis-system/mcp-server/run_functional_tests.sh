#!/bin/bash
# Functional Test Runner for RAG-Redis MCP Server

set -e

echo "🚀 RAG-Redis MCP Server Functional Test Runner"
echo "=" | head -c 60
echo ""

# Check if Redis is available
if ! command -v redis-server &> /dev/null; then
    echo "❌ Redis server not found. Please install Redis first."
    echo "   Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   macOS: brew install redis"
    echo "   Windows: Download from https://redis.io/download"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Please run this script from the mcp-server directory"
    exit 1
fi

# Add redis dependency temporarily for tests
echo "📦 Adding test dependencies..."
if ! grep -q "redis = " Cargo.toml; then
    sed -i '/\[dev-dependencies\]/a redis = "0.24"' Cargo.toml
fi

# Add functional test binary if not present
if ! grep -q "functional-test" Cargo.toml; then
    cat >> Cargo.toml << 'EOF'

[[bin]]
name = "functional-test"
path = "tests/functional_test.rs"
EOF
fi

echo "🔧 Building test suite..."
cargo build --bin functional-test

echo "🧪 Running functional tests..."
echo "   Note: This will start a Redis instance for testing"
echo ""

# Run the functional tests
cargo run --bin functional-test

echo ""
echo "✅ Functional tests completed!"
echo "📊 Check the output above for detailed results"