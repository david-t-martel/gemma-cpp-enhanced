#!/bin/bash
# Real functional test for RAG-Redis MCP Server

set -e

echo "=========================================="
echo "RAG-Redis MCP Server Functional Test Suite"
echo "=========================================="

# Check Redis is running
echo -n "Checking Redis connection... "
if redis-cli -p 6380 ping > /dev/null 2>&1; then
    echo "✓ Redis is running on port 6380"
else
    echo "✗ Redis is not running. Starting Redis..."
    redis-server --port 6380 &
    sleep 3
fi

# Build the MCP server
echo -n "Building MCP server... "
cd /c/codedev/llm/rag-redis/rag-redis-system/mcp-server
cargo build --release 2>/dev/null && echo "✓ Build successful" || echo "✗ Build failed"

# Create test input for MCP server
echo "Creating test inputs..."

# Test 1: Initialize MCP connection
cat > /tmp/test_init.json << 'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05","capabilities":{},"client_info":{"name":"test-client","version":"1.0.0"}}}
EOF

# Test 2: List available tools
cat > /tmp/test_tools.json << 'EOF'
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":null}
EOF

# Test 3: Health check
cat > /tmp/test_health.json << 'EOF'
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"health_check","arguments":{}}}
EOF

# Test 4: Ingest a document
cat > /tmp/test_ingest.json << 'EOF'
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ingest_document","arguments":{"content":"The RAG-Redis system provides high-performance vector search and multi-tier memory management for AI applications. It uses SIMD optimizations for fast similarity calculations.","metadata":{"title":"RAG-Redis Overview","type":"documentation"}}}}
EOF

# Test 5: Search for content
cat > /tmp/test_search.json << 'EOF'
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"search","arguments":{"query":"vector search performance","limit":5}}}
EOF

# Test 6: Store memory
cat > /tmp/test_memory_store.json << 'EOF'
{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"memory_store","arguments":{"content":"User prefers Rust-based implementations for performance","memory_type":"long_term","importance":0.9}}}
EOF

# Test 7: Recall memory
cat > /tmp/test_memory_recall.json << 'EOF'
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"memory_recall","arguments":{"query":"Rust implementations","memory_type":"long_term"}}}
EOF

echo ""
echo "Running functional tests..."
echo "=========================================="

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local expected_contains=$3

    echo -n "Testing $test_name... "

    # Send request to MCP server and capture response
    response=$(timeout 5 cat $test_file | ./target/release/rag-redis-mcp-server 2>/dev/null | head -1)

    if [[ $response == *"$expected_contains"* ]]; then
        echo "✓ PASS"
        echo "  Response: ${response:0:100}..."
        return 0
    else
        echo "✗ FAIL"
        echo "  Expected to contain: $expected_contains"
        echo "  Got: ${response:0:200}..."
        return 1
    fi
}

# Run all tests
passed=0
failed=0

if run_test "Initialize" "/tmp/test_init.json" '"protocol_version"'; then
    ((passed++))
else
    ((failed++))
fi

if run_test "List Tools" "/tmp/test_tools.json" '"tools"'; then
    ((passed++))
else
    ((failed++))
fi

if run_test "Health Check" "/tmp/test_health.json" '"status"'; then
    ((passed++))
else
    ((failed++))
fi

if run_test "Document Ingestion" "/tmp/test_ingest.json" '"document_id"'; then
    ((passed++))
else
    ((failed++))
fi

if run_test "Search" "/tmp/test_search.json" '"results"'; then
    ((passed++))
else
    ((failed++))
fi

if run_test "Memory Store" "/tmp/test_memory_store.json" '"memory_id"'; then
    ((passed++))
else
    ((failed++))
fi

if run_test "Memory Recall" "/tmp/test_memory_recall.json" '"memories"'; then
    ((passed++))
else
    ((failed++))
fi

echo ""
echo "=========================================="
echo "Test Results:"
echo "  Passed: $passed"
echo "  Failed: $failed"
echo "=========================================="

# Test Redis data persistence
echo ""
echo "Verifying Redis data persistence..."
echo -n "Checking stored documents... "
doc_count=$(redis-cli -p 6380 keys "doc:*" | wc -l)
echo "Found $doc_count documents"

echo -n "Checking stored embeddings... "
embed_count=$(redis-cli -p 6380 keys "embedding:*" | wc -l)
echo "Found $embed_count embeddings"

echo -n "Checking stored memories... "
mem_count=$(redis-cli -p 6380 keys "mem:*" | wc -l)
echo "Found $mem_count memories"

echo ""
if [ $failed -eq 0 ]; then
    echo "✅ All functional tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed. Please review the output above."
    exit 1
fi