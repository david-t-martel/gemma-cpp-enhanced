#!/bin/bash
# Direct functional test for MCP server

echo "============================================"
echo "Testing RAG-Redis MCP Server - REAL TESTS"
echo "============================================"

MCP_SERVER="./rag-redis-system/mcp-server/target/release/mcp-server.exe"
export REDIS_URL="redis://127.0.0.1:6380"

# Test 1: Initialize
echo -e "\n1. Testing Initialize..."
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05","capabilities":{},"client_info":{"name":"test","version":"1.0"}}}' | $MCP_SERVER 2>/dev/null | head -1

# Test 2: List tools
echo -e "\n2. Testing List Tools..."
echo -e '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05","capabilities":{},"client_info":{"name":"test","version":"1.0"}}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":null}' | $MCP_SERVER 2>/dev/null | tail -1 | jq '.result.tools | length'

# Test 3: Health check
echo -e "\n3. Testing Health Check..."
echo -e '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05","capabilities":{},"client_info":{"name":"test","version":"1.0"}}}\n{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"health_check","arguments":{}}}' | $MCP_SERVER 2>/dev/null | tail -1 | jq '.result'

echo -e "\n============================================"
echo "Tests Complete"
echo "============================================"