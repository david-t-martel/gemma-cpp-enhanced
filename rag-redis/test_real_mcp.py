#!/usr/bin/env python3
"""
Real functional test for RAG-Redis MCP Server
Tests actual functionality, not mocks
"""

import json
import subprocess
import time
import sys
import os

def send_mcp_request(request):
    """Send a request to the MCP server and get response"""
    mcp_server_path = r"c:\codedev\llm\rag-redis\rag-redis-system\mcp-server\target\release\mcp-server.exe"

    # Set environment for Redis connection
    env = os.environ.copy()
    env['REDIS_URL'] = 'redis://127.0.0.1:6380'
    env['RUST_LOG'] = 'info'

    try:
        # Send request to MCP server
        process = subprocess.Popen(
            [mcp_server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        # Send request and get response
        stdout, stderr = process.communicate(input=json.dumps(request) + '\n', timeout=5)

        # Parse response
        if stdout:
            return json.loads(stdout.strip())
        else:
            print(f"Error: {stderr}")
            return None
    except subprocess.TimeoutExpired:
        process.kill()
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_mcp_server():
    """Run comprehensive functional tests"""
    print("=" * 60)
    print("RAG-Redis MCP Server - Real Functional Tests")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Initialize MCP connection
    print("\n1. Testing MCP Initialize...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocol_version": "2024-11-05",
            "capabilities": {},
            "client_info": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    response = send_mcp_request(init_request)
    if response and "result" in response:
        print(f"   ✓ Initialize successful: {response['result'].get('protocol_version', 'unknown')}")
        tests_passed += 1
    else:
        print(f"   ✗ Initialize failed: {response}")
        tests_failed += 1

    # Test 2: List available tools
    print("\n2. Testing List Tools...")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": None
    }

    response = send_mcp_request(tools_request)
    if response and "result" in response and "tools" in response["result"]:
        tools = response["result"]["tools"]
        print(f"   ✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"      - {tool['name']}: {tool.get('description', 'No description')[:50]}...")
        tests_passed += 1
    else:
        print(f"   ✗ List tools failed: {response}")
        tests_failed += 1

    # Test 3: Health check
    print("\n3. Testing Health Check...")
    health_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "health_check",
            "arguments": {}
        }
    }

    response = send_mcp_request(health_request)
    if response and "result" in response:
        print(f"   ✓ Health check successful")
        if "content" in response["result"]:
            content = json.loads(response["result"]["content"][0]["text"])
            print(f"      Status: {content.get('status', 'unknown')}")
            print(f"      Redis: {content.get('redis_connected', False)}")
        tests_passed += 1
    else:
        print(f"   ✗ Health check failed: {response}")
        tests_failed += 1

    # Test 4: Document ingestion
    print("\n4. Testing Document Ingestion...")
    ingest_request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "ingest_document",
            "arguments": {
                "content": "The RAG-Redis system is a high-performance retrieval-augmented generation system. It uses Rust for performance and Redis for distributed caching.",
                "metadata": {
                    "title": "RAG-Redis Overview",
                    "type": "technical_doc",
                    "timestamp": "2024-09-21"
                }
            }
        }
    }

    response = send_mcp_request(ingest_request)
    if response and "result" in response:
        print(f"   ✓ Document ingested successfully")
        if "content" in response["result"]:
            content = json.loads(response["result"]["content"][0]["text"])
            print(f"      Document ID: {content.get('document_id', 'unknown')}")
        tests_passed += 1
    else:
        print(f"   ✗ Document ingestion failed: {response}")
        tests_failed += 1

    # Test 5: Search functionality
    print("\n5. Testing Search...")
    search_request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {
                "query": "high-performance Rust Redis",
                "limit": 5
            }
        }
    }

    response = send_mcp_request(search_request)
    if response and "result" in response:
        print(f"   ✓ Search executed successfully")
        if "content" in response["result"]:
            content = json.loads(response["result"]["content"][0]["text"])
            results = content.get("results", [])
            print(f"      Found {len(results)} results")
        tests_passed += 1
    else:
        print(f"   ✗ Search failed: {response}")
        tests_failed += 1

    # Test 6: Memory store
    print("\n6. Testing Memory Store...")
    memory_request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "memory_store",
            "arguments": {
                "content": "User prefers technical documentation with code examples",
                "memory_type": "long_term",
                "importance": 0.9
            }
        }
    }

    response = send_mcp_request(memory_request)
    if response and "result" in response:
        print(f"   ✓ Memory stored successfully")
        if "content" in response["result"]:
            content = json.loads(response["result"]["content"][0]["text"])
            print(f"      Memory ID: {content.get('memory_id', 'unknown')}")
        tests_passed += 1
    else:
        print(f"   ✗ Memory store failed: {response}")
        tests_failed += 1

    # Test 7: Memory recall
    print("\n7. Testing Memory Recall...")
    recall_request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "memory_recall",
            "arguments": {
                "query": "technical documentation preferences",
                "memory_type": "long_term"
            }
        }
    }

    response = send_mcp_request(recall_request)
    if response and "result" in response:
        print(f"   ✓ Memory recall successful")
        if "content" in response["result"]:
            content = json.loads(response["result"]["content"][0]["text"])
            memories = content.get("memories", [])
            print(f"      Recalled {len(memories)} memories")
        tests_passed += 1
    else:
        print(f"   ✗ Memory recall failed: {response}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  ✓ Passed: {tests_passed}")
    print(f"  ✗ Failed: {tests_failed}")
    print("=" * 60)

    return tests_failed == 0

def test_redis_data():
    """Check actual data in Redis"""
    print("\nChecking Redis Data Persistence...")

    import subprocess

    # Check documents
    result = subprocess.run(['redis-cli', '-p', '6380', 'keys', 'doc:*'],
                          capture_output=True, text=True)
    doc_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    print(f"  Documents in Redis: {doc_count}")

    # Check embeddings
    result = subprocess.run(['redis-cli', '-p', '6380', 'keys', 'embedding:*'],
                          capture_output=True, text=True)
    embed_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    print(f"  Embeddings in Redis: {embed_count}")

    # Check memories
    result = subprocess.run(['redis-cli', '-p', '6380', 'keys', 'mem:*'],
                          capture_output=True, text=True)
    mem_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    print(f"  Memories in Redis: {mem_count}")

if __name__ == "__main__":
    # Run the tests
    success = test_mcp_server()

    # Check Redis data
    test_redis_data()

    # Exit with appropriate code
    sys.exit(0 if success else 1)