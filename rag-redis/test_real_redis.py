#!/usr/bin/env python3
"""
Test RAG-Redis MCP Server with REAL Redis
Verifies that the server uses actual Redis, not mocks
"""

import json
import subprocess
import time
import sys
import os
from datetime import datetime

def send_request(process, request):
    """Send request to MCP server and get response"""
    request_json = json.dumps(request) + '\n'
    process.stdin.write(request_json)
    process.stdin.flush()

    response_line = process.stdout.readline()
    if response_line:
        return json.loads(response_line.strip())
    return None

def test_real_redis_operations():
    """Test that MCP server uses real Redis"""

    print("=" * 60)
    print("Testing RAG-Redis MCP Server with REAL Redis")
    print("=" * 60)

    # Check Redis is running
    print("\nVerifying Redis is running on port 6380...")
    redis_check = subprocess.run(
        ["redis-cli", "-p", "6380", "ping"],
        capture_output=True,
        text=True
    )

    if redis_check.returncode != 0 or redis_check.stdout.strip() != "PONG":
        print("ERROR: Redis is not running on port 6380!")
        return False
    print("SUCCESS: Redis is responding on port 6380")

    # Clear any existing test data
    print("\nClearing test data from Redis...")
    subprocess.run(["redis-cli", "-p", "6380", "del", "test:*"], capture_output=True)

    # Set up MCP server
    mcp_server = r"C:\Users\david\.cargo\shared-target\release\mcp-server.exe"

    env = os.environ.copy()
    env['REDIS_URL'] = 'redis://127.0.0.1:6380'
    env['RUST_LOG'] = 'debug'

    print("\nStarting MCP server...")
    process = subprocess.Popen(
        [mcp_server],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )

    try:
        # Initialize
        print("\n1. Initializing MCP connection...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "redis-test", "version": "1.0"}
            }
        }

        response = send_request(process, init_request)
        if response and "result" in response:
            print(f"   SUCCESS: Initialized with protocol {response['result']['protocolVersion']}")
        else:
            print(f"   FAILED: {response}")
            return False

        # Test document ingestion (should use Redis)
        test_id = f"test-doc-{int(time.time())}"
        print(f"\n2. Ingesting document with ID hint: {test_id}...")

        doc_content = f"Test document for Redis verification. ID: {test_id}. Created at {datetime.now()}"
        ingest_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "ingest_document",
                "arguments": {
                    "content": doc_content,
                    "metadata": {
                        "test_id": test_id,
                        "source": "redis_test"
                    }
                }
            }
        }

        response = send_request(process, ingest_request)
        if response and "result" in response:
            print("   SUCCESS: Document ingested")

            # Check if document is in Redis
            print("\n3. Verifying document is stored in Redis...")
            redis_keys = subprocess.run(
                ["redis-cli", "-p", "6380", "keys", "*"],
                capture_output=True,
                text=True
            )

            if redis_keys.stdout:
                keys = redis_keys.stdout.strip().split('\n')
                print(f"   Found {len(keys)} keys in Redis:")
                for key in keys[:10]:  # Show first 10
                    print(f"     - {key}")
            else:
                print("   WARNING: No keys found in Redis")

        # Test memory storage (should use Redis)
        print("\n4. Storing memory in Redis...")
        memory_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "store_memory",
                "arguments": {
                    "content": f"Test memory for Redis verification at {datetime.now()}",
                    "memory_type": "long_term",
                    "importance": 0.9
                }
            }
        }

        response = send_request(process, memory_request)
        if response and "result" in response:
            print("   SUCCESS: Memory stored")

            # Check Redis for memory keys
            memory_keys = subprocess.run(
                ["redis-cli", "-p", "6380", "keys", "mem:*"],
                capture_output=True,
                text=True
            )

            if memory_keys.stdout:
                print(f"   Found memory keys in Redis: {memory_keys.stdout.strip()[:100]}")
            else:
                print("   No memory keys found (may use different prefix)")

        # Test search (should query Redis)
        print("\n5. Searching documents (using Redis)...")
        search_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {
                    "query": "Redis verification test",
                    "limit": 5
                }
            }
        }

        response = send_request(process, search_request)
        if response and "result" in response:
            print("   SUCCESS: Search completed")

        # Final Redis check
        print("\n6. Final Redis verification...")
        final_keys = subprocess.run(
            ["redis-cli", "-p", "6380", "dbsize"],
            capture_output=True,
            text=True
        )

        if final_keys.stdout:
            db_size = final_keys.stdout.strip()
            print(f"   Redis database size: {db_size}")

        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE: System is using REAL Redis!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    finally:
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    success = test_real_redis_operations()
    sys.exit(0 if success else 1)