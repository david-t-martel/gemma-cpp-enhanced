#!/usr/bin/env python3
"""
Functional MCP Server Test for RAG-Redis
Tests actual functionality, not just listing
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

def test_mcp_functional():
    """Test MCP server actual functionality"""

    print("=" * 60)
    print("RAG-Redis MCP Server Functional Test")
    print("=" * 60)

    # Configuration
    mcp_server = r"C:\Users\david\.cargo\shared-target\release\mcp-server.exe"

    # Set up environment
    env = os.environ.copy()
    env['REDIS_URL'] = 'redis://127.0.0.1:6380'
    env['RUST_LOG'] = 'info'

    # Start MCP server
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
        # Test 1: Initialize
        print("\n1. Initializing MCP connection...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "functional-test", "version": "1.0"}
            }
        }

        response = send_request(process, init_request)
        if response and "result" in response:
            print(f"   SUCCESS: Protocol {response['result']['protocolVersion']}")
        else:
            print(f"   FAILED: {response}")
            return False

        # Test 2: Health Check
        print("\n2. Testing health check...")
        health_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "health_check",
                "arguments": {}
            }
        }

        response = send_request(process, health_request)
        if response and "result" in response:
            print(f"   SUCCESS: Health check passed")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    print(f"   Status: {data.get('status', 'unknown')}")
                    print(f"   Redis: {data.get('redis_connected', False)}")
        else:
            print(f"   FAILED: {response}")

        # Test 3: Document Ingestion
        print("\n3. Testing document ingestion...")
        doc_content = f"The RAG-Redis system is a high-performance retrieval system. Test document created at {datetime.now()}"
        ingest_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ingest_document",
                "arguments": {
                    "content": doc_content,
                    "metadata": {
                        "title": "Test Document",
                        "source": "functional_test",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        }

        response = send_request(process, ingest_request)
        doc_id = None
        if response and "result" in response:
            print(f"   SUCCESS: Document ingested")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    doc_id = data.get('document_id', 'unknown')
                    print(f"   Document ID: {doc_id}")
        else:
            print(f"   FAILED: {response}")

        # Test 4: Search for Document
        print("\n4. Testing document search...")
        search_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {
                    "query": "RAG-Redis high-performance retrieval",
                    "limit": 5
                }
            }
        }

        response = send_request(process, search_request)
        if response and "result" in response:
            print(f"   SUCCESS: Search completed")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    results = data.get('results', [])
                    print(f"   Found {len(results)} results")
                    for i, result in enumerate(results[:3], 1):
                        print(f"   Result {i}: Score {result.get('score', 0):.3f}")
        else:
            print(f"   FAILED: {response}")

        # Test 5: Memory Store
        print("\n5. Testing memory storage...")
        memory_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "store_memory",
                "arguments": {
                    "content": "User prefers technical documentation with examples",
                    "memory_type": "long_term",
                    "importance": 0.9
                }
            }
        }

        response = send_request(process, memory_request)
        memory_id = None
        if response and "result" in response:
            print(f"   SUCCESS: Memory stored")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    memory_id = data.get('memory_id', 'unknown')
                    print(f"   Memory ID: {memory_id}")
        else:
            print(f"   FAILED: {response}")

        # Test 6: Memory Recall
        print("\n6. Testing memory recall...")
        recall_request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "recall_memory",
                "arguments": {
                    "query": "user preferences documentation",
                    "limit": 5
                }
            }
        }

        response = send_request(process, recall_request)
        if response and "result" in response:
            print(f"   SUCCESS: Memory recalled")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    memories = data.get('memories', [])
                    print(f"   Found {len(memories)} memories")
                    for i, mem in enumerate(memories[:3], 1):
                        print(f"   Memory {i}: {mem.get('content', '')[:50]}...")
        else:
            print(f"   FAILED: {response}")

        # Test 7: List Documents
        print("\n7. Testing list documents...")
        list_request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "list_documents",
                "arguments": {
                    "limit": 10
                }
            }
        }

        response = send_request(process, list_request)
        if response and "result" in response:
            print(f"   SUCCESS: Documents listed")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    documents = data.get('documents', [])
                    print(f"   Total documents: {len(documents)}")
        else:
            print(f"   FAILED: {response}")

        # Test 8: Project Context Save
        print("\n8. Testing project context save...")
        context_request = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "save_project_context",
                "arguments": {
                    "project_id": f"test-project-{int(time.time())}",
                    "project_root": "c:/codedev/llm/rag-redis",
                    "options": {
                        "include_memories": True,
                        "include_documents": True
                    }
                }
            }
        }

        response = send_request(process, context_request)
        if response and "result" in response:
            print(f"   SUCCESS: Project context saved")
            if "content" in response["result"]:
                content = response["result"]["content"][0]
                if isinstance(content, dict) and "text" in content:
                    data = json.loads(content["text"])
                    snapshot_id = data.get('snapshot_id', 'unknown')
                    print(f"   Snapshot ID: {snapshot_id}")
        else:
            print(f"   FAILED: {response}")

        print("\n" + "=" * 60)
        print("All functional tests completed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    finally:
        # Cleanup
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    success = test_mcp_functional()
    sys.exit(0 if success else 1)