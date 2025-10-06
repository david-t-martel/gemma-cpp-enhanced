#!/usr/bin/env python3
"""
Simple MCP Server Test for RAG-Redis
"""

import json
import subprocess
import time
import sys
import os

def test_mcp_server():
    """Test MCP server basic functionality"""

    print("=" * 60)
    print("RAG-Redis MCP Server Simple Test")
    print("=" * 60)

    # Configuration
    mcp_server = r"C:\Users\david\.cargo\shared-target\release\mcp-server.exe"

    # Check if server exists
    if not os.path.exists(mcp_server):
        print(f"ERROR: MCP server not found at {mcp_server}")
        return False
    else:
        print(f"Found MCP server: {mcp_server}")

    # Test 1: Initialize
    print("\nTest 1: Initialize MCP connection")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    try:
        env = os.environ.copy()
        env['REDIS_URL'] = 'redis://127.0.0.1:6380'

        process = subprocess.Popen(
            [mcp_server],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        stdout, stderr = process.communicate(
            input=json.dumps(request) + '\n',
            timeout=5
        )

        if stdout:
            response = json.loads(stdout.strip())
            if "result" in response:
                print(f"  SUCCESS: {response['result']}")
            elif "error" in response:
                print(f"  ERROR: {response['error']}")
            else:
                print(f"  Response: {response}")
        else:
            print(f"  ERROR: No response")
            if stderr:
                print(f"  Stderr: {stderr[:200]}")

    except subprocess.TimeoutExpired:
        print("  ERROR: Request timed out")
        process.kill()
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON response: {e}")
        print(f"  stdout: {stdout[:200] if stdout else 'None'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 2: List tools
    print("\nTest 2: List available tools")
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": None
    }

    try:
        process = subprocess.Popen(
            [mcp_server],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        # Send initialize first
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }

        process.stdin.write(json.dumps(init_request) + '\n')
        process.stdin.flush()

        # Read init response
        init_response = process.stdout.readline()

        # Send tools/list
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()

        # Read tools response
        tools_response = process.stdout.readline()

        if tools_response:
            response = json.loads(tools_response.strip())
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                print(f"  SUCCESS: Found {len(tools)} tools:")
                for tool in tools[:5]:  # Show first 5
                    print(f"    - {tool['name']}")
            else:
                print(f"  ERROR: {response}")

        process.terminate()

    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)