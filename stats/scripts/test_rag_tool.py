#!/usr/bin/env python3
"""
Test RAG tool functionality
"""

import asyncio
import json
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO)


async def test_health_check():
    """Test the health_check tool"""
    server_path = "C:/Users/david/.cargo/shared-target/release/rag-redis-mcp-server.exe"

    # Start server
    process = await asyncio.create_subprocess_exec(
        server_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "RUST_LOG": "info"},
    )

    try:
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
                "clientInfo": {"name": "Test-Client", "version": "1.0.0"},
            },
        }

        process.stdin.write((json.dumps(init_request) + "\n").encode())
        await process.stdin.drain()

        # Read init response
        init_response = await process.stdout.readline()
        logger.info(f"Init: {init_response.decode().strip()}")

        # Send initialized notification
        initialized = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        process.stdin.write((json.dumps(initialized) + "\n").encode())
        await process.stdin.drain()

        # Test health check tool
        health_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "health_check", "arguments": {"verbose": True}},
        }

        process.stdin.write((json.dumps(health_request) + "\n").encode())
        await process.stdin.drain()

        # Read response
        health_response = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
        response_data = json.loads(health_response.decode().strip())

        logger.info("Health check response:")
        logger.info(json.dumps(response_data, indent=2))

        if "result" in response_data:
            logger.info("✅ Health check tool works!")
            return True
        else:
            logger.error("❌ Health check tool failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


if __name__ == "__main__":
    success = asyncio.run(test_health_check())
    sys.exit(0 if success else 1)
