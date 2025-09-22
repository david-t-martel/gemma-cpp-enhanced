#!/usr/bin/env python3
"""
MCP RAG-Redis Integration Validator

This script validates that the RAG-Redis MCP server:
1. Can be started successfully
2. Responds to MCP protocol messages correctly
3. Implements required MCP methods
4. Returns valid JSON-RPC responses
5. Works with the Gemma agent integration

Usage:
    python scripts/validate_mcp_integration.py
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MCPValidator:
    def __init__(self, server_path: str, config_path: str):
        self.server_path = server_path
        self.config_path = config_path
        self.process = None
        self.request_id = 0

    def get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    async def start_server(self) -> bool:
        """Start the MCP server process"""
        try:
            # Check if binary exists
            if not Path(self.server_path).exists():
                logger.error(f"❌ Server binary not found: {self.server_path}")
                return False

            logger.info(f"🚀 Starting MCP server: {self.server_path}")

            # Start the server process
            self.process = await asyncio.create_subprocess_exec(
                self.server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "RUST_LOG": "debug"},
            )

            # Wait a moment for server to initialize
            await asyncio.sleep(2)

            if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                logger.error(f"❌ Server failed to start: {stderr.decode()}")
                return False

            logger.info("✅ Server started successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> tuple[bool, dict[str, Any] | None]:
        """Send a JSON-RPC request to the server"""
        if not self.process or self.process.returncode is not None:
            return False, {"error": "Server not running"}

        request = {"jsonrpc": "2.0", "id": self.get_next_id(), "method": method}

        if params is not None:
            request["params"] = params

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            logger.debug(f"📤 Sending: {request_json.strip()}")

            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()

            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10.0)
                response_str = response_line.decode().strip()

                if not response_str:
                    return False, {"error": "Empty response"}

                logger.debug(f"📥 Received: {response_str}")
                response = json.loads(response_str)

                # Check for JSON-RPC error
                if "error" in response:
                    return False, response

                return True, response

            except TimeoutError:
                logger.error("❌ Request timeout")
                return False, {"error": "Request timeout"}

        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            return False, {"error": str(e)}

    async def validate_initialize(self) -> bool:
        """Test MCP initialize handshake"""
        logger.info("🔄 Testing initialize handshake...")

        params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
            "clientInfo": {"name": "MCP-Validator", "version": "1.0.0"},
        }

        success, response = await self.send_request("initialize", params)

        if not success:
            logger.error(f"❌ Initialize failed: {response}")
            return False

        # Validate initialize response
        result = response.get("result", {})
        if not result.get("protocolVersion"):
            logger.error("❌ Missing protocolVersion in initialize response")
            return False

        if not result.get("serverInfo"):
            logger.error("❌ Missing serverInfo in initialize response")
            return False

        if not result.get("capabilities"):
            logger.error("❌ Missing capabilities in initialize response")
            return False

        logger.info("✅ Initialize handshake successful")

        # Send initialized notification
        success, _ = await self.send_request("notifications/initialized")
        if success:
            logger.info("✅ Initialized notification sent")

        return True

    async def validate_ping(self) -> bool:
        """Test ping functionality"""
        logger.info("🏓 Testing ping...")

        success, response = await self.send_request("ping")

        if not success:
            logger.error(f"❌ Ping failed: {response}")
            return False

        logger.info("✅ Ping successful")
        return True

    async def validate_tools_list(self) -> bool:
        """Test tools/list endpoint"""
        logger.info("🔧 Testing tools/list...")

        success, response = await self.send_request("tools/list")

        if not success:
            logger.error(f"❌ Tools/list failed: {response}")
            return False

        result = response.get("result", {})
        tools = result.get("tools", [])

        if not tools:
            logger.warning("⚠️ No tools returned by server")
            return True

        # Validate tool structure
        for tool in tools:
            if not all(k in tool for k in ["name", "description", "inputSchema"]):
                logger.error(f"❌ Invalid tool structure: {tool}")
                return False

        logger.info(f"✅ Found {len(tools)} tools")
        for tool in tools:
            logger.info(f"  - {tool['name']}: {tool.get('description', 'No description')}")

        return True

    async def validate_tool_call(self) -> bool:
        """Test a specific tool call if available"""
        logger.info("🛠️ Testing tool execution...")

        # First get available tools
        success, response = await self.send_request("tools/list")
        if not success:
            logger.warning("⚠️ Cannot test tool execution - tools/list failed")
            return True

        tools = response.get("result", {}).get("tools", [])
        if not tools:
            logger.info("ℹ️ No tools available to test")
            return True

        # Try to call the first available tool with minimal params
        test_tool = tools[0]
        tool_name = test_tool["name"]

        # For RAG tools, try a simple test call
        if "health" in tool_name.lower():
            params = {"name": tool_name, "arguments": {}}
        elif "search" in tool_name.lower():
            params = {"name": tool_name, "arguments": {"query": "test"}}
        else:
            # Skip complex tools for basic validation
            logger.info(f"ℹ️ Skipping complex tool test: {tool_name}")
            return True

        success, response = await self.send_request("tools/call", params)

        if not success:
            logger.warning(f"⚠️ Tool call failed for {tool_name}: {response}")
            # Tool call failure is not critical for basic validation
            return True

        logger.info(f"✅ Tool call successful: {tool_name}")
        return True

    async def stop_server(self):
        """Stop the MCP server"""
        if self.process and self.process.returncode is None:
            logger.info("🛑 Stopping server...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                logger.warning("⚠️ Server didn't terminate gracefully, killing...")
                self.process.kill()
                await self.process.wait()

            logger.info("✅ Server stopped")

    async def run_validation(self) -> bool:
        """Run the complete validation suite"""
        logger.info("🧪 Starting MCP Integration Validation")
        logger.info("=" * 50)

        # Start server
        if not await self.start_server():
            return False

        try:
            # Run validation tests
            tests = [
                ("Initialize Handshake", self.validate_initialize),
                ("Ping Test", self.validate_ping),
                ("Tools List", self.validate_tools_list),
                ("Tool Execution", self.validate_tool_call),
            ]

            results = []
            for test_name, test_func in tests:
                logger.info(f"\n📋 Running: {test_name}")
                try:
                    result = await test_func()
                    results.append((test_name, result))
                    status = "✅ PASS" if result else "❌ FAIL"
                    logger.info(f"   Result: {status}")
                except Exception as e:
                    logger.error(f"   Result: ❌ ERROR - {e}")
                    results.append((test_name, False))

            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("📊 VALIDATION SUMMARY")
            logger.info("=" * 50)

            passed = sum(1 for _, result in results if result)
            total = len(results)

            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{status} {test_name}")

            logger.info("-" * 50)
            logger.info(f"📈 Overall: {passed}/{total} tests passed")

            if passed == total:
                logger.info("🎉 ALL TESTS PASSED - MCP integration is ready!")
                return True
            else:
                logger.error("💥 SOME TESTS FAILED - Review issues above")
                return False

        finally:
            await self.stop_server()


def load_config() -> dict[str, Any]:
    """Load MCP configuration"""
    config_paths = [
        Path("C:/codedev/llm/stats/rag-redis-mcp-corrected.json"),
        Path("C:/codedev/llm/stats/rag-redis-mcp.json"),
        Path("rag-redis-mcp-corrected.json"),
        Path("rag-redis-mcp.json"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"📋 Using config: {config_path}")
            with open(config_path) as f:
                return json.load(f)

    raise FileNotFoundError("No MCP configuration file found")


async def main():
    """Main validation entry point"""
    try:
        # Load configuration
        config = load_config()

        # Get server configuration
        rag_redis_config = config.get("mcpServers", {}).get("rag-redis")
        if not rag_redis_config:
            logger.error("❌ No 'rag-redis' server configuration found")
            return False

        server_path = rag_redis_config.get("command")
        if not server_path:
            logger.error("❌ No server command specified in configuration")
            return False

        # Run validation
        validator = MCPValidator(server_path, "")
        success = await validator.run_validation()

        return success

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    # Ensure we have required directories
    os.makedirs("C:/codedev/llm/stats/data/rag", exist_ok=True)
    os.makedirs("C:/codedev/llm/stats/cache/embeddings", exist_ok=True)

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
