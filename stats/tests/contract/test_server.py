#!/usr/bin/env python3
"""
Test script for FastAPI server with WebSocket support.

This script comprehensively tests the server in lightweight mode (no model required).
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import aiohttp
import pytest
import websockets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws"

# Test timeout
TIMEOUT = 30


class ServerTester:
    """Comprehensive server testing class."""

    def __init__(self):
        self.session: aiohttp.ClientSession = None
        self.results: dict[str, Any] = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_health_endpoint(self) -> bool:
        """Test the /health endpoint."""
        logger.info("Testing /health endpoint...")
        try:
            async with self.session.get(f"{BASE_URL}/health/") as response:
                data = await response.json()
                logger.info(f"Health response: {data}")
                success = response.status in [200, 503]  # Allow unhealthy state
                self.results["health"] = {
                    "success": success,
                    "status": response.status,
                    "data": data,
                }
                return success
        except Exception as e:
            logger.error(f"Health endpoint failed: {e}")
            self.results["health"] = {"success": False, "error": str(e)}
            return False

    async def test_ready_endpoint(self) -> bool:
        """Test the /health/ready endpoint."""
        logger.info("Testing /health/ready endpoint...")
        try:
            async with self.session.get(f"{BASE_URL}/health/ready") as response:
                data = await response.json()
                logger.info(f"Ready response: {data}")
                success = response.status in [200, 503]  # Allow not ready state
                self.results["ready"] = {
                    "success": success,
                    "status": response.status,
                    "data": data,
                }
                return success
        except Exception as e:
            logger.error(f"Ready endpoint failed: {e}")
            self.results["ready"] = {"success": False, "error": str(e)}
            return False

    async def test_live_endpoint(self) -> bool:
        """Test the /health/live endpoint."""
        logger.info("Testing /health/live endpoint...")
        try:
            async with self.session.get(f"{BASE_URL}/health/live") as response:
                data = await response.json()
                logger.info(f"Live response: {data}")
                success = response.status == 200
                self.results["live"] = {"success": success, "status": response.status, "data": data}
                return success
        except Exception as e:
            logger.error(f"Live endpoint failed: {e}")
            self.results["live"] = {"success": False, "error": str(e)}
            return False

    async def test_metrics_endpoint(self) -> bool:
        """Test the /health/metrics endpoint."""
        logger.info("Testing /health/metrics endpoint...")
        try:
            async with self.session.get(f"{BASE_URL}/health/metrics") as response:
                data = await response.json()
                logger.info(f"Metrics response keys: {list(data.keys())}")
                success = response.status == 200
                self.results["metrics"] = {
                    "success": success,
                    "status": response.status,
                    "data": data,
                }
                return success
        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            self.results["metrics"] = {"success": False, "error": str(e)}
            return False

    async def test_status_endpoint(self) -> bool:
        """Test the /health/status endpoint."""
        logger.info("Testing /health/status endpoint...")
        try:
            async with self.session.get(f"{BASE_URL}/health/status") as response:
                data = await response.json()
                logger.info(f"Status response keys: {list(data.keys())}")
                success = response.status == 200
                self.results["status"] = {
                    "success": success,
                    "status": response.status,
                    "data": data,
                }
                return success
        except Exception as e:
            logger.error(f"Status endpoint failed: {e}")
            self.results["status"] = {"success": False, "error": str(e)}
            return False

    async def test_root_endpoint(self) -> bool:
        """Test the root / endpoint."""
        logger.info("Testing root / endpoint...")
        try:
            async with self.session.get(f"{BASE_URL}/") as response:
                data = await response.json()
                logger.info(f"Root response: {data}")
                success = response.status == 200
                self.results["root"] = {"success": success, "status": response.status, "data": data}
                return success
        except Exception as e:
            logger.error(f"Root endpoint failed: {e}")
            self.results["root"] = {"success": False, "error": str(e)}
            return False

    async def test_generate_endpoint(self) -> bool:
        """Test the /v1/chat/completions endpoint."""
        logger.info("Testing /v1/chat/completions endpoint...")
        try:
            payload = {
                "model": "gemma-2b-it",
                "messages": [{"role": "user", "content": "Hello, test message"}],
                "max_tokens": 50,
                "temperature": 0.7,
                "stream": False,
            }

            async with self.session.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                data = await response.text()  # Get as text first
                logger.info(f"Chat completion status: {response.status}")
                logger.info(f"Chat completion response: {data[:200]}...")

                success = response.status in [200, 500, 503]  # Allow inference failure
                self.results["generate"] = {
                    "success": success,
                    "status": response.status,
                    "data": data[:500],  # Truncate for readability
                }
                return success
        except Exception as e:
            logger.error(f"Generate endpoint failed: {e}")
            self.results["generate"] = {"success": False, "error": str(e)}
            return False

    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection."""
        logger.info("Testing WebSocket connection...")
        try:
            async with websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=10, close_timeout=10
            ) as websocket:
                logger.info("WebSocket connected successfully")

                # Test ping message
                ping_msg = {"type": "ping", "data": {}}
                await websocket.send(json.dumps(ping_msg))
                response = await asyncio.wait_for(websocket.recv(), timeout=5)

                logger.info(f"WebSocket ping response: {response}")

                # Test chat message (will likely fail due to no model, but connection should work)
                chat_msg = {"type": "chat", "data": {"message": "Hello WebSocket", "stream": True}}
                await websocket.send(json.dumps(chat_msg))

                # Try to receive a response (may be error due to no model)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    logger.info(f"WebSocket chat response: {response}")
                except TimeoutError:
                    logger.info("WebSocket chat response timed out (expected without model)")

                success = True
                self.results["websocket"] = {"success": success, "connected": True}
                return success

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.results["websocket"] = {"success": False, "error": str(e)}
            return False

    async def test_websocket_session_endpoint(self) -> bool:
        """Test WebSocket session-specific endpoint."""
        logger.info("Testing WebSocket session endpoint...")
        try:
            session_id = "test-session-123"
            ws_session_url = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/{session_id}"

            async with websockets.connect(
                ws_session_url, ping_interval=20, ping_timeout=10, close_timeout=10
            ) as websocket:
                logger.info(f"WebSocket session connected: {session_id}")

                # Test ping
                ping_msg = {"type": "ping", "data": {}}
                await websocket.send(json.dumps(ping_msg))
                response = await asyncio.wait_for(websocket.recv(), timeout=5)

                logger.info(f"WebSocket session ping response: {response}")

                success = True
                self.results["websocket_session"] = {"success": success, "connected": True}
                return success

        except Exception as e:
            logger.error(f"WebSocket session connection failed: {e}")
            self.results["websocket_session"] = {"success": False, "error": str(e)}
            return False

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all tests and return results."""
        tests = [
            ("root", self.test_root_endpoint),
            ("health", self.test_health_endpoint),
            ("live", self.test_live_endpoint),
            ("ready", self.test_ready_endpoint),
            ("metrics", self.test_metrics_endpoint),
            ("status", self.test_status_endpoint),
            ("generate", self.test_generate_endpoint),
            ("websocket", self.test_websocket_connection),
            ("websocket_session", self.test_websocket_session_endpoint),
        ]

        logger.info("=" * 60)
        logger.info("Starting comprehensive server tests...")
        logger.info("=" * 60)

        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} test ---")
            try:
                success = await test_func()
                status = "✅ PASS" if success else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name}: ❌ ERROR - {e}")
                self.results[test_name] = {"success": False, "error": str(e)}

        return self.results


def print_summary(results: dict[str, Any]):
    """Print test results summary."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.get("success", False))
    failed_tests = total_tests - passed_tests

    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

    logger.info("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not result.get("success", False) and "error" in result:
            logger.info(f"    Error: {result['error']}")
        if "status" in result:
            logger.info(f"    HTTP Status: {result['status']}")


async def check_server_running() -> bool:
    """Check if server is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
    except:
        return False


async def main():
    """Main test function."""
    logger.info(f"Testing server at {BASE_URL}")

    # Check if server is running
    if not await check_server_running():
        logger.error(f"Server is not running at {BASE_URL}")
        logger.error("Please start the server first with:")
        logger.error("  uv run python -m src.server.main")
        return 1

    # Run tests
    async with ServerTester() as tester:
        results = await tester.run_all_tests()
        print_summary(results)

        # Determine exit code
        failed_tests = sum(1 for result in results.values() if not result.get("success", False))
        return min(failed_tests, 1)  # Return 1 if any tests failed, 0 if all passed


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
