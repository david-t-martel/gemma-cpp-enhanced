#!/usr/bin/env python3
"""
Complete test of FastAPI server functionality with proper API key.
"""

import asyncio
import json
import os
import sys

import aiohttp
import websockets

# Set API key for testing
API_KEY = "test-api-key-123"
os.environ["GEMMA_API_KEYS"] = API_KEY

BASE_URL = "http://localhost:8000"


async def test_chat_completion():
    """Test chat completion with proper authentication."""
    print("Testing /v1/chat/completions endpoint...")

    payload = {
        "model": "gemma-2b-it",
        "messages": [{"role": "user", "content": "Hello, this is a test message"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers
            ) as response:
                status = response.status
                data = await response.text()
                print(f"   Status: {status}")
                print(f"   Response: {data[:300]}...")

                if status == 200:
                    try:
                        json_data = json.loads(data)
                        if json_data.get("choices"):
                            content = json_data["choices"][0].get("message", {}).get("content", "")
                            print(f"   Generated content: {content}")
                    except:
                        pass

                return status == 200

        except Exception as e:
            print(f"   Error: {e}")
            return False


async def test_streaming_completion():
    """Test streaming chat completion."""
    print("\nTesting streaming chat completion...")

    payload = {
        "model": "gemma-2b-it",
        "messages": [{"role": "user", "content": "Tell me a short story"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers
            ) as response:
                status = response.status
                print(f"   Status: {status}")

                if status == 200:
                    content = []
                    async for line in response.content:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_part = line_str[6:]  # Remove 'data: '
                            if data_part == "[DONE]":
                                print("   Stream completed")
                                break
                            try:
                                chunk = json.loads(data_part)
                                if chunk.get("choices"):
                                    delta_content = (
                                        chunk["choices"][0].get("delta", {}).get("content", "")
                                    )
                                    if delta_content:
                                        content.append(delta_content)
                                        print(f"   Chunk: {delta_content}")
                            except:
                                pass

                    print(f"   Full streamed content: {''.join(content)}")
                    return True

                return False

        except Exception as e:
            print(f"   Error: {e}")
            return False


async def test_websocket_detailed():
    """Test WebSocket functionality in detail."""
    print("\nTesting WebSocket functionality...")

    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print("   WebSocket connected successfully")

            # Test 1: Ping
            ping_msg = {"type": "ping", "data": {}}
            await websocket.send(json.dumps(ping_msg))
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"   Ping response: {json.loads(response)['type']}")

            # Test 2: Chat message
            chat_msg = {
                "type": "chat",
                "data": {"message": "Hello WebSocket, tell me about AI", "stream": True},
            }
            await websocket.send(json.dumps(chat_msg))

            # Collect streaming responses
            responses = []
            start_time = asyncio.get_event_loop().time()
            while len(responses) < 10 and (asyncio.get_event_loop().time() - start_time) < 10:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    resp_data = json.loads(response)
                    responses.append(resp_data)
                    print(f"   Response {len(responses)}: {resp_data['type']}")

                    if resp_data.get("type") == "assistant_message_complete":
                        print("   Chat response completed")
                        break
                except TimeoutError:
                    break

            print(f"   Received {len(responses)} WebSocket responses")
            return len(responses) > 0

    except Exception as e:
        print(f"   WebSocket error: {e}")
        return False


async def test_all_health_endpoints():
    """Test all health and monitoring endpoints."""
    print("\nTesting health and monitoring endpoints...")

    endpoints = [
        "/",
        "/health/",
        "/health/live",
        "/health/ready",
        "/health/metrics",
        "/health/status",
    ]

    results = {}
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.get(f"{BASE_URL}{endpoint}") as response:
                    status = response.status
                    results[endpoint] = status == 200
                    print(f"   {endpoint}: {status} ({'✅' if status == 200 else '❌'})")
            except Exception as e:
                results[endpoint] = False
                print(f"   {endpoint}: ERROR - {e}")

    return all(results.values())


async def main():
    """Run comprehensive server tests."""
    print("=" * 60)
    print("COMPREHENSIVE FastAPI SERVER TEST REPORT")
    print("=" * 60)

    results = {}

    # Test health endpoints
    results["health_endpoints"] = await test_all_health_endpoints()

    # Test REST API
    results["chat_completion"] = await test_chat_completion()
    results["streaming_completion"] = await test_streaming_completion()

    # Test WebSocket
    results["websocket"] = await test_websocket_detailed()

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")

    print(
        f"\nOverall: {passed_tests}/{total_tests} tests passed ({(passed_tests / total_tests) * 100:.1f}%)"
    )

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Server is working perfectly.")
    else:
        print("⚠️  Some tests failed. Check the details above.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
