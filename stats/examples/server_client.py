#!/usr/bin/env python3
"""Example client for the Gemma Chatbot HTTP server.

This script demonstrates how to interact with the server using different methods:
- REST API calls for chat completions
- WebSocket connections for real-time chat
- Server-Sent Events for streaming responses

Run the server first with:
    python -m src.server.main

Then run this client:
    python examples/server_client.py
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from sse_client import SSEClient
import websockets

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"


async def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f}s")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    return True


async def test_models_endpoint():
    """Test the models listing endpoint."""
    print("\n📋 Testing models endpoint...")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/models")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['data'])} models:")
            for model in data["data"][:3]:  # Show first 3
                print(f"   - {model['id']} (owned by {model['owned_by']})")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False

    return True


async def test_chat_completion():
    """Test the chat completions endpoint."""
    print("\n💬 Testing chat completion...")

    messages = [{"role": "user", "content": "Hello! Can you tell me a short joke?"}]

    request_data = {
        "model": "gemma-2b-it",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BASE_URL}/v1/chat/completions", json=request_data)

        if response.status_code == 200:
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            usage = data["usage"]

            print("✅ Chat completion successful:")
            print(f"   Response: {message[:100]}...")
            print(f"   Tokens used: {usage['total_tokens']}")
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            if response.headers.get("content-type", "").startswith("application/json"):
                print(f"   Error: {response.json()}")
            return False

    return True


async def test_streaming_completion():
    """Test streaming chat completion."""
    print("\n🌊 Testing streaming completion...")

    messages = [{"role": "user", "content": "Tell me a very short story about a robot."}]

    request_data = {
        "model": "gemma-2b-it",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.8,
        "stream": True,
    }

    try:
        async with (
            httpx.AsyncClient(timeout=30.0) as client,
            client.stream("POST", f"{BASE_URL}/v1/chat/completions", json=request_data) as response,
        ):
            if response.status_code != 200:
                print(f"❌ Streaming failed: {response.status_code}")
                return False

            print("✅ Streaming response:")
            full_response = ""

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_response += content
                    except json.JSONDecodeError:
                        continue

            print(f"\n   Full response length: {len(full_response)} characters")

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

    return True


async def test_websocket_chat():
    """Test WebSocket chat functionality."""
    print("\n🔌 Testing WebSocket chat...")

    try:
        uri = f"{WS_URL}/ws"
        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)

            if welcome_data["type"] == "connection":
                session_id = welcome_data["session_id"]
                print(f"✅ Connected to session: {session_id}")

                # Send a chat message
                chat_message = {
                    "type": "chat",
                    "session_id": session_id,
                    "data": {
                        "message": "Hello via WebSocket! Tell me about yourself.",
                        "stream": True,
                        "user_id": "test_user",
                    },
                }

                await websocket.send(json.dumps(chat_message))
                print("📤 Sent chat message")

                # Listen for responses
                response_parts = []
                timeout_count = 0
                max_timeout = 10

                while timeout_count < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        response_data = json.loads(response)

                        if response_data["type"] == "assistant_message_chunk":
                            content = response_data["data"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                response_parts.append(content)
                        elif response_data["type"] == "assistant_message_complete":
                            print("\n✅ WebSocket chat complete!")
                            print(f"   Full response: {''.join(response_parts)[:100]}...")
                            break
                        elif response_data["type"] == "error":
                            print(f"\n❌ WebSocket error: {response_data['data']['error']}")
                            return False

                    except TimeoutError:
                        timeout_count += 1
                        if timeout_count >= max_timeout:
                            print("\n⏰ WebSocket response timeout")
                            return False

            else:
                print(f"❌ Unexpected welcome message: {welcome_data}")
                return False

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False

    return True


async def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("\n📊 Testing metrics endpoint...")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health/metrics")

        if response.status_code == 200:
            data = response.json()
            print("✅ Metrics collected:")
            print(f"   Total requests: {data['requests_total']}")
            print(f"   Active connections: {data['active_connections']}")
            print(f"   Cache hit rate: {data['cache_hit_rate']:.1f}%")
            print(f"   Average response time: {data['average_response_time']:.3f}s")
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False

    return True


async def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Starting Gemma Chatbot Server Tests")
    print("=" * 50)

    tests = [
        ("Health Check", test_health_check),
        ("Models Endpoint", test_models_endpoint),
        ("Chat Completion", test_chat_completion),
        ("Streaming Completion", test_streaming_completion),
        ("WebSocket Chat", test_websocket_chat),
        ("Metrics Endpoint", test_metrics_endpoint),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1

    print("=" * 50)
    print(f"Total: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Server is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
