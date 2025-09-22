#!/usr/bin/env python3
"""
Quick test to verify authentication fix on the running server.
"""

import asyncio
import json

import aiohttp


async def test_chat_with_auth():
    """Test chat endpoint with authentication."""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "gemma-2b-it",
        "messages": [{"role": "user", "content": "Hello, test message"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        print("Testing /v1/chat/completions endpoint...")

        # Test without auth (should fail if auth is enabled)
        print("1. Testing without authentication:")
        async with session.post(url, json=payload) as response:
            status = response.status
            data = await response.text()
            print(f"   Status: {status}")
            print(f"   Response: {data[:200]}...")

        # Test with API key (if auth is enabled)
        print("\n2. Testing with API key:")
        headers = {"Authorization": "Bearer test-api-key"}
        async with session.post(url, json=payload, headers=headers) as response:
            status = response.status
            data = await response.text()
            print(f"   Status: {status}")
            print(f"   Response: {data[:200]}...")


async def test_websocket_chat():
    """Test WebSocket chat functionality."""
    import websockets

    print("\n3. Testing WebSocket chat:")
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            # Send chat message
            chat_msg = {"type": "chat", "data": {"message": "Hello WebSocket test", "stream": True}}
            await websocket.send(json.dumps(chat_msg))

            # Receive responses
            for i in range(3):  # Get a few responses
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    print(f"   WebSocket response {i + 1}: {response[:100]}...")
                except TimeoutError:
                    print(f"   Response {i + 1}: Timeout")
                    break

    except Exception as e:
        print(f"   WebSocket error: {e}")


async def main():
    print("Testing FastAPI server authentication and chat functionality...")
    print("=" * 60)

    await test_chat_with_auth()
    await test_websocket_chat()


if __name__ == "__main__":
    asyncio.run(main())
