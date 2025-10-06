"""
MCP Gemma Client - Python client for connecting to the MCP Gemma server.

This module provides client implementations for all supported transport protocols:
- stdio: Direct command-line integration
- HTTP: REST API client
- WebSocket: Real-time streaming client

Example usage:

    # Stdio client (for local usage)
    from client import GemmaStdioClient
    async with GemmaStdioClient(model_path="/path/to/model.sbs") as client:
        response = await client.generate_text("Hello, world!")
        print(response)

    # HTTP client (for remote usage)
    from client import GemmaHTTPClient
    async with GemmaHTTPClient("http://localhost:8080") as client:
        response = await client.generate_text("Hello, world!")
        print(response)

    # WebSocket client (for streaming)
    from client import GemmaWebSocketClient
    async with GemmaWebSocketClient("ws://localhost:8081") as client:
        async for chunk in client.generate_text_stream("Hello, world!"):
            print(chunk, end="")
"""

__version__ = "1.0.0"
__author__ = "LLM Development Team"

from .base_client import BaseGemmaClient, GemmaClientError
from .http_client import GemmaHTTPClient
from .stdio_client import GemmaStdioClient
from .websocket_client import GemmaWebSocketClient

__all__ = [
    "GemmaStdioClient",
    "GemmaHTTPClient",
    "GemmaWebSocketClient",
    "BaseGemmaClient",
    "GemmaClientError",
]
