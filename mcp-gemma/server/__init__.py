"""
MCP Gemma Server - Model Context Protocol server for gemma.cpp integration.

This module provides MCP server capabilities for gemma.cpp, enabling it to be used
as an MCP server for other projects with support for:
- Text generation/completion
- Token streaming
- Model loading/switching
- Memory management (via RAG-Redis)
- Performance metrics

Supports multiple transport layers:
- stdio (for direct CLI integration)
- HTTP (for REST API access)
- WebSocket (for real-time streaming)
"""

__version__ = "1.0.0"
__author__ = "LLM Development Team"

from .base import GemmaServer
from .handlers import GenerationHandler, MemoryHandler, MetricsHandler, ModelHandler
from .transports import HTTPTransport, StdioTransport, WebSocketTransport

__all__ = [
    "GemmaServer",
    "StdioTransport",
    "HTTPTransport",
    "WebSocketTransport",
    "GenerationHandler",
    "ModelHandler",
    "MemoryHandler",
    "MetricsHandler",
]
