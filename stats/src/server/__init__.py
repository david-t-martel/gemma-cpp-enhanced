"""Gemma Chatbot HTTP Server.

This package provides a production-ready FastAPI HTTP server for the Gemma chatbot
with features including:

- RESTful chat completion endpoints (OpenAI-compatible)
- WebSocket support for real-time chat
- Server-Sent Events (SSE) for streaming responses
- Rate limiting and authentication middleware
- Prometheus metrics collection
- Comprehensive health checks
- Graceful shutdown handling
- Connection pooling and resource management

Usage:
    # Start the server
    python -m src.server.main

    # Or use uvicorn directly
    uvicorn src.server.main:app --host 0.0.0.0 --port 8000

    # Or use the project entry point
    gemma-server
"""

from .main import app
from .main import create_app
from .main import main
from .main import run_server

__all__ = ["app", "create_app", "main", "run_server"]
