"""FastAPI server for the Gemma chatbot with production-ready features.

This module provides a comprehensive HTTP API server with:
- RESTful chat completion endpoints
- WebSocket support for real-time chat
- Server-Sent Events (SSE) for streaming
- OpenAI-compatible API
- Rate limiting and authentication
- Request/response logging
- Prometheus metrics
- CORS configuration
- Graceful shutdown
- Connection pooling
"""

import asyncio
import contextlib
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any
from typing import Dict
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import generate_latest

from ..application.inference.service import InferenceService
from ..shared.config.settings import Settings
from ..shared.config.settings import get_settings
from ..shared.exceptions import ConfigurationException
from ..shared.exceptions import InferenceException
from ..shared.exceptions import ValidationException
from ..shared.logging import get_logger
from .api.chat import chat_router
from .api.health import health_router
from .api.models import models_router
from .middleware import add_all_middleware
from .middleware import add_auth_middleware
from .middleware import add_cors_middleware
from .middleware import add_input_validation_middleware
from .middleware import add_logging_middleware
from .middleware import add_metrics_middleware
from .middleware import add_rate_limiting_middleware
from .state import app_state
from .state import get_inference_service
from .state import get_shutdown_event
from .state import get_startup_time
from .state import get_uptime
from .state import set_inference_service
from .state import set_startup_time
from .websocket import WebSocketHandler

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    settings = get_settings()

    # Startup
    logger.info("Starting Gemma Chatbot Server...")
    set_startup_time()

    try:
        # Initialize inference service
        logger.info("Initializing inference service...")
        inference_service = InferenceService(settings=settings)
        await inference_service.initialize()
        set_inference_service(inference_service)

        # Warm up the service
        logger.info("Warming up inference service...")
        await inference_service._warm_up_model()

        startup_duration = get_uptime()
        logger.info(f"Server startup completed in {startup_duration:.2f} seconds")

        yield

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise ConfigurationException(f"Server startup failed: {e}")

    finally:
        # Shutdown
        logger.info("Shutting down Gemma Chatbot Server...")
        get_shutdown_event().set()

        inference_service = get_inference_service()
        if inference_service:
            try:
                await inference_service.cleanup()
            except Exception as e:
                logger.error(f"Error during inference service cleanup: {e}")

        logger.info("Server shutdown completed")


def create_app(settings: Settings = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Configuration settings (uses default if None)

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    # Create FastAPI app with custom configuration
    app = FastAPI(
        title="Gemma Chatbot API",
        description="Production-ready API server for Gemma LLM chatbot with streaming support",
        version=settings.version,
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan,
    )

    # Configure logging
    configure_logging(settings)

    # Add all middleware in the correct order
    add_all_middleware(app, settings)

    # Add routers
    app.include_router(chat_router, prefix="/v1", tags=["chat"])
    app.include_router(models_router, prefix="/v1", tags=["models"])
    app.include_router(health_router, prefix="/health", tags=["health"])

    # Add WebSocket endpoints
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, session_id: str | None = None):
        """WebSocket endpoint for real-time chat."""
        logger = get_logger("server.websocket")

        inference_service = get_inference_service()
        if not inference_service:
            await websocket.close(code=1011, reason="Service unavailable")
            return

        handler = WebSocketHandler(inference_service)
        await handler.handle_connection(websocket, session_id)

    @app.websocket("/ws/{session_id}")
    async def websocket_session_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for joining a specific session."""
        logger = get_logger("server.websocket")

        inference_service = get_inference_service()
        if not inference_service:
            await websocket.close(code=1011, reason="Service unavailable")
            return

        handler = WebSocketHandler(inference_service)
        await handler.handle_connection(websocket, session_id)

    # Add root endpoints
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "status": "running",
            "docs": "/docs" if not settings.is_production() else "disabled",
            "endpoints": {
                "chat": "/v1/chat/completions",
                "completions": "/v1/completions",
                "models": "/v1/models",
                "websocket": "/ws",
                "health": "/health",
                "metrics": "/metrics",
            },
        }

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Global exception handlers
    @app.exception_handler(ValidationException)
    async def validation_exception_handler(request: Request, exc: ValidationException):
        """Handle validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "message": str(exc),
                    "details": getattr(exc, "details", None),
                }
            },
        )

    @app.exception_handler(InferenceException)
    async def inference_exception_handler(request: Request, exc: InferenceException):
        """Handle inference errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "inference_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ConfigurationException)
    async def configuration_exception_handler(request: Request, exc: ConfigurationException):
        """Handle configuration errors."""
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "configuration_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger = get_logger("server.main")
        logger.error(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "message": "An internal server error occurred",
                }
            },
        )

    return app


def configure_logging(settings: Settings) -> None:
    """Configure application logging.

    Args:
        settings: Configuration settings
    """
    # Configure root logger
    logging.basicConfig(
        level=settings.log_level,
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(settings.log_file)] if settings.log_file else []),
        ],
    )

    # Set specific logger levels
    get_logger("uvicorn.access").setLevel(logging.INFO)
    get_logger("uvicorn.error").setLevel(logging.INFO)
    get_logger("fastapi").setLevel(logging.INFO)


def setup_signal_handlers(app: FastAPI) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        app: FastAPI application instance
    """
    logger = get_logger("server.main")

    def signal_handler(signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        app_state["shutdown_event"].set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """Run the server with custom configuration.

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    settings = get_settings()
    app = create_app(settings)

    # Setup signal handlers
    setup_signal_handlers(app)

    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Single worker in reload mode
        log_level=settings.log_level.value.lower(),
        access_log=settings.server.access_log,
        use_colors=not settings.is_production(),
        server_header=False,
        date_header=False,
    )

    server = uvicorn.Server(config)

    # Run server
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger = get_logger("server.main")
        logger.info("Server interrupted by user")
    finally:
        # Ensure cleanup
        app_state["shutdown_event"].set()


def main() -> None:
    """Main entry point for the server application."""
    settings = get_settings()

    # Create and configure app
    app = create_app(settings)

    # Run server
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_level=settings.log_level.value.lower(),
        access_log=settings.server.access_log,
    )


# Export the app for ASGI servers
settings = get_settings()
app = create_app(settings)

if __name__ == "__main__":
    main()
