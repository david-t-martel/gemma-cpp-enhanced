#!/usr/bin/env python3
"""
Start the FastAPI server in test mode without loading heavy models.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI
import uvicorn

# Add src to path for imports
sys.path.insert(0, "src")

from src.server.main import create_app
from src.shared.config.settings import Settings, get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestInferenceService:
    """Mock inference service for testing without loading models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.TestInferenceService")
        self._initialized = False

    async def initialize(self):
        """Mock initialization."""
        self.logger.info("Initializing test inference service (no model loading)")
        await asyncio.sleep(0.1)  # Simulate brief initialization
        self._initialized = True

    async def _warm_up_model(self):
        """Mock warmup."""
        self.logger.info("Warming up test service (no actual model)")
        await asyncio.sleep(0.1)

    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy", "test_mode": True}

    async def get_model_info(self):
        """Mock model info."""
        return {"name": "test-model", "type": "mock", "loaded": True}

    async def get_statistics(self):
        """Mock statistics."""
        return {
            "request_count": 0,
            "avg_request_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    async def generate_response(self, session, message, **kwargs):
        """Mock response generation."""
        from src.domain.models.chat import ChatMessage, MessageRole, TokenUsage

        await asyncio.sleep(0.5)  # Simulate processing time

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"This is a test response to: {message}",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        )

    async def generate_streaming_response(self, session, message, **kwargs):
        """Mock streaming response."""
        import uuid

        from src.domain.models.chat import StreamingResponse

        test_response = f"This is a test streaming response to: {message}"
        words = test_response.split()

        # Generate IDs for the streaming response
        session_id = session.id if hasattr(session, "id") else str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield StreamingResponse(
                session_id=session_id,
                message_id=message_id,
                content=chunk,
                is_complete=(i == len(words) - 1),
                token_count=i + 1,
            )
            await asyncio.sleep(0.1)  # Simulate streaming delay

    async def cleanup(self):
        """Mock cleanup."""
        self.logger.info("Cleaning up test inference service")
        self._initialized = False


@asynccontextmanager
async def test_lifespan(app: FastAPI):
    """Test lifespan manager that uses mock inference service."""
    logger.info("Starting test server...")

    # Import the state management
    from src.server.state import get_shutdown_event, set_inference_service, set_startup_time

    set_startup_time()

    try:
        # Create and initialize test inference service
        settings = get_settings()
        test_service = TestInferenceService(settings)
        await test_service.initialize()
        await test_service._warm_up_model()

        # Set the service in state
        set_inference_service(test_service)

        logger.info("Test server startup completed")
        yield

    except Exception as e:
        logger.error(f"Failed to start test server: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down test server...")
        get_shutdown_event().set()
        logger.info("Test server shutdown completed")


def create_test_app() -> FastAPI:
    """Create FastAPI app configured for testing."""
    settings = get_settings()

    # Override security settings for testing
    settings.security.api_key_required = False
    settings.security.enable_rate_limiting = False
    settings.security.enable_request_validation = True

    # Create the app but replace the lifespan
    app = create_app(settings)
    app.router.lifespan_context = test_lifespan

    return app


async def run_test_server(
    host: str = "localhost",
    port: int = 8000,
    reload: bool = False,
):
    """Run the test server."""
    logger.info(f"Starting test server on {host}:{port}")
    logger.info("Note: This server uses mock responses and does not load actual models")

    app = create_test_app()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        use_colors=True,
    )

    server = uvicorn.Server(config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Test server interrupted by user")
    except Exception as e:
        logger.error(f"Test server error: {e}")
        raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Start FastAPI server in test mode")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Set environment variable to indicate test mode
    os.environ["GEMMA_ENVIRONMENT"] = "test"

    try:
        asyncio.run(run_test_server(args.host, args.port, args.reload))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
