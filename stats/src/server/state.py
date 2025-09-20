"""Global application state management.

This module manages the global state of the FastAPI application including
inference service instances, startup time, and shutdown events.
"""

import asyncio
import time
from typing import Any
from typing import Dict
from typing import Optional

from ..application.inference.service import InferenceService

# Global application state
app_state: dict[str, Any] = {
    "inference_service": None,
    "startup_time": None,
    "shutdown_event": asyncio.Event(),
}


def get_inference_service() -> InferenceService | None:
    """Get the current inference service instance.

    Returns:
        InferenceService instance or None if not available
    """
    return app_state.get("inference_service")


def set_inference_service(service: InferenceService) -> None:
    """Set the inference service instance.

    Args:
        service: InferenceService instance to set
    """
    app_state["inference_service"] = service


def get_startup_time() -> float | None:
    """Get the server startup time.

    Returns:
        Startup timestamp or None if not set
    """
    return app_state.get("startup_time")


def set_startup_time(timestamp: float | None = None) -> None:
    """Set the server startup time.

    Args:
        timestamp: Startup timestamp (uses current time if None)
    """
    app_state["startup_time"] = timestamp or time.time()


def get_shutdown_event() -> asyncio.Event:
    """Get the shutdown event.

    Returns:
        Asyncio Event for shutdown coordination
    """
    event = app_state.get("shutdown_event")
    if not event:
        event = asyncio.Event()
        app_state["shutdown_event"] = event
    return event


def get_uptime() -> float:
    """Get the current server uptime in seconds.

    Returns:
        Uptime in seconds or 0 if not started
    """
    startup_time = get_startup_time()
    if not startup_time:
        return 0.0
    return time.time() - startup_time
