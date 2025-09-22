"""Health check and monitoring endpoints.

This module provides comprehensive health checks, system monitoring,
and performance metrics endpoints for production deployment.
"""

import asyncio
import platform
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import psutil
import torch
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import status

from ...application.inference.service import InferenceService
from ...shared.exceptions import InferenceException
from ...shared.logging import get_logger
from ..state import get_inference_service as get_inference_service_from_state
from ..state import get_shutdown_event
from ..state import get_startup_time
from ..state import get_uptime
from ..websocket import manager as websocket_manager
from .schemas import HealthResponse
from .schemas import MetricsResponse

logger = get_logger(__name__)

health_router = APIRouter()


def get_inference_service() -> InferenceService | None:
    """Dependency to get the inference service (optional for health checks)."""
    return get_inference_service_from_state()


@health_router.get(
    "/",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(
    request: Request,
    inference_service: InferenceService | None = Depends(get_inference_service),
):
    """Basic health check endpoint.

    Returns comprehensive health status including system resources,
    model status, and performance metrics.
    """
    logger.debug("Performing health check")
    start_time = time.time()

    try:
        # Calculate uptime
        uptime_seconds = get_uptime()

        # Check model status
        model_loaded = False
        model_healthy = False
        if inference_service:
            try:
                health_result = await inference_service.health_check()
                model_loaded = health_result.get("status") == "healthy"
                model_healthy = model_loaded
            except Exception as e:
                logger.warning(f"Model health check failed: {e}")

        # Get system information
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Check GPU availability
        gpu_available = torch.cuda.is_available() if torch is not None else False
        gpu_memory = {}
        if gpu_available:
            try:
                gpu_memory = {
                    "total": torch.cuda.get_device_properties(0).total_memory,
                    "allocated": torch.cuda.memory_allocated(0),
                    "cached": torch.cuda.memory_reserved(0),
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}")
                gpu_available = False

        # Get performance statistics
        performance_stats = {}
        if inference_service:
            try:
                performance_stats = await inference_service.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get performance statistics: {e}")

        # Determine overall health status
        is_healthy = (
            model_healthy
            and memory_info.percent < 90  # Memory usage below 90%
            and cpu_percent < 95  # CPU usage below 95%
            and uptime_seconds > 0  # Service has been running
        )

        status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        health_status = "healthy" if is_healthy else "unhealthy"

        response = HealthResponse(
            status=health_status,
            version="0.1.0",  # This should come from settings
            uptime_seconds=uptime_seconds,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage={
                "total": memory_info.total,
                "used": memory_info.used,
                "available": memory_info.available,
                "percent": memory_info.percent,
                **(
                    {
                        "gpu_total": gpu_memory.get("total", 0),
                        "gpu_allocated": gpu_memory.get("allocated", 0),
                        "gpu_cached": gpu_memory.get("cached", 0),
                    }
                    if gpu_memory
                    else {}
                ),
            },
            performance_stats=performance_stats,
        )

        # Set appropriate status code
        if not is_healthy:
            raise HTTPException(status_code=status_code, detail=response.dict())

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            },
        )


@health_router.get(
    "/ready",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_check(
    inference_service: InferenceService | None = Depends(get_inference_service),
):
    """Readiness check endpoint.

    This endpoint is used by orchestrators (like Kubernetes) to determine
    if the service is ready to accept traffic.
    """
    logger.debug("Performing readiness check")

    try:
        # Check if inference service is available and initialized
        if not inference_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"ready": False, "reason": "Inference service not available"},
            )

        # Check if model is loaded
        model_info = await inference_service.get_model_info()
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"ready": False, "reason": "Model not loaded"},
            )

        # Perform a quick health check
        health_result = await inference_service.health_check()
        if health_result.get("status") != "healthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"ready": False, "reason": "Model health check failed"},
            )

        return {
            "ready": True,
            "model": model_info.get("name", "unknown"),
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"ready": False, "reason": str(e)},
        )


@health_router.get(
    "/live",
    responses={
        200: {"description": "Service is alive"},
        503: {"description": "Service is not alive"},
    },
)
async def liveness_check():
    """Liveness check endpoint.

    This endpoint is used by orchestrators (like Kubernetes) to determine
    if the service is alive and should be restarted if not.
    """
    logger.debug("Performing liveness check")

    try:
        # Basic checks to ensure the service is alive
        startup_time = get_startup_time()
        if not startup_time:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"alive": False, "reason": "Service not properly started"},
            )

        # Check if shutdown has been initiated
        shutdown_event = get_shutdown_event()
        if shutdown_event.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"alive": False, "reason": "Service is shutting down"},
            )

        # Check basic system resources
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 98:  # Very high memory usage
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"alive": False, "reason": "System memory critically low"},
            )

        return {
            "alive": True,
            "uptime": get_uptime(),
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"alive": False, "reason": str(e)},
        )


@health_router.get(
    "/metrics",
    response_model=MetricsResponse,
    responses={
        200: {"description": "System and application metrics"},
        503: {"description": "Metrics unavailable"},
    },
)
async def get_metrics(
    inference_service: InferenceService | None = Depends(get_inference_service),
):
    """Get comprehensive system and application metrics.

    Returns detailed metrics for monitoring and observability.
    """
    logger.debug("Collecting metrics")

    try:
        # System metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk_info = psutil.disk_usage("/")

        # Network metrics
        network_info = psutil.net_io_counters()

        # Process metrics
        process = psutil.Process()
        process_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_info": process.memory_info()._asdict(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time(),
        }

        # Application metrics
        uptime = get_uptime()

        # WebSocket metrics
        websocket_stats = websocket_manager.get_statistics()

        # Model metrics
        model_info = {}
        performance_stats = {}
        if inference_service:
            try:
                model_info = await inference_service.get_model_info()
                performance_stats = await inference_service.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get model metrics: {e}")

        # GPU metrics
        gpu_info = {}
        if torch.cuda.is_available():
            try:
                gpu_info = {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(0),
                    "memory_reserved": torch.cuda.memory_reserved(0),
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")

        return MetricsResponse(
            requests_total=performance_stats.get("request_count", 0),
            requests_per_second=performance_stats.get("request_count", 0) / max(uptime, 1),
            average_response_time=performance_stats.get("avg_request_time", 0),
            active_connections=websocket_stats.get("active_connections", 0),
            cache_hit_rate=performance_stats.get("cache_hit_rate", 0) * 100,
            model_info=model_info,
            system_info={
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory_info.total,
                    "used": memory_info.used,
                    "available": memory_info.available,
                    "percent": memory_info.percent,
                },
                "disk": {
                    "total": disk_info.total,
                    "used": disk_info.used,
                    "free": disk_info.free,
                    "percent": (disk_info.used / disk_info.total) * 100,
                },
                "network": {
                    "bytes_sent": network_info.bytes_sent,
                    "bytes_recv": network_info.bytes_recv,
                    "packets_sent": network_info.packets_sent,
                    "packets_recv": network_info.packets_recv,
                },
                "process": process_info,
                "websockets": websocket_stats,
                "gpu": gpu_info,
                "uptime_seconds": uptime,
            },
        )

    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics collection failed"
        )


@health_router.get(
    "/status",
    responses={
        200: {"description": "Detailed service status"},
    },
)
async def get_status(
    inference_service: InferenceService | None = Depends(get_inference_service),
):
    """Get detailed service status information.

    This endpoint provides comprehensive status information including
    configuration, capabilities, and current state.
    """
    logger.debug("Getting service status")

    try:
        uptime = get_uptime()

        # Service information
        service_info = {
            "name": "Gemma Chatbot API",
            "version": "0.1.0",
            "environment": "development",  # This should come from settings
            "uptime_seconds": uptime,
            "started_at": time.ctime(get_startup_time()) if get_startup_time() else "unknown",
        }

        # Configuration status
        config_status = {
            "model_loaded": False,
            "cache_enabled": True,
            "streaming_enabled": True,
            "websocket_enabled": True,
            "auth_enabled": False,  # This should come from settings
        }

        # Capabilities
        capabilities = [
            "chat-completion",
            "text-completion",
            "streaming-responses",
            "websocket-chat",
            "server-sent-events",
            "openai-compatible-api",
        ]

        # Model status
        model_status = {}
        if inference_service:
            try:
                model_info = await inference_service.get_model_info()
                health_result = await inference_service.health_check()
                model_status = {
                    "name": model_info.get("name", "unknown"),
                    "type": model_info.get("type", "unknown"),
                    "loaded": health_result.get("status") == "healthy",
                    "health": health_result.get("status", "unknown"),
                }
                config_status["model_loaded"] = model_status["loaded"]
            except Exception as e:
                logger.warning(f"Failed to get model status: {e}")
                model_status = {"error": str(e)}

        # System status
        memory_info = psutil.virtual_memory()
        system_status = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total": memory_info.total,
            "memory_used": memory_info.used,
            "memory_percent": memory_info.percent,
            "gpu_available": torch.cuda.is_available() if torch is not None else False,
        }

        # Connection status
        websocket_stats = websocket_manager.get_statistics()
        connection_status = {
            "active_websocket_connections": websocket_stats.get("active_connections", 0),
            "active_sessions": websocket_stats.get("active_sessions", 0),
            "total_connections": websocket_stats.get("total_connections", 0),
            "total_messages": websocket_stats.get("total_messages", 0),
        }

        return {
            "service": service_info,
            "configuration": config_status,
            "capabilities": capabilities,
            "model": model_status,
            "system": system_status,
            "connections": connection_status,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Status collection failed"
        )


@health_router.post(
    "/warmup",
    responses={
        200: {"description": "Warmup completed successfully"},
        503: {"description": "Warmup failed"},
    },
)
async def warmup(
    inference_service: InferenceService | None = Depends(get_inference_service),
):
    """Warm up the model and services.

    This endpoint can be called to ensure the model is loaded and ready
    for inference requests.
    """
    logger.info("Starting service warmup")

    try:
        if not inference_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service not available",
            )

        start_time = time.time()

        # Warm up the model
        await inference_service._warm_up_model()

        warmup_time = time.time() - start_time

        return {
            "status": "warmed_up",
            "warmup_time_seconds": warmup_time,
            "model_ready": True,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Warmup failed: {e!s}"
        )
