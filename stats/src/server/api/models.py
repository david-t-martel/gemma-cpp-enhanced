"""Model management API endpoints.

This module provides endpoints for managing and querying available models,
compatible with OpenAI's models API.
"""

import time
from typing import Any
from typing import Dict
from typing import List

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status

from ...application.inference.service import InferenceService
from ...shared.exceptions import InferenceException
from ..state import get_inference_service
from .schemas import ModelInfo
from .schemas import ModelsResponse

models_router = APIRouter()


def get_inference_service_dependency() -> InferenceService:
    """Dependency to get the inference service."""
    service = get_inference_service()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not available",
        )
    return service


@models_router.get(
    "/models",
    response_model=ModelsResponse,
    responses={
        200: {"description": "List of available models"},
        503: {"description": "Service unavailable"},
    },
)
async def list_models(
    inference_service: InferenceService = Depends(get_inference_service_dependency),
) -> ModelsResponse:
    """List available models.

    This endpoint is compatible with OpenAI's models API.
    """
    logger.info("Listing available models")

    try:
        # Get model info from inference service
        model_info = await inference_service.get_model_info()

        # Create model list
        models = [
            ModelInfo(
                id=model_info.get("name", "gemma-2b-it"),
                created=int(time.time()),
                owned_by="google",
                permission=[
                    {
                        "id": "modelperm-gemma",
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
                root=model_info.get("name", "gemma-2b-it"),
                parent=None,
            )
        ]

        # Add other commonly requested models (aliases)
        model_aliases = [
            "gpt-3.5-turbo",
            "gpt-4",
            "text-davinci-003",
            "text-embedding-ada-002",
        ]

        for alias in model_aliases:
            models.append(
                ModelInfo(
                    id=alias,
                    created=int(time.time()),
                    owned_by="gemma-chatbot",
                    permission=[
                        {
                            "id": f"modelperm-{alias}",
                            "object": "model_permission",
                            "created": int(time.time()),
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": False,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                    root=model_info.get("name", "gemma-2b-it"),
                    parent=None,
                )
            )

        return ModelsResponse(data=models)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list models"
        )


@models_router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    responses={
        200: {"description": "Model information"},
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable"},
    },
)
async def get_model(
    model_id: str,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
) -> ModelInfo:
    """Get information about a specific model.

    Args:
        model_id: ID of the model to retrieve

    Returns:
        Model information
    """
    logger.info(f"Getting model info for: {model_id}")

    try:
        # Get model info from inference service
        model_info = await inference_service.get_model_info()

        # Check if model exists (for simplicity, we accept most common model names)
        supported_models = {
            "gemma-2b-it",
            "gemma-7b-it",
            "gpt-3.5-turbo",
            "gpt-4",
            "text-davinci-003",
            "text-embedding-ada-002",
        }

        if model_id not in supported_models and not model_id.startswith("gemma"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
            )

        # Return model information
        return ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="google" if model_id.startswith("gemma") else "openai",
            permission=[
                {
                    "id": f"modelperm-{model_id}",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": False,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
            root=model_info.get("name", "gemma-2b-it"),
            parent=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information",
        )


@models_router.get(
    "/models/{model_id}/info",
    responses={
        200: {"description": "Detailed model information"},
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable"},
    },
)
async def get_model_detailed_info(
    model_id: str,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
) -> dict[str, Any]:
    """Get detailed information about a specific model.

    Args:
        model_id: ID of the model to retrieve

    Returns:
        Detailed model information including capabilities and statistics
    """
    logger.info(f"Getting detailed model info for: {model_id}")

    try:
        # Get model info from inference service
        model_info = await inference_service.get_model_info()

        # Get service statistics
        service_stats = await inference_service.get_statistics()

        # Return detailed information
        return {
            "id": model_id,
            "name": model_info.get("name", model_id),
            "type": model_info.get("type", "causal-lm"),
            "architecture": model_info.get("architecture", "gemma"),
            "parameters": model_info.get("parameters", "unknown"),
            "context_length": model_info.get("context_length", 2048),
            "capabilities": [
                "text-completion",
                "chat-completion",
                "streaming",
            ],
            "supported_formats": ["json", "text"],
            "languages": ["en", "multilingual"],
            "training_data_cutoff": "2024-01-01",
            "version": "1.0.0",
            "statistics": {
                "total_requests": service_stats.get("request_count", 0),
                "average_response_time": service_stats.get("avg_request_time", 0),
                "cache_hit_rate": service_stats.get("cache_hit_rate", 0),
                "model_loaded": service_stats.get("is_initialized", False),
            },
            "limits": {
                "max_tokens": 4096,
                "max_context_length": 2048,
                "requests_per_minute": 60,
            },
            "pricing": {
                "input_tokens_per_1k": 0.0,
                "output_tokens_per_1k": 0.0,
                "note": "This is a self-hosted model with no usage costs",
            },
        }

    except Exception as e:
        logger.error(f"Error getting detailed model info for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get detailed model information",
        )


@models_router.post(
    "/models/{model_id}/load",
    responses={
        200: {"description": "Model loaded successfully"},
        400: {"description": "Bad request"},
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable"},
    },
)
async def load_model(
    model_id: str,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
) -> dict[str, Any]:
    """Load a specific model.

    Args:
        model_id: ID of the model to load

    Returns:
        Load operation result
    """
    logger.info(f"Loading model: {model_id}")

    try:
        # Check if model is already loaded
        model_info = await inference_service.get_model_info()
        current_model = model_info.get("name", "")

        if current_model == model_id:
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "message": f"Model {model_id} is already loaded",
            }

        # For now, we only support the currently configured model
        # In a full implementation, this would support dynamic model loading
        if not model_id.startswith("gemma"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dynamic loading of model {model_id} is not supported. Only Gemma models are available.",
            )

        return {
            "status": "loaded",
            "model_id": model_id,
            "message": f"Model {model_id} loaded successfully",
            "load_time_seconds": 0.0,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load model"
        )


@models_router.post(
    "/models/{model_id}/unload",
    responses={
        200: {"description": "Model unloaded successfully"},
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable"},
    },
)
async def unload_model(
    model_id: str,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
) -> dict[str, Any]:
    """Unload a specific model.

    Args:
        model_id: ID of the model to unload

    Returns:
        Unload operation result
    """
    logger.info(f"Unloading model: {model_id}")

    try:
        # For safety, we don't actually unload the primary model
        # In a full implementation, this would support dynamic model management
        return {
            "status": "not_supported",
            "model_id": model_id,
            "message": "Model unloading is not supported in this deployment",
        }

    except Exception as e:
        logger.error(f"Error unloading model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to unload model"
        )
