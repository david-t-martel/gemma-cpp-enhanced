"""
Client implementations following Single Responsibility Principle.
Each client has exactly one responsibility.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .contracts import (
    GenerationRequest,
    GenerationResponse,
    HealthStatus,
    IGenerationClient,
    IMemoryClient,
    IMetricsClient,
    IModelClient,
    ITransportAdapter,
    MemoryEntry,
    MetricsData,
    ModelInfo,
    RequestError,
)


class GenerationClient(IGenerationClient):
    """Client for text generation operations only."""

    def __init__(self, transport: ITransportAdapter):
        self.transport = transport
        self.logger = logging.getLogger(__name__)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a request."""
        params = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
            "context": request.context,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            result = await self.transport.send_request("generate_text", params)

            # Parse result based on response format
            if isinstance(result, str):
                text = result
                tokens = len(text.split())
                response_time = 0.0
            elif isinstance(result, dict):
                text = result.get("text", result.get("result", ""))
                tokens = result.get("tokens_generated", len(text.split()))
                response_time = result.get("response_time", 0.0)
            else:
                raise RequestError(f"Unexpected response type: {type(result)}")

            return GenerationResponse(
                text=text,
                tokens_generated=tokens,
                response_time=response_time,
                metadata={"method": "generate_text"},
            )

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RequestError(f"Generation failed: {e}")

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        params = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "context": request.context,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            async for chunk in self.transport.send_stream_request("generate_text", params):
                if isinstance(chunk, str):
                    yield chunk
                elif isinstance(chunk, dict):
                    yield chunk.get("text", chunk.get("chunk", ""))

        except Exception as e:
            self.logger.error(f"Stream generation failed: {e}")
            raise RequestError(f"Stream generation failed: {e}")


class ModelClient(IModelClient):
    """Client for model management operations only."""

    def __init__(self, transport: ITransportAdapter):
        self.transport = transport
        self.logger = logging.getLogger(__name__)

    async def switch_model(self, model_info: ModelInfo) -> bool:
        """Switch to a different model."""
        params = {
            "model_path": model_info.path,
            "tokenizer_path": model_info.tokenizer_path,
        }

        try:
            result = await self.transport.send_request("switch_model", params)
            return "success" in str(result).lower() or "switched" in str(result).lower()

        except Exception as e:
            self.logger.error(f"Model switch failed: {e}")
            raise RequestError(f"Model switch failed: {e}")

    async def get_current_model(self) -> ModelInfo:
        """Get current model information."""
        try:
            result = await self.transport.send_request("get_current_model", {})

            # Parse result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            return ModelInfo(
                name=data.get("name", "unknown"),
                path=data.get("path", ""),
                type=data.get("type", "unknown"),
                tokenizer_path=data.get("tokenizer_path"),
            )

        except Exception as e:
            self.logger.error(f"Failed to get current model: {e}")
            raise RequestError(f"Failed to get current model: {e}")

    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        try:
            result = await self.transport.send_request("list_models", {})

            # Parse result
            if isinstance(result, str):
                models_data = json.loads(result)
            else:
                models_data = result

            models = []
            for model in models_data:
                models.append(
                    ModelInfo(
                        name=model.get("name", "unknown"),
                        path=model.get("path", ""),
                        type=model.get("type", "unknown"),
                        size=model.get("size"),
                        tokenizer_path=model.get("tokenizer_path"),
                    )
                )

            return models

        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise RequestError(f"Failed to list models: {e}")


class MemoryClient(IMemoryClient):
    """Client for memory operations only."""

    def __init__(self, transport: ITransportAdapter):
        self.transport = transport
        self.logger = logging.getLogger(__name__)

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        params = {
            "key": entry.key,
            "content": entry.content,
            "metadata": entry.metadata,
        }

        try:
            result = await self.transport.send_request("store_memory", params)

            # Extract ID from result
            if isinstance(result, dict):
                return result.get("id", "")
            else:
                # Parse ID from response string
                import re

                match = re.search(r"id:\s*([a-f0-9-]+)", str(result))
                if match:
                    return match.group(1)
                return ""

        except Exception as e:
            self.logger.error(f"Memory store failed: {e}")
            raise RequestError(f"Memory store failed: {e}")

    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        try:
            result = await self.transport.send_request("retrieve_memory", {"key": key})

            # Parse result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            if "error" in data:
                return None

            return MemoryEntry(
                key=data.get("key", key),
                content=data.get("content", ""),
                metadata=data.get("metadata", {}),
                timestamp=data.get("timestamp"),
                id=data.get("id"),
            )

        except Exception as e:
            self.logger.error(f"Memory retrieve failed: {e}")
            return None

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries."""
        params = {"query": query, "limit": limit}

        try:
            result = await self.transport.send_request("search_memory", params)

            # Parse result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            entries = []
            for item in data.get("results", []):
                entries.append(
                    MemoryEntry(
                        key=item.get("key", ""),
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {}),
                        timestamp=item.get("timestamp"),
                        id=item.get("id"),
                    )
                )

            return entries

        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            raise RequestError(f"Memory search failed: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        try:
            result = await self.transport.send_request("delete_memory", {"key": key})
            return "deleted" in str(result).lower() or "success" in str(result).lower()

        except Exception as e:
            self.logger.error(f"Memory delete failed: {e}")
            raise RequestError(f"Memory delete failed: {e}")

    async def list_keys(self) -> List[str]:
        """List all memory keys."""
        try:
            result = await self.transport.send_request("list_memory_keys", {})

            # Parse result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            return data.get("keys", [])

        except Exception as e:
            self.logger.error(f"Failed to list memory keys: {e}")
            raise RequestError(f"Failed to list memory keys: {e}")


class MetricsClient(IMetricsClient):
    """Client for metrics operations only."""

    def __init__(self, transport: ITransportAdapter):
        self.transport = transport
        self.logger = logging.getLogger(__name__)

    async def get_metrics(self) -> MetricsData:
        """Get current metrics."""
        try:
            result = await self.transport.send_request("get_metrics", {})

            # Parse result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            return MetricsData(
                total_requests=data.get("total_requests", 0),
                total_tokens=data.get("total_tokens", 0),
                avg_response_time=data.get("average_response_time", 0.0),
                requests_per_minute=data.get("requests_per_minute", 0.0),
                uptime_seconds=data.get("uptime_seconds", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise RequestError(f"Failed to get metrics: {e}")

    async def health_check(self) -> HealthStatus:
        """Perform health check."""
        try:
            result = await self.transport.send_request("health_check", {})

            # Parse result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            return HealthStatus(
                status=data.get("status", "unknown"),
                checks=data.get("checks", {}),
                timestamp=data.get("timestamp", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise RequestError(f"Health check failed: {e}")

    async def reset_metrics(self) -> bool:
        """Reset metrics."""
        try:
            result = await self.transport.send_request("reset_metrics", {})
            return "success" in str(result).lower() or "reset" in str(result).lower()

        except Exception as e:
            self.logger.error(f"Failed to reset metrics: {e}")
            raise RequestError(f"Failed to reset metrics: {e}")
