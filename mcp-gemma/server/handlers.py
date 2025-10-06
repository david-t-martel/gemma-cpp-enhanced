"""
Handler modules for specific MCP Gemma server functionality.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseHandler:
    """Base class for all handlers."""

    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)


class GenerationHandler(BaseHandler):
    """Handler for text generation operations."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with the current model."""
        max_tokens = kwargs.get("max_tokens", self.server.config.max_tokens)
        temperature = kwargs.get("temperature", self.server.config.temperature)
        stream = kwargs.get("stream", False)

        # Store original parameters
        original_max_tokens = self.server.gemma_interface.max_tokens
        original_temperature = self.server.gemma_interface.temperature

        try:
            # Set new parameters
            self.server.gemma_interface.max_tokens = max_tokens
            self.server.gemma_interface.temperature = temperature

            if stream:
                response = await self.server.gemma_interface.generate_response(
                    prompt, stream_callback=kwargs.get("stream_callback")
                )
            else:
                response = await self.server.gemma_interface.generate_response(prompt)

            return response

        finally:
            # Restore original parameters
            self.server.gemma_interface.max_tokens = original_max_tokens
            self.server.gemma_interface.temperature = original_temperature

    async def generate_with_context(
        self, prompt: str, context: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text with conversation context."""
        # Build full prompt with context
        full_prompt = self._build_contextual_prompt(prompt, context)
        return await self.generate(full_prompt, **kwargs)

    def _build_contextual_prompt(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Build a prompt with conversation context."""
        context_lines = []
        for turn in context:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                context_lines.append(f"Human: {content}")
            elif role == "assistant":
                context_lines.append(f"Assistant: {content}")

        context_lines.append(f"Human: {prompt}")
        context_lines.append("Assistant:")

        return "\n".join(context_lines)


class ModelHandler(BaseHandler):
    """Handler for model management operations."""

    def __init__(self, server):
        super().__init__(server)
        self.available_models = self._discover_models()

    def _discover_models(self) -> List[Dict[str, str]]:
        """Discover available models in the models directory."""
        models = []
        models_dir = Path("/c/codedev/llm/.models")

        if models_dir.exists():
            for model_file in models_dir.glob("*.sbs"):
                models.append(
                    {
                        "name": model_file.stem,
                        "path": str(model_file),
                        "size": model_file.stat().st_size,
                        "type": "sbs",
                    }
                )

        # Also check for other model formats
        for pattern in ["*.bin", "*.safetensors", "*.gguf"]:
            for model_file in models_dir.glob(pattern):
                models.append(
                    {
                        "name": model_file.stem,
                        "path": str(model_file),
                        "size": model_file.stat().st_size,
                        "type": model_file.suffix[1:],  # Remove the dot
                    }
                )

        return models

    async def switch_model(self, model_path: str, tokenizer_path: Optional[str] = None) -> bool:
        """Switch to a different model."""
        try:
            # Import GemmaInterface
            import sys

            sys.path.append(str(Path(__file__).parent.parent.parent / "gemma"))
            from gemma_cli import GemmaInterface

            # Create new interface
            new_interface = GemmaInterface(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                gemma_executable=self.server.config.gemma_executable,
                max_tokens=self.server.config.max_tokens,
                temperature=self.server.config.temperature,
            )

            # Test the new interface
            await new_interface.generate_response("Hello", max_tokens=10)

            # If successful, replace the old interface
            self.server.gemma_interface = new_interface
            self.server.config.model_path = model_path
            if tokenizer_path:
                self.server.config.tokenizer_path = tokenizer_path

            self.logger.info(f"Successfully switched to model: {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to switch model: {e}")
            raise

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        return {
            "model_path": self.server.config.model_path,
            "tokenizer_path": self.server.config.tokenizer_path,
            "max_tokens": self.server.config.max_tokens,
            "temperature": self.server.config.temperature,
            "gemma_executable": self.server.config.gemma_executable,
        }

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return self.available_models


class MemoryHandler(BaseHandler):
    """Handler for memory management operations using Redis."""

    def __init__(self, server):
        super().__init__(server)
        self.memory_prefix = "gemma_mcp"

    async def store(self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory."""
        if not self.server.redis_client:
            raise Exception("Redis not available for memory operations")

        memory_data = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "id": str(uuid.uuid4()),
        }

        full_key = f"{self.memory_prefix}:memory:{key}"
        self.server.redis_client.set(full_key, json.dumps(memory_data))

        # Also store in a searchable index
        await self._update_search_index(key, content, metadata)

        return memory_data["id"]

    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve content from memory."""
        if not self.server.redis_client:
            raise Exception("Redis not available for memory operations")

        full_key = f"{self.memory_prefix}:memory:{key}"
        memory_data = self.server.redis_client.get(full_key)

        if memory_data:
            return json.loads(memory_data)
        return None

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory by content similarity."""
        if not self.server.redis_client:
            raise Exception("Redis not available for memory operations")

        # Simple text-based search - could be enhanced with vector similarity
        pattern = f"{self.memory_prefix}:memory:*"
        keys = self.server.redis_client.keys(pattern)

        results = []
        for key in keys:
            if len(results) >= limit:
                break

            memory_data = self.server.redis_client.get(key)
            if memory_data:
                data = json.loads(memory_data)
                # Simple text matching
                if query.lower() in data["content"].lower():
                    results.append(
                        {
                            "key": key.replace(f"{self.memory_prefix}:memory:", ""),
                            "content": data["content"],
                            "metadata": data.get("metadata", {}),
                            "timestamp": data.get("timestamp"),
                            "relevance_score": self._calculate_relevance(query, data["content"]),
                        }
                    )

        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    async def _update_search_index(
        self, key: str, content: str, metadata: Optional[Dict[str, Any]]
    ):
        """Update search index for efficient querying."""
        # Add content to search index
        index_key = f"{self.memory_prefix}:search_index"

        # Simple keyword extraction
        keywords = self._extract_keywords(content)
        for keyword in keywords:
            self.server.redis_client.sadd(f"{index_key}:{keyword}", key)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content for indexing."""
        # Simple keyword extraction - could be enhanced with NLP
        import re

        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        return list(set(words))[:50]  # Limit to 50 unique keywords

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        if union == 0:
            return 0.0

        return intersection / union

    async def list_keys(self) -> List[str]:
        """List all memory keys."""
        if not self.server.redis_client:
            raise Exception("Redis not available for memory operations")

        pattern = f"{self.memory_prefix}:memory:*"
        keys = self.server.redis_client.keys(pattern)
        return [key.replace(f"{self.memory_prefix}:memory:", "") for key in keys]

    async def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        if not self.server.redis_client:
            raise Exception("Redis not available for memory operations")

        full_key = f"{self.memory_prefix}:memory:{key}"
        result = self.server.redis_client.delete(full_key)
        return result > 0

    async def clear_all(self) -> int:
        """Clear all memory entries."""
        if not self.server.redis_client:
            raise Exception("Redis not available for memory operations")

        pattern = f"{self.memory_prefix}:*"
        keys = self.server.redis_client.keys(pattern)
        if keys:
            return self.server.redis_client.delete(*keys)
        return 0


class MetricsHandler(BaseHandler):
    """Handler for performance metrics and monitoring."""

    def __init__(self, server):
        super().__init__(server)
        self.start_time = time.time()

    def get_server_metrics(self) -> Dict[str, Any]:
        """Get comprehensive server metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Calculate requests per minute
        total_requests = self.server.metrics["total_requests"]
        requests_per_minute = (total_requests / (uptime / 60)) if uptime > 0 else 0

        return {
            "server_info": {
                "name": self.server.server.name,
                "version": self.server.server.version,
                "uptime_seconds": uptime,
                "start_time": self.start_time,
                "current_time": current_time,
            },
            "model_info": {
                "model_path": self.server.config.model_path,
                "tokenizer_path": self.server.config.tokenizer_path,
                "max_tokens": self.server.config.max_tokens,
                "temperature": self.server.config.temperature,
            },
            "performance": {
                **self.server.metrics,
                "requests_per_minute": requests_per_minute,
                "uptime_seconds": uptime,
            },
            "memory": {
                "redis_enabled": self.server.redis_client is not None,
                "redis_connected": (
                    self._check_redis_connection() if self.server.redis_client else False
                ),
            },
        }

    def _check_redis_connection(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.server.redis_client.ping()
            return True
        except:
            return False

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model-specific metrics."""
        return {
            "total_tokens_generated": self.server.metrics["total_tokens_generated"],
            "average_tokens_per_request": (
                self.server.metrics["total_tokens_generated"]
                / max(1, self.server.metrics["total_requests"])
            ),
            "current_model": self.server.config.model_path,
            "generation_parameters": {
                "max_tokens": self.server.config.max_tokens,
                "temperature": self.server.config.temperature,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of all server components."""
        health = {"status": "healthy", "checks": {}}

        # Check Gemma interface
        try:
            # Quick test generation
            test_response = await self.server.gemma_interface.generate_response(
                "Test", max_tokens=5
            )
            health["checks"]["gemma_interface"] = {
                "status": "healthy",
                "test_response_length": len(test_response),
            }
        except Exception as e:
            health["checks"]["gemma_interface"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "unhealthy"

        # Check Redis connection
        if self.server.redis_client:
            try:
                self.server.redis_client.ping()
                health["checks"]["redis"] = {"status": "healthy"}
            except Exception as e:
                health["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        else:
            health["checks"]["redis"] = {"status": "disabled"}

        return health

    def reset_metrics(self):
        """Reset performance metrics."""
        self.server.metrics = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "average_response_time": 0.0,
            "requests_per_minute": 0,
            "last_request_time": None,
        }
        self.start_time = time.time()
        self.logger.info("Metrics reset")
