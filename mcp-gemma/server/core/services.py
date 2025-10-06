"""
Service implementations following Single Responsibility Principle.
Each service has exactly one reason to change.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .contracts import (
    GenerationRequest,
    GenerationResponse,
    IGenerationService,
    IMemoryRepository,
    IMemoryService,
    IMetricsObserver,
    IMetricsService,
    IModelRepository,
    IModelService,
    MemoryEntry,
    MetricsSnapshot,
    ModelInfo,
)
from .repositories import FileModelRepository, InMemoryRepository, RedisMemoryRepository


class ModelService(IModelService):
    """Service responsible for model management only."""

    def __init__(self, repository: IModelRepository, config: Any):
        self.repository = repository
        self.config = config
        self.current_model: Optional[ModelInfo] = None
        self.gemma_interface = None
        self.logger = logging.getLogger(__name__)
        self.observers: List[IMetricsObserver] = []

    def add_observer(self, observer: IMetricsObserver):
        """Add an observer for model events."""
        self.observers.append(observer)

    async def load_model(self, model_info: ModelInfo) -> None:
        """Load a model."""
        try:
            # Import GemmaInterface
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "gemma"))
            from gemma_cli import GemmaInterface

            self.gemma_interface = GemmaInterface(
                model_path=model_info.path,
                tokenizer_path=model_info.tokenizer_path,
                gemma_executable=self.config.gemma_executable,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            self.current_model = model_info
            self.logger.info(f"Loaded model: {model_info.name}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    async def switch_model(self, model_info: ModelInfo) -> None:
        """Switch to a different model."""
        old_model = self.current_model.name if self.current_model else None

        # Load the new model
        await self.load_model(model_info)

        # Notify observers
        for observer in self.observers:
            observer.on_model_switched(old_model, model_info.name)

    def get_current_model(self) -> ModelInfo:
        """Get current model information."""
        if not self.current_model:
            raise RuntimeError("No model loaded")
        return self.current_model

    def list_available_models(self) -> List[ModelInfo]:
        """List all available models."""
        return self.repository.find_all()

    def get_interface(self):
        """Get the underlying Gemma interface."""
        if not self.gemma_interface:
            raise RuntimeError("No model loaded")
        return self.gemma_interface


class GenerationService(IGenerationService):
    """Service responsible for text generation only."""

    def __init__(
        self, model_service: ModelService, metrics_service: Optional[IMetricsService] = None
    ):
        self.model_service = model_service
        self.metrics_service = metrics_service
        self.logger = logging.getLogger(__name__)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a request."""
        start_time = time.time()

        try:
            # Get the model interface
            interface = self.model_service.get_interface()

            # Apply generation parameters
            original_max_tokens = interface.max_tokens
            original_temperature = interface.temperature

            if request.max_tokens:
                interface.max_tokens = request.max_tokens
            if request.temperature is not None:
                interface.temperature = request.temperature

            try:
                # Build prompt with context if provided
                prompt = self._build_prompt_with_context(request.prompt, request.context)

                # Generate response
                text = await interface.generate_response(prompt)

                # Calculate metrics
                response_time = time.time() - start_time
                tokens_generated = len(text.split())

                # Record metrics if service is available
                if self.metrics_service:
                    self.metrics_service.record_request(response_time, tokens_generated)

                return GenerationResponse(
                    text=text,
                    tokens_generated=tokens_generated,
                    response_time=response_time,
                    metadata={
                        "model": self.model_service.get_current_model().name,
                        "temperature": interface.temperature,
                        "max_tokens": interface.max_tokens,
                    },
                )

            finally:
                # Restore original parameters
                interface.max_tokens = original_max_tokens
                interface.temperature = original_temperature

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            if self.metrics_service:
                # Record error in metrics
                response_time = time.time() - start_time
                self.metrics_service.record_request(response_time, 0)
            raise

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        interface = self.model_service.get_interface()

        # Build prompt with context
        prompt = self._build_prompt_with_context(request.prompt, request.context)

        # Stream chunks
        async def stream_callback(chunk: str):
            yield chunk

        # Apply parameters temporarily
        original_max_tokens = interface.max_tokens
        original_temperature = interface.temperature

        if request.max_tokens:
            interface.max_tokens = request.max_tokens
        if request.temperature is not None:
            interface.temperature = request.temperature

        try:
            response = await interface.generate_response(prompt, stream_callback=stream_callback)
            yield response

        finally:
            interface.max_tokens = original_max_tokens
            interface.temperature = original_temperature

    def _build_prompt_with_context(
        self, prompt: str, context: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build prompt with conversation context."""
        if not context:
            return prompt

        lines = []
        for turn in context:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                lines.append(f"Human: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")

        lines.append(f"Human: {prompt}")
        lines.append("Assistant:")

        return "\n".join(lines)


class MemoryService(IMemoryService):
    """Service responsible for memory management only."""

    def __init__(self, repository: IMemoryRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        # Generate ID if not provided
        if not entry.id:
            entry.id = str(uuid.uuid4())

        # Set timestamp if not provided
        if not entry.timestamp:
            entry.timestamp = time.time()

        await self.repository.save(entry)
        self.logger.debug(f"Stored memory entry: {entry.key}")
        return entry.id

    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key."""
        entry = await self.repository.find_by_key(key)
        if entry:
            self.logger.debug(f"Retrieved memory entry: {key}")
        return entry

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries."""
        entries = await self.repository.find_by_query(query, limit)
        self.logger.debug(f"Found {len(entries)} entries for query: {query}")
        return entries

    async def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        result = await self.repository.delete_by_key(key)
        if result:
            self.logger.debug(f"Deleted memory entry: {key}")
        return result

    async def list_keys(self) -> List[str]:
        """List all memory keys."""
        keys = await self.repository.find_all_keys()
        self.logger.debug(f"Listed {len(keys)} memory keys")
        return keys


class MetricsService(IMetricsService, IMetricsObserver):
    """Service responsible for metrics and monitoring only."""

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_response_time = 0.0
        self.model_switches = 0
        self.errors = 0
        self.last_request_time = None
        self.logger = logging.getLogger(__name__)

        # Nested class for health checking
        self._health_checker = self.HealthChecker(self)

    class HealthChecker:
        """Nested class for health check operations."""

        def __init__(self, metrics_service):
            self.metrics_service = metrics_service

        async def check_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
            """Check health of various components."""
            health = {"status": "healthy", "checks": {}}

            for name, component in components.items():
                try:
                    if hasattr(component, "health_check"):
                        health["checks"][name] = await component.health_check()
                    else:
                        health["checks"][name] = {"status": "healthy"}
                except Exception as e:
                    health["checks"][name] = {"status": "unhealthy", "error": str(e)}
                    health["status"] = "degraded"

            return health

    def record_request(self, response_time: float, tokens_generated: int) -> None:
        """Record a request for metrics."""
        self.total_requests += 1
        self.total_tokens += tokens_generated
        self.total_response_time += response_time
        self.last_request_time = time.time()
        self.logger.debug(f"Recorded request: {response_time:.2f}s, {tokens_generated} tokens")

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Calculate averages
        avg_response_time = (
            self.total_response_time / self.total_requests if self.total_requests > 0 else 0.0
        )

        # Calculate requests per minute
        requests_per_minute = (self.total_requests / uptime) * 60 if uptime > 0 else 0.0

        return MetricsSnapshot(
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
            avg_response_time=avg_response_time,
            requests_per_minute=requests_per_minute,
            uptime_seconds=uptime,
            timestamp=current_time,
        )

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_response_time = 0.0
        self.model_switches = 0
        self.errors = 0
        self.last_request_time = None
        self.logger.info("Metrics reset")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "metrics": {
                "total_requests": self.total_requests,
                "uptime_seconds": time.time() - self.start_time,
                "errors": self.errors,
            },
        }

    # IMetricsObserver implementation
    def on_request_completed(self, response_time: float, tokens: int) -> None:
        """Called when a request is completed."""
        self.record_request(response_time, tokens)

    def on_model_switched(self, old_model: str, new_model: str) -> None:
        """Called when model is switched."""
        self.model_switches += 1
        self.logger.info(f"Model switched from {old_model} to {new_model}")

    def on_error_occurred(self, error: Exception) -> None:
        """Called when an error occurs."""
        self.errors += 1
        self.logger.error(f"Error recorded: {error}")
