"""
Base client implementation for MCP Gemma clients.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional


class GemmaClientError(Exception):
    """Base exception for Gemma client errors."""

    pass


class GemmaConnectionError(GemmaClientError):
    """Exception raised for connection-related errors."""

    pass


class GemmaTimeoutError(GemmaClientError):
    """Exception raised for timeout errors."""

    pass


class GemmaServerError(GemmaClientError):
    """Exception raised for server-side errors."""

    pass


@dataclass
class GenerationRequest:
    """Request configuration for text generation."""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    context: Optional[List[Dict[str, str]]] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_generated: Optional[int] = None
    response_time: Optional[float] = None


@dataclass
class MemoryEntry:
    """Memory entry for storage and retrieval."""

    key: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    id: Optional[str] = None


class BaseGemmaClient(ABC):
    """Abstract base class for all Gemma MCP clients."""

    def __init__(self, timeout: float = 30.0, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass

    @abstractmethod
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the server."""
        pass

    @abstractmethod
    async def generate_text_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    # Convenience methods
    async def simple_generate(self, prompt: str, **kwargs) -> str:
        """Simple text generation with default parameters."""
        request = GenerationRequest(prompt=prompt, **kwargs)
        response = await self.generate_text(request)
        return response.text

    async def chat(
        self, message: str, context: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """Chat-style generation with conversation context."""
        request = GenerationRequest(prompt=message, context=context or [], **kwargs)
        response = await self.generate_text(request)
        return response.text

    # Model management methods (implemented by subclasses if supported)
    async def switch_model(self, model_path: str, tokenizer_path: Optional[str] = None) -> bool:
        """Switch to a different model."""
        raise NotImplementedError("Model switching not supported by this client")

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        raise NotImplementedError("Model info not supported by this client")

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        raise NotImplementedError("Model listing not supported by this client")

    # Memory management methods (implemented by subclasses if supported)
    async def store_memory(self, entry: MemoryEntry) -> str:
        """Store content in memory."""
        raise NotImplementedError("Memory storage not supported by this client")

    async def retrieve_memory(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve content from memory."""
        raise NotImplementedError("Memory retrieval not supported by this client")

    async def search_memory(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory by content."""
        raise NotImplementedError("Memory search not supported by this client")

    async def list_memory_keys(self) -> List[str]:
        """List all memory keys."""
        raise NotImplementedError("Memory listing not supported by this client")

    async def delete_memory(self, key: str) -> bool:
        """Delete a memory entry."""
        raise NotImplementedError("Memory deletion not supported by this client")

    # Metrics and monitoring methods
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        raise NotImplementedError("Metrics not supported by this client")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        raise NotImplementedError("Health check not supported by this client")

    # Utility methods
    def _validate_request(self, request: GenerationRequest) -> None:
        """Validate generation request parameters."""
        if not request.prompt:
            raise ValueError("Prompt cannot be empty")

        if request.max_tokens is not None and request.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

    def _log_request(self, request: GenerationRequest) -> None:
        """Log request details if debug is enabled."""
        if self.debug:
            self.logger.debug(
                f"Generation request: prompt_length={len(request.prompt)}, "
                f"max_tokens={request.max_tokens}, temperature={request.temperature}, "
                f"stream={request.stream}"
            )

    def _log_response(self, response: GenerationResponse) -> None:
        """Log response details if debug is enabled."""
        if self.debug:
            self.logger.debug(
                f"Generation response: text_length={len(response.text)}, "
                f"tokens_generated={response.tokens_generated}, "
                f"response_time={response.response_time}"
            )


class BatchGemmaClient:
    """Client for batch processing multiple requests."""

    def __init__(self, client: BaseGemmaClient, max_concurrent: int = 5):
        self.client = client
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate responses for multiple requests concurrently."""

        async def _generate_with_semaphore(request):
            async with self.semaphore:
                return await self.client.generate_text(request)

        tasks = [_generate_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def generate_simple_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts with same parameters."""
        requests = [GenerationRequest(prompt=prompt, **kwargs) for prompt in prompts]
        responses = await self.generate_batch(requests)
        return [response.text for response in responses]


class ClientPool:
    """Pool of clients for high-throughput scenarios."""

    def __init__(self, client_factory, pool_size: int = 3):
        self.client_factory = client_factory
        self.pool_size = pool_size
        self.clients = []
        self.available_clients = asyncio.Queue()
        self._initialized = False

    async def initialize(self):
        """Initialize the client pool."""
        if self._initialized:
            return

        for _ in range(self.pool_size):
            client = self.client_factory()
            await client.connect()
            self.clients.append(client)
            await self.available_clients.put(client)

        self._initialized = True

    async def acquire(self) -> BaseGemmaClient:
        """Acquire a client from the pool."""
        if not self._initialized:
            await self.initialize()
        return await self.available_clients.get()

    async def release(self, client: BaseGemmaClient):
        """Release a client back to the pool."""
        await self.available_clients.put(client)

    async def close(self):
        """Close all clients in the pool."""
        for client in self.clients:
            try:
                await client.disconnect()
            except Exception as e:
                logging.error(f"Error closing client: {e}")

        self.clients.clear()
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
