"""
Client interface definitions following Interface Segregation Principle.
Clients should only depend on the interfaces they actually use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional


# Data Transfer Objects (DTOs)
@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    context: Optional[List[Dict[str, str]]] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    tokens_generated: int
    response_time: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MemoryEntry:
    """Memory storage entry."""

    key: str
    content: str
    metadata: Dict[str, Any]
    timestamp: Optional[float] = None
    id: Optional[str] = None


@dataclass
class ModelInfo:
    """Model information."""

    name: str
    path: str
    type: str
    size: Optional[int] = None
    tokenizer_path: Optional[str] = None


@dataclass
class MetricsData:
    """Metrics data."""

    total_requests: int
    total_tokens: int
    avg_response_time: float
    requests_per_minute: float
    uptime_seconds: float


@dataclass
class HealthStatus:
    """Health check status."""

    status: str  # healthy, degraded, unhealthy
    checks: Dict[str, Any]
    timestamp: float


# Client Interfaces (ISP - Interface Segregation Principle)
class IGenerationClient(ABC):
    """Interface for text generation operations."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a request."""
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        pass


class IModelClient(ABC):
    """Interface for model management operations."""

    @abstractmethod
    async def switch_model(self, model_info: ModelInfo) -> bool:
        """Switch to a different model."""
        pass

    @abstractmethod
    async def get_current_model(self) -> ModelInfo:
        """Get current model information."""
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        pass


class IMemoryClient(ABC):
    """Interface for memory operations."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def list_keys(self) -> List[str]:
        """List all memory keys."""
        pass


class IMetricsClient(ABC):
    """Interface for metrics operations."""

    @abstractmethod
    async def get_metrics(self) -> MetricsData:
        """Get current metrics."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Perform health check."""
        pass

    @abstractmethod
    async def reset_metrics(self) -> bool:
        """Reset metrics."""
        pass


# Transport Adapter Interface (Adapter Pattern)
class ITransportAdapter(ABC):
    """Interface for transport adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        pass

    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request to the server."""
        pass

    @abstractmethod
    async def send_stream_request(
        self, method: str, params: Dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        """Send a streaming request to the server."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        pass


# Exception hierarchy
class GemmaClientError(Exception):
    """Base exception for Gemma client errors."""

    pass


class ConnectionError(GemmaClientError):
    """Connection-related errors."""

    pass


class RequestError(GemmaClientError):
    """Request-related errors."""

    pass


class TimeoutError(GemmaClientError):
    """Timeout errors."""

    pass


class ServerError(GemmaClientError):
    """Server-side errors."""

    pass
