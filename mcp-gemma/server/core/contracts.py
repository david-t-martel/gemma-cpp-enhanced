"""
Interface definitions following the Interface Segregation Principle.
These contracts define the boundaries between components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union


# Data Transfer Objects
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
    timestamp: float
    id: str


@dataclass
class ModelInfo:
    """Model information."""

    path: str
    name: str
    size: int
    type: str
    tokenizer_path: Optional[str] = None


@dataclass
class MetricsSnapshot:
    """Performance metrics snapshot."""

    total_requests: int
    total_tokens: int
    avg_response_time: float
    requests_per_minute: float
    uptime_seconds: float
    timestamp: float


# Service Interfaces (following ISP - Interface Segregation Principle)
class IGenerationService(ABC):
    """Interface for text generation operations."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a request."""
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        pass


class IModelService(ABC):
    """Interface for model management operations."""

    @abstractmethod
    async def load_model(self, model_info: ModelInfo) -> None:
        """Load a model."""
        pass

    @abstractmethod
    async def switch_model(self, model_info: ModelInfo) -> None:
        """Switch to a different model."""
        pass

    @abstractmethod
    def get_current_model(self) -> ModelInfo:
        """Get current model information."""
        pass

    @abstractmethod
    def list_available_models(self) -> List[ModelInfo]:
        """List all available models."""
        pass


class IMemoryService(ABC):
    """Interface for memory management operations."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key."""
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


class IMetricsService(ABC):
    """Interface for metrics and monitoring operations."""

    @abstractmethod
    def record_request(self, response_time: float, tokens_generated: int) -> None:
        """Record a request for metrics."""
        pass

    @abstractmethod
    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass


# Transport and Handler Interfaces
class ITransport(ABC):
    """Interface for transport implementations (Strategy Pattern)."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        pass

    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming request."""
        pass


class IRequestHandler(ABC):
    """Interface for request handlers."""

    @abstractmethod
    async def handle(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle a specific request."""
        pass

    @abstractmethod
    def can_handle(self, method: str) -> bool:
        """Check if this handler can handle the method."""
        pass


# Repository Pattern for data access
class IModelRepository(ABC):
    """Repository interface for model storage."""

    @abstractmethod
    def find_all(self) -> List[ModelInfo]:
        """Find all available models."""
        pass

    @abstractmethod
    def find_by_name(self, name: str) -> Optional[ModelInfo]:
        """Find a model by name."""
        pass

    @abstractmethod
    def find_by_path(self, path: str) -> Optional[ModelInfo]:
        """Find a model by path."""
        pass


class IMemoryRepository(ABC):
    """Repository interface for memory storage."""

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        pass

    @abstractmethod
    async def find_by_key(self, key: str) -> Optional[MemoryEntry]:
        """Find entry by key."""
        pass

    @abstractmethod
    async def find_by_query(self, query: str, limit: int) -> List[MemoryEntry]:
        """Find entries by query."""
        pass

    @abstractmethod
    async def delete_by_key(self, key: str) -> bool:
        """Delete entry by key."""
        pass

    @abstractmethod
    async def find_all_keys(self) -> List[str]:
        """Find all keys."""
        pass


# Observer Pattern for metrics
class IMetricsObserver(ABC):
    """Observer interface for metrics events."""

    @abstractmethod
    def on_request_completed(self, response_time: float, tokens: int) -> None:
        """Called when a request is completed."""
        pass

    @abstractmethod
    def on_model_switched(self, old_model: str, new_model: str) -> None:
        """Called when model is switched."""
        pass

    @abstractmethod
    def on_error_occurred(self, error: Exception) -> None:
        """Called when an error occurs."""
        pass
