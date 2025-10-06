"""
Composite client that combines all client interfaces.
Follows the Composite pattern for convenience.
"""

from typing import AsyncGenerator, Dict, List, Optional

from .clients import GenerationClient, MemoryClient, MetricsClient, ModelClient
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
)


class CompositeClient(IGenerationClient, IModelClient, IMemoryClient, IMetricsClient):
    """
    Composite client that provides all interfaces.
    Users can choose to use this or individual clients based on their needs.
    """

    def __init__(self, transport: ITransportAdapter):
        self.transport = transport

        # Create individual clients (composition over inheritance)
        self.generation = GenerationClient(transport)
        self.model = ModelClient(transport)
        self.memory = MemoryClient(transport)
        self.metrics = MetricsClient(transport)

    # IGenerationClient implementation
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a request."""
        return await self.generation.generate(request)

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        async for chunk in self.generation.generate_stream(request):
            yield chunk

    # IModelClient implementation
    async def switch_model(self, model_info: ModelInfo) -> bool:
        """Switch to a different model."""
        return await self.model.switch_model(model_info)

    async def get_current_model(self) -> ModelInfo:
        """Get current model information."""
        return await self.model.get_current_model()

    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        return await self.model.list_models()

    # IMemoryClient implementation
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        return await self.memory.store(entry)

    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        return await self.memory.retrieve(key)

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries."""
        return await self.memory.search(query, limit)

    async def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        return await self.memory.delete(key)

    async def list_keys(self) -> List[str]:
        """List all memory keys."""
        return await self.memory.list_keys()

    # IMetricsClient implementation
    async def get_metrics(self) -> MetricsData:
        """Get current metrics."""
        return await self.metrics.get_metrics()

    async def health_check(self) -> HealthStatus:
        """Perform health check."""
        return await self.metrics.health_check()

    async def reset_metrics(self) -> bool:
        """Reset metrics."""
        return await self.metrics.reset_metrics()

    # Connection management
    async def connect(self) -> None:
        """Connect to the server."""
        await self.transport.connect()

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        await self.transport.disconnect()

    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self.transport.is_connected()

    # Context manager support
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
        response = await self.generate(request)
        return response.text

    async def chat(
        self, message: str, context: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """Chat-style generation with conversation context."""
        request = GenerationRequest(prompt=message, context=context or [], **kwargs)
        response = await self.generate(request)
        return response.text
