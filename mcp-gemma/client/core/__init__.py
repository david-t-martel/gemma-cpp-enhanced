"""
Core client components following SOLID principles.
"""

from .clients import (
    GenerationClient,
    MemoryClient,
    MetricsClient,
    ModelClient,
)
from .composite import CompositeClient
from .contracts import (
    IGenerationClient,
    IMemoryClient,
    IMetricsClient,
    IModelClient,
    ITransportAdapter,
)
from .factory import ClientFactory

__all__ = [
    # Contracts
    "IGenerationClient",
    "IMemoryClient",
    "IMetricsClient",
    "IModelClient",
    "ITransportAdapter",
    # Implementations
    "GenerationClient",
    "MemoryClient",
    "MetricsClient",
    "ModelClient",
    "CompositeClient",
    "ClientFactory",
]
