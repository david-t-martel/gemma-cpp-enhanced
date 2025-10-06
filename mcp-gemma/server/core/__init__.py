"""
Core components for MCP Gemma server.

This module provides the core architectural components following SOLID principles.
"""

from .config import Configuration, ConfigurationBuilder
from .contracts import (
    IGenerationService,
    IMemoryService,
    IMetricsService,
    IModelService,
    IRequestHandler,
    ITransport,
)
from .factory import ServerFactory
from .server import MCPServer
from .services import (
    GenerationService,
    MemoryService,
    MetricsService,
    ModelService,
)

__all__ = [
    # Configuration
    "Configuration",
    "ConfigurationBuilder",
    # Contracts
    "IModelService",
    "IMemoryService",
    "IMetricsService",
    "IGenerationService",
    "ITransport",
    "IRequestHandler",
    # Services
    "ModelService",
    "MemoryService",
    "MetricsService",
    "GenerationService",
    # Server
    "MCPServer",
    "ServerFactory",
]
