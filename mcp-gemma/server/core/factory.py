"""
Factory for creating server components.
Follows Factory pattern to encapsulate object creation logic.
"""

import logging
from typing import Optional

import redis

from .config import Configuration
from .contracts import IMemoryService, IMetricsService, IModelService
from .repositories import FileModelRepository, MemoryRepositoryFactory
from .server import MCPServer
from .services import GenerationService, MemoryService, MetricsService, ModelService


class ServerFactory:
    """Factory for creating MCP server with all dependencies."""

    @staticmethod
    def create_from_config(config: Configuration) -> MCPServer:
        """Create a fully configured MCP server."""
        logging.info("Creating MCP server from configuration")

        # Create repositories
        model_repository = FileModelRepository()

        # Create services
        model_service = ServerFactory._create_model_service(config, model_repository)
        metrics_service = ServerFactory._create_metrics_service(config)
        memory_service = ServerFactory._create_memory_service(config)

        # Wire up observers
        if metrics_service:
            model_service.add_observer(metrics_service)

        # Create generation service with dependencies
        generation_service = GenerationService(model_service, metrics_service)

        # Create and return server
        server = MCPServer(
            config=config,
            generation_service=generation_service,
            model_service=model_service,
            memory_service=memory_service,
            metrics_service=metrics_service,
        )

        return server

    @staticmethod
    def _create_model_service(
        config: Configuration, repository: FileModelRepository
    ) -> ModelService:
        """Create model service."""
        return ModelService(repository, config)

    @staticmethod
    def _create_metrics_service(config: Configuration) -> Optional[IMetricsService]:
        """Create metrics service if enabled."""
        if config.metrics_enabled:
            return MetricsService()
        return None

    @staticmethod
    def _create_memory_service(config: Configuration) -> Optional[IMemoryService]:
        """Create memory service based on configuration."""
        if not config.enable_memory:
            return None

        # Create repository based on backend
        if config.memory_backend == "redis":
            try:
                # Test Redis connection
                redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=True,
                )
                redis_client.ping()

                repository = MemoryRepositoryFactory.create(
                    "redis", host=config.redis_host, port=config.redis_port, db=config.redis_db
                )
            except Exception as e:
                logging.warning(f"Redis not available: {e}, falling back to in-memory")
                repository = MemoryRepositoryFactory.create("inmemory")
        else:
            repository = MemoryRepositoryFactory.create(config.memory_backend)

        return MemoryService(repository)


class TransportFactory:
    """Factory for creating transport implementations."""

    @staticmethod
    def create_stdio_transport(server: MCPServer):
        """Create stdio transport."""
        from ..transports import StdioTransportStrategy

        return StdioTransportStrategy(server)

    @staticmethod
    def create_http_transport(server: MCPServer, host: str = "localhost", port: int = 8080):
        """Create HTTP transport."""
        from ..transports import HTTPTransportStrategy

        return HTTPTransportStrategy(server, host, port)

    @staticmethod
    def create_websocket_transport(server: MCPServer, host: str = "localhost", port: int = 8081):
        """Create WebSocket transport."""
        from ..transports import WebSocketTransportStrategy

        return WebSocketTransportStrategy(server, host, port)

    @staticmethod
    def create_from_mode(server: MCPServer, mode: str, **kwargs):
        """Create transport based on mode."""
        if mode == "stdio":
            return TransportFactory.create_stdio_transport(server)
        elif mode == "http":
            return TransportFactory.create_http_transport(
                server, kwargs.get("host", "localhost"), kwargs.get("port", 8080)
            )
        elif mode == "websocket":
            return TransportFactory.create_websocket_transport(
                server, kwargs.get("host", "localhost"), kwargs.get("port", 8081)
            )
        else:
            raise ValueError(f"Unknown transport mode: {mode}")
