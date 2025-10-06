"""
Factory for creating clients with appropriate transports.
Follows Factory pattern to encapsulate object creation.
"""

from typing import Optional, Union

from ..transport_adapters import HTTPAdapter, StdioAdapter, WebSocketAdapter
from .clients import GenerationClient, MemoryClient, MetricsClient, ModelClient
from .composite import CompositeClient
from .contracts import ITransportAdapter


class ClientFactory:
    """Factory for creating MCP Gemma clients."""

    @staticmethod
    def create_stdio_client(
        composite: bool = True, timeout: float = 30.0, debug: bool = False
    ) -> Union[CompositeClient, GenerationClient]:
        """Create a client with stdio transport."""
        transport = StdioAdapter(timeout=timeout, debug=debug)

        if composite:
            return CompositeClient(transport)
        else:
            return GenerationClient(transport)

    @staticmethod
    def create_http_client(
        url: str,
        composite: bool = True,
        timeout: float = 30.0,
        debug: bool = False,
        headers: Optional[dict] = None,
    ) -> Union[CompositeClient, GenerationClient]:
        """Create a client with HTTP transport."""
        transport = HTTPAdapter(url=url, timeout=timeout, debug=debug, headers=headers)

        if composite:
            return CompositeClient(transport)
        else:
            return GenerationClient(transport)

    @staticmethod
    def create_websocket_client(
        url: str, composite: bool = True, timeout: float = 30.0, debug: bool = False
    ) -> Union[CompositeClient, GenerationClient]:
        """Create a client with WebSocket transport."""
        transport = WebSocketAdapter(url=url, timeout=timeout, debug=debug)

        if composite:
            return CompositeClient(transport)
        else:
            return GenerationClient(transport)

    @staticmethod
    def create_client_with_transport(
        transport: ITransportAdapter, composite: bool = True
    ) -> Union[CompositeClient, GenerationClient]:
        """Create a client with a custom transport."""
        if composite:
            return CompositeClient(transport)
        else:
            return GenerationClient(transport)

    @staticmethod
    def create_specialized_clients(transport: ITransportAdapter) -> dict:
        """Create all specialized clients for those who want separate clients."""
        return {
            "generation": GenerationClient(transport),
            "model": ModelClient(transport),
            "memory": MemoryClient(transport),
            "metrics": MetricsClient(transport),
        }


class ClientBuilder:
    """Builder for creating configured clients."""

    def __init__(self):
        self._transport_type = "http"
        self._url = "http://localhost:8080"
        self._timeout = 30.0
        self._debug = False
        self._headers = None
        self._composite = True

    def with_stdio(self) -> "ClientBuilder":
        """Use stdio transport."""
        self._transport_type = "stdio"
        return self

    def with_http(self, url: str) -> "ClientBuilder":
        """Use HTTP transport."""
        self._transport_type = "http"
        self._url = url
        return self

    def with_websocket(self, url: str) -> "ClientBuilder":
        """Use WebSocket transport."""
        self._transport_type = "websocket"
        self._url = url
        return self

    def with_timeout(self, timeout: float) -> "ClientBuilder":
        """Set timeout."""
        self._timeout = timeout
        return self

    def with_debug(self, enabled: bool = True) -> "ClientBuilder":
        """Enable debug mode."""
        self._debug = enabled
        return self

    def with_headers(self, headers: dict) -> "ClientBuilder":
        """Set custom headers for HTTP transport."""
        self._headers = headers
        return self

    def as_composite(self, enabled: bool = True) -> "ClientBuilder":
        """Create composite client (default) or individual client."""
        self._composite = enabled
        return self

    def build(self) -> Union[CompositeClient, GenerationClient]:
        """Build the configured client."""
        if self._transport_type == "stdio":
            return ClientFactory.create_stdio_client(
                composite=self._composite, timeout=self._timeout, debug=self._debug
            )

        elif self._transport_type == "http":
            return ClientFactory.create_http_client(
                url=self._url,
                composite=self._composite,
                timeout=self._timeout,
                debug=self._debug,
                headers=self._headers,
            )

        elif self._transport_type == "websocket":
            return ClientFactory.create_websocket_client(
                url=self._url, composite=self._composite, timeout=self._timeout, debug=self._debug
            )

        else:
            raise ValueError(f"Unknown transport type: {self._transport_type}")
