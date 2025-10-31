"""Production-ready MCP client manager for Gemma CLI.

This module provides a robust, async-first MCP client implementation with:
- Connection pooling and automatic reconnection
- Tool discovery with intelligent caching
- Comprehensive error handling and retry logic
- Health checks and session lifecycle management
- Full type safety (mypy --strict compatible)
"""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class MCPTransportType(str, Enum):
    """MCP transport protocol types."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"


class MCPServerStatus(str, Enum):
    """MCP server connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Exception raised when connection to MCP server fails."""

    pass


class MCPToolExecutionError(MCPError):
    """Exception raised when tool execution fails."""

    pass


class MCPResourceError(MCPError):
    """Exception raised when resource operations fail."""

    pass


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection.

    Attributes:
        name: Unique server identifier
        transport: Transport protocol type
        command: Command to execute for stdio transport
        args: Command arguments for stdio transport
        env: Environment variables for server process
        url: Server URL for HTTP/SSE/WebSocket transports
        enabled: Whether server is enabled
        auto_reconnect: Enable automatic reconnection
        max_reconnect_attempts: Maximum reconnection attempts
        reconnect_delay: Initial delay between reconnection attempts (exponential backoff)
        connection_timeout: Connection timeout in seconds
        request_timeout: Request timeout in seconds
        health_check_interval: Interval between health checks in seconds
    """

    name: str
    transport: MCPTransportType
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    url: Optional[str] = None
    enabled: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    connection_timeout: float = 10.0
    request_timeout: float = 30.0
    health_check_interval: float = 60.0

    @field_validator("transport", mode="before")
    @classmethod
    def validate_transport(cls, v: Any) -> MCPTransportType:
        """Validate and convert transport type."""
        if isinstance(v, str):
            return MCPTransportType(v.lower())
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if self.transport == MCPTransportType.STDIO:
            if not self.command:
                raise ValueError("command is required for stdio transport")
        elif self.transport in (
            MCPTransportType.HTTP,
            MCPTransportType.SSE,
            MCPTransportType.WEBSOCKET,
        ):
            if not self.url:
                raise ValueError(f"url is required for {self.transport} transport")


@dataclass
class CachedTool:
    """Cached tool information with TTL.

    Attributes:
        tool: The MCP tool definition
        cached_at: Timestamp when tool was cached
        ttl: Time-to-live in seconds
    """

    tool: Tool
    cached_at: float
    ttl: float

    def is_expired(self) -> bool:
        """Check if cached tool has expired.

        Returns:
            True if cache entry is expired
        """
        return time.time() - self.cached_at > self.ttl


@dataclass
class ServerConnection:
    """Active server connection with metadata.

    Attributes:
        config: Server configuration
        session: Active MCP client session
        status: Current connection status
        connected_at: Connection timestamp
        last_health_check: Last health check timestamp
        reconnect_attempts: Number of reconnection attempts
        stats: Connection statistics
    """

    config: MCPServerConfig
    session: ClientSession
    status: MCPServerStatus = MCPServerStatus.CONNECTED
    connected_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    reconnect_attempts: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize statistics."""
        if not self.stats:
            self.stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_latency": 0.0,
                "min_latency": float("inf"),
                "max_latency": 0.0,
            }


class MCPToolRegistry:
    """Tool registry with intelligent caching and invalidation.

    Provides efficient tool discovery and caching with configurable TTL.
    """

    def __init__(self, default_ttl: float = 3600.0) -> None:
        """Initialize tool registry.

        Args:
            default_ttl: Default cache TTL in seconds (default: 1 hour)
        """
        self._cache: dict[str, dict[str, CachedTool]] = {}
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get_tools(
        self,
        server_name: str,
        fetch_fn: Callable[[], Any],
        force_refresh: bool = False,
    ) -> list[Tool]:
        """Get tools for a server with caching.

        Args:
            server_name: Server identifier
            fetch_fn: Async function to fetch tools from server
            force_refresh: Force cache refresh

        Returns:
            List of available tools

        Raises:
            MCPError: If tool fetching fails
        """
        async with self._lock:
            # Check cache first
            if not force_refresh and server_name in self._cache:
                cached_tools = self._cache[server_name]
                valid_tools = []

                for tool_name, cached_tool in cached_tools.items():
                    if not cached_tool.is_expired():
                        valid_tools.append(cached_tool.tool)
                    else:
                        logger.debug(f"Cache expired for tool: {tool_name}")

                if valid_tools:
                    logger.debug(f"Using {len(valid_tools)} cached tools for {server_name}")
                    return valid_tools

            # Fetch fresh tools
            try:
                logger.debug(f"Fetching tools from server: {server_name}")
                result = await fetch_fn()
                tools = result.tools if hasattr(result, "tools") else []

                # Update cache
                self._cache[server_name] = {
                    tool.name: CachedTool(
                        tool=tool, cached_at=time.time(), ttl=self._default_ttl
                    )
                    for tool in tools
                }

                logger.info(f"Cached {len(tools)} tools for {server_name}")
                return tools

            except Exception as e:
                logger.error(f"Failed to fetch tools from {server_name}: {e}")
                raise MCPError(f"Tool discovery failed: {e}") from e

    async def invalidate(self, server_name: Optional[str] = None) -> None:
        """Invalidate tool cache.

        Args:
            server_name: Server to invalidate, or None for all servers
        """
        async with self._lock:
            if server_name:
                self._cache.pop(server_name, None)
                logger.debug(f"Invalidated tool cache for {server_name}")
            else:
                self._cache.clear()
                logger.debug("Invalidated all tool caches")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_tools = sum(len(tools) for tools in self._cache.values())
        expired_tools = sum(
            sum(1 for tool in tools.values() if tool.is_expired())
            for tools in self._cache.values()
        )

        return {
            "servers_cached": len(self._cache),
            "total_tools": total_tools,
            "expired_tools": expired_tools,
            "valid_tools": total_tools - expired_tools,
        }


class MCPClientManager:
    """Production-ready MCP client manager.

    Manages multiple MCP server connections with:
    - Automatic connection pooling
    - Intelligent reconnection with exponential backoff
    - Tool discovery and caching
    - Health monitoring
    - Comprehensive error handling
    - Detailed statistics and metrics
    """

    def __init__(self, tool_cache_ttl: float = 3600.0) -> None:
        """Initialize MCP client manager.

        Args:
            tool_cache_ttl: Tool cache TTL in seconds (default: 1 hour)
        """
        self._connections: dict[str, ServerConnection] = {}
        self._tool_registry = MCPToolRegistry(default_ttl=tool_cache_ttl)
        self._lock = asyncio.Lock()
        self._health_check_tasks: dict[str, asyncio.Task[None]] = {}
        self._shutdown_event = asyncio.Event()

    async def connect_server(
        self,
        name: str,
        config: MCPServerConfig,
    ) -> bool:
        """Connect to an MCP server.

        Args:
            name: Unique server identifier
            config: Server configuration

        Returns:
            True if connection successful

        Raises:
            MCPConnectionError: If connection fails after all retries
        """
        async with self._lock:
            # Check if already connected
            if name in self._connections:
                conn = self._connections[name]
                if conn.status == MCPServerStatus.CONNECTED:
                    logger.warning(f"Server {name} already connected")
                    return True

            # Attempt connection
            conn = await self._establish_connection(config)
            if not conn:
                raise MCPConnectionError(f"Failed to connect to server: {name}")

            self._connections[name] = conn
            logger.info(f"Successfully connected to MCP server: {name}")

            # Start health check task
            if config.health_check_interval > 0:
                task = asyncio.create_task(self._health_check_loop(name))
                self._health_check_tasks[name] = task

            return True

    async def _establish_connection(
        self, config: MCPServerConfig
    ) -> Optional[ServerConnection]:
        """Establish connection to MCP server.

        Args:
            config: Server configuration

        Returns:
            ServerConnection if successful, None otherwise
        """
        if config.transport == MCPTransportType.STDIO:
            return await self._connect_stdio(config)
        elif config.transport == MCPTransportType.HTTP:
            return await self._connect_http(config)
        else:
            logger.error(f"Unsupported transport type: {config.transport}")
            return None

    async def _connect_stdio(self, config: MCPServerConfig) -> Optional[ServerConnection]:
        """Connect to MCP server via stdio transport.

        Args:
            config: Server configuration

        Returns:
            ServerConnection if successful, None otherwise
        """
        if not config.command:
            logger.error("No command specified for stdio transport")
            return None

        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env=config.env,
            )

            # Create stdio client with timeout
            async with asyncio.timeout(config.connection_timeout):
                read, write = await asyncio.wait_for(
                    stdio_client(server_params), timeout=config.connection_timeout
                )

                session = ClientSession(read, write)
                await session.initialize()

                logger.info(
                    f"Stdio connection established: {config.command} {config.args or []}"
                )

                return ServerConnection(
                    config=config,
                    session=session,
                    status=MCPServerStatus.CONNECTED,
                )

        except TimeoutError:
            logger.error(f"Connection timeout for stdio server: {config.name}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect via stdio: {e}", exc_info=True)
            return None

    async def _connect_http(self, config: MCPServerConfig) -> Optional[ServerConnection]:
        """Connect to MCP server via HTTP/SSE transport.

        Args:
            config: Server configuration

        Returns:
            ServerConnection if successful, None otherwise

        Note:
            HTTP/SSE transport implementation depends on mcp package updates.
            This is a placeholder for future implementation.
        """
        logger.warning(
            f"HTTP/SSE transport not yet fully implemented for server: {config.name}"
        )
        # TODO: Implement HTTP/SSE client when available in mcp package
        return None

    async def disconnect_server(self, name: str) -> bool:
        """Disconnect from an MCP server.

        Args:
            name: Server identifier

        Returns:
            True if disconnection successful
        """
        async with self._lock:
            if name not in self._connections:
                logger.warning(f"Server {name} not connected")
                return False

            conn = self._connections[name]

            # Cancel health check task
            if name in self._health_check_tasks:
                self._health_check_tasks[name].cancel()
                del self._health_check_tasks[name]

            # Close session
            try:
                # MCP sessions don't have an explicit close method yet
                # Mark as disconnected
                conn.status = MCPServerStatus.DISCONNECTED
                del self._connections[name]
                logger.info(f"Disconnected from server: {name}")
                return True

            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
                return False

    async def list_tools(
        self, server: str, force_refresh: bool = False
    ) -> list[Tool]:
        """Get available tools from a server.

        Args:
            server: Server identifier
            force_refresh: Force cache refresh

        Returns:
            List of available tools

        Raises:
            MCPConnectionError: If server not connected
            MCPError: If tool listing fails
        """
        conn = self._get_connection(server)

        async def fetch_tools() -> ListToolsResult:
            start_time = time.time()
            try:
                result = await conn.session.list_tools()
                self._update_stats(conn, success=True, latency=time.time() - start_time)
                return result
            except Exception as e:
                self._update_stats(conn, success=False, latency=time.time() - start_time)
                raise

        return await self._tool_registry.get_tools(server, fetch_tools, force_refresh)

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: Optional[dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Any:
        """Execute a tool on an MCP server with retry logic.

        Args:
            server: Server identifier
            tool: Tool name
            args: Tool arguments
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (exponential backoff)

        Returns:
            Tool execution result

        Raises:
            MCPConnectionError: If server not connected
            MCPToolExecutionError: If tool execution fails after all retries
        """
        conn = self._get_connection(server)
        args = args or {}

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    conn.session.call_tool(tool, arguments=args),
                    timeout=conn.config.request_timeout,
                )

                self._update_stats(conn, success=True, latency=time.time() - start_time)

                # Extract content from result
                if result.content:
                    # Return first text content if available
                    for content in result.content:
                        if isinstance(content, TextContent):
                            return content.text
                    return result.content

                return result

            except TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Tool call timeout (attempt {attempt + 1}/{max_retries}): {tool}"
                )
                await asyncio.sleep(retry_delay * (2**attempt))

            except Exception as e:
                last_error = e
                self._update_stats(conn, success=False, latency=time.time() - start_time)
                logger.error(
                    f"Tool execution error (attempt {attempt + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(retry_delay * (2**attempt))

        # All retries exhausted
        error_msg = f"Tool execution failed after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise MCPToolExecutionError(error_msg)

    async def list_resources(self, server: str) -> list[Any]:
        """Get available resources from a server.

        Args:
            server: Server identifier

        Returns:
            List of available resources

        Raises:
            MCPConnectionError: If server not connected
            MCPResourceError: If resource listing fails
        """
        conn = self._get_connection(server)

        try:
            start_time = time.time()
            result = await conn.session.list_resources()
            self._update_stats(conn, success=True, latency=time.time() - start_time)
            return result.resources if hasattr(result, "resources") else []

        except Exception as e:
            self._update_stats(conn, success=False, latency=time.time() - start_time)
            logger.error(f"Failed to list resources from {server}: {e}")
            raise MCPResourceError(f"Resource listing failed: {e}") from e

    async def read_resource(self, server: str, uri: str) -> Any:
        """Read a resource from a server.

        Args:
            server: Server identifier
            uri: Resource URI

        Returns:
            Resource content

        Raises:
            MCPConnectionError: If server not connected
            MCPResourceError: If resource reading fails
        """
        conn = self._get_connection(server)

        try:
            start_time = time.time()
            result = await conn.session.read_resource(uri)
            self._update_stats(conn, success=True, latency=time.time() - start_time)

            # Extract content from result
            if result.contents:
                for content in result.contents:
                    if isinstance(content, TextContent):
                        return content.text
                return result.contents

            return result

        except Exception as e:
            self._update_stats(conn, success=False, latency=time.time() - start_time)
            logger.error(f"Failed to read resource {uri} from {server}: {e}")
            raise MCPResourceError(f"Resource reading failed: {e}") from e

    async def health_check(self, server: str) -> bool:
        """Check server health.

        Args:
            server: Server identifier

        Returns:
            True if server is healthy
        """
        try:
            conn = self._get_connection(server)
            conn.last_health_check = time.time()

            # Simple health check: try to list tools
            await conn.session.list_tools()
            conn.status = MCPServerStatus.CONNECTED
            return True

        except Exception as e:
            logger.warning(f"Health check failed for {server}: {e}")
            if server in self._connections:
                self._connections[server].status = MCPServerStatus.ERROR
            return False

    async def _health_check_loop(self, server: str) -> None:
        """Background health check loop for a server.

        Args:
            server: Server identifier
        """
        if server not in self._connections:
            return

        config = self._connections[server].config

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(config.health_check_interval)

                if server not in self._connections:
                    break

                is_healthy = await self.health_check(server)

                if not is_healthy and config.auto_reconnect:
                    logger.info(f"Attempting to reconnect to {server}")
                    await self._attempt_reconnect(server)

            except asyncio.CancelledError:
                logger.debug(f"Health check loop cancelled for {server}")
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {server}: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _attempt_reconnect(self, server: str) -> bool:
        """Attempt to reconnect to a server.

        Args:
            server: Server identifier

        Returns:
            True if reconnection successful
        """
        if server not in self._connections:
            return False

        conn = self._connections[server]
        config = conn.config

        if conn.reconnect_attempts >= config.max_reconnect_attempts:
            logger.error(
                f"Max reconnection attempts reached for {server}, giving up"
            )
            conn.status = MCPServerStatus.ERROR
            return False

        conn.status = MCPServerStatus.RECONNECTING
        conn.reconnect_attempts += 1

        delay = config.reconnect_delay * (2 ** (conn.reconnect_attempts - 1))
        logger.info(
            f"Reconnecting to {server} (attempt {conn.reconnect_attempts}/"
            f"{config.max_reconnect_attempts}) in {delay:.1f}s"
        )
        await asyncio.sleep(delay)

        new_conn = await self._establish_connection(config)
        if new_conn:
            # Replace connection
            self._connections[server] = new_conn
            logger.info(f"Successfully reconnected to {server}")
            return True

        logger.warning(f"Reconnection attempt failed for {server}")
        return False

    def _get_connection(self, server: str) -> ServerConnection:
        """Get connection for a server.

        Args:
            server: Server identifier

        Returns:
            ServerConnection

        Raises:
            MCPConnectionError: If server not connected
        """
        if server not in self._connections:
            raise MCPConnectionError(f"Server not connected: {server}")

        conn = self._connections[server]
        if conn.status != MCPServerStatus.CONNECTED:
            raise MCPConnectionError(
                f"Server {server} not in connected state: {conn.status}"
            )

        return conn

    def _update_stats(
        self, conn: ServerConnection, success: bool, latency: float
    ) -> None:
        """Update connection statistics.

        Args:
            conn: Server connection
            success: Whether request was successful
            latency: Request latency in seconds
        """
        stats = conn.stats
        stats["total_requests"] += 1

        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1

        stats["total_latency"] += latency
        stats["min_latency"] = min(stats["min_latency"], latency)
        stats["max_latency"] = max(stats["max_latency"], latency)

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics for all servers.

        Returns:
            Dictionary mapping server names to their statistics
        """
        stats: dict[str, Any] = {
            "servers": {},
            "tool_cache": self._tool_registry.get_cache_stats(),
        }

        for name, conn in self._connections.items():
            server_stats = conn.stats.copy()
            server_stats["status"] = conn.status.value
            server_stats["uptime"] = time.time() - conn.connected_at
            server_stats["last_health_check"] = (
                datetime.fromtimestamp(conn.last_health_check).isoformat()
            )

            # Calculate average latency
            if server_stats["total_requests"] > 0:
                server_stats["avg_latency"] = (
                    server_stats["total_latency"] / server_stats["total_requests"]
                )
            else:
                server_stats["avg_latency"] = 0.0

            # Calculate success rate
            if server_stats["total_requests"] > 0:
                server_stats["success_rate"] = (
                    server_stats["successful_requests"] / server_stats["total_requests"]
                )
            else:
                server_stats["success_rate"] = 1.0

            stats["servers"][name] = server_stats

        return stats

    async def shutdown(self) -> None:
        """Shutdown all connections and cleanup resources."""
        logger.info("Shutting down MCP client manager")
        self._shutdown_event.set()

        # Cancel all health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()

        # Wait for health check tasks to complete
        if self._health_check_tasks:
            await asyncio.gather(*self._health_check_tasks.values(), return_exceptions=True)

        # Disconnect all servers
        server_names = list(self._connections.keys())
        for name in server_names:
            await self.disconnect_server(name)

        # Clear caches
        await self._tool_registry.invalidate()

        logger.info("MCP client manager shutdown complete")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MCPClientManager(connections={len(self._connections)}, "
            f"active_health_checks={len(self._health_check_tasks)})"
        )
