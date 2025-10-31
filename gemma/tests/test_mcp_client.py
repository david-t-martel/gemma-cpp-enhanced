"""Comprehensive tests for MCP client manager.

Tests cover:
- Connection lifecycle
- Tool discovery and caching
- Tool execution with retries
- Resource operations
- Health checks
- Error handling
- Statistics collection
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from gemma_cli.mcp.client import (
    CachedTool,
    MCPClientManager,
    MCPConnectionError,
    MCPError,
    MCPResourceError,
    MCPServerConfig,
    MCPServerStatus,
    MCPToolExecutionError,
    MCPToolRegistry,
    MCPTransportType,
    ServerConnection,
)
from gemma_cli.mcp.config_loader import MCPConfigLoader, validate_mcp_config


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_valid_stdio_config(self) -> None:
        """Test valid stdio configuration."""
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransportType.STDIO,
            command="test-command",
            args=["--arg1", "value1"],
        )

        assert config.name == "test-server"
        assert config.transport == MCPTransportType.STDIO
        assert config.command == "test-command"
        assert config.args == ["--arg1", "value1"]
        assert config.enabled is True
        assert config.auto_reconnect is True

    def test_valid_http_config(self) -> None:
        """Test valid HTTP configuration."""
        config = MCPServerConfig(
            name="http-server",
            transport=MCPTransportType.HTTP,
            url="http://localhost:8080/mcp",
        )

        assert config.name == "http-server"
        assert config.transport == MCPTransportType.HTTP
        assert config.url == "http://localhost:8080/mcp"

    def test_missing_command_stdio(self) -> None:
        """Test validation error for stdio without command."""
        with pytest.raises(ValueError, match="command is required"):
            MCPServerConfig(
                name="bad-stdio",
                transport=MCPTransportType.STDIO,
            )

    def test_missing_url_http(self) -> None:
        """Test validation error for HTTP without URL."""
        with pytest.raises(ValueError, match="url is required"):
            MCPServerConfig(
                name="bad-http",
                transport=MCPTransportType.HTTP,
            )

    def test_transport_string_conversion(self) -> None:
        """Test transport type string conversion."""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="test",
        )

        assert config.transport == MCPTransportType.STDIO


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry."""

    @pytest.fixture
    def registry(self) -> MCPToolRegistry:
        """Create tool registry instance."""
        return MCPToolRegistry(default_ttl=60.0)

    @pytest.fixture
    def mock_tools(self) -> list[Any]:
        """Create mock tools."""
        tool1 = Mock()
        tool1.name = "tool1"
        tool1.description = "Test tool 1"

        tool2 = Mock()
        tool2.name = "tool2"
        tool2.description = "Test tool 2"

        return [tool1, tool2]

    @pytest.mark.asyncio
    async def test_get_tools_first_fetch(self, registry: MCPToolRegistry, mock_tools: list[Any]) -> None:
        """Test fetching tools for the first time."""

        async def fetch_fn() -> Any:
            result = Mock()
            result.tools = mock_tools
            return result

        tools = await registry.get_tools("server1", fetch_fn)

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    @pytest.mark.asyncio
    async def test_get_tools_from_cache(self, registry: MCPToolRegistry, mock_tools: list[Any]) -> None:
        """Test retrieving tools from cache."""

        fetch_count = 0

        async def fetch_fn() -> Any:
            nonlocal fetch_count
            fetch_count += 1
            result = Mock()
            result.tools = mock_tools
            return result

        # First fetch
        tools1 = await registry.get_tools("server1", fetch_fn)
        assert fetch_count == 1

        # Second fetch - should use cache
        tools2 = await registry.get_tools("server1", fetch_fn)
        assert fetch_count == 1  # Should not have called fetch again
        assert len(tools2) == 2

    @pytest.mark.asyncio
    async def test_get_tools_force_refresh(self, registry: MCPToolRegistry, mock_tools: list[Any]) -> None:
        """Test forcing cache refresh."""

        fetch_count = 0

        async def fetch_fn() -> Any:
            nonlocal fetch_count
            fetch_count += 1
            result = Mock()
            result.tools = mock_tools
            return result

        # First fetch
        await registry.get_tools("server1", fetch_fn)
        assert fetch_count == 1

        # Force refresh
        await registry.get_tools("server1", fetch_fn, force_refresh=True)
        assert fetch_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_specific_server(self, registry: MCPToolRegistry, mock_tools: list[Any]) -> None:
        """Test invalidating cache for specific server."""

        async def fetch_fn() -> Any:
            result = Mock()
            result.tools = mock_tools
            return result

        # Cache tools for two servers
        await registry.get_tools("server1", fetch_fn)
        await registry.get_tools("server2", fetch_fn)

        # Invalidate server1
        await registry.invalidate("server1")

        # server1 should be gone, server2 should remain
        assert "server1" not in registry._cache
        assert "server2" in registry._cache

    @pytest.mark.asyncio
    async def test_invalidate_all(self, registry: MCPToolRegistry, mock_tools: list[Any]) -> None:
        """Test invalidating all caches."""

        async def fetch_fn() -> Any:
            result = Mock()
            result.tools = mock_tools
            return result

        # Cache tools for two servers
        await registry.get_tools("server1", fetch_fn)
        await registry.get_tools("server2", fetch_fn)

        # Invalidate all
        await registry.invalidate()

        # Both should be gone
        assert len(registry._cache) == 0

    def test_cache_stats(self, registry: MCPToolRegistry) -> None:
        """Test cache statistics."""
        stats = registry.get_cache_stats()

        assert "servers_cached" in stats
        assert "total_tools" in stats
        assert "expired_tools" in stats
        assert "valid_tools" in stats


class TestMCPClientManager:
    """Tests for MCPClientManager."""

    @pytest.fixture
    def manager(self) -> MCPClientManager:
        """Create client manager instance."""
        return MCPClientManager(tool_cache_ttl=60.0)

    @pytest.fixture
    def mock_config(self) -> MCPServerConfig:
        """Create mock server configuration."""
        return MCPServerConfig(
            name="test-server",
            transport=MCPTransportType.STDIO,
            command="test-command",
            args=["--test"],
            connection_timeout=5.0,
            request_timeout=10.0,
            health_check_interval=30.0,
        )

    @pytest.mark.asyncio
    async def test_connect_server_success(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test successful server connection."""
        with patch.object(
            manager, "_establish_connection", return_value=Mock(spec=ServerConnection)
        ) as mock_establish:
            result = await manager.connect_server("test-server", mock_config)

            assert result is True
            assert "test-server" in manager._connections
            mock_establish.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_connect_server_already_connected(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test connecting to already connected server."""
        # Set up existing connection
        mock_session = Mock()
        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        result = await manager.connect_server("test-server", mock_config)

        assert result is True

    @pytest.mark.asyncio
    async def test_connect_server_failure(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test connection failure."""
        with patch.object(manager, "_establish_connection", return_value=None):
            with pytest.raises(MCPConnectionError):
                await manager.connect_server("test-server", mock_config)

    @pytest.mark.asyncio
    async def test_disconnect_server(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test disconnecting from server."""
        # Set up connection
        mock_session = Mock()
        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        result = await manager.disconnect_server("test-server")

        assert result is True
        assert "test-server" not in manager._connections

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_server(self, manager: MCPClientManager) -> None:
        """Test disconnecting from non-existent server."""
        result = await manager.disconnect_server("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_tools(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test listing tools from server."""
        # Set up connection
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.tools = [Mock(name="tool1"), Mock(name="tool2")]
        mock_session.list_tools.return_value = mock_result

        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        tools = await manager.list_tools("test-server")

        assert len(tools) == 2
        mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self, manager: MCPClientManager) -> None:
        """Test listing tools from disconnected server."""
        with pytest.raises(MCPConnectionError):
            await manager.list_tools("nonexistent")

    @pytest.mark.asyncio
    async def test_call_tool_success(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test successful tool execution."""
        # Set up connection
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_text_content = Mock()
        mock_text_content.text = "Tool result"
        mock_result.content = [mock_text_content]
        mock_session.call_tool.return_value = mock_result

        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        result = await manager.call_tool("test-server", "test_tool", {"arg": "value"})

        assert result == "Tool result"
        mock_session.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_retry_on_timeout(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test tool execution retry on timeout."""
        # Set up connection
        mock_session = AsyncMock()

        # First two calls timeout, third succeeds
        mock_result = Mock()
        mock_text_content = Mock()
        mock_text_content.text = "Success"
        mock_result.content = [mock_text_content]

        mock_session.call_tool.side_effect = [
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            mock_result,
        ]

        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        result = await manager.call_tool(
            "test-server", "test_tool", {}, max_retries=3, retry_delay=0.1
        )

        assert result == "Success"
        assert mock_session.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_call_tool_all_retries_fail(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test tool execution failure after all retries."""
        # Set up connection
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = Exception("Tool error")

        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        with pytest.raises(MCPToolExecutionError):
            await manager.call_tool(
                "test-server", "test_tool", {}, max_retries=2, retry_delay=0.1
            )

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test successful health check."""
        # Set up connection
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = Mock(tools=[])

        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        result = await manager.health_check("test-server")

        assert result is True
        assert conn.status == MCPServerStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test failed health check."""
        # Set up connection
        mock_session = AsyncMock()
        mock_session.list_tools.side_effect = Exception("Health check failed")

        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        result = await manager.health_check("test-server")

        assert result is False
        assert conn.status == MCPServerStatus.ERROR

    def test_get_stats(self, manager: MCPClientManager, mock_config: MCPServerConfig) -> None:
        """Test getting connection statistics."""
        # Set up connection with stats
        mock_session = Mock()
        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        conn.stats["total_requests"] = 10
        conn.stats["successful_requests"] = 8
        conn.stats["failed_requests"] = 2
        conn.stats["total_latency"] = 5.0

        manager._connections["test-server"] = conn

        stats = manager.get_stats()

        assert "servers" in stats
        assert "tool_cache" in stats
        assert "test-server" in stats["servers"]

        server_stats = stats["servers"]["test-server"]
        assert server_stats["total_requests"] == 10
        assert server_stats["successful_requests"] == 8
        assert server_stats["success_rate"] == 0.8
        assert server_stats["avg_latency"] == 0.5

    @pytest.mark.asyncio
    async def test_shutdown(
        self, manager: MCPClientManager, mock_config: MCPServerConfig
    ) -> None:
        """Test manager shutdown."""
        # Set up connection
        mock_session = Mock()
        conn = ServerConnection(
            config=mock_config,
            session=mock_session,
            status=MCPServerStatus.CONNECTED,
        )
        manager._connections["test-server"] = conn

        await manager.shutdown()

        assert len(manager._connections) == 0
        assert manager._shutdown_event.is_set()


class TestMCPConfigLoader:
    """Tests for MCPConfigLoader."""

    @pytest.fixture
    def temp_config_file(self, tmp_path: Path) -> Path:
        """Create temporary config file."""
        config_content = """
[test-server]
enabled = true
transport = "stdio"
command = "test-command"
args = ["--test"]

[disabled-server]
enabled = false
transport = "stdio"
command = "disabled-command"
"""
        config_file = tmp_path / "mcp_servers.toml"
        config_file.write_text(config_content)
        return config_file

    def test_load_servers(self, temp_config_file: Path) -> None:
        """Test loading server configurations."""
        loader = MCPConfigLoader(temp_config_file)
        servers = loader.load_servers()

        assert len(servers) == 1  # Only enabled server
        assert "test-server" in servers
        assert "disabled-server" not in servers

        config = servers["test-server"]
        assert config.name == "test-server"
        assert config.command == "test-command"

    def test_load_specific_server(self, temp_config_file: Path) -> None:
        """Test loading specific server."""
        loader = MCPConfigLoader(temp_config_file)
        config = loader.load_server("test-server")

        assert config is not None
        assert config.name == "test-server"

    def test_get_enabled_servers(self, temp_config_file: Path) -> None:
        """Test getting enabled server names."""
        loader = MCPConfigLoader(temp_config_file)
        servers = loader.get_enabled_servers()

        assert servers == ["test-server"]

    def test_validate_config_valid(self, temp_config_file: Path) -> None:
        """Test validating valid configuration."""
        loader = MCPConfigLoader(temp_config_file)
        is_valid, errors = loader.validate_config()

        assert is_valid
        assert len(errors) == 0

    def test_validate_config_missing_file(self) -> None:
        """Test validating with missing config file."""
        loader = MCPConfigLoader(Path("nonexistent.toml"))
        is_valid, errors = loader.validate_config()

        assert not is_valid
        assert "not found" in errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
