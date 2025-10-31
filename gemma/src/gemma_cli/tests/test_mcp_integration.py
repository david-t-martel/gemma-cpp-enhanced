"""Tests for MCP integration in Gemma CLI."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemma_cli.mcp.client import (
    MCPClientManager,
    MCPConnectionError,
    MCPServerConfig,
    MCPServerStatus,
    MCPTransportType,
)
from gemma_cli.mcp.config_loader import MCPConfigLoader, validate_mcp_config


@pytest.fixture
def sample_config_toml(tmp_path: Path) -> Path:
    """Create a sample MCP servers configuration file."""
    config_content = """
[test-server]
enabled = true
transport = "stdio"
command = "test-mcp-server"
args = ["--config", "test.toml"]
auto_reconnect = true
max_reconnect_attempts = 3
connection_timeout = 10.0
request_timeout = 30.0

[disabled-server]
enabled = false
transport = "stdio"
command = "disabled-server"
"""
    config_path = tmp_path / "mcp_servers.toml"
    config_path.write_text(config_content)
    return config_path


class TestMCPConfigLoader:
    """Test suite for MCPConfigLoader."""

    def test_load_servers_success(self, sample_config_toml: Path) -> None:
        """Test successful loading of server configurations."""
        loader = MCPConfigLoader(sample_config_toml)
        servers = loader.load_servers()

        assert "test-server" in servers
        assert "disabled-server" not in servers  # Disabled servers excluded

        test_server = servers["test-server"]
        assert test_server.name == "test-server"
        assert test_server.transport == MCPTransportType.STDIO
        assert test_server.command == "test-mcp-server"
        assert test_server.enabled is True

    def test_load_server_by_name(self, sample_config_toml: Path) -> None:
        """Test loading a specific server by name."""
        loader = MCPConfigLoader(sample_config_toml)
        server = loader.load_server("test-server")

        assert server is not None
        assert server.name == "test-server"

    def test_get_enabled_servers(self, sample_config_toml: Path) -> None:
        """Test getting list of enabled server names."""
        loader = MCPConfigLoader(sample_config_toml)
        enabled = loader.get_enabled_servers()

        assert "test-server" in enabled
        assert "disabled-server" not in enabled

    def test_validate_config_success(self, sample_config_toml: Path) -> None:
        """Test validation of valid configuration."""
        loader = MCPConfigLoader(sample_config_toml)
        is_valid, errors = loader.validate_config()

        assert is_valid
        assert len(errors) == 0

    def test_validate_config_missing_file(self, tmp_path: Path) -> None:
        """Test validation with missing configuration file."""
        loader = MCPConfigLoader(tmp_path / "nonexistent.toml")
        is_valid, errors = loader.validate_config()

        assert not is_valid
        assert "Configuration file not found" in errors[0]


class TestMCPClientManager:
    """Test suite for MCPClientManager."""

    @pytest.fixture
    def manager(self) -> MCPClientManager:
        """Create MCPClientManager instance."""
        return MCPClientManager(tool_cache_ttl=3600.0)

    @pytest.fixture
    def server_config(self) -> MCPServerConfig:
        """Create test server configuration."""
        return MCPServerConfig(
            name="test-server",
            transport=MCPTransportType.STDIO,
            command="test-command",
            args=["--test"],
            enabled=True,
            auto_reconnect=False,
        )

    @pytest.mark.asyncio
    async def test_connect_server_success(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test successful server connection."""
        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=MagicMock(),
                status=MCPServerStatus.CONNECTED,
            )

            result = await manager.connect_server("test-server", server_config)

            assert result is True
            assert "test-server" in manager._connections

    @pytest.mark.asyncio
    async def test_connect_server_already_connected(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test connecting to already connected server."""
        # First connection
        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=MagicMock(),
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        # Second connection (should handle gracefully)
        result = await manager.connect_server("test-server", server_config)
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_server(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test server disconnection."""
        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=MagicMock(),
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        result = await manager.disconnect_server("test-server")
        assert result is True
        assert "test-server" not in manager._connections

    @pytest.mark.asyncio
    async def test_list_tools(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test listing tools from a server."""
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(
            return_value=MagicMock(
                tools=[
                    MagicMock(name="tool1", description="Test tool 1"),
                    MagicMock(name="tool2", description="Test tool 2"),
                ]
            )
        )

        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=mock_session,
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        tools = await manager.list_tools("test-server")
        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    @pytest.mark.asyncio
    async def test_call_tool_success(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test successful tool execution."""
        from mcp.types import TextContent

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=MagicMock(
                content=[TextContent(type="text", text="Tool result")]
            )
        )

        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=mock_session,
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        result = await manager.call_tool(
            server="test-server",
            tool="test-tool",
            args={"key": "value"},
        )

        assert result == "Tool result"

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test server health check."""
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=mock_session,
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        is_healthy = await manager.health_check("test-server")
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test getting connection statistics."""
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=mock_session,
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        stats = manager.get_stats()

        assert "servers" in stats
        assert "tool_cache" in stats
        assert "test-server" in stats["servers"]

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        manager: MCPClientManager,
        server_config: MCPServerConfig,
    ) -> None:
        """Test manager shutdown."""
        with patch.object(
            manager, "_establish_connection", return_value=MagicMock()
        ) as mock_connect:
            mock_connect.return_value = MagicMock(
                config=server_config,
                session=MagicMock(),
                status=MCPServerStatus.CONNECTED,
            )
            await manager.connect_server("test-server", server_config)

        await manager.shutdown()

        # Verify all connections are closed
        assert len(manager._connections) == 0


class TestMCPIntegrationCLI:
    """Test CLI integration with MCP."""

    def test_mcp_commands_registered(self) -> None:
        """Test that MCP commands are registered with CLI."""
        from gemma_cli.cli import cli

        # Check that mcp command group is registered
        assert "mcp" in [cmd.name for cmd in cli.commands.values()]

    def test_chat_command_has_enable_mcp_flag(self) -> None:
        """Test that chat command has --enable-mcp flag."""
        from gemma_cli.cli import chat

        # Check that enable_mcp parameter exists
        params = {p.name for p in chat.params}
        assert "enable_mcp" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
