"""Configuration loader for MCP servers.

Utilities to load and validate MCP server configurations from TOML files.
"""

import logging
from pathlib import Path
from typing import Optional

import toml

from gemma_cli.mcp.client import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPConfigLoader:
    """Loader for MCP server configurations from TOML files."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize configuration loader.

        Args:
            config_path: Path to MCP servers configuration file
        """
        self.config_path = config_path or self._find_config_file()

    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Find MCP servers configuration file in standard locations.

        Returns:
            Path to configuration file if found, None otherwise
        """
        possible_paths = [
            Path("config/mcp_servers.toml"),
            Path.cwd() / "config" / "mcp_servers.toml",
            Path.home() / ".gemma_cli" / "mcp_servers.toml",
        ]

        for path in possible_paths:
            if path.exists():
                logger.debug(f"Found MCP config at: {path}")
                return path

        logger.warning("MCP servers configuration file not found")
        return None

    def load_servers(self) -> dict[str, MCPServerConfig]:
        """Load all server configurations.

        Returns:
            Dictionary mapping server names to their configurations

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(
                "MCP servers configuration file not found. "
                "Expected at config/mcp_servers.toml or ~/.gemma_cli/mcp_servers.toml"
            )

        try:
            with open(self.config_path, encoding="utf-8") as f:
                config_data = toml.load(f)
        except (OSError, toml.TomlDecodeError) as e:
            raise ValueError(f"Error loading MCP config: {e}") from e

        servers: dict[str, MCPServerConfig] = {}

        for name, server_config in config_data.items():
            try:
                # Add name to config
                server_config["name"] = name

                # Create server config
                config = MCPServerConfig(**server_config)

                # Only include enabled servers
                if config.enabled:
                    servers[name] = config
                    logger.info(f"Loaded MCP server config: {name} ({config.transport})")
                else:
                    logger.debug(f"Skipping disabled server: {name}")

            except Exception as e:
                logger.error(f"Failed to load server config '{name}': {e}")
                continue

        if not servers:
            logger.warning("No enabled MCP servers found in configuration")

        return servers

    def load_server(self, name: str) -> Optional[MCPServerConfig]:
        """Load a specific server configuration.

        Args:
            name: Server name

        Returns:
            Server configuration if found and enabled, None otherwise
        """
        servers = self.load_servers()
        return servers.get(name)

    def get_enabled_servers(self) -> list[str]:
        """Get list of enabled server names.

        Returns:
            List of enabled server names
        """
        servers = self.load_servers()
        return list(servers.keys())

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate MCP server configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self.config_path or not self.config_path.exists():
            return False, ["Configuration file not found"]

        errors: list[str] = []

        try:
            servers = self.load_servers()

            if not servers:
                errors.append("No enabled servers found in configuration")

            # Check for duplicate server names (case-insensitive)
            names_lower = [name.lower() for name in servers.keys()]
            if len(names_lower) != len(set(names_lower)):
                errors.append("Duplicate server names detected (case-insensitive)")

            # Validate each server configuration
            for name, config in servers.items():
                # Validate stdio transport
                if config.transport.value == "stdio":
                    if not config.command:
                        errors.append(f"Server '{name}': command required for stdio transport")

                # Validate HTTP/SSE/WebSocket transports
                elif config.transport.value in ("http", "sse", "websocket"):
                    if not config.url:
                        errors.append(
                            f"Server '{name}': url required for {config.transport} transport"
                        )

                # Validate timeout values
                if config.connection_timeout <= 0:
                    errors.append(f"Server '{name}': connection_timeout must be positive")

                if config.request_timeout <= 0:
                    errors.append(f"Server '{name}': request_timeout must be positive")

                # Validate reconnection settings
                if config.max_reconnect_attempts < 0:
                    errors.append(
                        f"Server '{name}': max_reconnect_attempts must be non-negative"
                    )

                if config.reconnect_delay <= 0:
                    errors.append(f"Server '{name}': reconnect_delay must be positive")

                # Validate health check interval
                if config.health_check_interval < 0:
                    errors.append(
                        f"Server '{name}': health_check_interval must be non-negative"
                    )

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        return len(errors) == 0, errors


def load_mcp_servers(
    config_path: Optional[Path] = None,
) -> dict[str, MCPServerConfig]:
    """Convenience function to load MCP server configurations.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Dictionary mapping server names to configurations

    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If configuration is invalid
    """
    loader = MCPConfigLoader(config_path)
    return loader.load_servers()


def validate_mcp_config(config_path: Optional[Path] = None) -> tuple[bool, list[str]]:
    """Convenience function to validate MCP configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    loader = MCPConfigLoader(config_path)
    return loader.validate_config()
