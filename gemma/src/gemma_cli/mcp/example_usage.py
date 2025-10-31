"""Example usage of MCP client manager.

This example demonstrates:
- Loading server configurations
- Connecting to multiple servers
- Tool discovery and execution
- Resource operations
- Health monitoring
- Error handling
- Statistics collection
"""

import asyncio
import logging
from pathlib import Path

from gemma_cli.mcp.client import MCPClientManager, MCPConnectionError, MCPToolExecutionError
from gemma_cli.mcp.config_loader import load_mcp_servers, validate_mcp_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def example_basic_usage() -> None:
    """Basic usage example: connect and list tools."""
    # Create client manager
    manager = MCPClientManager(tool_cache_ttl=3600.0)

    # Load server configurations
    try:
        servers = load_mcp_servers(Path("config/mcp_servers.toml"))
        logger.info(f"Loaded {len(servers)} server configurations")
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return

    # Connect to servers
    for name, config in servers.items():
        try:
            await manager.connect_server(name, config)
            logger.info(f"Connected to server: {name}")
        except MCPConnectionError as e:
            logger.error(f"Failed to connect to {name}: {e}")

    # List tools from each server
    for name in manager._connections.keys():
        try:
            tools = await manager.list_tools(name)
            logger.info(f"Server '{name}' has {len(tools)} tools:")
            for tool in tools:
                logger.info(f"  - {tool.name}: {tool.description}")
        except Exception as e:
            logger.error(f"Failed to list tools from {name}: {e}")

    # Cleanup
    await manager.shutdown()


async def example_tool_execution() -> None:
    """Example: Execute tools with error handling."""
    manager = MCPClientManager()

    try:
        # Load and connect to a specific server
        servers = load_mcp_servers()
        if "rag-redis" in servers:
            await manager.connect_server("rag-redis", servers["rag-redis"])

            # List available tools
            tools = await manager.list_tools("rag-redis")
            logger.info(f"Available tools: {[tool.name for tool in tools]}")

            # Execute a tool with retry logic
            result = await manager.call_tool(
                server="rag-redis",
                tool="store_memory",
                args={
                    "content": "Test memory entry",
                    "memory_type": "working",
                    "importance": 0.8,
                },
                max_retries=3,
                retry_delay=1.0,
            )
            logger.info(f"Tool execution result: {result}")

        else:
            logger.warning("rag-redis server not configured")

    except MCPConnectionError as e:
        logger.error(f"Connection error: {e}")
    except MCPToolExecutionError as e:
        logger.error(f"Tool execution error: {e}")
    finally:
        await manager.shutdown()


async def example_resource_operations() -> None:
    """Example: Resource listing and reading."""
    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        # Connect to filesystem server if available
        if "filesystem" in servers:
            await manager.connect_server("filesystem", servers["filesystem"])

            # List available resources
            resources = await manager.list_resources("filesystem")
            logger.info(f"Available resources: {len(resources)}")

            # Read a specific resource
            if resources:
                resource_uri = resources[0].uri if hasattr(resources[0], "uri") else None
                if resource_uri:
                    content = await manager.read_resource("filesystem", resource_uri)
                    logger.info(f"Resource content: {content[:100]}...")

    except Exception as e:
        logger.error(f"Resource operation error: {e}")
    finally:
        await manager.shutdown()


async def example_health_monitoring() -> None:
    """Example: Health monitoring and statistics."""
    manager = MCPClientManager()

    try:
        # Load and connect to servers
        servers = load_mcp_servers()
        for name, config in list(servers.items())[:2]:  # Connect to first 2 servers
            try:
                await manager.connect_server(name, config)
            except MCPConnectionError as e:
                logger.warning(f"Skipping {name}: {e}")

        # Let health checks run for a bit
        await asyncio.sleep(5)

        # Perform manual health checks
        for name in manager._connections.keys():
            is_healthy = await manager.health_check(name)
            logger.info(f"Server '{name}' health: {'✓' if is_healthy else '✗'}")

        # Get statistics
        stats = manager.get_stats()
        logger.info("Connection Statistics:")
        for name, server_stats in stats["servers"].items():
            logger.info(f"  {name}:")
            logger.info(f"    Status: {server_stats['status']}")
            logger.info(f"    Uptime: {server_stats['uptime']:.1f}s")
            logger.info(f"    Total Requests: {server_stats['total_requests']}")
            logger.info(f"    Success Rate: {server_stats['success_rate']:.2%}")
            logger.info(f"    Avg Latency: {server_stats['avg_latency']:.3f}s")

        logger.info(f"Tool Cache: {stats['tool_cache']}")

    finally:
        await manager.shutdown()


async def example_error_handling() -> None:
    """Example: Comprehensive error handling."""
    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        # Attempt to connect to non-existent server
        try:
            from gemma_cli.mcp.client import MCPServerConfig, MCPTransportType

            bad_config = MCPServerConfig(
                name="bad-server",
                transport=MCPTransportType.STDIO,
                command="nonexistent-command",
                connection_timeout=5.0,
            )
            await manager.connect_server("bad-server", bad_config)
        except MCPConnectionError as e:
            logger.info(f"Expected error caught: {e}")

        # Connect to a valid server
        if "rag-redis" in servers:
            await manager.connect_server("rag-redis", servers["rag-redis"])

            # Try to execute non-existent tool
            try:
                await manager.call_tool(
                    server="rag-redis",
                    tool="nonexistent_tool",
                    args={},
                    max_retries=2,
                )
            except MCPToolExecutionError as e:
                logger.info(f"Expected error caught: {e}")

            # Try to call tool on disconnected server
            await manager.disconnect_server("rag-redis")
            try:
                await manager.call_tool("rag-redis", "some_tool", {})
            except MCPConnectionError as e:
                logger.info(f"Expected error caught: {e}")

    finally:
        await manager.shutdown()


async def example_configuration_validation() -> None:
    """Example: Configuration validation."""
    logger.info("Validating MCP configuration...")

    is_valid, errors = validate_mcp_config(Path("config/mcp_servers.toml"))

    if is_valid:
        logger.info("✓ Configuration is valid")
    else:
        logger.error("✗ Configuration has errors:")
        for error in errors:
            logger.error(f"  - {error}")


async def example_concurrent_operations() -> None:
    """Example: Concurrent tool execution on multiple servers."""
    manager = MCPClientManager()

    try:
        # Load and connect to all servers
        servers = load_mcp_servers()
        await asyncio.gather(
            *[
                manager.connect_server(name, config)
                for name, config in servers.items()
            ],
            return_exceptions=True,
        )

        # Execute tools concurrently
        tasks = []
        for name in manager._connections.keys():
            # Get tools for each server
            task = manager.list_tools(name, force_refresh=True)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(manager._connections.keys(), results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Error from {name}: {result}")
            else:
                logger.info(f"Server {name} returned {len(result)} tools")

    finally:
        await manager.shutdown()


async def example_advanced_features() -> None:
    """Example: Advanced features (caching, reconnection, etc.)."""
    manager = MCPClientManager(tool_cache_ttl=300.0)  # 5 minute cache

    try:
        servers = load_mcp_servers()

        if "rag-redis" in servers:
            config = servers["rag-redis"]
            await manager.connect_server("rag-redis", config)

            # First call - will cache tools
            logger.info("First tool listing (will cache):")
            tools1 = await manager.list_tools("rag-redis")
            logger.info(f"  Found {len(tools1)} tools")

            # Second call - will use cache
            logger.info("Second tool listing (from cache):")
            tools2 = await manager.list_tools("rag-redis")
            logger.info(f"  Found {len(tools2)} tools")

            # Force refresh cache
            logger.info("Third tool listing (force refresh):")
            tools3 = await manager.list_tools("rag-redis", force_refresh=True)
            logger.info(f"  Found {len(tools3)} tools")

            # Test reconnection by simulating disconnect
            logger.info("Testing auto-reconnection...")
            # Simulate server failure (in real scenario, this would be external)
            # The health check loop will attempt reconnection

            # Wait for health check
            await asyncio.sleep(config.health_check_interval + 5)

            # Check final stats
            stats = manager.get_stats()
            logger.info(f"Final statistics: {stats}")

    finally:
        await manager.shutdown()


async def main() -> None:
    """Run all examples."""
    logger.info("=" * 60)
    logger.info("MCP Client Manager Examples")
    logger.info("=" * 60)

    examples = [
        ("Configuration Validation", example_configuration_validation),
        ("Basic Usage", example_basic_usage),
        ("Tool Execution", example_tool_execution),
        ("Resource Operations", example_resource_operations),
        ("Health Monitoring", example_health_monitoring),
        ("Error Handling", example_error_handling),
        ("Concurrent Operations", example_concurrent_operations),
        ("Advanced Features", example_advanced_features),
    ]

    for title, example_func in examples:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Example: {title}")
        logger.info("=" * 60)

        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example failed: {e}", exc_info=True)

        # Brief pause between examples
        await asyncio.sleep(2)

    logger.info(f"\n{'=' * 60}")
    logger.info("All examples completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
