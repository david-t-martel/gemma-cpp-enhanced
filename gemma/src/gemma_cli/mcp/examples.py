"""Example usage of MCP client in Gemma CLI.

This module demonstrates how to use the MCP client manager to connect to
MCP servers, discover tools, and execute tool calls.
"""

import asyncio
import logging
from pathlib import Path

from gemma_cli.mcp.client import MCPClientManager
from gemma_cli.mcp.config_loader import load_mcp_servers, validate_mcp_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def example_basic_connection() -> None:
    """Example: Basic server connection and tool discovery."""
    print("\n=== Example 1: Basic Connection ===\n")

    manager = MCPClientManager()

    try:
        # Load server configurations
        servers = load_mcp_servers()

        if not servers:
            print("No MCP servers configured. Edit config/mcp_servers.toml to add servers.")
            return

        # Connect to first available server
        server_name = list(servers.keys())[0]
        server_config = servers[server_name]

        print(f"Connecting to server: {server_name}")
        await manager.connect_server(server_name, server_config)
        print(f"✓ Connected to {server_name}")

        # List available tools
        tools = await manager.list_tools(server_name)
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool.name}: {tool.description or 'No description'}")

    except Exception as e:
        logger.error(f"Error in basic connection example: {e}")

    finally:
        await manager.shutdown()


async def example_tool_execution() -> None:
    """Example: Execute an MCP tool with arguments."""
    print("\n=== Example 2: Tool Execution ===\n")

    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        # Connect to memory server (if available)
        if "memory" in servers:
            print("Connecting to memory server...")
            await manager.connect_server("memory", servers["memory"])

            # Store a memory
            print("\nStoring memory...")
            result = await manager.call_tool(
                server="memory",
                tool="store_memory",
                args={
                    "key": "test_key",
                    "value": "This is a test memory from MCP example",
                },
            )
            print(f"Store result: {result}")

            # Retrieve the memory
            print("\nRetrieving memory...")
            result = await manager.call_tool(
                server="memory",
                tool="get_memory",
                args={"key": "test_key"},
            )
            print(f"Retrieved value: {result}")

        else:
            print("Memory server not configured. Enable it in mcp_servers.toml")

    except Exception as e:
        logger.error(f"Error in tool execution example: {e}")

    finally:
        await manager.shutdown()


async def example_multiple_servers() -> None:
    """Example: Connect to multiple servers and use their tools."""
    print("\n=== Example 3: Multiple Servers ===\n")

    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        # Connect to all enabled servers
        connected_servers = []
        for name, config in servers.items():
            try:
                await manager.connect_server(name, config)
                connected_servers.append(name)
                print(f"✓ Connected to {name}")
            except Exception as e:
                print(f"✗ Failed to connect to {name}: {e}")

        # List tools from all connected servers
        print(f"\nConnected to {len(connected_servers)} servers\n")
        for server_name in connected_servers:
            tools = await manager.list_tools(server_name)
            print(f"{server_name}: {len(tools)} tools")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"  • {tool.name}")

        # Get statistics
        print("\nConnection Statistics:")
        stats = manager.get_stats()
        for server_name, server_stats in stats["servers"].items():
            print(f"\n{server_name}:")
            print(f"  Total Requests: {server_stats['total_requests']}")
            print(f"  Success Rate: {server_stats['success_rate']:.1%}")
            print(f"  Avg Latency: {server_stats['avg_latency']:.3f}s")

    except Exception as e:
        logger.error(f"Error in multiple servers example: {e}")

    finally:
        await manager.shutdown()


async def example_health_monitoring() -> None:
    """Example: Monitor server health."""
    print("\n=== Example 4: Health Monitoring ===\n")

    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        # Connect to servers
        for name, config in servers.items():
            try:
                await manager.connect_server(name, config)
                print(f"✓ Connected to {name}")
            except Exception as e:
                print(f"✗ Failed to connect to {name}: {e}")

        # Check health of all servers
        print("\nHealth Check:")
        for server_name in manager._connections.keys():
            is_healthy = await manager.health_check(server_name)
            status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
            print(f"  {server_name}: {status}")

    except Exception as e:
        logger.error(f"Error in health monitoring example: {e}")

    finally:
        await manager.shutdown()


async def example_resource_operations() -> None:
    """Example: List and read resources from a server."""
    print("\n=== Example 5: Resource Operations ===\n")

    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        # Connect to filesystem server (if available)
        if "filesystem" in servers:
            print("Connecting to filesystem server...")
            await manager.connect_server("filesystem", servers["filesystem"])

            # List available resources
            print("\nListing resources...")
            resources = await manager.list_resources("filesystem")
            print(f"Found {len(resources)} resources")

            # Note: Reading specific resources depends on server implementation
            # Example for filesystem server:
            # result = await manager.read_resource(
            #     server="filesystem",
            #     uri="file:///path/to/file.txt"
            # )

        else:
            print("Filesystem server not configured. Enable it in mcp_servers.toml")

    except Exception as e:
        logger.error(f"Error in resource operations example: {e}")

    finally:
        await manager.shutdown()


async def example_error_handling() -> None:
    """Example: Error handling with retry logic."""
    print("\n=== Example 6: Error Handling ===\n")

    manager = MCPClientManager()

    try:
        servers = load_mcp_servers()

        if not servers:
            print("No servers configured")
            return

        server_name = list(servers.keys())[0]
        await manager.connect_server(server_name, servers[server_name])

        # Attempt to call a non-existent tool (will retry)
        print("Attempting to call non-existent tool (will retry 3 times)...")
        try:
            result = await manager.call_tool(
                server=server_name,
                tool="nonexistent_tool",
                args={},
                max_retries=3,
                retry_delay=1.0,
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"Expected error after retries: {type(e).__name__}: {e}")

    except Exception as e:
        logger.error(f"Error in error handling example: {e}")

    finally:
        await manager.shutdown()


async def example_config_validation() -> None:
    """Example: Validate MCP configuration before use."""
    print("\n=== Example 7: Configuration Validation ===\n")

    # Validate configuration
    is_valid, errors = validate_mcp_config()

    if is_valid:
        print("✓ MCP configuration is valid")

        # Load and display servers
        servers = load_mcp_servers()
        print(f"\nEnabled servers ({len(servers)}):")
        for name, config in servers.items():
            print(f"  • {name} ({config.transport.value})")
    else:
        print("✗ MCP configuration has errors:")
        for error in errors:
            print(f"  - {error}")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MCP Client Manager Examples")
    print("=" * 60)

    # Run examples
    await example_config_validation()
    await example_basic_connection()
    await example_tool_execution()
    await example_multiple_servers()
    await example_health_monitoring()
    await example_resource_operations()
    await example_error_handling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
