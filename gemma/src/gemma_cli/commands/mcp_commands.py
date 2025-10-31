"""MCP command group for managing MCP servers and tools."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from gemma_cli.mcp.client import (
    MCPClientManager,
    MCPConnectionError,
    MCPError,
    MCPServerStatus,
    MCPToolExecutionError,
)
from gemma_cli.mcp.config_loader import MCPConfigLoader, validate_mcp_config

console = Console()
logger = logging.getLogger(__name__)


@click.group()
def mcp() -> None:
    """Manage MCP (Model Context Protocol) servers and tools.

    MCP enables AI assistants to interact with external tools and data sources.
    Use these commands to manage server connections and execute tools.
    """
    pass


@mcp.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to MCP servers configuration file",
)
def list(config: Optional[Path]) -> None:
    """List all configured MCP servers.

    Shows server names, transport types, enabled status, and descriptions.
    """
    try:
        loader = MCPConfigLoader(config)
        servers = loader.load_servers()

        if not servers:
            console.print("[yellow]No MCP servers configured or enabled.[/yellow]")
            console.print("\nTo add servers, edit: [cyan]config/mcp_servers.toml[/cyan]")
            return

        # Create table
        table = Table(title="Configured MCP Servers", show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Transport", style="green")
        table.add_column("Enabled", style="yellow")
        table.add_column("Command/URL", style="white")

        for name, server_config in servers.items():
            # Determine command or URL
            if server_config.transport.value == "stdio":
                cmd_info = f"{server_config.command} {' '.join(server_config.args or [])}"[:60]
            else:
                cmd_info = server_config.url or "N/A"

            enabled_str = "[green]✓[/green]" if server_config.enabled else "[red]✗[/red]"

            table.add_row(
                name,
                server_config.transport.value,
                enabled_str,
                cmd_info,
            )

        console.print(table)
        console.print(f"\n[dim]Total servers: {len(servers)}[/dim]")

    except Exception as e:
        console.print(f"[red]Error loading MCP configuration: {e}[/red]")
        logger.error(f"Failed to list MCP servers: {e}", exc_info=True)


@mcp.command()
@click.argument("server", required=True)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to MCP servers configuration file",
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Force refresh tool cache",
)
def tools(server: str, config: Optional[Path], force_refresh: bool) -> None:
    """List available tools from a specific MCP server.

    ARGUMENTS:
        SERVER: Name of the MCP server to query

    Example:
        gemma-cli mcp tools filesystem
        gemma-cli mcp tools rag-redis --force-refresh
    """
    asyncio.run(_list_tools_async(server, config, force_refresh))


async def _list_tools_async(
    server: str,
    config: Optional[Path],
    force_refresh: bool,
) -> None:
    """Async implementation for listing tools."""
    manager = MCPClientManager()

    try:
        # Load server configuration
        loader = MCPConfigLoader(config)
        server_config = loader.load_server(server)

        if not server_config:
            console.print(f"[red]Server '{server}' not found or not enabled.[/red]")
            console.print("\nAvailable servers:")
            for name in loader.get_enabled_servers():
                console.print(f"  • [cyan]{name}[/cyan]")
            return

        # Connect to server
        with console.status(f"[bold green]Connecting to {server}..."):
            await manager.connect_server(server, server_config)

        # List tools
        with console.status(f"[bold green]Fetching tools from {server}..."):
            tool_list = await manager.list_tools(server, force_refresh=force_refresh)

        if not tool_list:
            console.print(f"[yellow]No tools available from server '{server}'.[/yellow]")
            return

        # Display tools in a tree
        tree = Tree(f"[bold cyan]Tools from {server}[/bold cyan] ({len(tool_list)} available)")

        for tool in tool_list:
            tool_branch = tree.add(f"[green]{tool.name}[/green]")
            tool_branch.add(f"[dim]Description:[/dim] {tool.description or 'No description'}")

            # Display input schema if available
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                schema = tool.inputSchema
                if isinstance(schema, dict):
                    properties = schema.get("properties", {})
                    if properties:
                        params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in properties.items())
                        tool_branch.add(f"[dim]Parameters:[/dim] {params_str}")

        console.print(tree)

    except MCPConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        logger.error(f"Failed to connect to MCP server: {e}")
    except MCPError as e:
        console.print(f"[red]MCP error: {e}[/red]")
        logger.error(f"MCP operation failed: {e}")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.error(f"Unexpected error listing tools: {e}", exc_info=True)
    finally:
        await manager.shutdown()


@mcp.command()
@click.argument("server", required=True)
@click.argument("tool", required=True)
@click.argument("args", required=False)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to MCP servers configuration file",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    help="Maximum retry attempts",
)
def call(
    server: str,
    tool: str,
    args: Optional[str],
    config: Optional[Path],
    max_retries: int,
) -> None:
    """Call an MCP tool directly.

    ARGUMENTS:
        SERVER: Name of the MCP server
        TOOL: Name of the tool to call
        ARGS: JSON string with tool arguments (optional)

    Examples:
        # Simple call without arguments
        gemma-cli mcp call memory list_memories

        # Call with JSON arguments
        gemma-cli mcp call memory store_memory '{"key": "test", "value": "data"}'

        # File system operations
        gemma-cli mcp call filesystem read_file '{"path": "/path/to/file"}'
    """
    asyncio.run(_call_tool_async(server, tool, args, config, max_retries))


async def _call_tool_async(
    server: str,
    tool: str,
    args: Optional[str],
    config: Optional[Path],
    max_retries: int,
) -> None:
    """Async implementation for calling tools."""
    manager = MCPClientManager()

    try:
        # Parse arguments
        tool_args: dict[str, Any] = {}
        if args:
            try:
                tool_args = json.loads(args)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON arguments: {e}[/red]")
                console.print("[yellow]Arguments must be valid JSON, e.g., '{\"key\": \"value\"}'[/yellow]")
                return

        # Load server configuration
        loader = MCPConfigLoader(config)
        server_config = loader.load_server(server)

        if not server_config:
            console.print(f"[red]Server '{server}' not found or not enabled.[/red]")
            return

        # Connect to server
        with console.status(f"[bold green]Connecting to {server}..."):
            await manager.connect_server(server, server_config)

        # Call tool
        with console.status(f"[bold green]Executing {tool}..."):
            result = await manager.call_tool(
                server=server,
                tool=tool,
                args=tool_args,
                max_retries=max_retries,
            )

        # Display result
        console.print(Panel(
            str(result),
            title=f"[bold green]Result from {server}.{tool}[/bold green]",
            border_style="green",
        ))

    except MCPToolExecutionError as e:
        console.print(f"[red]Tool execution failed: {e}[/red]")
        logger.error(f"Tool execution error: {e}")
    except MCPConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        logger.error(f"Failed to connect to MCP server: {e}")
    except MCPError as e:
        console.print(f"[red]MCP error: {e}[/red]")
        logger.error(f"MCP operation failed: {e}")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.error(f"Unexpected error calling tool: {e}", exc_info=True)
    finally:
        await manager.shutdown()


@mcp.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to MCP servers configuration file",
)
def status(config: Optional[Path]) -> None:
    """Check status and health of all MCP servers.

    Attempts to connect to each enabled server and reports connection status,
    uptime, and basic statistics.
    """
    asyncio.run(_status_async(config))


async def _status_async(config: Optional[Path]) -> None:
    """Async implementation for status check."""
    manager = MCPClientManager()

    try:
        # Load server configurations
        loader = MCPConfigLoader(config)
        servers = loader.load_servers()

        if not servers:
            console.print("[yellow]No MCP servers configured or enabled.[/yellow]")
            return

        # Create table
        table = Table(title="MCP Server Status", show_header=True, header_style="bold magenta")
        table.add_column("Server", style="cyan", no_wrap=True)
        table.add_column("Status", style="yellow")
        table.add_column("Transport", style="green")
        table.add_column("Health", style="white")

        # Connect to each server and check health
        for name, server_config in servers.items():
            status_str = "[yellow]Connecting...[/yellow]"
            health_str = "[dim]N/A[/dim]"

            try:
                # Attempt connection
                await manager.connect_server(name, server_config)
                status_str = "[green]Connected[/green]"

                # Check health
                is_healthy = await manager.health_check(name)
                health_str = "[green]Healthy[/green]" if is_healthy else "[red]Unhealthy[/red]"

            except MCPConnectionError:
                status_str = "[red]Failed[/red]"
                health_str = "[red]Error[/red]"
            except Exception as e:
                status_str = "[red]Error[/red]"
                health_str = f"[red]{type(e).__name__}[/red]"

            table.add_row(
                name,
                status_str,
                server_config.transport.value,
                health_str,
            )

        console.print(table)

        # Print statistics if any servers connected
        stats = manager.get_stats()
        if stats["servers"]:
            console.print("\n[bold]Connection Statistics:[/bold]")
            for server_name, server_stats in stats["servers"].items():
                console.print(f"  [cyan]{server_name}[/cyan]:")
                console.print(f"    Requests: {server_stats['total_requests']}")
                console.print(f"    Success Rate: {server_stats['success_rate']:.1%}")
                console.print(f"    Avg Latency: {server_stats['avg_latency']:.3f}s")

    except Exception as e:
        console.print(f"[red]Error checking server status: {e}[/red]")
        logger.error(f"Failed to check server status: {e}", exc_info=True)
    finally:
        await manager.shutdown()


@mcp.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to MCP servers configuration file",
)
def validate(config: Optional[Path]) -> None:
    """Validate MCP servers configuration file.

    Checks configuration syntax, required fields, and timeout values.
    Reports any errors or warnings found.
    """
    try:
        is_valid, errors = validate_mcp_config(config)

        if is_valid:
            console.print("[green]✓ MCP configuration is valid[/green]")

            # Show enabled servers
            loader = MCPConfigLoader(config)
            servers = loader.load_servers()
            if servers:
                console.print(f"\n[dim]Enabled servers: {', '.join(servers.keys())}[/dim]")
        else:
            console.print("[red]✗ MCP configuration has errors:[/red]\n")
            for error in errors:
                console.print(f"  • [red]{error}[/red]")

    except FileNotFoundError:
        console.print("[red]✗ MCP configuration file not found[/red]")
        console.print("\nExpected locations:")
        console.print("  • [cyan]config/mcp_servers.toml[/cyan]")
        console.print("  • [cyan]~/.gemma_cli/mcp_servers.toml[/cyan]")
    except Exception as e:
        console.print(f"[red]Error validating configuration: {e}[/red]")
        logger.error(f"Failed to validate MCP configuration: {e}", exc_info=True)


@mcp.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to MCP servers configuration file",
)
def info(config: Optional[Path]) -> None:
    """Show detailed information about MCP integration.

    Displays configuration paths, enabled features, and usage examples.
    """
    console.print(Panel.fit(
        "[bold cyan]MCP (Model Context Protocol) Integration[/bold cyan]\n\n"
        "MCP enables Gemma CLI to interact with external tools and data sources.\n"
        "Servers can provide file operations, memory management, web search, and more.",
        border_style="cyan",
    ))

    # Configuration paths
    console.print("\n[bold]Configuration:[/bold]")
    loader = MCPConfigLoader(config)
    if loader.config_path:
        console.print(f"  Config file: [cyan]{loader.config_path}[/cyan]")
    else:
        console.print("  Config file: [yellow]Not found[/yellow]")

    # Available server types
    console.print("\n[bold]Available Server Types:[/bold]")
    server_types = [
        ("filesystem", "File read/write operations", "npx @modelcontextprotocol/server-filesystem"),
        ("memory", "Key-value memory store", "npx @modelcontextprotocol/server-memory"),
        ("github", "GitHub API integration", "npx @modelcontextprotocol/server-github"),
        ("brave-search", "Web search via Brave API", "npx @modelcontextprotocol/server-brave-search"),
        ("fetch", "HTTP/HTTPS requests", "npx @modelcontextprotocol/server-fetch"),
        ("rag-redis", "High-performance RAG", "rag-redis-server (Rust)"),
    ]

    for name, description, command in server_types:
        console.print(f"  • [green]{name}[/green]: {description}")
        console.print(f"    [dim]{command}[/dim]")

    # Usage examples
    console.print("\n[bold]Usage Examples:[/bold]")
    console.print("  [cyan]gemma-cli mcp list[/cyan]                    # List configured servers")
    console.print("  [cyan]gemma-cli mcp tools filesystem[/cyan]        # Show available tools")
    console.print("  [cyan]gemma-cli mcp status[/cyan]                  # Check server health")
    console.print("  [cyan]gemma-cli mcp call memory list_memories[/cyan] # Execute a tool")
    console.print("\n[dim]See 'gemma-cli mcp --help' for more commands[/dim]")


# Register command group
def register(cli_group: click.Group) -> None:
    """Register MCP command group with main CLI."""
    cli_group.add_command(mcp)
