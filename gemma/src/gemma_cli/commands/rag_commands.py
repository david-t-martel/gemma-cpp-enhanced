"""CLI command handlers for Phase 2 MCP and RAG features.

This module provides comprehensive CLI commands for:
- RAG memory system (5-tier architecture)
- MCP server management and tool execution
- Memory consolidation and cleanup
- Document ingestion and retrieval

All commands use Rich for beautiful terminal output and Click for argument parsing.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gemma_cli.config.settings import Settings, load_config
from gemma_cli.rag.memory import MemoryTier
from gemma_cli.rag.python_backend import PythonRAGBackend
from gemma_cli.rag.hybrid_rag import RecallMemoriesParams, StoreMemoryParams, IngestDocumentParams


# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Initialization and Helpers (Stateless)
# ============================================================================


def get_console() -> Console:
    """Get a Console instance for output formatting.

    Returns:
        Console: Rich Console for beautiful terminal output
    """
    return Console()


def load_settings_or_default() -> Settings:
    """Load settings from config file or return defaults.

    Returns:
        Settings: Configuration object

    Raises:
        click.Abort: If user intervention needed for critical errors
    """
    console = get_console()
    try:
        return load_config()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading config: {e}")
        logger.warning("Using default settings")
        return Settings()


async def create_rag_backend(settings: Settings, console: Console) -> PythonRAGBackend:
    """Create and initialize RAG backend instance.

    Args:
        settings: Configuration settings
        console: Console for output

    Returns:
        PythonRAGBackend: Initialized RAG backend

    Raises:
        click.Abort: If initialization fails
    """
    backend = PythonRAGBackend(
        redis_host=settings.redis.host,
        redis_port=settings.redis.port,
        redis_db=settings.redis.db,
        pool_size=settings.redis.pool_size,
    )
    if not await backend.initialize():
        logger.error("Failed to initialize RAG backend")
        logger.error(
            f"Make sure Redis is running on "
            f"{settings.redis.host}:{settings.redis.port}"
        )
        raise click.Abort()
    return backend


def format_memory_entry(entry: Any) -> Table:
    """Format a memory entry as a Rich table."""
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("ID", entry.id[:12] + "...")
    table.add_row("Type", entry.memory_type)
    table.add_row("Importance", f"{entry.importance:.2f}")
    table.add_row("Access Count", str(entry.access_count))
    table.add_row("Created", entry.created_at.strftime("%Y-%m-%d %H:%M:%S"))

    # Show similarity score if available
    if hasattr(entry, "similarity_score"):
        table.add_row("Similarity", f"{entry.similarity_score:.3f}")

    # Show content preview
    content_preview = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
    table.add_row("Content", content_preview)

    # Show tags if present
    if entry.tags:
        table.add_row("Tags", ", ".join(entry.tags))

    return table


def format_memory_stats(stats: dict[str, Any]) -> Table:
    """Format memory statistics as a Rich table."""
    table = Table(title="Memory System Statistics", show_header=True)
    table.add_column("Tier", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Max Size", justify="right")

    tier_config = PythonRAGBackend.TIER_CONFIG

    for tier in [
        MemoryTier.WORKING,
        MemoryTier.SHORT_TERM,
        MemoryTier.LONG_TERM,
        MemoryTier.EPISODIC,
        MemoryTier.SEMANTIC,
    ]:
        count = stats.get(tier, 0)
        max_size = tier_config[tier]["max_size"]
        table.add_row(tier, str(count), str(max_size))

    table.add_row("", "", "", style="dim")
    table.add_row("TOTAL", str(stats.get("total", 0)), "", style="bold")

    return table


# ============================================================================
# Memory Commands Group
# ============================================================================


@click.group()
def memory_commands():
    """Memory and RAG system commands.

    Manage the 5-tier memory architecture:
    - WORKING: Short-term context (15 min)
    - SHORT_TERM: Recent interactions (1 hour)
    - LONG_TERM: Consolidated knowledge (30 days)
    - EPISODIC: Event sequences (7 days)
    - SEMANTIC: Permanent concepts (no expiry)
    """
    pass


@memory_commands.command("dashboard")
@click.option("--refresh", "-r", type=int, default=0, help="Auto-refresh interval in seconds")
def memory_dashboard(refresh: int):
    """Show comprehensive memory system dashboard.

    Examples:
        gemma /memory dashboard
        gemma /memory dashboard --refresh 5
    """
    console = get_console()
    settings = load_settings_or_default()

    async def _show_dashboard():
        backend = await create_rag_backend(settings, console)
        try:
            stats = await backend.get_memory_stats()

            # Display statistics table
            console.clear()
            console.print(Panel.fit("Memory System Dashboard", style="bold magenta"))
            console.print()
            console.print(format_memory_stats(stats))

            # Redis memory usage
            redis_memory = stats.get("redis_memory", 0)
            if redis_memory > 0:
                memory_mb = redis_memory / (1024 * 1024)
                console.print()
                console.print(f"[cyan]Redis Memory Usage:[/cyan] {memory_mb:.2f} MB")

            console.print()
            console.print("[dim]Press Ctrl+C to exit[/dim]")
        finally:
            # Cleanup backend connection
            if hasattr(backend, "close"):
                await backend.close()

    if refresh > 0:
        try:
            while True:
                asyncio.run(_show_dashboard())
                import time

                time.sleep(refresh)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped")
    else:
        asyncio.run(_show_dashboard())


@memory_commands.command("recall")
@click.argument("query")
@click.option(
    "--tier",
    "-t",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    help="Specific memory tier to search",
)
@click.option("--limit", "-l", type=int, default=5, help="Maximum number of results")
def recall_command(query: str, tier: Optional[str], limit: int):
    """Recall memories similar to query using semantic search.

    This command uses embedding-based similarity search to find
    relevant memories across one or all memory tiers.

    Examples:
        gemma /recall "machine learning concepts"
        gemma /recall "error handling" --tier=long_term --limit=10
        gemma /recall "yesterday's discussion" --tier=episodic
    """
    async def _recall():
        backend = await get_rag_backend()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Searching {tier or 'all tiers'}...", total=None)
            recall_params = RecallMemoriesParams(query=query, memory_type=tier, limit=limit)
            results = await backend.recall_memories(params=recall_params)
            progress.stop()

        if not results:
            logger.warning(f"No memories found for query: {query}")
            return

        console.print(f"\n[green]Found {len(results)} relevant memories:[/green]\n")

        for i, entry in enumerate(results, 1):
            console.print(f"[bold cyan]Result {i}[/bold cyan]")
            console.print(format_memory_entry(entry))
            console.print()

    asyncio.run(_recall())


@memory_commands.command("store")
@click.argument("text")
@click.option(
    "--tier",
    "-t",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    default="short_term",
    help="Memory tier for storage",
)
@click.option(
    "--importance",
    "-i",
    type=float,
    default=0.5,
    help="Importance score (0.0-1.0)",
)
@click.option("--tags", "-g", multiple=True, help="Tags to attach to memory")
def store_command(text: str, tier: str, importance: float, tags: tuple[str, ...]):
    """Store text content in specified memory tier.

    Importance scores guide consolidation and retention:
    - 0.0-0.3: Low importance (may be pruned quickly)
    - 0.4-0.6: Normal importance
    - 0.7-0.9: High importance (retained longer)
    - 1.0: Critical importance (highest priority)

    Examples:
        gemma /store "Python uses duck typing" --tier=semantic --importance=0.8
        gemma /store "Meeting at 3pm tomorrow" --tier=episodic --tags=meeting --tags=reminder
        gemma /store "Debugging session findings" --tier=long_term --importance=0.7
    """
    async def _store():
        backend = await get_rag_backend()

        # Validate importance range
        if not 0.0 <= importance <= 1.0:
            logger.error("Importance must be between 0.0 and 1.0")
            raise click.Abort()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Storing memory...", total=None)
            store_params = StoreMemoryParams(
                content=text,
                memory_type=tier,
                importance=importance,
                tags=list(tags) if tags else None
            )
            entry_id = await backend.store_memory(params=store_params)
            progress.stop()

        if entry_id:
            console.print(f"[green]✓ Memory stored successfully[/green]")
            console.print(f"[dim]ID: {entry_id[:12]}...[/dim]")
            console.print(f"[dim]Tier: {tier}[/dim]")
        else:
            logger.error("Failed to store memory")

    asyncio.run(_store())


@memory_commands.command("search")
@click.argument("query")
@click.option(
    "--tier",
    "-t",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    help="Specific memory tier to search",
)
@click.option(
    "--min-importance",
    "-m",
    type=float,
    default=0.0,
    help="Minimum importance threshold",
)
def search_command(query: str, tier: Optional[str], min_importance: float):
    """Search memories by content and importance (keyword-based).

    Unlike 'recall' which uses semantic similarity, 'search' performs
    exact substring matching with importance filtering.

    Examples:
        gemma /search "error" --min-importance=0.5
        gemma /search "API" --tier=long_term
        gemma /search "critical" --min-importance=0.8 --tier=semantic
    """
    async def _search():
        backend = await get_rag_backend()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            search_params = SearchParams(query=query, memory_type=tier, min_importance=min_importance)
            results = await backend.search_memories(params=search_params)
            # TODO: [RAG Backend] Implement a Pydantic model for search_memories parameters
            # and update the backend call to use it. (This is already done, but keeping the TODO for clarity)
            progress.stop()

        if not results:
            logger.warning(f"No memories found matching: {query}")
            if min_importance > 0:
                logger.debug(f"With minimum importance: {min_importance}")
            return

        console.print(f"\n[green]Found {len(results)} matching memories:[/green]\n")

        for i, entry in enumerate(results, 1):
            console.print(f"[bold cyan]Result {i}[/bold cyan]")
            console.print(format_memory_entry(entry))
            console.print()

    asyncio.run(_search())


@memory_commands.command("ingest")
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--tier",
    "-t",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    default="long_term",
    help="Memory tier for document chunks",
)
@click.option(
    "--chunk-size",
    "-c",
    type=int,
    default=500,
    help="Chunk size in tokens/characters",
)
def ingest_command(file_path: str, tier: str, chunk_size: int):
    """Ingest a document into the memory system by chunking.

    Supported formats: .txt, .md, .html, .json
    Documents are intelligently chunked and stored with embeddings.

    Examples:
        gemma /ingest docs/readme.md
        gemma /ingest notes.txt --tier=semantic --chunk-size=1000
        gemma /ingest data.json --tier=long_term
    """
    async def _ingest():
        backend = await get_rag_backend()
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise click.Abort()

        console.print(f"[cyan]Ingesting document:[/cyan] {path.name}")
        console.print(f"[dim]Chunk size: {chunk_size} | Tier: {tier}[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing document...", total=None)
            ingest_params = IngestDocumentParams(file_path=file_path, memory_type=tier, chunk_size=chunk_size)
            chunks_stored = await backend.ingest_document(params=ingest_params)
            progress.stop()

        if chunks_stored > 0:
            console.print(f"\n[green]✓ Successfully ingested {chunks_stored} chunks[/green]")
        else:
            logger.error("Failed to ingest document")

    asyncio.run(_ingest())


@memory_commands.command("cleanup")
@click.option("--dry-run", "-d", is_flag=True, help="Show what would be deleted without deleting")
def cleanup_command(dry_run: bool):
    """Clean up expired memory entries across all tiers.

    This removes entries that have exceeded their TTL (Time To Live).
    Permanent memories (semantic tier) are never cleaned up.

    Examples:
        gemma /cleanup
        gemma /cleanup --dry-run
    """
    async def _cleanup():
        backend = await get_rag_backend()

        if dry_run:
            logger.warning("DRY RUN: No entries will be deleted")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning for expired entries...", total=None)

            if dry_run:
                # For dry run, just show stats
                stats = await backend.get_memory_stats()
                progress.stop()
                console.print("\n[cyan]Current memory statistics:[/cyan]")
                console.print(format_memory_stats(stats))
            else:
                cleaned = await backend.cleanup_expired()
                progress.stop()

                if cleaned > 0:
                    console.print(f"\n[green]✓ Cleaned up {cleaned} expired entries[/green]")
                else:
                    logger.info("No expired entries found")

    asyncio.run(_cleanup())


@memory_commands.command("consolidate")
@click.option("--force", "-f", is_flag=True, help="Force consolidation even if threshold not met")
def consolidate_command(force: bool):
    """Run memory consolidation across tiers.

    Consolidation moves important memories from lower tiers (working, short_term)
    to higher tiers (long_term, semantic) based on access patterns and importance.

    Examples:
        gemma /consolidate
        gemma /consolidate --force
    """
    async def _consolidate():
        logger.warning("Note: Full consolidation not yet implemented")
        logger.info("This will be available in Phase 2B")

        # Placeholder for future implementation
        if force:
            logger.info("Force flag noted for future implementation")

    asyncio.run(_consolidate())


# ============================================================================
# MCP Commands Group
# ============================================================================

# TODO: [MCP Integration] Implement full MCP client functionality for connecting to
# and interacting with MCP servers. This includes dynamic tool discovery,
# execution, and result handling.


@click.group()
def mcp_commands():
    """MCP server and tool commands.

    Manage Model Context Protocol servers and execute tools.
    MCP provides a standardized way to connect LLMs with external
    tools and resources.
    """
    pass


@mcp_commands.command("status")
def mcp_status():
    """Show MCP server status and connection health.

    Examples:
        gemma /mcp status
    """
    logger.warning("Note: MCP integration not yet implemented")
    logger.info("This will be available in Phase 2B")
    table = Table(title="MCP Server Status")
    table.add_column("Server", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Tools", justify="right")

    table.add_row("filesystem", "[dim]not connected[/dim]", "0")
    table.add_row("redis-cache", "[dim]not connected[/dim]", "0")
    table.add_row("sequential-thinking", "[dim]not connected[/dim]", "0")

    console.print(table)


@mcp_commands.command("list")
@click.argument("item_type", type=click.Choice(["tools", "resources", "servers"]), default="servers")
def mcp_list(item_type: str):
    """List MCP items (tools, resources, or servers).

    Examples:
        gemma /mcp list servers
        gemma /mcp list tools
        gemma /mcp list resources
    """
    logger.warning("Note: MCP integration not yet implemented")
    logger.info("This will be available in Phase 2B")


@mcp_commands.command("call")
@click.argument("server")
@click.argument("tool")
@click.argument("args", nargs=-1)
def mcp_call(server: str, tool: str, args: tuple[str, ...]):
    """Execute an MCP tool on specified server.

    Examples:
        gemma /mcp call filesystem read_file /path/to/file.txt
        gemma /mcp call redis-cache get key123
    """
    logger.warning("Note: MCP integration not yet implemented")
    logger.info("This will be available in Phase 2B")


@mcp_commands.command("connect")
@click.argument("server")
def mcp_connect(server: str):
    """Connect to an MCP server.

    Examples:
        gemma /mcp connect filesystem
        gemma /mcp connect redis-cache
    """
    console.print(f"[cyan]Connecting to MCP server: {server}[/cyan]")
    logger.warning("Note: MCP integration not yet implemented")
    logger.info("This will be available in Phase 2B")


@mcp_commands.command("disconnect")
@click.argument("server")
def mcp_disconnect(server: str):
    """Disconnect from an MCP server.

    Examples:
        gemma /mcp disconnect filesystem
        gemma /mcp disconnect redis-cache
    """
    logger.warning("Note: MCP integration not yet implemented")
    logger.info("This will be available in Phase 2B")


@mcp_commands.command("health")
@click.argument("server", required=False)
def mcp_health(server: Optional[str]):
    """Perform health check on MCP servers.

    Examples:
        gemma /mcp health           # Check all servers
        gemma /mcp health filesystem  # Check specific server
    """
    if server:
        console.print(f"[cyan]Health check for: {server}[/cyan]")
    else:
        console.print("[cyan]Health check for all MCP servers[/cyan]")

    logger.warning("Note: MCP integration not yet implemented")
    logger.info("This will be available in Phase 2B")


# ============================================================================
# Main CLI Group
# ============================================================================


@click.group()
def cli():
    """Gemma CLI - Memory and MCP commands."""
    pass


# Register command groups
cli.add_command(memory_commands, name="memory")
cli.add_command(mcp_commands, name="mcp")


if __name__ == "__main__":
    cli()
