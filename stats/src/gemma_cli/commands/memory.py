"""Memory and RAG commands (refactored for Click)."""

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gemma_cli.rag.adapter import HybridRAGManager, MemoryType, DocumentMetadata

console = Console()


@click.group(name="memory")
def memory_group() -> None:
    """Memory and RAG system management."""
    pass


@memory_group.command(name="recall")
@click.argument("query", required=True)
@click.option(
    "--tier",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    help="Memory tier to search in",
)
@click.option("--limit", type=int, default=5, help="Maximum number of results")
@click.option("--backend", type=click.Choice(["mcp", "ffi", "python"]), help="Preferred backend")
@click.pass_context
async def recall(
    ctx: click.Context,
    query: str,
    tier: Optional[str],
    limit: int,
    backend: Optional[str],
) -> None:
    """Recall memories semantically similar to query.

    Uses vector similarity search to find the most relevant memories
    across one or more memory tiers.

    \b
    Examples:
      gemma memory recall "machine learning algorithms"
      gemma memory recall "python functions" --tier long_term --limit 10
      gemma memory recall "recent chats" --tier working
    """
    from gemma_cli.rag.adapter import BackendType

    memory_type = MemoryType(tier) if tier else None
    prefer_backend = BackendType(backend) if backend else None

    # Initialize RAG manager
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager(prefer_backend=prefer_backend)
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    # Recall memories
    with console.status(f"[cyan]Searching for: {query}..."):
        memories = await rag.recall_memories(query, memory_type, limit)

    if not memories:
        console.print(f"[yellow]No memories found matching: {query}[/yellow]")
        await rag.close()
        return

    # Display results
    console.print(f"\n[green]Found {len(memories)} relevant memories:[/green]\n")

    for i, memory in enumerate(memories, 1):
        score = getattr(memory, "similarity_score", 0.0)
        importance = memory.importance

        # Create panel for each memory
        content_preview = memory.content[:300]
        if len(memory.content) > 300:
            content_preview += "..."

        panel_content = f"[white]{content_preview}[/white]\n\n"
        panel_content += f"[dim]Type: {memory.memory_type.value} | "
        panel_content += f"Score: {score:.3f} | "
        panel_content += f"Importance: {importance:.2f} | "
        panel_content += f"Created: {memory.created_at.strftime('%Y-%m-%d %H:%M')}"

        if memory.tags:
            panel_content += f"\nTags: {', '.join(memory.tags)}"
        panel_content += "[/dim]"

        console.print(Panel(
            panel_content,
            title=f"[cyan]Memory {i}[/cyan]",
            border_style="blue",
        ))

    await rag.close()


@memory_group.command(name="store")
@click.argument("content", required=True)
@click.option(
    "--tier",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    required=True,
    help="Memory tier to store in",
)
@click.option(
    "--importance",
    type=float,
    default=0.5,
    help="Importance score (0.0-1.0)",
)
@click.option("--tags", multiple=True, help="Tags for categorization (repeatable)")
@click.option("--backend", type=click.Choice(["mcp", "ffi", "python"]), help="Preferred backend")
@click.pass_context
async def store(
    ctx: click.Context,
    content: str,
    tier: str,
    importance: float,
    tags: tuple,
    backend: Optional[str],
) -> None:
    """Store content in memory tier.

    \b
    Examples:
      gemma memory store "Important fact" --tier long_term --importance 0.8
      gemma memory store "Quick note" --tier working --tags note --tags temp
    """
    from gemma_cli.rag.adapter import BackendType

    if not 0.0 <= importance <= 1.0:
        console.print("[red]Importance must be between 0.0 and 1.0[/red]")
        raise click.Abort()

    memory_type = MemoryType(tier)
    prefer_backend = BackendType(backend) if backend else None

    # Initialize RAG
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager(prefer_backend=prefer_backend)
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    # Store memory
    with console.status("[cyan]Storing memory..."):
        memory_id = await rag.store_memory(
            content,
            memory_type,
            importance,
            list(tags) if tags else None,
        )

    console.print(f"[green]✓ Memory stored successfully[/green]")
    console.print(f"  ID: [cyan]{memory_id}[/cyan]")
    console.print(f"  Tier: [cyan]{tier}[/cyan]")
    console.print(f"  Importance: [cyan]{importance:.2f}[/cyan]")
    if tags:
        console.print(f"  Tags: [cyan]{', '.join(tags)}[/cyan]")

    await rag.close()


@memory_group.command(name="stats")
@click.option(
    "--tier",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    help="Show stats for specific tier",
)
@click.option("--backend", type=click.Choice(["mcp", "ffi", "python"]), help="Preferred backend")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
async def stats(
    ctx: click.Context,
    tier: Optional[str],
    backend: Optional[str],
    output_json: bool,
) -> None:
    """Show memory usage statistics.

    Displays detailed statistics about memory usage across all tiers,
    including entry counts, TTLs, and backend performance metrics.

    \b
    Examples:
      gemma memory stats
      gemma memory stats --tier long_term
      gemma memory stats --json > stats.json
    """
    from gemma_cli.rag.adapter import BackendType

    prefer_backend = BackendType(backend) if backend else None

    # Initialize RAG
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager(prefer_backend=prefer_backend)
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    # Get stats
    with console.status("[cyan]Gathering statistics..."):
        memory_stats = await rag.get_memory_stats()
        backend_stats = rag.get_backend_stats()
        active_backend = rag.get_active_backend()

    if output_json:
        import json
        output = {
            "memory_stats": memory_stats,
            "backend_stats": {k.value: v for k, v in backend_stats.items()},
            "active_backend": active_backend.value if active_backend else None,
        }
        console.print_json(data=output)
        await rag.close()
        return

    # Display memory statistics
    console.print(f"\n[bold cyan]Memory Statistics[/bold cyan]")
    console.print(f"Active Backend: [green]{active_backend.value if active_backend else 'None'}[/green]\n")

    # Tier statistics table
    tier_table = Table(title="Memory Tiers", show_header=True)
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Entries", justify="right", style="green")
    tier_table.add_column("Max Size", justify="right", style="white")
    tier_table.add_column("TTL", justify="right", style="yellow")
    tier_table.add_column("Usage %", justify="right", style="magenta")

    tier_configs = {
        "working": {"ttl": "15m", "max_size": 15},
        "short_term": {"ttl": "1h", "max_size": 100},
        "long_term": {"ttl": "30d", "max_size": 10000},
        "episodic": {"ttl": "7d", "max_size": 5000},
        "semantic": {"ttl": "∞", "max_size": 50000},
    }

    for tier_name, config in tier_configs.items():
        if tier and tier != tier_name:
            continue

        count = memory_stats.get(tier_name, 0)
        max_size = config["max_size"]
        ttl = config["ttl"]
        usage_pct = (count / max_size) * 100 if max_size > 0 else 0

        usage_color = "green" if usage_pct < 70 else "yellow" if usage_pct < 90 else "red"

        tier_table.add_row(
            tier_name,
            str(count),
            str(max_size),
            ttl,
            f"[{usage_color}]{usage_pct:.1f}%[/{usage_color}]",
        )

    console.print(tier_table)

    # Backend performance table
    if active_backend and active_backend in backend_stats:
        console.print(f"\n[bold cyan]Backend Performance[/bold cyan]\n")

        perf_table = Table(show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")

        stats = backend_stats[active_backend]
        perf_table.add_row("Successful Calls", str(stats["successful_calls"]))
        perf_table.add_row("Failed Calls", str(stats["failed_calls"]))
        perf_table.add_row("Avg Latency", f"{stats['avg_latency_ms']:.2f}ms")
        perf_table.add_row("Success Rate", f"{stats['success_rate'] * 100:.1f}%")
        perf_table.add_row("Init Time", f"{stats['initialization_time_ms']:.2f}ms")

        if stats["last_error"]:
            perf_table.add_row("Last Error", f"[red]{stats['last_error']}[/red]")

        console.print(perf_table)

    # Additional stats from backend
    if "total" in memory_stats:
        console.print(f"\n[cyan]Total Entries:[/cyan] [green]{memory_stats['total']}[/green]")

    await rag.close()


@memory_group.command(name="ingest")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tier",
    type=click.Choice(["short_term", "long_term", "semantic"]),
    default="long_term",
    help="Memory tier for document chunks",
)
@click.option("--importance", type=float, default=0.7, help="Importance score (0.0-1.0)")
@click.option("--chunk-size", type=int, default=512, help="Text chunk size in tokens")
@click.option("--title", type=str, help="Document title (defaults to filename)")
@click.option("--tags", multiple=True, help="Tags for document (repeatable)")
@click.pass_context
async def ingest(
    ctx: click.Context,
    file_path: Path,
    tier: str,
    importance: float,
    chunk_size: int,
    title: Optional[str],
    tags: tuple,
) -> None:
    """Ingest a document into memory system.

    Reads a text file, chunks it intelligently, and stores each chunk
    in the specified memory tier with embeddings for semantic search.

    \b
    Supported formats:
    - Plain text (.txt)
    - Markdown (.md)
    - JSON (.json)
    - Code files (.py, .js, .rs, etc.)

    \b
    Examples:
      gemma memory ingest document.txt
      gemma memory ingest notes.md --tier semantic --importance 0.9
      gemma memory ingest code.py --chunk-size 256 --tags python --tags code
    """
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise click.Abort()

    if not 0.0 <= importance <= 1.0:
        console.print("[red]Importance must be between 0.0 and 1.0[/red]")
        raise click.Abort()

    memory_type = MemoryType(tier)

    # Initialize RAG
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager()
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    # Create document metadata
    metadata = DocumentMetadata(
        title=title or file_path.name,
        source=str(file_path),
        doc_type=file_path.suffix[1:] if file_path.suffix else "text",
        tags=list(tags) if tags else [],
        importance=importance,
    )

    # Ingest document with progress
    console.print(f"\n[cyan]Ingesting document:[/cyan] {file_path.name}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing chunks...", total=None)

        doc_id = await rag.ingest_document(file_path, metadata, chunk_size)

        progress.update(task, completed=True)

    console.print(f"\n[green]✓ Document ingested successfully[/green]")
    console.print(f"  Document ID: [cyan]{doc_id}[/cyan]")
    console.print(f"  Tier: [cyan]{tier}[/cyan]")
    console.print(f"  Chunk Size: [cyan]{chunk_size}[/cyan]")

    await rag.close()


@memory_group.command(name="search")
@click.argument("query", required=True)
@click.option(
    "--tier",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    help="Memory tier to search in",
)
@click.option("--min-importance", type=float, default=0.0, help="Minimum importance threshold")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
@click.pass_context
async def search(
    ctx: click.Context,
    query: str,
    tier: Optional[str],
    min_importance: float,
    limit: int,
) -> None:
    """Search memories with similarity scoring.

    More advanced than recall - includes importance filtering and
    detailed scoring information.

    \b
    Examples:
      gemma memory search "transformer architecture"
      gemma memory search "important facts" --min-importance 0.7
    """
    memory_type = MemoryType(tier) if tier else None

    # Initialize RAG
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager()
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    # Search memories
    with console.status(f"[cyan]Searching for: {query}..."):
        results = await rag.search_memories(query, memory_type, min_importance)

    if not results:
        console.print(f"[yellow]No results found matching: {query}[/yellow]")
        await rag.close()
        return

    # Limit results
    results = results[:limit]

    # Display results
    console.print(f"\n[green]Found {len(results)} results:[/green]\n")

    for i, result in enumerate(results, 1):
        content_preview = result.content[:250]
        if len(result.content) > 250:
            content_preview += "..."

        panel_content = f"[white]{content_preview}[/white]\n\n"
        panel_content += f"[dim]Score: {result.score:.3f} | "
        panel_content += f"ID: {result.id[:12]}..."

        if result.metadata:
            meta_str = ", ".join(f"{k}: {v}" for k, v in result.metadata.items())
            panel_content += f"\nMetadata: {meta_str}"
        panel_content += "[/dim]"

        console.print(Panel(
            panel_content,
            title=f"[cyan]Result {i}[/cyan]",
            border_style="blue",
        ))

    await rag.close()


@memory_group.command(name="cleanup")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.pass_context
async def cleanup(ctx: click.Context, dry_run: bool) -> None:
    """Clean up expired memory entries.

    Removes entries that have exceeded their TTL based on memory tier
    configuration.

    \b
    Examples:
      gemma memory cleanup --dry-run
      gemma memory cleanup
    """
    # Initialize RAG
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager()
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]\n")
        # In real implementation, would get count without deleting
        console.print("[cyan]Would clean up expired entries[/cyan]")
    else:
        with console.status("[cyan]Cleaning up expired memories..."):
            cleaned = await rag.cleanup_expired()

        console.print(f"[green]✓ Cleanup complete: {cleaned} entries removed[/green]")

    await rag.close()


@memory_group.command(name="health")
@click.pass_context
async def health(ctx: click.Context) -> None:
    """Check memory system health.

    Performs comprehensive health check of the memory system including
    backend connectivity, performance metrics, and system resources.
    """
    # Initialize RAG
    with console.status("[cyan]Initializing memory system..."):
        rag = HybridRAGManager()
        if not await rag.initialize():
            console.print("[red]Failed to initialize memory system[/red]")
            raise click.Abort()

    # Run health check
    with console.status("[cyan]Running health check..."):
        health_status = await rag.health_check()

    # Display health status
    status = health_status.get("status", "unknown")
    status_color = "green" if status == "healthy" else "red" if status == "unhealthy" else "yellow"

    console.print(f"\n[bold {status_color}]Status: {status.upper()}[/bold {status_color}]\n")

    if "active_backend" in health_status:
        console.print(f"Active Backend: [cyan]{health_status['active_backend']}[/cyan]")

    if "performance" in health_status:
        perf = health_status["performance"]
        console.print(f"\n[cyan]Performance Metrics:[/cyan]")
        console.print(f"  Avg Latency: [green]{perf.get('avg_latency_ms', 0):.2f}ms[/green]")
        console.print(f"  Success Rate: [green]{perf.get('success_rate', 0) * 100:.1f}%[/green]")
        console.print(f"  Total Calls: [green]{perf.get('total_calls', 0)}[/green]")

    if "error" in health_status:
        console.print(f"\n[red]Error: {health_status['error']}[/red]")

    await rag.close()

    if status != "healthy":
        raise click.Abort()
