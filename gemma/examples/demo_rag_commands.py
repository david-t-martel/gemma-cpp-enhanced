#!/usr/bin/env python3
"""Demo script for RAG commands.

This script demonstrates the usage of RAG memory commands
in the Gemma CLI. It showcases:
- Storing memories in different tiers
- Recalling memories with semantic search
- Searching with keyword filtering
- Ingesting documents
- Viewing memory statistics
- Cleanup operations

Run this after starting Redis on port 6380.
"""

import asyncio
import tempfile
from pathlib import Path

from gemma_cli.commands.rag_commands import (
    cleanup_command,
    ingest_command,
    recall_command,
    search_command,
    store_command,
)
from click.testing import CliRunner
from rich.console import Console

console = Console()


def demo_store_memories():
    """Demonstrate storing memories in different tiers."""
    console.print("\n[bold cyan]Demo 1: Storing Memories[/bold cyan]")
    console.print("=" * 60)

    runner = CliRunner()

    # Store a fact in semantic tier (permanent)
    console.print("\n[yellow]Storing a permanent fact...[/yellow]")
    result = runner.invoke(
        store_command,
        [
            "Python is a dynamically typed language with first-class functions",
            "--tier",
            "semantic",
            "--importance",
            "0.9",
            "--tags",
            "python",
            "--tags",
            "programming",
        ],
    )
    console.print(result.output)

    # Store a short-term note
    console.print("\n[yellow]Storing a short-term note...[/yellow]")
    result = runner.invoke(
        store_command,
        [
            "Need to review the error handling in the API module",
            "--tier",
            "short_term",
            "--importance",
            "0.6",
            "--tags",
            "todo",
        ],
    )
    console.print(result.output)

    # Store an episodic memory
    console.print("\n[yellow]Storing an episodic event...[/yellow]")
    result = runner.invoke(
        store_command,
        [
            "Team meeting discussed new RAG architecture implementation",
            "--tier",
            "episodic",
            "--importance",
            "0.7",
            "--tags",
            "meeting",
        ],
    )
    console.print(result.output)

    console.print("\n[green]✓ Stored 3 memories across different tiers[/green]")


def demo_recall_memories():
    """Demonstrate semantic memory recall."""
    console.print("\n[bold cyan]Demo 2: Recalling Memories (Semantic Search)[/bold cyan]")
    console.print("=" * 60)

    runner = CliRunner()

    # Recall Python-related memories
    console.print("\n[yellow]Recalling Python-related memories...[/yellow]")
    result = runner.invoke(recall_command, ["Python programming", "--limit", "3"])
    console.print(result.output)

    # Recall from specific tier
    console.print("\n[yellow]Recalling from episodic tier only...[/yellow]")
    result = runner.invoke(recall_command, ["meeting", "--tier", "episodic"])
    console.print(result.output)


def demo_search_memories():
    """Demonstrate keyword-based search."""
    console.print("\n[bold cyan]Demo 3: Searching Memories (Keyword-Based)[/bold cyan]")
    console.print("=" * 60)

    runner = CliRunner()

    # Search for specific keyword
    console.print("\n[yellow]Searching for 'error'...[/yellow]")
    result = runner.invoke(search_command, ["error"])
    console.print(result.output)

    # Search with importance filter
    console.print("\n[yellow]Searching high-importance entries...[/yellow]")
    result = runner.invoke(search_command, ["Python", "--min-importance", "0.8"])
    console.print(result.output)


def demo_ingest_document():
    """Demonstrate document ingestion."""
    console.print("\n[bold cyan]Demo 4: Ingesting Documents[/bold cyan]")
    console.print("=" * 60)

    runner = CliRunner()

    # Create a temporary document
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            """
# Machine Learning Concepts

## Supervised Learning
Supervised learning uses labeled data to train models. Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

## Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Reinforcement Learning
Agents learn through trial and error with rewards and penalties.
"""
        )
        temp_file = f.name

    try:
        console.print(f"\n[yellow]Ingesting document: {Path(temp_file).name}[/yellow]")
        result = runner.invoke(
            ingest_command,
            [temp_file, "--tier", "long_term", "--chunk-size", "200"],
        )
        console.print(result.output)

    finally:
        # Cleanup temp file
        Path(temp_file).unlink()


def demo_cleanup():
    """Demonstrate cleanup operations."""
    console.print("\n[bold cyan]Demo 5: Cleanup Operations[/bold cyan]")
    console.print("=" * 60)

    runner = CliRunner()

    # Dry run first
    console.print("\n[yellow]Running cleanup dry-run...[/yellow]")
    result = runner.invoke(cleanup_command, ["--dry-run"])
    console.print(result.output)

    # Actual cleanup
    console.print("\n[yellow]Running actual cleanup...[/yellow]")
    result = runner.invoke(cleanup_command)
    console.print(result.output)


def demo_memory_dashboard():
    """Demonstrate memory dashboard."""
    console.print("\n[bold cyan]Demo 6: Memory Dashboard[/bold cyan]")
    console.print("=" * 60)

    runner = CliRunner()

    # Note: Dashboard is typically run interactively, but we can show stats
    console.print("\n[yellow]Showing memory statistics...[/yellow]")
    console.print("[dim]Use 'gemma /memory dashboard' for interactive view[/dim]\n")

    # Dashboard command would be invoked here in actual CLI
    console.print("[green]Dashboard displays:[/green]")
    console.print("  • Memory tier counts")
    console.print("  • Total entries")
    console.print("  • Redis memory usage")
    console.print("  • Capacity utilization")


def main():
    """Run all demos."""
    console.print("\n[bold magenta]RAG Commands Demo[/bold magenta]")
    console.print("[dim]Make sure Redis is running on localhost:6380[/dim]")
    console.print()

    try:
        # Run demos sequentially
        demo_store_memories()
        demo_recall_memories()
        demo_search_memories()
        demo_ingest_document()
        demo_cleanup()
        demo_memory_dashboard()

        console.print("\n[bold green]✓ All demos completed successfully![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Try commands interactively: gemma /memory --help")
        console.print("  2. View dashboard: gemma /memory dashboard")
        console.print("  3. Store your own memories: gemma /store 'your text'")
        console.print("  4. Recall memories: gemma /recall 'your query'")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Check Redis is running: redis-cli -p 6380 ping")
        console.print("  2. Verify config: cat config/config.toml")
        console.print("  3. Check logs: tail -f ~/.gemma_cli/gemma.log")


if __name__ == "__main__":
    main()
