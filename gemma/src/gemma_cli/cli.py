"""Main CLI entry point for gemma-cli."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from .commands.setup import config, health, init, reset, tutorial
from .commands.model import model, profile

console = Console()

# Version info
__version__ = "2.0.0"


def check_first_run() -> bool:
    """
    Check if this is the first run (no config exists).

    Returns:
        True if first run, False otherwise
    """
    config_path = Path.home() / ".gemma_cli" / "config.toml"
    return not config_path.exists()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with verbose output",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Path | None) -> None:
    """
    Gemma CLI - Modern terminal interface for Gemma LLM.

    A powerful CLI wrapper for Google's Gemma models with:
    • 5-tier memory system with RAG
    • MCP tool integration
    • Advanced conversation management
    • Performance monitoring
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store global options in context
    ctx.obj["debug"] = debug
    ctx.obj["config_path"] = config or Path.home() / ".gemma_cli" / "config.toml"

    # Check for first run (only if not running init command)
    if ctx.invoked_subcommand != "init" and check_first_run():
        console.print(
            "[yellow]No configuration found. Running first-time setup...[/yellow]\n"
        )

        # Auto-run onboarding
        from .onboarding import OnboardingWizard

        wizard = OnboardingWizard()
        asyncio.run(wizard.run())


@cli.command()
@click.option(
    "--model",
    "-m",
    help="Override model path from config",
)
@click.option(
    "--tokenizer",
    "-t",
    help="Override tokenizer path from config",
)
@click.option(
    "--enable-rag",
    is_flag=True,
    help="Enable RAG context enhancement",
)
@click.option(
    "--max-tokens",
    type=int,
    default=2048,
    help="Maximum tokens to generate",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (0.0-1.0)",
)
@click.pass_context
def chat(
    ctx: click.Context,
    model: str | None,
    tokenizer: str | None,
    enable_rag: bool,
    max_tokens: int,
    temperature: float,
) -> None:
    """
    Start interactive chat session.

    Opens an interactive chat interface with Gemma.
    Uses configuration from ~/.gemma_cli/config.toml.
    """
    # Import here to avoid circular dependencies
    from .config.settings import load_config

    try:
        # Load configuration
        settings = load_config(ctx.obj["config_path"])

        # Override with CLI options if provided
        model_path = model if model else (settings.gemma.default_model if settings.gemma else None)
        tokenizer_path = tokenizer if tokenizer else (settings.gemma.default_tokenizer if settings.gemma else None)

        if not model_path:
            console.print(
                "[red]Error: No model path configured. Run: gemma-cli init[/red]"
            )
            sys.exit(1)

        # Launch chat interface
        asyncio.run(_run_chat_session(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            enable_rag=enable_rag,
            max_tokens=max_tokens,
            temperature=temperature,
            debug=ctx.obj["debug"],
        ))

    except FileNotFoundError:
        console.print(
            "[red]Configuration not found. Run: gemma-cli init[/red]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error starting chat: {e}[/red]")
        if ctx.obj["debug"]:
            raise
        sys.exit(1)


async def _run_chat_session(
    model_path: str,
    tokenizer_path: str | None,
    enable_rag: bool,
    max_tokens: int,
    temperature: float,
    debug: bool,
) -> None:
    """
    Run the interactive chat session with Rich UI.

    Args:
        model_path: Path to model weights
        tokenizer_path: Path to tokenizer (optional for single-file models)
        enable_rag: Whether to enable RAG context enhancement
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        debug: Debug mode flag
    """
    from datetime import datetime
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text

    from .core.conversation import ConversationManager
    from .core.gemma import GemmaInterface
    from .ui.formatters import (
        format_assistant_message,
        format_error_message,
        format_system_message,
        format_user_message,
    )
    from .ui.components import create_panel

    # Optional RAG import
    rag_manager = None
    if enable_rag:
        try:
            from .rag.hybrid_rag import HybridRAGManager
            rag_manager = HybridRAGManager()
            await rag_manager.initialize()
            console.print("[green]✓ RAG system initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ RAG initialization failed: {e}[/yellow]")
            if debug:
                raise

    # Initialize components
    try:
        gemma = GemmaInterface(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        conversation = ConversationManager()

        # Display startup banner
        banner = create_panel(
            "[bold cyan]Gemma CLI v2.0.0[/bold cyan]\n\n"
            f"Model: [yellow]{Path(model_path).name}[/yellow]\n"
            f"RAG: [{'green' if enable_rag else 'yellow'}]{'Enabled' if enable_rag else 'Disabled'}[/]\n"
            f"Max Tokens: [cyan]{max_tokens}[/cyan] | Temperature: [magenta]{temperature}[/magenta]\n\n"
            "[dim]Commands: /quit, /clear, /save, /stats, /help[/dim]",
            title="Welcome",
            border_style="cyan",
        )
        console.print(banner)
        console.print()

    except FileNotFoundError as e:
        console.print(format_error_message(
            str(e),
            suggestion="Check model path and ensure model files exist"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(format_error_message(f"Failed to initialize: {e}"))
        if debug:
            raise
        sys.exit(1)

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[cyan bold]You[/cyan bold]")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input[1:].lower().strip()

                if command == "quit" or command == "exit":
                    console.print(format_system_message(
                        "Goodbye! Chat session ended.",
                        message_type="info"
                    ))
                    break

                elif command == "clear":
                    conversation.clear()
                    console.print(format_system_message(
                        "Conversation history cleared.",
                        message_type="success"
                    ))
                    continue

                elif command == "save":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = Path.home() / ".gemma_cli" / "sessions" / f"chat_{timestamp}.json"
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    success = await conversation.save_to_file(save_path)
                    if success:
                        console.print(format_system_message(
                            f"Conversation saved to {save_path}",
                            message_type="success"
                        ))
                    else:
                        console.print(format_error_message("Failed to save conversation"))
                    continue

                elif command == "stats":
                    stats = conversation.get_stats()
                    stats_text = (
                        f"Messages: [cyan]{stats['message_count']}[/cyan]\n"
                        f"Session Duration: [yellow]{stats['session_duration']}[/yellow]\n"
                        f"Characters: [green]{stats['total_characters']:,}[/green]\n"
                        f"Context Usage: [magenta]{stats['context_utilization']:.1%}[/magenta]"
                    )
                    console.print(create_panel(
                        stats_text,
                        title="Session Statistics",
                        border_style="blue",
                    ))
                    continue

                elif command == "help":
                    help_text = (
                        "[cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit chat session\n"
                        "[cyan]/clear[/cyan] - Clear conversation history\n"
                        "[cyan]/save[/cyan] - Save conversation to file\n"
                        "[cyan]/stats[/cyan] - Show session statistics\n"
                        "[cyan]/help[/cyan] - Show this help message"
                    )
                    console.print(create_panel(
                        help_text,
                        title="Available Commands",
                        border_style="blue",
                    ))
                    continue

                else:
                    console.print(format_error_message(
                        f"Unknown command: /{command}",
                        suggestion="Type /help for available commands"
                    ))
                    continue

            # Display user message
            console.print(format_user_message(user_input, datetime.now()))

            # Add to conversation history
            conversation.add_message("user", user_input)

            # Build prompt with context
            prompt = conversation.get_context_prompt()

            # Add RAG context if enabled
            if rag_manager:
                try:
                    memories = await rag_manager.recall_memories(user_input, limit=3)
                    if memories:
                        rag_context = "\n\n".join([
                            f"[Context from memory: {m.content}]"
                            for m in memories
                        ])
                        prompt = f"{rag_context}\n\n{prompt}"
                except Exception as e:
                    if debug:
                        console.print(f"[dim yellow]RAG recall failed: {e}[/dim yellow]")

            # Stream response with Live updates
            response_text = ""
            start_time = datetime.now()

            with Live(
                format_assistant_message("", metadata={"tokens": 0, "time_ms": 0}),
                console=console,
                refresh_per_second=10,
            ) as live:
                async def stream_callback(chunk: str) -> None:
                    nonlocal response_text
                    response_text += chunk

                    # Calculate elapsed time
                    elapsed = (datetime.now() - start_time).total_seconds() * 1000

                    # Update live display
                    live.update(format_assistant_message(
                        response_text,
                        metadata={"time_ms": elapsed}
                    ))

                # Generate response
                response = await gemma.generate_response(
                    prompt=prompt,
                    stream_callback=stream_callback,
                )

            # Final display with complete metadata
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            console.print(format_assistant_message(
                response,
                metadata={"time_ms": elapsed_ms}
            ))

            # Add assistant response to conversation
            conversation.add_message("assistant", response)

            # Store in RAG if enabled
            if rag_manager:
                try:
                    await rag_manager.store_memory(
                        content=f"Q: {user_input}\nA: {response}",
                        memory_type="episodic",
                        importance=0.6,
                    )
                except Exception as e:
                    if debug:
                        console.print(f"[dim yellow]RAG storage failed: {e}[/dim yellow]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit properly[/yellow]")
            continue

        except Exception as e:
            console.print(format_error_message(
                f"Error during chat: {e}",
                suggestion="Check model configuration and try again"
            ))
            if debug:
                raise
            continue

    # Cleanup
    if rag_manager:
        try:
            await rag_manager.close()
        except Exception:
            pass


@cli.command()
@click.argument("query", nargs=-1)
@click.option(
    "--model",
    "-m",
    help="Model to use",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum tokens to generate",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (0.0-1.0)",
)
@click.pass_context
def ask(
    ctx: click.Context,
    query: tuple[str, ...],
    model: str | None,
    max_tokens: int,
    temperature: float,
) -> None:
    """
    Ask a single question (non-interactive).

    Example:
        gemma-cli ask "What is the capital of France?"
    """
    if not query:
        console.print("[red]Error: No query provided[/red]")
        console.print("[dim]Usage: gemma-cli ask \"your question here\"[/dim]")
        sys.exit(1)

    query_text = " ".join(query)

    # Run async query
    asyncio.run(_run_single_query(
        query=query_text,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        debug=ctx.obj["debug"],
        config_path=ctx.obj["config_path"],
    ))


async def _run_single_query(
    query: str,
    model: str | None,
    max_tokens: int,
    temperature: float,
    debug: bool,
    config_path: Path,
) -> None:
    """
    Run a single-shot query without conversation history.

    Args:
        query: Query text
        model: Optional model path override
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        debug: Debug mode flag
        config_path: Path to configuration file
    """
    from datetime import datetime
    from rich.live import Live

    from .config.settings import load_config
    from .core.gemma import GemmaInterface
    from .ui.formatters import (
        format_assistant_message,
        format_error_message,
        format_user_message,
    )

    try:
        # Load configuration
        settings = load_config(config_path)
        model_path = model if model else (settings.gemma.default_model if settings.gemma else None)
        tokenizer_path = settings.gemma.default_tokenizer if settings.gemma else None

        if not model_path:
            console.print(format_error_message(
                "No model path configured",
                suggestion="Run: gemma-cli init"
            ))
            sys.exit(1)

        # Initialize Gemma
        gemma = GemmaInterface(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Display query
        console.print(format_user_message(query, datetime.now()))

        # Stream response
        response_text = ""
        start_time = datetime.now()

        with Live(
            format_assistant_message("", metadata={"tokens": 0, "time_ms": 0}),
            console=console,
            refresh_per_second=10,
        ) as live:
            async def stream_callback(chunk: str) -> None:
                nonlocal response_text
                response_text += chunk
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                live.update(format_assistant_message(
                    response_text,
                    metadata={"time_ms": elapsed}
                ))

            response = await gemma.generate_response(
                prompt=query,
                stream_callback=stream_callback,
            )

        # Final display
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        console.print(format_assistant_message(
            response,
            metadata={"time_ms": elapsed_ms}
        ))

    except FileNotFoundError as e:
        console.print(format_error_message(str(e)))
        sys.exit(1)
    except Exception as e:
        console.print(format_error_message(f"Query failed: {e}"))
        if debug:
            raise
        sys.exit(1)


@cli.command()
@click.argument("document", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tier",
    type=click.Choice(["working", "short_term", "long_term", "episodic", "semantic"]),
    default="long_term",
    help="Memory tier to store document",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    help="Chunk size for document splitting",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    document: Path,
    tier: str,
    chunk_size: int,
) -> None:
    """
    Ingest a document into the RAG memory system.

    Splits the document into chunks and stores them with embeddings
    for semantic search during conversations.
    """
    asyncio.run(_run_document_ingestion(
        document=document,
        tier=tier,
        chunk_size=chunk_size,
        debug=ctx.obj["debug"],
    ))


async def _run_document_ingestion(
    document: Path,
    tier: str,
    chunk_size: int,
    debug: bool,
) -> None:
    """
    Ingest a document into RAG memory system.

    Args:
        document: Path to document file
        tier: Memory tier to store chunks
        chunk_size: Chunk size for splitting
        debug: Debug mode flag
    """
    from .rag.hybrid_rag import HybridRAGManager
    from .ui.formatters import format_error_message, format_system_message

    try:
        # Initialize RAG
        with console.status(f"[cyan]Initializing RAG system..."):
            rag_manager = HybridRAGManager()
            await rag_manager.initialize()

        console.print(f"[cyan]Ingesting document:[/cyan] {document.name}\n")

        # Ingest document
        with console.status(f"[cyan]Processing and chunking document..."):
            chunks_stored = await rag_manager.ingest_document(
                file_path=str(document),
                memory_type=tier,
                chunk_size=chunk_size,
            )

        if chunks_stored > 0:
            console.print(format_system_message(
                f"Successfully ingested {chunks_stored} chunks from {document.name}",
                message_type="success"
            ))
        else:
            console.print(format_error_message(
                "No chunks were stored",
                suggestion="Check document format and content"
            ))

        # Cleanup
        await rag_manager.close()

    except FileNotFoundError:
        console.print(format_error_message(
            f"Document not found: {document}",
            suggestion="Check file path and try again"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(format_error_message(f"Ingestion failed: {e}"))
        if debug:
            raise
        sys.exit(1)


@cli.command()
@click.option(
    "--tier",
    type=click.Choice(["all", "working", "short_term", "long_term", "episodic", "semantic"]),
    default="all",
    help="Memory tier to display stats for",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output stats as JSON",
)
@click.pass_context
def memory(ctx: click.Context, tier: str, output_json: bool) -> None:
    """
    Display memory system statistics.

    Shows usage and capacity across all memory tiers.
    """
    asyncio.run(_show_memory_stats(
        tier=tier if tier != "all" else None,
        output_json=output_json,
        debug=ctx.obj["debug"],
    ))


async def _show_memory_stats(
    tier: str | None,
    output_json: bool,
    debug: bool,
) -> None:
    """
    Display memory system statistics.

    Args:
        tier: Optional specific tier to show
        output_json: Whether to output as JSON
        debug: Debug mode flag
    """
    from .rag.hybrid_rag import HybridRAGManager
    from .ui.formatters import format_error_message, format_memory_stats
    from .ui.widgets import MemoryDashboard

    try:
        # Initialize RAG
        with console.status("[cyan]Fetching memory statistics..."):
            rag_manager = HybridRAGManager()
            await rag_manager.initialize()
            stats = await rag_manager.get_memory_stats()

        # Display results
        if output_json:
            console.print_json(data=stats)
        else:
            # Use MemoryDashboard widget for visual display
            dashboard = MemoryDashboard()
            console.print(dashboard.render(stats))

            # Also show table format
            console.print()
            table = format_memory_stats(stats)
            console.print(table)

        # Cleanup
        await rag_manager.close()

    except Exception as e:
        console.print(format_error_message(f"Failed to fetch memory stats: {e}"))
        if debug:
            raise
        sys.exit(1)


# Register setup commands
cli.add_command(init)
cli.add_command(health)
cli.add_command(tutorial)
cli.add_command(reset)
cli.add_command(config)

# Register model management commands (Phase 4)
cli.add_command(model)
cli.add_command(profile)


def main() -> None:
    """Main entry point for gemma-cli command."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
