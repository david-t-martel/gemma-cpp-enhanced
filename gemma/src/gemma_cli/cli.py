"""Main CLI entry point for gemma-cli."""

# TODO: [Deployment] Implement packaging and distribution for standalone local LLM TUI.
# This includes bundling Python dependencies, the gemma.exe executable, and models.
# Consider PyInstaller or similar tools for creating a single executable.

import asyncio
import json
import sys
from pathlib import Path

import click
import logging

# Performance optimizations: lazy imports for heavy modules
from .utils.profiler import LazyImport, PerformanceMonitor

# Essential imports (lightweight, always needed)
from .ui.console import create_console

# Lazy load command modules (only loaded when command is used)
setup_group = LazyImport('gemma_cli.commands.setup', 'setup_group')
model = LazyImport('gemma_cli.commands.model_simple', 'model')
mcp = LazyImport('gemma_cli.commands.mcp_commands', 'mcp')

# Lazy load Gemma params (only needed for chat command)
GemmaRuntimeParams = LazyImport('gemma_cli.core.gemma', 'GemmaRuntimeParams')

logger = logging.getLogger(__name__) 

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

    # Create console and inject into context (dependency injection pattern)
    console = create_console()
    ctx.obj["console"] = console

    # Store global options in context
    ctx.obj["debug"] = debug
    ctx.obj["config_path"] = config or Path.home() / ".gemma_cli" / "config.toml"

    # Check for first run (only if not running init command)
    if ctx.invoked_subcommand != "init" and check_first_run():
        logger.warning("No configuration found. Running first-time setup...")

        # Auto-run onboarding
        from .onboarding import OnboardingWizard

        wizard = OnboardingWizard(console=console)
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
    "--enable-mcp",
    is_flag=True,
    help="Enable MCP tool integration",
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
    enable_mcp: bool,
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

        # Model loading priority:
        # Priority 1: --model CLI argument (direct path or model name)
        # Priority 2: default_model from config
        model_path = None
        tokenizer_path = None

        if model:
            # Priority 1: Direct path or model name provided via --model
            model_path_obj = Path(model)
            if model_path_obj.exists() and model_path_obj.suffix == ".sbs":
                # Direct path to model file
                model_path = str(model_path_obj.resolve())
                # Auto-detect tokenizer if not provided
                if not tokenizer:
                    tokenizer_file = model_path_obj.parent / "tokenizer.spm"
                    tokenizer_path = str(tokenizer_file) if tokenizer_file.exists() else None
                else:
                    tokenizer_path = str(Path(tokenizer).resolve())
            else:
                # Try to resolve as model name from detected/configured models
                from .config.settings import get_model_by_name
                resolved = get_model_by_name(model, settings)
                if resolved:
                    model_path, tokenizer_path = resolved
                    if tokenizer:  # Override with CLI tokenizer if provided
                        tokenizer_path = str(Path(tokenizer).resolve())
                else:
                    logger.error(f"Model not found: {model}")
                    logger.info("Use 'gemma-cli model list' to see available models")
                    sys.exit(1)
        else:
            # Priority 2: Use default_model from config
            if settings.gemma.default_model:
                model_path = settings.gemma.default_model
                tokenizer_path = settings.gemma.default_tokenizer

        if not model_path:
            logger.error("No model configured. Options:")
            logger.info("  1. Run: gemma-cli init")
            logger.info("  2. Run: gemma-cli model detect")
            logger.info("  3. Run: gemma-cli model set-default <name>")
            logger.info("  4. Use: gemma-cli chat --model /path/to/model.sbs")
            logger.info("  5. Use: gemma-cli chat --model <model-name>")
            sys.exit(1)

        # Launch chat interface
        asyncio.run(_run_chat_session(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            enable_rag=enable_rag,
            enable_mcp=enable_mcp,
            max_tokens=max_tokens,
            temperature=temperature,
            debug=ctx.obj["debug"],
            config_path=ctx.obj["config_path"],
            console=ctx.obj["console"],
        ))

    except FileNotFoundError:
        logger.error("Configuration not found. Run: gemma-cli init")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error starting chat: {e}") # Use logger.exception for full traceback
        sys.exit(1)


async def _run_chat_session(
    model_path: str,
    tokenizer_path: str | None,
    enable_rag: bool,
    enable_mcp: bool,
    max_tokens: int,
    temperature: float,
    debug: bool,
    config_path: Path,
    console,  # Add console parameter
) -> None:
    """
    Run the interactive chat session with Rich UI.

    Args:
        model_path: Path to model weights
        tokenizer_path: Path to tokenizer (optional for single-file models)
        enable_rag: Whether to enable RAG context enhancement
        enable_mcp: Whether to enable MCP tool integration
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        debug: Debug mode flag
        config_path: Path to configuration file
    """
    from datetime import datetime
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text

    from .core.conversation import ConversationManager
    from .core.gemma import GemmaInterface, create_gemma_interface, GemmaRuntimeParams
    from .ui.formatters import (
        format_assistant_message,
        format_error_message,
        format_system_message,
        format_user_message,
    )
    from .ui.components import create_panel

    # Use optimized config loading with caching
    from .config.optimized_settings import load_config_cached
    settings = load_config_cached(config_path)

    # Optional RAG import
    rag_manager = None
    if enable_rag:
        try:
            from .rag.hybrid_rag import HybridRAGManager, RecallMemoriesParams, StoreMemoryParams, IngestDocumentParams
            rag_manager = HybridRAGManager(use_embedded_store=settings.redis.enable_fallback)
            await rag_manager.initialize()
            console.print("[green]✓ RAG system initialized[/green]")
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            if debug:
                raise

    # Optional MCP integration
    mcp_manager = None
    available_tools = {}
    if enable_mcp and settings.mcp.enabled:
        try:
            from .mcp.client import MCPClientManager
            from .mcp.config_loader import load_mcp_servers

            mcp_manager = MCPClientManager(tool_cache_ttl=settings.mcp.tool_cache_ttl)
            servers = load_mcp_servers(Path(settings.mcp.servers_config))

            # Connect to enabled servers
            connected_count = 0
            for name, server_config in servers.items():
                try:
                    await mcp_manager.connect_server(name, server_config)
                    tools = await mcp_manager.list_tools(name)
                    available_tools[name] = tools
                    connected_count += 1
                except Exception as e:
                    logger.warning(f"Failed to connect to MCP server '{name}': {e}")
                    if debug:
                        logger.debug(f"MCP connection error details: {e}", exc_info=True)

            if connected_count > 0:
                tool_count = sum(len(tools) for tools in available_tools.values())
                console.print(f"[green]✓ MCP enabled: {connected_count} servers, {tool_count} tools[/green]")
            else:
                console.print("[yellow]⚠ MCP enabled but no servers connected[/yellow]")
                mcp_manager = None

        except Exception as e:
            logger.warning(f"MCP initialization failed: {e}")
            if debug:
                raise
            mcp_manager = None

    # Initialize tool orchestrator if MCP is enabled
    tool_orchestrator = None
    if mcp_manager and available_tools:
        from .core.tool_orchestrator import ToolOrchestrator, ToolCallFormat

        tool_orchestrator = ToolOrchestrator(
            mcp_manager=mcp_manager,
            available_tools=available_tools,
            format_type=ToolCallFormat.JSON_BLOCK,
            max_tool_depth=5,
            require_confirmation=False  # Can be made configurable
        )
        logger.info("Tool orchestrator initialized with autonomous tool calling")

    # Initialize components
    try:
        gemma_params = GemmaRuntimeParams(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            gemma_executable=settings.gemma.executable_path if settings.gemma else None,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_mode=debug,
        )
        use_optimized = settings.performance.use_optimized_gemma if hasattr(settings, "performance") else True
        gemma = create_gemma_interface(params=gemma_params, use_optimized=use_optimized)
        conversation = ConversationManager()

        # Add system prompt with tool instructions if orchestrator is available
        if tool_orchestrator:
            system_prompt = tool_orchestrator.get_system_prompt()
            conversation.add_message("system", system_prompt)
            logger.debug("Added system prompt with tool instructions")

        # Display startup banner
        mcp_status = "Enabled" if mcp_manager else "Disabled"
        mcp_color = "green" if mcp_manager else "yellow"
        banner = create_panel(
            "[bold cyan]Gemma CLI v2.0.0[/bold cyan]\n\n"
            f"Model: [yellow]{Path(model_path).name}[/yellow]\n"
            f"RAG: [{'green' if enable_rag else 'yellow'}]{'Enabled' if enable_rag else 'Disabled'}[/]\n"
            f"MCP: [{mcp_color}]{mcp_status}[/]\n"
            f"Max Tokens: [cyan]{max_tokens}[/cyan] | Temperature: [magenta]{temperature}[/magenta]\n\n"
            "[dim]Commands: /quit, /clear, /save, /stats, /tools, /help[/dim]",
            title="Welcome",
            border_style="cyan",
        )
        console.print(banner)
        console.print()

    except FileNotFoundError as e:
        logger.error(f"Gemma initialization failed: {e}. Suggestion: Check model path and ensure model files exist.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to initialize GemmaInterface: {e}")
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

                elif command == "tools":
                    if not mcp_manager:
                        console.print(format_system_message(
                            "MCP tools not enabled. Use --enable-mcp flag.",
                            message_type="warning"
                        ))
                        continue

                    # Display available MCP tools
                    from rich.tree import Tree
                    tool_tree = Tree("[bold cyan]Available MCP Tools[/bold cyan]")

                    for server_name, tools in available_tools.items():
                        server_branch = tool_tree.add(f"[green]{server_name}[/green] ({len(tools)} tools)")
                        for tool in tools:
                            server_branch.add(f"[yellow]{tool.name}[/yellow]: {tool.description or 'No description'}")

                    console.print(tool_tree)
                    continue

                elif command == "help":
                    help_text = (
                        "[cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit chat session\n"
                        "[cyan]/clear[/cyan] - Clear conversation history\n"
                        "[cyan]/save[/cyan] - Save conversation to file\n"
                        "[cyan]/stats[/cyan] - Show session statistics\n"
                        "[cyan]/tools[/cyan] - Show available MCP tools\n"
                        "[cyan]/help[/cyan] - Show this help message"
                    )
                    console.print(create_panel(
                        help_text,
                        title="Available Commands",
                        border_style="blue",
                    ))
                    continue

                else:
                    logger.warning(f"Unknown command: /{command}. Suggestion: Type /help for available commands.")
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
                    recall_params = RecallMemoriesParams(query=user_input, limit=3)
                    memories = await rag_manager.recall_memories(params=recall_params)
                    if memories:
                        rag_context = "\n\n".join([
                            f"[Context from memory: {m.content}]"
                            for m in memories
                        ])
                        prompt = f"{rag_context}\n\n{prompt}"
                except Exception as e:
                    if debug:
                        logger.debug(f"RAG recall failed: {e}")

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

            # Process tool calls if orchestrator is available
            if tool_orchestrator:
                # Check if response contains tool calls
                processed_response, tool_results = await tool_orchestrator.process_response_with_tools(
                    response=response,
                    console=console
                )

                # If tools were called, we might need to generate a follow-up response
                if tool_results:
                    # Add tool results to conversation context
                    tool_context = "\n\n".join([
                        f"Tool {i+1} result: {json.dumps(result.output, indent=2) if result.success else result.error}"
                        for i, result in enumerate(tool_results)
                    ])

                    # Generate final response with tool results
                    console.print(format_system_message(
                        "Processing tool results...",
                        message_type="info"
                    ))

                    # Update conversation with tool results
                    conversation.add_message("system", f"Tool execution results:\n{tool_context}")

                    # Generate final response incorporating tool results
                    final_prompt = conversation.get_context_prompt() + "\n\nAssistant: Based on the tool results, "

                    final_response_text = ""
                    with Live(
                        format_assistant_message("", metadata={"tokens": 0, "time_ms": 0}),
                        console=console,
                        refresh_per_second=10,
                    ) as live:
                        async def final_stream_callback(chunk: str) -> None:
                            nonlocal final_response_text
                            final_response_text += chunk
                            elapsed = (datetime.now() - start_time).total_seconds() * 1000
                            live.update(format_assistant_message(
                                final_response_text,
                                metadata={"time_ms": elapsed}
                            ))

                        final_response = await gemma.generate_response(
                            prompt=final_prompt,
                            stream_callback=final_stream_callback,
                        )

                    response = final_response  # Use the final response

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
                    store_params = StoreMemoryParams(
                        content=f"Q: {user_input}\nA: {response}",
                        memory_type="episodic",
                        importance=0.6,
                    )
                    await rag_manager.store_memory(params=store_params)
                except Exception as e:
                    if debug:
                        logger.debug(f"RAG storage failed: {e}")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit properly[/yellow]")
            continue

        except Exception as e:
            logger.exception(f"Error during chat: {e}. Suggestion: Check model configuration and try again.")
            continue

    # Cleanup
    if rag_manager:
        try:
            await rag_manager.close()
        except Exception:
            pass

    if mcp_manager:
        try:
            await mcp_manager.shutdown()
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
        logger.error("Error: No query provided. Usage: gemma-cli ask \"your question here\"")
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
        console=ctx.obj["console"],
    ))


async def _run_single_query(
    query: str,
    model: str | None,
    max_tokens: int,
    temperature: float,
    debug: bool,
    config_path: Path,
    console,  # Add console parameter
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
    from .core.gemma import GemmaInterface, create_gemma_interface, GemmaRuntimeParams
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
            logger.error("No model path configured. Suggestion: Run: gemma-cli init")
            sys.exit(1)

        # Initialize Gemma
        gemma_params = GemmaRuntimeParams(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            gemma_executable=settings.gemma.executable_path if settings.gemma else None,
            max_tokens=max_tokens,
            temperature=temperature,
            debug_mode=debug,
        )
        use_optimized = settings.performance.use_optimized_gemma if hasattr(settings, "performance") else True
        gemma = create_gemma_interface(params=gemma_params, use_optimized=use_optimized)

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
        logger.error(f"Query failed due to FileNotFoundError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Query failed: {e}")
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
        config_path=ctx.obj["config_path"],
        console=ctx.obj["console"],
    ))


async def _run_document_ingestion(
    document: Path,
    tier: str,
    chunk_size: int,
    debug: bool,
    config_path: Path,
    console,  # Add console parameter
) -> None:
    """
    Ingest a document into RAG memory system.

    Args:
        document: Path to document file
        tier: Memory tier to store chunks
        chunk_size: Chunk size for splitting
        debug: Debug mode flag
        config_path: Path to configuration file
    """
    from .config.settings import load_config # Add load_config import
    from .rag.hybrid_rag import HybridRAGManager, IngestDocumentParams
    from .ui.formatters import format_error_message, format_system_message

    try:
        # Initialize RAG
        settings = load_config(config_path)
        logger.info("Initializing RAG system...")
        rag_manager = HybridRAGManager(use_embedded_store=settings.redis.enable_fallback)
        await rag_manager.initialize()

        logger.info(f"Ingesting document: {document.name}")

        # Ingest document
        logger.info("Processing and chunking document...")
        ingest_params = IngestDocumentParams(
            file_path=str(document.absolute()),
            memory_type=tier,
            chunk_size=chunk_size,
        )
        chunks_stored = await rag_manager.ingest_document(params=ingest_params)

        if chunks_stored > 0:
            logger.info(f"Successfully ingested {chunks_stored} chunks from {document.name}")
        else:
            logger.warning("No chunks were stored. Suggestion: Check document format and content.")

        # Cleanup
        await rag_manager.close()

    except FileNotFoundError:
        logger.error(f"Document not found: {document}. Suggestion: Check file path and try again.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
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
        config_path=ctx.obj["config_path"],
        console=ctx.obj["console"],
    ))


async def _show_memory_stats(
    tier: str | None,
    output_json: bool,
    debug: bool,
    config_path: Path,
    console,  # Add console parameter
) -> None:
    """
    Display memory system statistics.

    Args:
        tier: Optional specific tier to show
        output_json: Whether to output as JSON
        debug: Debug mode flag
    """
    from .config.settings import load_config # Add load_config import
    from .rag.hybrid_rag import HybridRAGManager
    from .ui.formatters import format_error_message, format_memory_stats
    from .ui.widgets import MemoryDashboard

    try:
        # Initialize RAG
        settings = load_config(config_path)
        logger.info("Fetching memory statistics...")
        rag_manager = HybridRAGManager(use_embedded_store=settings.redis.enable_fallback)
        await rag_manager.initialize()
        stats = await rag_manager.get_memory_stats()

        # Display results
        if output_json:
            console.print_json(data=stats)
        else:
            # Use MemoryDashboard widget for visual display (with injected console)
            dashboard = MemoryDashboard(console=console)
            console.print(dashboard.render(stats))

            # Also show table format
            console.print()
            table = format_memory_stats(stats)
            console.print(table)

        # Cleanup
        await rag_manager.close()

    except Exception as e:
        logger.exception(f"Failed to fetch memory stats: {e}")
        sys.exit(1)


cli.add_command(setup_group)

# Register simplified model management commands (no more complex profiles)
cli.add_command(model)

# Register MCP commands
cli.add_command(mcp)


def main() -> None:
    """Main entry point for gemma-cli command."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True) # Use exc_info=True to log traceback
        sys.exit(1)


if __name__ == "__main__":
    main()
