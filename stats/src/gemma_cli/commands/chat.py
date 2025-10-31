"""Chat and conversation commands."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


@click.group(name="chat")
def chat_group() -> None:
    """Interactive chat and conversation management."""
    pass


@chat_group.command(name="interactive")
@click.option(
    "--model",
    type=str,
    help="Model preset to use (2b, 4b, 9b, 27b)",
)
@click.option(
    "--profile",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced",
    help="Performance profile",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Enable streaming responses",
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
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--enable-rag/--no-rag",
    default=True,
    help="Enable RAG memory enhancement",
)
@click.pass_context
async def interactive(
    ctx: click.Context,
    model: Optional[str],
    profile: str,
    stream: bool,
    max_tokens: int,
    temperature: float,
    enable_rag: bool,
) -> None:
    """Start interactive chat session.

    \b
    Features:
    - Multi-turn conversations with context
    - Real-time streaming responses
    - RAG-enhanced memory retrieval
    - Save/load conversation history
    - In-chat commands (prefix with /)

    \b
    In-chat Commands:
      /help      - Show available commands
      /clear     - Clear conversation history
      /save      - Save conversation
      /history   - View message history
      /settings  - Show current settings
      /quit      - Exit chat session

    \b
    Example:
      gemma chat interactive --model 4b --temperature 0.8
    """
    from gemma_cli.core.conversation import ConversationManager
    from gemma_cli.core.gemma import GemmaInterface
    from gemma_cli.rag.adapter import HybridRAGManager, MemoryType
    from gemma_cli.utils.config import load_config
    from gemma_cli.widgets.spinner import create_spinner

    config = load_config(ctx.obj["CONFIG_PATH"])

    # Initialize components
    console.print(Panel(
        "[cyan]Initializing Gemma CLI[/cyan]\n"
        f"Profile: [green]{profile}[/green] | "
        f"Model: [green]{model or config.get('default_model', '4b')}[/green] | "
        f"RAG: [green]{'enabled' if enable_rag else 'disabled'}[/green]",
        title="Chat Session",
        border_style="cyan"
    ))

    # Load Gemma interface
    with console.status("[cyan]Loading model...") as status:
        try:
            gemma = GemmaInterface(
                model_path=config["model_path"],
                tokenizer_path=config.get("tokenizer_path"),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            conversation = ConversationManager(
                max_context_length=config.get("max_context", 8192)
            )
        except Exception as e:
            console.print(f"[red]Failed to initialize Gemma: {e}[/red]")
            raise click.Abort()

    # Initialize RAG if enabled
    rag_manager = None
    if enable_rag:
        with console.status("[cyan]Initializing memory system...") as status:
            try:
                rag_manager = HybridRAGManager(
                    redis_url=config.get("redis_url", "redis://localhost:6379")
                )
                if await rag_manager.initialize():
                    backend = rag_manager.get_active_backend()
                    console.print(f"[green]✓ Memory system active[/green] (backend: {backend.value})")
                else:
                    console.print("[yellow]⚠ Memory system unavailable, continuing without RAG[/yellow]")
                    enable_rag = False
            except Exception as e:
                console.print(f"[yellow]⚠ RAG initialization warning: {e}[/yellow]")
                enable_rag = False

    # Add system message
    system_prompt = config.get(
        "system_prompt",
        "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
    )
    conversation.add_message("system", system_prompt)

    # Display welcome message
    console.print("\n" + "─" * 60)
    console.print("[cyan]Type your message and press Enter to chat[/cyan]")
    console.print("[dim]Use /help for commands, /quit to exit[/dim]")
    console.print("─" * 60 + "\n")

    # Main chat loop
    running = True
    while running:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                running = await _handle_chat_command(
                    user_input,
                    conversation,
                    gemma,
                    rag_manager,
                    console,
                )
                continue

            # Add user message
            conversation.add_message("user", user_input)

            # Get RAG context if enabled
            rag_context = ""
            if enable_rag and rag_manager:
                try:
                    # Store current query in working memory
                    await rag_manager.store_memory(
                        user_input,
                        memory_type=MemoryType.WORKING,
                        importance=0.6,
                        tags=["conversation"],
                    )

                    # Recall relevant memories
                    memories = await rag_manager.recall_memories(user_input, limit=3)
                    if memories:
                        rag_context = "\n[Context from memory:\n"
                        for i, mem in enumerate(memories, 1):
                            preview = mem.content[:200]
                            rag_context += f"{i}. {preview}...\n"
                        rag_context += "]\n"
                except Exception as e:
                    console.print(f"[dim yellow]RAG lookup warning: {e}[/dim yellow]")

            # Build full prompt with context
            context_prompt = conversation.get_context_prompt()
            full_prompt = f"{rag_context}{context_prompt}\nAssistant:"

            # Generate response
            console.print("\n[bold magenta]Assistant[/bold magenta]: ", end="")

            response_text = []

            async def stream_callback(token: str) -> None:
                """Callback for streaming tokens."""
                response_text.append(token)
                console.print(token, end="", style="white")

            try:
                if stream:
                    full_response = await gemma.generate_response(
                        full_prompt,
                        stream_callback=stream_callback,
                    )
                    console.print()  # Newline after streaming
                else:
                    with console.status("[cyan]Thinking..."):
                        full_response = await gemma.generate_response(full_prompt)
                    console.print(full_response, style="white")

                # Extract assistant response (remove echo)
                if "Assistant:" in full_response:
                    assistant_response = full_response.split("Assistant:")[-1].strip()
                else:
                    assistant_response = full_response.strip()

                # Add to conversation
                conversation.add_message("assistant", assistant_response)

                # Store in episodic memory
                if enable_rag and rag_manager:
                    try:
                        await rag_manager.store_memory(
                            f"Q: {user_input}\nA: {assistant_response}",
                            memory_type=MemoryType.EPISODIC,
                            importance=0.7,
                            tags=["conversation", "qa_pair"],
                        )
                    except Exception as e:
                        pass  # Silent failure for memory storage

            except KeyboardInterrupt:
                console.print("\n[yellow]Generation interrupted[/yellow]")
                gemma.stop_generation()
            except Exception as e:
                console.print(f"\n[red]Error generating response: {e}[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except EOFError:
            running = False

    # Cleanup
    if rag_manager:
        await rag_manager.close()

    console.print("\n[cyan]Goodbye! Chat session ended.[/cyan]")


async def _handle_chat_command(
    command: str,
    conversation,
    gemma,
    rag_manager,
    console: Console,
) -> bool:
    """Handle in-chat commands.

    Returns:
        True to continue chat, False to exit
    """
    command = command.strip().lower()

    if command in ["/quit", "/exit"]:
        return False

    elif command == "/help":
        _print_chat_help(console)

    elif command == "/clear":
        conversation.clear()
        console.print("[green]✓ Conversation history cleared[/green]")

    elif command.startswith("/save"):
        parts = command.split(maxsplit=1)
        filename = parts[1] if len(parts) > 1 else f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path.home() / ".gemma_conversations" / filename
        filepath.parent.mkdir(exist_ok=True)

        if conversation.save_to_file(filepath):
            console.print(f"[green]✓ Saved to {filepath}[/green]")
        else:
            console.print("[red]✗ Failed to save conversation[/red]")

    elif command == "/history":
        _print_conversation_history(conversation, console)

    elif command == "/settings":
        _print_current_settings(gemma, conversation, rag_manager, console)

    elif command == "/stats" and rag_manager:
        stats = await rag_manager.get_memory_stats()
        console.print(f"\n[cyan]Memory Statistics:[/cyan]")
        console.print(f"  Backend: {rag_manager.get_active_backend().value}")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[dim]Type /help for available commands[/dim]")

    return True


def _print_chat_help(console: Console) -> None:
    """Print in-chat help message."""
    help_table = Table(title="In-Chat Commands", show_header=True)
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")

    commands = [
        ("/help", "Show this help message"),
        ("/clear", "Clear conversation history"),
        ("/save [filename]", "Save conversation to file"),
        ("/history", "View message history"),
        ("/settings", "Show current settings"),
        ("/stats", "Show memory statistics (if RAG enabled)"),
        ("/quit", "Exit chat session"),
    ]

    for cmd, desc in commands:
        help_table.add_row(cmd, desc)

    console.print(help_table)


def _print_conversation_history(conversation, console: Console) -> None:
    """Print conversation history."""
    table = Table(title="Conversation History", show_header=True)
    table.add_column("Role", style="cyan")
    table.add_column("Content", style="white")
    table.add_column("Time", style="dim")

    for msg in conversation.messages:
        role = msg["role"].capitalize()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        timestamp = msg.get("timestamp", "N/A")

        table.add_row(role, content, timestamp)

    console.print(table)


def _print_current_settings(gemma, conversation, rag_manager, console: Console) -> None:
    """Print current settings."""
    table = Table(title="Current Settings", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    settings = [
        ("Model Path", gemma.model_path),
        ("Max Tokens", str(gemma.max_tokens)),
        ("Temperature", f"{gemma.temperature:.2f}"),
        ("Context Length", str(conversation.max_context_length)),
        ("Messages in History", str(len(conversation.messages))),
        ("RAG Enabled", "Yes" if rag_manager else "No"),
    ]

    if rag_manager:
        settings.append(("RAG Backend", rag_manager.get_active_backend().value))

    for key, value in settings:
        table.add_row(key, value)

    console.print(table)


@chat_group.command(name="ask")
@click.argument("prompt", required=True)
@click.option("--model", type=str, help="Model preset to use")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
@click.option("--context", type=click.File("r"), help="Context file to include")
@click.pass_context
async def ask(
    ctx: click.Context,
    prompt: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    context: Optional[click.File],
) -> None:
    """Ask a single question (non-interactive mode).

    \b
    Example:
      gemma chat ask "What is a transformer model?"
      gemma chat ask "Summarize this" --context document.txt
      gemma chat ask "Explain quantum computing" --temperature 0.3
    """
    from gemma_cli.core.gemma import GemmaInterface
    from gemma_cli.utils.config import load_config

    config = load_config(ctx.obj["CONFIG_PATH"])

    # Load context if provided
    context_text = ""
    if context:
        context_text = f"\nContext:\n{context.read()}\n\n"

    # Initialize Gemma
    with console.status("[cyan]Initializing model..."):
        gemma = GemmaInterface(
            model_path=config["model_path"],
            tokenizer_path=config.get("tokenizer_path"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # Generate response
    full_prompt = f"{context_text}User: {prompt}\nAssistant:"

    with console.status("[cyan]Generating response..."):
        response = await gemma.generate_response(full_prompt)

    # Extract and display response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    console.print(Panel(
        Markdown(response),
        title="Response",
        border_style="green",
    ))


@chat_group.command(name="history")
@click.option("--export", type=click.Path(), help="Export history to file")
@click.option("--format", type=click.Choice(["json", "markdown", "text"]), default="text")
@click.pass_context
def history(ctx: click.Context, export: Optional[str], format: str) -> None:
    """View conversation history.

    \b
    Example:
      gemma chat history
      gemma chat history --export history.md --format markdown
    """
    # List saved conversations
    conversations_dir = Path.home() / ".gemma_conversations"

    if not conversations_dir.exists():
        console.print("[yellow]No conversation history found[/yellow]")
        return

    files = sorted(conversations_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        console.print("[yellow]No saved conversations found[/yellow]")
        return

    table = Table(title="Saved Conversations", show_header=True)
    table.add_column("Filename", style="cyan")
    table.add_column("Modified", style="green")
    table.add_column("Size", style="white")

    for file in files:
        stat = file.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size = f"{stat.st_size / 1024:.1f} KB"
        table.add_row(file.name, modified, size)

    console.print(table)

    if export:
        console.print(f"\n[cyan]To export a conversation:[/cyan]")
        console.print(f"  gemma chat export {files[0].name} --output {export}")
