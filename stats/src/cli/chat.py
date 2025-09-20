"""Interactive chat interface for the Gemma chatbot.

This module provides interactive chat commands with Rich formatting,
streaming support, session management, and context persistence.
"""

import asyncio
from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from ..domain.models.chat import ChatSession
from ..domain.models.chat import MessageRole
from ..infrastructure.llm.gemma import GemmaLLM
from ..shared.config.settings import Settings
from .utils import get_console
from .utils import handle_exceptions
from .utils import load_session
from .utils import save_session

# Create chat subcommand app
chat_app = typer.Typer(name="chat", help="💬 Interactive chat commands", rich_markup_mode="rich")

console = get_console()


@chat_app.command("interactive")
@handle_exceptions(console)
def interactive(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use for chat")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option("--session", "-s", help="Resume existing session by ID")
    ] = None,
    system_prompt: Annotated[
        str | None, typer.Option("--system", help="System prompt to use")
    ] = None,
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="Generation temperature", min=0.0, max=2.0)
    ] = 0.7,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Maximum tokens per response", min=1)
    ] = 1024,
    stream: Annotated[bool, typer.Option("--stream", help="Enable streaming responses")] = True,
    save_session: Annotated[
        bool, typer.Option("--save/--no-save", help="Save session for later resume")
    ] = True,
) -> None:
    """Start an interactive chat session."""
    asyncio.run(
        _run_interactive_chat(
            model=model,
            session_id=session_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            save_session_flag=save_session,
        )
    )


async def _run_interactive_chat(
    model: str | None,
    session_id: str | None,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    stream: bool,
    save_session_flag: bool,
) -> None:
    """Run the interactive chat session."""

    # Initialize or load session
    if session_id:
        session = await load_session(session_id)
        if not session:
            console.print(f"❌ Session '{session_id}' not found", style="red")
            raise typer.Exit(1)
        console.print(f"📂 Resumed session: {session.title or session.id}", style="green")
    else:
        session = ChatSession(
            title=f"Chat Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            system_prompt=system_prompt,
        )
        console.print(f"🆕 Started new chat session: {session.id}", style="blue")

    # Load settings and model
    settings = Settings()
    if model:
        settings.model.name = model

    # Initialize the LLM
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        try:
            llm = GemmaLLM(settings)
            await llm.load()
            progress.remove_task(task)
            console.print("✅ Model loaded successfully", style="green")
        except Exception as e:
            progress.remove_task(task)
            console.print(f"❌ Failed to load model: {e}", style="red")
            raise typer.Exit(1)

    # Show welcome message and instructions
    await _show_welcome_message(session, llm, console)

    # Main chat loop
    try:
        while True:
            # Get user input
            user_input = await _get_user_input(console)

            if user_input is None:  # User wants to exit
                break

            if user_input.startswith("/"):  # Handle commands
                if await _handle_chat_command(user_input, session, llm, console):
                    continue
                else:
                    break

            # Add user message to session
            session.add_user_message(user_input)

            # Generate response
            try:
                if stream:
                    await _stream_response(session, llm, temperature, max_tokens, console)
                else:
                    await _generate_response(session, llm, temperature, max_tokens, console)

            except KeyboardInterrupt:
                console.print("\n⏸️  Generation interrupted", style="yellow")
                continue
            except Exception as e:
                console.print(f"\n❌ Error generating response: {e}", style="red")
                continue

    except KeyboardInterrupt:
        console.print("\n👋 Chat session interrupted", style="yellow")
    finally:
        # Save session if requested
        if save_session_flag and session.messages:
            try:
                await save_session(session)
                console.print(f"💾 Session saved: {session.id}", style="green")
            except Exception as e:
                console.print(f"⚠️  Failed to save session: {e}", style="yellow")

        # Cleanup
        if hasattr(llm, "cleanup"):
            await llm.cleanup()

        console.print("👋 Goodbye!", style="blue")


async def _show_welcome_message(session: ChatSession, llm: GemmaLLM, console: Console) -> None:
    """Show welcome message and session info."""
    welcome_text = Text()
    welcome_text.append("🤖 Gemma Chat Session\n", style="bold cyan")
    welcome_text.append(f"Session ID: {session.id}\n", style="dim")

    # Model info
    model_info = await llm.get_model_info()
    welcome_text.append(f"Model: {model_info.get('model_name', 'Unknown')}\n", style="blue")
    welcome_text.append(f"Device: {model_info.get('device', 'Unknown')}\n", style="blue")

    if model_info.get("memory_usage"):
        mem = model_info["memory_usage"]
        welcome_text.append(
            f"GPU Memory: {mem.get('allocated', 0):.1f}GB allocated\n", style="green"
        )

    welcome_text.append("\n📝 Commands:\n", style="bold")
    welcome_text.append("  /help     - Show help\n", style="dim")
    welcome_text.append("  /clear    - Clear session\n", style="dim")
    welcome_text.append("  /history  - Show message history\n", style="dim")
    welcome_text.append("  /save     - Save current session\n", style="dim")
    welcome_text.append("  /stats    - Show session statistics\n", style="dim")
    welcome_text.append("  /exit     - Exit chat\n", style="dim")
    welcome_text.append(
        "\nPress Ctrl+C to interrupt generation, Ctrl+D or /exit to quit.\n", style="yellow"
    )

    panel = Panel(welcome_text, title="Chat Session", border_style="cyan", padding=(1, 2))
    console.print(panel)


async def _get_user_input(console: Console) -> str | None:
    """Get user input with proper handling."""
    try:
        user_input = Prompt.ask(
            "[bold blue]You[/bold blue]",
            console=console,
        ).strip()

        if not user_input:
            return await _get_user_input(console)

        return user_input

    except EOFError:
        return None  # User pressed Ctrl+D
    except KeyboardInterrupt:
        console.print("\nUse /exit to quit or Ctrl+D")
        return await _get_user_input(console)


async def _handle_chat_command(
    command: str, session: ChatSession, llm: GemmaLLM, console: Console
) -> bool:
    """Handle chat commands. Returns True to continue, False to exit."""
    cmd_parts = command[1:].split()
    if not cmd_parts:
        return True

    cmd = cmd_parts[0].lower()

    if cmd in {"exit", "quit"}:
        return False

    elif cmd == "help":
        _show_help(console)

    elif cmd == "clear":
        session.clear_messages()
        console.print("🗑️  Session cleared", style="yellow")

    elif cmd == "history":
        await _show_history(session, console)

    elif cmd == "save":
        try:
            await save_session(session)
            console.print(f"💾 Session saved: {session.id}", style="green")
        except Exception as e:
            console.print(f"❌ Failed to save session: {e}", style="red")

    elif cmd == "stats":
        await _show_stats(session, llm, console)

    elif cmd == "model":
        if len(cmd_parts) > 1:
            # Switch model (would require reloading)
            console.print("Model switching not yet implemented", style="yellow")
        else:
            model_info = await llm.get_model_info()
            console.print(f"Current model: {model_info.get('model_name', 'Unknown')}", style="blue")

    else:
        console.print(f"❓ Unknown command: {cmd}. Type /help for available commands.", style="red")

    return True


def _show_help(console: Console) -> None:
    """Show help information."""
    help_text = Text()
    help_text.append("Available Commands:\n", style="bold")
    help_text.append("/help     - Show this help message\n")
    help_text.append("/clear    - Clear all messages in current session\n")
    help_text.append("/history  - Show conversation history\n")
    help_text.append("/save     - Save current session to disk\n")
    help_text.append("/stats    - Show session and model statistics\n")
    help_text.append("/model    - Show current model information\n")
    help_text.append("/exit     - Exit the chat session\n")

    panel = Panel(help_text, title="Help", border_style="blue", padding=(1, 2))
    console.print(panel)


async def _show_history(session: ChatSession, console: Console) -> None:
    """Show conversation history."""
    if not session.messages:
        console.print("📭 No messages in session", style="yellow")
        return

    console.print(f"📜 Message History ({len(session.messages)} messages):", style="bold blue")
    console.print()

    for i, message in enumerate(session.messages, 1):
        role_style = "blue" if message.role == MessageRole.USER else "green"
        role_name = "You" if message.role == MessageRole.USER else "Gemma"

        timestamp = message.created_at.strftime("%H:%M:%S")

        console.print(
            f"[{i}] [{role_style}]{role_name}[/{role_style}] ({timestamp}):", style="bold"
        )

        # Truncate long messages
        content = message.content
        if len(content) > 200:
            content = content[:200] + "..."

        console.print(content, style="dim")
        console.print()


async def _show_stats(session: ChatSession, llm: GemmaLLM, console: Console) -> None:
    """Show session and model statistics."""
    stats_table = Table(title="Session Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    # Session stats
    stats_table.add_row("Session ID", session.id)
    stats_table.add_row("Messages", str(len(session.messages)))
    stats_table.add_row("User Messages", str(session.get_user_message_count()))
    stats_table.add_row("Assistant Messages", str(session.get_assistant_message_count()))
    stats_table.add_row("Total Characters", str(session.get_conversation_length()))

    # Token usage
    if session.total_token_usage.total_tokens > 0:
        usage = session.total_token_usage
        stats_table.add_row("Total Tokens", str(usage.total_tokens))
        stats_table.add_row("Prompt Tokens", str(usage.prompt_tokens))
        stats_table.add_row("Completion Tokens", str(usage.completion_tokens))

    # Duration
    if session.messages:
        duration = datetime.utcnow() - session.created_at
        stats_table.add_row("Session Duration", str(duration).split(".")[0])

    console.print(stats_table)

    # Model info
    try:
        model_info = await llm.get_model_info()
        console.print()

        model_table = Table(title="Model Information")
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="white")

        model_table.add_row("Model", str(model_info.get("model_name", "Unknown")))
        model_table.add_row("Device", str(model_info.get("device", "Unknown")))
        model_table.add_row("Parameters", str(model_info.get("parameter_count", "Unknown")))

        if "memory_usage" in model_info:
            mem = model_info["memory_usage"]
            model_table.add_row(
                "GPU Memory", f"{mem.get('allocated', 0):.1f}GB / {mem.get('reserved', 0):.1f}GB"
            )

        console.print(model_table)

    except Exception as e:
        console.print(f"⚠️  Could not retrieve model info: {e}", style="yellow")


async def _generate_response(
    session: ChatSession, llm: GemmaLLM, temperature: float, max_tokens: int, console: Console
) -> None:
    """Generate a non-streaming response."""
    messages = session.get_messages_for_inference()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Generating response...", total=None)

        try:
            start_time = datetime.utcnow()
            response = await llm.generate(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
            end_time = datetime.utcnow()

            progress.remove_task(task)

            # Calculate processing time
            processing_time = int((end_time - start_time).total_seconds() * 1000)

            # Add response to session
            session.add_assistant_message(
                content=response,
                processing_time_ms=processing_time,
            )

            # Display response
            console.print("\n[bold green]Gemma:[/bold green]")
            console.print(Markdown(response))
            console.print(f"\n[dim]⏱️  {processing_time}ms[/dim]")
            console.print()

        except Exception as e:
            progress.remove_task(task)
            raise


async def _stream_response(
    session: ChatSession, llm: GemmaLLM, temperature: float, max_tokens: int, console: Console
) -> None:
    """Generate a streaming response."""
    messages = session.get_messages_for_inference()

    console.print("\n[bold green]Gemma:[/bold green]")

    full_response = ""
    start_time = datetime.utcnow()

    try:
        with Live(console=console, refresh_per_second=10) as live:
            async for chunk in llm.generate_streaming(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_tokens,
            ):
                full_response += chunk

                # Update display with current response
                if full_response.strip():
                    live.update(Markdown(full_response))

        end_time = datetime.utcnow()
        processing_time = int((end_time - start_time).total_seconds() * 1000)

        # Add response to session
        session.add_assistant_message(
            content=full_response,
            processing_time_ms=processing_time,
        )

        console.print(f"\n[dim]⏱️  {processing_time}ms[/dim]")
        console.print()

    except Exception as e:
        console.print(f"\n❌ Streaming error: {e}", style="red")
        raise


@chat_app.command("list")
@handle_exceptions(console)
def list_sessions(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Maximum number of sessions to show", min=1)
    ] = 10,
    all_sessions: Annotated[bool, typer.Option("--all", "-a", help="Show all sessions")] = False,
) -> None:
    """List saved chat sessions."""
    asyncio.run(_list_sessions(limit if not all_sessions else None))


async def _list_sessions(limit: int | None) -> None:
    """List saved sessions."""
    try:
        from .utils import list_sessions

        sessions = await list_sessions(limit)

        if not sessions:
            console.print("📭 No saved sessions found", style="yellow")
            return

        table = Table(title=f"Saved Chat Sessions ({len(sessions)})")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Messages", justify="right", style="blue")
        table.add_column("Created", style="dim")
        table.add_column("Updated", style="dim")

        for session_info in sessions:
            table.add_row(
                session_info["id"][:8] + "...",
                session_info.get("title", "Untitled")[:40],
                str(session_info.get("message_count", 0)),
                session_info.get("created_at", "Unknown"),
                session_info.get("updated_at", "Unknown"),
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error listing sessions: {e}", style="red")


@chat_app.command("delete")
@handle_exceptions(console)
def delete_session(
    session_id: Annotated[str, typer.Argument(help="Session ID to delete")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Delete a saved chat session."""
    asyncio.run(_delete_session(session_id, force))


async def _delete_session(session_id: str, force: bool) -> None:
    """Delete a session."""
    try:
        from .utils import delete_session

        if not force:
            confirm = Prompt.ask(
                f"Are you sure you want to delete session {session_id}? [y/N]", default="n"
            )
            if confirm.lower() != "y":
                console.print("❌ Deletion cancelled", style="yellow")
                return

        success = await delete_session(session_id)
        if success:
            console.print(f"🗑️  Session {session_id} deleted", style="green")
        else:
            console.print(f"❌ Session {session_id} not found", style="red")

    except Exception as e:
        console.print(f"❌ Error deleting session: {e}", style="red")


async def quick_generate(
    message: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    stream: bool,
    console: Console,
) -> None:
    """Generate a quick response without session management."""
    # Load settings
    settings = Settings()
    if model:
        settings.model.name = model

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        try:
            llm = GemmaLLM(settings)
            await llm.load()
            progress.remove_task(task)
        except Exception as e:
            progress.remove_task(task)
            console.print(f"❌ Failed to load model: {e}", style="red")
            raise typer.Exit(1)

    # Generate response
    messages = [{"role": "user", "content": message}]

    console.print(f"[bold blue]You:[/bold blue] {message}")
    console.print("\n[bold green]Gemma:[/bold green]")

    try:
        if stream:
            full_response = ""
            with Live(console=console, refresh_per_second=10) as live:
                async for chunk in llm.generate_streaming(
                    messages=messages,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                ):
                    full_response += chunk
                    if full_response.strip():
                        live.update(Markdown(full_response))
        else:
            response = await llm.generate(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
            console.print(Markdown(response))

    except Exception as e:
        console.print(f"❌ Error generating response: {e}", style="red")
        raise typer.Exit(1)
    finally:
        if hasattr(llm, "cleanup"):
            await llm.cleanup()
