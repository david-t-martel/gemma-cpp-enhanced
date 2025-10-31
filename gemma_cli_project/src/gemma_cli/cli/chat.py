"""
Interactive chat interface for the Gemma CLI.

This module provides interactive chat commands with Rich formatting,
and streaming support.
"""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt

from ..utils import get_console, handle_exceptions
from ..core.config import Settings
from ..core.rag import RagClient

# Create chat subcommand app
chat_app = typer.Typer(name="chat", help="💬 Interactive chat commands", rich_markup_mode="rich")

console = get_console()


@chat_app.command("interactive")
@handle_exceptions(console)
def interactive(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use for chat")
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", "-t", help="Generation temperature")
    ] = None,
    stream: Annotated[bool, typer.Option("--stream", help="Enable streaming responses")] = True,
) -> None:
    """Start an interactive chat session."""
    settings = Settings()
    if model:
        settings.model_name = model
    if temperature:
        settings.temperature = temperature

    asyncio.run(
        _run_interactive_chat(
            settings=settings,
            stream=stream,
        )
    )


async def _run_interactive_chat(
    settings: Settings,
    stream: bool,
) -> None:
    """Run the interactive chat session."""

    console.print("🤖 Gemma CLI - Interactive Chat", style="bold cyan")
    console.print(f"Model: {settings.model_name}, Temperature: {settings.temperature}", style="dim")
    console.print("Press Ctrl+D or type /exit to quit.", style="dim")

    rag_client = RagClient("C:\\codedev\\llm\\rag-redis\\rag-binaries\\bin\\rag-server.exe")
    rag_client.start_server()

    # Main chat loop
    try:
        while True:
            # Get user input
            user_input = await _get_user_input(console)

            if user_input is None:  # User wants to exit
                break

            if user_input.startswith("/"):  # Handle commands
                if await _handle_chat_command(user_input, console, rag_client):
                    continue
                else:
                    break

            # Generate response
            try:
                if stream:
                    await _stream_response(user_input, console)
                else:
                    await _generate_response(user_input, console)

            except KeyboardInterrupt:
                console.print("\n⏸️  Generation interrupted", style="yellow")
                continue
            except Exception as e:
                console.print(f"\n❌ Error generating response: {e}", style="red")
                continue

    except KeyboardInterrupt:
        console.print("\n👋 Chat session interrupted", style="yellow")
    finally:
        rag_client.stop_server()
        console.print("👋 Goodbye!", style="blue")


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
    command: str, console: Console, rag_client: RagClient
) -> bool:
    """Handle chat commands. Returns True to continue, False to exit."""
    cmd_parts = command[1:].split()
    if not cmd_parts:
        return True

    cmd = cmd_parts[0].lower()

    if cmd in {"exit", "quit"}:
        return False
    elif cmd == "rag":
        response = rag_client.rag_command("search", query="What is the capital of France?")
        console.print(response)
    else:
        console.print(f"❓ Unknown command: {cmd}. Type /exit to quit.", style="red")

    return True


async def _generate_response(
    message: str, console: Console
) -> None:
    """Generate a non-streaming response."""
    console.print("\n[bold green]Gemma:[/bold green]")
    console.print(f"Echo: {message}")
    console.print()


async def _stream_response(
    message: str, console: Console
) -> None:
    """Generate a streaming response."""
    console.print("\n[bold green]Gemma:[/bold green]")

    with Live(console=console, refresh_per_second=10) as live:
        full_response = ""
        for char in f"Echo: {message}":
            full_response += char
            live.update(Markdown(full_response))
            await asyncio.sleep(0.05)
    console.print()
