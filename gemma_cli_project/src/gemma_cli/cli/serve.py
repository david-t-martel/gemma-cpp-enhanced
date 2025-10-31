"""Server management commands for Gemma CLI.

This module provides commands to start the chatbot server.
"""

import asyncio
import subprocess
import sys
import os
import psutil
from pathlib import Path
from typing import Annotated

import typer

from ..utils import get_console, handle_exceptions

# Create server subcommand app
serve_app = typer.Typer(name="serve", help="🚀 Server management commands", rich_markup_mode="rich")

console = get_console()
PID_FILE = Path.home() / ".gemma_cli.pid"


@serve_app.command("start")
@handle_exceptions(console)
def start_server(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8000,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for development")
    ] = False,
) -> None:
    """Start the Gemma CLI server."""
    _start_server(
        host=host,
        port=port,
        reload=reload,
    )


def _start_server(
    host: str,
    port: int,
    reload: bool,
) -> None:
    """Start server implementation."""
    console.print(f"🚀 Starting Gemma server on {host}:{port}", style="green bold")

    # Start in foreground
    _start_foreground_server(host, port, reload)


def _start_foreground_server(host: str, port: int, reload: bool) -> None:
    """Start server in foreground."""
    try:
        console.print(
            f"🌐 Server will be available at: http://{host}:{port}",
            style="blue",
        )
        console.print("Press Ctrl+C to stop the server", style="dim")
        console.print()

        # Create and run server
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "gemma_cli.server.main:app",
            f"--host={host}",
            f"--port={port}",
        ]
        if reload:
            cmd.append("--reload")

        process = subprocess.Popen(cmd)
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

    except Exception as e:
        console.print(f"❌ Server error: {e}", style="red")
        raise

@serve_app.command("stop")
@handle_exceptions(console)
def stop_server() -> None:
    """Stop the Gemma CLI server."""
    if not PID_FILE.exists():
        console.print("🤷 Server is not running.", style="yellow")
        return

    with open(PID_FILE, "r") as f:
        pid = int(f.read())

    try:
        process = psutil.Process(pid)
        process.terminate()
        console.print(f"🛑 Server stopped (PID: {pid}).", style="green")
    except psutil.NoSuchProcess:
        console.print(f"🤷 Server with PID {pid} not found.", style="yellow")
    except Exception as e:
        console.print(f"❌ Error stopping server: {e}", style="red")

    PID_FILE.unlink()
