"""Server management commands for Gemma chatbot.

This module provides commands to start, stop, and manage the chatbot server,
including health checks, log monitoring, and performance metrics.
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any

import psutil
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .utils import format_size
from .utils import get_available_port
from .utils import get_console
from .utils import handle_exceptions

# Create server subcommand app
serve_app = typer.Typer(name="serve", help="🚀 Server management commands", rich_markup_mode="rich")

console = get_console()


@serve_app.command("start")
@handle_exceptions(console)
def start_server(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8000,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to load on startup")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of worker processes")] = 1,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for development")
    ] = False,
    daemon: Annotated[bool, typer.Option("--daemon", "-d", help="Run as daemon process")] = False,
    log_level: Annotated[str, typer.Option("--log-level", help="Logging level")] = "info",
    access_log: Annotated[
        bool, typer.Option("--access-log/--no-access-log", help="Enable access logging")
    ] = True,
    cors: Annotated[bool, typer.Option("--cors/--no-cors", help="Enable CORS")] = True,
    api_key: Annotated[
        str | None, typer.Option("--api-key", help="API key for authentication")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Server configuration file")
    ] = None,
) -> None:
    """Start the Gemma chatbot server."""
    asyncio.run(
        _start_server(
            host=host,
            port=port,
            model=model,
            workers=workers,
            reload=reload,
            daemon=daemon,
            log_level=log_level,
            access_log=access_log,
            cors=cors,
            api_key=api_key,
            config_file=config_file,
        )
    )


async def _start_server(
    host: str,
    port: int,
    model: str | None,
    workers: int,
    reload: bool,
    daemon: bool,
    log_level: str,
    access_log: bool,
    cors: bool,
    api_key: str | None,
    config_file: Path | None,
) -> None:
    """Start server implementation."""
    try:
        # Check if port is available
        if not _is_port_available(host, port):
            console.print(f"❌ Port {port} is already in use", style="red")
            # Try to find an alternative port
            alternative_port = get_available_port(port)
            if alternative_port != port:
                console.print(f"💡 Alternative port available: {alternative_port}", style="yellow")
                if typer.confirm(f"Use port {alternative_port} instead?"):
                    port = alternative_port
                else:
                    raise typer.Exit(1)
            else:
                raise typer.Exit(1)

        # Check if server is already running
        running_server = await _get_running_server_info()
        if running_server:
            console.print(
                f"⚠️  Server already running on port {running_server['port']}", style="yellow"
            )
            if not typer.confirm("Stop existing server and start new one?"):
                raise typer.Exit(1)
            await _stop_server_by_pid(running_server["pid"])

        console.print(f"🚀 Starting Gemma server on {host}:{port}", style="green bold")

        # Load configuration
        server_config = await _load_server_config(config_file)

        # Override with command line options
        server_config.update(
            {
                "host": host,
                "port": port,
                "workers": workers,
                "reload": reload,
                "log_level": log_level,
                "access_log": access_log,
                "cors": cors,
            }
        )

        if model:
            server_config["model"] = model
        if api_key:
            server_config["api_key"] = api_key

        # Show server configuration
        _show_server_config(server_config, console)

        if daemon:
            # Start as daemon process
            await _start_daemon_server(server_config)
        else:
            # Start in foreground
            await _start_foreground_server(server_config)

    except KeyboardInterrupt:
        console.print("\n⏸️  Server startup interrupted", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Failed to start server: {e}", style="red")
        raise typer.Exit(1)


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False


async def _load_server_config(config_file: Path | None) -> dict[str, Any]:
    """Load server configuration from file."""
    default_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "reload": False,
        "log_level": "info",
        "access_log": True,
        "cors": True,
        "timeout_keep_alive": 5,
        "timeout_graceful_shutdown": 30,
    }

    if config_file and config_file.exists():
        try:
            with open(config_file) as f:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    import yaml

                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)

            default_config.update(file_config.get("server", {}))
            console.print(f"📖 Loaded configuration from: {config_file}", style="blue")

        except Exception as e:
            console.print(f"⚠️  Failed to load config file: {e}", style="yellow")

    return default_config


def _show_server_config(config: dict[str, Any], console: Console) -> None:
    """Display server configuration."""
    config_table = Table(title="Server Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")

    # Show relevant configuration
    for key, value in config.items():
        if key not in ["api_key"]:  # Hide sensitive data
            config_table.add_row(key.replace("_", " ").title(), str(value))

    if config.get("api_key"):
        config_table.add_row("API Key", "***hidden***")

    console.print(config_table)
    console.print()


async def _start_daemon_server(config: dict[str, Any]) -> None:
    """Start server as daemon process."""
    try:
        import uvicorn

        from ..server.main import create_app

        # Save PID file
        pid_file = Path.home() / ".gemma_server.pid"

        # Create server config file
        config_file = Path.home() / ".gemma_server_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Start server process
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.server.main:create_app",
            "--factory",
            f"--host={config['host']}",
            f"--port={config['port']}",
            f"--workers={config['workers']}",
            f"--log-level={config['log_level']}",
        ]

        if not config.get("access_log", True):
            cmd.append("--no-access-log")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "GEMMA_CONFIG_FILE": str(config_file)},
        )

        # Save PID
        with open(pid_file, "w") as f:
            f.write(str(process.pid))

        # Wait a bit to check if server started successfully
        await asyncio.sleep(2)

        if process.poll() is None:
            console.print(f"✅ Server started as daemon (PID: {process.pid})", style="green")
            console.print(f"🌐 Server URL: http://{config['host']}:{config['port']}", style="blue")
            console.print(
                f"📊 Health check: http://{config['host']}:{config['port']}/health", style="dim"
            )
        else:
            _stdout, stderr = process.communicate()
            console.print("❌ Server failed to start:", style="red")
            if stderr:
                console.print(stderr.decode(), style="red")

    except Exception as e:
        console.print(f"❌ Failed to start daemon server: {e}", style="red")
        raise


async def _start_foreground_server(config: dict[str, Any]) -> None:
    """Start server in foreground."""
    try:
        import uvicorn

        from ..server.main import create_app

        console.print(
            f"🌐 Server will be available at: http://{config['host']}:{config['port']}",
            style="blue",
        )
        console.print("Press Ctrl+C to stop the server", style="dim")
        console.print()

        # Create and run server
        app = create_app(config)

        uvicorn_config = uvicorn.Config(
            app,
            host=config["host"],
            port=config["port"],
            log_level=config["log_level"],
            access_log=config.get("access_log", True),
            reload=config.get("reload", False),
        )

        server = uvicorn.Server(uvicorn_config)
        await server.serve()

    except KeyboardInterrupt:
        console.print("\n⏸️  Server stopped", style="yellow")
    except Exception as e:
        console.print(f"❌ Server error: {e}", style="red")
        raise


@serve_app.command("stop")
@handle_exceptions(console)
def stop_server(
    force: Annotated[bool, typer.Option("--force", "-f", help="Force stop the server")] = False,
    port: Annotated[
        int | None, typer.Option("--port", "-p", help="Stop server running on specific port")
    ] = None,
) -> None:
    """Stop the running Gemma server."""
    asyncio.run(_stop_server(force, port))


async def _stop_server(force: bool, port: int | None) -> None:
    """Stop server implementation."""
    try:
        if port:
            # Stop server on specific port
            servers = await _find_servers_by_port(port)
            if not servers:
                console.print(f"❌ No server found running on port {port}", style="red")
                return

            for server in servers:
                await _stop_server_by_pid(server["pid"], force)
        else:
            # Stop main server using PID file
            pid_file = Path.home() / ".gemma_server.pid"

            if not pid_file.exists():
                console.print("❌ No running server found (PID file missing)", style="red")
                return

            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())

                await _stop_server_by_pid(pid, force)
                pid_file.unlink()  # Remove PID file

            except Exception as e:
                console.print(f"❌ Error reading PID file: {e}", style="red")

    except Exception as e:
        console.print(f"❌ Failed to stop server: {e}", style="red")


async def _stop_server_by_pid(pid: int, force: bool = False) -> None:
    """Stop server by PID."""
    try:
        if not psutil.pid_exists(pid):
            console.print(f"❌ Process {pid} not found", style="red")
            return

        process = psutil.Process(pid)

        console.print(f"🛑 Stopping server (PID: {pid})...", style="yellow")

        if force:
            process.kill()
            console.print("✅ Server force-stopped", style="green")
        else:
            process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                console.print("✅ Server stopped gracefully", style="green")
            except psutil.TimeoutExpired:
                console.print("⏰ Graceful shutdown timeout, forcing stop...", style="yellow")
                process.kill()
                console.print("✅ Server force-stopped", style="green")

    except psutil.NoSuchProcess:
        console.print(f"❌ Process {pid} not found", style="red")
    except psutil.AccessDenied:
        console.print(f"❌ Access denied when stopping process {pid}", style="red")
    except Exception as e:
        console.print(f"❌ Error stopping server: {e}", style="red")


@serve_app.command("status")
@handle_exceptions(console)
def server_status(
    detailed: Annotated[
        bool, typer.Option("--detailed", "-d", help="Show detailed status information")
    ] = False,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Watch status continuously")] = False,
    refresh_interval: Annotated[
        int, typer.Option("--refresh", "-r", help="Refresh interval in seconds (for --watch)")
    ] = 2,
) -> None:
    """Show server status and health information."""
    asyncio.run(_show_server_status(detailed, watch, refresh_interval))


async def _show_server_status(detailed: bool, watch: bool, refresh_interval: int) -> None:
    """Show server status implementation."""
    if watch:
        try:
            with Live(console=console, refresh_per_second=1 / refresh_interval) as live:
                while True:
                    status_display = await _get_status_display(detailed)
                    live.update(status_display)
                    await asyncio.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n⏸️  Status monitoring stopped", style="yellow")
    else:
        status_display = await _get_status_display(detailed)
        console.print(status_display)


async def _get_status_display(detailed: bool) -> Panel:
    """Get formatted status display."""
    try:
        server_info = await _get_running_server_info()

        if not server_info:
            status_text = Text("❌ Server not running", style="red")
            return Panel(status_text, title="Server Status", border_style="red")

        # Basic status
        status_text = Text()
        status_text.append("✅ Server running\n", style="green bold")
        status_text.append(f"PID: {server_info['pid']}\n", style="blue")
        status_text.append(f"Port: {server_info['port']}\n", style="blue")
        status_text.append(f"Uptime: {server_info['uptime']}\n", style="cyan")

        if detailed:
            # Detailed status
            process = psutil.Process(server_info["pid"])

            status_text.append("\n📊 Resource Usage:\n", style="bold")
            status_text.append(f"CPU: {process.cpu_percent():.1f}%\n", style="white")

            memory_info = process.memory_info()
            status_text.append(f"Memory: {format_size(memory_info.rss)}\n", style="white")

            # Network connections
            try:
                connections = process.connections()
                active_connections = len([c for c in connections if c.status == "ESTABLISHED"])
                status_text.append(f"Active connections: {active_connections}\n", style="white")
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

            # Health check
            health_status = await _check_server_health(server_info["port"])
            if health_status:
                status_text.append(f"\n🏥 Health: {health_status['status']}\n", style="green")
                if "model" in health_status:
                    status_text.append(f"Model: {health_status['model']}\n", style="blue")
                if "version" in health_status:
                    status_text.append(f"Version: {health_status['version']}\n", style="dim")
            else:
                status_text.append("\n🏥 Health: Failed to connect\n", style="red")

        return Panel(
            status_text,
            title=f"Server Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="green",
        )

    except Exception as e:
        error_text = Text(f"❌ Error getting status: {e}", style="red")
        return Panel(error_text, title="Server Status", border_style="red")


async def _get_running_server_info() -> dict[str, Any] | None:
    """Get information about running server."""
    pid_file = Path.home() / ".gemma_server.pid"

    if not pid_file.exists():
        return None

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        if not psutil.pid_exists(pid):
            pid_file.unlink()  # Remove stale PID file
            return None

        process = psutil.Process(pid)

        # Find the port from command line or connections
        port = None
        try:
            cmdline = process.cmdline()
            for i, arg in enumerate(cmdline):
                if arg.startswith("--port="):
                    port = int(arg.split("=")[1])
                    break
                elif arg == "--port" and i + 1 < len(cmdline):
                    port = int(cmdline[i + 1])
                    break

            if not port:
                # Try to find from network connections
                connections = process.connections()
                for conn in connections:
                    if conn.status == "LISTEN":
                        port = conn.laddr.port
                        break
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        uptime = datetime.now() - datetime.fromtimestamp(process.create_time())

        return {
            "pid": pid,
            "port": port or "unknown",
            "uptime": str(uptime).split(".")[0],  # Remove microseconds
            "status": process.status(),
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
        }

    except Exception:
        return None


async def _find_servers_by_port(port: int) -> list[dict[str, Any]]:
    """Find servers running on specific port."""
    servers = []

    for process in psutil.process_iter(["pid", "name", "connections"]):
        try:
            if process.info["connections"]:
                for conn in process.info["connections"]:
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        servers.append(
                            {"pid": process.info["pid"], "name": process.info["name"], "port": port}
                        )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return servers


async def _check_server_health(port: int) -> dict[str, Any] | None:
    """Check server health via HTTP."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": f"HTTP {response.status_code}"}

    except Exception:
        return None


@serve_app.command("logs")
@handle_exceptions(console)
def show_logs(
    lines: Annotated[int, typer.Option("--lines", "-n", help="Number of lines to show")] = 50,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Follow log output")] = False,
    level: Annotated[str | None, typer.Option("--level", help="Filter by log level")] = None,
) -> None:
    """Show server logs."""
    asyncio.run(_show_server_logs(lines, follow, level))


async def _show_server_logs(lines: int, follow: bool, level: str | None) -> None:
    """Show server logs implementation."""
    log_file = Path.home() / ".gemma_server.log"

    if not log_file.exists():
        console.print("❌ Log file not found", style="red")
        console.print("Make sure the server is running with logging enabled", style="yellow")
        return

    try:
        if follow:
            console.print("📋 Following logs (Ctrl+C to stop)...", style="blue")
            # Implement log following
            await _follow_logs(log_file, level)
        else:
            # Show last N lines
            console.print(f"📋 Last {lines} log entries:", style="blue")
            await _show_log_tail(log_file, lines, level)

    except KeyboardInterrupt:
        console.print("\n⏸️  Log monitoring stopped", style="yellow")
    except Exception as e:
        console.print(f"❌ Error reading logs: {e}", style="red")


async def _show_log_tail(log_file: Path, lines: int, level: str | None) -> None:
    """Show last N lines of log file."""
    try:
        with open(log_file) as f:
            all_lines = f.readlines()

        # Filter by level if specified
        if level:
            filtered_lines = [line for line in all_lines if level.upper() in line]
        else:
            filtered_lines = all_lines

        # Show last N lines
        for line in filtered_lines[-lines:]:
            _format_log_line(line.strip(), console)

    except Exception as e:
        console.print(f"❌ Error reading log file: {e}", style="red")


async def _follow_logs(log_file: Path, level: str | None) -> None:
    """Follow log file output."""
    try:
        with open(log_file) as f:
            # Go to end of file
            f.seek(0, 2)

            while True:
                line = f.readline()
                if line:
                    if not level or level.upper() in line:
                        _format_log_line(line.strip(), console)
                else:
                    await asyncio.sleep(0.1)  # Wait a bit before checking again

    except Exception as e:
        console.print(f"❌ Error following logs: {e}", style="red")


def _format_log_line(line: str, console: Console) -> None:
    """Format and colorize log line."""
    if "ERROR" in line:
        console.print(line, style="red")
    elif "WARNING" in line:
        console.print(line, style="yellow")
    elif "INFO" in line:
        console.print(line, style="white")
    elif "DEBUG" in line:
        console.print(line, style="dim")
    else:
        console.print(line)
