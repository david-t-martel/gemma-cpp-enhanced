"""CLI utilities and helpers for Gemma chatbot.

This module provides common utilities, formatting functions, error handling,
and helper functions used across all CLI commands.
"""

import functools
import json
import logging
import platform
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import TypeVar

import psutil
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from ..domain.models.chat import ChatSession
from ..shared.config.settings import Settings

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Global console instance
_console: Console | None = None


def get_console() -> Console:
    """Get the global Rich console instance."""
    global _console
    if _console is None:
        _console = Console(
            color_system="auto",
            force_terminal=True,
            legacy_windows=False,
        )
    return _console


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Setup logging configuration with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    handlers = [RichHandler(console=get_console(), rich_tracebacks=True)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")

    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def handle_exceptions(console: Console) -> Callable[[F], F]:
    """Decorator to handle exceptions in CLI commands."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                console.print("\n⏸️  Operation interrupted by user", style="yellow")
                raise typer.Exit(1)
            except typer.Exit:
                # Re-raise typer exits
                raise
            except Exception as e:
                console.print(f"❌ Unexpected error: {e}", style="red")
                if "--verbose" in sys.argv or "-v" in sys.argv:
                    import traceback

                    console.print(traceback.format_exc(), style="dim red")
                raise typer.Exit(1)

        return wrapper

    return decorator


def validate_file_path(path: Path, must_exist: bool = True, must_be_file: bool = True) -> bool:
    """Validate file path and show appropriate error messages."""
    console = get_console()

    if must_exist and not path.exists():
        console.print(f"❌ Path does not exist: {path}", style="red")
        return False

    if path.exists() and must_be_file and not path.is_file():
        console.print(f"❌ Path is not a file: {path}", style="red")
        return False

    return True


def format_size(size_bytes: int) -> str:
    """Format byte size as human readable string."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    import math

    i = math.floor(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_available_port(preferred_port: int = 8000) -> int:
    """Find an available port starting from preferred port."""
    for port in range(preferred_port, preferred_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue
    return preferred_port  # Fallback


def show_banner(console: Console) -> None:
    """Show the application banner."""
    banner_text = Text()
    banner_text.append("🤖 ", style="bold blue")
    banner_text.append("Gemma Chatbot CLI", style="bold cyan")
    banner_text.append(" - Your AI Assistant\n", style="bold blue")
    banner_text.append("Powered by Google Gemma & PyTorch", style="dim")

    panel = Panel(banner_text, border_style="blue", padding=(1, 2))
    console.print(panel)
    console.print()


def validate_environment() -> dict[str, Any]:
    """Validate the environment and return status information."""
    validation_result = {"success": True, "warnings": [], "critical_errors": [], "info": {}}

    try:
        # Check Python version

        # Check PyTorch installation
        try:
            import torch

            validation_result["info"]["torch_version"] = torch.__version__
            validation_result["info"]["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                validation_result["info"]["cuda_version"] = torch.version.cuda
                validation_result["info"]["gpu_count"] = torch.cuda.device_count()
        except ImportError:
            validation_result["critical_errors"].append("PyTorch not installed")

        # Check transformers installation
        try:
            import transformers

            validation_result["info"]["transformers_version"] = transformers.__version__
        except ImportError:
            validation_result["critical_errors"].append("Transformers not installed")

        # Check available disk space
        try:
            disk_usage = psutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 5:
                validation_result["warnings"].append(f"Low disk space: {free_gb:.1f}GB free")
        except Exception:
            pass

        # Check available memory
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb < 4:
                validation_result["warnings"].append(f"Low memory: {available_gb:.1f}GB available")
        except Exception:
            pass

        # Check if running in supported environment
        if platform.system() not in ["Windows", "Linux", "Darwin"]:
            validation_result["warnings"].append(f"Untested platform: {platform.system()}")

    except Exception as e:
        validation_result["warnings"].append(f"Environment validation error: {e}")

    validation_result["success"] = len(validation_result["critical_errors"]) == 0
    return validation_result


def check_system_status() -> dict[str, Any]:
    """Check comprehensive system status."""
    status = {
        "environment": {"ready": True},
        "gpu": {"available": False},
        "models": {"available_count": 0},
        "config": {"valid": True},
        "warnings": [],
        "errors": [],
    }

    try:
        # Environment checks
        env_validation = validate_environment()
        status["environment"]["ready"] = env_validation["success"]
        status["warnings"].extend(env_validation["warnings"])
        status["errors"].extend(env_validation["critical_errors"])

        # GPU status
        if torch.cuda.is_available():
            status["gpu"]["available"] = True
            status["gpu"]["device_count"] = torch.cuda.device_count()

            # GPU memory info
            try:
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
                status["gpu"]["memory"] = {
                    "total": total_memory,
                    "used": allocated_memory,
                    "free": total_memory - allocated_memory,
                }
            except Exception:
                pass

        # Model availability
        try:
            models = get_local_models()
            status["models"]["available_count"] = len(models)
        except Exception:
            pass

        # Configuration validation
        try:
            settings = Settings()
            # Basic validation - if Settings() doesn't raise, config is likely valid
        except Exception as e:
            status["config"]["valid"] = False
            status["errors"].append(f"Invalid configuration: {e}")

        # Server status (if running)
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            server_info = loop.run_until_complete(_get_running_server_info())
            if server_info:
                status["server"] = {
                    "running": True,
                    "port": server_info.get("port", "unknown"),
                    "pid": server_info.get("pid"),
                }
            loop.close()
        except Exception:
            pass

    except Exception as e:
        status["errors"].append(f"System status check error: {e}")

    return status


async def _get_running_server_info() -> dict[str, Any] | None:
    """Get information about running server (duplicate from serve.py for independence)."""
    pid_file = Path.home() / ".gemma_server.pid"

    if not pid_file.exists():
        return None

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        if not psutil.pid_exists(pid):
            pid_file.unlink()
            return None

        process = psutil.Process(pid)

        # Find port
        port = None
        try:
            connections = process.connections()
            for conn in connections:
                if conn.status == "LISTEN":
                    port = conn.laddr.port
                    break
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        return {"pid": pid, "port": port}

    except Exception:
        return None


def get_local_models() -> list[dict[str, Any]]:
    """Get list of locally available models."""
    models = []

    # Common model directories
    model_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "torch" / "hub",
        Path("./models"),
        Path("./fine_tuned_model"),
    ]

    for model_dir in model_dirs:
        if not model_dir.exists():
            continue

        try:
            for item in model_dir.iterdir():
                if item.is_dir():
                    # Check if it looks like a model directory
                    if any(
                        (item / f).exists()
                        for f in ["config.json", "pytorch_model.bin", "model.safetensors"]
                    ):
                        model_info = {
                            "name": item.name,
                            "path": str(item),
                            "type": "local",
                            "size": _get_directory_size(item),
                        }

                        # Check for checkpoints
                        checkpoint_count = len(list(item.glob("checkpoint-*")))
                        if checkpoint_count > 0:
                            model_info["checkpoint_count"] = checkpoint_count

                        models.append(model_info)
        except Exception:
            continue

    return models


def _get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    try:
        for item in directory.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except Exception:
        pass
    return total_size


def find_model_checkpoints(model_dir: Path) -> list[dict[str, Any]]:
    """Find model checkpoints in directory."""
    checkpoints = []

    if not model_dir.exists():
        return checkpoints

    try:
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                except (IndexError, ValueError):
                    step = 0

                checkpoint_info = {
                    "name": item.name,
                    "path": str(item),
                    "step": step,
                    "size": _get_directory_size(item),
                    "modified": time.ctime(item.stat().st_mtime),
                }
                checkpoints.append(checkpoint_info)

        # Sort by step number
        checkpoints.sort(key=lambda x: x["step"], reverse=True)

    except Exception:
        pass

    return checkpoints


async def list_models(include_remote: bool = False) -> list[dict[str, Any]]:
    """List available models (local and optionally remote)."""
    models = get_local_models()

    if include_remote:
        # Add remote models from centralized config
        from ..shared.config.model_configs import MODEL_REGISTRY

        remote_models = []
        seen_models = set()

        for spec in MODEL_REGISTRY.values():
            # Only include unique specs (avoid duplicates from aliases)
            if spec.hf_model_id not in seen_models:
                seen_models.add(spec.hf_model_id)

                # Estimate size based on parameter count
                if spec.parameter_count:
                    size_gb = (
                        spec.parameter_count * 2 / 1_000_000_000
                    )  # Rough estimate: 2 bytes per param
                    size_str = f"~{size_gb:.0f}GB"
                else:
                    size_str = "Unknown"

                remote_models.append(
                    {
                        "name": spec.hf_model_id,
                        "type": "remote",
                        "description": f"{spec.family.value.title()} {spec.size.value} {spec.type.value} model",
                        "size": size_str,
                    }
                )

        models.extend(remote_models)

    return models


def display_models_table(models: list[dict[str, Any]], console: Console) -> None:
    """Display models in a formatted table."""
    if not models:
        console.print("📭 No models found", style="yellow")
        return

    table = Table(title=f"Available Models ({len(models)})")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Size", style="white")
    table.add_column("Description/Path", style="dim")

    for model in models:
        size_str = model.get("size", "Unknown")
        if isinstance(size_str, int):
            size_str = format_size(size_str)

        description = model.get("description", str(model.get("path", "")))
        if len(description) > 60:
            description = description[:57] + "..."

        table.add_row(model["name"], model.get("type", "unknown"), str(size_str), description)

    console.print(table)


async def download_model(model_name: str, console: Console) -> bool:
    """Download a model from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        console.print(f"📥 Downloading model: {model_name}")

        # Create progress display
        with Progress(console=console) as progress:
            task = progress.add_task("Downloading...", total=100)

            # Download model files
            try:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

                # Download tokenizer
                progress.update(task, completed=25, description="Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

                # Download model
                progress.update(task, completed=50, description="Downloading model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16,
                    device_map=None,  # Don't load to GPU during download
                )

                progress.update(task, completed=100, description="Download complete")

            except Exception as e:
                console.print(f"❌ Download failed: {e}", style="red")
                return False

        return True

    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}", style="red")
        console.print("Install with: pip install transformers huggingface_hub", style="yellow")
        return False
    except Exception as e:
        console.print(f"❌ Download error: {e}", style="red")
        return False


async def get_model_info(model_name: str) -> dict[str, Any] | None:
    """Get detailed information about a model."""
    try:
        # Check if model exists locally
        local_models = get_local_models()
        local_model = next((m for m in local_models if model_name in m["name"]), None)

        if local_model:
            model_info = local_model.copy()

            # Add more detailed info
            model_path = Path(local_model["path"])
            if model_path.exists():
                # Check for config.json
                config_file = model_path / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                            model_info["config"] = config
                            model_info["model_type"] = config.get("model_type", "unknown")
                            model_info["vocab_size"] = config.get("vocab_size", "unknown")
                    except Exception:
                        pass

                # Count parameters (approximate)
                model_files = list(model_path.glob("*.bin")) + list(
                    model_path.glob("*.safetensors")
                )
                if model_files:
                    total_size = sum(f.stat().st_size for f in model_files)
                    # Rough estimate: 4 bytes per float32 parameter
                    estimated_params = total_size // 4
                    model_info["estimated_parameters"] = estimated_params

            return model_info

        # If not local, try to get info from Hugging Face Hub
        try:
            from huggingface_hub import model_info

            hub_info = model_info(model_name)

            return {
                "name": model_name,
                "type": "remote",
                "description": getattr(hub_info, "description", ""),
                "tags": getattr(hub_info, "tags", []),
                "downloads": getattr(hub_info, "downloads", 0),
                "likes": getattr(hub_info, "likes", 0),
                "hub_url": f"https://huggingface.co/{model_name}",
            }
        except Exception:
            pass

        return None

    except Exception:
        return None


def display_model_info(model_info: dict[str, Any], console: Console) -> None:
    """Display detailed model information."""
    info_table = Table(title=f"Model Information: {model_info['name']}")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")

    # Basic info
    info_table.add_row("Name", model_info["name"])
    info_table.add_row("Type", model_info.get("type", "unknown"))

    size = model_info.get("size")
    if isinstance(size, int):
        size = format_size(size)
    info_table.add_row("Size", str(size))

    # Model-specific info
    if "model_type" in model_info:
        info_table.add_row("Model Type", str(model_info["model_type"]))

    if "vocab_size" in model_info:
        info_table.add_row("Vocabulary Size", str(model_info["vocab_size"]))

    if "estimated_parameters" in model_info:
        params = model_info["estimated_parameters"]
        if params > 1e9:
            param_str = f"{params / 1e9:.1f}B"
        elif params > 1e6:
            param_str = f"{params / 1e6:.1f}M"
        else:
            param_str = f"{params:,}"
        info_table.add_row("Est. Parameters", param_str)

    # Remote model info
    if "downloads" in model_info:
        info_table.add_row("Downloads", f"{model_info['downloads']:,}")

    if "likes" in model_info:
        info_table.add_row("Likes", str(model_info["likes"]))

    if "hub_url" in model_info:
        info_table.add_row("Hub URL", model_info["hub_url"])

    # Local model info
    if "path" in model_info:
        info_table.add_row("Local Path", str(model_info["path"]))

    if "checkpoint_count" in model_info:
        info_table.add_row("Checkpoints", str(model_info["checkpoint_count"]))

    console.print(info_table)

    # Show description if available
    if model_info.get("description"):
        console.print("\n📝 Description:")
        console.print(model_info["description"], style="dim")

    # Show tags if available
    if model_info.get("tags"):
        console.print(f"\n🏷️  Tags: {', '.join(model_info['tags'])}", style="blue")


# Session management utilities
async def save_session(session: ChatSession) -> None:
    """Save a chat session to disk."""
    sessions_dir = Path.home() / ".gemma" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    session_file = sessions_dir / f"{session.id}.json"
    session_data = session.model_dump(mode="json")

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2, default=str)


async def load_session(session_id: str) -> ChatSession | None:
    """Load a chat session from disk."""
    sessions_dir = Path.home() / ".gemma" / "sessions"
    session_file = sessions_dir / f"{session_id}.json"

    if not session_file.exists():
        return None

    try:
        with open(session_file) as f:
            session_data = json.load(f)
        return ChatSession(**session_data)
    except Exception:
        return None


async def list_sessions(limit: int | None = None) -> list[dict[str, Any]]:
    """List saved chat sessions."""
    sessions_dir = Path.home() / ".gemma" / "sessions"

    if not sessions_dir.exists():
        return []

    sessions = []

    try:
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    session_data = json.load(f)

                session_info = {
                    "id": session_data.get("id", session_file.stem),
                    "title": session_data.get("title", "Untitled"),
                    "message_count": len(session_data.get("messages", [])),
                    "created_at": session_data.get("created_at", "Unknown"),
                    "updated_at": session_data.get("updated_at", "Unknown"),
                }
                sessions.append(session_info)

            except Exception:
                continue

        # Sort by updated time (most recent first)
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)

        if limit:
            sessions = sessions[:limit]

    except Exception:
        pass

    return sessions


async def delete_session(session_id: str) -> bool:
    """Delete a saved chat session."""
    sessions_dir = Path.home() / ".gemma" / "sessions"
    session_file = sessions_dir / f"{session_id}.json"

    if session_file.exists():
        try:
            session_file.unlink()
            return True
        except Exception:
            return False

    return False
