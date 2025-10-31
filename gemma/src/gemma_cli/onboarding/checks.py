"""Environment validation and health checks."""

import asyncio
import platform
import sys
from pathlib import Path
from typing import Any

import psutil
from rich.console import Console
from rich.table import Table

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

console = Console()


async def check_system_requirements() -> list[tuple[str, bool, str]]:
    """
    Check system requirements for gemma-cli.

    Returns:
        List of tuples (check_name, passed, message)
    """
    checks: list[tuple[str, bool, str]] = []

    # Python version check (>= 3.10)
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    py_msg = (
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro}"
        if py_ok
        else f"Python {py_version.major}.{py_version.minor} (requires >= 3.10)"
    )
    checks.append(("Python Version", py_ok, py_msg))

    # Available memory check (recommend >= 4GB)
    mem = psutil.virtual_memory()
    mem_gb = mem.available / (1024**3)
    mem_ok = mem_gb >= 4.0
    mem_msg = (
        f"{mem_gb:.1f} GB available"
        if mem_ok
        else f"{mem_gb:.1f} GB available (recommend >= 4GB)"
    )
    checks.append(("Available Memory", mem_ok, mem_msg))

    # Disk space check (recommend >= 10GB free)
    try:
        disk = psutil.disk_usage("/")
        disk_gb = disk.free / (1024**3)
        disk_ok = disk_gb >= 10.0
        disk_msg = (
            f"{disk_gb:.1f} GB free"
            if disk_ok
            else f"{disk_gb:.1f} GB free (recommend >= 10GB)"
        )
    except Exception:
        disk_ok = True
        disk_msg = "Could not check disk space"
    checks.append(("Disk Space", disk_ok, disk_msg))

    # Redis availability check
    if REDIS_AVAILABLE:
        redis_ok, redis_msg = await check_redis_connection("localhost", 6379)
    else:
        redis_ok = False
        redis_msg = "Redis library not installed"
    checks.append(("Redis Connection", redis_ok, redis_msg))

    # Check for optional dependencies
    optional_checks = [
        ("sentence-transformers", "sentence_transformers"),
        ("colorama", "colorama"),
        ("prompt-toolkit", "prompt_toolkit"),
        ("rich", "rich"),
    ]

    for name, module_name in optional_checks:
        try:
            __import__(module_name)
            checks.append((f"{name} library", True, "Installed"))
        except ImportError:
            checks.append((f"{name} library", False, "Not installed"))

    return checks


async def check_redis_connection(
    host: str, port: int, timeout: float = 5.0
) -> tuple[bool, str]:
    """
    Test Redis connection.

    Args:
        host: Redis host
        port: Redis port
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    if not REDIS_AVAILABLE:
        return False, "Redis library not available"

    try:
        redis_client = aioredis.Redis(
            host=host,
            port=port,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            decode_responses=True,
        )

        # Try to ping Redis
        result = await asyncio.wait_for(redis_client.ping(), timeout=timeout)
        await redis_client.aclose()

        if result:
            return True, f"Connected to {host}:{port}"
        else:
            return False, f"Failed to ping {host}:{port}"

    except asyncio.TimeoutError:
        return False, f"Connection timeout to {host}:{port}"
    except ConnectionRefusedError:
        return False, f"Connection refused by {host}:{port} (is Redis running?)"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


async def check_model_files(model_path: str | Path) -> tuple[bool, str]:
    """
    Validate model files exist and are readable.

    Args:
        model_path: Path to model file or directory

    Returns:
        Tuple of (success, message)
    """
    path = Path(model_path) if isinstance(model_path, str) else model_path

    # Check if path exists
    if not path.exists():
        return False, f"Model path does not exist: {path}"

    # If it's a directory, look for .sbs files
    if path.is_dir():
        sbs_files = list(path.glob("*.sbs"))
        if not sbs_files:
            return False, f"No .sbs model files found in {path}"

        # Check if tokenizer exists
        spm_files = list(path.glob("*.spm"))
        if spm_files:
            return (
                True,
                f"Found {len(sbs_files)} model(s) and tokenizer in {path.name}",
            )
        else:
            return (
                True,
                f"Found {len(sbs_files)} model(s) in {path.name} (no tokenizer)",
            )

    # If it's a file, check if it's a .sbs file
    if path.is_file():
        if path.suffix == ".sbs":
            size_mb = path.stat().st_size / (1024**2)
            return True, f"Model file found ({size_mb:.1f} MB)"
        else:
            return False, f"Not a .sbs model file: {path.suffix}"

    return False, f"Invalid model path: {path}"


def display_health_check_results(results: list[tuple[str, bool, str]]) -> bool:
    """
    Display health check results in Rich table.

    Args:
        results: List of tuples (check_name, passed, message)

    Returns:
        True if all checks passed, False otherwise
    """
    table = Table(title="System Health Check", show_header=True, header_style="bold")
    table.add_column("Check", style="cyan", width=30)
    table.add_column("Status", width=10)
    table.add_column("Details", style="white")

    all_passed = True
    for check_name, passed, message in results:
        status = "[green]✓ PASS[/green]" if passed else "[yellow]⚠ WARN[/yellow]"
        if not passed:
            all_passed = False
        table.add_row(check_name, status, message)

    console.print(table)
    return all_passed


async def diagnose_redis_issues() -> dict[str, Any]:
    """
    Diagnose common Redis connection issues.

    Returns:
        Dictionary with diagnostic information
    """
    diagnostics: dict[str, Any] = {
        "redis_installed": REDIS_AVAILABLE,
        "localhost_reachable": False,
        "common_ports": {},
        "suggestions": [],
    }

    if not REDIS_AVAILABLE:
        diagnostics["suggestions"].append(
            "Install Redis library: uv pip install redis"
        )
        return diagnostics

    # Check localhost reachability
    common_ports = [6379, 6380, 6381]
    for port in common_ports:
        success, msg = await check_redis_connection("localhost", port, timeout=2.0)
        diagnostics["common_ports"][port] = {"success": success, "message": msg}

        if success:
            diagnostics["localhost_reachable"] = True

    # Provide suggestions
    if not diagnostics["localhost_reachable"]:
        diagnostics["suggestions"].extend(
            [
                "Start Redis server: redis-server",
                "Check if Redis is installed: redis-cli --version",
                "Install Redis: https://redis.io/docs/install/",
            ]
        )

    return diagnostics


async def validate_environment_variables() -> dict[str, tuple[bool, str]]:
    """
    Validate required environment variables.

    Returns:
        Dictionary mapping variable name to (is_set, value_or_message)
    """
    import os

    optional_vars = {
        "GEMMA_MODEL_PATH": "Path to default Gemma model",
        "GEMMA_TOKENIZER_PATH": "Path to default tokenizer",
        "REDIS_HOST": "Redis server hostname",
        "REDIS_PORT": "Redis server port",
    }

    results = {}
    for var, description in optional_vars.items():
        value = os.environ.get(var)
        if value:
            results[var] = (True, value)
        else:
            results[var] = (False, f"Not set ({description})")

    return results


async def check_disk_space_for_models(
    model_dir: Path | None = None,
) -> tuple[bool, str]:
    """
    Check if sufficient disk space for model downloads.

    Args:
        model_dir: Directory to check (defaults to system default)

    Returns:
        Tuple of (sufficient, message)
    """
    if model_dir is None:
        model_dir = Path.home() / ".cache" / "gemma"

    try:
        if not model_dir.exists():
            model_dir = model_dir.parent

        disk = psutil.disk_usage(str(model_dir))
        free_gb = disk.free / (1024**3)

        # Recommend at least 10GB for models
        sufficient = free_gb >= 10.0

        if sufficient:
            return True, f"{free_gb:.1f} GB available at {model_dir}"
        else:
            return False, f"Only {free_gb:.1f} GB available (recommend >= 10GB)"

    except Exception as e:
        return False, f"Could not check disk space: {e}"


async def run_comprehensive_checks() -> dict[str, Any]:
    """
    Run comprehensive environment checks.

    Returns:
        Dictionary with all check results
    """
    results = {
        "system": await check_system_requirements(),
        "redis_diagnostics": await diagnose_redis_issues(),
        "environment_vars": await validate_environment_variables(),
        "disk_space": await check_disk_space_for_models(),
        "platform_info": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    return results
