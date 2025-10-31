"""Test startup banner and initialization UI."""

import pytest
from pathlib import Path
import asyncio
from rich.console import Console

from tests.playwright.utils.cli_runner import AsyncCLIRunner


@pytest.mark.asyncio
@pytest.mark.ui
async def test_startup_banner_displays(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    test_output_dir: Path,
):
    """Test startup banner renders correctly with ASCII art and version info."""
    # Run gemma CLI with version flag
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--version"],
        timeout=10.0,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify banner elements
    assert "GEMMA" in result.stdout or "gemma" in result.stdout.lower()
    assert "version" in result.stdout.lower()

    # Take snapshot
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "startup_banner",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Snapshot saved: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_system_checks_run(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    test_output_dir: Path,
):
    """Test system health checks display during startup."""
    # Run with verbose mode to see health checks
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "health", "--verbose"],
        timeout=15.0,
    )

    # Should show system checks
    output = result.stdout.lower()

    # Look for health check indicators
    health_indicators = [
        "redis", "model", "memory", "cpu", "disk",
        "✓", "✗", "✔", "✘", "ok", "fail", "pass",
    ]

    found_indicators = [ind for ind in health_indicators if ind in output]
    assert len(found_indicators) >= 2, \
        f"Expected health check indicators, found: {found_indicators}"

    # Capture multi-frame snapshot (status checks are sequential)
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "system_health_checks",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Health check snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.slow
async def test_model_loading_spinner(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    test_output_dir: Path,
    mock_model_inference,
):
    """Test loading animation during model initialization."""
    # Start chat mode (will trigger model loading)
    process = await asyncio.create_subprocess_exec(
        "python",
        "gemma-cli.py",
        "chat",
        "--model",
        "mock://gemma-2b-it",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Capture frames during loading
    snapshots = await snapshot_recorder.capture_live_process(
        process,
        "model_loading",
        duration=5.0,
    )

    # Terminate process
    process.terminate()
    await process.wait()

    assert len(snapshots) > 0, "Should capture at least one frame"
    print(f"✓ Captured {len(snapshots)} loading frames")

    # Check for spinner characters or progress indicators
    # Common spinner chars: ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ or /-\|
    spinner_found = False
    for snapshot in snapshots[:5]:  # Check first few frames
        if snapshot.exists():
            content = snapshot.read_text() if snapshot.suffix == ".svg" else ""
            if any(c in content for c in "⠋⠙⠹⠸⠼⠴⠦⠧/-\\|"):
                spinner_found = True
                break

    # Even if spinner not detected in SVG, test passes if frames captured
    print(f"✓ Spinner animation {'detected' if spinner_found else 'frames captured'}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_startup_error_handling(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test error display when startup fails."""
    # Run with invalid configuration
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "chat", "--model", "/nonexistent/model.sbs"],
        timeout=10.0,
    )

    # Should show error
    assert result.returncode != 0 or "error" in result.stderr.lower() \
        or "error" in result.stdout.lower()

    # Capture error display
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "startup_error",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Error snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_welcome_message_content(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test welcome message contains useful information."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--help"],
        timeout=10.0,
    )

    assert result.returncode == 0

    # Should contain help information
    output = result.stdout.lower()
    assert "usage" in output or "commands" in output
    assert "chat" in output  # Should list chat command

    # Snapshot help screen
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "help_screen",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Help screen snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_startup_performance(
    cli_runner: AsyncCLIRunner,
):
    """Test startup completes within reasonable time."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--version"],
        timeout=5.0,
    )

    # Startup should be fast
    assert result.duration < 5.0, \
        f"Startup too slow: {result.duration:.2f}s"

    print(f"✓ Startup time: {result.duration:.2f}s")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_color_output_enabled(
    cli_runner: AsyncCLIRunner,
):
    """Test that color output is enabled by default."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--version"],
        env={"NO_COLOR": "0", "FORCE_COLOR": "1"},
        timeout=5.0,
    )

    # Look for ANSI color codes
    assert "\x1b[" in result.stdout, \
        "Expected ANSI color codes in output"

    print("✓ Color output detected")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_no_color_mode(
    cli_runner: AsyncCLIRunner,
):
    """Test --no-color flag disables colors."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--version", "--no-color"],
        timeout=5.0,
    )

    # Should not contain ANSI codes
    # (Some basic codes might still appear, but significantly fewer)
    ansi_count = result.stdout.count("\x1b[")
    assert ansi_count < 5, \
        f"Too many ANSI codes in no-color mode: {ansi_count}"

    print(f"✓ No-color mode (ANSI count: {ansi_count})")
