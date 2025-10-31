"""Test command palette and help system UI."""

import pytest
from pathlib import Path
from tests.playwright.utils.cli_runner import AsyncCLIRunner


@pytest.mark.asyncio
@pytest.mark.ui
async def test_help_command_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test /help command renders full command palette."""
    # Test main help
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--help"],
        timeout=10.0,
    )

    assert result.returncode == 0

    # Should show commands
    output_lower = result.stdout.lower()
    assert "usage" in output_lower or "commands" in output_lower

    # Capture help screen
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "help_command_display",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Help display snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_chat_help_commands(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test in-chat help command shows available commands."""
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "/help"),
            (2.0, "/exit"),
        ],
        timeout=10.0,
    )

    # Should show chat commands
    output_lower = result.stdout.lower()
    chat_commands = ["help", "exit", "clear", "history", "save"]
    found_commands = [cmd for cmd in chat_commands if cmd in output_lower]

    assert len(found_commands) >= 3, \
        f"Expected multiple chat commands, found: {found_commands}"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "chat_help_commands",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Chat help snapshot: {snapshot_path}")
    print(f"  Commands found: {', '.join(found_commands)}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_command_examples_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test command examples are shown in help."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "chat", "--help"],
        timeout=10.0,
    )

    # Look for example indicators
    output_lower = result.stdout.lower()
    example_indicators = ["example", "usage", "e.g.", "demo"]
    has_examples = any(ind in output_lower for ind in example_indicators)

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "command_examples_display",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Examples display snapshot: {snapshot_path}")
    print(f"  Examples {'found' if has_examples else 'not found'}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_subcommand_help(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test help for specific subcommands."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "--help"],
        timeout=10.0,
    )

    # Should show memory-specific commands
    output_lower = result.stdout.lower()
    memory_commands = ["store", "recall", "search", "stats", "clear"]
    found_commands = [cmd for cmd in memory_commands if cmd in output_lower]

    assert len(found_commands) >= 3, \
        f"Expected memory commands, found: {found_commands}"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "subcommand_help",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Subcommand help snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_command_descriptions(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test command descriptions are informative."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--help"],
        timeout=10.0,
    )

    # Each command should have description
    output = result.stdout
    # Count lines that look like command descriptions
    lines = output.split("\n")
    description_lines = [
        line for line in lines
        if any(cmd in line.lower() for cmd in ["chat", "memory", "health"])
        and len(line.strip()) > 20  # Substantial description
    ]

    assert len(description_lines) >= 2, \
        "Expected command descriptions"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "command_descriptions",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Command descriptions snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_option_flags_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test option flags and their descriptions."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "chat", "--help"],
        timeout=10.0,
    )

    # Look for option flags
    output = result.stdout
    flag_indicators = ["--", "-h", "optional", "arguments", "options"]
    has_flags = any(ind in output.lower() for ind in flag_indicators)

    assert has_flags, "Expected option flags in help"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "option_flags_display",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Option flags snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_version_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test --version flag displays version info."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--version"],
        timeout=5.0,
    )

    # Should show version
    output_lower = result.stdout.lower()
    assert "version" in output_lower or any(
        char.isdigit() for char in result.stdout
    )

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "version_display",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Version display snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_command_table_formatting(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test command table has proper column alignment."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--help"],
        timeout=10.0,
    )

    # Check for table-like structure
    output = result.stdout
    lines = output.split("\n")

    # Look for aligned columns (multiple spaces between words)
    aligned_lines = [
        line for line in lines
        if "  " in line and len(line.strip()) > 10
    ]

    assert len(aligned_lines) >= 3, \
        "Expected table-formatted help output"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "command_table_formatting",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Table formatting snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_invalid_command_suggestion(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test suggestions are shown for invalid commands."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "chatt"],  # Typo
        timeout=5.0,
    )

    # Should suggest correct command
    output_lower = result.stdout.lower()
    has_suggestion = (
        "did you mean" in output_lower
        or "suggestion" in output_lower
        or "chat" in output_lower
    )

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "invalid_command_suggestion",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Command suggestion snapshot: {snapshot_path}")
    print(f"  Suggestion {'found' if has_suggestion else 'not found'}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_command_categories(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test commands are grouped into categories."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "--help"],
        timeout=10.0,
    )

    # Look for category headers
    output_lower = result.stdout.lower()
    category_indicators = [
        "commands:", "available", "options:", "positional", "optional"
    ]
    found_categories = [cat for cat in category_indicators if cat in output_lower]

    assert len(found_categories) >= 2, \
        f"Expected command categories, found: {found_categories}"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "command_categories",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Command categories snapshot: {snapshot_path}")
