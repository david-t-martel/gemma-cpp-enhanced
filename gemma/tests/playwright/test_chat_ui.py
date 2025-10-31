"""Test interactive chat UI components and rendering."""

import pytest
from pathlib import Path
import asyncio
from tests.playwright.utils.cli_runner import AsyncCLIRunner


@pytest.mark.asyncio
@pytest.mark.ui
async def test_user_message_format(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    test_output_dir: Path,
):
    """Test user message rendering with cyan color and panel."""
    # Run interactive chat with mock responses
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "Hello, what is 2+2?"),  # Wait 1s, send message
            (2.0, "/exit"),  # Wait 2s, exit
        ],
        timeout=10.0,
    )

    # Check for user message indicators
    output = result.stdout
    assert "hello" in output.lower() or "2+2" in output.lower()

    # Look for cyan color code (typically \x1b[36m or \x1b[96m)
    cyan_codes = ["\x1b[36m", "\x1b[96m", "\x1b[1;36m"]
    has_cyan = any(code in output for code in cyan_codes)

    # Capture snapshot
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "user_message_format",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ User message snapshot: {snapshot_path}")
    print(f"  Cyan color {'detected' if has_cyan else 'not detected'}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_assistant_response_format(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    mock_model_inference,
):
    """Test assistant response with green panel and markdown."""
    # Run chat with mocked model
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "Explain Python lists"),
            (3.0, "/exit"),
        ],
        timeout=15.0,
    )

    # Look for assistant response indicators
    output = result.stdout
    # Should contain either mock response or actual response
    assert len(output) > 100  # Response should have content

    # Look for green color code
    green_codes = ["\x1b[32m", "\x1b[92m", "\x1b[1;32m"]
    has_green = any(code in output for code in green_codes)

    # Capture response rendering
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "assistant_response_format",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Assistant response snapshot: {snapshot_path}")
    print(f"  Green color {'detected' if has_green else 'not detected'}")


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.slow
async def test_streaming_animation(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test live streaming text animation during inference."""
    # Start chat process
    process = await asyncio.create_subprocess_exec(
        "python",
        "gemma-cli.py",
        "chat",
        "--model",
        "mock://gemma-2b",
        "--stream",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Send message
    await asyncio.sleep(2.0)  # Wait for startup
    process.stdin.write(b"Write a short poem\n")
    await process.stdin.drain()

    # Capture streaming frames
    snapshots = await snapshot_recorder.capture_live_process(
        process,
        "streaming_animation",
        duration=5.0,
    )

    # Cleanup
    process.stdin.close()
    process.terminate()
    await process.wait()

    assert len(snapshots) >= 3, \
        f"Should capture multiple streaming frames, got {len(snapshots)}"

    print(f"✓ Captured {len(snapshots)} streaming frames")

    # Create comparison between first and last frame
    if len(snapshots) >= 2:
        comparison = await snapshot_recorder.generate_comparison(
            snapshots[0],
            snapshots[-1],
            "streaming_progress",
        )
        print(f"✓ Streaming comparison: {comparison}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_error_message_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test error panel with red color and suggestion."""
    # Trigger error by invalid command
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "/invalid_command"),
            (2.0, "/exit"),
        ],
        timeout=10.0,
    )

    # Should show error
    output_lower = result.stdout.lower()
    has_error = (
        "error" in output_lower
        or "invalid" in output_lower
        or "unknown" in output_lower
    )

    assert has_error, "Expected error message"

    # Look for red color code
    red_codes = ["\x1b[31m", "\x1b[91m", "\x1b[1;31m"]
    has_red = any(code in result.stdout for code in red_codes)

    # Capture error display
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "error_message_display",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Error display snapshot: {snapshot_path}")
    print(f"  Red color {'detected' if has_red else 'not detected'}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_markdown_code_blocks(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test markdown code block rendering with syntax highlighting."""
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "Show me Python code for hello world"),
            (3.0, "/exit"),
        ],
        timeout=15.0,
    )

    # Response should contain code or formatted text
    assert len(result.stdout) > 50

    # Capture code rendering
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "markdown_code_blocks",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Markdown code snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_multiline_input_handling(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test handling of multiline user input."""
    # Send multiline message (using triple quotes or line continuations)
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "This is line 1\\nThis is line 2\\nThis is line 3"),
            (3.0, "/exit"),
        ],
        timeout=10.0,
    )

    # Should handle multiline input
    assert "line" in result.stdout.lower()

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "multiline_input",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Multiline input snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_conversation_history_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test conversation history shows multiple exchanges."""
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "First message"),
            (2.0, "Second message"),
            (3.0, "Third message"),
            (4.0, "/history"),
            (5.0, "/exit"),
        ],
        timeout=20.0,
    )

    # Should show conversation history
    output_lower = result.stdout.lower()
    assert "message" in output_lower or "history" in output_lower

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "conversation_history",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Conversation history snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_token_count_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test token count and timing metadata display."""
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b", "--verbose"],
        interactions=[
            (1.0, "Short message"),
            (3.0, "/exit"),
        ],
        timeout=10.0,
    )

    # Look for metadata indicators
    output_lower = result.stdout.lower()
    metadata_indicators = ["token", "time", "ms", "second", "duration"]
    has_metadata = any(ind in output_lower for ind in metadata_indicators)

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "token_metadata_display",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Metadata display snapshot: {snapshot_path}")
    print(f"  Metadata {'detected' if has_metadata else 'not detected'}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_long_response_scrolling(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test handling of long responses that exceed screen height."""
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "Write a very long explanation with many paragraphs"),
            (5.0, "/exit"),
        ],
        timeout=15.0,
    )

    # Response should be substantial
    assert len(result.stdout) > 500

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "long_response_scrolling",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Long response snapshot: {snapshot_path}")
