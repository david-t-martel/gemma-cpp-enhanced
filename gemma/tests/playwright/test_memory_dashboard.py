"""Test RAG memory dashboard visualization and statistics."""

import pytest
from pathlib import Path
import asyncio
from tests.playwright.utils.cli_runner import AsyncCLIRunner


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_tier_bars(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test 5-tier memory progress bars display correctly."""
    # Run memory stats command
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "stats"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Should show memory tiers
    output_lower = result.stdout.lower()
    memory_tiers = ["working", "short", "long", "episodic", "semantic"]

    found_tiers = [tier for tier in memory_tiers if tier in output_lower]
    assert len(found_tiers) >= 3, \
        f"Expected memory tier names, found: {found_tiers}"

    # Capture dashboard
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_tier_bars",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Memory dashboard snapshot: {snapshot_path}")
    print(f"  Tiers found: {', '.join(found_tiers)}")


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.slow
async def test_memory_dashboard_live_update(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test live updating memory dashboard during operations."""
    # Start dashboard in watch mode (if supported)
    process = await asyncio.create_subprocess_exec(
        "python",
        "gemma-cli.py",
        "memory",
        "stats",
        "--watch",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Capture frames over time
    snapshots = await snapshot_recorder.capture_live_process(
        process,
        "memory_dashboard_live",
        duration=5.0,
    )

    # Cleanup
    process.terminate()
    await process.wait()

    assert len(snapshots) >= 3, \
        f"Should capture multiple dashboard frames, got {len(snapshots)}"

    print(f"✓ Captured {len(snapshots)} dashboard update frames")

    # Generate comparison
    if len(snapshots) >= 2:
        comparison = await snapshot_recorder.generate_comparison(
            snapshots[0],
            snapshots[-1],
            "memory_dashboard_changes",
        )
        print(f"✓ Dashboard comparison: {comparison}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_table_rendering(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test memory entry table display with columns."""
    # Add some test memories first
    await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "store", "Test memory 1"],
        timeout=5.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "store", "Test memory 2"],
        timeout=5.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Recall memories (shows table)
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "recall", "test"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Should show table with memories
    output_lower = result.stdout.lower()
    assert "memory" in output_lower or "test" in output_lower

    # Capture table rendering
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_table_rendering",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Memory table snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_capacity_indicators(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test capacity indicators show fill percentages."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "stats", "--verbose"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Look for capacity indicators (numbers, percentages, fractions)
    output = result.stdout
    capacity_patterns = ["%", "/", "capacity", "used", "available", "full"]
    found_indicators = [p for p in capacity_patterns if p in output.lower()]

    assert len(found_indicators) >= 2, \
        f"Expected capacity indicators, found: {found_indicators}"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_capacity_indicators",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Capacity indicators snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_search_results(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test search results display with relevance scores."""
    # Store some searchable memories
    test_memories = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Rust is a systems programming language",
    ]

    for memory in test_memories:
        await cli_runner.run(
            ["python", "gemma-cli.py", "memory", "store", memory],
            timeout=5.0,
            env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
        )

    # Search for memories
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "search", "programming"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Should show search results
    output_lower = result.stdout.lower()
    assert "programming" in output_lower or "result" in output_lower

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_search_results",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Search results snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_consolidation_display(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test memory consolidation process visualization."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "consolidate", "--dry-run"],
        timeout=15.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Should show consolidation information
    output_lower = result.stdout.lower()
    consolidation_terms = ["consolidate", "merge", "compress", "tier"]
    found_terms = [t for t in consolidation_terms if t in output_lower]

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_consolidation_display",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Consolidation display snapshot: {snapshot_path}")
    print(f"  Terms found: {', '.join(found_terms)}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_export_preview(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test memory export preview display."""
    # Add memories
    await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "store", "Export test memory"],
        timeout=5.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Preview export
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "export", "--preview"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Should show export preview
    assert len(result.stdout) > 20

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_export_preview",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Export preview snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_memory_visualization_colors(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test color-coded memory tiers are distinct."""
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "stats"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Count distinct color codes
    color_codes = [
        "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[34m", "\x1b[35m", "\x1b[36m",  # Basic
        "\x1b[91m", "\x1b[92m", "\x1b[93m", "\x1b[94m", "\x1b[95m", "\x1b[96m",  # Bright
    ]

    found_colors = [code for code in color_codes if code in result.stdout]
    unique_colors = len(set(found_colors))

    assert unique_colors >= 3, \
        f"Expected multiple colors for tiers, found {unique_colors}"

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_tier_colors",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Color-coded tiers snapshot: {snapshot_path}")
    print(f"  Unique colors detected: {unique_colors}")


@pytest.mark.asyncio
@pytest.mark.ui
async def test_empty_memory_state(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test display when no memories exist."""
    # Clear any existing memories
    await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "clear", "--confirm"],
        timeout=5.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Show stats for empty state
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "stats"],
        timeout=10.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Should indicate empty state
    output_lower = result.stdout.lower()
    empty_indicators = ["empty", "no memories", "0", "none"]
    has_empty_indicator = any(ind in output_lower for ind in empty_indicators)

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "empty_memory_state",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Empty state snapshot: {snapshot_path}")
    print(f"  Empty indicator {'found' if has_empty_indicator else 'not found'}")
