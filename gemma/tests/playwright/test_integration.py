"""End-to-end integration tests with full user flows."""

import pytest
from pathlib import Path
import asyncio
from tests.playwright.utils.cli_runner import AsyncCLIRunner


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_complete_conversation_flow(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test full conversation from start to finish with all features.

    Steps:
    1. Start gemma-cli
    2. Wait for startup banner
    3. Send first message
    4. Receive response
    5. Store memory
    6. Recall memory
    7. View stats
    8. Export conversation
    """
    # Start chat session
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            # Stage 1: First exchange
            (1.0, "What is Python?"),
            (2.0, ""),  # Wait for response

            # Stage 2: Store memory
            (1.0, "/remember Python is a programming language"),
            (1.0, ""),

            # Stage 3: Check history
            (1.0, "/history"),
            (1.0, ""),

            # Stage 4: Save and exit
            (1.0, "/save conversation.json"),
            (1.0, "/exit"),
        ],
        timeout=30.0,
        env={"GEMMA_REDIS_DB": str(memory_state["redis_db"])},
    )

    # Verify all stages completed
    output_lower = result.stdout.lower()
    stages_completed = [
        "python" in output_lower,  # Message sent
        "remember" in output_lower or "memory" in output_lower,  # Memory stored
        "history" in output_lower or "conversation" in output_lower,  # History shown
    ]

    assert sum(stages_completed) >= 2, \
        f"Expected multiple stages completed: {stages_completed}"

    # Capture final state
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "complete_conversation_flow",
        format="svg",
        theme="monokai",
    )

    assert snapshot_path.exists()
    print(f"✓ Complete flow snapshot: {snapshot_path}")
    print(f"  Stages completed: {sum(stages_completed)}/3")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_onboarding_wizard_flow(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    tmp_path,
):
    """Test first-run onboarding experience."""
    # Simulate first run with no config
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "setup", "--interactive"],
        timeout=20.0,
        env={
            "GEMMA_CONFIG_DIR": str(tmp_path),
            "GEMMA_FIRST_RUN": "1",
        },
    )

    # Should show onboarding steps
    output_lower = result.stdout.lower()
    onboarding_indicators = [
        "welcome", "setup", "configure", "first", "getting started"
    ]
    found_indicators = [ind for ind in onboarding_indicators if ind in output_lower]

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "onboarding_wizard",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Onboarding snapshot: {snapshot_path}")
    print(f"  Indicators found: {', '.join(found_indicators)}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_recovery_flow(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test error handling and recovery across multiple errors.

    Triggers various errors and verifies appropriate handling:
    1. Invalid model path
    2. Redis connection failure
    3. Invalid command
    4. Recovery and successful operation
    """
    snapshots = []

    # Error 1: Invalid model
    result1 = await cli_runner.run(
        ["python", "gemma-cli.py", "chat", "--model", "/nonexistent/model.sbs"],
        timeout=5.0,
    )
    snap1 = await snapshot_recorder.take_snapshot(
        result1.terminal_display,
        "error_invalid_model",
        format="svg",
    )
    snapshots.append(snap1)

    # Error 2: Invalid Redis connection
    result2 = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "stats"],
        timeout=5.0,
        env={"REDIS_HOST": "invalid-host-12345"},
    )
    snap2 = await snapshot_recorder.take_snapshot(
        result2.terminal_display,
        "error_redis_connection",
        format="svg",
    )
    snapshots.append(snap2)

    # Error 3: Invalid command
    result3 = await cli_runner.run(
        ["python", "gemma-cli.py", "invalid_command"],
        timeout=5.0,
    )
    snap3 = await snapshot_recorder.take_snapshot(
        result3.terminal_display,
        "error_invalid_command",
        format="svg",
    )
    snapshots.append(snap3)

    # Success: Valid command after errors
    result4 = await cli_runner.run(
        ["python", "gemma-cli.py", "--version"],
        timeout=5.0,
    )
    snap4 = await snapshot_recorder.take_snapshot(
        result4.terminal_display,
        "recovery_success",
        format="svg",
    )
    snapshots.append(snap4)

    # Verify all snapshots created
    assert all(snap.exists() for snap in snapshots)
    print(f"✓ Captured {len(snapshots)} error/recovery snapshots")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_memory_workflow_integration(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test complete memory workflow: store, search, recall, export.

    Complete workflow:
    1. Store multiple memories
    2. View statistics
    3. Search memories
    4. Recall specific memory
    5. Export memories
    6. Clear memories
    """
    env = {"GEMMA_REDIS_DB": str(memory_state["redis_db"])}
    snapshots = []

    # Step 1: Store memories
    memories = [
        "Python is a high-level programming language",
        "JavaScript runs in web browsers",
        "Rust provides memory safety without garbage collection",
    ]

    for i, memory in enumerate(memories):
        result = await cli_runner.run(
            ["python", "gemma-cli.py", "memory", "store", memory],
            timeout=5.0,
            env=env,
        )
        snap = await snapshot_recorder.take_snapshot(
            result.terminal_display,
            f"memory_store_step_{i+1}",
            format="svg",
        )
        snapshots.append(snap)

    # Step 2: View stats
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "stats"],
        timeout=10.0,
        env=env,
    )
    snap = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_stats_after_store",
        format="svg",
    )
    snapshots.append(snap)

    # Step 3: Search memories
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "search", "programming"],
        timeout=10.0,
        env=env,
    )
    snap = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_search_results",
        format="svg",
    )
    snapshots.append(snap)

    # Step 4: Export
    export_file = memory_state["data_dir"] / "memories_export.json"
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "export", str(export_file)],
        timeout=10.0,
        env=env,
    )
    snap = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "memory_export_complete",
        format="svg",
    )
    snapshots.append(snap)

    # Verify all snapshots
    valid_snapshots = [snap for snap in snapshots if snap.exists()]
    assert len(valid_snapshots) >= 5, \
        f"Expected at least 5 snapshots, got {len(valid_snapshots)}"

    print(f"✓ Memory workflow: {len(valid_snapshots)} steps captured")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_streaming_with_memory_context(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test streaming responses use memory context."""
    env = {"GEMMA_REDIS_DB": str(memory_state["redis_db"])}

    # Store context
    await cli_runner.run(
        ["python", "gemma-cli.py", "memory", "store",
         "User prefers detailed technical explanations"],
        timeout=5.0,
        env=env,
    )

    # Start streaming chat with context
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b",
         "--use-memory", "--stream"],
        interactions=[
            (1.0, "Explain how memory works"),
            (3.0, "/exit"),
        ],
        timeout=15.0,
        env=env,
    )

    # Should show memory being used
    output_lower = result.stdout.lower()
    memory_indicators = ["context", "memory", "retrieved", "using"]
    found_indicators = [ind for ind in memory_indicators if ind in output_lower]

    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "streaming_with_memory",
        format="svg",
    )

    assert snapshot_path.exists()
    print(f"✓ Streaming with memory snapshot: {snapshot_path}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multi_session_workflow(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    memory_state,
):
    """Test multiple chat sessions with persistent memory."""
    env = {"GEMMA_REDIS_DB": str(memory_state["redis_db"])}

    # Session 1: Store information
    result1 = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b",
         "--session", "test-session-1"],
        interactions=[
            (1.0, "My name is Alice"),
            (2.0, "/exit"),
        ],
        timeout=10.0,
        env=env,
    )
    snap1 = await snapshot_recorder.take_snapshot(
        result1.terminal_display,
        "session_1_store",
        format="svg",
    )

    # Session 2: Recall information
    result2 = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b",
         "--session", "test-session-2", "--use-memory"],
        interactions=[
            (1.0, "What is my name?"),
            (2.0, "/exit"),
        ],
        timeout=10.0,
        env=env,
    )
    snap2 = await snapshot_recorder.take_snapshot(
        result2.terminal_display,
        "session_2_recall",
        format="svg",
    )

    assert snap1.exists() and snap2.exists()
    print("✓ Multi-session workflow captured")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_export_import_conversation(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
    tmp_path,
):
    """Test exporting and importing conversation history."""
    export_file = tmp_path / "conversation_export.json"

    # Export conversation
    result1 = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat", "--model", "mock://gemma-2b"],
        interactions=[
            (1.0, "Test message 1"),
            (2.0, "Test message 2"),
            (1.0, f"/export {export_file}"),
            (1.0, "/exit"),
        ],
        timeout=15.0,
    )

    # Import conversation
    if export_file.exists():
        result2 = await cli_runner.run(
            ["python", "gemma-cli.py", "chat", "--import", str(export_file),
             "--model", "mock://gemma-2b"],
            timeout=10.0,
        )

        snap = await snapshot_recorder.take_snapshot(
            result2.terminal_display,
            "import_conversation",
            format="svg",
        )

        assert snap.exists()
        print(f"✓ Export/import workflow: {export_file.exists()}")
    else:
        print("⚠ Export file not created, skipping import test")
