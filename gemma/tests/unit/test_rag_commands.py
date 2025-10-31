"""Unit tests for RAG command handlers.

Tests cover:
- Memory command parsing and execution
- Error handling and user feedback
- Output formatting
- Async operation handling
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from gemma_cli.commands.rag_commands import (
    cleanup_command,
    consolidate_command,
    ingest_command,
    memory_commands,
    recall_command,
    search_command,
    store_command,
)
from gemma_cli.rag.memory import MemoryEntry, MemoryTier


@pytest.fixture
def cli_runner():
    """Create Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_rag_backend():
    """Create mock RAG backend."""
    backend = AsyncMock()
    backend.initialize = AsyncMock(return_value=True)
    backend.store_memory = AsyncMock(return_value="test-id-123")
    backend.recall_memories = AsyncMock(return_value=[])
    backend.search_memories = AsyncMock(return_value=[])
    backend.ingest_document = AsyncMock(return_value=5)
    backend.get_memory_stats = AsyncMock(
        return_value={
            "working": 5,
            "short_term": 20,
            "long_term": 100,
            "episodic": 50,
            "semantic": 200,
            "total": 375,
            "redis_memory": 1024 * 1024 * 10,  # 10 MB
        }
    )
    backend.cleanup_expired = AsyncMock(return_value=10)
    return backend


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.redis.host = "localhost"
    settings.redis.port = 6380
    settings.redis.db = 0
    settings.redis.pool_size = 10
    return settings


# ============================================================================
# Memory Command Tests
# ============================================================================


class TestRecallCommand:
    """Tests for memory recall command."""

    def test_recall_basic(self, cli_runner, mock_rag_backend, mock_settings):
        """Test basic recall command."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(recall_command, ["test query"])
                assert result.exit_code == 0
                assert "No memories found" in result.output or "relevant memories" in result.output

    def test_recall_with_tier(self, cli_runner, mock_rag_backend, mock_settings):
        """Test recall with specific tier."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(recall_command, ["test", "--tier", "long_term"])
                assert result.exit_code == 0

    def test_recall_with_limit(self, cli_runner, mock_rag_backend, mock_settings):
        """Test recall with custom limit."""
        # Create mock entries with required attributes
        mock_entries = []
        for i in range(3):
            entry = MagicMock(spec=MemoryEntry)
            entry.id = f"entry-{i}"
            entry.memory_type = "long_term"
            entry.importance = 0.8
            entry.access_count = 5
            entry.created_at = MagicMock()
            entry.created_at.strftime = MagicMock(return_value="2025-01-01 12:00:00")
            entry.content = f"Test content {i}"
            entry.tags = ["test"]
            entry.similarity_score = 0.9
            mock_entries.append(entry)

        mock_rag_backend.recall_memories = AsyncMock(return_value=mock_entries)

        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(recall_command, ["query", "--limit", "10"])
                assert result.exit_code == 0
                assert "Found 3 relevant memories" in result.output


class TestStoreCommand:
    """Tests for memory store command."""

    def test_store_basic(self, cli_runner, mock_rag_backend, mock_settings):
        """Test basic store command."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(store_command, ["Test memory content"])
                assert result.exit_code == 0
                assert "Memory stored successfully" in result.output

    def test_store_with_tier(self, cli_runner, mock_rag_backend, mock_settings):
        """Test store with specific tier."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(
                    store_command, ["Test", "--tier", "semantic", "--importance", "0.9"]
                )
                assert result.exit_code == 0

    def test_store_with_tags(self, cli_runner, mock_rag_backend, mock_settings):
        """Test store with tags."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(
                    store_command, ["Content", "--tags", "tag1", "--tags", "tag2"]
                )
                assert result.exit_code == 0

    def test_store_invalid_importance(self, cli_runner, mock_rag_backend, mock_settings):
        """Test store with invalid importance score."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(store_command, ["Test", "--importance", "1.5"])
                assert result.exit_code != 0
                assert "Importance must be between 0.0 and 1.0" in result.output


class TestSearchCommand:
    """Tests for memory search command."""

    def test_search_basic(self, cli_runner, mock_rag_backend, mock_settings):
        """Test basic search command."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(search_command, ["error"])
                assert result.exit_code == 0

    def test_search_with_importance_filter(self, cli_runner, mock_rag_backend, mock_settings):
        """Test search with importance filter."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(search_command, ["API", "--min-importance", "0.7"])
                assert result.exit_code == 0


class TestIngestCommand:
    """Tests for document ingest command."""

    def test_ingest_basic(self, cli_runner, mock_rag_backend, mock_settings, tmp_path):
        """Test basic document ingest."""
        # Create temporary test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test document content\nWith multiple lines.")

        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(ingest_command, [str(test_file)])
                assert result.exit_code == 0
                assert "Successfully ingested" in result.output

    def test_ingest_nonexistent_file(self, cli_runner, mock_rag_backend, mock_settings):
        """Test ingest with nonexistent file."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(ingest_command, ["/nonexistent/file.txt"])
                assert result.exit_code != 0

    def test_ingest_with_options(self, cli_runner, mock_rag_backend, mock_settings, tmp_path):
        """Test ingest with tier and chunk size options."""
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Test Document\n\nContent here.")

        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(
                    ingest_command, [str(test_file), "--tier", "semantic", "--chunk-size", "1000"]
                )
                assert result.exit_code == 0


class TestCleanupCommand:
    """Tests for cleanup command."""

    def test_cleanup_basic(self, cli_runner, mock_rag_backend, mock_settings):
        """Test basic cleanup."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(cleanup_command)
                assert result.exit_code == 0
                assert "Cleaned up" in result.output or "No expired entries" in result.output

    def test_cleanup_dry_run(self, cli_runner, mock_rag_backend, mock_settings):
        """Test cleanup with dry-run flag."""
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(cleanup_command, ["--dry-run"])
                assert result.exit_code == 0
                assert "DRY RUN" in result.output


class TestConsolidateCommand:
    """Tests for consolidate command."""

    def test_consolidate_basic(self, cli_runner):
        """Test basic consolidate (placeholder)."""
        result = cli_runner.invoke(consolidate_command)
        assert result.exit_code == 0
        assert "not yet implemented" in result.output

    def test_consolidate_force(self, cli_runner):
        """Test consolidate with force flag."""
        result = cli_runner.invoke(consolidate_command, ["--force"])
        assert result.exit_code == 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in commands."""

    def test_redis_connection_failure(self, cli_runner, mock_settings):
        """Test handling of Redis connection failure."""
        failing_backend = AsyncMock()
        failing_backend.initialize = AsyncMock(return_value=False)

        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            async def failing_get():
                backend = failing_backend
                if not await backend.initialize():
                    raise Exception("Connection failed")
                return backend

            mock_get.return_value = failing_get()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                result = cli_runner.invoke(recall_command, ["test"])
                # Should handle error gracefully
                assert "Failed to initialize" in result.output or result.exit_code != 0

    def test_config_load_failure(self, cli_runner):
        """Test handling of config load failure."""
        with patch("gemma_cli.commands.rag_commands.load_config") as mock_load:
            mock_load.side_effect = FileNotFoundError("Config not found")

            with patch("gemma_cli.commands.rag_commands.Settings") as mock_settings_class:
                mock_settings_class.return_value = MagicMock()

                # Should fall back to default settings
                result = cli_runner.invoke(recall_command, ["test"])
                # Command should still attempt to run with defaults


# ============================================================================
# Output Formatting Tests
# ============================================================================


class TestOutputFormatting:
    """Tests for Rich output formatting."""

    def test_memory_entry_formatting(self):
        """Test memory entry table formatting."""
        from gemma_cli.commands.rag_commands import format_memory_entry

        entry = MagicMock(spec=MemoryEntry)
        entry.id = "test-id-12345678"
        entry.memory_type = "long_term"
        entry.importance = 0.85
        entry.access_count = 10
        entry.created_at = MagicMock()
        entry.created_at.strftime = MagicMock(return_value="2025-01-01 12:00:00")
        entry.content = "Test content"
        entry.tags = ["tag1", "tag2"]
        entry.similarity_score = 0.92

        table = format_memory_entry(entry)
        assert table is not None
        # Table should have formatted data

    def test_memory_stats_formatting(self):
        """Test memory statistics table formatting."""
        from gemma_cli.commands.rag_commands import format_memory_stats

        stats = {
            "working": 5,
            "short_term": 20,
            "long_term": 100,
            "episodic": 50,
            "semantic": 200,
            "total": 375,
        }

        table = format_memory_stats(stats)
        assert table is not None
        # Table should have all tiers


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for command workflows."""

    def test_store_and_recall_workflow(self, cli_runner, mock_rag_backend, mock_settings):
        """Test storing and then recalling a memory."""
        # Store a memory
        with patch("gemma_cli.commands.rag_commands.get_rag_backend") as mock_get:
            mock_get.return_value = asyncio.coroutine(lambda: mock_rag_backend)()

            with patch("gemma_cli.commands.rag_commands.get_settings") as mock_get_settings:
                mock_get_settings.return_value = mock_settings

                store_result = cli_runner.invoke(
                    store_command, ["Python uses duck typing", "--tier", "semantic"]
                )
                assert store_result.exit_code == 0

                # Recall it
                recall_result = cli_runner.invoke(recall_command, ["Python typing"])
                assert recall_result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
