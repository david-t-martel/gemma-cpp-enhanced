"""Unit tests for GemmaInterface class.

Tests cover:
- Initialization with valid and invalid paths
- Response generation with streaming
- Error handling and recovery
- Security validation (prompt length, forbidden characters)
- Edge cases (Unicode, long prompts, concurrent requests)
- Process cleanup and timeout handling

Target: 95% code coverage
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from gemma_cli.core.gemma import GemmaInterface


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_executable(tmp_path):
    """Create a temporary mock executable file."""
    exe_path = tmp_path / "gemma.exe"
    exe_path.touch()
    return str(exe_path)


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary mock model file."""
    model_path = tmp_path / "model.sbs"
    model_path.touch()
    return str(model_path)


@pytest.fixture
def mock_tokenizer_path(tmp_path):
    """Create a temporary mock tokenizer file."""
    tokenizer_path = tmp_path / "tokenizer.spm"
    tokenizer_path.touch()
    return str(tokenizer_path)


@pytest.fixture
def gemma_interface(mock_executable, mock_model_path):
    """Create a basic GemmaInterface instance."""
    return GemmaInterface(
        model_path=mock_model_path,
        gemma_executable=mock_executable,
        max_tokens=100,
        temperature=0.5,
    )


@pytest.fixture
def mock_process():
    """Create a mock asyncio subprocess."""
    process = AsyncMock()
    process.returncode = None
    process.stdout = AsyncMock()
    process.stderr = AsyncMock()
    process.wait = AsyncMock(return_value=0)
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests for GemmaInterface initialization."""

    def test_init_valid_paths(self, mock_executable, mock_model_path, mock_tokenizer_path):
        """Test initialization with valid paths."""
        interface = GemmaInterface(
            model_path=mock_model_path,
            tokenizer_path=mock_tokenizer_path,
            gemma_executable=mock_executable,
            max_tokens=2048,
            temperature=0.7,
        )

        assert interface.model_path == mock_model_path
        assert interface.tokenizer_path == mock_tokenizer_path
        assert interface.gemma_executable == mock_executable
        assert interface.max_tokens == 2048
        assert interface.temperature == 0.7
        assert interface.process is None
        assert interface.debug_mode is False

    def test_init_without_tokenizer(self, mock_executable, mock_model_path):
        """Test initialization without tokenizer (single-file model)."""
        interface = GemmaInterface(
            model_path=mock_model_path,
            tokenizer_path=None,
            gemma_executable=mock_executable,
        )

        assert interface.tokenizer_path is None
        assert interface.model_path == mock_model_path

    def test_init_invalid_executable(self, mock_model_path):
        """Test initialization with non-existent executable."""
        with pytest.raises(FileNotFoundError) as exc_info:
            GemmaInterface(
                model_path=mock_model_path,
                gemma_executable="/nonexistent/gemma.exe",
            )

        assert "Gemma executable not found" in str(exc_info.value)
        assert "/nonexistent/gemma.exe" in str(exc_info.value)

    def test_init_default_parameters(self, mock_executable, mock_model_path):
        """Test initialization with default parameters."""
        interface = GemmaInterface(
            model_path=mock_model_path,
            gemma_executable=mock_executable,
        )

        # Check defaults are set
        assert interface.max_tokens == 2048
        assert interface.temperature == 0.7

    def test_init_path_normalization(self, mock_executable, mock_model_path, mock_tokenizer_path):
        """Test that paths are normalized properly."""
        # Use paths with mixed separators
        model_with_forward = mock_model_path.replace("\\", "/")
        tokenizer_with_forward = mock_tokenizer_path.replace("\\", "/")

        interface = GemmaInterface(
            model_path=model_with_forward,
            tokenizer_path=tokenizer_with_forward,
            gemma_executable=mock_executable,
        )

        # Paths should be normalized to OS-specific format
        assert interface.model_path == mock_model_path
        assert interface.tokenizer_path == mock_tokenizer_path


# ============================================================================
# Command Building Tests
# ============================================================================


class TestCommandBuilding:
    """Tests for _build_command method."""

    def test_build_command_basic(self, gemma_interface):
        """Test basic command building."""
        cmd = gemma_interface._build_command("test prompt")

        assert gemma_interface.gemma_executable in cmd
        assert "--weights" in cmd
        assert gemma_interface.model_path in cmd
        assert "--max_generated_tokens" in cmd
        assert "100" in cmd
        assert "--temperature" in cmd
        assert "0.5" in cmd
        assert "--prompt" in cmd
        assert "test prompt" in cmd

    def test_build_command_with_tokenizer(self, mock_executable, mock_model_path, mock_tokenizer_path):
        """Test command building with tokenizer."""
        interface = GemmaInterface(
            model_path=mock_model_path,
            tokenizer_path=mock_tokenizer_path,
            gemma_executable=mock_executable,
        )

        cmd = interface._build_command("test")

        assert "--tokenizer" in cmd
        assert mock_tokenizer_path in cmd

    def test_build_command_without_tokenizer(self, gemma_interface):
        """Test command building without tokenizer."""
        cmd = gemma_interface._build_command("test")

        assert "--tokenizer" not in cmd

    def test_build_command_prompt_too_long(self, gemma_interface):
        """Test command building with prompt exceeding max length."""
        long_prompt = "x" * (GemmaInterface.MAX_PROMPT_LENGTH + 1)

        with pytest.raises(ValueError) as exc_info:
            gemma_interface._build_command(long_prompt)

        assert "Prompt exceeds maximum length" in str(exc_info.value)
        assert str(GemmaInterface.MAX_PROMPT_LENGTH) in str(exc_info.value)
        assert "command injection" in str(exc_info.value)

    def test_build_command_forbidden_null_byte(self, gemma_interface):
        """Test command building with null byte in prompt."""
        prompt_with_null = "test\x00prompt"

        with pytest.raises(ValueError) as exc_info:
            gemma_interface._build_command(prompt_with_null)

        assert "forbidden characters" in str(exc_info.value)
        assert "command injection" in str(exc_info.value)

    def test_build_command_forbidden_escape_sequence(self, gemma_interface):
        """Test command building with escape sequence in prompt."""
        prompt_with_escape = "test\x1bprompt"

        with pytest.raises(ValueError) as exc_info:
            gemma_interface._build_command(prompt_with_escape)

        assert "forbidden characters" in str(exc_info.value)
        assert "terminal manipulation" in str(exc_info.value)

    def test_build_command_multiple_forbidden_chars(self, gemma_interface):
        """Test command building with multiple forbidden characters."""
        prompt_with_multiple = "test\x00\x1bprompt"

        with pytest.raises(ValueError) as exc_info:
            gemma_interface._build_command(prompt_with_multiple)

        assert "forbidden characters" in str(exc_info.value)


# ============================================================================
# Response Generation Tests
# ============================================================================


class TestResponseGeneration:
    """Tests for generate_response method."""

    @pytest.mark.asyncio
    async def test_generate_response_success(self, gemma_interface, mock_process):
        """Test successful response generation."""
        # Mock process output
        mock_process.stdout.read = AsyncMock(side_effect=[
            b"Hello ",
            b"world!",
            b"",  # EOF
        ])
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test prompt")

        assert response == "Hello world!"
        assert gemma_interface.process is None  # Cleanup occurred

    @pytest.mark.asyncio
    async def test_generate_response_empty_prompt(self, gemma_interface, mock_process):
        """Test response generation with empty prompt."""
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("")

        # Should still work, just with empty prompt
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_response_with_streaming(self, gemma_interface, mock_process):
        """Test response generation with streaming callback."""
        # Mock process output
        mock_process.stdout.read = AsyncMock(side_effect=[
            b"chunk1 ",
            b"chunk2 ",
            b"chunk3",
            b"",  # EOF
        ])
        mock_process.wait = AsyncMock(return_value=0)

        streamed_chunks = []

        def stream_callback(chunk):
            streamed_chunks.append(chunk)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test", stream_callback)

        assert response == "chunk1 chunk2 chunk3"
        assert streamed_chunks == ["chunk1 ", "chunk2 ", "chunk3"]

    @pytest.mark.asyncio
    async def test_generate_response_process_failure(self, gemma_interface, mock_process):
        """Test response generation with process returning non-zero exit code."""
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.stderr.read = AsyncMock(return_value=b"Error: Model not found")
        mock_process.wait = AsyncMock(return_value=1)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        assert "Error" in response
        assert "Model not found" in response

    @pytest.mark.asyncio
    async def test_generate_response_unicode_handling(self, gemma_interface, mock_process):
        """Test response generation with Unicode characters."""
        # Mock Unicode output (emoji, accented characters, Chinese)
        unicode_text = "Hello 世界 café 🎉"
        mock_process.stdout.read = AsyncMock(side_effect=[
            unicode_text.encode("utf-8"),
            b"",  # EOF
        ])
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        assert response == unicode_text

    @pytest.mark.asyncio
    async def test_generate_response_invalid_utf8(self, gemma_interface, mock_process):
        """Test response generation with invalid UTF-8 bytes."""
        # Mock invalid UTF-8 sequence
        mock_process.stdout.read = AsyncMock(side_effect=[
            b"Valid text ",
            b"\xff\xfe",  # Invalid UTF-8
            b" more text",
            b"",  # EOF
        ])
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        # Should handle gracefully with errors='ignore'
        assert "Valid text" in response
        assert "more text" in response

    @pytest.mark.asyncio
    async def test_generate_response_max_size_exceeded(self, gemma_interface, mock_process):
        """Test response generation exceeding max response size."""
        # Create chunk that exceeds max size
        large_chunk = b"x" * (GemmaInterface.MAX_RESPONSE_SIZE + 1000)
        mock_process.stdout.read = AsyncMock(side_effect=[
            large_chunk,
            b"",  # EOF
        ])
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        assert "Error" in response
        assert "exceeded maximum size" in response

    @pytest.mark.asyncio
    async def test_generate_response_oserror(self, gemma_interface):
        """Test response generation with OSError."""
        with patch("asyncio.create_subprocess_exec", side_effect=OSError("Permission denied")):
            response = await gemma_interface.generate_response("test")

        assert "Error" in response
        assert "Permission denied" in response

    @pytest.mark.asyncio
    async def test_generate_response_valueerror(self, gemma_interface):
        """Test response generation with ValueError from command building."""
        # Trigger ValueError with forbidden character
        response = await gemma_interface.generate_response("test\x00prompt")

        assert "Error" in response
        assert "forbidden characters" in response

    @pytest.mark.asyncio
    async def test_generate_response_read_error(self, gemma_interface, mock_process):
        """Test response generation with read error."""
        # Mock OSError during read
        mock_process.stdout.read = AsyncMock(side_effect=OSError("Broken pipe"))
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        # Should handle gracefully and return whatever was read
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_response_debug_mode(self, gemma_interface, mock_process, capsys):
        """Test response generation with debug mode enabled."""
        gemma_interface.debug_mode = True
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await gemma_interface.generate_response("test")

        captured = capsys.readouterr()
        assert "Debug - Command:" in captured.out


# ============================================================================
# Process Cleanup Tests
# ============================================================================


class TestProcessCleanup:
    """Tests for process cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_process_already_finished(self, gemma_interface):
        """Test cleanup when process already finished."""
        # Mock process that already finished
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        gemma_interface.process = mock_proc

        await gemma_interface._cleanup_process()

        # Should not call terminate/kill
        mock_proc.terminate.assert_not_called()
        mock_proc.kill.assert_not_called()
        assert gemma_interface.process is None

    @pytest.mark.asyncio
    async def test_cleanup_process_running(self, gemma_interface):
        """Test cleanup of running process."""
        # Mock running process
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        gemma_interface.process = mock_proc

        await gemma_interface._cleanup_process()

        # Should terminate gracefully
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
        assert gemma_interface.process is None

    @pytest.mark.asyncio
    async def test_cleanup_process_timeout(self, gemma_interface):
        """Test cleanup when process doesn't terminate within timeout."""
        # Mock process that times out
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        gemma_interface.process = mock_proc

        # Patch wait_for to raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            await gemma_interface._cleanup_process()

        # Should force kill after timeout
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert gemma_interface.process is None

    @pytest.mark.asyncio
    async def test_cleanup_process_oserror(self, gemma_interface):
        """Test cleanup handling OSError."""
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock(side_effect=OSError("No such process"))
        gemma_interface.process = mock_proc

        # Should handle error gracefully
        await gemma_interface._cleanup_process()

        assert gemma_interface.process is None

    @pytest.mark.asyncio
    async def test_cleanup_process_process_lookup_error(self, gemma_interface):
        """Test cleanup handling ProcessLookupError."""
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock(side_effect=ProcessLookupError())
        gemma_interface.process = mock_proc

        # Should handle error gracefully
        await gemma_interface._cleanup_process()

        assert gemma_interface.process is None

    @pytest.mark.asyncio
    async def test_cleanup_process_no_process(self, gemma_interface):
        """Test cleanup when no process exists."""
        gemma_interface.process = None

        # Should not raise error
        await gemma_interface._cleanup_process()

        assert gemma_interface.process is None

    @pytest.mark.asyncio
    async def test_stop_generation(self, gemma_interface):
        """Test stop_generation calls cleanup."""
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        gemma_interface.process = mock_proc

        await gemma_interface.stop_generation()

        # Should have cleaned up process
        mock_proc.terminate.assert_called_once()
        assert gemma_interface.process is None


# ============================================================================
# Parameter Management Tests
# ============================================================================


class TestParameterManagement:
    """Tests for parameter setting and configuration."""

    def test_set_parameters_both(self, gemma_interface):
        """Test setting both max_tokens and temperature."""
        gemma_interface.set_parameters(max_tokens=4096, temperature=0.9)

        assert gemma_interface.max_tokens == 4096
        assert gemma_interface.temperature == 0.9

    def test_set_parameters_max_tokens_only(self, gemma_interface):
        """Test setting only max_tokens."""
        original_temp = gemma_interface.temperature
        gemma_interface.set_parameters(max_tokens=1024)

        assert gemma_interface.max_tokens == 1024
        assert gemma_interface.temperature == original_temp

    def test_set_parameters_temperature_only(self, gemma_interface):
        """Test setting only temperature."""
        original_tokens = gemma_interface.max_tokens
        gemma_interface.set_parameters(temperature=0.3)

        assert gemma_interface.max_tokens == original_tokens
        assert gemma_interface.temperature == 0.3

    def test_set_parameters_none(self, gemma_interface):
        """Test setting no parameters (no-op)."""
        original_tokens = gemma_interface.max_tokens
        original_temp = gemma_interface.temperature

        gemma_interface.set_parameters()

        assert gemma_interface.max_tokens == original_tokens
        assert gemma_interface.temperature == original_temp

    def test_get_config(self, mock_executable, mock_model_path, mock_tokenizer_path):
        """Test getting current configuration."""
        interface = GemmaInterface(
            model_path=mock_model_path,
            tokenizer_path=mock_tokenizer_path,
            gemma_executable=mock_executable,
            max_tokens=2048,
            temperature=0.7,
        )
        interface.debug_mode = True

        config = interface.get_config()

        assert config["model_path"] == mock_model_path
        assert config["tokenizer_path"] == mock_tokenizer_path
        assert config["executable"] == mock_executable
        assert config["max_tokens"] == 2048
        assert config["temperature"] == 0.7
        assert config["debug_mode"] is True

    def test_get_config_no_tokenizer(self, gemma_interface):
        """Test getting configuration without tokenizer."""
        config = gemma_interface.get_config()

        assert config["tokenizer_path"] is None


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_long_valid_prompt(self, gemma_interface, mock_process):
        """Test with maximum allowed prompt length."""
        # Create prompt at exactly max length
        max_prompt = "x" * GemmaInterface.MAX_PROMPT_LENGTH
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response(max_prompt)

        # Should succeed without error
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, gemma_interface, mock_process):
        """Test prompt with special characters (but not forbidden ones)."""
        special_prompt = "Test with 'quotes' and \"double quotes\" and <tags> & symbols!"
        mock_process.stdout.read = AsyncMock(return_value=b"response")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response(special_prompt)

        assert "response" in response

    @pytest.mark.asyncio
    async def test_multiline_prompt(self, gemma_interface, mock_process):
        """Test prompt with multiple lines."""
        multiline_prompt = "Line 1\nLine 2\rLine 3\r\nLine 4"
        mock_process.stdout.read = AsyncMock(return_value=b"response")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response(multiline_prompt)

        assert "response" in response

    @pytest.mark.asyncio
    async def test_empty_response_from_process(self, gemma_interface, mock_process):
        """Test when process returns empty response."""
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        assert response == ""

    @pytest.mark.asyncio
    async def test_response_exactly_at_max_size(self, gemma_interface, mock_process):
        """Test response at exactly max size boundary."""
        # Create response exactly at max size
        exact_size_response = b"x" * GemmaInterface.MAX_RESPONSE_SIZE
        mock_process.stdout.read = AsyncMock(side_effect=[
            exact_size_response,
            b"",  # EOF
        ])
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response = await gemma_interface.generate_response("test")

        # Should succeed at exact boundary
        assert len(response) == GemmaInterface.MAX_RESPONSE_SIZE

    def test_constants_are_defined(self):
        """Test that security constants are properly defined."""
        assert GemmaInterface.MAX_RESPONSE_SIZE == 10 * 1024 * 1024
        assert GemmaInterface.MAX_PROMPT_LENGTH == 50_000
        assert GemmaInterface.BUFFER_SIZE == 8192
        assert "\x00" in GemmaInterface.FORBIDDEN_CHARS
        assert "\x1b" in GemmaInterface.FORBIDDEN_CHARS


# ============================================================================
# Concurrent Request Tests
# ============================================================================


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""

    @pytest.mark.asyncio
    async def test_sequential_requests(self, gemma_interface, mock_process):
        """Test multiple sequential requests."""
        mock_process.stdout.read = AsyncMock(side_effect=[
            b"response1",
            b"",  # EOF for request 1
        ])
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response1 = await gemma_interface.generate_response("prompt1")
            assert "response1" in response1

        # Reset mock for second request
        mock_process.stdout.read = AsyncMock(side_effect=[
            b"response2",
            b"",  # EOF for request 2
        ])

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response2 = await gemma_interface.generate_response("prompt2")
            assert "response2" in response2

    @pytest.mark.asyncio
    async def test_cleanup_between_requests(self, gemma_interface, mock_process):
        """Test that cleanup happens between requests."""
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await gemma_interface.generate_response("test1")
            assert gemma_interface.process is None

            await gemma_interface.generate_response("test2")
            assert gemma_interface.process is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=gemma_cli.core.gemma", "--cov-report=term-missing"])
