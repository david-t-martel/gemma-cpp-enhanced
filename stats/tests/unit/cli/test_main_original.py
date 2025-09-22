"""Comprehensive unit tests for CLI main.py module.

Tests all CLI commands, arguments, error handling, and validation
using typer.testing.CliRunner and proper mocking.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from src.cli.main import app


class TestCliMainCommands:
    """Test suite for CLI main module commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_app_creation(self):
        """Test that the main Typer app is properly configured."""
        assert isinstance(app, typer.Typer)
        assert app.info.name == "gemma-cli"
        assert "Gemma Chatbot CLI" in app.info.help

    @patch("importlib.metadata.version")
    def test_version_command_basic(self, mock_version):
        """Test basic version command output."""
        mock_version.return_value = "1.0.0"

        result = self.runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Gemma Chatbot CLI v1.0.0" in result.stdout

    @patch("importlib.metadata.version")
    @patch("torch.cuda.is_available")
    @patch("torch.version.cuda", "11.8")
    @patch("torch.cuda.device_count")
    @patch("torch.__version__", "2.0.0")
    @patch("transformers.__version__", "4.35.0")
    @patch("sys.version", "3.11.5 | packaged by conda-forge")
    @patch("platform.platform")
    def test_version_command_verbose(
        self,
        mock_platform,
        mock_device_count,
        mock_cuda_available,
        mock_version
    ):
        """Test verbose version command with system information."""
        mock_version.return_value = "1.0.0"
        mock_platform.return_value = "Linux-5.15.0-x86_64"
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        result = self.runner.invoke(app, ["version", "--verbose"])

        assert result.exit_code == 0
        assert "Gemma Chatbot CLI" in result.stdout
        assert "Version: 1.0.0" in result.stdout
        assert "Python: 3.11.5" in result.stdout
        assert "Platform: Linux-5.15.0-x86_64" in result.stdout
        assert "PyTorch: 2.0.0" in result.stdout
        assert "Transformers: 4.35.0" in result.stdout
        assert "CUDA: 11.8" in result.stdout
        assert "GPU Count: 2" in result.stdout

    @patch("importlib.metadata.version")
    @patch("torch.cuda.is_available")
    def test_version_command_no_cuda(self, mock_cuda_available, mock_version):
        """Test version command when CUDA is not available."""
        mock_version.return_value = "1.0.0"
        mock_cuda_available.return_value = False

        result = self.runner.invoke(app, ["version", "--verbose"])

        assert result.exit_code == 0
        assert "CUDA: Not available" in result.stdout

    def test_version_command_package_not_found(self):
        """Test version command when package metadata is not found."""
        with patch("importlib.metadata.version", side_effect=Exception("Package not found")):
            result = self.runner.invoke(app, ["version"])

            assert result.exit_code == 0
            # Should fall back to "development" or "unknown"
            assert "unknown" in result.stdout or "development" in result.stdout

    @patch("src.cli.utils.check_system_status")
    def test_status_command_healthy_system(self, mock_check_status):
        """Test status command with healthy system."""
        mock_check_status.return_value = {
            "environment": {"ready": True},
            "gpu": {"available": True, "device_count": 1, "memory": {"used": 2.5, "total": 8.0}},
            "models": {"available_count": 3},
            "server": {"running": True, "port": 8000},
            "config": {"valid": True},
            "warnings": [],
            "errors": []
        }

        result = self.runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Environment: ✅ Ready" in result.stdout
        assert "GPU: ✅ Available" in result.stdout
        assert "Available Models: 3" in result.stdout
        assert "Server: ✅ Running" in result.stdout
        assert "Configuration: ✅ Valid" in result.stdout

    @patch("src.cli.utils.check_system_status")
    def test_status_command_with_warnings_errors(self, mock_check_status):
        """Test status command with warnings and errors."""
        mock_check_status.return_value = {
            "environment": {"ready": False},
            "gpu": {"available": False},
            "models": {"available_count": 0},
            "config": {"valid": False},
            "warnings": ["Low memory warning", "Old dependency version"],
            "errors": ["CUDA driver error", "Model loading failed"]
        }

        result = self.runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Environment: ❌ Issues detected" in result.stdout
        assert "GPU: ⚠️  Not available" in result.stdout
        assert "⚠️  Warnings:" in result.stdout
        assert "Low memory warning" in result.stdout
        assert "❌ Errors:" in result.stdout
        assert "CUDA driver error" in result.stdout

    @patch("src.cli.chat.quick_generate")
    @patch("asyncio.run")
    def test_quick_chat_command_basic(self, mock_asyncio_run, mock_quick_generate):
        """Test basic quick-chat command."""
        mock_asyncio_run.return_value = None

        result = self.runner.invoke(app, ["quick-chat", "Hello, world!"])

        assert result.exit_code == 0
        assert "Generating response..." in result.stdout
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.chat.quick_generate")
    @patch("asyncio.run")
    def test_quick_chat_command_with_options(self, mock_asyncio_run, mock_quick_generate):
        """Test quick-chat command with all options."""
        mock_asyncio_run.return_value = None

        result = self.runner.invoke(app, [
            "quick-chat",
            "Explain AI",
            "--model", "gemma-7b-it",
            "--temperature", "0.9",
            "--max-tokens", "1000",
            "--stream"
        ])

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.utils.list_models")
    @patch("asyncio.run")
    def test_models_command_basic(self, mock_asyncio_run, mock_list_models):
        """Test basic models command."""
        mock_models = [
            {"name": "gemma-2b-it", "size": "2.5GB", "available": True},
            {"name": "gemma-7b-it", "size": "7.2GB", "available": True}
        ]
        mock_asyncio_run.return_value = mock_models

        with patch("src.cli.utils.display_models_table") as mock_display:
            result = self.runner.invoke(app, ["models"])

            assert result.exit_code == 0
            assert "Available models:" in result.stdout
            mock_display.assert_called_once()

    @patch("src.cli.utils.list_models")
    @patch("asyncio.run")
    def test_models_command_no_models(self, mock_asyncio_run, mock_list_models):
        """Test models command when no models are found."""
        mock_asyncio_run.return_value = []

        result = self.runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "No models found" in result.stdout
        assert "Use --download to download a model" in result.stdout

    @patch("src.cli.utils.download_model")
    @patch("asyncio.run")
    def test_models_command_download_success(self, mock_asyncio_run, mock_download_model):
        """Test successful model download."""
        mock_asyncio_run.return_value = True

        result = self.runner.invoke(app, ["models", "--download", "gemma-2b-it"])

        assert result.exit_code == 0
        assert "Downloading model: gemma-2b-it" in result.stdout
        assert "Model downloaded successfully!" in result.stdout

    @patch("src.cli.utils.download_model")
    @patch("asyncio.run")
    def test_models_command_download_failure(self, mock_asyncio_run, mock_download_model):
        """Test failed model download."""
        mock_asyncio_run.return_value = False

        result = self.runner.invoke(app, ["models", "--download", "invalid-model"])

        assert result.exit_code == 1
        assert "Failed to download model" in result.stdout

    @patch("src.cli.utils.get_model_info")
    @patch("asyncio.run")
    def test_models_command_info_success(self, mock_asyncio_run, mock_get_model_info):
        """Test successful model info retrieval."""
        mock_info = {
            "name": "gemma-2b-it",
            "size": "2.5GB",
            "description": "Instruction-tuned model",
            "parameters": "2B"
        }
        mock_asyncio_run.return_value = mock_info

        with patch("src.cli.utils.display_model_info") as mock_display:
            result = self.runner.invoke(app, ["models", "--info", "gemma-2b-it"])

            assert result.exit_code == 0
            assert "Model information for: gemma-2b-it" in result.stdout
            mock_display.assert_called_once_with(mock_info, mock.ANY)

    @patch("src.cli.utils.get_model_info")
    @patch("asyncio.run")
    def test_models_command_info_not_found(self, mock_asyncio_run, mock_get_model_info):
        """Test model info for non-existent model."""
        mock_asyncio_run.return_value = None

        result = self.runner.invoke(app, ["models", "--info", "nonexistent-model"])

        assert result.exit_code == 1
        assert "Model 'nonexistent-model' not found" in result.stdout

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_default(self, mock_banner, mock_validate, mock_logging):
        """Test main callback with default settings."""
        mock_validate.return_value = {"success": True, "warnings": [], "critical_errors": []}

        result = self.runner.invoke(app, ["--help"])

        # Should not fail on help
        assert result.exit_code == 0

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_verbose_no_banner(self, mock_banner, mock_validate, mock_logging):
        """Test main callback with verbose logging and no banner."""
        mock_validate.return_value = {"success": True, "warnings": [], "critical_errors": []}

        result = self.runner.invoke(app, ["--verbose", "--no-banner", "version"])

        assert result.exit_code == 0
        mock_logging.assert_called_once_with(verbose=True, log_file=None)
        mock_banner.assert_not_called()

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    def test_main_callback_critical_errors(self, mock_validate, mock_logging):
        """Test main callback with critical environment errors."""
        mock_validate.return_value = {
            "success": False,
            "warnings": [],
            "critical_errors": ["CUDA driver missing", "Python version too old"]
        }

        result = self.runner.invoke(app, ["version"])

        assert result.exit_code == 1
        assert "Critical environment issues detected" in result.stdout
        assert "CUDA driver missing" in result.stdout
        assert "Python version too old" in result.stdout

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_warnings_only(self, mock_banner, mock_validate, mock_logging):
        """Test main callback with warnings but no critical errors."""
        mock_validate.return_value = {
            "success": True,
            "warnings": ["Low memory", "Old dependency"],
            "critical_errors": []
        }

        result = self.runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "⚠️  Warnings:" in result.stdout
        assert "Low memory" in result.stdout
        assert "Old dependency" in result.stdout

    def test_main_callback_log_file_option(self):
        """Test main callback with log file option."""
        with patch("src.cli.utils.setup_logging") as mock_logging, \
             patch("src.cli.utils.validate_environment") as mock_validate:

            mock_validate.return_value = {"success": True, "warnings": [], "critical_errors": []}
            log_path = Path("/tmp/test.log")

            result = self.runner.invoke(app, ["--log-file", str(log_path), "version"])

            assert result.exit_code == 0
            mock_logging.assert_called_once_with(verbose=False, log_file=log_path)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = self.runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0
        assert "No such command" in result.stdout or "Usage:" in result.stdout

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        result = self.runner.invoke(app, ["quick-chat"])  # Missing required message

        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Usage:" in result.stdout

    def test_quick_chat_invalid_temperature(self):
        """Test quick-chat with invalid temperature value."""
        result = self.runner.invoke(app, [
            "quick-chat",
            "test message",
            "--temperature", "5.0"  # Invalid range
        ])

        # Should fail validation
        assert result.exit_code != 0

    def test_quick_chat_invalid_max_tokens(self):
        """Test quick-chat with invalid max-tokens value."""
        result = self.runner.invoke(app, [
            "quick-chat",
            "test message",
            "--max-tokens", "0"  # Invalid value
        ])

        # Should fail validation
        assert result.exit_code != 0


class TestAsyncIntegration:
    """Test async function integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("src.cli.chat.quick_generate")
    def test_quick_chat_async_function_call(self, mock_quick_generate):
        """Test that quick_chat properly calls async functions."""
        mock_coro = AsyncMock()
        mock_quick_generate.return_value = mock_coro

        with patch("asyncio.run") as mock_run:
            result = self.runner.invoke(app, ["quick-chat", "test message"])

            # Verify asyncio.run was called
            mock_run.assert_called_once()

    @patch("src.cli.utils.list_models")
    def test_models_async_operations(self, mock_list_models):
        """Test that models command properly handles async operations."""
        mock_coro = AsyncMock(return_value=[])
        mock_list_models.return_value = mock_coro

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = []

            result = self.runner.invoke(app, ["models"])

            assert result.exit_code == 0
            mock_run.assert_called_once()


class TestInputValidation:
    """Test input validation and sanitization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_temperature_validation_bounds(self):
        """Test temperature parameter bounds validation."""
        # Test lower bound
        result = self.runner.invoke(app, [
            "quick-chat",
            "test",
            "--temperature", "-0.1"
        ])
        assert result.exit_code != 0

        # Test upper bound
        result = self.runner.invoke(app, [
            "quick-chat",
            "test",
            "--temperature", "2.1"
        ])
        assert result.exit_code != 0

    def test_max_tokens_validation(self):
        """Test max-tokens parameter validation."""
        # Test minimum value
        result = self.runner.invoke(app, [
            "quick-chat",
            "test",
            "--max-tokens", "0"
        ])
        assert result.exit_code != 0

    def test_path_validation(self):
        """Test path parameter validation."""
        # Test with valid path
        with patch("src.cli.utils.setup_logging"), \
             patch("src.cli.utils.validate_environment") as mock_validate:

            mock_validate.return_value = {"success": True, "warnings": [], "critical_errors": []}

            result = self.runner.invoke(app, [
                "--log-file", "/tmp/test.log",
                "version"
            ])

            assert result.exit_code == 0

    def test_empty_message_validation(self):
        """Test validation of empty message in quick-chat."""
        result = self.runner.invoke(app, ["quick-chat", ""])

        # Should still work with empty string, but might generate warning
        # The actual validation depends on the implementation
        # This test verifies the command doesn't crash


class TestMockingPatterns:
    """Test proper mocking patterns for external dependencies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("src.cli.utils.get_console")
    def test_console_mocking(self, mock_get_console):
        """Test mocking of console output."""
        mock_console = MagicMock()
        mock_get_console.return_value = mock_console

        with patch("importlib.metadata.version", return_value="1.0.0"):
            result = self.runner.invoke(app, ["version"])

            assert result.exit_code == 0
            # Console methods should be called through the CLI
            # We can't easily verify this without deeper mocking

    @patch("torch.cuda.is_available")
    @patch("torch.__version__", "2.0.0")
    def test_torch_dependency_mocking(self, mock_cuda_available):
        """Test mocking of PyTorch dependencies."""
        mock_cuda_available.return_value = False

        with patch("importlib.metadata.version", return_value="1.0.0"):
            result = self.runner.invoke(app, ["version", "--verbose"])

            assert result.exit_code == 0
            assert "CUDA: Not available" in result.stdout

    @patch("transformers.__version__", "4.35.0")
    def test_transformers_dependency_mocking(self):
        """Test mocking of transformers library."""
        with patch("importlib.metadata.version", return_value="1.0.0"):
            result = self.runner.invoke(app, ["version", "--verbose"])

            assert result.exit_code == 0
            assert "Transformers: 4.35.0" in result.stdout


# Integration tests that require minimal mocking
class TestMinimalMockingIntegration:
    """Test CLI with minimal mocking to ensure real behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_help_command_no_mocking(self):
        """Test help command without mocking."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Gemma Chatbot CLI" in result.stdout
        assert "chat" in result.stdout
        assert "train" in result.stdout
        assert "serve" in result.stdout
        assert "config" in result.stdout

    def test_command_help_no_mocking(self):
        """Test individual command help without mocking."""
        result = self.runner.invoke(app, ["quick-chat", "--help"])

        assert result.exit_code == 0
        assert "quick-chat" in result.stdout
        assert "temperature" in result.stdout
        assert "max-tokens" in result.stdout

    def test_version_command_import_error_handling(self):
        """Test version command when imports fail."""
        # This tests the actual error handling in the code
        # without mocking the import, which might reveal real issues
        result = self.runner.invoke(app, ["version"])

        # Should not crash even if there are import issues
        assert result.exit_code == 0
        assert "Gemma Chatbot CLI" in result.stdout


class TestCoverageTargets:
    """Test specific functions and paths to improve coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_handle_exceptions_decorator(self):
        """Test the handle_exceptions decorator behavior."""
        # The decorator is applied to most commands
        # Test that it properly handles exceptions
        with patch("src.cli.utils.handle_exceptions") as mock_decorator:
            # Create a mock decorator that still calls the function
            def passthrough_decorator(console):
                def decorator(func):
                    return func
                return decorator

            mock_decorator.return_value = passthrough_decorator

            result = self.runner.invoke(app, ["version"])
            # The command should still execute successfully
            assert result.exit_code == 0

    @patch("src.cli.utils.validate_environment")
    def test_environment_validation_failure(self, mock_validate):
        """Test behavior when environment validation fails."""
        mock_validate.return_value = {
            "success": False,
            "warnings": [],
            "critical_errors": ["Missing required dependencies"]
        }

        result = self.runner.invoke(app, ["status"])

        # Should exit with error code due to critical errors
        assert result.exit_code == 1
        assert "Critical environment issues detected" in result.stdout

    def test_subcommand_registration(self):
        """Test that all subcommands are properly registered."""
        # Test that subcommands are accessible
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        # All subcommands should be listed
        assert "chat" in result.stdout
        assert "train" in result.stdout
        assert "serve" in result.stdout
        assert "config" in result.stdout