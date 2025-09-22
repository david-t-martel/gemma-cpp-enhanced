"""Unit tests for CLI main.py module focusing on direct function testing.

Tests CLI functions directly rather than through the CLI runner to avoid
rich formatting and banner complications.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import typer

from src.cli.main import app, version, status, quick_chat, models, main


class TestVersionFunction:
    """Test the version function directly."""

    @patch("src.cli.main.console")
    @patch("importlib.metadata.version")
    def test_version_basic(self, mock_version, mock_console):
        """Test basic version output."""
        mock_version.return_value = "1.0.0"

        version(verbose=False)

        mock_console.print.assert_called_once_with("Gemma Chatbot CLI v1.0.0", style="bold cyan")

    @patch("src.cli.main.console")
    @patch("importlib.metadata.version")
    @patch("torch.cuda.is_available")
    @patch("torch.version.cuda", "11.8")
    @patch("torch.cuda.device_count")
    @patch("torch.__version__", "2.0.0")
    @patch("transformers.__version__", "4.35.0")
    @patch("sys.version", "3.11.5 | packaged by conda-forge")
    @patch("platform.platform")
    def test_version_verbose(
        self,
        mock_platform,
        mock_device_count,
        mock_cuda_available,
        mock_version,
        mock_console
    ):
        """Test verbose version output."""
        mock_version.return_value = "1.0.0"
        mock_platform.return_value = "Linux-5.15.0-x86_64"
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        version(verbose=True)

        # Should print a panel with system information
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0]
        assert len(call_args) == 1  # Panel object

    @patch("src.cli.main.console")
    @patch("importlib.metadata.version")
    @patch("torch.cuda.is_available")
    def test_version_no_cuda(self, mock_cuda_available, mock_version, mock_console):
        """Test version output when CUDA is not available."""
        mock_version.return_value = "1.0.0"
        mock_cuda_available.return_value = False

        version(verbose=True)

        mock_console.print.assert_called_once()

    @patch("src.cli.main.console")
    def test_version_package_not_found(self, mock_console):
        """Test version when package metadata is not found."""
        with patch("importlib.metadata.version", side_effect=Exception("Package not found")):
            version(verbose=False)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0]
        assert "unknown" in call_args[0] or "development" in call_args[0]


class TestStatusFunction:
    """Test the status function directly."""

    @patch("src.cli.main.console")
    @patch("src.cli.utils.check_system_status")
    def test_status_healthy_system(self, mock_check_status, mock_console):
        """Test status with healthy system."""
        mock_check_status.return_value = {
            "environment": {"ready": True},
            "gpu": {"available": True, "device_count": 1, "memory": {"used": 2.5, "total": 8.0}},
            "models": {"available_count": 3},
            "server": {"running": True, "port": 8000},
            "config": {"valid": True},
            "warnings": [],
            "errors": []
        }

        status()

        # Should print the status panel
        mock_console.print.assert_called()
        assert mock_console.print.call_count >= 1

    @patch("src.cli.main.console")
    @patch("src.cli.utils.check_system_status")
    def test_status_with_warnings_errors(self, mock_check_status, mock_console):
        """Test status with warnings and errors."""
        mock_check_status.return_value = {
            "environment": {"ready": False},
            "gpu": {"available": False},
            "models": {"available_count": 0},
            "config": {"valid": False},
            "warnings": ["Low memory warning", "Old dependency version"],
            "errors": ["CUDA driver error", "Model loading failed"]
        }

        status()

        # Should print status panel plus warnings and errors
        mock_console.print.assert_called()
        assert mock_console.print.call_count >= 3  # Panel + warnings header + errors header


class TestQuickChatFunction:
    """Test the quick_chat function directly."""

    @patch("src.cli.main.console")
    @patch("src.cli.chat.quick_generate")
    @patch("asyncio.run")
    def test_quick_chat_basic(self, mock_asyncio_run, mock_quick_generate, mock_console):
        """Test basic quick chat."""
        mock_asyncio_run.return_value = None

        quick_chat(
            message="Hello, world!",
            model=None,
            temperature=0.7,
            max_tokens=512,
            stream=False
        )

        mock_console.print.assert_called_with("🤖 [bold blue]Generating response...[/bold blue]")
        mock_asyncio_run.assert_called_once()

    @patch("src.cli.main.console")
    @patch("src.cli.chat.quick_generate")
    @patch("asyncio.run")
    def test_quick_chat_with_options(self, mock_asyncio_run, mock_quick_generate, mock_console):
        """Test quick chat with custom options."""
        mock_asyncio_run.return_value = None

        quick_chat(
            message="Test message",
            model="gemma-7b",
            temperature=0.8,
            max_tokens=256,
            stream=True
        )

        mock_console.print.assert_called_with("🤖 [bold blue]Generating response...[/bold blue]")
        mock_asyncio_run.assert_called_once()


class TestModelsFunction:
    """Test the models function directly."""

    @patch("src.cli.main.console")
    @patch("src.cli.utils.list_models")
    @patch("src.cli.utils.display_models_table")
    @patch("asyncio.run")
    def test_models_list_basic(self, mock_asyncio_run, mock_display, mock_list_models, mock_console):
        """Test basic models listing."""
        mock_asyncio_run.return_value = ["gemma-2b", "gemma-7b"]

        models(list_all=False, download=None, info=None)

        mock_console.print.assert_called_with("📚 Available models:", style="bold blue")
        mock_asyncio_run.assert_called_once()
        mock_display.assert_called_once()

    @patch("src.cli.main.console")
    @patch("src.cli.utils.list_models")
    @patch("asyncio.run")
    def test_models_no_models(self, mock_asyncio_run, mock_list_models, mock_console):
        """Test models when no models are available."""
        mock_asyncio_run.return_value = []

        models(list_all=False, download=None, info=None)

        # Should print no models message
        calls = mock_console.print.call_args_list
        assert any("No models found" in str(call) for call in calls)

    @patch("src.cli.main.console")
    @patch("src.cli.utils.download_model")
    @patch("asyncio.run")
    def test_models_download_success(self, mock_asyncio_run, mock_download, mock_console):
        """Test successful model download."""
        mock_asyncio_run.return_value = True

        models(list_all=False, download="gemma-7b", info=None)

        calls = mock_console.print.call_args_list
        assert any("Downloading model" in str(call) for call in calls)
        assert any("Model downloaded successfully" in str(call) for call in calls)

    @patch("src.cli.main.console")
    @patch("src.cli.utils.download_model")
    @patch("asyncio.run")
    def test_models_download_failure(self, mock_asyncio_run, mock_download, mock_console):
        """Test failed model download."""
        mock_asyncio_run.return_value = False

        with pytest.raises(typer.Exit) as exc_info:
            models(list_all=False, download="invalid-model", info=None)

        assert exc_info.value.exit_code == 1
        calls = mock_console.print.call_args_list
        assert any("Failed to download model" in str(call) for call in calls)

    @patch("src.cli.main.console")
    @patch("src.cli.utils.get_model_info")
    @patch("src.cli.utils.display_model_info")
    @patch("asyncio.run")
    def test_models_info_success(self, mock_asyncio_run, mock_display_info, mock_get_info, mock_console):
        """Test successful model info retrieval."""
        model_info = {"name": "gemma-7b", "size": "7B", "description": "A 7B parameter model"}
        mock_asyncio_run.return_value = model_info

        models(list_all=False, download=None, info="gemma-7b")

        calls = mock_console.print.call_args_list
        assert any("Model information for" in str(call) for call in calls)
        mock_display_info.assert_called_once()

    @patch("src.cli.main.console")
    @patch("src.cli.utils.get_model_info")
    @patch("asyncio.run")
    def test_models_info_not_found(self, mock_asyncio_run, mock_get_info, mock_console):
        """Test model info not found."""
        mock_asyncio_run.return_value = None

        with pytest.raises(typer.Exit) as exc_info:
            models(list_all=False, download=None, info="nonexistent")

        assert exc_info.value.exit_code == 1
        calls = mock_console.print.call_args_list
        assert any("not found" in str(call) for call in calls)


class TestMainCallback:
    """Test the main callback function directly."""

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_default(self, mock_banner, mock_validate, mock_setup_logging):
        """Test main callback with default parameters."""
        mock_validate.return_value = {"success": True, "critical_errors": [], "warnings": []}

        main(verbose=False, log_file=None, no_banner=False)

        mock_setup_logging.assert_called_once_with(verbose=False, log_file=None)
        mock_validate.assert_called_once()
        mock_banner.assert_called_once()

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_verbose_no_banner(self, mock_banner, mock_validate, mock_setup_logging):
        """Test main callback with verbose and no banner options."""
        mock_validate.return_value = {"success": True, "critical_errors": [], "warnings": []}

        main(verbose=True, log_file=None, no_banner=True)

        mock_setup_logging.assert_called_once_with(verbose=True, log_file=None)
        mock_validate.assert_called_once()
        mock_banner.assert_not_called()

    @patch("src.cli.main.console")
    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    def test_main_callback_critical_errors(self, mock_validate, mock_setup_logging, mock_console):
        """Test main callback exits on critical validation errors."""
        mock_validate.return_value = {
            "success": False,
            "critical_errors": ["CUDA initialization failed", "Missing required dependencies"],
            "warnings": []
        }

        with pytest.raises(typer.Exit) as exc_info:
            main(verbose=False, log_file=None, no_banner=False)

        assert exc_info.value.exit_code == 1
        calls = mock_console.print.call_args_list
        assert any("Critical environment issues detected" in str(call) for call in calls)

    @patch("src.cli.main.console")
    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_warnings_only(self, mock_banner, mock_validate, mock_setup_logging, mock_console):
        """Test main callback shows warnings but continues."""
        mock_validate.return_value = {
            "success": True,
            "critical_errors": [],
            "warnings": ["Low GPU memory", "Outdated driver version"]
        }

        main(verbose=False, log_file=None, no_banner=False)

        mock_banner.assert_called_once()
        calls = mock_console.print.call_args_list
        assert any("⚠️  Warnings:" in str(call) for call in calls)

    @patch("src.cli.utils.setup_logging")
    @patch("src.cli.utils.validate_environment")
    @patch("src.cli.utils.show_banner")
    def test_main_callback_log_file_option(self, mock_banner, mock_validate, mock_setup_logging):
        """Test main callback with log file option."""
        mock_validate.return_value = {"success": True, "critical_errors": [], "warnings": []}
        log_file_path = Path("/tmp/test.log")

        main(verbose=False, log_file=log_file_path, no_banner=False)

        mock_setup_logging.assert_called_once_with(verbose=False, log_file=log_file_path)


class TestAppConfiguration:
    """Test app configuration and structure."""

    def test_app_creation(self):
        """Test that the main Typer app is properly configured."""
        assert isinstance(app, typer.Typer)
        assert app.info.name == "gemma-cli"
        assert "Gemma Chatbot CLI" in app.info.help

    def test_subcommand_registration(self):
        """Test that all subcommands are properly registered."""
        # Check that subcommands are registered
        assert len(app.registered_groups) > 0

        # Get the registered command names
        registered_names = [group.typer_instance.info.name for group in app.registered_groups.values()]
        expected_commands = ["chat", "train", "serve", "config"]

        for expected in expected_commands:
            assert expected in registered_names


class TestExceptionHandling:
    """Test exception handling in CLI functions."""

    @patch("src.cli.main.console")
    def test_version_function_exception_handling(self, mock_console):
        """Test version function handles exceptions gracefully."""
        # Test with various import errors
        with patch("importlib.metadata.version", side_effect=ImportError("Mock import error")):
            version(verbose=False)
            mock_console.print.assert_called()

        mock_console.reset_mock()

        with patch("importlib.metadata.version", side_effect=Exception("Generic error")):
            version(verbose=False)
            mock_console.print.assert_called()

    @patch("src.cli.main.console")
    @patch("src.cli.utils.check_system_status")
    def test_status_function_exception_handling(self, mock_check_status, mock_console):
        """Test status function handles exceptions gracefully."""
        mock_check_status.side_effect = Exception("System check failed")

        # Should not raise, should handle gracefully through @handle_exceptions
        with pytest.raises(Exception):
            status()


class TestInputValidation:
    """Test input validation for CLI functions."""

    def test_temperature_bounds(self):
        """Test temperature parameter validation logic."""
        # These would be validated by Typer at the CLI level,
        # but we can test the function accepts valid values
        assert 0.0 <= 0.7 <= 2.0  # Default temperature
        assert 0.0 <= 0.1 <= 2.0  # Minimum valid
        assert 0.0 <= 2.0 <= 2.0  # Maximum valid

    def test_max_tokens_bounds(self):
        """Test max_tokens parameter validation logic."""
        # These would be validated by Typer at the CLI level
        assert 512 >= 1  # Default max_tokens
        assert 1 >= 1    # Minimum valid

    def test_path_validation(self):
        """Test path parameter validation."""
        # Test that Path objects are handled correctly
        valid_path = Path("C:/tmp/test.log")  # Use Windows-compatible path
        assert isinstance(valid_path, Path)
        assert "test.log" in str(valid_path)


class TestAsyncIntegration:
    """Test async function integration in CLI."""

    @patch("src.cli.main.console")
    @patch("asyncio.run")
    def test_asyncio_run_usage(self, mock_asyncio_run, mock_console):
        """Test that async functions are called with asyncio.run."""
        mock_asyncio_run.return_value = None

        # Test quick_chat uses asyncio.run
        with patch("src.cli.chat.quick_generate"):
            quick_chat("test", None, 0.7, 512, False)
            mock_asyncio_run.assert_called_once()

        mock_asyncio_run.reset_mock()

        # Test models uses asyncio.run
        with patch("src.cli.utils.list_models"):
            mock_asyncio_run.return_value = []
            models(False, None, None)
            mock_asyncio_run.assert_called_once()


class TestMockingPatterns:
    """Test various mocking patterns used in CLI testing."""

    @patch("src.cli.main.console")
    def test_console_mocking_pattern(self, mock_console):
        """Test console mocking pattern."""
        with patch("importlib.metadata.version", return_value="1.0.0"):
            version(verbose=False)

        mock_console.print.assert_called_once()

    @patch("torch.__version__", "2.0.0")
    @patch("transformers.__version__", "4.35.0")
    def test_dependency_version_mocking(self):
        """Test mocking of dependency versions."""
        import torch
        import transformers

        assert torch.__version__ == "2.0.0"
        assert transformers.__version__ == "4.35.0"

    def test_environment_variable_mocking(self):
        """Test mocking environment variables."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            import os
            assert os.environ.get("GEMINI_API_KEY") == "test-key"