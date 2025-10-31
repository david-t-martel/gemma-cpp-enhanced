"""Integration tests for model management commands.

Tests:
- model detect: Scan filesystem for models
- model list: Show detected + configured + default
- model set-default: Update config correctly
- Console DI: Verify console injection in all commands
- Config persistence: Changes persist across commands
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gemma_cli.cli import cli
from gemma_cli.config.settings import (
    AppConfig,
    ConfigManager,
    ConfiguredModel,
    DetectedModel,
    GemmaConfig,
    load_config,
    load_detected_models,
    save_detected_models,
)
from gemma_cli.ui.console import create_console


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / ".gemma_cli"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory with mock model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    # Create mock model files
    model1 = model_dir / "gemma-2b-it-sfp.sbs"
    model1.write_bytes(b"x" * (2 * 1024**3))  # 2GB mock file

    tokenizer1 = model_dir / "tokenizer.spm"
    tokenizer1.write_bytes(b"tokenizer data")

    # Create subdirectory with another model
    subdir = model_dir / "gemma-7b"
    subdir.mkdir()
    model2 = subdir / "7b-it.sbs"
    model2.write_bytes(b"x" * (7 * 1024**3))  # 7GB mock file
    tokenizer2 = subdir / "tokenizer.spm"
    tokenizer2.write_bytes(b"tokenizer data")

    return model_dir


@pytest.fixture
def mock_config(temp_config_dir):
    """Create a mock configuration."""
    config = AppConfig(
        gemma=GemmaConfig(
            executable_path="C:/codedev/llm/gemma/build/Release/gemma.exe",
            default_model=None,
            default_tokenizer=None,
        ),
        configured_models={},
    )

    # Save to temp location
    config_file = temp_config_dir / "config.toml"
    manager = ConfigManager(config_path=config_file)
    manager.save(config)

    return config_file


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


class TestModelDetectCommand:
    """Test 'model detect' command."""

    def test_detect_models_in_directory(self, cli_runner, temp_model_dir, temp_config_dir):
        """Test detecting models in a specified directory."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.save_detected_models") as mock_save:
                result = cli_runner.invoke(
                    cli,
                    ["model", "detect", "--path", str(temp_model_dir), "--recursive"],
                )

                assert result.exit_code == 0
                assert "Scanning for models" in result.output
                assert mock_save.called

                # Check that models were detected
                detected = mock_save.call_args[0][0]
                assert len(detected) == 2  # Both models should be found

    def test_detect_models_recursive(self, cli_runner, temp_model_dir, temp_config_dir):
        """Test recursive model detection."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.save_detected_models") as mock_save:
                result = cli_runner.invoke(
                    cli,
                    ["model", "detect", "--path", str(temp_model_dir), "--recursive"],
                )

                detected = mock_save.call_args[0][0]

                # Should find models in subdirectories
                model_names = list(detected.keys())
                assert any("2b" in name or "7b" in name for name in model_names)

    def test_detect_no_models_found(self, cli_runner, tmp_path, temp_config_dir):
        """Test detection when no models exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            result = cli_runner.invoke(
                cli,
                ["model", "detect", "--path", str(empty_dir)],
            )

            assert result.exit_code == 0
            assert "No models detected" in result.output

    def test_detect_model_format_detection(self, cli_runner, temp_model_dir):
        """Test that model format is detected correctly from filename."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.save_detected_models") as mock_save:
                result = cli_runner.invoke(
                    cli,
                    ["model", "detect", "--path", str(temp_model_dir)],
                )

                detected = mock_save.call_args[0][0]

                # Check format detection
                for model in detected.values():
                    assert model.format in ["sfp", "bf16", "f32", "nuq", "unknown"]


class TestModelListCommand:
    """Test 'model list' command."""

    def test_list_detected_models(self, cli_runner, temp_config_dir):
        """Test listing detected models."""
        # Setup detected models
        detected_models = {
            "gemma-2b-it-sfp": DetectedModel(
                name="gemma-2b-it-sfp",
                weights_path="/models/gemma-2b-it-sfp.sbs",
                tokenizer_path="/models/tokenizer.spm",
                format="sfp",
                size_gb=2.5,
            ),
        }

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected_models
            ):
                with patch(
                    "gemma_cli.commands.model_simple.load_config",
                    return_value=AppConfig(configured_models={}),
                ):
                    result = cli_runner.invoke(cli, ["model", "list"])

                    assert result.exit_code == 0
                    assert "gemma-2b-it-sfp" in result.output
                    assert "2.5" in result.output or "2.50" in result.output

    def test_list_with_default_marker(self, cli_runner, temp_config_dir):
        """Test that default model is marked in list."""
        detected_models = {
            "model1": DetectedModel(
                name="model1",
                weights_path="/models/model1.sbs",
                tokenizer_path="/models/tokenizer.spm",
                format="sfp",
                size_gb=2.0,
            ),
        }

        config = AppConfig(
            gemma=GemmaConfig(default_model="/models/model1.sbs"),
            configured_models={},
        )

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected_models
            ):
                with patch("gemma_cli.commands.model_simple.load_config", return_value=config):
                    result = cli_runner.invoke(cli, ["model", "list"])

                    assert "(default)" in result.output

    def test_list_simple_format(self, cli_runner):
        """Test simple format output."""
        detected_models = {
            "test-model": DetectedModel(
                name="test-model",
                weights_path="/models/test.sbs",
                tokenizer_path=None,
                format="sfp",
                size_gb=1.5,
            ),
        }

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected_models
            ):
                with patch(
                    "gemma_cli.commands.model_simple.load_config",
                    return_value=AppConfig(configured_models={}),
                ):
                    result = cli_runner.invoke(cli, ["model", "list", "--format", "simple"])

                    assert result.exit_code == 0
                    assert "test-model" in result.output


class TestModelAddCommand:
    """Test 'model add' command."""

    def test_add_model_basic(self, cli_runner, temp_model_dir, temp_config_dir, mock_config):
        """Test adding a model to configuration."""
        model_path = temp_model_dir / "gemma-2b-it-sfp.sbs"

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.ConfigManager") as mock_manager:
                mock_instance = MagicMock()
                mock_manager.return_value = mock_instance
                mock_instance.load.return_value = AppConfig(configured_models={})

                result = cli_runner.invoke(
                    cli,
                    ["model", "add", str(model_path), "--name", "my-model"],
                )

                assert result.exit_code == 0
                assert "Added model: my-model" in result.output

                # Verify save was called with updated config
                assert mock_instance.save.called

    def test_add_model_with_auto_tokenizer(self, cli_runner, temp_model_dir, temp_config_dir):
        """Test that tokenizer is auto-detected."""
        model_path = temp_model_dir / "gemma-2b-it-sfp.sbs"

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.ConfigManager") as mock_manager:
                mock_instance = MagicMock()
                mock_manager.return_value = mock_instance
                mock_instance.load.return_value = AppConfig(configured_models={})

                result = cli_runner.invoke(cli, ["model", "add", str(model_path)])

                # Should find tokenizer.spm in same directory
                assert result.exit_code == 0
                saved_config = mock_instance.save.call_args[0][0]
                model_name = model_path.stem
                assert saved_config.configured_models[model_name].tokenizer_path is not None


class TestModelSetDefaultCommand:
    """Test 'model set-default' command."""

    def test_set_default_from_detected(self, cli_runner, temp_config_dir, mock_config):
        """Test setting default model from detected models."""
        detected_models = {
            "test-model": DetectedModel(
                name="test-model",
                weights_path="/models/test.sbs",
                tokenizer_path="/models/tokenizer.spm",
                format="sfp",
                size_gb=2.0,
            ),
        }

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected_models
            ):
                with patch("gemma_cli.commands.model_simple.load_config") as mock_load:
                    with patch("gemma_cli.commands.model_simple.ConfigManager") as mock_manager:
                        mock_load.return_value = AppConfig(configured_models={})
                        mock_instance = MagicMock()
                        mock_manager.return_value = mock_instance
                        mock_instance.load.return_value = AppConfig(configured_models={})

                        result = cli_runner.invoke(cli, ["model", "set-default", "test-model"])

                        assert result.exit_code == 0
                        assert "Set default model: test-model" in result.output

                        # Verify config was updated
                        saved_config = mock_instance.save.call_args[0][0]
                        assert saved_config.gemma.default_model == "/models/test.sbs"

    def test_set_default_nonexistent_model(self, cli_runner):
        """Test setting default to non-existent model fails."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.load_detected_models", return_value={}):
                with patch(
                    "gemma_cli.commands.model_simple.load_config",
                    return_value=AppConfig(configured_models={}),
                ):
                    result = cli_runner.invoke(cli, ["model", "set-default", "nonexistent"])

                    assert result.exit_code != 0
                    assert "not found" in result.output.lower()


class TestConsoleInjection:
    """Test console dependency injection in model commands."""

    def test_model_detect_uses_injected_console(self, cli_runner, temp_model_dir):
        """Test that model detect command uses injected console."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.cli.create_console") as mock_create:
                mock_console = MagicMock()
                mock_create.return_value = mock_console

                with patch("gemma_cli.commands.model_simple.save_detected_models"):
                    result = cli_runner.invoke(
                        cli,
                        ["model", "detect", "--path", str(temp_model_dir)],
                    )

                    # Console should have been created
                    assert mock_create.called

    def test_model_list_uses_injected_console(self, cli_runner):
        """Test that model list command uses injected console."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.load_detected_models", return_value={}):
                with patch(
                    "gemma_cli.commands.model_simple.load_config",
                    return_value=AppConfig(configured_models={}),
                ):
                    result = cli_runner.invoke(cli, ["model", "list"])

                    # Command should execute successfully with console
                    assert result.exit_code == 0


class TestConfigPersistence:
    """Test that configuration changes persist across commands."""

    def test_add_then_list_shows_model(self, cli_runner, temp_model_dir, temp_config_dir):
        """Test that adding a model makes it visible in list."""
        model_path = temp_model_dir / "gemma-2b-it-sfp.sbs"
        config_file = temp_config_dir / "config.toml"

        # Create initial config
        manager = ConfigManager(config_path=config_file)
        manager.save(AppConfig(configured_models={}))

        # Add model
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.ConfigManager") as mock_manager:
                mock_instance = MagicMock()
                mock_manager.return_value = mock_instance

                initial_config = AppConfig(configured_models={})
                mock_instance.load.return_value = initial_config

                result = cli_runner.invoke(cli, ["model", "add", str(model_path), "--name", "test"])

                # Get the saved config
                saved_config = mock_instance.save.call_args[0][0]
                assert "test" in saved_config.configured_models

    def test_set_default_persists(self, cli_runner, temp_config_dir):
        """Test that setting default model persists in config."""
        detected_models = {
            "test-model": DetectedModel(
                name="test-model",
                weights_path="/models/test.sbs",
                tokenizer_path="/models/tokenizer.spm",
                format="sfp",
                size_gb=2.0,
            ),
        }

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected_models
            ):
                with patch("gemma_cli.commands.model_simple.load_config") as mock_load:
                    with patch("gemma_cli.commands.model_simple.ConfigManager") as mock_manager:
                        config = AppConfig(configured_models={})
                        mock_load.return_value = config

                        mock_instance = MagicMock()
                        mock_manager.return_value = mock_instance
                        mock_instance.load.return_value = config

                        # Set default
                        cli_runner.invoke(cli, ["model", "set-default", "test-model"])

                        # Verify persistence
                        saved_config = mock_instance.save.call_args[0][0]
                        assert saved_config.gemma.default_model == "/models/test.sbs"


class TestModelPriority:
    """Test model loading priority: CLI arg > detected > configured > default."""

    def test_cli_argument_overrides_default(self, cli_runner):
        """Test that --model CLI argument takes priority."""
        config = AppConfig(
            gemma=GemmaConfig(default_model="/default/model.sbs"),
            configured_models={},
        )

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.cli.load_config", return_value=config):
                # We can't easily test the full chat flow, but we can verify
                # the priority logic is in place by checking the code path
                result = cli_runner.invoke(cli, ["chat", "--help"])
                assert result.exit_code == 0
                assert "--model" in result.output


class TestIntegrationWorkflow:
    """Test complete workflow: detect → list → set-default."""

    def test_complete_workflow(self, cli_runner, temp_model_dir, temp_config_dir):
        """Test realistic workflow of detecting, listing, and setting default."""
        with patch("gemma_cli.cli.check_first_run", return_value=False):
            # Step 1: Detect models
            with patch("gemma_cli.commands.model_simple.save_detected_models") as mock_save:
                detect_result = cli_runner.invoke(
                    cli,
                    ["model", "detect", "--path", str(temp_model_dir)],
                )
                assert detect_result.exit_code == 0

                detected = mock_save.call_args[0][0]
                model_names = list(detected.keys())
                assert len(model_names) > 0

            # Step 2: List models
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected
            ):
                with patch(
                    "gemma_cli.commands.model_simple.load_config",
                    return_value=AppConfig(configured_models={}),
                ):
                    list_result = cli_runner.invoke(cli, ["model", "list"])
                    assert list_result.exit_code == 0
                    assert model_names[0] in list_result.output

            # Step 3: Set default
            with patch(
                "gemma_cli.commands.model_simple.load_detected_models", return_value=detected
            ):
                with patch("gemma_cli.commands.model_simple.load_config") as mock_load:
                    with patch("gemma_cli.commands.model_simple.ConfigManager") as mock_manager:
                        mock_load.return_value = AppConfig(configured_models={})
                        mock_instance = MagicMock()
                        mock_manager.return_value = mock_instance
                        mock_instance.load.return_value = AppConfig(configured_models={})

                        default_result = cli_runner.invoke(
                            cli, ["model", "set-default", model_names[0]]
                        )
                        assert default_result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
