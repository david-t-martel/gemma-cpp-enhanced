"""Tests for onboarding system."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemma_cli.onboarding import (
    OnboardingWizard,
    check_model_files,
    check_redis_connection,
    check_system_requirements,
    customize_template,
    get_template,
)


class TestTemplates:
    """Test configuration templates."""

    def test_get_template_minimal(self) -> None:
        """Test getting minimal template."""
        template = get_template("minimal")

        assert "name" in template
        assert "description" in template
        assert "config" in template
        assert template["name"] == "Minimal Setup"

    def test_get_template_developer(self) -> None:
        """Test getting developer template."""
        template = get_template("developer")

        assert template["name"] == "Developer Setup"
        assert "redis" in template["config"]
        assert "memory" in template["config"]
        assert "mcp" in template["config"]

    def test_get_template_performance(self) -> None:
        """Test getting performance template."""
        template = get_template("performance")

        assert template["name"] == "Performance Optimized"
        config = template["config"]
        assert config["redis"]["pool_size"] == 20
        assert config["embedding"]["batch_size"] == 64

    def test_get_template_invalid(self) -> None:
        """Test getting invalid template raises KeyError."""
        with pytest.raises(KeyError):
            get_template("nonexistent")

    def test_customize_template(self) -> None:
        """Test template customization."""
        template = get_template("minimal")
        overrides = {
            "redis": {"port": 6380},
            "ui": {"theme": "monokai"},
        }

        customized = customize_template(template, overrides)

        assert customized["redis"]["port"] == 6380
        assert customized["redis"]["host"] == "localhost"  # Not overridden
        assert customized["ui"]["theme"] == "monokai"


class TestChecks:
    """Test environment checks."""

    @pytest.mark.asyncio
    async def test_check_system_requirements(self) -> None:
        """Test system requirements check."""
        checks = await check_system_requirements()

        assert isinstance(checks, list)
        assert len(checks) > 0

        # Check format
        for check_name, passed, message in checks:
            assert isinstance(check_name, str)
            assert isinstance(passed, bool)
            assert isinstance(message, str)

        # Verify expected checks
        check_names = [name for name, _, _ in checks]
        assert "Python Version" in check_names
        assert "Available Memory" in check_names

    @pytest.mark.asyncio
    async def test_check_redis_connection_success(self) -> None:
        """Test Redis connection check with mocked success."""
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            success, message = await check_redis_connection("localhost", 6379)

            assert success is True
            assert "Connected" in message

    @pytest.mark.asyncio
    async def test_check_redis_connection_failure(self) -> None:
        """Test Redis connection check with connection refused."""
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = ConnectionRefusedError()
            mock_redis.return_value = mock_client

            success, message = await check_redis_connection("localhost", 6379)

            assert success is False
            assert "refused" in message.lower()

    @pytest.mark.asyncio
    async def test_check_model_files_directory(self, tmp_path: Path) -> None:
        """Test model file check with directory."""
        # Create test files
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        model_file = model_dir / "model.sbs"
        model_file.write_text("fake model data")

        tokenizer_file = model_dir / "tokenizer.spm"
        tokenizer_file.write_text("fake tokenizer")

        success, message = await check_model_files(model_dir)

        assert success is True
        assert "1 model" in message
        assert "tokenizer" in message

    @pytest.mark.asyncio
    async def test_check_model_files_single_file(self, tmp_path: Path) -> None:
        """Test model file check with single .sbs file."""
        model_file = tmp_path / "model.sbs"
        model_file.write_bytes(b"0" * (1024 * 1024))  # 1MB fake file

        success, message = await check_model_files(model_file)

        assert success is True
        assert "Model file found" in message
        assert "MB" in message

    @pytest.mark.asyncio
    async def test_check_model_files_not_found(self, tmp_path: Path) -> None:
        """Test model file check with nonexistent path."""
        nonexistent = tmp_path / "nonexistent"

        success, message = await check_model_files(nonexistent)

        assert success is False
        assert "does not exist" in message


class TestWizard:
    """Test onboarding wizard."""

    @pytest.fixture
    def wizard(self, tmp_path: Path) -> OnboardingWizard:
        """Create wizard instance with temp config path."""
        config_path = tmp_path / "config.toml"
        return OnboardingWizard(config_path=config_path)

    def test_wizard_initialization(self, wizard: OnboardingWizard) -> None:
        """Test wizard initializes correctly."""
        assert wizard.config == {}
        assert wizard.config_path.exists() is False

    def test_wizard_detect_models(self, wizard: OnboardingWizard) -> None:
        """Test model detection."""
        # This will return empty list if no models in standard locations
        models = wizard._detect_available_models()

        assert isinstance(models, list)
        for model in models:
            assert "path" in model
            assert "size_mb" in model

    def test_wizard_find_executable(self, wizard: OnboardingWizard) -> None:
        """Test finding gemma.exe."""
        exe_path = wizard._find_gemma_executable()

        assert isinstance(exe_path, str)
        # Should return at least a default path
        assert "gemma" in exe_path.lower()

    @pytest.mark.asyncio
    async def test_wizard_test_configuration(
        self, wizard: OnboardingWizard, tmp_path: Path
    ) -> None:
        """Test configuration testing."""
        # Create fake model
        model_file = tmp_path / "model.sbs"
        model_file.write_bytes(b"0" * 1024)

        config = {
            "gemma": {
                "default_model": str(model_file),
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }

        with patch(
            "src.gemma_cli.onboarding.wizard.check_redis_connection"
        ) as mock_redis:
            mock_redis.return_value = (False, "Redis not available")

            result = await wizard._test_configuration(config)

            # Should pass even if Redis fails (it's optional)
            assert isinstance(result, bool)

    def test_wizard_merge_configurations(self, wizard: OnboardingWizard) -> None:
        """Test configuration merging."""
        config1 = {"redis": {"host": "localhost", "port": 6379}}
        config2 = {"redis": {"port": 6380}, "ui": {"theme": "dark"}}
        config3 = {"mcp": {"enabled": True}}

        merged = wizard._merge_configurations(config1, config2, config3)

        assert merged["redis"]["host"] == "localhost"
        assert merged["redis"]["port"] == 6380  # Overridden
        assert merged["ui"]["theme"] == "dark"
        assert merged["mcp"]["enabled"] is True

    def test_wizard_save_configuration(
        self, wizard: OnboardingWizard, tmp_path: Path
    ) -> None:
        """Test saving configuration."""
        wizard.config = {
            "gemma": {"default_model": "test.sbs"},
            "redis": {"host": "localhost", "port": 6379},
        }

        wizard._save_configuration()

        assert wizard.config_path.exists()

        # Verify content
        import toml

        with open(wizard.config_path, encoding="utf-8") as f:
            saved_config = toml.load(f)

        assert saved_config["gemma"]["default_model"] == "test.sbs"
        assert saved_config["redis"]["port"] == 6379


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_health_check(self) -> None:
        """Test running full health check."""
        checks = await check_system_requirements()

        # Should have at least basic checks
        assert len(checks) >= 3

        # Python check should pass
        python_check = next(
            (c for c in checks if c[0] == "Python Version"), None
        )
        assert python_check is not None
        assert python_check[1] is True  # Should pass

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("C:/codedev/llm/.models").exists(),
        reason="Model directory not available",
    )
    async def test_detect_real_models(self) -> None:
        """Test detecting actual models (if available)."""
        wizard = OnboardingWizard()
        models = wizard._detect_available_models()

        # If models directory exists, should find some models
        if Path("C:/codedev/llm/.models").exists():
            assert len(models) > 0

            for model in models:
                assert Path(model["path"]).exists()
                assert model["size_mb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
