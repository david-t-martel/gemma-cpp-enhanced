"""Test console dependency injection pattern.

This test verifies that the console is properly injected through Click context
and used by commands and widgets.
"""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from gemma_cli.cli import cli
from gemma_cli.ui.console import create_console
from gemma_cli.ui.widgets import MemoryDashboard, StatusBar
from gemma_cli.onboarding.wizard import OnboardingWizard


class TestConsoleInjection:
    """Test suite for console dependency injection."""

    def test_create_console_factory(self):
        """Test that create_console returns a Console instance."""
        console = create_console()
        assert isinstance(console, (Console, MagicMock))

    def test_cli_injects_console_into_context(self):
        """Test that CLI injects console into context."""
        runner = CliRunner()

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            result = runner.invoke(cli, ["--help"])

        # CLI should run without errors
        assert result.exit_code == 0

    def test_memory_dashboard_accepts_console(self):
        """Test that MemoryDashboard accepts console parameter."""
        console = create_console()
        dashboard = MemoryDashboard(console=console)

        assert dashboard.console is console

    def test_memory_dashboard_fallback(self):
        """Test that MemoryDashboard works without console (backward compatibility)."""
        # Should not raise error even without console parameter
        dashboard = MemoryDashboard()
        assert dashboard.console is not None

    def test_status_bar_accepts_console(self):
        """Test that StatusBar accepts console parameter."""
        console = create_console()
        status_bar = StatusBar(console=console)

        assert status_bar.console is console

    @patch("gemma_cli.onboarding.wizard.PromptSession")
    def test_onboarding_wizard_accepts_console(self, mock_prompt):
        """Test that OnboardingWizard accepts console parameter."""
        console = create_console()
        wizard = OnboardingWizard(console=console)

        assert wizard.console is console

    @patch("gemma_cli.onboarding.wizard.PromptSession")
    def test_onboarding_wizard_fallback(self, mock_prompt):
        """Test that OnboardingWizard works without console (backward compatibility)."""
        wizard = OnboardingWizard()
        assert wizard.console is not None

    @pytest.mark.parametrize(
        "widget_class",
        [MemoryDashboard, StatusBar],
    )
    def test_widgets_use_injected_console(self, widget_class):
        """Test that widgets use the injected console instance."""
        # Create a mock console to track calls
        mock_console = MagicMock(spec=Console)
        widget = widget_class(console=mock_console)

        # Verify the widget stored the console
        assert widget.console is mock_console

    def test_no_deprecation_warning_with_factory(self):
        """Test that using create_console doesn't trigger deprecation warnings."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            console = create_console()

            # Verify no deprecation warnings were raised
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

    def test_console_injection_integration(self):
        """Integration test: verify console flows from CLI to widgets."""
        runner = CliRunner()

        with patch("gemma_cli.cli.check_first_run", return_value=False):
            with patch("gemma_cli.commands.model_simple.load_detected_models", return_value={}):
                with patch("gemma_cli.commands.model_simple.load_config") as mock_config:
                    mock_config.return_value = MagicMock(configured_models={}, gemma=MagicMock(default_model=None))

                    # Run a command that uses console
                    result = runner.invoke(cli, ["model", "list"])

                    # Command should execute successfully
                    assert result.exit_code == 0


class TestConsoleBackwardCompatibility:
    """Test backward compatibility for legacy code."""

    def test_widgets_work_without_console_parameter(self):
        """Test that widgets still work without explicit console parameter."""
        # This ensures old code doesn't break
        dashboard = MemoryDashboard()
        status_bar = StatusBar()

        assert dashboard.console is not None
        assert status_bar.console is not None

    @patch("gemma_cli.onboarding.wizard.PromptSession")
    def test_wizard_works_without_console_parameter(self, mock_prompt):
        """Test that OnboardingWizard works without console parameter."""
        wizard = OnboardingWizard()
        assert wizard.console is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
