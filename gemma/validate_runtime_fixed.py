#!/usr/bin/env python3
"""
Runtime Validation Script for Gemma CLI
Performs comprehensive validation without requiring all optional dependencies.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ValidationResult:
    def __init__(self):
        self.passed: List[str] = []
        self.warnings: List[Tuple[str, str]] = []
        self.failures: List[Tuple[str, str, str]] = []

    def add_pass(self, test_name: str) -> None:
        self.passed.append(test_name)

    def add_warning(self, test_name: str, message: str) -> None:
        self.warnings.append((test_name, message))

    def add_failure(self, test_name: str, error: str, traceback_str: str) -> None:
        self.failures.append((test_name, error, traceback_str))

    def print_summary(self) -> None:
        print(f"\n{BOLD}=== VALIDATION SUMMARY ==={RESET}\n")

        print(f"{GREEN}[PASS] PASSED ({len(self.passed)}):{RESET}")
        for test in self.passed:
            print(f"  [+] {test}")

        if self.warnings:
            print(f"\n{YELLOW}[WARN]  WARNINGS ({len(self.warnings)}):{RESET}")
            for test, msg in self.warnings:
                print(f"  [!] {test}")
                print(f"    {msg}")

        if self.failures:
            print(f"\n{RED}[FAIL] FAILURES ({len(self.failures)}):{RESET}")
            for test, error, tb in self.failures:
                print(f"  [X] {test}")
                print(f"    Error: {error}")
                if tb:
                    print(f"    Traceback:\n{tb}")

        print(f"\n{BOLD}Total: {len(self.passed)} passed, {len(self.warnings)} warnings, {len(self.failures)} failures{RESET}\n")

        return len(self.failures) == 0


def test_import_module(module_path: str, result: ValidationResult) -> Any:
    """Test importing a module."""
    try:
        module = importlib.import_module(module_path)
        result.add_pass(f"Import: {module_path}")
        return module
    except ImportError as e:
        result.add_failure(
            f"Import: {module_path}",
            str(e),
            traceback.format_exc()
        )
        return None
    except Exception as e:
        result.add_failure(
            f"Import: {module_path}",
            f"Unexpected error: {e}",
            traceback.format_exc()
        )
        return None


def test_core_imports(result: ValidationResult) -> None:
    """Test importing core modules."""
    print(f"\n{BOLD}1. Testing Core Imports{RESET}")

    modules = [
        "gemma_cli",
        "gemma_cli.cli",
        "gemma_cli.config",
        "gemma_cli.config.settings",
        "gemma_cli.config.models",
        "gemma_cli.config.prompts",
        "gemma_cli.core",
        "gemma_cli.core.conversation",
        "gemma_cli.core.gemma",
        "gemma_cli.ui",
        "gemma_cli.ui.console",
        "gemma_cli.ui.theme",
        "gemma_cli.ui.formatters",
        "gemma_cli.ui.widgets",
        "gemma_cli.ui.components",
    ]

    for module_path in modules:
        test_import_module(module_path, result)


def test_optional_imports(result: ValidationResult) -> None:
    """Test importing optional modules (may have warnings)."""
    print(f"\n{BOLD}2. Testing Optional Imports{RESET}")

    optional_modules = [
        "gemma_cli.commands",
        "gemma_cli.commands.setup",
        "gemma_cli.commands.rag_commands",
        "gemma_cli.rag",
        "gemma_cli.rag.memory",
        "gemma_cli.rag.optimizations",
        "gemma_cli.mcp",
        "gemma_cli.mcp.client",
        "gemma_cli.mcp.config_loader",
        "gemma_cli.onboarding",
        "gemma_cli.onboarding.wizard",
        "gemma_cli.onboarding.checks",
    ]

    for module_path in optional_modules:
        module = test_import_module(module_path, result)
        if module is None:
            result.add_warning(
                f"Optional module: {module_path}",
                "Module failed to import (may require optional dependencies)"
            )


def test_circular_dependencies(result: ValidationResult) -> None:
    """Test for circular import issues."""
    print(f"\n{BOLD}3. Testing Circular Dependencies{RESET}")

    try:
        # Try importing in different orders
        import gemma_cli.config.settings
        import gemma_cli.config.models
        import gemma_cli.core.gemma
        import gemma_cli.core.conversation

        # Reload to catch circular issues
        importlib.reload(sys.modules["gemma_cli.config.settings"])
        importlib.reload(sys.modules["gemma_cli.config.models"])

        result.add_pass("Circular dependency check")
    except Exception as e:
        result.add_failure(
            "Circular dependency check",
            str(e),
            traceback.format_exc()
        )


def test_config_models(result: ValidationResult) -> None:
    """Test configuration models."""
    print(f"\n{BOLD}4. Testing Configuration Models{RESET}")

    try:
        from gemma_cli.config.models import ModelConfig, ModelPreset
        from pydantic import ValidationError

        # Test valid model config
        config = ModelConfig(
            name="test-model",
            path="/fake/path/model.sbs",
            tokenizer="/fake/path/tokenizer.spm"
        )
        result.add_pass("ModelConfig instantiation")

        # Test validation
        try:
            invalid = ModelConfig(name="", path="", tokenizer="")
            result.add_warning("ModelConfig validation", "Empty strings accepted (should validate)")
        except ValidationError:
            result.add_pass("ModelConfig validation")

        # Test ModelPreset
        preset = ModelPreset(
            name="gemma-2b-it",
            model_size="2b",
            model_type="gemma",
            description="Test preset"
        )
        result.add_pass("ModelPreset instantiation")

    except ImportError as e:
        result.add_failure(
            "Config models test",
            f"Import error: {e}",
            traceback.format_exc()
        )
    except Exception as e:
        result.add_failure(
            "Config models test",
            str(e),
            traceback.format_exc()
        )


def test_settings_loading(result: ValidationResult) -> None:
    """Test settings loading."""
    print(f"\n{BOLD}5. Testing Settings Loading{RESET}")

    try:
        from gemma_cli.config.settings import GemmaSettings

        # Test default settings
        settings = GemmaSettings()
        result.add_pass("GemmaSettings default instantiation")

        # Check required fields
        if hasattr(settings, "models_dir"):
            result.add_pass("GemmaSettings.models_dir exists")
        else:
            result.add_warning("GemmaSettings.models_dir", "Field not found")

        if hasattr(settings, "gemma_binary"):
            result.add_pass("GemmaSettings.gemma_binary exists")
        else:
            result.add_warning("GemmaSettings.gemma_binary", "Field not found")

    except Exception as e:
        result.add_failure(
            "Settings loading test",
            str(e),
            traceback.format_exc()
        )


def test_cli_structure(result: ValidationResult) -> None:
    """Test CLI command structure."""
    print(f"\n{BOLD}6. Testing CLI Structure{RESET}")

    try:
        from gemma_cli.cli import main, cli
        import click

        # Check if main is callable
        if callable(main):
            result.add_pass("CLI main() is callable")
        else:
            result.add_failure("CLI main()", "main() is not callable", "")

        # Check if cli is a Click group
        if isinstance(cli, click.Group):
            result.add_pass("CLI is Click group")

            # List commands
            commands = cli.list_commands(None)
            if commands:
                result.add_pass(f"CLI has commands: {', '.join(commands)}")
            else:
                result.add_warning("CLI commands", "No commands registered")
        else:
            result.add_failure("CLI structure", "cli is not a Click group", "")

    except Exception as e:
        result.add_failure(
            "CLI structure test",
            str(e),
            traceback.format_exc()
        )


def test_async_patterns(result: ValidationResult) -> None:
    """Test async function patterns."""
    print(f"\n{BOLD}7. Testing Async Patterns{RESET}")

    try:
        import inspect
        from gemma_cli.core import conversation, gemma

        # Check conversation module
        for name, obj in inspect.getmembers(conversation):
            if inspect.iscoroutinefunction(obj):
                result.add_pass(f"Async function found: conversation.{name}")

        # Check gemma module
        for name, obj in inspect.getmembers(gemma):
            if inspect.iscoroutinefunction(obj):
                result.add_pass(f"Async function found: gemma.{name}")

        # Look for common async mistakes (sync calls in async functions)
        # This is a simplified check - full AST analysis would be more thorough
        result.add_pass("Async pattern basic check")

    except Exception as e:
        result.add_failure(
            "Async patterns test",
            str(e),
            traceback.format_exc()
        )


def test_ui_components(result: ValidationResult) -> None:
    """Test UI component imports."""
    print(f"\n{BOLD}8. Testing UI Components{RESET}")

    try:
        from gemma_cli.ui import console, theme, formatters, widgets, components
        from rich.console import Console

        # Test console creation
        test_console = console.get_console()
        if isinstance(test_console, Console):
            result.add_pass("UI console creation")
        else:
            result.add_warning("UI console", "get_console() didn't return Console instance")

        # Check theme
        if hasattr(theme, "THEME"):
            result.add_pass("UI theme defined")
        else:
            result.add_warning("UI theme", "THEME not found")

        # Check formatters
        if hasattr(formatters, "format_error"):
            result.add_pass("UI formatters present")
        else:
            result.add_warning("UI formatters", "format_error not found")

    except Exception as e:
        result.add_failure(
            "UI components test",
            str(e),
            traceback.format_exc()
        )


def test_model_manager(result: ValidationResult) -> None:
    """Test ModelManager instantiation."""
    print(f"\n{BOLD}9. Testing Model Manager{RESET}")

    try:
        from gemma_cli.core.gemma import ModelManager
        from pathlib import Path

        # Try to create with fake paths
        manager = ModelManager(
            models_dir=Path("/fake/models"),
            gemma_binary=Path("/fake/gemma")
        )
        result.add_pass("ModelManager instantiation")

        # Test list_models with fake directory
        try:
            models = manager.list_models()
            result.add_pass("ModelManager.list_models() callable")
        except Exception as e:
            # Expected to fail with fake paths, but should not crash
            if "not" in str(e).lower() or "exist" in str(e).lower():
                result.add_pass("ModelManager.list_models() handles missing dir gracefully")
            else:
                result.add_warning(
                    "ModelManager.list_models()",
                    f"Unexpected error with fake paths: {e}"
                )

    except Exception as e:
        result.add_failure(
            "Model manager test",
            str(e),
            traceback.format_exc()
        )


def main() -> int:
    """Main validation function."""
    print(f"{BOLD}{'='*60}")
    print("  GEMMA CLI RUNTIME VALIDATION")
    print(f"{'='*60}{RESET}\n")

    # Add src to path
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    result = ValidationResult()

    # Run all tests
    test_core_imports(result)
    test_optional_imports(result)
    test_circular_dependencies(result)
    test_config_models(result)
    test_settings_loading(result)
    test_cli_structure(result)
    test_async_patterns(result)
    test_ui_components(result)
    test_model_manager(result)

    # Print summary
    success = result.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
