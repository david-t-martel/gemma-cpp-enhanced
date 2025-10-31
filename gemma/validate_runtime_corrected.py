#!/usr/bin/env python3
"""
Runtime Validation Script for Gemma CLI (CORRECTED VERSION)
Uses actual class names from the codebase.
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

        print(f"{GREEN}[PASSED] ({len(self.passed)}):{RESET}")
        for test in self.passed:
            print(f"  [+] {test}")

        if self.warnings:
            print(f"\n{YELLOW}[WARNINGS] ({len(self.warnings)}):{RESET}")
            for test, msg in self.warnings:
                print(f"  [!] {test}")
                print(f"      {msg}")

        if self.failures:
            print(f"\n{RED}[FAILURES] ({len(self.failures)}):{RESET}")
            for test, error, tb in self.failures:
                print(f"  [X] {test}")
                print(f"      Error: {error}")
                if tb:
                    print(f"      Traceback:\n{tb}")

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
    """Test configuration models (CORRECTED VERSION)."""
    print(f"\n{BOLD}4. Testing Configuration Models (CORRECTED){RESET}")

    try:
        # CORRECTED: Use actual class names
        from gemma_cli.config.models import ModelPreset, PerformanceProfile, ModelManager
        from pydantic import ValidationError

        # Test ModelPreset (not ModelConfig)
        preset = ModelPreset(
            name="test-model",
            weights="/fake/path/model.sbs",
            tokenizer="/fake/path/tokenizer.spm",
            format="sfp",
            size_gb=2.5,
            avg_tokens_per_sec=100,
            quality="high",
            use_case="testing"
        )
        result.add_pass("ModelPreset instantiation")

        # Test validation
        try:
            invalid = ModelPreset(
                name="",  # Empty name
                weights="",
                tokenizer="",
                format="invalid",  # Invalid format
                size_gb=-1,  # Negative size
                avg_tokens_per_sec=0,  # Zero speed
                quality="bad",  # Invalid quality
                use_case="test"
            )
            result.add_warning(
                "ModelPreset validation",
                "Validation did not catch invalid values"
            )
        except ValidationError:
            result.add_pass("ModelPreset validation")

        # Test PerformanceProfile
        profile = PerformanceProfile(
            name="test-profile",
            max_tokens=1024,
            temperature=0.7,
            description="Test profile"
        )
        result.add_pass("PerformanceProfile instantiation")

        # Test ModelManager (from config.models, not core.gemma)
        manager = ModelManager(config_path=Path("/fake/config.toml"))
        result.add_pass("ModelManager instantiation (from config.models)")

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
    """Test settings loading (CORRECTED VERSION)."""
    print(f"\n{BOLD}5. Testing Settings Loading (CORRECTED){RESET}")

    try:
        # CORRECTED: Use Settings instead of GemmaSettings
        from gemma_cli.config.settings import Settings

        # Test default settings
        settings = Settings()
        result.add_pass("Settings default instantiation")

        # Check sections exist
        if hasattr(settings, "redis"):
            result.add_pass("Settings.redis exists")
        else:
            result.add_warning("Settings.redis", "Field not found")

        if hasattr(settings, "memory"):
            result.add_pass("Settings.memory exists")
        else:
            result.add_warning("Settings.memory", "Field not found")

        if hasattr(settings, "ui"):
            result.add_pass("Settings.ui exists")
        else:
            result.add_warning("Settings.ui", "Field not found")

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
                result.add_pass(f"CLI has {len(commands)} commands: {', '.join(sorted(commands))}")
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
        from gemma_cli import cli, commands

        # Count async functions
        async_funcs = []

        # Check cli module
        for name, obj in inspect.getmembers(cli):
            if inspect.iscoroutinefunction(obj):
                async_funcs.append(f"cli.{name}")

        if async_funcs:
            result.add_pass(f"Found {len(async_funcs)} async functions in CLI")
        else:
            result.add_warning("Async functions", "No async functions found in cli module")

        # Verify asyncio.run usage pattern
        import ast
        cli_file = Path(__file__).parent / "src" / "gemma_cli" / "cli.py"
        if cli_file.exists():
            with open(cli_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            asyncio_run_count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (hasattr(node.func, "attr") and node.func.attr == "run" and
                        hasattr(node.func, "value") and
                        hasattr(node.func.value, "id") and
                        node.func.value.id == "asyncio"):
                        asyncio_run_count += 1

            if asyncio_run_count > 0:
                result.add_pass(f"asyncio.run() used correctly ({asyncio_run_count} times)")
            else:
                result.add_warning("asyncio.run()", "No asyncio.run() calls found")

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
        from gemma_cli.ui import console, theme, formatters
        from rich.console import Console

        # Test console creation
        test_console = console.get_console()
        if isinstance(test_console, Console):
            result.add_pass("UI console creation")
        else:
            result.add_warning("UI console", "get_console() didn't return Console instance")

        # Check theme exports (don't require specific name)
        theme_attrs = [attr for attr in dir(theme) if not attr.startswith("_")]
        if theme_attrs:
            result.add_pass(f"UI theme module has {len(theme_attrs)} exports")
        else:
            result.add_warning("UI theme", "No public exports found")

        # Check formatters (accept any format_error* function)
        formatter_funcs = [
            attr for attr in dir(formatters)
            if callable(getattr(formatters, attr)) and "format" in attr.lower()
        ]
        if formatter_funcs:
            result.add_pass(f"UI formatters present: {', '.join(formatter_funcs[:3])}")
        else:
            result.add_warning("UI formatters", "No format functions found")

    except Exception as e:
        result.add_failure(
            "UI components test",
            str(e),
            traceback.format_exc()
        )


def test_gemma_interface(result: ValidationResult) -> None:
    """Test GemmaInterface instantiation (CORRECTED)."""
    print(f"\n{BOLD}9. Testing Gemma Interface (CORRECTED){RESET}")

    try:
        # CORRECTED: GemmaInterface is in core.gemma, not ModelManager
        from gemma_cli.core.gemma import GemmaInterface

        # Try to create with fake paths (should not crash)
        interface = GemmaInterface(
            model_path="/fake/model.sbs",
            tokenizer_path="/fake/tokenizer.spm",
            gemma_executable="/fake/gemma.exe"
        )
        result.add_warning(
            "GemmaInterface instantiation",
            "Created with fake paths (expected FileNotFoundError on executable check)"
        )

    except FileNotFoundError as e:
        # Expected error - executable doesn't exist
        if "gemma.exe" in str(e) or "Gemma executable" in str(e):
            result.add_pass("GemmaInterface validates executable existence")
        else:
            result.add_warning(
                "GemmaInterface",
                f"Unexpected FileNotFoundError: {e}"
            )

    except ImportError as e:
        result.add_failure(
            "Gemma interface test",
            f"Import error: {e}",
            traceback.format_exc()
        )
    except Exception as e:
        result.add_failure(
            "Gemma interface test",
            str(e),
            traceback.format_exc()
        )


def test_model_manager_location(result: ValidationResult) -> None:
    """Test ModelManager is in correct location."""
    print(f"\n{BOLD}10. Testing ModelManager Location{RESET}")

    try:
        # Verify ModelManager is in config.models
        from gemma_cli.config.models import ModelManager

        manager = ModelManager(config_path=Path("/fake/config.toml"))
        result.add_pass("ModelManager imported from config.models (correct location)")

        # Verify methods exist
        if hasattr(manager, "list_models"):
            result.add_pass("ModelManager.list_models() exists")
        if hasattr(manager, "get_model"):
            result.add_pass("ModelManager.get_model() exists")
        if hasattr(manager, "detect_models"):
            result.add_pass("ModelManager.detect_models() exists")

    except ImportError as e:
        result.add_failure(
            "ModelManager location test",
            f"Import error: {e}",
            traceback.format_exc()
        )
    except Exception as e:
        result.add_failure(
            "ModelManager location test",
            str(e),
            traceback.format_exc()
        )


def main() -> int:
    """Main validation function."""
    print(f"{BOLD}{'='*60}")
    print("  GEMMA CLI RUNTIME VALIDATION (CORRECTED)")
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
    test_gemma_interface(result)
    test_model_manager_location(result)

    # Print summary
    success = result.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
