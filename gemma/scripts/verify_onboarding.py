#!/usr/bin/env python
"""Verify onboarding system installation and functionality."""

import sys
from pathlib import Path


def check_imports() -> bool:
    """Check if all onboarding modules can be imported."""
    print("Checking module imports...")

    try:
        from src.gemma_cli.onboarding import (
            OnboardingWizard,
            InteractiveTutorial,
            get_template,
            check_system_requirements,
        )
        print("  ✓ Core onboarding modules")
    except ImportError as e:
        print(f"  ✗ Failed to import core modules: {e}")
        return False

    try:
        from src.gemma_cli.commands.setup import init, health, tutorial, reset, config
        print("  ✓ Setup commands")
    except ImportError as e:
        print(f"  ✗ Failed to import setup commands: {e}")
        return False

    try:
        from src.gemma_cli.cli import cli, main
        print("  ✓ Main CLI")
    except ImportError as e:
        print(f"  ✗ Failed to import main CLI: {e}")
        return False

    return True


def check_files() -> bool:
    """Check if all expected files exist."""
    print("\nChecking file structure...")

    expected_files = [
        "src/gemma_cli/onboarding/__init__.py",
        "src/gemma_cli/onboarding/wizard.py",
        "src/gemma_cli/onboarding/checks.py",
        "src/gemma_cli/onboarding/templates.py",
        "src/gemma_cli/onboarding/tutorial.py",
        "src/gemma_cli/commands/setup.py",
        "src/gemma_cli/cli.py",
        "config/config.example.toml",
        "docs/ONBOARDING.md",
        "tests/test_onboarding.py",
        "ONBOARDING_IMPLEMENTATION.md",
    ]

    all_exist = True
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_exist = False

    return all_exist


def check_templates() -> bool:
    """Check if configuration templates work."""
    print("\nChecking configuration templates...")

    try:
        from src.gemma_cli.onboarding.templates import get_template, list_templates

        templates = list_templates()
        print(f"  ✓ Found {len(templates)} templates")

        for key, name, desc in templates:
            template = get_template(key)
            print(f"    • {name}: {len(template['config'])} sections")

        return True
    except Exception as e:
        print(f"  ✗ Template check failed: {e}")
        return False


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    print("\nChecking dependencies...")

    dependencies = [
        ("click", "Click"),
        ("rich", "Rich"),
        ("prompt_toolkit", "Prompt Toolkit"),
        ("toml", "TOML"),
        ("pydantic", "Pydantic"),
        ("psutil", "Psutil"),
        ("redis", "Redis"),
    ]

    all_available = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT INSTALLED")
            all_available = False

    return all_available


async def run_basic_checks() -> bool:
    """Run basic system checks."""
    print("\nRunning basic health checks...")

    try:
        from src.gemma_cli.onboarding.checks import check_system_requirements

        checks = await check_system_requirements()

        passed = sum(1 for _, success, _ in checks if success)
        total = len(checks)

        print(f"  ✓ Completed {total} checks ({passed} passed)")

        return True
    except Exception as e:
        print(f"  ✗ Health checks failed: {e}")
        return False


def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print("Gemma CLI Onboarding System Verification")
    print("=" * 60)

    results = []

    # Check imports
    results.append(("Module imports", check_imports()))

    # Check files
    results.append(("File structure", check_files()))

    # Check templates
    results.append(("Configuration templates", check_templates()))

    # Check dependencies
    results.append(("Dependencies", check_dependencies()))

    # Check basic functionality
    try:
        import asyncio
        results.append(("Basic health checks", asyncio.run(run_basic_checks())))
    except Exception as e:
        print(f"\n✗ Could not run async checks: {e}")
        results.append(("Basic health checks", False))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8} {check_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! Onboarding system is ready.")
        print("\nNext steps:")
        print("  1. Install missing dependencies: uv pip install psutil")
        print("  2. Run tests: uv run pytest tests/test_onboarding.py -v")
        print("  3. Try it: uv run python -m gemma_cli.cli init")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
