#!/usr/bin/env python
"""Test runner script for terminal UI tests with snapshot generation.

This script provides a convenient interface for running terminal UI tests
with various options for snapshot generation, comparison, and reporting.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import argparse


def run_pytest(
    args: List[str],
    verbose: bool = False,
    capture_output: bool = True,
) -> int:
    """Run pytest with specified arguments.

    Args:
        args: Additional pytest arguments
        verbose: Enable verbose output
        capture_output: Capture test output

    Returns:
        Exit code from pytest
    """
    cmd = ["pytest"]

    if verbose:
        cmd.append("-vvs")
    else:
        cmd.append("-v")

    if not capture_output:
        cmd.append("--capture=no")

    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run terminal UI tests with snapshot generation"
    )

    # Test selection
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific test files or test names to run",
    )

    # Test filtering
    parser.add_argument(
        "-m", "--marker",
        help="Run tests with specific marker (ui, slow, integration, etc.)",
    )
    parser.add_argument(
        "-k", "--keyword",
        help="Run tests matching keyword expression",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (-vvs)",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Don't capture test output",
    )

    # Snapshot options
    parser.add_argument(
        "--snapshot-format",
        choices=["svg", "png", "html"],
        default="svg",
        help="Snapshot format (default: svg)",
    )
    parser.add_argument(
        "--snapshot-theme",
        choices=["monokai", "dimmed", "light"],
        default="monokai",
        help="Snapshot color theme (default: monokai)",
    )

    # Performance options
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        help="Number of parallel workers (pytest-xdist)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks",
    )

    # Coverage options
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Generate coverage report",
    )

    # Cleanup options
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean old snapshots before running",
    )

    args = parser.parse_args()

    # Build pytest command
    pytest_args = []

    # Add test directory
    test_dir = Path(__file__).parent
    if args.tests:
        # Specific tests provided
        pytest_args.extend(args.tests)
    else:
        # Run all tests in directory
        pytest_args.append(str(test_dir))

    # Add markers
    if args.marker:
        pytest_args.extend(["-m", args.marker])

    # Add keyword filter
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Add parallel execution
    if args.num_workers:
        pytest_args.extend(["-n", str(args.num_workers)])

    # Add benchmark
    if args.benchmark:
        pytest_args.append("--benchmark-only")

    # Add coverage
    if args.cov:
        pytest_args.extend([
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html",
        ])

    # Clean old snapshots
    if args.clean:
        print("Cleaning old snapshots...")
        screenshots_dir = test_dir / "screenshots"
        if screenshots_dir.exists():
            import shutil
            shutil.rmtree(screenshots_dir)
            screenshots_dir.mkdir()
        print("✓ Snapshots cleaned")

    # Set environment for snapshot configuration
    import os
    if args.snapshot_format:
        os.environ["GEMMA_TEST_SNAPSHOT_FORMAT"] = args.snapshot_format
    if args.snapshot_theme:
        os.environ["GEMMA_TEST_SNAPSHOT_THEME"] = args.snapshot_theme

    # Run tests
    exit_code = run_pytest(
        pytest_args,
        verbose=args.verbose,
        capture_output=not args.no_capture,
    )

    # Summary
    if exit_code == 0:
        print("\n✓ All tests passed!")
        print(f"\nSnapshots saved to: {test_dir / 'screenshots'}")
    else:
        print(f"\n✗ Tests failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
