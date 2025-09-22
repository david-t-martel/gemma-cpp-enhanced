#!/usr/bin/env python3
"""
Gemma.cpp Test Runner
Convenient script to run different categories of tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


class GemmaTestRunner:
    def __init__(self, build_dir: Optional[str] = None):
        self.project_root = Path(__file__).parent
        self.gemma_cpp_dir = self.project_root / "gemma.cpp"
        self.build_dir = Path(build_dir) if build_dir else self.gemma_cpp_dir / "build"

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run a command and return success status"""
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {cwd or Path.cwd()}")
        print("-" * 50)

        try:
            result = subprocess.run(cmd, cwd=cwd, check=False)
            success = result.returncode == 0
            print("-" * 50)
            print(f"Command {'SUCCEEDED' if success else 'FAILED'} (exit code: {result.returncode})")
            print()
            return success
        except Exception as e:
            print(f"Error running command: {e}")
            return False

    def build_tests(self, preset: str = "make") -> bool:
        """Build the test executables"""
        print("Building test executables...")

        # Configure CMake
        configure_cmd = [
            "cmake", "--preset", preset,
            "-DGEMMA_ENABLE_ENHANCED_TESTS=ON",
            "-DGEMMA_ENABLE_TESTS=ON",
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        if not self._run_command(configure_cmd, self.gemma_cpp_dir):
            return False

        # Build tests
        build_cmd = [
            "cmake", "--build", "--preset", preset,
            "--target", "test_gemma_core",
            "--target", "test_backends_integration",
            "--target", "test_model_loading_integration",
            "--target", "benchmark_inference",
            "--parallel", "4"
        ]

        return self._run_command(build_cmd, self.gemma_cpp_dir)

    def run_unit_tests(self) -> bool:
        """Run unit tests"""
        print("Running unit tests...")

        cmd = [
            "ctest", "--test-dir", str(self.build_dir),
            "-L", "unit",
            "--output-on-failure",
            "--parallel", "4"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)

    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("Running integration tests...")

        cmd = [
            "ctest", "--test-dir", str(self.build_dir),
            "-L", "integration",
            "--output-on-failure",
            "--parallel", "2"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)

    def run_performance_tests(self) -> bool:
        """Run performance benchmarks"""
        print("Running performance benchmarks...")

        benchmark_exe = self.build_dir / "benchmark_inference"
        if not benchmark_exe.exists():
            print(f"Benchmark executable not found: {benchmark_exe}")
            return False

        cmd = [
            str(benchmark_exe),
            "--benchmark_min_time=0.1",
            "--benchmark_out=benchmark_results.json",
            "--benchmark_out_format=json"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)

    def run_all_tests(self) -> bool:
        """Run all test categories"""
        print("Running all tests...")

        cmd = [
            "ctest", "--test-dir", str(self.build_dir),
            "--output-on-failure",
            "--parallel", "4"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)

    def run_quick_tests(self) -> bool:
        """Run only the fastest tests for quick feedback"""
        print("Running quick tests (sampling only)...")

        cmd = [
            "ctest", "--test-dir", str(self.build_dir),
            "-R", "test_sampling",
            "--output-on-failure"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)

    def list_tests(self) -> bool:
        """List available tests"""
        print("Available tests:")

        cmd = [
            "ctest", "--test-dir", str(self.build_dir),
            "--show-only"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)

    def run_specific_test(self, test_name: str) -> bool:
        """Run a specific test by name"""
        print(f"Running specific test: {test_name}")

        cmd = [
            "ctest", "--test-dir", str(self.build_dir),
            "-R", test_name,
            "--output-on-failure",
            "--verbose"
        ]

        return self._run_command(cmd, self.gemma_cpp_dir)


def main():
    parser = argparse.ArgumentParser(description='Gemma.cpp Test Runner')
    parser.add_argument('--build-dir', help='Build directory path')
    parser.add_argument('--preset', default='make', help='CMake preset to use (default: make)')

    subparsers = parser.add_subparsers(dest='command', help='Test commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build test executables')

    # Test commands
    unit_parser = subparsers.add_parser('unit', help='Run unit tests')
    integration_parser = subparsers.add_parser('integration', help='Run integration tests')
    performance_parser = subparsers.add_parser('performance', help='Run performance benchmarks')
    all_parser = subparsers.add_parser('all', help='Run all tests')
    quick_parser = subparsers.add_parser('quick', help='Run quick tests only')

    # Utility commands
    list_parser = subparsers.add_parser('list', help='List available tests')
    run_parser = subparsers.add_parser('run', help='Run specific test')
    run_parser.add_argument('test_name', help='Test name to run')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    runner = GemmaTestRunner(args.build_dir)

    # Ensure build directory exists
    if not runner.build_dir.exists() and args.command != 'build':
        print(f"Build directory not found: {runner.build_dir}")
        print("Run 'python run_tests.py build' first to build the tests.")
        return 1

    success = True

    if args.command == 'build':
        success = runner.build_tests(args.preset)
    elif args.command == 'unit':
        success = runner.run_unit_tests()
    elif args.command == 'integration':
        success = runner.run_integration_tests()
    elif args.command == 'performance':
        success = runner.run_performance_tests()
    elif args.command == 'all':
        success = runner.run_all_tests()
    elif args.command == 'quick':
        success = runner.run_quick_tests()
    elif args.command == 'list':
        success = runner.list_tests()
    elif args.command == 'run':
        success = runner.run_specific_test(args.test_name)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())