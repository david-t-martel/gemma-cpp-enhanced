#!/usr/bin/env python3
"""
Comprehensive Test Runner for Gemma.cpp
========================================

This script orchestrates the complete test suite for Gemma.cpp, including:
- Backend validation and dependency checking
- Compilation tests for all configurations
- Functional integration tests
- Performance benchmarks and comparisons
- Report generation and analysis

Usage:
    python run_comprehensive_tests.py [options]

Features:
- Parallel test execution
- Detailed progress reporting
- Automatic failure analysis
- Cross-platform compatibility
- Integration with existing test infrastructure
"""

import os
import sys
import subprocess
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil


@dataclass
class TestSuite:
    """Configuration for a test suite."""
    name: str
    description: str
    command: List[str]
    working_directory: Path
    timeout: int = 300
    required: bool = True
    depends_on: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of running a test suite."""
    suite_name: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
    return_code: int = 0


class ComprehensiveTestRunner:
    """Orchestrates all test suites for comprehensive validation."""

    def __init__(self, project_root: Path, build_dir: Path, output_dir: Path):
        self.project_root = project_root
        self.build_dir = build_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Ensure directories exist
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_test_suites(self) -> List[TestSuite]:
        """Create all test suite configurations."""
        suites = []

        # 1. Dependency validation
        suites.append(TestSuite(
            name="dependency_validation",
            description="Validate backend dependencies and toolchains",
            command=[
                sys.executable,
                str(self.project_root / "validate_backends.py"),
                "--project-root", str(self.project_root),
                "--output-dir", str(self.output_dir / "validation")
            ],
            working_directory=self.project_root,
            timeout=300,
            required=True
        ))

        # 2. Compilation tests
        suites.append(TestSuite(
            name="compilation_tests",
            description="Test compilation across all backend configurations",
            command=[
                sys.executable,
                str(self.project_root / "compile_test.py"),
                "--project-root", str(self.project_root),
                "--build-dir", str(self.build_dir / "compile-test"),
                "--output-dir", str(self.output_dir / "compilation"),
                "--parallel", "2"
            ],
            working_directory=self.project_root,
            timeout=1800,  # 30 minutes
            required=True,
            depends_on=["dependency_validation"]
        ))

        # 3. CMake configuration and build
        suites.append(TestSuite(
            name="cmake_configure",
            description="Configure project with CMake",
            command=[
                "cmake",
                str(self.project_root),
                "-DGEMMA_BUILD_ENHANCED_TESTS=ON",
                "-DGEMMA_BUILD_BACKEND_TESTS=ON",
                "-DGEMMA_AUTO_DETECT_BACKENDS=ON",
                "-DCMAKE_BUILD_TYPE=Release"
            ],
            working_directory=self.build_dir,
            timeout=300,
            required=True,
            depends_on=["dependency_validation"]
        ))

        suites.append(TestSuite(
            name="cmake_build",
            description="Build project with CMake",
            command=[
                "cmake",
                "--build", ".",
                "--config", "Release",
                "-j", "4"
            ],
            working_directory=self.build_dir,
            timeout=1200,  # 20 minutes
            required=True,
            depends_on=["cmake_configure"]
        ))

        # 4. Unit tests
        suites.append(TestSuite(
            name="unit_tests",
            description="Run unit tests",
            command=["ctest", "--output-on-failure", "-L", "unit"],
            working_directory=self.build_dir,
            timeout=600,
            required=True,
            depends_on=["cmake_build"]
        ))

        # 5. Integration tests
        suites.append(TestSuite(
            name="integration_tests",
            description="Run integration tests",
            command=["ctest", "--output-on-failure", "-L", "integration"],
            working_directory=self.build_dir,
            timeout=600,
            required=True,
            depends_on=["cmake_build"]
        ))

        # 6. Functional tests
        suites.append(TestSuite(
            name="functional_tests",
            description="Run functional backend tests",
            command=["ctest", "--output-on-failure", "-L", "functional"],
            working_directory=self.build_dir,
            timeout=900,
            required=True,
            depends_on=["cmake_build"]
        ))

        # 7. Backend-specific tests
        suites.append(TestSuite(
            name="backend_tests",
            description="Run backend-specific tests",
            command=["ctest", "--output-on-failure", "-L", "backend"],
            working_directory=self.build_dir,
            timeout=900,
            required=False,  # Optional if no backends available
            depends_on=["cmake_build"]
        ))

        # 8. Performance tests
        suites.append(TestSuite(
            name="performance_tests",
            description="Run performance benchmarks",
            command=["ctest", "--output-on-failure", "-L", "performance"],
            working_directory=self.build_dir,
            timeout=1200,
            required=False,  # Optional
            depends_on=["cmake_build"]
        ))

        # 9. Memory tests (if valgrind available)
        if shutil.which("valgrind"):
            suites.append(TestSuite(
                name="memory_tests",
                description="Run memory tests with valgrind",
                command=[
                    "ctest", "--output-on-failure",
                    "-T", "memcheck",
                    "-L", "unit"
                ],
                working_directory=self.build_dir,
                timeout=1800,
                required=False,
                depends_on=["cmake_build"]
            ))

        return suites

    def run_test_suite(self, suite: TestSuite) -> TestResult:
        """Run a single test suite."""
        self.logger.info(f"Running test suite: {suite.name}")
        self.logger.debug(f"Command: {' '.join(suite.command)}")
        self.logger.debug(f"Working directory: {suite.working_directory}")

        start_time = time.time()

        try:
            result = subprocess.run(
                suite.command,
                cwd=suite.working_directory,
                capture_output=True,
                text=True,
                timeout=suite.timeout
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            return TestResult(
                suite_name=suite.name,
                success=success,
                duration=duration,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                suite_name=suite.name,
                success=False,
                duration=duration,
                error=f"Test suite timed out after {suite.timeout} seconds",
                return_code=-1
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                suite_name=suite.name,
                success=False,
                duration=duration,
                error=f"Test suite failed with exception: {e}",
                return_code=-1
            )

    def resolve_dependencies(self, suites: List[TestSuite]) -> List[List[TestSuite]]:
        """Resolve test suite dependencies and create execution order."""
        # Create dependency graph
        suite_map = {suite.name: suite for suite in suites}

        # Topological sort to resolve dependencies
        execution_order = []
        visited = set()
        temp_visited = set()

        def visit(suite_name: str, current_batch: List[TestSuite]):
            if suite_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {suite_name}")
            if suite_name in visited:
                return

            temp_visited.add(suite_name)
            suite = suite_map[suite_name]

            # Visit all dependencies first
            for dep_name in suite.depends_on:
                if dep_name in suite_map:
                    visit(dep_name, current_batch)

            temp_visited.remove(suite_name)
            visited.add(suite_name)

            # Add to current batch if no dependencies or all dependencies resolved
            deps_resolved = all(dep in visited for dep in suite.depends_on if dep in suite_map)
            if deps_resolved:
                current_batch.append(suite)

        # Group suites into batches that can run in parallel
        batches = []
        remaining_suites = list(suites)

        while remaining_suites:
            current_batch = []

            for suite in remaining_suites[:]:
                deps_resolved = all(dep in visited for dep in suite.depends_on if dep in suite_map)
                if deps_resolved:
                    current_batch.append(suite)
                    remaining_suites.remove(suite)
                    visited.add(suite.name)

            if not current_batch:
                # No progress made, check for unresolvable dependencies
                unresolved = []
                for suite in remaining_suites:
                    missing_deps = [dep for dep in suite.depends_on
                                  if dep not in suite_map and dep not in visited]
                    if missing_deps:
                        unresolved.append((suite.name, missing_deps))

                if unresolved:
                    self.logger.warning(f"Skipping suites with missing dependencies: {unresolved}")
                    break
                else:
                    raise ValueError("Circular dependency detected in test suites")

            batches.append(current_batch)

        return batches

    def run_all_tests(self, max_parallel: int = 2) -> List[TestResult]:
        """Run all test suites with dependency resolution."""
        suites = self.create_test_suites()
        execution_batches = self.resolve_dependencies(suites)

        all_results = []
        failed_suites = set()

        self.logger.info(f"Executing {len(suites)} test suites in {len(execution_batches)} batches")

        for batch_idx, batch in enumerate(execution_batches):
            self.logger.info(f"Running batch {batch_idx + 1}/{len(execution_batches)}: "
                           f"{[suite.name for suite in batch]}")

            # Check if any dependencies failed
            batch_to_run = []
            for suite in batch:
                if any(dep in failed_suites for dep in suite.depends_on):
                    self.logger.warning(f"Skipping {suite.name} due to failed dependency")
                    all_results.append(TestResult(
                        suite_name=suite.name,
                        success=False,
                        duration=0.0,
                        error="Skipped due to failed dependency"
                    ))
                else:
                    batch_to_run.append(suite)

            if not batch_to_run:
                continue

            # Run batch in parallel
            if len(batch_to_run) == 1 or max_parallel == 1:
                # Sequential execution
                for suite in batch_to_run:
                    result = self.run_test_suite(suite)
                    all_results.append(result)

                    if not result.success:
                        failed_suites.add(suite.name)
                        if suite.required:
                            self.logger.error(f"Required test suite {suite.name} failed")
                        else:
                            self.logger.warning(f"Optional test suite {suite.name} failed")
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=min(max_parallel, len(batch_to_run))) as executor:
                    future_to_suite = {
                        executor.submit(self.run_test_suite, suite): suite
                        for suite in batch_to_run
                    }

                    for future in as_completed(future_to_suite):
                        suite = future_to_suite[future]
                        try:
                            result = future.result()
                            all_results.append(result)

                            if not result.success:
                                failed_suites.add(suite.name)
                                if suite.required:
                                    self.logger.error(f"Required test suite {suite.name} failed")
                                else:
                                    self.logger.warning(f"Optional test suite {suite.name} failed")
                        except Exception as e:
                            self.logger.error(f"Test suite {suite.name} raised exception: {e}")
                            all_results.append(TestResult(
                                suite_name=suite.name,
                                success=False,
                                duration=0.0,
                                error=str(e)
                            ))
                            failed_suites.add(suite.name)

        return all_results

    def generate_comprehensive_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_suites = len(results)
        successful_suites = sum(1 for r in results if r.success)
        failed_suites = total_suites - successful_suites
        total_duration = sum(r.duration for r in results)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_test_suites": total_suites,
                "successful_suites": successful_suites,
                "failed_suites": failed_suites,
                "success_rate": (successful_suites / total_suites * 100) if total_suites > 0 else 0,
                "total_duration_minutes": total_duration / 60.0,
                "overall_status": "PASS" if failed_suites == 0 else "FAIL"
            },
            "results": [
                {
                    "suite_name": r.suite_name,
                    "success": r.success,
                    "duration": r.duration,
                    "return_code": r.return_code,
                    "output_length": len(r.output),
                    "error_length": len(r.error),
                    "has_output": bool(r.output.strip()),
                    "has_error": bool(r.error.strip())
                } for r in results
            ],
            "failed_suites": [
                {
                    "suite_name": r.suite_name,
                    "duration": r.duration,
                    "return_code": r.return_code,
                    "error_snippet": r.error[:500] if r.error else "",
                    "output_snippet": r.output[-500:] if r.output else ""
                } for r in results if not r.success
            ]
        }

        # Save detailed results
        for result in results:
            suite_dir = self.output_dir / "detailed" / result.suite_name
            suite_dir.mkdir(parents=True, exist_ok=True)

            with open(suite_dir / "result.json", 'w') as f:
                json.dump({
                    "suite_name": result.suite_name,
                    "success": result.success,
                    "duration": result.duration,
                    "return_code": result.return_code
                }, f, indent=2)

            if result.output:
                with open(suite_dir / "stdout.txt", 'w') as f:
                    f.write(result.output)

            if result.error:
                with open(suite_dir / "stderr.txt", 'w') as f:
                    f.write(result.error)

        # Save comprehensive report
        report_file = self.output_dir / f"comprehensive_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate text summary
        self._generate_text_summary(report)

        return report

    def _generate_text_summary(self, report: Dict[str, Any]):
        """Generate human-readable summary."""
        summary_file = self.output_dir / f"test_summary_{int(time.time())}.txt"

        with open(summary_file, 'w') as f:
            f.write("GEMMA.CPP COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Test Date: {report['timestamp']}\n")
            f.write(f"Overall Status: {report['summary']['overall_status']}\n")
            f.write(f"Total Duration: {report['summary']['total_duration_minutes']:.1f} minutes\n\n")

            f.write("SUMMARY:\n")
            f.write("-" * 10 + "\n")
            f.write(f"Total Test Suites: {report['summary']['total_test_suites']}\n")
            f.write(f"Successful: {report['summary']['successful_suites']}\n")
            f.write(f"Failed: {report['summary']['failed_suites']}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n\n")

            f.write("TEST SUITE RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in report['results']:
                status = "✓" if result['success'] else "✗"
                f.write(f"  {status} {result['suite_name']} ({result['duration']:.1f}s)")
                if not result['success']:
                    f.write(f" - Exit code: {result['return_code']}")
                f.write("\n")

            if report['failed_suites']:
                f.write("\nFAILED SUITES DETAILS:\n")
                f.write("-" * 25 + "\n")
                for failed in report['failed_suites']:
                    f.write(f"\n{failed['suite_name']}:\n")
                    f.write(f"  Duration: {failed['duration']:.1f}s\n")
                    f.write(f"  Exit Code: {failed['return_code']}\n")
                    if failed['error_snippet']:
                        f.write(f"  Error: {failed['error_snippet']}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemma.cpp Comprehensive Test Suite")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Path to Gemma.cpp project root")
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "build-comprehensive",
                       help="Build directory for tests")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "test-results",
                       help="Output directory for reports")
    parser.add_argument("--parallel", type=int, default=2,
                       help="Maximum parallel test suites")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build directory before testing")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--suite", action="append",
                       help="Run only specific test suite(s)")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting comprehensive Gemma.cpp test suite...")

    # Clean build directory if requested
    if args.clean and args.build_dir.exists():
        logger.info(f"Cleaning build directory: {args.build_dir}")
        shutil.rmtree(args.build_dir)

    # Run tests
    runner = ComprehensiveTestRunner(args.project_root, args.build_dir, args.output_dir)
    results = runner.run_all_tests(args.parallel)

    # Generate report
    report = runner.generate_comprehensive_report(results)

    # Print summary
    print(f"\nCOMPREHENSIVE TEST SUMMARY:")
    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Test Suites: {report['summary']['successful_suites']}/{report['summary']['total_test_suites']} passed")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Total Duration: {report['summary']['total_duration_minutes']:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")

    # Return appropriate exit code
    return 0 if report['summary']['overall_status'] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())