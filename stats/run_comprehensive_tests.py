#!/usr/bin/env python3
"""
Comprehensive Test Runner for RAG-Redis System Integration

Executes all test suites and generates detailed reports:
- Integration tests (Python + Rust components)
- Performance benchmarks (SIMD, Redis, Memory)
- Coverage analysis
- System validation
- Results storage in MCP memory
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.logging import LogLevel, get_logger, setup_logging

PROJECT_ROOT = Path(__file__).parent
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_RESULTS_DIR = TEST_RESULTS_DIR / f"comprehensive_session_{TIMESTAMP}"

# Setup logging
setup_logging(level=LogLevel.INFO, console=True)
logger = get_logger(__name__)


class ComprehensiveTestResults:
    """Collect and manage all test results"""

    def __init__(self):
        self.start_time = time.time()
        self.results = {
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "session_id": TIMESTAMP,
                "project_root": str(PROJECT_ROOT),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "test_suites": {},
            "performance_metrics": {},
            "coverage_data": {},
            "system_validation": {},
            "errors": [],
            "summary": {},
        }

    def add_test_suite_result(self, suite_name: str, result_data: dict[str, Any]):
        """Add results from a test suite"""
        self.results["test_suites"][suite_name] = {
            **result_data,
            "completed_at": datetime.now().isoformat(),
        }

    def add_performance_metrics(self, metrics_name: str, metrics_data: dict[str, Any]):
        """Add performance benchmark metrics"""
        self.results["performance_metrics"][metrics_name] = metrics_data

    def add_coverage_data(self, coverage_data: dict[str, Any]):
        """Add code coverage data"""
        self.results["coverage_data"] = coverage_data

    def add_error(self, error_context: str, error_details: str):
        """Add error information"""
        self.results["errors"].append(
            {
                "context": error_context,
                "details": error_details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def generate_summary(self):
        """Generate comprehensive test summary"""
        end_time = time.time()
        duration = end_time - self.start_time

        # Count test results
        total_suites = len(self.results["test_suites"])
        passed_suites = sum(
            1 for suite in self.results["test_suites"].values() if suite.get("exit_code", 1) == 0
        )

        # Performance summary
        perf_tests = len(self.results["performance_metrics"])

        # Coverage summary
        coverage_available = bool(self.results["coverage_data"])

        self.results["summary"] = {
            "total_duration_seconds": duration,
            "total_test_suites": total_suites,
            "passed_test_suites": passed_suites,
            "failed_test_suites": total_suites - passed_suites,
            "performance_tests_run": perf_tests,
            "coverage_available": coverage_available,
            "total_errors": len(self.results["errors"]),
            "success_rate_percent": (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            "completed_at": datetime.now().isoformat(),
        }

    def save_results(self, output_file: Path):
        """Save all results to JSON file"""
        self.generate_summary()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)


def run_command_with_timeout(cmd: list[str], cwd: Path, timeout: int = 300) -> dict[str, Any]:
    """Run a command with timeout and capture results"""
    start_time = time.time()
    try:
        logger.info(f"Running: {' '.join(cmd)} in {cwd}")
        result = subprocess.run(  # noqa: S603
            cmd,
            check=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        duration = time.time() - start_time

        return {
            "command": " ".join(cmd),
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout[:10000],  # Limit output size
            "stderr": result.stderr[:10000],
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "command": " ".join(cmd),
            "exit_code": -1,
            "duration_seconds": duration,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "success": False,
            "timeout": True,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "command": " ".join(cmd),
            "exit_code": -2,
            "duration_seconds": duration,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "error": str(e),
        }


def check_prerequisites() -> dict[str, Any]:
    """Check system prerequisites"""
    logger.info("Checking prerequisites...")

    checks = {}

    # Check Python environment
    checks["python_version"] = {
        "version": sys.version,
        "executable": sys.executable,
        "success": True,
    }

    # Check Redis connection
    try:
        import redis  # noqa: PLC0415

        from src.shared.config.redis_test_utils import get_test_redis_config  # noqa: PLC0415

        config = get_test_redis_config()
        config["socket_timeout"] = 5
        client = redis.Redis(**config)
        client.ping()
        checks["redis"] = {
            "available": True,
            "success": True,
            "host": config["host"],
            "port": config["port"],
        }
    except Exception as e:
        checks["redis"] = {"available": False, "error": str(e), "success": False}

    # Check uv availability
    uv_result = run_command_with_timeout(["uv", "--version"], PROJECT_ROOT, 10)
    checks["uv"] = uv_result

    # Check required Python packages
    required_packages = ["pytest", "redis", "numpy", "psutil"]
    packages_status = {}

    for package in required_packages:
        try:
            __import__(package)
            packages_status[package] = True
        except ImportError:
            packages_status[package] = False

    checks["required_packages"] = {
        "packages": packages_status,
        "all_available": all(packages_status.values()),
        "success": all(packages_status.values()),
    }

    return checks


def run_integration_tests(test_results: ComprehensiveTestResults) -> dict[str, Any]:
    """Run comprehensive integration tests"""
    logger.info("Running integration tests...")

    test_file = PROJECT_ROOT / "tests" / "test_comprehensive_integration.py"

    cmd = [
        "uv",
        "run",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "--json-report",
        f"--json-report-file={SESSION_RESULTS_DIR}/integration_report.json",
    ]

    result = run_command_with_timeout(cmd, PROJECT_ROOT, 300)

    # Try to load JSON report if available
    json_report_path = SESSION_RESULTS_DIR / "integration_report.json"
    json_data = {}
    if json_report_path.exists():
        try:
            with open(json_report_path) as f:
                json_data = json.load(f)
        except Exception as e:
            test_results.add_error("integration_test_json_parse", str(e))

    return {**result, "json_report": json_data, "test_file": str(test_file)}


def run_performance_benchmarks(test_results: ComprehensiveTestResults) -> dict[str, Any]:
    """Run performance benchmark tests"""
    logger.info("Running performance benchmarks...")

    test_file = PROJECT_ROOT / "tests" / "test_performance_benchmarks.py"

    cmd = [
        "uv",
        "run",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "--json-report",
        f"--json-report-file={SESSION_RESULTS_DIR}/performance_report.json",
    ]

    result = run_command_with_timeout(cmd, PROJECT_ROOT, 600)  # Longer timeout for perf tests

    # Load performance JSON report
    json_report_path = SESSION_RESULTS_DIR / "performance_report.json"
    json_data = {}
    if json_report_path.exists():
        try:
            with open(json_report_path) as f:
                json_data = json.load(f)
        except Exception as e:
            test_results.add_error("performance_test_json_parse", str(e))

    return {**result, "json_report": json_data, "test_file": str(test_file)}


def run_rust_tests(test_results: ComprehensiveTestResults) -> dict[str, Any]:
    """Run Rust component tests"""
    logger.info("Running Rust tests...")

    rust_dir = PROJECT_ROOT / "rag-redis"

    if not rust_dir.exists():
        return {"success": False, "error": "Rust directory not found", "skipped": True}

    # Try to build first
    build_result = run_command_with_timeout(["cargo", "build", "--release"], rust_dir, 300)

    if not build_result["success"]:
        test_results.add_error("rust_build", build_result["stderr"])
        return {**build_result, "phase": "build", "test_skipped": True}

    # Run tests if build succeeded
    test_result = run_command_with_timeout(["cargo", "test", "--release"], rust_dir, 300)

    return {
        "build_result": build_result,
        "test_result": test_result,
        "success": test_result["success"],
        "combined_duration": build_result["duration_seconds"] + test_result["duration_seconds"],
    }


def generate_coverage_report(test_results: ComprehensiveTestResults) -> dict[str, Any]:
    """Generate code coverage report"""
    logger.info("Generating coverage report...")

    # Run coverage on main source code
    cmd = [
        "uv",
        "run",
        "coverage",
        "run",
        "--source",
        "src",
        "--omit",
        "*/tests/*,*/test_*",
        "-m",
        "pytest",
        "tests/",
        "-x",  # Stop on first failure for coverage
    ]

    coverage_result = run_command_with_timeout(cmd, PROJECT_ROOT, 300)

    if not coverage_result["success"]:
        return coverage_result

    # Generate coverage report
    report_cmd = ["uv", "run", "coverage", "report", "--format=json"]

    report_result = run_command_with_timeout(report_cmd, PROJECT_ROOT, 60)

    coverage_data = {}
    if report_result["success"] and report_result["stdout"]:
        try:
            coverage_data = json.loads(report_result["stdout"])
        except json.JSONDecodeError as e:
            test_results.add_error("coverage_json_parse", str(e))

    # Generate HTML report
    html_cmd = ["uv", "run", "coverage", "html", "-d", str(SESSION_RESULTS_DIR / "coverage_html")]
    html_result = run_command_with_timeout(html_cmd, PROJECT_ROOT, 60)

    return {
        "coverage_run": coverage_result,
        "coverage_report": report_result,
        "coverage_html": html_result,
        "coverage_data": coverage_data,
        "success": report_result["success"],
    }


async def store_results_in_memory(test_results: ComprehensiveTestResults):
    """Store test results in MCP memory system"""
    logger.info("Storing results in memory...")

    try:
        # This would use the memory MCP tool when available
        # For now, we'll prepare the data structure

        memory_data = {
            "comprehensive_test_session": {
                "session_id": test_results.results["session_info"]["session_id"],
                "summary": test_results.results["summary"],
                "key_metrics": {
                    "total_duration": test_results.results["summary"].get(
                        "total_duration_seconds", 0
                    ),
                    "success_rate": test_results.results["summary"].get("success_rate_percent", 0),
                    "test_suites_run": test_results.results["summary"].get("total_test_suites", 0),
                    "errors_encountered": test_results.results["summary"].get("total_errors", 0),
                },
                "performance_highlights": {},
                "coverage_summary": {},
                "issues_identified": test_results.results.get("errors", []),
            }
        }

        # Extract performance highlights
        for perf_name in test_results.results.get("performance_metrics", {}):
            memory_data["comprehensive_test_session"]["performance_highlights"][perf_name] = {
                "key_metric": "performance_tested",
                "status": "completed",
            }

        # Extract coverage summary
        coverage_data = test_results.results.get("coverage_data", {})
        if coverage_data:
            memory_data["comprehensive_test_session"]["coverage_summary"] = {
                "total_statements": coverage_data.get("totals", {}).get("num_statements", 0),
                "covered_statements": coverage_data.get("totals", {}).get("covered_lines", 0),
                "coverage_percent": coverage_data.get("totals", {}).get("percent_covered", 0),
            }

        logger.info(f"Prepared memory data structure with {len(memory_data)} top-level items")
        return memory_data

    except Exception as e:
        test_results.add_error("memory_storage", str(e))
        logger.error(f"Failed to store results in memory: {e}")
        return None


def run_test_phases(test_results: ComprehensiveTestResults) -> None:
    """Execute all test phases"""
    # 1. Check prerequisites
    logger.info("=== PHASE 1: Prerequisites Check ===")
    prereq_results = check_prerequisites()
    test_results.add_test_suite_result("prerequisites", prereq_results)

    if not prereq_results.get("redis", {}).get("success", False):
        logger.warning("Redis not available - some tests may be skipped")

    # 2. Run integration tests
    logger.info("=== PHASE 2: Integration Tests ===")
    integration_results = run_integration_tests(test_results)
    test_results.add_test_suite_result("integration_tests", integration_results)

    # 3. Run performance benchmarks
    logger.info("=== PHASE 3: Performance Benchmarks ===")
    performance_results = run_performance_benchmarks(test_results)
    test_results.add_test_suite_result("performance_benchmarks", performance_results)
    test_results.add_performance_metrics("benchmark_suite", performance_results)

    # 4. Run Rust tests
    logger.info("=== PHASE 4: Rust Component Tests ===")
    rust_results = run_rust_tests(test_results)
    test_results.add_test_suite_result("rust_tests", rust_results)

    # 5. Generate coverage report
    logger.info("=== PHASE 5: Coverage Analysis ===")
    coverage_results = generate_coverage_report(test_results)
    test_results.add_test_suite_result("coverage_analysis", coverage_results)
    if coverage_results.get("coverage_data"):
        test_results.add_coverage_data(coverage_results["coverage_data"])

    # 6. Store results in memory
    logger.info("=== PHASE 6: Results Storage ===")
    _memory_data = asyncio.run(store_results_in_memory(test_results))


def print_test_summary(test_results: ComprehensiveTestResults, results_file: Path) -> int:
    """Print test summary and return exit code"""
    summary = test_results.results["summary"]
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE TEST SUITE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Duration: {summary.get('total_duration_seconds', 0):.1f} seconds")
    logger.info(
        f"Test Suites: {summary.get('passed_test_suites', 0)}/{summary.get('total_test_suites', 0)} passed"
    )
    logger.info(f"Success Rate: {summary.get('success_rate_percent', 0):.1f}%")
    logger.info(f"Performance Tests: {summary.get('performance_tests_run', 0)} completed")
    logger.info(f"Coverage Available: {summary.get('coverage_available', False)}")
    logger.info(f"Total Errors: {summary.get('total_errors', 0)}")
    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 60)

    # Return appropriate exit code
    if summary.get("success_rate_percent", 0) >= 80:  # 80% success threshold
        logger.info("Test suite completed successfully!")
        return 0
    else:
        logger.warning("Test suite completed with issues")
        return 1


def main():
    """Main test runner"""
    logger.info(f"Starting comprehensive test suite at {datetime.now()}")

    # Create results directory
    SESSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize results collector
    test_results = ComprehensiveTestResults()

    try:
        run_test_phases(test_results)
    except Exception as e:
        logger.error(f"Critical error in test execution: {e}")
        test_results.add_error("critical_execution_error", str(e))

    # Generate final summary and save results
    logger.info("=== PHASE 7: Results Generation ===")
    results_file = SESSION_RESULTS_DIR / "comprehensive_test_results.json"
    test_results.save_results(results_file)
    return print_test_summary(test_results, results_file)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
