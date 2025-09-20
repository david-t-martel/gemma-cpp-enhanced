#!/usr/bin/env python3
"""
Enhanced Gemma.cpp Compilation Test Suite
=========================================

This script provides comprehensive testing of all hardware backend configurations,
toolchain detection, and compilation verification for the enhanced Gemma.cpp project.

Features:
- Automatic detection of available hardware backends and SDKs
- Parallel compilation testing across all configurations
- Detailed compilation reports with timing and error analysis
- Dependency verification and environment validation
- Platform-specific toolchain detection
"""

import os
import sys
import subprocess
import json
import time
import threading
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging


@dataclass
class CompilerInfo:
    """Information about detected compilers."""
    name: str
    path: str
    version: str
    target_arch: str
    is_available: bool = True


@dataclass
class BackendInfo:
    """Information about hardware backend availability."""
    name: str
    cmake_option: str
    required_packages: List[str]
    detection_commands: List[str]
    is_available: bool = False
    version: str = ""
    notes: str = ""


@dataclass
class CompilationResult:
    """Result of a compilation attempt."""
    backend: str
    compiler: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
    binary_size: int = 0
    warnings: List[str] = field(default_factory=list)


class ToolchainDetector:
    """Detects available compilers and SDKs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compilers: List[CompilerInfo] = []
        self.backends: List[BackendInfo] = []

    def detect_compilers(self) -> List[CompilerInfo]:
        """Detect available C++ compilers."""
        compilers = []

        # Common compiler candidates
        candidates = [
            ("gcc", "g++", ["-dumpversion"]),
            ("clang", "clang++", ["--version"]),
            ("msvc", "cl.exe", []),  # Windows only
            ("icc", "icpc", ["--version"]),  # Intel C++
            ("icx", "icx", ["--version"]),   # Intel oneAPI
        ]

        for name, executable, version_cmd in candidates:
            try:
                # Try to find the executable
                path = shutil.which(executable)
                if not path:
                    continue

                # Get version information
                if version_cmd:
                    result = subprocess.run(
                        [executable] + version_cmd,
                        capture_output=True, text=True, timeout=10
                    )
                    version = result.stdout.strip() if result.returncode == 0 else "unknown"
                else:
                    version = "unknown"

                # Get target architecture
                if name == "gcc" or name == "clang":
                    arch_result = subprocess.run(
                        [executable, "-dumpmachine"],
                        capture_output=True, text=True, timeout=10
                    )
                    target_arch = arch_result.stdout.strip() if arch_result.returncode == 0 else "unknown"
                else:
                    target_arch = "unknown"

                compilers.append(CompilerInfo(
                    name=name,
                    path=path,
                    version=version,
                    target_arch=target_arch
                ))

                self.logger.info(f"Detected compiler: {name} ({version}) at {path}")

            except Exception as e:
                self.logger.debug(f"Failed to detect {name}: {e}")

        self.compilers = compilers
        return compilers

    def detect_backends(self) -> List[BackendInfo]:
        """Detect available hardware acceleration backends."""
        backends = [
            BackendInfo(
                name="CUDA",
                cmake_option="GEMMA_BUILD_CUDA_BACKEND",
                required_packages=["CUDAToolkit"],
                detection_commands=["nvcc --version", "nvidia-smi"]
            ),
            BackendInfo(
                name="SYCL",
                cmake_option="GEMMA_BUILD_SYCL_BACKEND",
                required_packages=["IntelSYCL"],
                detection_commands=["icpx --version", "sycl-ls"]
            ),
            BackendInfo(
                name="Vulkan",
                cmake_option="GEMMA_BUILD_VULKAN_BACKEND",
                required_packages=["Vulkan"],
                detection_commands=["vulkaninfo --summary"]
            ),
            BackendInfo(
                name="OpenCL",
                cmake_option="GEMMA_BUILD_OPENCL_BACKEND",
                required_packages=["OpenCL"],
                detection_commands=["clinfo"]
            ),
            BackendInfo(
                name="Metal",
                cmake_option="GEMMA_BUILD_METAL_BACKEND",
                required_packages=["Metal"],
                detection_commands=["system_profiler SPDisplaysDataType"]
            )
        ]

        for backend in backends:
            backend.is_available = self._check_backend_availability(backend)

        self.backends = backends
        return backends

    def _check_backend_availability(self, backend: BackendInfo) -> bool:
        """Check if a specific backend is available."""
        try:
            # Special handling for Metal (macOS only)
            if backend.name == "Metal":
                import platform
                if platform.system() != "Darwin":
                    backend.notes = "Metal only available on macOS"
                    return False

            # Try detection commands
            available = False
            for cmd in backend.detection_commands:
                try:
                    result = subprocess.run(
                        cmd.split(), capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        available = True
                        # Extract version info if possible
                        if backend.name == "CUDA" and "nvcc" in cmd:
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if "release" in line.lower():
                                    backend.version = line.strip()
                                    break
                        elif backend.name == "Vulkan":
                            backend.version = "SDK Available"
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            if available:
                self.logger.info(f"Backend {backend.name} is available")
            else:
                backend.notes = "Required tools not found in PATH"

            return available

        except Exception as e:
            backend.notes = f"Detection failed: {e}"
            return False


class CompilationTester:
    """Tests compilation of different backend configurations."""

    def __init__(self, project_root: Path, build_dir: Path):
        self.project_root = project_root
        self.build_dir = build_dir
        self.logger = logging.getLogger(__name__)

    def test_configuration(self, backend: BackendInfo, compiler: CompilerInfo) -> CompilationResult:
        """Test compilation of a specific backend configuration."""
        start_time = time.time()

        # Create unique build directory for this configuration
        config_name = f"{backend.name.lower()}_{compiler.name}"
        config_build_dir = self.build_dir / config_name
        config_build_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Configure CMake
            cmake_args = [
                "cmake",
                str(self.project_root),
                f"-DCMAKE_CXX_COMPILER={compiler.path}",
                f"-D{backend.cmake_option}=ON",
                "-DGEMMA_BUILD_ENHANCED_TESTS=ON",
                "-DGEMMA_BUILD_BACKEND_TESTS=ON",
                "-DCMAKE_BUILD_TYPE=Release"
            ]

            self.logger.info(f"Configuring {config_name}...")
            configure_result = subprocess.run(
                cmake_args,
                cwd=config_build_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if configure_result.returncode != 0:
                return CompilationResult(
                    backend=backend.name,
                    compiler=compiler.name,
                    success=False,
                    duration=time.time() - start_time,
                    error=f"CMake configuration failed: {configure_result.stderr}",
                    output=configure_result.stdout
                )

            # Build
            self.logger.info(f"Building {config_name}...")
            build_result = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "-j", "4"],
                cwd=config_build_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )

            duration = time.time() - start_time
            success = build_result.returncode == 0

            # Check for binary creation and get size
            binary_size = 0
            gemma_binary = config_build_dir / "gemma.cpp" / "gemma"
            if gemma_binary.exists():
                binary_size = gemma_binary.stat().st_size

            # Extract warnings
            warnings = []
            for line in build_result.stderr.split('\n'):
                if 'warning:' in line.lower():
                    warnings.append(line.strip())

            return CompilationResult(
                backend=backend.name,
                compiler=compiler.name,
                success=success,
                duration=duration,
                output=build_result.stdout,
                error=build_result.stderr if not success else "",
                binary_size=binary_size,
                warnings=warnings
            )

        except subprocess.TimeoutExpired:
            return CompilationResult(
                backend=backend.name,
                compiler=compiler.name,
                success=False,
                duration=time.time() - start_time,
                error="Compilation timed out"
            )
        except Exception as e:
            return CompilationResult(
                backend=backend.name,
                compiler=compiler.name,
                success=False,
                duration=time.time() - start_time,
                error=f"Unexpected error: {e}"
            )


class CompilationReporter:
    """Generates detailed compilation reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self,
                       compilers: List[CompilerInfo],
                       backends: List[BackendInfo],
                       results: List[CompilationResult]) -> Dict[str, Any]:
        """Generate comprehensive compilation report."""

        # Calculate statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - successful_tests

        avg_duration = sum(r.duration for r in results) / total_tests if total_tests > 0 else 0
        total_warnings = sum(len(r.warnings) for r in results)

        # Group results by backend and compiler
        by_backend = {}
        by_compiler = {}

        for result in results:
            if result.backend not in by_backend:
                by_backend[result.backend] = []
            by_backend[result.backend].append(result)

            if result.compiler not in by_compiler:
                by_compiler[result.compiler] = []
            by_compiler[result.compiler].append(result)

        # Create comprehensive report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_configurations_tested": total_tests,
                "successful_builds": successful_tests,
                "failed_builds": failed_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "average_build_time": avg_duration,
                "total_warnings": total_warnings
            },
            "environment": {
                "detected_compilers": [
                    {
                        "name": c.name,
                        "version": c.version,
                        "path": c.path,
                        "target_arch": c.target_arch
                    } for c in compilers
                ],
                "detected_backends": [
                    {
                        "name": b.name,
                        "available": b.is_available,
                        "version": b.version,
                        "notes": b.notes
                    } for b in backends
                ]
            },
            "results_by_backend": {
                backend: {
                    "total_tests": len(results),
                    "successful": sum(1 for r in results if r.success),
                    "average_duration": sum(r.duration for r in results) / len(results),
                    "details": [
                        {
                            "compiler": r.compiler,
                            "success": r.success,
                            "duration": r.duration,
                            "binary_size": r.binary_size,
                            "warning_count": len(r.warnings),
                            "error": r.error if not r.success else None
                        } for r in results
                    ]
                } for backend, results in by_backend.items()
            },
            "results_by_compiler": {
                compiler: {
                    "total_tests": len(results),
                    "successful": sum(1 for r in results if r.success),
                    "average_duration": sum(r.duration for r in results) / len(results),
                    "details": [
                        {
                            "backend": r.backend,
                            "success": r.success,
                            "duration": r.duration,
                            "binary_size": r.binary_size,
                            "warning_count": len(r.warnings)
                        } for r in results
                    ]
                } for compiler, results in by_compiler.items()
            },
            "detailed_results": [
                {
                    "backend": r.backend,
                    "compiler": r.compiler,
                    "success": r.success,
                    "duration": r.duration,
                    "binary_size": r.binary_size,
                    "warnings": r.warnings,
                    "error": r.error,
                    "output_excerpt": r.output[-500:] if r.output else ""  # Last 500 chars
                } for r in results
            ]
        }

        # Save JSON report
        json_file = self.output_dir / f"compilation_report_{int(time.time())}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        self._generate_text_summary(report, results)

        return report

    def _generate_text_summary(self, report: Dict[str, Any], results: List[CompilationResult]):
        """Generate human-readable text summary."""
        summary_file = self.output_dir / f"compilation_summary_{int(time.time())}.txt"

        with open(summary_file, 'w') as f:
            f.write("GEMMA.CPP COMPILATION TEST REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Test Date: {report['timestamp']}\n")
            f.write(f"Total Configurations Tested: {report['summary']['total_configurations_tested']}\n")
            f.write(f"Successful Builds: {report['summary']['successful_builds']}\n")
            f.write(f"Failed Builds: {report['summary']['failed_builds']}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Average Build Time: {report['summary']['average_build_time']:.1f}s\n")
            f.write(f"Total Warnings: {report['summary']['total_warnings']}\n\n")

            f.write("DETECTED COMPILERS:\n")
            f.write("-" * 20 + "\n")
            for compiler in report['environment']['detected_compilers']:
                f.write(f"  {compiler['name']}: {compiler['version']} ({compiler['path']})\n")

            f.write("\nDETECTED BACKENDS:\n")
            f.write("-" * 20 + "\n")
            for backend in report['environment']['detected_backends']:
                status = "AVAILABLE" if backend['available'] else "UNAVAILABLE"
                f.write(f"  {backend['name']}: {status}")
                if backend['version']:
                    f.write(f" ({backend['version']})")
                if backend['notes']:
                    f.write(f" - {backend['notes']}")
                f.write("\n")

            f.write("\nBUILD RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                status = "SUCCESS" if result.success else "FAILED"
                f.write(f"  {result.backend} + {result.compiler}: {status} ({result.duration:.1f}s)")
                if result.binary_size > 0:
                    f.write(f" - Binary: {result.binary_size // 1024}KB")
                if result.warnings:
                    f.write(f" - Warnings: {len(result.warnings)}")
                f.write("\n")
                if not result.success and result.error:
                    f.write(f"    Error: {result.error[:100]}...\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemma.cpp Compilation Test Suite")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Path to Gemma.cpp project root")
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "build-test",
                       help="Build directory for tests")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "test-reports",
                       help="Output directory for reports")
    parser.add_argument("--parallel", type=int, default=2,
                       help="Number of parallel build jobs")
    parser.add_argument("--backends", nargs="*",
                       choices=["CUDA", "SYCL", "Vulkan", "OpenCL", "Metal"],
                       help="Specific backends to test")
    parser.add_argument("--compilers", nargs="*",
                       choices=["gcc", "clang", "msvc", "icc", "icx"],
                       help="Specific compilers to test")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build directories before testing")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Gemma.cpp compilation test suite...")

    # Clean build directory if requested
    if args.clean and args.build_dir.exists():
        logger.info(f"Cleaning build directory: {args.build_dir}")
        shutil.rmtree(args.build_dir)

    # Detect available tools
    detector = ToolchainDetector()
    compilers = detector.detect_compilers()
    backends = detector.detect_backends()

    if not compilers:
        logger.error("No C++ compilers detected!")
        return 1

    if not any(b.is_available for b in backends):
        logger.warning("No hardware backends detected, testing CPU-only build")
        # Add CPU-only "backend" for testing
        backends = [BackendInfo(
            name="CPU",
            cmake_option="GEMMA_BUILD_BACKENDS=OFF",
            required_packages=[],
            detection_commands=[],
            is_available=True,
            version="CPU-only",
            notes="Standard CPU implementation"
        )]

    # Filter based on command line arguments
    if args.compilers:
        compilers = [c for c in compilers if c.name in args.compilers]
    if args.backends:
        backends = [b for b in backends if b.name in args.backends and b.is_available]

    logger.info(f"Testing {len(compilers)} compilers with {len(backends)} backends")

    # Run compilation tests
    tester = CompilationTester(args.project_root, args.build_dir)
    results = []

    # Create test configurations
    test_configs = [(backend, compiler) for backend in backends for compiler in compilers]

    if args.parallel > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_config = {
                executor.submit(tester.test_configuration, backend, compiler): (backend, compiler)
                for backend, compiler in test_configs
            }

            for future in as_completed(future_to_config):
                backend, compiler = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "SUCCESS" if result.success else "FAILED"
                    logger.info(f"{backend.name} + {compiler.name}: {status} ({result.duration:.1f}s)")
                except Exception as e:
                    logger.error(f"Test failed for {backend.name} + {compiler.name}: {e}")
    else:
        # Sequential execution
        for backend, compiler in test_configs:
            result = tester.test_configuration(backend, compiler)
            results.append(result)
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"{backend.name} + {compiler.name}: {status} ({result.duration:.1f}s)")

    # Generate report
    reporter = CompilationReporter(args.output_dir)
    report = reporter.generate_report(compilers, backends, results)

    # Print summary
    print(f"\nCOMPILATION TEST SUMMARY:")
    print(f"Total Tests: {report['summary']['total_configurations_tested']}")
    print(f"Successful: {report['summary']['successful_builds']}")
    print(f"Failed: {report['summary']['failed_builds']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Reports saved to: {args.output_dir}")

    # Return non-zero exit code if any tests failed
    return 0 if report['summary']['failed_builds'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())