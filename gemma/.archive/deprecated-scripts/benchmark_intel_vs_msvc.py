#!/usr/bin/env python3
"""
Intel vs MSVC Performance Benchmark Comparison for Gemma.cpp

This script runs comprehensive benchmarks comparing Intel ICX optimized builds
against MSVC builds for Gemma inference performance.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import statistics


@dataclass
class BenchmarkResult:
    """Performance benchmark result data structure"""
    build_type: str              # "intel" or "msvc"
    model_name: str              # e.g., "gemma-2b", "gemma-4b"
    tokens_per_second: float     # Inference speed
    total_time_ms: float         # Total benchmark time
    memory_usage_mb: float       # Peak memory usage
    cpu_utilization: float       # Average CPU utilization %
    simd_instructions: int       # SIMD instruction count (if available)
    cache_misses: int            # L3 cache misses (if available)
    compiler_version: str        # Compiler version used
    optimization_flags: str      # Optimization flags applied
    timestamp: str               # Benchmark timestamp


class GemmaBenchmarkRunner:
    """Manages benchmark execution for different Gemma builds"""

    def __init__(self, gemma_root: Path):
        self.gemma_root = Path(gemma_root)
        self.models_dir = self.gemma_root.parent / ".models"
        self.results: List[BenchmarkResult] = []

        # Define available models for benchmarking
        self.models = {
            "gemma-2b": {
                "weights": self.models_dir / "gemma-gemmacpp-2b-it-v3" / "2b-it.sbs",
                "tokenizer": self.models_dir / "gemma-gemmacpp-2b-it-v3" / "tokenizer.spm"
            },
            "gemma-4b": {
                "weights": self.models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1" / "4b-it-sfp.sbs",
                "tokenizer": self.models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1" / "tokenizer.spm"
            }
        }

        # Define build configurations
        self.builds = {
            "msvc": {
                "build_dir": self.gemma_root / "build",
                "executable": "single_benchmark.exe"
            },
            "intel": {
                "build_dir": self.gemma_root / "build-intel-optimized",
                "executable": "single_benchmark.exe"
            }
        }

    def verify_setup(self) -> bool:
        """Verify that models and build directories exist"""
        print("🔍 Verifying benchmark setup...")

        # Check models
        missing_models = []
        for model_name, paths in self.models.items():
            if not paths["weights"].exists():
                missing_models.append(f"{model_name} weights: {paths['weights']}")
            if not paths["tokenizer"].exists():
                missing_models.append(f"{model_name} tokenizer: {paths['tokenizer']}")

        if missing_models:
            print("❌ Missing model files:")
            for missing in missing_models:
                print(f"   - {missing}")
            return False

        # Check build directories
        missing_builds = []
        for build_name, config in self.builds.items():
            build_dir = config["build_dir"]
            executable = build_dir / "Release" / config["executable"]

            if not build_dir.exists():
                missing_builds.append(f"{build_name} build directory: {build_dir}")
            elif not executable.exists():
                # Try different locations
                alt_paths = [
                    build_dir / config["executable"],
                    build_dir / "Debug" / config["executable"],
                    build_dir / "gemma.cpp" / "Release" / config["executable"]
                ]

                found = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        config["executable_path"] = alt_path
                        found = True
                        break

                if not found:
                    missing_builds.append(f"{build_name} executable: {executable}")
            else:
                config["executable_path"] = executable

        if missing_builds:
            print("❌ Missing build artifacts:")
            for missing in missing_builds:
                print(f"   - {missing}")
            return False

        print("✅ All required files found")
        return True

    def run_single_benchmark(self, build_type: str, model_name: str, iterations: int = 3) -> Optional[BenchmarkResult]:
        """Run benchmark for a single build/model combination"""
        build_config = self.builds[build_type]
        model_config = self.models[model_name]

        executable = build_config.get("executable_path")
        if not executable:
            print(f"❌ Executable not found for {build_type} build")
            return None

        print(f"🏃 Running {build_type.upper()} benchmark for {model_name}...")

        # Prepare benchmark command
        cmd = [
            str(executable),
            "--weights", str(model_config["weights"]),
            "--tokenizer", str(model_config["tokenizer"]),
            "--max_tokens", "100",  # Fixed token count for comparison
            "--json_output"  # Request JSON output if supported
        ]

        results = []
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}...")

            try:
                # Run benchmark with timeout
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=executable.parent
                )
                end_time = time.time()

                if result.returncode != 0:
                    print(f"⚠️  Benchmark failed (iteration {i+1}): {result.stderr}")
                    continue

                # Parse benchmark output
                tokens_per_second = self._extract_tokens_per_second(result.stdout)
                memory_usage = self._extract_memory_usage(result.stdout)

                iteration_result = {
                    "tokens_per_second": tokens_per_second,
                    "total_time_ms": (end_time - start_time) * 1000,
                    "memory_usage_mb": memory_usage,
                    "raw_output": result.stdout
                }

                results.append(iteration_result)
                print(f"    Speed: {tokens_per_second:.2f} tokens/sec")

            except subprocess.TimeoutExpired:
                print(f"⚠️  Benchmark timed out (iteration {i+1})")
                continue
            except Exception as e:
                print(f"⚠️  Benchmark error (iteration {i+1}): {e}")
                continue

        if not results:
            print(f"❌ All benchmark iterations failed for {build_type} {model_name}")
            return None

        # Calculate statistics from multiple iterations
        tokens_per_sec_values = [r["tokens_per_second"] for r in results if r["tokens_per_second"] > 0]
        time_values = [r["total_time_ms"] for r in results]
        memory_values = [r["memory_usage_mb"] for r in results if r["memory_usage_mb"] > 0]

        if not tokens_per_sec_values:
            print(f"❌ No valid token/sec measurements for {build_type} {model_name}")
            return None

        # Create benchmark result with averaged values
        benchmark_result = BenchmarkResult(
            build_type=build_type,
            model_name=model_name,
            tokens_per_second=statistics.mean(tokens_per_sec_values),
            total_time_ms=statistics.mean(time_values),
            memory_usage_mb=statistics.mean(memory_values) if memory_values else 0.0,
            cpu_utilization=0.0,  # TODO: Implement CPU monitoring
            simd_instructions=0,   # TODO: Implement SIMD counting
            cache_misses=0,        # TODO: Implement cache monitoring
            compiler_version=self._get_compiler_version(build_type),
            optimization_flags=self._get_optimization_flags(build_type),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        return benchmark_result

    def _extract_tokens_per_second(self, output: str) -> float:
        """Extract tokens per second from benchmark output"""
        try:
            # Look for various possible output formats
            patterns = [
                "tokens per second:",
                "tok/s:",
                "speed:",
                "throughput:"
            ]

            lines = output.lower().split('\n')
            for line in lines:
                for pattern in patterns:
                    if pattern in line:
                        # Extract number after the pattern
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            number_part = parts[1].strip().split()[0]
                            try:
                                return float(number_part)
                            except ValueError:
                                continue

            # Fallback: look for any floating point number that might be tokens/sec
            import re
            numbers = re.findall(r'\d+\.?\d*', output)
            if numbers:
                # Return a reasonable-looking number (not too small, not too large)
                for num_str in numbers:
                    num = float(num_str)
                    if 1 < num < 10000:  # Reasonable range for tokens/sec
                        return num

            return 0.0

        except Exception:
            return 0.0

    def _extract_memory_usage(self, output: str) -> float:
        """Extract memory usage from benchmark output"""
        try:
            # Look for memory usage patterns
            patterns = [
                "memory usage:",
                "peak memory:",
                "mem:",
                "mb used:"
            ]

            lines = output.lower().split('\n')
            for line in lines:
                for pattern in patterns:
                    if pattern in line and "mb" in line:
                        # Extract number before "mb"
                        parts = line.split("mb")[0].split()
                        if parts:
                            try:
                                return float(parts[-1])
                            except ValueError:
                                continue

            return 0.0

        except Exception:
            return 0.0

    def _get_compiler_version(self, build_type: str) -> str:
        """Get compiler version information"""
        if build_type == "intel":
            try:
                result = subprocess.run(
                    ["C:/users/david/.local/bin/intel-icx.cmd", "--version"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            except:
                pass
            return "Intel ICX (version unknown)"
        else:
            return "MSVC 19.44.35214.0"

    def _get_optimization_flags(self, build_type: str) -> str:
        """Get optimization flags used for the build"""
        if build_type == "intel":
            return "-O3 -xHost -march=native -mtune=native -ffast-math -fopenmp -mkl=parallel"
        else:
            return "/O2 /LTCG /arch:AVX2"

    def run_all_benchmarks(self, iterations: int = 3) -> List[BenchmarkResult]:
        """Run benchmarks for all build/model combinations"""
        print("🚀 Starting comprehensive benchmark comparison...")
        print(f"   Iterations per test: {iterations}")
        print(f"   Models: {list(self.models.keys())}")
        print(f"   Builds: {list(self.builds.keys())}")
        print()

        all_results = []

        for model_name in self.models.keys():
            for build_type in self.builds.keys():
                result = self.run_single_benchmark(build_type, model_name, iterations)
                if result:
                    all_results.append(result)
                    self.results.append(result)
                print()

        return all_results

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate comprehensive benchmark comparison report"""
        if not self.results:
            return "No benchmark results available"

        report = []
        report.append("=" * 80)
        report.append("INTEL vs MSVC GEMMA.CPP PERFORMANCE COMPARISON")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total benchmarks: {len(self.results)}")
        report.append("")

        # Group results by model
        by_model = {}
        for result in self.results:
            if result.model_name not in by_model:
                by_model[result.model_name] = {}
            by_model[result.model_name][result.build_type] = result

        # Generate detailed comparison for each model
        for model_name, builds in by_model.items():
            report.append(f"📊 {model_name.upper()} PERFORMANCE")
            report.append("-" * 50)

            if "intel" in builds and "msvc" in builds:
                intel_result = builds["intel"]
                msvc_result = builds["msvc"]

                # Calculate performance improvements
                speed_improvement = ((intel_result.tokens_per_second - msvc_result.tokens_per_second)
                                   / msvc_result.tokens_per_second * 100)
                time_improvement = ((msvc_result.total_time_ms - intel_result.total_time_ms)
                                  / msvc_result.total_time_ms * 100)

                report.append(f"Intel ICX:  {intel_result.tokens_per_second:.2f} tokens/sec")
                report.append(f"MSVC:       {msvc_result.tokens_per_second:.2f} tokens/sec")
                report.append(f"Speedup:    {speed_improvement:+.1f}% ({intel_result.tokens_per_second/msvc_result.tokens_per_second:.2f}x)")
                report.append("")
                report.append(f"Intel Time: {intel_result.total_time_ms:.0f}ms")
                report.append(f"MSVC Time:  {msvc_result.total_time_ms:.0f}ms")
                report.append(f"Time Saved: {time_improvement:+.1f}%")
                report.append("")

                if intel_result.memory_usage_mb > 0 and msvc_result.memory_usage_mb > 0:
                    memory_diff = ((intel_result.memory_usage_mb - msvc_result.memory_usage_mb)
                                 / msvc_result.memory_usage_mb * 100)
                    report.append(f"Intel Mem:  {intel_result.memory_usage_mb:.0f}MB")
                    report.append(f"MSVC Mem:   {msvc_result.memory_usage_mb:.0f}MB")
                    report.append(f"Memory Δ:   {memory_diff:+.1f}%")

            else:
                # Show individual results if comparison not available
                for build_type, result in builds.items():
                    report.append(f"{build_type.upper()}: {result.tokens_per_second:.2f} tokens/sec")

            report.append("")

        # Summary statistics
        intel_results = [r for r in self.results if r.build_type == "intel"]
        msvc_results = [r for r in self.results if r.build_type == "msvc"]

        if intel_results and msvc_results:
            avg_intel_speed = statistics.mean([r.tokens_per_second for r in intel_results])
            avg_msvc_speed = statistics.mean([r.tokens_per_second for r in msvc_results])
            overall_improvement = ((avg_intel_speed - avg_msvc_speed) / avg_msvc_speed * 100)

            report.append("📈 OVERALL SUMMARY")
            report.append("-" * 50)
            report.append(f"Average Intel Speed: {avg_intel_speed:.2f} tokens/sec")
            report.append(f"Average MSVC Speed:  {avg_msvc_speed:.2f} tokens/sec")
            report.append(f"Overall Improvement: {overall_improvement:+.1f}%")
            report.append("")

        # Compiler information
        report.append("🔧 BUILD CONFIGURATION")
        report.append("-" * 50)
        if intel_results:
            intel_result = intel_results[0]
            report.append(f"Intel Compiler: {intel_result.compiler_version}")
            report.append(f"Intel Flags:    {intel_result.optimization_flags}")
        if msvc_results:
            msvc_result = msvc_results[0]
            report.append(f"MSVC Compiler:  {msvc_result.compiler_version}")
            report.append(f"MSVC Flags:     {msvc_result.optimization_flags}")

        report_text = "\n".join(report)

        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_text)
            print(f"📄 Report saved to: {output_file}")

        return report_text

    def export_json(self, output_file: Path):
        """Export benchmark results as JSON"""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gemma_root": str(self.gemma_root),
            "results": [asdict(result) for result in self.results]
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📊 JSON data exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Intel vs MSVC performance benchmark for Gemma.cpp"
    )
    parser.add_argument(
        "--gemma-root",
        type=Path,
        default=Path(__file__).parent,
        help="Path to Gemma.cpp root directory"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations per test"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "benchmark_results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gemma-2b", "gemma-4b"],
        default=["gemma-2b", "gemma-4b"],
        help="Models to benchmark"
    )

    args = parser.parse_args()

    # Initialize benchmark runner
    runner = GemmaBenchmarkRunner(args.gemma_root)

    # Filter models if specified
    if args.models != ["gemma-2b", "gemma-4b"]:
        runner.models = {k: v for k, v in runner.models.items() if k in args.models}

    # Verify setup
    if not runner.verify_setup():
        print("❌ Benchmark setup verification failed")
        sys.exit(1)

    print()

    # Run benchmarks
    try:
        results = runner.run_all_benchmarks(args.iterations)

        if not results:
            print("❌ No successful benchmark results")
            sys.exit(1)

        print("✅ All benchmarks completed successfully!")
        print()

        # Generate and display report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = args.output_dir / f"intel_vs_msvc_report_{timestamp}.txt"
        json_file = args.output_dir / f"intel_vs_msvc_results_{timestamp}.json"

        report = runner.generate_report(report_file)
        runner.export_json(json_file)

        print(report)

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()