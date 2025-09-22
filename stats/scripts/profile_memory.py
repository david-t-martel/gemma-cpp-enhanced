#!/usr/bin/env python3
"""
Profile memory usage of the application.

This script provides comprehensive memory profiling capabilities including
real-time monitoring, leak detection, and optimization recommendations.
"""

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Configure logging
s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("profile_memory.log"),
    ],
)

@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of system memory
    available_mb: float  # Available system memory
    tracemalloc_mb: float | None = None  # Tracemalloc tracked memory
    top_traces: list[dict] | None = None  # Top memory allocations

    @property
    def timestamp_str(self) -> str:
        """Get formatted timestamp."""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

@dataclass
class MemoryLeak:
    """Represents a potential memory leak."""

    location: str
    size_mb: float
    growth_rate_mb_s: float
    duration_seconds: float
    allocation_count: int
    stack_trace: list[str]

class MemoryProfiler:
    """Advanced memory profiler for Python applications."""

    def __init__(self, project_root: Path):
        """Initialize the memory profiler."""
        self.project_root = project_root
        self.process = psutil.Process()
        self.snapshots: list[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: threading.Thread | None = None
        self.tracemalloc_enabled = False

        # Profiling configuration
        self.snapshot_interval = 1.0  # seconds
        self.max_snapshots = 1000
        self.memory_threshold_mb = 100  # Alert threshold

        # Leak detection
        self.leak_detection_window = 60  # seconds
        self.leak_threshold_mb = 10  # MB growth
        self.potential_leaks: list[MemoryLeak] = []

    def start_tracemalloc(self) -> None:
        """Start tracemalloc for detailed memory tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Keep 10 frames
            self.tracemalloc_enabled = True
            logger.info("Started tracemalloc")

    def stop_tracemalloc(self) -> None:
        """Stop tracemalloc."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            logger.info("Stopped tracemalloc")

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            system_memory = psutil.virtual_memory()

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=memory_percent,
                available_mb=system_memory.available / (1024 * 1024)
            )

            # Add tracemalloc info if enabled
            if self.tracemalloc_enabled:
                current, _peak = tracemalloc.get_traced_memory()
                snapshot.tracemalloc_mb = current / (1024 * 1024)

                # Get top memory allocations
                top_stats = tracemalloc.take_snapshot().statistics('lineno')
                snapshot.top_traces = []

                for stat in top_stats[:10]:
                    snapshot.top_traces.append({
                        'filename': stat.traceback.format()[0],
                        'size_mb': stat.size / (1024 * 1024),
                        'count': stat.count
                    })

            return snapshot

        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0,
                vms_mb=0,
                percent=0,
                available_mb=0
            )

    def monitor_memory(self, duration: float | None = None) -> None:
        """Monitor memory usage in real-time."""
        self.monitoring = True
        start_time = time.time()

        logger.info(f"Starting memory monitoring (interval: {self.snapshot_interval}s)")

        try:
            while self.monitoring:
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)

                # Check memory threshold
                if snapshot.rss_mb > self.memory_threshold_mb:
                    logger.warning(f"Memory usage high: {snapshot.rss_mb:.1f} MB")

                # Limit snapshot history
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots = self.snapshots[-self.max_snapshots:]

                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break

                time.sleep(self.snapshot_interval)

        except KeyboardInterrupt:
            logger.info("Memory monitoring interrupted")
        finally:
            self.monitoring = False

    def start_background_monitoring(self, duration: float | None = None) -> None:
        """Start memory monitoring in background thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring already running")
            return

        self.monitor_thread = threading.Thread(
            target=self.monitor_memory,
            args=(duration,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started background memory monitoring")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped memory monitoring")

    def detect_memory_leaks(self) -> list[MemoryLeak]:
        """Detect potential memory leaks from snapshots."""
        if len(self.snapshots) < 10:
            logger.warning("Not enough snapshots for leak detection")
            return []

        leaks = []
        window_snapshots = []

        # Analyze snapshots in sliding window
        for i in range(len(self.snapshots)):
            current_snapshot = self.snapshots[i]

            # Add to window
            window_snapshots.append(current_snapshot)

            # Remove old snapshots from window
            window_snapshots = [
                s for s in window_snapshots
                if current_snapshot.timestamp - s.timestamp <= self.leak_detection_window
            ]

            if len(window_snapshots) < 5:
                continue

            # Check for memory growth
            first_memory = window_snapshots[0].rss_mb
            last_memory = window_snapshots[-1].rss_mb
            memory_growth = last_memory - first_memory

            if memory_growth > self.leak_threshold_mb:
                duration = window_snapshots[-1].timestamp - window_snapshots[0].timestamp
                growth_rate = memory_growth / duration

                # Create leak report
                leak = MemoryLeak(
                    location=f"Process {self.process.pid}",
                    size_mb=memory_growth,
                    growth_rate_mb_s=growth_rate,
                    duration_seconds=duration,
                    allocation_count=len(window_snapshots),
                    stack_trace=[]
                )

                leaks.append(leak)

        # Deduplicate similar leaks
        unique_leaks = []
        for leak in leaks:
            is_duplicate = False
            for existing in unique_leaks:
                if abs(leak.growth_rate_mb_s - existing.growth_rate_mb_s) < 0.1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_leaks.append(leak)

        self.potential_leaks = unique_leaks
        logger.info(f"Detected {len(unique_leaks)} potential memory leaks")
        return unique_leaks

    def analyze_memory_patterns(self) -> dict[str, Any]:
        """Analyze memory usage patterns from snapshots."""
        if not self.snapshots:
            return {"error": "No snapshots available"}

        # Basic statistics
        rss_values = [s.rss_mb for s in self.snapshots]
        vms_values = [s.vms_mb for s in self.snapshots]

        analysis = {
            "summary": {
                "snapshots_count": len(self.snapshots),
                "duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
                "avg_rss_mb": sum(rss_values) / len(rss_values),
                "max_rss_mb": max(rss_values),
                "min_rss_mb": min(rss_values),
                "rss_growth_mb": rss_values[-1] - rss_values[0],
            },
            "patterns": {},
            "recommendations": []
        }

        # Growth patterns
        if len(self.snapshots) > 1:
            # Calculate growth rate
            duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
            growth_rate = (rss_values[-1] - rss_values[0]) / duration
            analysis["patterns"]["growth_rate_mb_s"] = growth_rate

            # Volatility (standard deviation)
            avg_rss = analysis["summary"]["avg_rss_mb"]
            variance = sum((x - avg_rss) ** 2 for x in rss_values) / len(rss_values)
            std_dev = variance ** 0.5
            analysis["patterns"]["volatility_mb"] = std_dev

            # Peak detection
            peaks = []
            for i in range(1, len(rss_values) - 1):
                if rss_values[i] > rss_values[i-1] and rss_values[i] > rss_values[i+1]:
                    if rss_values[i] > avg_rss + std_dev:
                        peaks.append({
                            "timestamp": self.snapshots[i].timestamp_str,
                            "memory_mb": rss_values[i]
                        })
            analysis["patterns"]["memory_peaks"] = peaks

        # Generate recommendations
        recommendations = []

        if analysis["summary"]["max_rss_mb"] > 500:
            recommendations.append("High memory usage detected (>500MB)")

        if analysis["patterns"].get("growth_rate_mb_s", 0) > 1:
            recommendations.append("Significant memory growth detected - check for leaks")

        if analysis["patterns"].get("volatility_mb", 0) > 50:
            recommendations.append("High memory volatility - optimize allocation patterns")

        if len(analysis["patterns"].get("memory_peaks", [])) > 5:
            recommendations.append("Frequent memory spikes - consider memory pooling")

        analysis["recommendations"] = recommendations

        return analysis

    def profile_function(self, func: Callable, *args, **kwargs) -> dict[str, Any]:
        """Profile memory usage of a specific function."""
        logger.info(f"Profiling function: {func.__name__}")

        # Take before snapshot
        self.start_tracemalloc()
        before_snapshot = self.take_snapshot()

        # Run function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Function execution failed: {e}")

        execution_time = time.time() - start_time

        # Take after snapshot
        after_snapshot = self.take_snapshot()

        # Calculate memory delta
        memory_delta_mb = after_snapshot.rss_mb - before_snapshot.rss_mb
        tracemalloc_delta_mb = (
            (after_snapshot.tracemalloc_mb or 0) - (before_snapshot.tracemalloc_mb or 0)
            if self.tracemalloc_enabled else None
        )

        # Get detailed allocation info
        allocations = []
        if self.tracemalloc_enabled and after_snapshot.top_traces:
            allocations = after_snapshot.top_traces[:5]

        profile_result = {
            "function_name": func.__name__,
            "execution_time_s": execution_time,
            "memory_delta_mb": memory_delta_mb,
            "tracemalloc_delta_mb": tracemalloc_delta_mb,
            "before_memory_mb": before_snapshot.rss_mb,
            "after_memory_mb": after_snapshot.rss_mb,
            "success": success,
            "error": error,
            "top_allocations": allocations,
            "recommendations": []
        }

        # Generate recommendations
        if memory_delta_mb > 10:
            profile_result["recommendations"].append(
                "High memory allocation - consider memory optimization"
            )

        if execution_time > 1 and memory_delta_mb > 0:
            profile_result["recommendations"].append(
                "Long execution with memory allocation - check for efficiency"
            )

        return profile_result

    def profile_script(self, script_path: Path) -> dict[str, Any]:
        """Profile memory usage of a Python script."""
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        logger.info(f"Profiling script: {script_path}")

        # Start monitoring
        self.start_tracemalloc()
        self.start_background_monitoring()

        try:
            # Execute script in subprocess to isolate memory
            start_time = time.time()
            result = subprocess.run([
                'uv', 'run', 'python', str(script_path)
            ], check=False, capture_output=True, text=True, timeout=300)

            execution_time = time.time() - start_time

            # Stop monitoring
            time.sleep(1)  # Allow final snapshot
            self.stop_monitoring()

            # Analyze results
            analysis = self.analyze_memory_patterns()
            leaks = self.detect_memory_leaks()

            profile_result = {
                "script_path": str(script_path),
                "execution_time_s": execution_time,
                "return_code": result.returncode,
                "stdout_lines": len(result.stdout.split('\n')),
                "stderr_lines": len(result.stderr.split('\n')),
                "memory_analysis": analysis,
                "potential_leaks": len(leaks),
                "success": result.returncode == 0
            }

            return profile_result

        except subprocess.TimeoutExpired:
            self.stop_monitoring()
            return {
                "script_path": str(script_path),
                "error": "Script execution timed out",
                "success": False
            }

        except Exception as e:
            self.stop_monitoring()
            return {
                "script_path": str(script_path),
                "error": str(e),
                "success": False
            }

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive memory profiling report."""
        analysis = self.analyze_memory_patterns()
        leaks = self.detect_memory_leaks()

        # System information
        system_info = {
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "platform": sys.platform,
        }

        # Process information
        process_info = {
            "pid": self.process.pid,
            "name": self.process.name(),
            "cpu_percent": self.process.cpu_percent(),
            "num_threads": self.process.num_threads(),
            "create_time": datetime.fromtimestamp(
                self.process.create_time()
            ).isoformat()
        }

        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "process_info": process_info,
            "memory_analysis": analysis,
            "potential_leaks": [asdict(leak) for leak in leaks],
            "snapshots_summary": {
                "count": len(self.snapshots),
                "first": asdict(self.snapshots[0]) if self.snapshots else None,
                "last": asdict(self.snapshots[-1]) if self.snapshots else None,
            },
            "tracemalloc_enabled": self.tracemalloc_enabled
        }

        return report

    def save_report(self, output_file: Path) -> None:
        """Save profiling report to file."""
        report = self.generate_report()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_summary(self) -> None:
        """Print memory profiling summary."""
        analysis = self.analyze_memory_patterns()

        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return

        print("\n" + "="*80)
        print("MEMORY PROFILING REPORT")
        print("="*80)

        # System info
        memory_info = psutil.virtual_memory()
        print("\n🖥️  SYSTEM INFORMATION")
        print("-" * 40)
        print(f"Total RAM: {memory_info.total / (1024**3):.1f} GB")
        print(f"Available: {memory_info.available / (1024**3):.1f} GB")
        print(f"Used: {memory_info.percent:.1f}%")

        # Memory summary
        summary = analysis["summary"]
        print("\n📊 MEMORY USAGE SUMMARY")
        print("-" * 40)
        print(f"Snapshots: {summary['snapshots_count']}")
        print(f"Duration: {summary['duration_seconds']:.1f}s")
        print(f"Average RSS: {summary['avg_rss_mb']:.1f} MB")
        print(f"Peak RSS: {summary['max_rss_mb']:.1f} MB")
        print(f"Memory Growth: {summary['rss_growth_mb']:.1f} MB")

        # Patterns
        if "patterns" in analysis:
            patterns = analysis["patterns"]
            print("\n📈 PATTERNS")
            print("-" * 40)

            if "growth_rate_mb_s" in patterns:
                rate = patterns["growth_rate_mb_s"]
                print(f"Growth Rate: {rate:.3f} MB/s")

            if "volatility_mb" in patterns:
                print(f"Volatility: {patterns['volatility_mb']:.1f} MB")

            if "memory_peaks" in patterns:
                peaks = patterns["memory_peaks"]
                print(f"Memory Peaks: {len(peaks)}")
                for peak in peaks[:3]:
                    print(f"  {peak['timestamp']}: {peak['memory_mb']:.1f} MB")

        # Leaks
        if self.potential_leaks:
            print(f"\n⚠️  POTENTIAL MEMORY LEAKS ({len(self.potential_leaks)})")
            print("-" * 40)
            for leak in self.potential_leaks[:5]:
                print(f"  Growth: {leak.size_mb:.1f} MB "
                      f"({leak.growth_rate_mb_s:.3f} MB/s)")
                print(f"  Duration: {leak.duration_seconds:.1f}s")

        # Recommendations
        if analysis.get("recommendations"):
            print("\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"{i}. {rec}")

        print("\n" + "="*80)

    def cleanup(self) -> None:
        """Clean up profiler resources."""
        self.stop_monitoring()
        self.stop_tracemalloc()
        gc.collect()

def signal_handler(signum, frame, profiler):
    """Handle interrupt signals gracefully."""
    logger.info("Received interrupt signal, cleaning up...")
    profiler.cleanup()
    sys.exit(0)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile memory usage of Python applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile_memory.py                       # Monitor current process
  python profile_memory.py --duration 60        # Monitor for 60 seconds
  python profile_memory.py --script main.py     # Profile script execution
  python profile_memory.py --output report.json # Save detailed report
  python profile_memory.py --tracemalloc        # Enable detailed tracing
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Monitoring duration in seconds",
    )

    parser.add_argument(
        "--script",
        type=Path,
        help="Profile specific Python script",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Snapshot interval in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        help="Enable tracemalloc for detailed memory tracking",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=100,
        help="Memory alert threshold in MB (default: 100)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save detailed report to JSON file",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:

    # Validate arguments
    if not args.project_root.exists():
        logger.error(f"Project root does not exist: {args.project_root}")
        sys.exit(1)

    # Initialize profiler
    profiler = MemoryProfiler(args.project_root)
    profiler.snapshot_interval = args.interval
    profiler.memory_threshold_mb = args.threshold

    # Set up signal handler
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, profiler))

    try:
        # Enable tracemalloc if requested
        if args.tracemalloc:
            profiler.start_tracemalloc()

        # Profile specific script
        if args.script:
            if not args.script.exists():
                logger.error(f"Script not found: {args.script}")
                sys.exit(1)

            result = profiler.profile_script(args.script)
            print("\n📝 Script Profiling Results:")
            print(f"Script: {result['script_path']}")
            print(f"Execution Time: {result['execution_time_s']:.2f}s")
            print(f"Success: {result['success']}")

            if 'memory_analysis' in result:
                profiler.print_summary()

        else:
            # Monitor current process
            profiler.start_background_monitoring(args.duration)

            if args.duration:
                # Wait for monitoring to complete
                profiler.monitor_thread.join()
            else:
                # Interactive monitoring
                print("Memory monitoring started. Press Ctrl+C to stop and view results.")
                try:
                    while profiler.monitoring:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass

            profiler.stop_monitoring()
            profiler.detect_memory_leaks()
            profiler.print_summary()

        # Save report if requested
        if args.output:
            profiler.save_report(args.output)

    finally:
        profiler.cleanup()

if __name__ == "__main__":
    main()
