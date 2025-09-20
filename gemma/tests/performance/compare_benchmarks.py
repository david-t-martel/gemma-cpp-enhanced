#!/usr/bin/env python3
"""
Benchmark Comparison Tool
Compare two benchmark result files and detect performance regressions
"""

import json
import sys
import argparse
from typing import Dict, List, Any
from pathlib import Path


class BenchmarkComparator:
    def __init__(self, baseline_file: str, current_file: str):
        self.baseline_file = Path(baseline_file)
        self.current_file = Path(current_file)
        self.baseline_data = self._load_json(self.baseline_file)
        self.current_data = self._load_json(self.current_file)

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            sys.exit(1)

    def _extract_benchmarks(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract benchmark results into a normalized format"""
        benchmarks = {}

        if 'benchmarks' not in data:
            print("Invalid benchmark data format")
            return benchmarks

        for bench in data['benchmarks']:
            name = bench['name']
            # Extract key metrics
            benchmarks[name] = {
                'real_time': bench.get('real_time', 0),
                'cpu_time': bench.get('cpu_time', 0),
                'iterations': bench.get('iterations', 0),
                'bytes_per_second': bench.get('bytes_per_second', 0),
                'items_per_second': bench.get('items_per_second', 0)
            }

        return benchmarks

    def compare(self, regression_threshold: float = 0.1) -> bool:
        """
        Compare benchmarks and detect regressions

        Args:
            regression_threshold: Threshold for detecting regression (e.g., 0.1 = 10%)

        Returns:
            True if no significant regressions detected
        """
        baseline_benchmarks = self._extract_benchmarks(self.baseline_data)
        current_benchmarks = self._extract_benchmarks(self.current_data)

        if not baseline_benchmarks:
            print("Warning: No baseline benchmarks found")
            return True

        if not current_benchmarks:
            print("Error: No current benchmarks found")
            return False

        print("=" * 80)
        print("BENCHMARK COMPARISON REPORT")
        print("=" * 80)
        print(f"Baseline: {self.baseline_file}")
        print(f"Current:  {self.current_file}")
        print(f"Regression threshold: {regression_threshold * 100:.1f}%")
        print()

        regressions_found = False
        improvements_found = False

        # Compare common benchmarks
        common_benchmarks = set(baseline_benchmarks.keys()) & set(current_benchmarks.keys())

        if not common_benchmarks:
            print("Warning: No common benchmarks found between baseline and current")
            return True

        print(f"{'Benchmark':<40} {'Baseline':<15} {'Current':<15} {'Change':<10} {'Status'}")
        print("-" * 90)

        for bench_name in sorted(common_benchmarks):
            baseline = baseline_benchmarks[bench_name]
            current = current_benchmarks[bench_name]

            # Compare real_time as primary metric
            baseline_time = baseline['real_time']
            current_time = current['real_time']

            if baseline_time == 0:
                continue

            change_ratio = (current_time - baseline_time) / baseline_time
            change_percent = change_ratio * 100

            # Determine status
            if change_ratio > regression_threshold:
                status = "REGRESSION ❌"
                regressions_found = True
            elif change_ratio < -regression_threshold:
                status = "IMPROVEMENT ✅"
                improvements_found = True
            else:
                status = "STABLE ➖"

            print(f"{bench_name:<40} {baseline_time:<15.2f} {current_time:<15.2f} {change_percent:>+7.1f}% {status}")

        print()

        # Summary
        if regressions_found:
            print("⚠️  PERFORMANCE REGRESSIONS DETECTED!")
            print("   Some benchmarks are significantly slower than baseline.")
        elif improvements_found:
            print("🎉 PERFORMANCE IMPROVEMENTS DETECTED!")
            print("   Some benchmarks are significantly faster than baseline.")
        else:
            print("✅ PERFORMANCE IS STABLE")
            print("   No significant changes from baseline.")

        print()

        # New benchmarks
        new_benchmarks = set(current_benchmarks.keys()) - set(baseline_benchmarks.keys())
        if new_benchmarks:
            print("New benchmarks (no baseline for comparison):")
            for bench_name in sorted(new_benchmarks):
                bench_data = current_benchmarks[bench_name]
                print(f"  - {bench_name}: {bench_data['real_time']:.2f} ns")
            print()

        # Missing benchmarks
        missing_benchmarks = set(baseline_benchmarks.keys()) - set(current_benchmarks.keys())
        if missing_benchmarks:
            print("Missing benchmarks (present in baseline but not current):")
            for bench_name in sorted(missing_benchmarks):
                print(f"  - {bench_name}")
            print()

        return not regressions_found

    def generate_report(self, output_file: str = None):
        """Generate a detailed comparison report"""
        baseline_benchmarks = self._extract_benchmarks(self.baseline_data)
        current_benchmarks = self._extract_benchmarks(self.current_data)

        report = {
            "comparison_metadata": {
                "baseline_file": str(self.baseline_file),
                "current_file": str(self.current_file),
                "baseline_benchmarks_count": len(baseline_benchmarks),
                "current_benchmarks_count": len(current_benchmarks)
            },
            "benchmark_comparisons": []
        }

        common_benchmarks = set(baseline_benchmarks.keys()) & set(current_benchmarks.keys())

        for bench_name in sorted(common_benchmarks):
            baseline = baseline_benchmarks[bench_name]
            current = current_benchmarks[bench_name]

            comparison = {
                "name": bench_name,
                "baseline": baseline,
                "current": current,
                "change_ratio": {
                    "real_time": (current['real_time'] - baseline['real_time']) / baseline['real_time'] if baseline['real_time'] > 0 else 0,
                    "cpu_time": (current['cpu_time'] - baseline['cpu_time']) / baseline['cpu_time'] if baseline['cpu_time'] > 0 else 0
                }
            }

            report["benchmark_comparisons"].append(comparison)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Detailed report saved to: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('baseline', help='Baseline benchmark results file (JSON)')
    parser.add_argument('current', help='Current benchmark results file (JSON)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Regression threshold (default: 0.1 = 10%%)')
    parser.add_argument('--report', help='Generate detailed report to file')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with non-zero code if regressions detected')

    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        sys.exit(1)

    if not Path(args.current).exists():
        print(f"Error: Current file not found: {args.current}")
        sys.exit(1)

    comparator = BenchmarkComparator(args.baseline, args.current)

    # Generate detailed report if requested
    if args.report:
        comparator.generate_report(args.report)

    # Perform comparison
    no_regressions = comparator.compare(args.threshold)

    if args.fail_on_regression and not no_regressions:
        print("\nExiting with error code due to performance regressions.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()