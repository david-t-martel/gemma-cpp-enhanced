"""
Test Report Generator

Generates comprehensive test reports with metrics, visualizations, and analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    test_category: str
    status: TestStatus
    duration_seconds: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class TestSuiteReport:
    """Complete test suite report."""
    suite_name: str
    start_time: float
    end_time: float
    test_results: List[TestResult]
    performance_metrics: Dict[str, Any]
    system_info: Dict[str, Any]
    configuration: Dict[str, Any]

    @property
    def duration(self) -> float:
        """Total test suite duration."""
        return self.end_time - self.start_time

    @property
    def total_tests(self) -> int:
        """Total number of tests."""
        return len(self.test_results)

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        return sum(1 for t in self.test_results if t.status == TestStatus.PASSED)

    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        return sum(1 for t in self.test_results if t.status == TestStatus.FAILED)

    @property
    def pass_rate(self) -> float:
        """Pass rate percentage."""
        if self.total_tests == 0:
            return 0
        return (self.passed_tests / self.total_tests) * 100

    def get_category_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary by test category."""
        categories = {}
        for test in self.test_results:
            if test.test_category not in categories:
                categories[test.test_category] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'skipped': 0,
                    'error': 0,
                    'timeout': 0
                }

            categories[test.test_category]['total'] += 1
            categories[test.test_category][test.status.value] += 1

        return categories


class ReportGenerator:
    """Generate test reports in various formats."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(self, report: TestSuiteReport) -> Path:
        """Generate JSON format report."""
        output_file = self.output_dir / f"test_report_{int(report.start_time)}.json"

        report_dict = {
            'suite_name': report.suite_name,
            'start_time': report.start_time,
            'end_time': report.end_time,
            'duration': report.duration,
            'summary': {
                'total_tests': report.total_tests,
                'passed': report.passed_tests,
                'failed': report.failed_tests,
                'pass_rate': report.pass_rate
            },
            'test_results': [
                {
                    'test_name': t.test_name,
                    'category': t.test_category,
                    'status': t.status.value,
                    'duration': t.duration_seconds,
                    'error_message': t.error_message,
                    'metrics': t.metrics,
                    'timestamp': t.timestamp
                }
                for t in report.test_results
            ],
            'performance_metrics': report.performance_metrics,
            'system_info': report.system_info,
            'configuration': report.configuration,
            'category_summary': report.get_category_summary()
        }

        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        return output_file

    def generate_html_report(self, report: TestSuiteReport) -> Path:
        """Generate HTML format report."""
        output_file = self.output_dir / f"test_report_{int(report.start_time)}.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {report.suite_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .pass {{ color: #10b981; }}
        .fail {{ color: #ef4444; }}
        .skip {{ color: #f59e0b; }}
        table {{
            width: 100%;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #666;
        }}
        td {{
            padding: 12px;
            border-top: 1px solid #e5e7eb;
        }}
        .status-passed {{ color: #10b981; font-weight: 600; }}
        .status-failed {{ color: #ef4444; font-weight: 600; }}
        .status-skipped {{ color: #f59e0b; font-weight: 600; }}
        .status-error {{ color: #dc2626; font-weight: 600; }}
        .status-timeout {{ color: #7c3aed; font-weight: 600; }}
        .category {{
            background: #f3f4f6;
            padding: 10px;
            margin: 20px 0 10px 0;
            border-radius: 5px;
            font-weight: 600;
        }}
        .metrics {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-container {{
            width: 100%;
            max-width: 600px;
            margin: 20px auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.suite_name}</h1>
        <p>Generated: {datetime.fromtimestamp(report.end_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Duration: {report.duration:.2f} seconds</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Tests</h3>
            <div class="value">{report.total_tests}</div>
        </div>
        <div class="summary-card">
            <h3>Passed</h3>
            <div class="value pass">{report.passed_tests}</div>
        </div>
        <div class="summary-card">
            <h3>Failed</h3>
            <div class="value fail">{report.failed_tests}</div>
        </div>
        <div class="summary-card">
            <h3>Pass Rate</h3>
            <div class="value">{report.pass_rate:.1f}%</div>
        </div>
    </div>

    <h2>Test Results</h2>
"""

        # Group tests by category
        categories = {}
        for test in report.test_results:
            if test.test_category not in categories:
                categories[test.test_category] = []
            categories[test.test_category].append(test)

        # Generate table for each category
        for category, tests in categories.items():
            html_content += f'<div class="category">{category}</div>'
            html_content += """
    <table>
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Error Message</th>
            </tr>
        </thead>
        <tbody>
"""
            for test in tests:
                status_class = f"status-{test.status.value}"
                error_msg = test.error_message or "-"
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."

                html_content += f"""
            <tr>
                <td>{test.test_name}</td>
                <td class="{status_class}">{test.status.value.upper()}</td>
                <td>{test.duration_seconds:.3f}</td>
                <td>{error_msg}</td>
            </tr>
"""
            html_content += """
        </tbody>
    </table>
"""

        # Add performance metrics
        if report.performance_metrics:
            html_content += """
    <div class="metrics">
        <h2>Performance Metrics</h2>
        <pre>
"""
            for key, value in report.performance_metrics.items():
                html_content += f"{key}: {value}\n"
            html_content += """
        </pre>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html_content)

        return output_file

    def generate_markdown_report(self, report: TestSuiteReport) -> Path:
        """Generate Markdown format report."""
        output_file = self.output_dir / f"test_report_{int(report.start_time)}.md"

        md_lines = [
            f"# Test Report - {report.suite_name}",
            "",
            f"**Generated:** {datetime.fromtimestamp(report.end_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {report.duration:.2f} seconds",
            "",
            "## Summary",
            "",
            f"- **Total Tests:** {report.total_tests}",
            f"- **Passed:** {report.passed_tests} ✅",
            f"- **Failed:** {report.failed_tests} ❌",
            f"- **Pass Rate:** {report.pass_rate:.1f}%",
            "",
            "## Results by Category",
            ""
        ]

        # Category summary
        category_summary = report.get_category_summary()
        for category, stats in category_summary.items():
            pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            md_lines.extend([
                f"### {category}",
                "",
                f"| Metric | Count |",
                f"|--------|-------|",
                f"| Total | {stats['total']} |",
                f"| Passed | {stats['passed']} |",
                f"| Failed | {stats['failed']} |",
                f"| Pass Rate | {pass_rate:.1f}% |",
                ""
            ])

        # Detailed test results
        md_lines.extend([
            "## Detailed Test Results",
            "",
            "| Test Name | Category | Status | Duration (s) | Error |",
            "|-----------|----------|--------|--------------|-------|"
        ])

        for test in report.test_results:
            status_icon = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌",
                TestStatus.SKIPPED: "⏭️",
                TestStatus.ERROR: "🔥",
                TestStatus.TIMEOUT: "⏱️"
            }.get(test.status, "❓")

            error_msg = test.error_message[:50] + "..." if test.error_message and len(test.error_message) > 50 else (test.error_message or "-")
            md_lines.append(
                f"| {test.test_name} | {test.test_category} | {status_icon} {test.status.value} | {test.duration_seconds:.3f} | {error_msg} |"
            )

        # Performance metrics
        if report.performance_metrics:
            md_lines.extend([
                "",
                "## Performance Metrics",
                "",
                "```json"
            ])
            md_lines.append(json.dumps(report.performance_metrics, indent=2))
            md_lines.append("```")

        # System info
        if report.system_info:
            md_lines.extend([
                "",
                "## System Information",
                "",
                "```json"
            ])
            md_lines.append(json.dumps(report.system_info, indent=2))
            md_lines.append("```")

        with open(output_file, 'w') as f:
            f.write("\n".join(md_lines))

        return output_file

    def generate_charts(self, report: TestSuiteReport) -> Optional[Path]:
        """Generate performance charts if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Test Report - {report.suite_name}', fontsize=16)

        # 1. Pass/Fail pie chart
        ax1 = axes[0, 0]
        labels = ['Passed', 'Failed', 'Other']
        sizes = [
            report.passed_tests,
            report.failed_tests,
            report.total_tests - report.passed_tests - report.failed_tests
        ]
        colors = ['#10b981', '#ef4444', '#f59e0b']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Test Results Distribution')

        # 2. Category performance bar chart
        ax2 = axes[0, 1]
        category_summary = report.get_category_summary()
        categories = list(category_summary.keys())
        pass_rates = [
            (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            for stats in category_summary.values()
        ]
        ax2.bar(categories, pass_rates, color='#667eea')
        ax2.set_title('Pass Rate by Category')
        ax2.set_ylabel('Pass Rate (%)')
        ax2.set_ylim(0, 100)

        # 3. Test duration histogram
        ax3 = axes[1, 0]
        durations = [t.duration_seconds for t in report.test_results]
        ax3.hist(durations, bins=20, color='#764ba2', edgecolor='white')
        ax3.set_title('Test Duration Distribution')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Number of Tests')

        # 4. Timeline of test execution
        ax4 = axes[1, 1]
        test_times = [(t.timestamp - report.start_time, t.duration_seconds) for t in report.test_results]
        test_times.sort(key=lambda x: x[0])

        cumulative_time = []
        current_time = 0
        for start, duration in test_times:
            current_time += duration
            cumulative_time.append(current_time)

        ax4.plot(range(len(cumulative_time)), cumulative_time, color='#ef4444', linewidth=2)
        ax4.set_title('Cumulative Test Execution Time')
        ax4.set_xlabel('Test Number')
        ax4.set_ylabel('Cumulative Time (seconds)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        chart_file = self.output_dir / f"test_charts_{int(report.start_time)}.png"
        plt.savefig(chart_file, dpi=100, bbox_inches='tight')
        plt.close()

        return chart_file

    def generate_all_formats(self, report: TestSuiteReport) -> Dict[str, Path]:
        """Generate reports in all available formats."""
        outputs = {
            'json': self.generate_json_report(report),
            'html': self.generate_html_report(report),
            'markdown': self.generate_markdown_report(report)
        }

        chart_path = self.generate_charts(report)
        if chart_path:
            outputs['charts'] = chart_path

        return outputs


def collect_system_info() -> Dict[str, Any]:
    """Collect system information for the report."""
    import platform
    import psutil

    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
        'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024
    }