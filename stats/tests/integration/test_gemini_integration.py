#!/usr/bin/env python
"""Test script for Gemini integration components.

This script demonstrates the usage of Gemini-powered code review,
analysis, and optimization tools.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure.gemini import (
    GeminiClient,
    GeminiCodeAnalyzer,
    GeminiCodeReviewer,
    GeminiOptimizer,
)
from src.infrastructure.gemini.client import GeminiConfig, GeminiModel
from src.infrastructure.gemini.optimizer import OptimizationProfile, OptimizationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Sample code for testing
SAMPLE_PYTHON_CODE = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def find_duplicates(lst):
    """Find duplicate elements in a list."""
    duplicates = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates

class DataProcessor:
    def __init__(self):
        self.data = []

    def process_data(self, raw_data):
        # Process data without validation
        for item in raw_data:
            result = item * 2
            self.data.append(result)
        return self.data

    def get_statistics(self):
        total = 0
        count = 0
        for item in self.data:
            total += item
            count += 1
        return total / count if count > 0 else 0
'''

SAMPLE_SQL_QUERY = """
SELECT
    u.id,
    u.name,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2023-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY total_spent DESC;
"""


async def test_basic_client():
    """Test basic Gemini client functionality."""
    print("\n" + "=" * 80)
    print("Testing Basic Gemini Client")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "⚠️  Warning: No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
        print("   Get your key from: https://makersuite.google.com/app/apikey")
        return False

    try:
        # Create client with configuration
        config = GeminiConfig(
            api_key=api_key,
            model=GeminiModel.GEMINI_2_5_FLASH,
            temperature=0.7,
        )

        async with GeminiClient(config) as client:
            # Test simple generation
            response = await client.generate(
                prompt="Explain what a fibonacci sequence is in 2 sentences.",
                temperature=0.5,
            )

            print(f"✅ Response received: {response.content[:200]}...")
            print(f"   Model: {response.model}")
            print(f"   Tokens used: {response.total_tokens}")

            # Test token counting
            token_count = await client.count_tokens("This is a test sentence.")
            print(f"✅ Token counting works: {token_count} tokens")

            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_code_reviewer():
    """Test code review functionality."""
    print("\n" + "=" * 80)
    print("Testing Code Reviewer")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Skipping: No API key configured")
        return False

    try:
        # Create reviewer
        reviewer = GeminiCodeReviewer()

        # Review sample code
        print("📝 Reviewing sample Python code...")
        result = await reviewer.review_code(
            code=SAMPLE_PYTHON_CODE,
            file_path="sample.py",
            language="python",
        )

        print("\n📊 Review Summary:")
        print(f"   {result.summary}")
        print(f"\n🔍 Issues Found: {len(result.issues)}")

        for issue in result.issues[:3]:  # Show first 3 issues
            print(f"\n   [{issue.severity.value.upper()}] {issue.category.value}")
            print(f"   📍 Line {issue.line_number}: {issue.message}")
            if issue.suggestion:
                print(f"   💡 Suggestion: {issue.suggestion}")

        print(f"\n📈 Overall Score: {result.overall_score:.1f}/100")

        if result.recommendations:
            print("\n💡 Recommendations:")
            for rec in result.recommendations[:3]:
                print(f"   • {rec}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_code_analyzer():
    """Test code analysis functionality."""
    print("\n" + "=" * 80)
    print("Testing Code Analyzer")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Skipping: No API key configured")
        return False

    try:
        # Create analyzer
        analyzer = GeminiCodeAnalyzer()

        # Analyze sample code
        print("🔬 Analyzing sample Python code...")
        module = await analyzer.analyze_code(
            code=SAMPLE_PYTHON_CODE,
            file_path="sample.py",
            language="python",
            deep_analysis=True,
        )

        print("\n📊 Analysis Results:")
        print(f"   File: {module.file_path}")
        print(f"   Lines of code: {module.lines_of_code}")
        print(f"   Classes: {len(module.classes)}")
        print(f"   Functions: {len(module.functions)}")
        print(f"   Comment ratio: {module.comment_ratio:.1%}")

        # Show function complexity
        print("\n🔢 Function Complexity:")
        for func in module.functions:
            print(
                f"   • {func.name}: {func.complexity_level.value} (complexity: {func.complexity})"
            )

        # Test complexity analysis
        print("\n🎯 Detailed Complexity Analysis:")
        complexity = await analyzer.analyze_complexity(SAMPLE_PYTHON_CODE, "python")
        if complexity and "functions" in complexity:
            for func_data in complexity.get("functions", [])[:3]:
                print(f"   • {func_data.get('name', 'Unknown')}: ")
                print(f"     Cyclomatic: {func_data.get('cyclomatic_complexity', 'N/A')}")
                print(f"     Level: {func_data.get('complexity_level', 'N/A')}")

        # Test pattern detection
        print("\n🎨 Design Pattern Detection:")
        patterns = await analyzer.detect_patterns(SAMPLE_PYTHON_CODE, "python")
        if patterns:
            for pattern in patterns:
                print(f"   • {pattern.value}")
        else:
            print("   No common patterns detected")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_optimizer():
    """Test code optimization functionality."""
    print("\n" + "=" * 80)
    print("Testing Code Optimizer")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Skipping: No API key configured")
        return False

    try:
        # Create optimizer
        optimizer = GeminiOptimizer()

        # Create optimization profile
        profile = OptimizationProfile(
            target_metrics=[OptimizationType.PERFORMANCE, OptimizationType.MEMORY],
            resource_constraints={},
            acceptable_trade_offs=["slight readability decrease"],
            preserve_readability=True,
        )

        # Optimize sample code
        print("⚡ Optimizing sample Python code...")
        result = await optimizer.optimize_code(
            code=SAMPLE_PYTHON_CODE,
            language="python",
            profile=profile,
        )

        print("\n📊 Optimization Results:")
        print(f"   Suggestions: {len(result.suggestions)}")
        print(f"   Potential speedup: {result.total_potential_speedup:.1f}x")
        print(f"   Memory savings: {result.total_memory_savings} bytes")
        print(f"   Confidence: {result.confidence_score:.1%}")
        print(f"   Estimated effort: {result.estimated_effort_hours:.1f} hours")

        # Show optimization suggestions
        print("\n🚀 Optimization Suggestions:")
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            print(f"\n   {i}. [{suggestion.priority.value.upper()}] {suggestion.title}")
            print(f"      Type: {suggestion.type.value}")
            print(f"      {suggestion.description}")
            if suggestion.optimized_code:
                print("      ✅ Optimized code provided")

        # Show implementation plan
        if result.implementation_plan:
            print("\n📋 Implementation Plan:")
            for step in result.implementation_plan[:5]:
                print(f"   {step}")

        # Test algorithm optimization
        print("\n🧮 Algorithm Optimization:")
        algo_result = await optimizer.optimize_algorithm(
            code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
            algorithm_description="Calculate fibonacci number",
        )

        if algo_result.suggestions:
            print(f"   Found {len(algo_result.suggestions)} algorithmic improvements")
            print(f"   Expected speedup: {algo_result.total_potential_speedup:.1f}x")

        # Test SQL optimization
        print("\n🗄️ SQL Query Optimization:")
        sql_suggestions = await optimizer.optimize_database_queries(
            queries=[SAMPLE_SQL_QUERY],
            database_type="postgresql",
        )

        if sql_suggestions:
            print(f"   Found {len(sql_suggestions)} query optimizations")
            for suggestion in sql_suggestions[:2]:
                print(f"   • {suggestion.title}: {suggestion.description}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_integration():
    """Test complete integration workflow."""
    print("\n" + "=" * 80)
    print("Testing Complete Integration Workflow")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Skipping: No API key configured")
        return False

    try:
        # 1. Analyze code
        analyzer = GeminiCodeAnalyzer()
        print("1️⃣ Analyzing code structure...")
        analysis = await analyzer.analyze_code(SAMPLE_PYTHON_CODE, language="python")
        print(f"   ✅ Found {len(analysis.functions)} functions, {len(analysis.classes)} classes")

        # 2. Review code
        reviewer = GeminiCodeReviewer()
        print("\n2️⃣ Reviewing code quality...")
        review = await reviewer.review_code(SAMPLE_PYTHON_CODE, language="python")
        print(f"   ✅ Identified {len(review.issues)} issues")

        # 3. Optimize based on findings
        optimizer = GeminiOptimizer()
        print("\n3️⃣ Generating optimizations...")

        # Focus on issues found in review
        focus_areas = []
        if any(i.category.value == "performance" for i in review.issues):
            focus_areas.append(OptimizationType.PERFORMANCE)
        if any(i.category.value == "memory" for i in review.issues):
            focus_areas.append(OptimizationType.MEMORY)

        profile = OptimizationProfile(
            target_metrics=focus_areas or [OptimizationType.PERFORMANCE],
            resource_constraints={},
            acceptable_trade_offs=[],
        )

        optimization = await optimizer.optimize_code(
            code=SAMPLE_PYTHON_CODE,
            language="python",
            profile=profile,
            context=f"Address these issues: {review.summary}",
        )

        print(f"   ✅ Generated {len(optimization.suggestions)} optimizations")

        # 4. Summary
        print("\n📊 Integration Summary:")
        print(f"   Code Quality Score: {review.overall_score:.1f}/100")
        print(f"   Optimization Potential: {optimization.total_potential_speedup:.1f}x speedup")
        print(f"   Total Issues: {len(review.issues)}")
        print(f"   Total Suggestions: {len(optimization.suggestions)}")
        print(f"   Estimated Improvement Effort: {optimization.estimated_effort_hours:.1f} hours")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("GEMINI INTEGRATION TEST SUITE")
    print("🚀 " * 20)

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: No Gemini API key configured!")
        print("To use Gemini features, please:")
        print("1. Get an API key from: https://makersuite.google.com/app/apikey")
        print("2. Set the environment variable: GEMINI_API_KEY=your-key-here")
        print("\nRunning tests without API key (limited functionality)...\n")

    results = {}

    # Run tests
    tests = [
        ("Basic Client", test_basic_client),
        ("Code Reviewer", test_code_reviewer),
        ("Code Analyzer", test_code_analyzer),
        ("Code Optimizer", test_optimizer),
        ("Integration", test_integration),
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with error: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
    elif api_key:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    else:
        print("\n💡 To enable all features, please configure your Gemini API key.")

    return passed == total


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
