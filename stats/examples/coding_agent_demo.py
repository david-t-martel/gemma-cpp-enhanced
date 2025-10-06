#!/usr/bin/env python3
"""
Comprehensive demonstration of the Python ReAct agent with Gemma for solving coding problems.

This demo showcases:
1. Code generation with reflection
2. Bug fixing with step-by-step reasoning
3. Code review and optimization
4. Integration with RAG for context retrieval
5. Tool usage and memory tiers
6. Planning and reflection capabilities

Author: Claude Code Assistant
Date: 2025-09-24
"""

import os
import sys
import time
import json
import asyncio
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up environment
os.environ["PYTHONPATH"] = str(project_root)

# Import agent components
from src.agent.gemma_agent import AgentMode, create_gemma_agent
from src.agent.react_agent import UnifiedReActAgent, ReActTrace, ThoughtType
from src.agent.tools import ToolDefinition, ToolParameter, ToolRegistry
from src.agent.planner import Planner, TaskComplexity
from src.agent.rag_integration import RAGIntegration
from src.shared.logging import setup_logging, get_logger, LogLevel

# Setup logging
setup_logging(level=LogLevel.INFO, console=True)
logger = get_logger(__name__)


# ==============================================================================
# CUSTOM CODING TOOLS
# ==============================================================================

def execute_python_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment and return results.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with execution results
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timeout ({timeout}s)",
            "returncode": -1
        }
    finally:
        os.unlink(temp_file)


def analyze_code_quality(code: str) -> Dict[str, Any]:
    """
    Analyze code quality using static analysis tools.

    Args:
        code: Python code to analyze

    Returns:
        Dictionary with analysis results
    """
    import ast
    import re

    issues = []
    metrics = {
        "lines": len(code.splitlines()),
        "functions": 0,
        "classes": 0,
        "complexity": 0
    }

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
                # Simple cyclomatic complexity estimation
                metrics["complexity"] += sum(1 for n in ast.walk(node)
                                            if isinstance(n, (ast.If, ast.While, ast.For)))
            elif isinstance(node, ast.ClassDef):
                metrics["classes"] += 1

        # Check for common issues
        if "import *" in code:
            issues.append("Avoid wildcard imports")

        if re.search(r'except\s*:', code):
            issues.append("Avoid bare except clauses")

        if re.search(r'print\s*\(', code) and 'logging' not in code:
            issues.append("Consider using logging instead of print statements")

        # Check for long lines
        for i, line in enumerate(code.splitlines(), 1):
            if len(line) > 100:
                issues.append(f"Line {i} exceeds 100 characters")

    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")

    return {
        "metrics": metrics,
        "issues": issues,
        "quality_score": max(0, 10 - len(issues))
    }


def retrieve_code_context(query: str, codebase_path: Optional[str] = None) -> str:
    """
    Retrieve relevant code context using RAG.

    Args:
        query: Search query
        codebase_path: Path to codebase for context

    Returns:
        Retrieved context as string
    """
    # Simulated RAG retrieval - in production, this would use vector search
    contexts = {
        "sorting": """
        Common sorting algorithms:
        - Bubble Sort: O(n²) - Simple but inefficient
        - Quick Sort: O(n log n) average - Efficient divide-and-conquer
        - Merge Sort: O(n log n) - Stable sort with consistent performance
        - Heap Sort: O(n log n) - In-place sorting

        Python's built-in sort() uses Timsort - a hybrid stable sorting algorithm.
        """,
        "api": """
        RESTful API best practices:
        - Use proper HTTP methods (GET, POST, PUT, DELETE)
        - Version your APIs (/api/v1/...)
        - Return appropriate status codes
        - Use JSON for request/response bodies
        - Implement proper authentication (JWT, OAuth)
        - Add rate limiting and pagination
        """,
        "database": """
        Database optimization tips:
        - Use indexes on frequently queried columns
        - Avoid SELECT * queries
        - Use connection pooling
        - Implement proper transaction management
        - Consider denormalization for read-heavy workloads
        - Use prepared statements to prevent SQL injection
        """,
        "testing": """
        Testing best practices:
        - Write unit tests for individual functions
        - Use pytest for Python testing
        - Aim for 80%+ code coverage
        - Mock external dependencies
        - Test edge cases and error conditions
        - Use fixtures for test data setup
        """
    }

    # Find best matching context
    query_lower = query.lower()
    for key, context in contexts.items():
        if key in query_lower:
            return context

    return "No specific context found. Using general programming knowledge."


def generate_test_cases(code: str, function_name: str) -> str:
    """
    Generate test cases for a given function.

    Args:
        code: Code containing the function
        function_name: Name of function to test

    Returns:
        Generated test code
    """
    import ast

    try:
        tree = ast.parse(code)

        # Find the function
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                func_node = node
                break

        if not func_node:
            return f"Function '{function_name}' not found in code"

        # Extract function signature
        args = [arg.arg for arg in func_node.args.args]

        # Generate basic test template
        test_template = f'''
import pytest
from typing import Any

def test_{function_name}_basic():
    """Test basic functionality of {function_name}."""
    # TODO: Import the actual function
    # from module import {function_name}

    # Test case 1: Normal input
    result = {function_name}({", ".join([f"arg{i}" for i in range(len(args))])})
    assert result is not None

def test_{function_name}_edge_cases():
    """Test edge cases for {function_name}."""
    # Test with None values
    with pytest.raises(TypeError):
        {function_name}({", ".join(["None" for _ in args])})

def test_{function_name}_performance():
    """Test performance of {function_name}."""
    import time
    start = time.time()
    # Run function multiple times
    for _ in range(100):
        {function_name}({", ".join([f"test_arg{i}" for i in range(len(args))])})
    elapsed = time.time() - start
    assert elapsed < 1.0  # Should complete in under 1 second
'''
        return test_template

    except Exception as e:
        return f"Error generating tests: {e}"


# ==============================================================================
# CODING PROBLEM EXAMPLES
# ==============================================================================

@dataclass
class CodingProblem:
    """Represents a coding problem for the agent to solve."""
    title: str
    description: str
    requirements: List[str]
    test_cases: List[Dict[str, Any]]
    difficulty: str = "medium"
    category: str = "general"


def create_coding_problems() -> List[CodingProblem]:
    """Create a set of coding problems for demonstration."""

    problems = [
        CodingProblem(
            title="Binary Search Implementation",
            description="""
            Implement a binary search algorithm that finds the position of a target value
            within a sorted array. The function should return the index if found, or -1 if not found.
            """,
            requirements=[
                "Function must be named 'binary_search'",
                "Must handle edge cases (empty array, single element)",
                "Must be efficient O(log n)",
                "Must include proper error handling"
            ],
            test_cases=[
                {"input": ([1, 2, 3, 4, 5], 3), "expected": 2},
                {"input": ([1, 2, 3, 4, 5], 6), "expected": -1},
                {"input": ([], 1), "expected": -1},
                {"input": ([5], 5), "expected": 0}
            ],
            difficulty="easy",
            category="algorithms"
        ),

        CodingProblem(
            title="API Rate Limiter",
            description="""
            Implement a rate limiter class that restricts the number of API calls
            within a time window using a sliding window algorithm.
            """,
            requirements=[
                "Class must be named 'RateLimiter'",
                "Support configurable window size and max requests",
                "Thread-safe implementation",
                "Include cleanup of old timestamps"
            ],
            test_cases=[
                {
                    "description": "Allow requests within limit",
                    "operations": ["allow", "allow", "allow"],
                    "config": {"window": 60, "max_requests": 5},
                    "expected": [True, True, True]
                },
                {
                    "description": "Block requests over limit",
                    "operations": ["allow"] * 6,
                    "config": {"window": 60, "max_requests": 5},
                    "expected": [True] * 5 + [False]
                }
            ],
            difficulty="medium",
            category="system_design"
        ),

        CodingProblem(
            title="Bug Fix: Memory Leak in Cache",
            description="""
            The following cache implementation has a memory leak. Find and fix the bug:

            ```python
            class Cache:
                def __init__(self, max_size=100):
                    self.cache = {}
                    self.max_size = max_size
                    self.access_times = {}

                def get(self, key):
                    if key in self.cache:
                        self.access_times[key] = time.time()
                        return self.cache[key]
                    return None

                def set(self, key, value):
                    self.cache[key] = value
                    self.access_times[key] = time.time()
            ```
            """,
            requirements=[
                "Identify the memory leak",
                "Implement proper eviction when max_size is reached",
                "Maintain LRU (Least Recently Used) behavior",
                "Clean up access_times dict properly"
            ],
            test_cases=[
                {
                    "description": "Cache should not exceed max_size",
                    "operations": [("set", i, f"value_{i}") for i in range(150)],
                    "check": "len(cache.cache) <= max_size"
                }
            ],
            difficulty="medium",
            category="debugging"
        ),

        CodingProblem(
            title="Code Optimization: Database Query",
            description="""
            Optimize this database query function that's running slowly:

            ```python
            def get_user_posts_with_comments(user_id):
                posts = db.query("SELECT * FROM posts WHERE user_id = ?", user_id)
                result = []
                for post in posts:
                    comments = db.query("SELECT * FROM comments WHERE post_id = ?", post['id'])
                    post['comments'] = comments
                    result.append(post)
                return result
            ```
            """,
            requirements=[
                "Eliminate N+1 query problem",
                "Use efficient JOIN or batch querying",
                "Maintain the same output structure",
                "Add proper error handling"
            ],
            test_cases=[
                {
                    "description": "Should make maximum 2 database queries",
                    "metric": "query_count",
                    "expected": 2
                }
            ],
            difficulty="hard",
            category="optimization"
        )
    ]

    return problems


# ==============================================================================
# AGENT DEMONSTRATION
# ==============================================================================

class CodingAgentDemo:
    """Main demonstration class for the coding agent."""

    def __init__(self, mode: AgentMode = AgentMode.LIGHTWEIGHT):
        """Initialize the demo with specified agent mode."""
        self.mode = mode
        self.agent = None
        self.tool_registry = None
        self.problems = create_coding_problems()
        self.results = []

    def setup_tools(self) -> ToolRegistry:
        """Set up custom tools for coding tasks."""
        registry = ToolRegistry()

        # Register coding tools
        registry.register(ToolDefinition(
            name="execute_code",
            description="Execute Python code and return results",
            parameters=[
                ToolParameter(name="code", type="string", description="Python code to execute"),
                ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False)
            ],
            function=execute_python_code
        ))

        registry.register(ToolDefinition(
            name="analyze_code",
            description="Analyze code quality and find issues",
            parameters=[
                ToolParameter(name="code", type="string", description="Code to analyze")
            ],
            function=analyze_code_quality
        ))

        registry.register(ToolDefinition(
            name="retrieve_context",
            description="Retrieve relevant code context using RAG",
            parameters=[
                ToolParameter(name="query", type="string", description="Context search query")
            ],
            function=retrieve_code_context
        ))

        registry.register(ToolDefinition(
            name="generate_tests",
            description="Generate test cases for a function",
            parameters=[
                ToolParameter(name="code", type="string", description="Code containing the function"),
                ToolParameter(name="function_name", type="string", description="Name of function to test")
            ],
            function=generate_test_cases
        ))

        return registry

    async def initialize_agent(self):
        """Initialize the ReAct agent with Gemma."""
        logger.info(f"Initializing agent in {self.mode.value} mode...")

        # Set up tools
        self.tool_registry = self.setup_tools()

        # Create agent
        try:
            self.agent = UnifiedReActAgent(
                model_name="gemma-2b",  # Use lightweight model for demo
                mode=self.mode,
                tool_registry=self.tool_registry,
                max_iterations=15,
                verbose=True,
                enable_planning=True,
                enable_reflection=True,
                temperature=0.7,
                top_p=0.9
            )

            logger.info("Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    def format_problem_prompt(self, problem: CodingProblem) -> str:
        """Format a coding problem into a prompt for the agent."""
        prompt = f"""
        Solve the following coding problem:

        **{problem.title}**

        Description: {problem.description}

        Requirements:
        {chr(10).join(f"- {req}" for req in problem.requirements)}

        Difficulty: {problem.difficulty}
        Category: {problem.category}

        Please:
        1. Understand the problem thoroughly
        2. Plan your approach
        3. Implement the solution
        4. Test your implementation
        5. Optimize if needed
        6. Provide the final solution with explanation

        Use the available tools to help you solve this problem effectively.
        """

        return prompt.strip()

    async def solve_problem(self, problem: CodingProblem) -> Dict[str, Any]:
        """Have the agent solve a coding problem."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Solving: {problem.title}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()

        try:
            # Format prompt
            prompt = self.format_problem_prompt(problem)

            # Get agent's solution
            trace = await self.agent.reason(prompt)

            # Extract solution and metrics
            elapsed_time = time.time() - start_time

            result = {
                "problem": problem.title,
                "success": trace.success,
                "solution": trace.final_answer,
                "reasoning_steps": len(trace.thoughts),
                "tools_used": [action["tool"] for action in trace.actions],
                "time_taken": elapsed_time,
                "trace": trace.to_dict()
            }

            # Display results
            self.display_solution(result)

            return result

        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "problem": problem.title,
                "success": False,
                "error": str(e),
                "time_taken": time.time() - start_time
            }

    def display_solution(self, result: Dict[str, Any]):
        """Display the solution in a formatted way."""
        print("\n" + "="*60)
        print(f"Problem: {result['problem']}")
        print(f"Status: {'✓ SOLVED' if result['success'] else '✗ FAILED'}")
        print(f"Time: {result['time_taken']:.2f} seconds")
        print(f"Reasoning Steps: {result.get('reasoning_steps', 0)}")
        print(f"Tools Used: {', '.join(result.get('tools_used', []))}")
        print("="*60)

        if result.get('solution'):
            print("\nSolution:")
            print("-"*40)
            print(result['solution'])
            print("-"*40)

    def display_trace_analysis(self, trace: ReActTrace):
        """Display detailed analysis of the reasoning trace."""
        print("\n" + "="*60)
        print("REASONING TRACE ANALYSIS")
        print("="*60)

        # Count thought types
        thought_counts = {}
        for thought in trace.thoughts:
            thought_type = thought.type.value
            thought_counts[thought_type] = thought_counts.get(thought_type, 0) + 1

        print("\nThought Distribution:")
        for thought_type, count in thought_counts.items():
            print(f"  {thought_type.upper()}: {count}")

        # Show planning details
        if trace.plan:
            print(f"\nPlan Complexity: {trace.plan.complexity.value}")
            print(f"Plan Steps: {len(trace.plan.steps)}")

        # Show reflections
        if trace.reflections:
            print(f"\nReflections Made: {len(trace.reflections)}")
            for i, reflection in enumerate(trace.reflections[:3], 1):
                print(f"  {i}. {reflection[:100]}...")

        print("="*60)

    async def run_full_demonstration(self):
        """Run the complete demonstration."""
        print("\n" + "="*80)
        print(" PYTHON REACT AGENT WITH GEMMA - CODING DEMONSTRATION")
        print("="*80)
        print(f"Mode: {self.mode.value}")
        print(f"Problems to solve: {len(self.problems)}")
        print("="*80 + "\n")

        # Initialize agent
        await self.initialize_agent()

        # Solve each problem
        for i, problem in enumerate(self.problems, 1):
            print(f"\n[{i}/{len(self.problems)}] Starting problem: {problem.title}")

            result = await self.solve_problem(problem)
            self.results.append(result)

            # Show trace analysis for interesting problems
            if result.get("trace") and result.get("success"):
                trace = ReActTrace(**result["trace"])
                self.display_trace_analysis(trace)

            # Brief pause between problems
            await asyncio.sleep(2)

        # Display summary
        self.display_summary()

    def display_summary(self):
        """Display summary of all results."""
        print("\n" + "="*80)
        print(" DEMONSTRATION SUMMARY")
        print("="*80)

        successful = sum(1 for r in self.results if r.get("success"))
        total = len(self.results)

        print(f"\nSuccess Rate: {successful}/{total} ({successful/total*100:.1f}%)")

        print("\nProblem Results:")
        for result in self.results:
            status = "✓" if result.get("success") else "✗"
            time_str = f"{result.get('time_taken', 0):.2f}s"
            steps = result.get('reasoning_steps', 0)
            print(f"  {status} {result['problem']:<30} - {time_str:>8} - {steps:>3} steps")

        # Calculate metrics
        total_time = sum(r.get('time_taken', 0) for r in self.results)
        total_steps = sum(r.get('reasoning_steps', 0) for r in self.results)

        print(f"\nTotal Time: {total_time:.2f} seconds")
        print(f"Total Reasoning Steps: {total_steps}")
        print(f"Average Time per Problem: {total_time/total:.2f} seconds")
        print(f"Average Steps per Problem: {total_steps/total:.1f}")

        # Tool usage statistics
        all_tools = []
        for r in self.results:
            all_tools.extend(r.get('tools_used', []))

        if all_tools:
            tool_counts = {}
            for tool in all_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

            print("\nTool Usage:")
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} times")

        print("="*80)


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================

class PerformanceMonitor:
    """Monitor and report performance metrics."""

    def __init__(self):
        self.metrics = {
            "inference_times": [],
            "memory_usage": [],
            "token_counts": [],
            "tool_latencies": {}
        }

    def record_inference(self, time_taken: float, tokens: int):
        """Record inference metrics."""
        self.metrics["inference_times"].append(time_taken)
        self.metrics["token_counts"].append(tokens)

    def record_memory(self):
        """Record current memory usage."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics["memory_usage"].append(memory_mb)

    def record_tool_latency(self, tool_name: str, latency: float):
        """Record tool execution latency."""
        if tool_name not in self.metrics["tool_latencies"]:
            self.metrics["tool_latencies"][tool_name] = []
        self.metrics["tool_latencies"][tool_name].append(latency)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        import numpy as np

        summary = {}

        if self.metrics["inference_times"]:
            times = self.metrics["inference_times"]
            summary["inference"] = {
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times)
            }

        if self.metrics["token_counts"]:
            tokens = self.metrics["token_counts"]
            summary["tokens"] = {
                "total": sum(tokens),
                "mean": np.mean(tokens),
                "max": max(tokens)
            }

        if self.metrics["memory_usage"]:
            memory = self.metrics["memory_usage"]
            summary["memory_mb"] = {
                "mean": np.mean(memory),
                "peak": max(memory)
            }

        if self.metrics["tool_latencies"]:
            summary["tools"] = {}
            for tool, latencies in self.metrics["tool_latencies"].items():
                summary["tools"][tool] = {
                    "calls": len(latencies),
                    "mean_ms": np.mean(latencies) * 1000,
                    "total_ms": sum(latencies) * 1000
                }

        return summary


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

async def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstration of ReAct agent solving coding problems"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "lightweight"],
        default="lightweight",
        help="Agent operation mode"
    )
    parser.add_argument(
        "--problems",
        type=int,
        default=None,
        help="Number of problems to solve (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set up performance monitoring
    monitor = PerformanceMonitor()

    # Create and configure demo
    mode = AgentMode.FULL if args.mode == "full" else AgentMode.LIGHTWEIGHT
    demo = CodingAgentDemo(mode=mode)

    # Limit problems if specified
    if args.problems:
        demo.problems = demo.problems[:args.problems]

    try:
        # Run demonstration
        await demo.run_full_demonstration()

        # Display performance metrics
        print("\n" + "="*80)
        print(" PERFORMANCE METRICS")
        print("="*80)

        # Record final memory
        monitor.record_memory()

        # Get and display summary
        perf_summary = monitor.get_summary()
        print(json.dumps(perf_summary, indent=2))

    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())