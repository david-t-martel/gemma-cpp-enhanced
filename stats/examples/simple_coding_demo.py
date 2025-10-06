#!/usr/bin/env python3
"""
Simple demonstration of the ReAct agent solving coding problems.
This is a simplified version that focuses on core functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.gemma_agent import AgentMode
from src.agent.react_agent import UnifiedReActAgent
from src.agent.tools import ToolRegistry, ToolDefinition, ToolParameter
from src.shared.logging import setup_logging, get_logger, LogLevel

# Setup logging
setup_logging(level=LogLevel.INFO, console=True)
logger = get_logger(__name__)


def execute_python(code: str) -> str:
    """Execute Python code and return output."""
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )

        output = result.stdout if result.returncode == 0 else result.stderr
        os.unlink(temp_file)
        return output or "Code executed successfully"
    except Exception as e:
        return f"Error: {e}"


def analyze_code(code: str) -> str:
    """Analyze code and provide metrics."""
    lines = code.splitlines()

    # Count functions and classes
    func_count = sum(1 for line in lines if line.strip().startswith('def '))
    class_count = sum(1 for line in lines if line.strip().startswith('class '))

    return f"Code Analysis:\n- Lines: {len(lines)}\n- Functions: {func_count}\n- Classes: {class_count}"


def main():
    """Main demonstration function."""

    print("="*70)
    print(" REACT AGENT - CODING DEMONSTRATION (SIMPLIFIED)")
    print("="*70)

    # Create tool registry
    print("\n1. Setting up tools...")
    tool_registry = ToolRegistry()

    # Add custom tools
    tool_registry.register(ToolDefinition(
        name="execute_python",
        description="Execute Python code and return output",
        parameters=[
            ToolParameter(name="code", type="string", description="Python code to execute")
        ],
        function=execute_python
    ))

    tool_registry.register(ToolDefinition(
        name="analyze_code",
        description="Analyze code structure and metrics",
        parameters=[
            ToolParameter(name="code", type="string", description="Code to analyze")
        ],
        function=analyze_code
    ))

    print(f"   Tools registered: {', '.join(tool_registry.tools.keys())}")

    # Initialize agent
    print("\n2. Initializing ReAct agent...")
    try:
        agent = UnifiedReActAgent(
            model_name="gemma-2b",
            mode=AgentMode.LIGHTWEIGHT,
            tool_registry=tool_registry,
            max_iterations=10,
            verbose=True,
            enable_planning=False,  # Disable for simplicity
            enable_reflection=False,  # Disable for simplicity
            temperature=0.7
        )
        print("   Agent initialized successfully")
    except Exception as e:
        print(f"   ERROR: Failed to initialize agent: {e}")
        return 1

    # Define coding problems
    problems = [
        {
            "title": "Fibonacci Function",
            "prompt": """
            Write a Python function called 'fibonacci' that returns the nth Fibonacci number.
            Use an efficient approach (not recursive). Handle n <= 0 by returning 0.
            Test it with n=10.
            """
        },
        {
            "title": "List Reversal",
            "prompt": """
            Write a Python function called 'reverse_list' that reverses a list in-place.
            Do not use the built-in reverse() method. Test it with [1, 2, 3, 4, 5].
            """
        },
        {
            "title": "Bug Fix",
            "prompt": """
            Fix this buggy code:

            def find_max(numbers):
                max_num = 0
                for num in numbers:
                    if num > max_num:
                        max_num = num
                return max_num

            The bug is that it doesn't handle negative numbers correctly.
            Fix it and test with [-5, -2, -8, -1].
            """
        }
    ]

    # Solve each problem
    print("\n3. Solving coding problems...")
    print("-"*70)

    for i, problem in enumerate(problems, 1):
        print(f"\n[Problem {i}] {problem['title']}")
        print("-"*50)

        try:
            # Get solution from agent
            solution = agent.solve(problem['prompt'])

            print("\nSolution:")
            print(solution[:500] + "..." if len(solution) > 500 else solution)

        except Exception as e:
            print(f"Error solving problem: {e}")

        print("-"*50)

    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())