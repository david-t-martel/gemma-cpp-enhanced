"""Simple usage examples for the ReAct agent."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.gemma_agent import AgentMode
from src.agent.react_agent import create_react_agent
from src.agent.tools import ToolDefinition, ToolParameter, ToolRegistry


def example_simple_reasoning():
    """Example: Simple reasoning task."""
    print("=" * 60)
    print("Example 1: Simple Reasoning")
    print("=" * 60)

    # Create a ReAct agent
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        verbose=True,
        enable_planning=False,  # Simple task doesn't need planning
    )

    # Ask a simple question
    result = agent.solve("What is 25 * 4 + 10?")

    print(f"\nFinal Answer: {result}")
    print("-" * 60)


def example_with_planning():
    """Example: Task that requires planning."""
    print("\n" + "=" * 60)
    print("Example 2: Task with Planning")
    print("=" * 60)

    # Create agent with planning enabled
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT, verbose=True, enable_planning=True, enable_reflection=True
    )

    # Complex task requiring multiple steps
    task = """
    I need to:
    1. Calculate the area of a circle with radius 5
    2. Write the result to a file called 'area.txt'
    3. Then read the file back to confirm it was saved correctly
    """

    result = agent.solve(task)

    print(f"\nFinal Answer: {result}")

    # Show the reasoning trace
    print("\nReasoning Summary:")
    print(agent.get_trace_summary())
    print("-" * 60)


def example_custom_tools():
    """Example: Using custom tools with ReAct agent."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Tools")
    print("=" * 60)

    # Create custom tools
    def convert_temperature(temp: float, from_unit: str, to_unit: str) -> str:
        """Convert temperature between units."""
        if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
            result = (temp * 9 / 5) + 32
            return f"{temp}°C = {result:.1f}°F"
        elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
            result = (temp - 32) * 5 / 9
            return f"{temp}°F = {result:.1f}°C"
        else:
            return "Unsupported conversion"

    # Create tool registry with custom tool
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="convert_temperature",
            description="Convert temperature between Celsius and Fahrenheit",
            parameters=[
                ToolParameter(name="temp", type="number", description="Temperature value"),
                ToolParameter(
                    name="from_unit", type="string", description="Source unit (Celsius/Fahrenheit)"
                ),
                ToolParameter(
                    name="to_unit", type="string", description="Target unit (Celsius/Fahrenheit)"
                ),
            ],
            function=convert_temperature,
        )
    )

    # Create agent with custom tools
    agent = create_react_agent(mode=AgentMode.LIGHTWEIGHT, tool_registry=registry, verbose=True)

    # Use the custom tool
    result = agent.solve("Convert 100 degrees Fahrenheit to Celsius")

    print(f"\nFinal Answer: {result}")
    print("-" * 60)


def example_error_recovery():
    """Example: Error recovery and self-correction."""
    print("\n" + "=" * 60)
    print("Example 4: Error Recovery")
    print("=" * 60)

    # Create agent with reflection for error recovery
    agent = create_react_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True, enable_reflection=True)

    # Task that might cause errors
    task = """
    Try to read a file called 'nonexistent.txt'.
    If it doesn't exist, create a new file called 'backup.txt'
    with the content 'File not found, created backup instead'.
    """

    result = agent.solve(task)

    print(f"\nFinal Answer: {result}")
    print("-" * 60)


def example_multi_step_analysis():
    """Example: Multi-step analysis with reflection."""
    print("\n" + "=" * 60)
    print("Example 5: Multi-Step Analysis")
    print("=" * 60)

    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        verbose=True,
        enable_planning=True,
        enable_reflection=True,
        max_iterations=15,
    )

    # Complex analytical task
    task = """
    Analyze the following:
    1. Calculate the average of these numbers: 45, 67, 89, 23, 56
    2. Determine if the average is above or below 50
    3. Based on the result, provide a recommendation:
       - If above 50: "Performance is good"
       - If below 50: "Needs improvement"
    4. Save your analysis to 'analysis.txt'
    """

    result = agent.solve(task)

    print(f"\nFinal Answer: {result}")

    # Save trace for later analysis
    agent.save_trace("analysis_trace.json")
    print("\nTrace saved to 'analysis_trace.json'")
    print("-" * 60)


def example_comparison():
    """Example: Comparing regular agent vs ReAct agent."""
    print("\n" + "=" * 60)
    print("Example 6: Regular Agent vs ReAct Agent")
    print("=" * 60)

    task = "Calculate the sum of 1 to 10, then multiply by 2"

    # Regular Gemma agent
    print("Regular Gemma Agent:")
    from src.agent.gemma_agent import create_gemma_agent

    regular_agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=False)
    regular_result = regular_agent.chat(task)
    print(f"Result: {regular_result}\n")

    # ReAct agent
    print("ReAct Agent with Reasoning:")
    react_agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT, verbose=True, enable_planning=False, enable_reflection=False
    )
    react_result = react_agent.solve(task)
    print(f"\nResult: {react_result}")
    print("-" * 60)


def interactive_example():
    """Interactive example where user can input queries."""
    print("\n" + "=" * 60)
    print("Interactive ReAct Agent")
    print("=" * 60)

    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT, verbose=True, enable_planning=True, enable_reflection=True
    )

    print("\nEnter your queries (type 'quit' to exit)")
    print("The agent will use reasoning to solve your problems.\n")

    while True:
        query = input("Your query: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not query:
            continue

        print(f"\nProcessing: {query}\n")
        print("-" * 40)

        result = agent.solve(query)

        print("-" * 40)
        print(f"\nFinal Answer: {result}\n")

        show_trace = input("Show reasoning trace? (y/n): ").strip().lower()
        if show_trace == "y":
            print(agent.get_trace_summary())

        print("=" * 60)


def main():
    """Run all examples or specific ones."""
    import argparse

    parser = argparse.ArgumentParser(description="ReAct Agent Usage Examples")
    parser.add_argument(
        "--example", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run specific example (1-6)"
    )
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║           ReAct Agent Usage Examples                      ║
║    Reasoning and Acting for Problem Solving               ║
╚══════════════════════════════════════════════════════════╝
    """)

    examples = [
        (1, "Simple Reasoning", example_simple_reasoning),
        (2, "Task with Planning", example_with_planning),
        (3, "Custom Tools", example_custom_tools),
        (4, "Error Recovery", example_error_recovery),
        (5, "Multi-Step Analysis", example_multi_step_analysis),
        (6, "Agent Comparison", example_comparison),
    ]

    if args.interactive:
        interactive_example()
    elif args.example:
        for num, name, func in examples:
            if num == args.example:
                print(f"Running Example {num}: {name}\n")
                func()
                break
    else:
        print("Running all examples...\n")
        for num, name, func in examples:
            try:
                input(f"\nPress Enter to run Example {num}: {name} (Ctrl+C to skip)...")
                func()
            except KeyboardInterrupt:
                print(f"\nSkipping Example {num}")
                continue
            except Exception as e:
                print(f"\nError in Example {num}: {e}")

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("\nTo run interactive mode:")
        print("  python react_usage.py --interactive")
        print("\nTo run a specific example:")
        print("  python react_usage.py --example 2")


if __name__ == "__main__":
    main()
