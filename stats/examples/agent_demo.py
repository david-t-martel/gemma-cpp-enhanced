"""Demonstration of LLM agent using various tools."""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.gemma_agent import AgentMode, create_gemma_agent
from src.agent.tools import tool_registry


def demo_calculator():
    """Demonstrate calculator tool usage."""
    print("\n" + "=" * 60)
    print("📐 CALCULATOR DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True)

    queries = [
        "What is 25 * 37?",
        "Calculate sin(pi/2) + cos(0)",
        "What's the square root of 144?",
        "Calculate (100 + 50) * 2 - 75",
    ]

    for query in queries:
        print(f"\n❓ Question: {query}")
        response = agent.chat(query)
        print(f"💬 Response: {response}")
        print("-" * 40)


def demo_file_operations():
    """Demonstrate file operation tools."""
    print("\n" + "=" * 60)
    print("📁 FILE OPERATIONS DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True)

    # Create a test file
    test_file = "test_demo.txt"
    print(f"\n❓ Question: Write 'Hello from LLM Agent!' to a file named {test_file}")
    response = agent.chat(f"Write 'Hello from LLM Agent!' to a file named {test_file}")
    print(f"💬 Response: {response}")

    # Read the file
    print(f"\n❓ Question: Read the contents of {test_file}")
    response = agent.chat(f"Read the contents of {test_file}")
    print(f"💬 Response: {response}")

    # List directory
    print("\n❓ Question: List the files in the current directory")
    response = agent.chat("List the files in the current directory")
    print(f"💬 Response: {response}")


def demo_system_info():
    """Demonstrate system information tool."""
    print("\n" + "=" * 60)
    print("💻 SYSTEM INFO DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True)

    queries = [
        "What operating system am I running?",
        "How much memory does this system have?",
        "Show me the CPU information",
        "What's the current disk usage?",
    ]

    for query in queries:
        print(f"\n❓ Question: {query}")
        response = agent.chat(query)
        print(f"💬 Response: {response}")
        print("-" * 40)


def demo_datetime():
    """Demonstrate datetime tool."""
    print("\n" + "=" * 60)
    print("🕐 DATETIME DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True)

    queries = [
        "What's the current date and time?",
        "What time is it in UTC?",
        "Show me the current Unix timestamp",
    ]

    for query in queries:
        print(f"\n❓ Question: {query}")
        response = agent.chat(query)
        print(f"💬 Response: {response}")
        print("-" * 40)


def demo_web_search():
    """Demonstrate web search tool."""
    print("\n" + "=" * 60)
    print("🔍 WEB SEARCH DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True)

    queries = [
        "Search for information about Python programming",
        "Find information about machine learning",
        "Search for LLM agents",
    ]

    for query in queries:
        print(f"\n❓ Question: {query}")
        response = agent.chat(query)
        print(f"💬 Response: {response}")
        print("-" * 40)


def demo_multi_tool():
    """Demonstrate using multiple tools in one conversation."""
    print("\n" + "=" * 60)
    print("🔧 MULTI-TOOL DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=True)

    print(
        "\n❓ Complex Question: Calculate 100 * 50, then write the result to a file called 'calculation.txt', and finally tell me what time it is."
    )
    response = agent.chat(
        "Calculate 100 * 50, then write the result to a file called 'calculation.txt', and finally tell me what time it is."
    )
    print(f"💬 Response: {response}")


def demo_conversation():
    """Demonstrate conversation with history."""
    print("\n" + "=" * 60)
    print("💬 CONVERSATION DEMO")
    print("=" * 60)

    agent = create_gemma_agent(mode=AgentMode.LIGHTWEIGHT, verbose=False)

    conversation = [
        "Hi! I need help with some calculations.",
        "Calculate 50 + 75 for me",
        "Now multiply that result by 2",
        "Great! Can you write this final result to a file called 'result.txt'?",
        "Now read that file back to me to confirm",
    ]

    for message in conversation:
        print(f"\n👤 You: {message}")
        response = agent.chat(message)
        print(f"🤖 Agent: {response}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("🚀 LLM AGENT TOOL DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo will showcase various tool capabilities.")
    print("Note: Using lightweight Gemma model for quick demonstrations.")

    demos = [
        ("Calculator", demo_calculator),
        ("System Info", demo_system_info),
        ("Datetime", demo_datetime),
        ("File Operations", demo_file_operations),
        ("Web Search", demo_web_search),
        ("Multi-Tool", demo_multi_tool),
        ("Conversation", demo_conversation),
    ]

    for name, demo_func in demos:
        try:
            input(f"\n\nPress Enter to run {name} demo (or Ctrl+C to skip)...")
            demo_func()
        except KeyboardInterrupt:
            print(f"\nSkipping {name} demo...")
            continue
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nTo run the interactive agent:")
    print("  python main.py --lightweight")
    print("\nTo use a different model:")
    print("  python main.py --model google/gemma-7b-it")
    print("\nFor more options:")
    print("  python main.py --help")


if __name__ == "__main__":
    main()
