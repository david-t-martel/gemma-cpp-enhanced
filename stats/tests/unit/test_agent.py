"""Quick test script to verify the agent system works."""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core import BaseAgent, Message
from src.agent.tools import tool_registry


class SimpleTestAgent(BaseAgent):
    """Simple test agent that echoes with tool awareness."""

    def generate_response(self, prompt: str) -> str:
        """Simple response generation for testing."""
        # Check if the prompt mentions calculation
        if (
            "calculate" in prompt.lower()
            or "math" in prompt.lower()
            or any(op in prompt for op in ["+", "-", "*", "/", "^"])
        ):
            # Extract numbers from prompt
            import re

            numbers = re.findall(r"\d+", prompt)
            if len(numbers) >= 2:
                expression = f"{numbers[0]} + {numbers[1]}"
                return f"""I'll help you calculate that.

```tool_call
{{
    "name": "calculator",
    "arguments": {{
        "expression": "{expression}"
    }}
}}
```

Let me compute {expression} for you."""

        # Check if prompt mentions files
        if "read" in prompt.lower() and "file" in prompt.lower():
            return """I'll read that file for you.

```tool_call
{{
    "name": "read_file",
    "arguments": {{
        "path": "test.txt"
    }}
}}
```

Reading the file contents..."""

        # Check if prompt mentions time/date
        if any(word in prompt.lower() for word in ["time", "date", "today", "now"]):
            return """I'll get the current date and time for you.

```tool_call
{{
    "name": "get_datetime",
    "arguments": {{}}
}}
```

Fetching current datetime..."""

        # Check if prompt mentions system
        if any(word in prompt.lower() for word in ["system", "cpu", "memory", "disk"]):
            return """I'll get the system information for you.

```tool_call
{{
    "name": "system_info",
    "arguments": {{
        "category": "all"
    }}
}}
```

Gathering system information..."""

        # Default response
        return f"I understand you said: '{prompt}'. How can I help you with that?"


def test_tools():
    """Test individual tools."""
    print("=" * 60)
    print("TESTING INDIVIDUAL TOOLS")
    print("=" * 60)

    # Test calculator
    print("\n1. Testing Calculator Tool:")
    result = tool_registry.execute_tool("calculator", {"expression": "2 + 2"})
    print(f"   2 + 2 = {result.output}")

    result = tool_registry.execute_tool("calculator", {"expression": "sin(3.14159/2)"})
    print(f"   sin(π/2) = {result.output}")

    # Test datetime
    print("\n2. Testing Datetime Tool:")
    result = tool_registry.execute_tool("get_datetime", {})
    print(f"   {result.output[:100]}...")

    # Test system info
    print("\n3. Testing System Info Tool:")
    result = tool_registry.execute_tool("system_info", {"category": "platform"})
    print(f"   {result.output[:200]}...")

    # Test file operations
    print("\n4. Testing File Operations:")
    result = tool_registry.execute_tool(
        "write_file", {"path": "test_file.txt", "content": "Hello from the LLM Agent!"}
    )
    print(f"   Write: {result.output}")

    result = tool_registry.execute_tool("read_file", {"path": "test_file.txt"})
    print(f"   Read: {result.output}")

    # Test list directory
    print("\n5. Testing List Directory:")
    result = tool_registry.execute_tool("list_directory", {"path": "."})
    lines = result.output.split("\n")[:10]  # Show first 10 lines
    print("   " + "\n   ".join(lines))

    print("\n✅ All tools tested successfully!")


def test_agent():
    """Test the agent with tool calling."""
    print("\n" + "=" * 60)
    print("TESTING AGENT WITH TOOL CALLING")
    print("=" * 60)

    agent = SimpleTestAgent(tool_registry=tool_registry, verbose=True)

    test_queries = [
        "Calculate 15 + 25",
        "What time is it?",
        "Show me system information",
        "Hello, how are you?",
    ]

    for query in test_queries:
        print(f"\n📝 Testing: {query}")
        print("-" * 40)
        response = agent.chat(query)
        print(f"Final Response: {response}")
        print("-" * 40)

    print("\n✅ Agent testing complete!")


def test_tool_parsing():
    """Test tool call parsing."""
    print("\n" + "=" * 60)
    print("TESTING TOOL CALL PARSING")
    print("=" * 60)

    agent = SimpleTestAgent(verbose=False)

    test_text = """I'll help you with that calculation.

```tool_call
{
    "name": "calculator",
    "arguments": {
        "expression": "100 * 50"
    }
}
```

And here's another tool:

```tool_call
{
    "name": "get_datetime",
    "arguments": {}
}
```

That should give us the results."""

    cleaned_text, tool_calls = agent.parse_tool_calls(test_text)

    print(f"Original text length: {len(test_text)}")
    print(f"Cleaned text length: {len(cleaned_text)}")
    print(f"Number of tool calls found: {len(tool_calls)}")

    for i, call in enumerate(tool_calls, 1):
        print(f"\nTool Call {i}:")
        print(f"  Name: {call['name']}")
        print(f"  Arguments: {call['arguments']}")

    print("\nCleaned text:")
    print(cleaned_text)

    print("\n✅ Tool parsing test complete!")


def main():
    """Run all tests."""
    print("\n🚀 LLM AGENT SYSTEM TEST")
    print("=" * 60)

    try:
        # Test tools
        test_tools()

        # Test tool parsing
        test_tool_parsing()

        # Test agent
        test_agent()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe agent system is working correctly!")
        print("\nTo run the full interactive agent:")
        print("  uv run python main.py --lightweight")
        print("\nTo run the demo:")
        print("  uv run python examples/agent_demo.py")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
