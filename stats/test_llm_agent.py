#!/usr/bin/env python
"""Test script for LLM agent functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.react_agent import create_react_agent
from src.domain.tools.base import ToolRegistry


def test_agent():
    """Test the LLM agent with a simple query."""
    print("🚀 Testing LLM Agent...")

    # Create agent
    agent = create_react_agent(lightweight=True)
    print("✅ Agent created successfully")

    # List tools
    registry = ToolRegistry()
    tools = registry.list_tools()
    print(f"✅ {len(tools)} tools loaded: {', '.join(tools)}")

    # Test simple query
    test_query = "What is 2 + 2?"
    print(f"\n📝 Test query: {test_query}")

    try:
        response = agent.process(test_query)
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test tool usage
    test_tool_query = "Use the calculator tool to compute 15 * 23"
    print(f"\n📝 Test tool query: {test_tool_query}")

    try:
        response = agent.process(test_tool_query)
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_agent()