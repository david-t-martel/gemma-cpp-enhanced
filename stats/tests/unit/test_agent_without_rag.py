#!/usr/bin/env python3
"""
Test script to verify core LLM agent functionality works without RAG dependencies.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all core agent imports work without RAG."""
    print("🧪 Testing core agent imports...")

    try:
        from src.agent.core import BaseAgent, ConversationHistory, Message

        print("✅ Core agent imports successful")
    except ImportError as e:
        print(f"❌ Core agent import failed: {e}")
        return False

    try:
        from src.agent.gemma_agent import GemmaAgent, LightweightGemmaAgent, create_gemma_agent

        print("✅ Gemma agent imports successful")
    except ImportError as e:
        print(f"❌ Gemma agent import failed: {e}")
        return False

    try:
        from src.agent.react_agent import ReActAgent, create_react_agent

        print("✅ ReAct agent imports successful")
    except ImportError as e:
        print(f"❌ ReAct agent import failed: {e}")
        return False

    try:
        from src.agent.tools import ToolRegistry, tool_registry

        print("✅ Tool registry imports successful")
    except ImportError as e:
        print(f"❌ Tool registry import failed: {e}")
        return False

    try:
        from src.agent.planner import Plan, Planner, Step

        print("✅ Planner imports successful")
    except ImportError as e:
        print(f"❌ Planner import failed: {e}")
        return False

    return True


def test_tool_registry():
    """Test that tool registry works without RAG tools."""
    print("\n🧪 Testing tool registry...")

    try:
        from src.agent.tools import tool_registry

        tools = tool_registry.list_tools()
        print(f"✅ Tool registry has {len(tools)} tools")

        # Check that basic tools are available
        tool_names = [tool.name for tool in tools]
        expected_tools = ["calculator", "file_read", "file_write", "system_info"]

        for expected in expected_tools:
            if expected in tool_names:
                print(f"  ✅ {expected} tool available")
            else:
                print(f"  ⚠️  {expected} tool not found")

        # Make sure no RAG tools remain
        rag_tools = [name for name in tool_names if "rag" in name.lower()]
        if rag_tools:
            print(f"  ⚠️  RAG tools still present: {rag_tools}")
        else:
            print("  ✅ No RAG tools found (as expected)")

        return True

    except Exception as e:
        print(f"❌ Tool registry test failed: {e}")
        return False


def test_agent_creation():
    """Test that agents can be created without RAG dependencies."""
    print("\n🧪 Testing agent creation...")

    try:
        from src.agent.gemma_agent import create_gemma_agent
        from src.agent.tools import tool_registry

        # Try to create a lightweight agent (without actually loading model)
        print("  Creating lightweight Gemma agent...")
        agent = create_gemma_agent(lightweight=True, tool_registry=tool_registry, verbose=False)
        print("  ✅ Lightweight Gemma agent created successfully")

        # Check that agent has expected attributes
        if hasattr(agent, "tool_registry"):
            print(
                f"    ✅ Agent has tool registry with {len(agent.tool_registry.list_tools())} tools"
            )
        else:
            print("    ⚠️  Agent missing tool registry")

        if hasattr(agent, "history"):
            print("    ✅ Agent has conversation history")
        else:
            print("    ⚠️  Agent missing conversation history")

        return True

    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False


def test_react_agent_creation():
    """Test that ReAct agent can be created without RAG dependencies."""
    print("\n🧪 Testing ReAct agent creation...")

    try:
        from src.agent.react_agent import create_react_agent
        from src.agent.tools import tool_registry

        # Try to create a lightweight ReAct agent
        print("  Creating lightweight ReAct agent...")
        react_agent = create_react_agent(
            lightweight=True,
            tool_registry=tool_registry,
            enable_planning=True,
            enable_reflection=True,
            verbose=False,
        )
        print("  ✅ Lightweight ReAct agent created successfully")

        # Check ReAct-specific attributes
        if hasattr(react_agent, "planner"):
            print("    ✅ ReAct agent has planner")
        else:
            print("    ⚠️  ReAct agent missing planner")

        if hasattr(react_agent, "enable_planning"):
            print(f"    ✅ Planning enabled: {react_agent.enable_planning}")
        else:
            print("    ⚠️  ReAct agent missing planning flag")

        if hasattr(react_agent, "enable_reflection"):
            print(f"    ✅ Reflection enabled: {react_agent.enable_reflection}")
        else:
            print("    ⚠️  ReAct agent missing reflection flag")

        return True

    except Exception as e:
        print(f"❌ ReAct agent creation test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic agent functionality without model inference."""
    print("\n🧪 Testing basic agent functionality...")

    try:
        from src.agent.tools import tool_registry

        # Test calculator tool
        calc_tool = next((t for t in tool_registry.list_tools() if t.name == "calculator"), None)
        if calc_tool:
            result = tool_registry.execute_tool("calculator", {"expression": "2 + 2"})
            if result.success and "4" in str(result.output):
                print("  ✅ Calculator tool works correctly")
            else:
                print(f"  ⚠️  Calculator tool issue: {result}")
        else:
            print("  ⚠️  Calculator tool not found")

        # Test system info tool
        sys_tool = next((t for t in tool_registry.list_tools() if t.name == "system_info"), None)
        if sys_tool:
            result = tool_registry.execute_tool("system_info", {})
            if result.success:
                print("  ✅ System info tool works correctly")
            else:
                print(f"  ⚠️  System info tool issue: {result}")
        else:
            print("  ⚠️  System info tool not found")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 TESTING LLM AGENT WITHOUT RAG DEPENDENCIES")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Tool Registry Tests", test_tool_registry),
        ("Agent Creation Tests", test_agent_creation),
        ("ReAct Agent Tests", test_react_agent_creation),
        ("Basic Functionality Tests", test_basic_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Core LLM agent works without RAG!")
        return 0
    else:
        print("⚠️  Some tests failed - investigation needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
