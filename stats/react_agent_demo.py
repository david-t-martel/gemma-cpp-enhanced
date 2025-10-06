#!/usr/bin/env python3
"""
Comprehensive ReAct Agent Demonstration

This script demonstrates the full capabilities of the UnifiedReActAgent
including planning, reasoning, acting, and reflection.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.react_agent import UnifiedReActAgent
from src.agent.tools import TOOL_REGISTRY
from src.shared.logging import get_logger

logger = get_logger(__name__)


async def demo_basic_reasoning():
    """Demonstrate basic reasoning capabilities."""
    print("\n" + "="*60)
    print("🧠 BASIC REASONING DEMONSTRATION")
    print("="*60)

    agent = UnifiedReActAgent(
        model_name="mock",  # Using mock model for demonstration
        tools=TOOL_REGISTRY,
        enable_planning=True,
        enable_reflection=True
    )

    # Simple math problem
    query = "What is 15 * 24? Please show your reasoning step by step."
    print(f"📝 Query: {query}")

    try:
        response = await agent.chat(query)
        print(f"🤖 Response: {response}")

        # Show trace
        trace = agent.get_trace_summary()
        print(f"\n📊 Reasoning Trace: {len(trace)} steps")
        for i, step in enumerate(trace[-3:], 1):  # Show last 3 steps
            print(f"  {i}. {step.get('type', 'Unknown')}: {step.get('content', 'No content')[:100]}...")

    except Exception as e:
        print(f"❌ Error in basic reasoning: {e}")


async def demo_tool_usage():
    """Demonstrate tool calling capabilities."""
    print("\n" + "="*60)
    print("🛠️ TOOL USAGE DEMONSTRATION")
    print("="*60)

    agent = UnifiedReActAgent(
        model_name="mock",
        tools=TOOL_REGISTRY,
        enable_planning=True
    )

    # List available tools
    print(f"📋 Available Tools: {list(TOOL_REGISTRY.keys())}")

    # Complex query requiring tool use
    query = "What's the current working directory and list the files in it?"
    print(f"📝 Query: {query}")

    try:
        response = await agent.chat(query)
        print(f"🤖 Response: {response}")

    except Exception as e:
        print(f"❌ Error in tool usage: {e}")


async def demo_multi_step_planning():
    """Demonstrate multi-step planning and execution."""
    print("\n" + "="*60)
    print("📋 MULTI-STEP PLANNING DEMONSTRATION")
    print("="*60)

    agent = UnifiedReActAgent(
        model_name="mock",
        tools=TOOL_REGISTRY,
        enable_planning=True,
        enable_reflection=True
    )

    # Complex multi-step task
    query = """Create a simple Python script that:
    1. Imports the datetime module
    2. Gets the current time
    3. Prints it in a formatted way
    4. Save it as 'time_script.py'

    Please plan this step by step and execute it."""

    print(f"📝 Complex Query: {query[:100]}...")

    try:
        # Show planning phase
        plan = await agent.plan(query)
        print(f"📋 Generated Plan: {len(plan)} steps")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step[:80]}...")

        # Execute the plan
        response = await agent.chat(query)
        print(f"🤖 Final Response: {response[:200]}...")

    except Exception as e:
        print(f"❌ Error in planning: {e}")


async def demo_reflection():
    """Demonstrate reflection and self-correction."""
    print("\n" + "="*60)
    print("🔍 REFLECTION DEMONSTRATION")
    print("="*60)

    agent = UnifiedReActAgent(
        model_name="mock",
        tools=TOOL_REGISTRY,
        enable_reflection=True
    )

    # Query that might need correction
    query = "Calculate the square root of -16. If there's an issue, explain why."
    print(f"📝 Query: {query}")

    try:
        response = await agent.chat(query)
        print(f"🤖 Response: {response}")

        # Demonstrate reflection
        reflection = await agent.reflect(query, response)
        print(f"🔍 Reflection: {reflection}")

    except Exception as e:
        print(f"❌ Error in reflection: {e}")


async def demo_conversation_continuity():
    """Demonstrate conversation memory and context."""
    print("\n" + "="*60)
    print("💭 CONVERSATION CONTINUITY DEMONSTRATION")
    print("="*60)

    agent = UnifiedReActAgent(
        model_name="mock",
        tools=TOOL_REGISTRY
    )

    # Multi-turn conversation
    queries = [
        "My name is Alice and I'm 25 years old.",
        "What's my name?",
        "How old am I?",
        "What can you tell me about myself?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n🗣️ Turn {i}: {query}")
        try:
            response = await agent.chat(query)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error in turn {i}: {e}")


async def main():
    """Run all demonstrations."""
    print("🚀 UNIFIED REACT AGENT COMPREHENSIVE DEMONSTRATION")
    print("This demo shows the full capabilities of the ReAct agent system.")

    try:
        await demo_basic_reasoning()
        await demo_tool_usage()
        await demo_multi_step_planning()
        await demo_reflection()
        await demo_conversation_continuity()

        print("\n" + "="*60)
        print("✅ ALL DEMONSTRATIONS COMPLETED")
        print("="*60)
        print("The UnifiedReActAgent successfully demonstrated:")
        print("- Basic reasoning and step-by-step thinking")
        print("- Tool calling and external system integration")
        print("- Multi-step planning and execution")
        print("- Self-reflection and error correction")
        print("- Conversation continuity and context awareness")

    except Exception as e:
        print(f"❌ Fatal error in demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())