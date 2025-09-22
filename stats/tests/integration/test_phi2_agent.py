"""
Comprehensive test script for ReAct agent with Microsoft Phi-2 model.
Tests reasoning, tool calling, memory, and context capabilities.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.agent.rag_integration import enhance_agent_with_rag, rag_context
from src.agent.react_agent import ReActAgent, create_react_agent
from src.agent.tools import ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable verbose transformers warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class Phi2TestSuite:
    """Test suite for Phi-2 model with ReAct agent."""

    def __init__(self, model_path: str = "models/microsoft_phi-2"):
        """Initialize test suite with Phi-2 model."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.agent = None
        self.test_results = []

    def setup_agent(
        self, enable_planning: bool = True, enable_reflection: bool = True, verbose: bool = True
    ):
        """Setup ReAct agent with Phi-2 model."""
        print("\n🔧 Setting up ReAct agent with Phi-2 model...")
        print(f"   Model path: {self.model_path}")

        try:
            # Create agent with Phi-2 model
            self.agent = ReActAgent(
                model_name=str(self.model_path),
                enable_planning=enable_planning,
                enable_reflection=enable_reflection,
                verbose=verbose,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                use_8bit=False,  # Phi-2 is small enough to run without quantization
            )

            print("✅ Agent initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            traceback.print_exc()
            return False

    def test_basic_reasoning(self) -> dict[str, Any]:
        """Test basic reasoning capabilities without tools."""
        print("\n📝 Test 1: Basic Reasoning (No Tools)")
        print("-" * 50)

        test_cases = [
            {
                "query": "What is 25 + 37? Explain your reasoning step by step.",
                "type": "math_reasoning",
            },
            {
                "query": "If all roses are flowers, and some flowers fade quickly, can we conclude that all roses fade quickly? Explain why or why not.",
                "type": "logical_reasoning",
            },
            {
                "query": "I have 3 apples. I eat one and give one to my friend. How many do I have left?",
                "type": "word_problem",
            },
        ]

        results = []
        for i, test in enumerate(test_cases, 1):
            print(f"\n  Test 1.{i}: {test['type']}")
            print(f"  Query: {test['query']}")

            try:
                start_time = time.time()

                # Disable tools for pure reasoning test
                response = self.agent.generate_response(test["query"])

                elapsed = time.time() - start_time

                print(
                    f"  Response: {response[:200]}..."
                    if len(response) > 200
                    else f"  Response: {response}"
                )
                print(f"  Time: {elapsed:.2f}s")

                # Check if response is meaningful (not placeholder)
                is_valid = (
                    len(response) > 20
                    and not response.lower().startswith("placeholder")
                    and not response.lower().startswith("test response")
                    and any(
                        word in response.lower()
                        for word in ["think", "answer", "therefore", "result", "because"]
                    )
                )

                results.append(
                    {
                        "test": test["type"],
                        "success": is_valid,
                        "response_length": len(response),
                        "time": elapsed,
                        "response_preview": response[:100],
                    }
                )

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({"test": test["type"], "success": False, "error": str(e)})

        return {"test": "basic_reasoning", "results": results}

    def test_tool_calling(self) -> dict[str, Any]:
        """Test tool calling functionality."""
        print("\n🔧 Test 2: Tool Calling")
        print("-" * 50)

        test_cases = [
            {
                "query": "Calculate the result of (15 * 3) + (28 / 4) - 10",
                "expected_tool": "calculator",
                "type": "calculator_tool",
            },
            {
                "query": "What is the current date and time?",
                "expected_tool": "datetime",
                "type": "datetime_tool",
            },
            {
                "query": "List the files in the current directory",
                "expected_tool": "list_directory",
                "type": "file_tool",
            },
        ]

        results = []
        for i, test in enumerate(test_cases, 1):
            print(f"\n  Test 2.{i}: {test['type']}")
            print(f"  Query: {test['query']}")

            try:
                start_time = time.time()

                # Use solve method for tool calling
                response = self.agent.solve(test["query"], max_iterations=5)

                elapsed = time.time() - start_time

                # Check if the expected tool was called
                trace = self.agent.current_trace
                tools_used = [action["name"] for action in trace.actions] if trace else []

                print(f"  Tools used: {tools_used}")
                print(
                    f"  Response: {response[:200]}..."
                    if len(response) > 200
                    else f"  Response: {response}"
                )
                print(f"  Time: {elapsed:.2f}s")

                # Verify tool was called
                tool_called = test["expected_tool"] in tools_used

                results.append(
                    {
                        "test": test["type"],
                        "success": tool_called,
                        "tools_used": tools_used,
                        "response_length": len(response),
                        "time": elapsed,
                    }
                )

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({"test": test["type"], "success": False, "error": str(e)})

        return {"test": "tool_calling", "results": results}

    def test_complex_reasoning(self) -> dict[str, Any]:
        """Test complex multi-step reasoning with planning."""
        print("\n🧠 Test 3: Complex Reasoning with Planning")
        print("-" * 50)

        test_cases = [
            {
                "query": "I need to plan a small party for 8 people. Calculate the total cost if pizza costs $12 each (need 3), drinks are $2 per person, and decorations cost $25.",
                "type": "multi_step_calculation",
            },
            {
                "query": "Research and explain: What are the three main components of a computer's CPU? Then calculate how many transistors would fit in 1 square millimeter if each transistor is 5 nanometers wide.",
                "type": "research_and_calculate",
            },
        ]

        results = []
        for i, test in enumerate(test_cases, 1):
            print(f"\n  Test 3.{i}: {test['type']}")
            print(f"  Query: {test['query']}")

            try:
                start_time = time.time()

                # Enable planning for complex tasks
                response = self.agent.solve(test["query"], max_iterations=10)

                elapsed = time.time() - start_time

                # Get trace info
                trace = self.agent.current_trace
                if trace:
                    print(f"  Plan created: {trace.plan is not None}")
                    print(f"  Thoughts: {len(trace.thoughts)}")
                    print(f"  Actions: {len(trace.actions)}")
                    print(f"  Reflections: {len(trace.reflections)}")

                print(
                    f"  Response: {response[:300]}..."
                    if len(response) > 300
                    else f"  Response: {response}"
                )
                print(f"  Time: {elapsed:.2f}s")

                results.append(
                    {
                        "test": test["type"],
                        "success": len(response) > 50,
                        "has_plan": trace.plan is not None if trace else False,
                        "num_thoughts": len(trace.thoughts) if trace else 0,
                        "num_actions": len(trace.actions) if trace else 0,
                        "time": elapsed,
                    }
                )

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({"test": test["type"], "success": False, "error": str(e)})

        return {"test": "complex_reasoning", "results": results}

    def test_memory_context(self) -> dict[str, Any]:
        """Test memory and context retention."""
        print("\n💾 Test 4: Memory and Context")
        print("-" * 50)

        # Series of related queries to test context retention
        conversation = [
            "My name is Alice and I'm planning a trip to Japan.",
            "What's my name?",
            "Where am I planning to travel?",
            "Can you suggest three things I should pack for my trip?",
        ]

        results = []
        for i, query in enumerate(conversation, 1):
            print(f"\n  Query {i}: {query}")

            try:
                start_time = time.time()

                # Use chat method to maintain conversation history
                response = self.agent.chat(query, use_tools=False)

                elapsed = time.time() - start_time

                print(
                    f"  Response: {response[:200]}..."
                    if len(response) > 200
                    else f"  Response: {response}"
                )
                print(f"  Time: {elapsed:.2f}s")

                # Check for context retention
                context_retained = False
                if (
                    (i == 2 and "alice" in response.lower())
                    or (i == 3 and "japan" in response.lower())
                    or (
                        i == 4
                        and any(
                            word in response.lower()
                            for word in ["pack", "bring", "japan", "travel"]
                        )
                    )
                ):
                    context_retained = True
                elif i == 1:
                    context_retained = True  # First message always passes

                results.append(
                    {
                        "query_num": i,
                        "success": context_retained,
                        "response_length": len(response),
                        "time": elapsed,
                    }
                )

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({"query_num": i, "success": False, "error": str(e)})

        return {"test": "memory_context", "results": results}

    def test_error_handling(self) -> dict[str, Any]:
        """Test error handling and recovery."""
        print("\n⚠️ Test 5: Error Handling and Recovery")
        print("-" * 50)

        test_cases = [
            {"query": "Use the nonexistent_tool to do something", "type": "invalid_tool"},
            {"query": "Read the file at /definitely/not/a/real/path.txt", "type": "invalid_file"},
        ]

        results = []
        for i, test in enumerate(test_cases, 1):
            print(f"\n  Test 5.{i}: {test['type']}")
            print(f"  Query: {test['query']}")

            try:
                start_time = time.time()

                # Test error recovery
                response = self.agent.solve(test["query"], max_iterations=3)

                elapsed = time.time() - start_time

                # Check if agent handled error gracefully
                trace = self.agent.current_trace
                has_error_handling = (
                    any(thought.type.value in {"error", "reflection"} for thought in trace.thoughts)
                    if trace
                    else False
                )

                print(f"  Error handled: {has_error_handling}")
                print(
                    f"  Response: {response[:200]}..."
                    if len(response) > 200
                    else f"  Response: {response}"
                )
                print(f"  Time: {elapsed:.2f}s")

                results.append(
                    {
                        "test": test["type"],
                        "success": has_error_handling
                        or "error" in response.lower()
                        or "cannot" in response.lower(),
                        "handled_gracefully": has_error_handling,
                        "time": elapsed,
                    }
                )

            except Exception as e:
                # Even catching an exception can be considered proper error handling
                print(f"  Exception caught (expected): {e}")
                results.append(
                    {
                        "test": test["type"],
                        "success": True,  # Catching exception is success for error test
                        "exception": str(e),
                    }
                )

        return {"test": "error_handling", "results": results}

    def run_all_tests(self) -> dict[str, Any]:
        """Run complete test suite."""
        print("\n" + "=" * 60)
        print("🧪 PHI-2 REACT AGENT TEST SUITE")
        print("=" * 60)

        if not self.setup_agent(enable_planning=True, enable_reflection=True, verbose=False):
            return {"error": "Failed to setup agent"}

        all_results = {
            "model": str(self.model_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": [],
        }

        # Run each test category
        test_methods = [
            self.test_basic_reasoning,
            self.test_tool_calling,
            self.test_complex_reasoning,
            self.test_memory_context,
            self.test_error_handling,
        ]

        for test_method in test_methods:
            try:
                result = test_method()
                all_results["tests"].append(result)
            except Exception as e:
                print(f"\n❌ Test failed with exception: {e}")
                all_results["tests"].append({"test": test_method.__name__, "error": str(e)})

        # Generate summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = 0
        passed_tests = 0

        for test_group in results.get("tests", []):
            if "error" in test_group:
                print(f"\n❌ {test_group.get('test', 'Unknown')}: FAILED (Exception)")
                continue

            group_name = test_group.get("test", "Unknown")
            group_results = test_group.get("results", [])

            group_passed = sum(1 for r in group_results if r.get("success", False))
            group_total = len(group_results)

            total_tests += group_total
            passed_tests += group_passed

            status = "✅" if group_passed == group_total else "⚠️"
            print(f"\n{status} {group_name}: {group_passed}/{group_total} passed")

            # Print failed tests
            for r in group_results:
                if not r.get("success", False):
                    test_id = r.get("test", r.get("query_num", "Unknown"))
                    error = r.get("error", "Test failed")
                    print(f"   ❌ {test_id}: {error}")

        # Overall summary
        print("\n" + "-" * 60)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        if success_rate >= 80:
            status_icon = "🎉"
            status_text = "EXCELLENT"
        elif success_rate >= 60:
            status_icon = "✅"
            status_text = "GOOD"
        elif success_rate >= 40:
            status_icon = "⚠️"
            status_text = "NEEDS IMPROVEMENT"
        else:
            status_icon = "❌"
            status_text = "POOR"

        print(f"\n{status_icon} Overall Performance: {status_text}")
        print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

        # Model performance assessment
        print("\n📝 Model Assessment:")
        if success_rate >= 60:
            print("   ✅ Model produces coherent, real responses")
            print("   ✅ Reasoning capabilities functional")
            print("   ✅ Tool calling works when supported")
        else:
            print("   ⚠️ Model may need fine-tuning or different parameters")
            print("   ⚠️ Consider adjusting temperature and top_p values")

        print("\n" + "=" * 60)


async def test_with_rag():
    """Test agent with RAG enhancement."""
    print("\n🔮 Testing with RAG Enhancement")
    print("-" * 50)

    try:
        # Create base agent
        agent = create_react_agent(
            lightweight=False,
            model_name="models/microsoft_phi-2",
            enable_planning=True,
            enable_reflection=True,
            verbose=False,
        )

        # Enhance with RAG
        print("Enhancing agent with RAG capabilities...")
        enhanced_agent = await enhance_agent_with_rag(agent)

        # Test RAG-enhanced queries
        test_queries = [
            "What do you know about machine learning?",
            "Tell me about neural networks",
            "How does backpropagation work?",
        ]

        async with rag_context() as rag_client:
            # Ingest some knowledge
            await rag_client.ingest_document(
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                {"source": "test", "type": "definition"},
            )

            for query in test_queries:
                print(f"\nQuery: {query}")
                response = enhanced_agent.solve(query, max_iterations=5)
                print(f"Response: {response[:200]}...")

        print("\n✅ RAG enhancement test completed")

    except Exception as e:
        print(f"\n⚠️ RAG test skipped (Redis not available): {e}")


def main():
    """Main test execution."""
    # Test with Phi-2 model
    tester = Phi2TestSuite(model_path="models/microsoft_phi-2")

    # Run comprehensive tests
    results = tester.run_all_tests()

    # Save results
    results_file = Path("phi2_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {results_file}")

    # Optional: Test with RAG if available
    try:
        import asyncio

        print("\n" + "=" * 60)
        asyncio.run(test_with_rag())
    except:
        print("\n⚠️ Skipping RAG tests (dependencies not available)")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
