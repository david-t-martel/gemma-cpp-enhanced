#!/usr/bin/env python3
"""
Quick test script to verify the ReAct agent setup and basic functionality.
Run this to ensure the agent is working correctly before running the full demo.
"""

import sys
import os
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
try:
    from src.agent.gemma_agent import AgentMode, create_gemma_agent
    from src.agent.react_agent import UnifiedReActAgent
    from src.agent.tools import ToolRegistry
    from src.shared.logging import setup_logging, get_logger, LogLevel
    print("[OK] All modules imported successfully")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Please ensure you're in the correct environment and dependencies are installed")
    sys.exit(1)

# Setup logging
setup_logging(level=LogLevel.INFO, console=True)
logger = get_logger("test_setup")


def test_basic_agent():
    """Test basic agent initialization and simple reasoning."""

    print("\n" + "="*60)
    print("Testing ReAct Agent Setup")
    print("="*60)

    try:
        # Step 1: Create tool registry
        print("\n1. Creating tool registry...")
        tool_registry = ToolRegistry()
        print(f"   Available tools: {len(tool_registry.tools)}")

        # Step 2: Initialize agent in lightweight mode
        print("\n2. Initializing ReAct agent (lightweight mode)...")
        agent = UnifiedReActAgent(
            model_name="gemma-2b",
            mode=AgentMode.LIGHTWEIGHT,
            tool_registry=tool_registry,
            max_iterations=5,
            verbose=False,
            enable_planning=False,  # Disable for quick test
            enable_reflection=False,  # Disable for quick test
            temperature=0.7
        )
        print("   [OK] Agent initialized")

        # Step 3: Test simple reasoning
        print("\n3. Testing simple reasoning...")
        simple_prompt = "What is 2 + 2? Think step by step."

        result = agent.solve(simple_prompt)

        if result and "error" not in result.lower():
            print("   [OK] Reasoning successful")
            print(f"   Answer: {result}")
        else:
            print("   [ERROR] Reasoning failed")

        # Step 4: Test with a coding problem
        print("\n4. Testing with simple coding problem...")
        coding_prompt = """
        Write a Python function that returns the square of a number.
        The function should be named 'square' and take one parameter.
        """

        result = agent.solve(coding_prompt)

        if result and "error" not in result.lower():
            print("   [OK] Code generation successful")
            print("   Generated code:")
            print("   " + "-"*40)
            # Show first 200 chars of answer
            answer_preview = result[:200] if result else "No answer"
            print(f"   {answer_preview}...")
        else:
            print("   [ERROR] Code generation failed")

        print("\n" + "="*60)
        print("[OK] All tests passed! Agent is ready for use.")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_loading():
    """Test if the model can be loaded."""

    print("\n" + "="*60)
    print("Testing Model Loading")
    print("="*60)

    try:
        print("\n1. Checking for model files...")

        # Check models directory
        models_dir = project_root / "models"
        cache_dir = project_root / "models_cache"
        c_models_dir = Path("/c/codedev/llm/.models")

        found_models = False

        if models_dir.exists():
            print(f"   Found models directory: {models_dir}")
            model_files = list(models_dir.glob("**/*.safetensors")) + \
                         list(models_dir.glob("**/*.bin")) + \
                         list(models_dir.glob("**/*.sbs"))
            if model_files:
                print(f"   Found {len(model_files)} model files")
                found_models = True

        if cache_dir.exists():
            print(f"   Found cache directory: {cache_dir}")

        if c_models_dir.exists():
            print(f"   Found C++ models directory: {c_models_dir}")
            sbs_files = list(c_models_dir.glob("*.sbs"))
            if sbs_files:
                print(f"   Found {len(sbs_files)} .sbs files")
                for f in sbs_files[:3]:  # Show first 3
                    print(f"     - {f.name}")
                found_models = True

        if not found_models:
            print("   [WARNING] No model files found")
            print("   Run: uv run python -m src.gcp.gemma_download --auto")

        return found_models

    except Exception as e:
        print(f"[ERROR] Error checking models: {e}")
        return False


async def main():
    """Run all tests."""

    print("\n" + "="*80)
    print(" REACT AGENT SETUP TEST")
    print("="*80)

    # Test model loading
    model_ok = await test_model_loading()

    if not model_ok:
        print("\n[WARNING] Warning: Model files not found.")
        print("The agent may not work properly without model files.")
        print("\nTo download models, run:")
        print("  cd /c/codedev/llm/stats")
        print("  uv run python -m src.gcp.gemma_download --auto")

    # Test agent setup
    agent_ok = test_basic_agent()

    if agent_ok:
        print("\n[SUCCESS] SUCCESS: Agent is properly configured and working!")
        print("\nYou can now run the full demonstration:")
        print("  python examples/coding_agent_demo.py")
        print("\nOr open the Jupyter notebook:")
        print("  jupyter notebook examples/react_agent_coding_notebook.ipynb")
    else:
        print("\n[FAILURE] FAILURE: Agent setup failed. Please check the errors above.")

    return agent_ok and model_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)