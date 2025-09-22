#!/usr/bin/env python3
"""Test script for native Gemma integration."""

import sys
import traceback
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.gemma_agent import UnifiedGemmaAgent, AgentMode
from src.agent.gemma_bridge import create_native_bridge


def test_native_bridge_direct():
    """Test the native bridge directly."""
    print("🧪 Testing native bridge directly...")
    
    try:
        # Create and test the bridge
        bridge = create_native_bridge(verbose=True)
        
        # Test loading
        print("📥 Loading native model...")
        if not bridge.load_native_model():
            raise RuntimeError("Failed to load native model")
        
        # Test generation
        print("🎯 Testing text generation...")
        test_prompt = "What is the capital of France?"
        response = bridge.generate_text(test_prompt, max_tokens=50)
        
        print(f"✅ Native bridge test successful!")
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response[:100]}...")
        
        # Test token counting
        token_count = bridge.count_tokens(test_prompt)
        print(f"   Token count: {token_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Native bridge test failed: {e}")
        traceback.print_exc()
        return False


def test_unified_agent_native():
    """Test the UnifiedGemmaAgent with native mode."""
    print("\n🧪 Testing UnifiedGemmaAgent with NATIVE mode...")
    
    try:
        # Create agent in native mode
        agent = UnifiedGemmaAgent(
            mode=AgentMode.NATIVE,
            verbose=True,
            max_new_tokens=50,
            temperature=0.7,
        )
        
        # Test generation
        test_prompt = "Explain what artificial intelligence is in one sentence."
        print(f"🎯 Testing with prompt: {test_prompt}")
        
        response = agent.generate_response(test_prompt)
        
        print(f"✅ Unified agent test successful!")
        print(f"   Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified agent test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting native Gemma integration tests...")
    print("=" * 60)
    
    # Test 1: Direct bridge test
    bridge_success = test_native_bridge_direct()
    
    # Test 2: Unified agent test
    agent_success = test_unified_agent_native()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Direct bridge test: {'✅ PASS' if bridge_success else '❌ FAIL'}")
    print(f"   Unified agent test: {'✅ PASS' if agent_success else '❌ FAIL'}")
    
    if bridge_success and agent_success:
        print("\n🎉 All tests passed! Native integration is working.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())