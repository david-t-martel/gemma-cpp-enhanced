#!/usr/bin/env python3
"""Test script for Gemma integration."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent.gemma_agent import UnifiedGemmaAgent, AgentMode


def test_native_mode():
    """Test native mode functionality."""
    print("🧪 Testing Native Mode")
    print("=" * 50)
    
    try:
        # Create native agent
        agent = UnifiedGemmaAgent(mode=AgentMode.NATIVE, verbose=True)
        print("✅ Native agent created successfully")
        
        # Test simple text generation
        prompt = "What is the capital of France?"
        print(f"\n📝 Testing with prompt: {prompt}")
        
        response = agent.generate_response(prompt)
        print(f"🤖 Response: {response}")
        
        # Test another prompt
        prompt2 = "Explain what Python is in one sentence."
        print(f"\n📝 Testing with prompt: {prompt2}")
        
        response2 = agent.generate_response(prompt2)
        print(f"🤖 Response: {response2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Native mode test failed: {e}")
        return False


def test_lightweight_mode():
    """Test lightweight mode with memory constraints."""
    print("\n🧪 Testing Lightweight Mode (Memory Constrained)")
    print("=" * 50)
    
    try:
        # This should fail due to memory issues we saw earlier
        agent = UnifiedGemmaAgent(mode=AgentMode.LIGHTWEIGHT, verbose=True)
        print("✅ Lightweight agent created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Lightweight mode test failed (expected): {e}")
        print("💡 This is expected due to memory constraints with the models")
        return False


def test_bridge_directly():
    """Test the native bridge directly."""
    print("\n🧪 Testing Native Bridge Directly")
    print("=" * 50)
    
    try:
        from src.agent.gemma_bridge import create_native_bridge
        
        # Create bridge
        bridge = create_native_bridge(model_type="2b-it", verbose=True)
        print("✅ Native bridge created successfully")
        
        # Test model loading
        if bridge.load_native_model():
            print("✅ Model loaded successfully")
        else:
            print("❌ Model loading failed")
            return False
        
        # Test text generation
        prompt = "Hello, world!"
        response = bridge.generate_text(prompt)
        print(f"🤖 Bridge response: {response}")
        
        # Test token counting
        token_count = bridge.count_tokens(prompt)
        print(f"🔢 Token count for '{prompt}': {token_count}")
        
        # Get model info
        info = bridge.get_model_info()
        print(f"ℹ️ Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Gemma Integration Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test native mode
    results['native_mode'] = test_native_mode()
    
    # Test bridge directly
    results['native_bridge'] = test_bridge_directly()
    
    # Test lightweight mode (expected to fail)
    results['lightweight_mode'] = test_lightweight_mode()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count >= 2:  # Native mode and bridge should work
        print("🎉 Integration test SUCCESSFUL!")
        return 0
    else:
        print("💥 Integration test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())