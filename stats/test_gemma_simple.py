#!/usr/bin/env python3
"""Simple test for Gemma integration components."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports."""
    try:
        from src.agent.gemma_agent import UnifiedGemmaAgent, AgentMode
        from src.agent.gemma_bridge import GemmaNativeBridge, create_native_bridge
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_bridge_creation():
    """Test bridge creation without loading model."""
    try:
        from src.agent.gemma_bridge import GemmaNativeBridge
        
        bridge = GemmaNativeBridge(
            gemma_exe_path=r"C:\codedev\llm\gemma\gemma.cpp\build-quick\Release\gemma.exe",
            model_path=r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs",
            tokenizer_path=r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm",
            verbose=True
        )
        print("✅ Bridge created successfully")
        
        # Test model info without loading
        info = bridge.get_model_info()
        print(f"📋 Model info: {info}")
        
        # Test token counting (uses heuristic)
        token_count = bridge.count_tokens("Hello world")
        print(f"🔢 Token count test: {token_count}")
        
        return True
    except Exception as e:
        print(f"❌ Bridge creation failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation with native mode."""
    try:
        from src.agent.gemma_agent import UnifiedGemmaAgent, AgentMode
        
        # Test with minimal configuration
        agent = UnifiedGemmaAgent(mode=AgentMode.NATIVE, verbose=False)
        print("✅ Native agent created")
        
        # Test basic functionality
        if hasattr(agent, 'native_bridge'):
            print("✅ Native bridge attribute exists")
            
            # Test text generation (will use placeholder)
            response = agent.generate_response("Test prompt")
            print(f"🤖 Generated response: {response[:100]}...")
            
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🧪 Simple Gemma Integration Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Bridge Creation", test_bridge_creation),
        ("Agent Creation", test_agent_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}:")
        try:
            result = test_func()
            results.append(result)
            print(f"{'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())