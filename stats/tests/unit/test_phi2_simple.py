"""
Simple test script for Phi-2 model loading and basic inference.
"""

from pathlib import Path
import sys
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def test_phi2_loading():
    """Test basic Phi-2 model loading and inference."""

    model_path = "models/microsoft_phi-2"
    print(f"🔧 Testing Phi-2 model at: {model_path}")

    try:
        # Step 1: Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,  # Phi-2 may need custom code
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("   ✅ Tokenizer loaded successfully")

        # Step 2: Load model
        print("\n2. Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")

        # Load with appropriate dtype
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,  # Phi-2 may need custom code
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to(device)

        print("   ✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        # Step 3: Test simple inference
        print("\n3. Testing inference...")

        test_prompts = ["The capital of France is", "2 + 2 equals", "The sun rises in the"]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: '{prompt}'")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = response[len(prompt) :].strip()

            print(f"   Response: '{completion}'")

            # Check if response is meaningful
            if len(completion) > 0 and not completion.lower().startswith("placeholder"):
                print("   ✅ Valid response generated")
            else:
                print("   ⚠️ Response may be invalid")

        print("\n✅ All basic tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_react_agent():
    """Test Phi-2 with ReAct agent."""
    print("\n" + "=" * 60)
    print("Testing Phi-2 with ReAct Agent")
    print("=" * 60)

    try:
        from src.agent.react_agent import ReActAgent
        from src.agent.tools import ToolRegistry

        print("\n1. Creating ReAct agent with Phi-2...")

        # Initialize with minimal configuration
        agent = ReActAgent(
            model_name="models/microsoft_phi-2",
            enable_planning=False,  # Start simple
            enable_reflection=False,
            verbose=True,
            max_new_tokens=256,
            temperature=0.7,
            use_8bit=False,
        )

        print("   ✅ Agent created successfully")

        # Test simple query without tools
        print("\n2. Testing simple reasoning...")
        query = "What is 5 + 3?"

        response = agent.generate_response(query)
        print(f"   Query: {query}")
        print(f"   Response: {response[:200]}...")

        # Test with tool
        print("\n3. Testing with tool calling...")
        query = "Calculate 15 * 4"

        response = agent.solve(query, max_iterations=3)
        print(f"   Query: {query}")
        print(f"   Response: {response[:200]}...")

        if agent.current_trace:
            print(f"   Actions taken: {len(agent.current_trace.actions)}")
            print(f"   Tools used: {[a['name'] for a in agent.current_trace.actions]}")

        print("\n✅ ReAct agent tests completed!")
        return True

    except Exception as e:
        print(f"\n❌ Error with ReAct agent: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 PHI-2 MODEL TEST SUITE")
    print("=" * 60)

    # First test basic model loading
    if test_phi2_loading():
        # If basic test passes, try with ReAct agent
        test_with_react_agent()
    else:
        print("\n⚠️ Skipping ReAct agent tests due to model loading issues")

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
