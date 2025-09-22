"""
Diagnostic script to check Phi-2 model files and system resources.
"""

import json
from pathlib import Path
import traceback

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import safetensors.torch
except ImportError:
    safetensors = None


def check_system_resources():
    """Check available system resources."""
    print("🖥️ System Resources:")
    print("-" * 40)

    # CPU info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"CPU usage: {psutil.cpu_percent()}%")

    # Memory info
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"Used RAM: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

    # PyTorch info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")


def check_model_files():
    """Check Phi-2 model files integrity."""
    print("\n📁 Model Files Check:")
    print("-" * 40)

    model_path = Path("models/microsoft_phi-2")

    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False

    # Check required files
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]

    missing_files = []
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file}: {size_mb:.2f} MB")
        else:
            missing_files.append(file)
            print(f"❌ {file}: Missing")

    # Check config
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("\n📋 Model Config:")
        print(f"   Architecture: {config.get('architectures', ['Unknown'])[0]}")
        print(f"   Hidden size: {config.get('hidden_size', 'Unknown')}")
        print(f"   Layers: {config.get('num_hidden_layers', 'Unknown')}")
        print(
            f"   Parameters: ~{config.get('hidden_size', 0) * config.get('num_hidden_layers', 0) * 4 / 1e9:.1f}B (estimate)"
        )

    # Check index file
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"\n   Weight shards: {len(index.get('weight_map', {}))}")

    return len(missing_files) == 0


def test_minimal_loading():
    """Test minimal model loading with error handling."""
    print("\n🔬 Minimal Loading Test:")
    print("-" * 40)

    try:
        # Try loading just the config
        config = AutoConfig.from_pretrained("models/microsoft_phi-2", trust_remote_code=True)
        print("✅ Config loaded successfully")
        print(f"   Model type: {config.model_type}")

        # Try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/microsoft_phi-2", trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")

        # Test tokenization
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   Tokenization test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")

        # Try loading model with safetensors directly
        print("\n⏳ Attempting to load model weights...")
        print("   (This may take a moment...)")

        # Try with minimal settings
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "models/microsoft_phi-2",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Reduce memory usage
                local_files_only=True,  # Don't try to download
            )
            print("✅ Model loaded successfully!")
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   Total parameters: {param_count / 1e9:.2f}B")
            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")

            # Try alternative loading method
            print("\n🔄 Trying alternative loading method...")
            try:
                if safetensors is None:
                    print("❌ safetensors not available")
                    return False

                # Load weights manually
                weights_1 = safetensors.torch.load_file(
                    "models/microsoft_phi-2/model-00001-of-00002.safetensors"
                )
                weights_2 = safetensors.torch.load_file(
                    "models/microsoft_phi-2/model-00002-of-00002.safetensors"
                )
                print("✅ Loaded weight shards manually")
                print(f"   Shard 1: {len(weights_1)} tensors")
                print(f"   Shard 2: {len(weights_2)} tensors")

                # Check tensor sizes
                total_params = 0
                for tensor in {**weights_1, **weights_2}.values():
                    total_params += tensor.numel()

                print(f"   Total parameters: {total_params / 1e9:.2f}B")
                return True

            except Exception as e2:
                print(f"❌ Manual loading also failed: {e2}")
                return False

    except Exception as e:
        print(f"❌ Error during minimal test: {e}")
        traceback.print_exc()
        return False


def suggest_fixes():
    """Suggest potential fixes based on diagnostics."""
    print("\n💡 Suggestions:")
    print("-" * 40)

    mem = psutil.virtual_memory()
    if mem.available / (1024**3) < 8:
        print("⚠️ Low RAM available. Phi-2 needs ~5-8GB RAM to load.")
        print("   Try: Close other applications or use quantization")

    if not torch.cuda.is_available():
        print("i No GPU detected. Model will run on CPU (slower)")
        print("   Consider using lightweight mode or smaller batch sizes")

    print("\n📝 Alternative approaches to try:")
    print("1. Use 8-bit quantization: use_8bit=True")
    print("2. Use the lightweight Gemma-2B model instead")
    print("3. Load model with low_cpu_mem_usage=True")
    print("4. Clear cache and re-download model files")
    print("5. Use a cloud service with more resources")


def main():
    """Run all diagnostics."""
    print("🔍 PHI-2 MODEL DIAGNOSTICS")
    print("=" * 60)

    # Check system resources
    check_system_resources()

    # Check model files
    files_ok = check_model_files()

    if files_ok:
        # Try minimal loading
        test_minimal_loading()

    # Provide suggestions
    suggest_fixes()

    print("\n" + "=" * 60)
    print("✅ Diagnostics complete!")


if __name__ == "__main__":
    main()
