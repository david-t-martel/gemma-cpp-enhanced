#!/usr/bin/env python3
"""
Test 4B model to check if tensor issue is specific to 2B model
"""

import subprocess
import sys
from pathlib import Path

def windows_to_wsl_path(windows_path):
    """Convert Windows path to WSL path format"""
    if isinstance(windows_path, str):
        windows_path = Path(windows_path)

    abs_path = windows_path.resolve()
    parts = abs_path.parts
    if len(parts) > 0 and ':' in parts[0]:
        drive = parts[0].replace(':', '').lower()
        remaining_parts = parts[1:]
        wsl_path = f"/mnt/{drive}/" + "/".join(remaining_parts)
        return wsl_path

    return str(abs_path).replace('\\', '/')

def test_4b_model():
    """Test 4B model loading"""
    print("Testing 4B Model Loading")
    print("========================")

    # Define paths
    models_dir = Path("C:/codedev/llm/.models")
    build_wsl_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build_wsl")

    # 4B Model files
    model_4b = {
        "weights": models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs",
        "tokenizer": models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm"
    }

    gemma_exe = build_wsl_dir / "gemma"

    # Check if files exist
    if not model_4b["weights"].exists():
        print(f"[ERROR] 4B Model weights not found: {model_4b['weights']}")
        return False

    if not model_4b["tokenizer"].exists():
        print(f"[ERROR] 4B Tokenizer not found: {model_4b['tokenizer']}")
        return False

    print("[OK] 4B model files found")

    # Convert paths to WSL format
    wsl_gemma = windows_to_wsl_path(gemma_exe)
    wsl_weights = windows_to_wsl_path(model_4b["weights"])
    wsl_tokenizer = windows_to_wsl_path(model_4b["tokenizer"])

    print(f"WSL weights: {wsl_weights}")
    print(f"WSL tokenizer: {wsl_tokenizer}")

    # Test model loading
    try:
        cmd = [
            'wsl', wsl_gemma,
            '--tokenizer', wsl_tokenizer,
            '--weights', wsl_weights,
            '--prompt', 'Hi'
        ]

        print(f"Command: {' '.join(cmd)}")
        print("Running 4B model loading test...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr length: {len(result.stderr)}")

        if result.stdout:
            print("Stdout preview:")
            print(result.stdout[:1000])
        if result.stderr:
            print("Stderr preview:")
            print(result.stderr[:1000])

        if result.returncode == 0:
            print("[SUCCESS] 4B model loading test SUCCESSFUL!")
            return True
        else:
            print(f"[FAILED] 4B model loading test FAILED with code {result.returncode}")

            # Check for specific error patterns
            output_text = result.stdout + result.stderr
            if "tensor" in output_text.lower() and "required" in output_text.lower():
                print("[ANALYSIS] Detected: Missing tensor error (same as 2B)")
            elif "post_att_ns_0" in output_text:
                print("[ANALYSIS] Detected: Same post_att_ns_0 tensor missing")
            elif "abort" in output_text.lower():
                print("[ANALYSIS] Detected: Abort/crash during loading")

            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] 4B model loading test TIMED OUT (120s)")
        return False
    except Exception as e:
        print(f"[ERROR] 4B model loading test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_4b_model()
    sys.exit(0 if success else 1)