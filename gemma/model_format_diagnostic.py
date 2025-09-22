#!/usr/bin/env python3
"""
Model Format Diagnostic Tool
Analyze and fix model format compatibility issues
"""

import subprocess
import sys
import struct
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

def analyze_sbs_file(sbs_path):
    """Analyze .sbs file structure to understand what tensors are included"""
    print(f"\nAnalyzing SBS file: {sbs_path}")

    if not sbs_path.exists():
        print("[ERROR] File not found")
        return None

    try:
        with open(sbs_path, 'rb') as f:
            # Read file header
            header = f.read(16)
            print(f"Header (first 16 bytes): {header.hex()}")

            # Try to parse SBS format
            f.seek(0)
            magic = f.read(4)
            print(f"Magic bytes: {magic}")

            if magic == b'SBS\n':
                print("[OK] Valid SBS format detected")

                # Read more structure
                version_data = f.read(4)
                print(f"Version data: {version_data.hex()}")

                # The exact format parsing would require understanding the full SBS spec
                # For now, let's just check file size and basic structure
                file_size = sbs_path.stat().st_size
                print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")

                return {
                    "valid_sbs": True,
                    "file_size": file_size,
                    "header": header.hex()
                }
            else:
                print("[WARNING] Not a standard SBS file")
                return {
                    "valid_sbs": False,
                    "file_size": sbs_path.stat().st_size,
                    "header": header.hex()
                }

    except Exception as e:
        print(f"[ERROR] Failed to analyze file: {e}")
        return None

def test_model_with_debug(model_path, tokenizer_path, model_name):
    """Test model loading with debug output to see what tensors are missing"""
    print(f"\n=== Testing {model_name} Model with Debug ===")

    build_wsl_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build_wsl")
    gemma_exe = build_wsl_dir / "gemma"

    wsl_gemma = windows_to_wsl_path(gemma_exe)
    wsl_weights = windows_to_wsl_path(model_path)
    wsl_tokenizer = windows_to_wsl_path(tokenizer_path)

    try:
        # Try with debug_prompt executable if available
        debug_exe = build_wsl_dir / "debug_prompt"
        if debug_exe.exists():
            wsl_debug = windows_to_wsl_path(debug_exe)
            print("Using debug_prompt executable...")

            cmd = [
                'wsl', wsl_debug,
                '--tokenizer', wsl_tokenizer,
                '--weights', wsl_weights,
                '--prompt', 'Hi'
            ]
        else:
            print("Using regular gemma executable...")
            cmd = [
                'wsl', wsl_gemma,
                '--tokenizer', wsl_tokenizer,
                '--weights', wsl_weights,
                '--prompt', 'Hi'
            ]

        print(f"Command: {' '.join(cmd)}")

        # Run with shorter timeout to capture the error quickly
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }

    except subprocess.TimeoutExpired:
        print("[ERROR] Command timed out")
        return {"error": "timeout"}
    except Exception as e:
        print(f"[ERROR] Command failed: {e}")
        return {"error": str(e)}

def attempt_model_migration():
    """Try to migrate the 2B model to a newer format"""
    print("\n=== Attempting Model Migration ===")

    build_wsl_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build_wsl")
    migrate_exe = build_wsl_dir / "migrate_weights"

    if not migrate_exe.exists():
        print("[ERROR] migrate_weights executable not found")
        return False

    models_dir = Path("C:/codedev/llm/.models")
    old_model_dir = models_dir / "gemma-gemmacpp-2b-it-v3"

    old_weights = old_model_dir / "2b-it.sbs"
    old_tokenizer = old_model_dir / "tokenizer.spm"
    new_weights = old_model_dir / "2b-it-migrated.sbs"

    if not old_weights.exists() or not old_tokenizer.exists():
        print("[ERROR] Original 2B model files not found")
        return False

    # Convert paths for WSL
    wsl_migrate = windows_to_wsl_path(migrate_exe)
    wsl_old_tokenizer = windows_to_wsl_path(old_tokenizer)
    wsl_old_weights = windows_to_wsl_path(old_weights)
    wsl_new_weights = windows_to_wsl_path(new_weights)

    try:
        cmd = [
            'wsl', wsl_migrate,
            '--tokenizer', wsl_old_tokenizer,
            '--weights', wsl_old_weights,
            '--output_weights', wsl_new_weights
        ]

        print(f"Migration command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        print(f"Migration return code: {result.returncode}")
        print(f"Migration stdout: {result.stdout}")
        print(f"Migration stderr: {result.stderr}")

        if result.returncode == 0 and new_weights.exists():
            print("[SUCCESS] Model migration completed!")
            print(f"New model saved to: {new_weights}")

            # Test the migrated model
            print("\nTesting migrated model...")
            test_result = test_model_with_debug(new_weights, old_tokenizer, "Migrated 2B")

            if test_result.get("success"):
                print("[SUCCESS] Migrated model works!")
                return True
            else:
                print("[WARNING] Migrated model still has issues")
                return False
        else:
            print("[FAILED] Model migration failed")
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Migration timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("Gemma Model Format Diagnostic Tool")
    print("==================================")

    models_dir = Path("C:/codedev/llm/.models")

    # Analyze both models
    models_to_check = [
        {
            "name": "2B",
            "weights": models_dir / "gemma-gemmacpp-2b-it-v3/2b-it.sbs",
            "tokenizer": models_dir / "gemma-gemmacpp-2b-it-v3/tokenizer.spm"
        },
        {
            "name": "4B",
            "weights": models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs",
            "tokenizer": models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm"
        }
    ]

    results = {}

    for model in models_to_check:
        print(f"\n{'='*50}")
        print(f"Analyzing {model['name']} Model")
        print(f"{'='*50}")

        # Analyze file format
        sbs_analysis = analyze_sbs_file(model["weights"])

        # Test model loading
        test_result = test_model_with_debug(model["weights"], model["tokenizer"], model["name"])

        results[model["name"]] = {
            "sbs_analysis": sbs_analysis,
            "test_result": test_result
        }

    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*50}")

    for model_name, result in results.items():
        test_success = result["test_result"].get("success", False)
        print(f"{model_name} Model: {'[WORKING]' if test_success else '[FAILED]'}")

        if not test_success and "stderr" in result["test_result"]:
            stderr = result["test_result"]["stderr"]
            if "post_att_ns_0" in stderr:
                print(f"  Issue: Missing post_att_ns_0 tensor (format incompatibility)")
            elif "Abort" in stderr:
                print(f"  Issue: Model loading abort - {stderr.strip()}")

    # If 2B model failed, try migration
    if not results.get("2B", {}).get("test_result", {}).get("success", False):
        print(f"\n{'='*50}")
        print("ATTEMPTING 2B MODEL FIX")
        print(f"{'='*50}")

        migration_success = attempt_model_migration()

        if migration_success:
            print("\n[FINAL] SUCCESS: 2B model has been fixed via migration!")
            return 0
        else:
            print("\n[FINAL] 2B model migration failed.")
            print("RECOMMENDATION: Use the 4B model which works correctly, or")
            print("download a newer version of the 2B model from Kaggle.")
            return 1
    else:
        print("\n[FINAL] All models are working correctly!")
        return 0

if __name__ == "__main__":
    sys.exit(main())