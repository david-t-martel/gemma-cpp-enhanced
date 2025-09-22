#!/usr/bin/env python3
"""
Test WSL model loading with corrected path conversion
"""

import subprocess
import sys
import os
from pathlib import Path, PureWindowsPath, PurePosixPath

def windows_to_wsl_path(windows_path):
    """Convert Windows path to WSL path format"""
    # Convert to Path object if string
    if isinstance(windows_path, str):
        windows_path = Path(windows_path)

    # Get the absolute path
    abs_path = windows_path.resolve()

    # Convert C:\ to /mnt/c/ format
    parts = abs_path.parts
    if len(parts) > 0 and ':' in parts[0]:
        drive = parts[0].replace(':', '').lower()
        remaining_parts = parts[1:]
        wsl_path = f"/mnt/{drive}/" + "/".join(remaining_parts)
        return wsl_path

    return str(abs_path).replace('\\', '/')

def test_wsl_gemma():
    """Test WSL gemma executable with proper path conversion"""

    # Define paths
    models_dir = Path("C:/codedev/llm/.models")
    build_wsl_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build_wsl")

    # Model files
    model_2b = {
        "weights": models_dir / "gemma-gemmacpp-2b-it-v3/2b-it.sbs",
        "tokenizer": models_dir / "gemma-gemmacpp-2b-it-v3/tokenizer.spm"
    }

    gemma_exe = build_wsl_dir / "gemma"

    # Check if files exist
    if not gemma_exe.exists():
        print(f"[ERROR] WSL gemma executable not found: {gemma_exe}")
        return False

    if not model_2b["weights"].exists():
        print(f"[ERROR] Model weights not found: {model_2b['weights']}")
        return False

    if not model_2b["tokenizer"].exists():
        print(f"[ERROR] Tokenizer not found: {model_2b['tokenizer']}")
        return False

    print("[OK] All files found, testing WSL path conversion...")

    # Convert paths to WSL format
    wsl_gemma = windows_to_wsl_path(gemma_exe)
    wsl_weights = windows_to_wsl_path(model_2b["weights"])
    wsl_tokenizer = windows_to_wsl_path(model_2b["tokenizer"])

    print(f"Windows gemma: {gemma_exe}")
    print(f"WSL gemma: {wsl_gemma}")
    print(f"Windows weights: {model_2b['weights']}")
    print(f"WSL weights: {wsl_weights}")
    print(f"Windows tokenizer: {model_2b['tokenizer']}")
    print(f"WSL tokenizer: {wsl_tokenizer}")

    # Test 1: Check if WSL executable exists and runs
    print("\n=== Test 1: WSL Executable Check ===")
    try:
        cmd = ['wsl', 'ls', '-la', wsl_gemma]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")

        if result.returncode != 0:
            print("[ERROR] WSL gemma executable not accessible")
            return False
        else:
            print("[OK] WSL gemma executable is accessible")
    except Exception as e:
        print(f"[ERROR] Failed to check WSL executable: {e}")
        return False

    # Test 2: Help command
    print("\n=== Test 2: Help Command ===")
    try:
        cmd = ['wsl', wsl_gemma, '--help']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr length: {len(result.stderr)}")

        if result.stdout:
            print("Output preview:")
            print(result.stdout[:500])
        if result.stderr:
            print("Error preview:")
            print(result.stderr[:500])

    except subprocess.TimeoutExpired:
        print("[ERROR] Help command timed out")
    except Exception as e:
        print(f"[ERROR] Help command failed: {e}")

    # Test 3: Model loading test with minimal prompt
    print("\n=== Test 3: Model Loading Test ===")
    try:
        cmd = [
            'wsl', wsl_gemma,
            '--tokenizer', wsl_tokenizer,
            '--weights', wsl_weights,
            '--prompt', 'Hi'
        ]

        print(f"Command: {' '.join(cmd)}")
        print("Running model loading test...")

        # Run with longer timeout for model loading
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
            print("[SUCCESS] Model loading test SUCCESSFUL!")
            return True
        else:
            print(f"[FAILED] Model loading test FAILED with code {result.returncode}")

            # Check for specific error patterns
            output_text = result.stdout + result.stderr
            if "segmentation fault" in output_text.lower():
                print("[ANALYSIS] Detected: Segmentation fault")
            elif "heap corruption" in output_text.lower():
                print("[ANALYSIS] Detected: Heap corruption")
            elif "access violation" in output_text.lower():
                print("[ANALYSIS] Detected: Access violation")
            elif "command not found" in output_text.lower():
                print("[ANALYSIS] Detected: Command not found (path issue)")
            elif "no such file" in output_text.lower():
                print("[ANALYSIS] Detected: File not found")

            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Model loading test TIMED OUT (120s)")
        return False
    except Exception as e:
        print(f"[ERROR] Model loading test FAILED: {e}")
        return False

def test_windows_event_viewer():
    """Check Windows Event Viewer for recent crashes"""
    print("\n=== Windows Event Viewer Check ===")

    try:
        # PowerShell command to get recent application crashes
        ps_cmd = """
Get-WinEvent -FilterHashtable @{LogName='Application'; Level=2; StartTime=(Get-Date).AddDays(-1)} -MaxEvents 20 |
Where-Object {$_.Message -like '*gemma*' -or $_.Message -like '*3221226356*' -or $_.Message -like '*0xC0000374*'} |
Format-Table TimeCreated, Id, LevelDisplayName, Message -Wrap
"""

        result = subprocess.run(
            ['powershell', '-Command', ps_cmd],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            if result.stdout.strip():
                print("Recent gemma-related crashes found:")
                print(result.stdout)
            else:
                print("No recent gemma-related crashes found in Event Viewer")
        else:
            print(f"Failed to query Event Viewer: {result.stderr}")

    except Exception as e:
        print(f"Error checking Event Viewer: {e}")

if __name__ == "__main__":
    print("Gemma WSL Model Loading Test")
    print("============================")

    success = test_wsl_gemma()
    test_windows_event_viewer()

    if success:
        print("\n[FINAL] SUCCESS: Model loading works with WSL!")
        sys.exit(0)
    else:
        print("\n[FINAL] FAILURE: Model loading still has issues")
        sys.exit(1)