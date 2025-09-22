#!/usr/bin/env python3
"""
Quick Start Script for Gemma CLI Wrapper
Automatically detects available models and launches the CLI
"""

import os
import sys
import subprocess
from pathlib import Path


def find_available_models():
    """Find available Gemma model files"""
    models_dir = Path("C:/codedev/llm/.models")
    available_models = []

    if models_dir.exists():
        # Look for .sbs files
        for sbs_file in models_dir.rglob("*.sbs"):
            model_info = {
                "model_path": str(sbs_file),
                "name": sbs_file.stem,
                "size_mb": sbs_file.stat().st_size / (1024 * 1024)
            }

            # Look for corresponding tokenizer
            tokenizer_file = sbs_file.parent / "tokenizer.spm"
            if tokenizer_file.exists():
                model_info["tokenizer_path"] = str(tokenizer_file)

            available_models.append(model_info)

    return available_models


def check_prerequisites():
    """Check if prerequisites are met"""
    print("Checking prerequisites...")

    # Check Python
    python_version = sys.version_info
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check WSL
    try:
        result = subprocess.run(["wsl", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ WSL available")
        else:
            print("⚠ WSL not responding properly")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠ WSL not found - install with: wsl --install")

    # Check colorama
    try:
        import colorama
        print("✓ Colorama available (colored output enabled)")
    except ImportError:
        print("⚠ Colorama not installed (optional) - install with: pip install colorama")

    # Check gemma executable
    gemma_exe = Path("C:/codedev/llm/gemma/gemma.cpp/build_wsl/gemma")
    if gemma_exe.exists():
        print(f"✓ Gemma executable found: {gemma_exe}")
    else:
        print(f"⚠ Gemma executable not found at: {gemma_exe}")
        print("   Build it with: cd gemma.cpp && cmake --preset make && cmake --build --preset make")

    print()


def select_model(models):
    """Let user select a model"""
    if not models:
        print("No models found in C:/codedev/llm/.models/")
        print("Please download models from Kaggle and extract to the models directory.")
        return None

    print("Available models:")
    for i, model in enumerate(models, 1):
        size_info = f"({model['size_mb']:.0f} MB)"
        tokenizer_info = "with tokenizer" if "tokenizer_path" in model else "no tokenizer"
        print(f"  {i}. {model['name']} {size_info} - {tokenizer_info}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None

            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")


def launch_cli(model_info):
    """Launch the CLI with selected model"""
    script_dir = Path(__file__).parent
    cli_script = script_dir / "gemma-cli.py"

    if not cli_script.exists():
        print(f"Error: gemma-cli.py not found at {cli_script}")
        return False

    # Build command
    cmd = [sys.executable, str(cli_script), "--model", model_info["model_path"]]

    if "tokenizer_path" in model_info:
        cmd.extend(["--tokenizer", model_info["tokenizer_path"]])

    print(f"\nLaunching Gemma CLI with model: {model_info['name']}")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to interrupt, type /quit to exit normally")
    print("=" * 60)

    try:
        # Launch the CLI
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nQuick start interrupted by user.")
    except Exception as e:
        print(f"Error launching CLI: {e}")
        return False

    return True


def main():
    """Main function"""
    print("Gemma CLI Wrapper - Quick Start")
    print("=" * 40)

    # Check prerequisites
    check_prerequisites()

    # Find available models
    print("Scanning for available models...")
    models = find_available_models()

    if not models:
        print("\nNo models found. Please:")
        print("1. Download Gemma models from Kaggle")
        print("2. Extract to C:/codedev/llm/.models/")
        print("3. Run this script again")
        return 1

    # Let user select model
    selected_model = select_model(models)
    if not selected_model:
        print("No model selected. Exiting.")
        return 0

    # Launch CLI
    success = launch_cli(selected_model)
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)