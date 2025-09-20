#!/usr/bin/env python3
"""
Gemma CLI Wrapper - Usage Examples
Demonstrates various ways to use the gemma-cli.py wrapper
"""

import os
import subprocess
import sys
from pathlib import Path


def example_basic_usage():
    """Show basic usage examples"""
    print("=" * 60)
    print("BASIC USAGE EXAMPLES")
    print("=" * 60)

    examples = [
        {
            "title": "Basic chat with 2B model",
            "command": [
                "python", "gemma-cli.py",
                "--model", r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs",
                "--tokenizer", r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm"
            ]
        },
        {
            "title": "Single-file model (no separate tokenizer)",
            "command": [
                "python", "gemma-cli.py",
                "--model", r"C:\codedev\llm\.models\gemma2-2b-it-sfp-single.sbs"
            ]
        },
        {
            "title": "High creativity (temperature 1.0)",
            "command": [
                "python", "gemma-cli.py",
                "--model", r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs",
                "--temperature", "1.0",
                "--max-tokens", "4096"
            ]
        },
        {
            "title": "Conservative/precise responses (temperature 0.1)",
            "command": [
                "python", "gemma-cli.py",
                "--model", r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs",
                "--temperature", "0.1",
                "--max-context", "16384"
            ]
        },
        {
            "title": "Debug mode for troubleshooting",
            "command": [
                "python", "gemma-cli.py",
                "--model", r"C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs",
                "--debug"
            ]
        }
    ]

    for example in examples:
        print(f"\n{example['title']}:")
        print(" ".join(example['command']))
        print()


def example_interactive_commands():
    """Show interactive command examples"""
    print("=" * 60)
    print("INTERACTIVE COMMAND EXAMPLES")
    print("=" * 60)

    commands = [
        {
            "command": "/help",
            "description": "Show all available commands and usage tips"
        },
        {
            "command": "/clear",
            "description": "Clear conversation history and start fresh"
        },
        {
            "command": "/save my_conversation.json",
            "description": "Save current conversation to a JSON file"
        },
        {
            "command": "/load previous_chat.json",
            "description": "Load a previously saved conversation"
        },
        {
            "command": "/list",
            "description": "List all saved conversations with timestamps"
        },
        {
            "command": "/status",
            "description": "Show session info (message count, duration, etc.)"
        },
        {
            "command": "/settings",
            "description": "Display current model and generation settings"
        },
        {
            "command": "/quit or /exit",
            "description": "Exit the application gracefully"
        }
    ]

    print("Once the CLI is running, you can use these commands:")
    print()
    for cmd in commands:
        print(f"  {cmd['command']:<25} - {cmd['description']}")
    print()


def example_conversation_flow():
    """Show example conversation flow"""
    print("=" * 60)
    print("EXAMPLE CONVERSATION FLOW")
    print("=" * 60)

    conversation = [
        ("You", "Hello! Can you help me understand quantum computing?"),
        ("Assistant", "Hello! I'd be happy to help explain quantum computing. Quantum computing is a revolutionary approach to computation that leverages quantum mechanical phenomena..."),
        ("You", "/save quantum_discussion.json"),
        ("System", "Conversation saved to: quantum_discussion.json"),
        ("You", "Can you give me a simple analogy?"),
        ("Assistant", "Certainly! Think of classical computers like a coin that can only be heads or tails (0 or 1). Quantum computers are like a spinning coin that can be both heads AND tails at the same time..."),
        ("You", "/status"),
        ("System", "Session Status:\n  Messages in conversation: 4\n  Session duration: 0:05:23\n  Session started: 2024-09-16 10:30:15"),
        ("You", "Thank you! /quit"),
        ("System", "Goodbye!")
    ]

    print("Example conversation:")
    print()
    for speaker, message in conversation:
        if speaker == "System":
            print(f"[{speaker}] {message}")
        else:
            print(f"{speaker}: {message}")
        print()


def example_batch_scripts():
    """Show example batch scripts for automation"""
    print("=" * 60)
    print("BATCH SCRIPT EXAMPLES")
    print("=" * 60)

    print("Windows Batch Script (gemma-quick.bat):")
    print("""
@echo off
REM Quick launch script for Gemma CLI
python gemma-cli.py ^
  --model "C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\2b-it.sbs" ^
  --tokenizer "C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\tokenizer.spm" ^
  --temperature 0.7 ^
  --max-tokens 2048
""")

    print("\nPowerShell Script (gemma-quick.ps1):")
    print("""
# Quick launch script for Gemma CLI
$modelPath = "C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\2b-it.sbs"
$tokenizerPath = "C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\tokenizer.spm"

python gemma-cli.py `
  --model $modelPath `
  --tokenizer $tokenizerPath `
  --temperature 0.7 `
  --max-tokens 2048
""")


def example_troubleshooting():
    """Show troubleshooting examples"""
    print("=" * 60)
    print("TROUBLESHOOTING EXAMPLES")
    print("=" * 60)

    issues = [
        {
            "issue": "Model file not found",
            "solution": "Check file path and ensure model file exists",
            "example": 'python gemma-cli.py --model "C:\\correct\\path\\to\\model.sbs"'
        },
        {
            "issue": "WSL not available",
            "solution": "Install WSL or use Windows-native executable",
            "example": "wsl --install (then restart)"
        },
        {
            "issue": "Permission denied",
            "solution": "Ensure gemma executable has execute permissions",
            "example": "chmod +x /path/to/gemma/executable"
        },
        {
            "issue": "Generation too slow",
            "solution": "Reduce max-tokens or use smaller model",
            "example": "python gemma-cli.py --model model.sbs --max-tokens 1024"
        },
        {
            "issue": "Out of memory",
            "solution": "Reduce context length or use compressed model",
            "example": "python gemma-cli.py --model model.sbs --max-context 4096"
        },
        {
            "issue": "Debug command execution",
            "solution": "Use debug mode to see exact commands",
            "example": "python gemma-cli.py --model model.sbs --debug"
        }
    ]

    for issue in issues:
        print(f"\nProblem: {issue['issue']}")
        print(f"Solution: {issue['solution']}")
        print(f"Example: {issue['example']}")
        print()


def main():
    """Run all examples"""
    print("Gemma CLI Wrapper - Usage Examples")
    print("This guide shows various ways to use the gemma-cli.py wrapper\n")

    example_basic_usage()
    example_interactive_commands()
    example_conversation_flow()
    example_batch_scripts()
    example_troubleshooting()

    print("=" * 60)
    print("GETTING STARTED")
    print("=" * 60)
    print("1. Ensure WSL is installed and working")
    print("2. Build gemma executable in WSL environment")
    print("3. Download model files to C:\\codedev\\llm\\.models\\")
    print("4. Run: python gemma-cli.py --model 'path\\to\\model.sbs'")
    print("5. Type /help for interactive commands")
    print("6. Start chatting!")

    print("\nFor help: python gemma-cli.py --help")


if __name__ == "__main__":
    main()