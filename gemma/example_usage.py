#!/usr/bin/env python3
"""
Example usage of gemma-cli.py with the native Windows gemma.exe

This script demonstrates how to use the updated CLI wrapper.
"""

# Example commands to run gemma-cli.py:

# 1. Basic usage with Gemma 3 4B model:
example1 = """
python gemma-cli.py \\
  --model "C:\\codedev\\llm\\.models\\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\\4b-it-sfp.sbs" \\
  --tokenizer "C:\\codedev\\llm\\.models\\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\\tokenizer.spm"
"""

# 2. With custom parameters:
example2 = """
python gemma-cli.py \\
  --model "C:\\codedev\\llm\\.models\\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\\4b-it-sfp.sbs" \\
  --tokenizer "C:\\codedev\\llm\\.models\\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\\tokenizer.spm" \\
  --max-tokens 1024 \\
  --temperature 0.8 \\
  --debug
"""

# 3. For single-file models (if available):
example3 = """
python gemma-cli.py \\
  --model "path\\to\\single-file-model.sbs"
"""

print("=== Gemma CLI Usage Examples ===")
print("\n1. Basic usage:")
print(example1)
print("\n2. With custom parameters:")
print(example2)
print("\n3. Single-file model (if available):")
print(example3)

print("\n=== Available Commands in Interactive Mode ===")
commands = [
    "/help              - Show help",
    "/clear             - Clear conversation",
    "/save [filename]   - Save conversation",
    "/load [filename]   - Load conversation",
    "/list              - List saved conversations",
    "/status            - Show session status",
    "/settings          - Show current settings",
    "/quit or /exit     - Exit application"
]

for cmd in commands:
    print(f"  {cmd}")

print("\n=== Requirements ===")
requirements = [
    "- Windows with gemma.exe built",
    "- Python 3.7+ with colorama (optional for colors)",
    "- Valid Gemma model files (.sbs format)",
    "- Sufficient RAM for model (4GB+ for 4B model)"
]

for req in requirements:
    print(f"  {req}")

print(f"\n=== Notes ===")
notes = [
    "- The CLI automatically uses the Windows gemma.exe at C:\\codedev\\llm\\gemma\\gemma.cpp\\gemma.exe",
    "- No WSL required - this is pure Windows execution",
    "- Supports streaming text generation",
    "- Conversation history is automatically managed",
    "- Use Ctrl+C to interrupt generation",
    "- Conversations are saved to ~/.gemma_conversations/"
]

for note in notes:
    print(f"  {note}")