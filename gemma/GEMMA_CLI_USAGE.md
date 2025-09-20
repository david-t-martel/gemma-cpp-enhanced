# Gemma CLI Wrapper Usage Guide

## Overview

The Gemma CLI Wrapper (`gemma-cli.py`) provides a chat-like interface for interacting with the WSL-built Gemma executable on Windows. It offers conversation management, streaming responses, and various utility commands.

## Prerequisites

1. **WSL (Windows Subsystem for Linux)** installed and configured
2. **Python 3.7+** installed on Windows
3. **Gemma executable** built in WSL at: `/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma`
4. **Model files** available (`.sbs` format and optionally `.spm` tokenizer)
5. **Optional: colorama** for colored output (`pip install colorama`)

## Installation

No installation required. Simply run the Python script directly:

```bash
python gemma-cli.py [options]
```

## Basic Usage

### Quick Start (with default model)

```bash
python gemma-cli.py --model "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs"
```

### With custom tokenizer

```bash
python gemma-cli.py --model "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs" --tokenizer "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm"
```

### Advanced configuration

```bash
python gemma-cli.py \
  --model "path\to\model.sbs" \
  --tokenizer "path\to\tokenizer.spm" \
  --max-tokens 4096 \
  --temperature 0.8 \
  --max-context 16384
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Path to Gemma model file (.sbs) | **Required** |
| `--tokenizer, -t` | Path to tokenizer file (.spm) | Optional |
| `--max-tokens` | Maximum tokens to generate | 2048 |
| `--temperature` | Sampling temperature (0.0-2.0) | 0.7 |
| `--max-context` | Maximum conversation context length | 8192 |

## Interactive Commands

Once running, you can use these commands within the chat interface:

### Core Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message with all commands |
| `/quit` or `/exit` | Exit the application |
| `/clear` | Clear conversation history |

### Conversation Management

| Command | Description |
|---------|-------------|
| `/save [filename]` | Save conversation to JSON file |
| `/load <filename>` | Load conversation from JSON file |
| `/list` | List all saved conversations |

### Status and Information

| Command | Description |
|---------|-------------|
| `/status` | Show current session status |
| `/settings` | Show current model settings |

## Features

### 🗨️ **Chat Interface**
- Natural conversation flow with context awareness
- Streaming token-by-token output for real-time response
- Conversation history maintained throughout session

### 💾 **Conversation Management**
- Save/load conversations in JSON format
- Automatic context trimming to stay within limits
- Session duration and message count tracking

### ⚙️ **Configuration**
- Adjustable generation parameters (temperature, max tokens)
- Flexible model and tokenizer file paths
- Configurable context window size

### 🎨 **User Experience**
- Colored output (with colorama installed)
- Graceful interrupt handling (Ctrl+C)
- Cross-platform Windows/WSL integration
- Progress indicators and status messages

## Example Session

```
╔══════════════════════════════════════════════════════════════╗
║                       Gemma CLI Wrapper                     ║
║              Chat interface for Gemma models                ║
╚══════════════════════════════════════════════════════════════╝

Model: C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs
Tokenizer: C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm
Max tokens: 2048, Temperature: 0.7
Type /help for commands, /quit to exit
------------------------------------------------------------

You: Hello! What can you help me with?
Assistant: Hello! I'd be happy to help you with various tasks. I can answer questions, help with coding, explain concepts, or assist with creative projects. What would you like to explore today?

You: /save my_first_chat.json
[System] Conversation saved to: C:\Users\username\.gemma_conversations\my_first_chat.json

You: /quit
Goodbye!
```

## Installation & Setup

### Step 1: Prerequisites

1. **Install WSL** (Windows Subsystem for Linux):
   ```bash
   wsl --install
   # Restart your computer after installation
   ```

2. **Install Python 3.7+** on Windows (if not already installed)

3. **Optional: Install colorama** for colored output:
   ```bash
   pip install colorama
   ```

### Step 2: Build Gemma Executable

Build the gemma executable in WSL:
```bash
cd /mnt/c/codedev/llm/gemma/gemma.cpp
cmake --preset make
cmake --build --preset make -j $(nproc)
chmod +x build/gemma
```

### Step 3: Download Model Files

1. Download Gemma model files from [Kaggle](https://www.kaggle.com/models/google/gemma-2/gemmaCpp)
2. Extract and place in `C:\codedev\llm\.models\`
3. Recommended starter: `gemma-gemmacpp-2b-it-v3` (2B parameter model)

### Step 4: Run the CLI

```bash
python gemma-cli.py --model "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs"
```

## Files Created

The CLI wrapper consists of several files:

- **`gemma-cli.py`** - Main CLI application
- **`GEMMA_CLI_USAGE.md`** - This documentation
- **`requirements.txt`** - Python dependencies
- **`demo_cli.py`** - Feature demonstration script
- **`examples/chat_examples.py`** - Usage examples
- **`test_cli.py`** - Testing script

## Architecture

### Components

1. **`ConversationManager`** - Handles chat history and context
2. **`GemmaInterface`** - Manages WSL subprocess communication
3. **`GemmaCLI`** - Main application with command handling
4. **Signal Handlers** - Graceful shutdown on Ctrl+C

### Cross-Platform Integration

- **Windows**: Native Python execution
- **WSL**: Seamless subprocess calls to Linux gemma binary
- **Path Conversion**: Automatic Windows↔WSL path translation
- **Fallback**: Direct execution if WSL unavailable

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Model file not found" | Check file path exists and use absolute paths |
| "WSL not available" | Install WSL: `wsl --install` |
| "Permission denied" | Make executable: `chmod +x gemma` |
| "Generation too slow" | Reduce `--max-tokens` or use smaller model |
| "Out of memory" | Reduce `--max-context` length |

### Debug Mode

Use `--debug` flag to see exact commands being executed:
```bash
python gemma-cli.py --model model.sbs --debug
```

## Performance Tips

1. **Model Selection**: Use `-sfp` models for 2x speed improvement
2. **Context Length**: Keep `--max-context` reasonable (8192 or less)
3. **Temperature**: Lower values (0.1-0.3) for faster, more focused responses
4. **Hardware**: Close CPU-intensive applications during generation

## Integration Examples

### Batch Script (Windows)
```batch
@echo off
python gemma-cli.py ^
  --model "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs" ^
  --temperature 0.7
```

### PowerShell Script
```powershell
python gemma-cli.py `
  --model "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs" `
  --temperature 0.7
```

## API Reference

### Command Line Arguments

```
usage: gemma-cli.py [-h] --model MODEL [--tokenizer TOKENIZER]
                    [--max-tokens MAX_TOKENS] [--temperature TEMPERATURE]
                    [--max-context MAX_CONTEXT] [--debug]

Gemma CLI Wrapper - Chat interface for Gemma models

required arguments:
  --model MODEL, -m MODEL
                        Path to the Gemma model file (.sbs)

optional arguments:
  --tokenizer TOKENIZER, -t TOKENIZER
                        Path to tokenizer file (.spm). Optional for single-file models.
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 2048)
  --temperature TEMPERATURE
                        Sampling temperature (default: 0.7)
  --max-context MAX_CONTEXT
                        Maximum conversation context length (default: 8192)
  --debug              Enable debug mode with verbose output
```

The gemma-cli.py wrapper transforms the command-line gemma executable into a modern, interactive chat interface that feels native on Windows while leveraging the power of WSL for seamless Linux executable integration.