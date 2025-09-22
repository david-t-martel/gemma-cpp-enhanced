# Enhanced Gemma CLI

A modern command-line interface for Gemma.cpp with interactive REPL functionality, featuring command history, tab completion, session management, and colored output.

## Features

### 🚀 Interactive REPL
- Modern readline-style interface with command history
- Tab completion for commands and file paths
- Colored output for better readability
- Progress bars for model loading and inference
- Real-time token streaming during generation

### 📝 Command System
- Built-in commands for model and backend management
- Session save/load functionality
- Configuration management
- Performance benchmarking tools

### 🎛️ Multiple Execution Modes
- **Interactive Mode**: Full REPL experience
- **Batch Mode**: Execute commands from file
- **Single Prompt**: Command-line compatibility

### 🎨 User Experience
- Colored output (can be disabled)
- Progress indicators
- Timing information
- Configurable prompts and behavior

## Building

### Prerequisites
- CMake 3.20+
- C++20 compatible compiler
- Built Gemma.cpp library

### Build Instructions

```bash
# From the project root
cmake -B build
cmake --build build --target gemma_cli

# Or build just the CLI
cd tools/cli
cmake -B build
cmake --build build
```

### Optional Dependencies
- **cxxopts**: Enhanced argument parsing
- **readline** (Linux/macOS): Better line editing
- **Windows Console API**: Enhanced Windows terminal support

## Usage

### Basic Usage

```bash
# Interactive mode (default)
./gemma_cli --tokenizer tokenizer.spm --weights model.sbs

# Force interactive mode
./gemma_cli --interactive --tokenizer tokenizer.spm --weights model.sbs

# Batch mode
./gemma_cli --batch commands.txt --tokenizer tokenizer.spm --weights model.sbs

# Single prompt (compatible with original gemma)
./gemma_cli --prompt "Hello, how are you?" --tokenizer tokenizer.spm --weights model.sbs
```

### CLI-Specific Options

```bash
--interactive, -i         Force interactive mode (default if no batch)
--batch FILE, -b FILE     Run commands from batch file
--history FILE, -h FILE   Use custom history file
--no-color                Disable colored output
--verbose, -v             Enable verbose output
--config FILE, -c FILE    Load CLI configuration from file
--help                    Show help message
```

All standard Gemma.cpp options are also supported.

## Interactive Commands

### Model Management
```
/model                    Show current model
/model <path>             Load new model
/backend                  Show current backend
/backend <name>           Switch backend (cpu/intel/cuda/vulkan)
```

### Session Management
```
/session save <file>      Save current conversation
/session load <file>      Load conversation from file
/session clear            Clear current conversation
```

### Information and Configuration
```
/config                   Show current configuration
/info                     Show model and system information
/history                  Show command history
/benchmark                Run performance benchmark
```

### Utility Commands
```
/help                     Show available commands
/clear                    Clear screen
/exit, /quit              Exit the CLI
```

## Key Bindings

- **Up/Down Arrows**: Navigate command history
- **Tab**: Command and path completion
- **Ctrl+C**: Interrupt current generation
- **Ctrl+D**: Exit CLI
- **Ctrl+L**: Clear screen

## Configuration

### Configuration File Format

Create `~/.gemma_config` (Linux/macOS) or `%USERPROFILE%\.gemma_config` (Windows):

```ini
# Enhanced Gemma CLI Configuration
enable_colors=true
enable_history=true
enable_completion=true
verbose=false
show_timing=true
show_progress=true
history_file=~/.gemma_history
max_history_size=1000
prompt_style=gemma> 
multiline_prompt=  ... 
```

### Environment Variables

- `GEMMA_CONFIG`: Override default config file location
- `GEMMA_HISTORY`: Override default history file location
- `NO_COLOR`: Disable colors when set

## Batch Mode

Create a batch file with commands and prompts:

```bash
# commands.txt
/config
/model
Tell me about quantum computing
/session save quantum_chat.txt
/exit
```

Run with:
```bash
./gemma_cli --batch commands.txt --tokenizer tokenizer.spm --weights model.sbs
```

## Session Files

Sessions are saved in a simple text format:

```
# Enhanced Gemma CLI Session
model=/path/to/model.sbs
backend=cpu
entries=2
---
timestamp=1640995200
prompt_len=25
Tell me about quantum computing
response_len=150
Quantum computing is a fascinating field...
```

## Backend Support

The CLI supports multiple backends:

- **cpu**: Default CPU backend
- **intel**: Intel oneAPI/SYCL backend (if available)
- **cuda**: NVIDIA CUDA backend (if available)
- **vulkan**: Vulkan compute backend (if available)

Backend availability depends on your build configuration and installed SDKs.

## Performance Features

### Progress Bars
Model loading and long operations show progress:
```
Loading model [████████████████████████████████████████] 100%
```

### Timing Information
Detailed timing for each generation:
```
Timing: Prefill: 45ms, Generation: 2340ms, Speed: 23.4 tokens/sec
```

### Benchmarking
Built-in performance testing:
```
/benchmark
```

## Troubleshooting

### Common Issues

1. **Colors not working**: Use `--no-color` or set `NO_COLOR=1`
2. **History not saved**: Check write permissions for history file
3. **Tab completion not working**: May require readline library on Linux/macOS
4. **Model loading fails**: Verify model path and format

### Debug Mode

Run with verbose output:
```bash
./gemma_cli --verbose --tokenizer tokenizer.smp --weights model.sbs
```

### Windows Notes

- Console features may be limited compared to Unix systems
- Use Windows Terminal for best experience
- PowerShell recommended over Command Prompt

## Examples

### Interactive Session
```bash
$ ./gemma_cli --tokenizer tokenizer.spm --weights gemma-2b-it.sbs

┌─────────────────────────────────────────────────────────────────┐
│                      Enhanced Gemma CLI                        │
│              Interactive REPL with Advanced Features           │
└─────────────────────────────────────────────────────────────────┘

Info: Enhanced Gemma CLI Ready. Type /help for commands or start chatting!
Info: Use Ctrl+C to interrupt generation, Ctrl+D or /exit to quit.

gemma> /info
Model Information:
  Model: gemma-2b-it.sbs
  Backend: cpu
  Layers: 18
  Model dimension: 2048
  Vocabulary size: 256000
  Max sequence length: 8192
  Conversation position: 0

gemma> What is the capital of France?