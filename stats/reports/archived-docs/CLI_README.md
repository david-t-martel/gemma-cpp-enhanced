# Gemma Chatbot CLI

A comprehensive command-line interface for the Google Gemma chatbot with PyTorch, featuring interactive chat, training, server management, and configuration capabilities.

## Features

- 🤖 **Interactive Chat**: Rich-formatted chat sessions with streaming support
- 🎓 **Training & Fine-tuning**: Complete training pipeline with LoRA support
- 🚀 **Server Management**: Full HTTP server lifecycle management
- ⚙️ **Configuration Management**: Comprehensive config file handling
- 📊 **Model Management**: Download, list, and manage models
- 🎨 **Rich UI**: Beautiful terminal interface with progress bars and formatting
- 💾 **Session Persistence**: Save and resume chat conversations
- 🔧 **Auto-completion**: Shell completion support

## Installation

The CLI is automatically installed when you install the gemma-chatbot package:

```bash
# Install the package
uv pip install -e .

# The CLI is available as:
gemma-cli --help

# Or run directly:
uv run python -m src.cli.main --help
```

## Quick Start

### 1. Check System Status
```bash
gemma-cli status
```

### 2. Initialize Configuration
```bash
gemma-cli config init --template basic
```

### 3. Download a Model
```bash
gemma-cli models --download google/gemma-2b-it
```

### 4. Start Interactive Chat
```bash
gemma-cli chat interactive
```

### 5. Quick Chat (One-off Response)
```bash
gemma-cli quick-chat "Explain quantum computing in simple terms"
```

## Command Structure

```
gemma-cli
├── version          # Show version information
├── status           # System and environment status
├── quick-chat       # One-off chat responses
├── models          # Model management
├── chat/           # Interactive chat commands
│   ├── interactive # Start chat session
│   ├── list        # List saved sessions
│   └── delete      # Delete sessions
├── train/          # Training and fine-tuning
│   ├── prepare     # Prepare training data
│   ├── finetune    # Fine-tune models
│   ├── evaluate    # Evaluate models
│   ├── list        # List models/checkpoints
│   └── config      # Generate training configs
├── serve/          # Server management
│   ├── start       # Start HTTP server
│   ├── stop        # Stop server
│   ├── status      # Server status
│   └── logs        # View server logs
└── config/         # Configuration management
    ├── show        # Show current config
    ├── set         # Set config values
    ├── unset       # Remove config values
    ├── reset       # Reset to defaults
    ├── validate    # Validate config
    ├── init        # Initialize config
    └── export      # Export config
```

## Detailed Usage

### Interactive Chat

Start an interactive chat session with advanced features:

```bash
# Basic interactive chat
gemma-cli chat interactive

# With specific model and settings
gemma-cli chat interactive \
  --model google/gemma-7b-it \
  --temperature 0.8 \
  --max-tokens 2048 \
  --stream

# Resume existing session
gemma-cli chat interactive --session <session-id>

# With system prompt
gemma-cli chat interactive --system "You are a helpful coding assistant"
```

**Chat Commands (during session):**
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/history` - Show message history
- `/save` - Save current session
- `/stats` - Show session statistics
- `/exit` - Exit chat session

### Training & Fine-tuning

Complete training pipeline with data preparation and fine-tuning:

```bash
# Prepare training data
gemma-cli train prepare data/raw_data.jsonl \
  --output data/processed \
  --format jsonl \
  --split "0.8,0.1,0.1" \
  --max-length 2048

# Generate training configuration
gemma-cli train config --template lora --output training_config.yaml

# Fine-tune model with LoRA
gemma-cli train finetune data/processed \
  --model google/gemma-2b \
  --output ./fine_tuned_model \
  --config training_config.yaml \
  --epochs 3 \
  --batch-size 4 \
  --lora \
  --lora-rank 16

# Evaluate fine-tuned model
gemma-cli train evaluate ./fine_tuned_model data/processed/test.jsonl \
  --metrics perplexity bleu rouge-1 \
  --output evaluation_results.json

# List available models and checkpoints
gemma-cli train list
gemma-cli train list ./fine_tuned_model
```

### Server Management

Full HTTP server lifecycle management:

```bash
# Start server
gemma-cli serve start \
  --host 0.0.0.0 \
  --port 8000 \
  --model google/gemma-2b-it \
  --workers 2 \
  --cors

# Start as daemon
gemma-cli serve start --daemon --port 8000

# Check server status
gemma-cli serve status --detailed

# Watch server status (live updates)
gemma-cli serve status --watch --refresh 5

# View server logs
gemma-cli serve logs --follow --lines 100

# Stop server
gemma-cli serve stop

# Force stop server
gemma-cli serve stop --force
```

### Configuration Management

Comprehensive configuration file handling:

```bash
# Show current configuration
gemma-cli config show

# Show specific section
gemma-cli config show --section model --format yaml

# Set configuration values
gemma-cli config set model.temperature 0.8
gemma-cli config set server.port 9000 --type int
gemma-cli config set performance.use_flash_attention true --type bool

# Remove configuration values
gemma-cli config unset model.top_k

# Reset configuration
gemma-cli config reset --section model
gemma-cli config reset  # Reset entire config

# Validate configuration
gemma-cli config validate --fix

# Initialize with templates
gemma-cli config init --template production --overwrite

# Export configuration
gemma-cli config export config_backup.yaml --format yaml
```

### Model Management

Download and manage models:

```bash
# List available models (local)
gemma-cli models

# List all models (local + remote)
gemma-cli models --all

# Download specific model
gemma-cli models --download google/gemma-2b-it

# Get model information
gemma-cli models --info google/gemma-7b-it
```

## Configuration

The CLI uses a hierarchical configuration system:

1. **Default settings** (built into the application)
2. **Config file** (`~/.gemma/config.yaml`)
3. **Environment variables** (prefix: `GEMMA_`)
4. **Command line arguments** (highest priority)

### Configuration Templates

Three built-in templates are available:

- **`basic`**: Simple setup for experimentation
- **`development`**: Debug mode with verbose logging
- **`production`**: Optimized for server deployment

### Configuration File Structure

```yaml
model:
  name: google/gemma-2b-it
  temperature: 0.7
  max_length: 2048
  top_p: 0.95
  top_k: 40

performance:
  device: auto
  precision: float16
  batch_size: 1
  use_cache: true
  use_flash_attention: true

server:
  host: 0.0.0.0
  port: 8000
  workers: 1
  cors: true
  log_level: info

logging:
  level: info
  file: ~/.gemma/logs/gemma.log
  rotation: daily
```

## Environment Variables

Common environment variables:

```bash
# Hugging Face token for model downloads
export HUGGINGFACE_HUB_TOKEN=your_token_here

# API key for server authentication
export GEMMA_API_KEY=your_api_key_here

# Override config file location
export GEMMA_CONFIG_FILE=~/.gemma/config.yaml

# Set logging level
export GEMMA_LOG_LEVEL=debug

# Override model cache directory
export GEMMA_CACHE_DIR=~/.cache/gemma
```

## Advanced Features

### Auto-completion

Enable shell completion:

```bash
# For bash
gemma-cli --install-completion bash
source ~/.bashrc

# For zsh
gemma-cli --install-completion zsh
source ~/.zshrc

# For fish
gemma-cli --install-completion fish
```

### Session Management

Chat sessions are automatically saved and can be resumed:

```bash
# List all saved sessions
gemma-cli chat list

# Resume specific session
gemma-cli chat interactive --session abc123def

# Delete old sessions
gemma-cli chat delete abc123def --force
```

### Streaming Responses

Enable streaming for real-time responses:

```bash
# Interactive chat with streaming
gemma-cli chat interactive --stream

# Quick chat with streaming
gemma-cli quick-chat "Write a story" --stream
```

### Custom System Prompts

Use system prompts to customize behavior:

```bash
gemma-cli chat interactive --system "You are a helpful Python coding assistant. Provide clear, well-commented code examples."
```

## Troubleshooting

### Common Issues

1. **Model not found**: Use `gemma-cli models --download <model-name>` to download
2. **CUDA out of memory**: Reduce `batch_size` or use quantization (`precision: int8`)
3. **Port already in use**: Use `--port` to specify different port
4. **Configuration errors**: Use `gemma-cli config validate --fix`

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
gemma-cli --verbose <command>
gemma-cli --log-file debug.log <command>
```

### System Requirements

Check system compatibility:

```bash
gemma-cli status
gemma-cli version --verbose
```

## Examples

### Example 1: Quick Setup and Chat

```bash
# Initialize configuration
gemma-cli config init

# Download a small model
gemma-cli models --download google/gemma-2b-it

# Start interactive chat
gemma-cli chat interactive --model google/gemma-2b-it
```

### Example 2: Fine-tuning Workflow

```bash
# Prepare data
gemma-cli train prepare my_data.jsonl --output ./prepared_data

# Generate config
gemma-cli train config --template lora --output lora_config.yaml

# Fine-tune
gemma-cli train finetune ./prepared_data \
  --config lora_config.yaml \
  --model google/gemma-2b \
  --output ./my_finetuned_model

# Evaluate
gemma-cli train evaluate ./my_finetuned_model ./prepared_data/test.jsonl
```

### Example 3: Production Server

```bash
# Initialize production config
gemma-cli config init --template production

# Start server as daemon
gemma-cli serve start --daemon --port 8000 --workers 4

# Monitor server
gemma-cli serve status --watch

# View logs
gemma-cli serve logs --follow
```

## Integration

The CLI can be integrated into scripts and workflows:

```python
import subprocess
import json

# Run CLI commands programmatically
result = subprocess.run([
    "gemma-cli", "quick-chat", "Hello, world!",
    "--format", "json"
], capture_output=True, text=True)

response = json.loads(result.stdout)
print(response["content"])
```

## Performance Tips

1. **Use quantization** for limited memory: `config set performance.precision int8`
2. **Enable optimizations**: `config set performance.use_torch_compile true`
3. **Adjust batch size** based on available memory
4. **Use streaming** for better user experience with long responses
5. **Cache models** locally to avoid repeated downloads

## Support

For issues and questions:

1. Check `gemma-cli status` for system issues
2. Use `gemma-cli config validate` for configuration problems
3. Enable verbose logging with `--verbose`
4. Consult the logs in `~/.gemma/logs/`

---

**Note**: This CLI is designed to work with Google Gemma models and requires proper setup of PyTorch and transformers libraries. CUDA support is recommended but not required.
