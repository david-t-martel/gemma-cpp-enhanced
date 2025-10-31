# Gemma CLI Refactoring - Click Framework Migration

## Overview

Complete refactoring of `gemma-cli.py` from a monolithic script to a modern, modular Click-based CLI with Rich integration.

## Architecture

### Directory Structure

```
src/gemma_cli/
├── __init__.py                 # Package exports
├── cli.py                      # Main CLI entry point (Click)
│
├── commands/                   # Click command groups
│   ├── __init__.py
│   ├── chat.py                # Interactive chat commands
│   ├── memory.py              # RAG memory commands
│   ├── mcp.py                 # MCP server commands
│   ├── config.py              # Configuration management
│   └── model.py               # Model management
│
├── core/                       # Core functionality (existing)
│   ├── conversation.py        # ConversationManager
│   └── gemma.py               # GemmaInterface
│
├── rag/                        # RAG system (existing)
│   └── adapter.py             # HybridRAGManager
│
├── mcp/                        # MCP integration (existing)
│   └── client.py              # MCPClientManager
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── config.py              # Configuration loading/validation
│   ├── system.py              # System information
│   └── health.py              # Health checks
│
└── widgets/                    # Rich UI widgets (future)
    └── spinner.py             # Custom spinners
```

## Key Features

### 1. Modern CLI Framework (Click)

**Benefits:**
- Subcommand structure: `gemma chat`, `gemma memory`, `gemma model`
- Auto-generated help text
- Type validation and coercion
- Shell completion support (bash, zsh, fish)
- Environment variable integration
- Async/await support

**Example Usage:**
```bash
# Interactive chat
gemma chat interactive --model 4b --temperature 0.8

# One-shot queries
gemma chat ask "Explain transformers"

# Memory operations
gemma memory recall "machine learning" --tier long_term --limit 5
gemma memory store "Important fact" --tier long_term --importance 0.9
gemma memory stats --json

# Model management
gemma model list
gemma model benchmark path/to/model.sbs

# Configuration
gemma config show
gemma config validate
gemma config init

# System health
gemma health
gemma info
```

### 2. Rich Integration

**Features:**
- Colored, formatted output
- Progress bars and spinners
- Tables and panels
- Markdown rendering
- Syntax highlighting
- Status messages

**Components:**
- `Console` - Main output interface
- `Table` - Structured data display
- `Panel` - Grouped content with borders
- `Progress` - Long-running operations
- `Syntax` - Code highlighting
- `Markdown` - Rich text rendering

### 3. Configuration System

**Format:** TOML-based configuration

**Location:** `config/config.toml`

**Validation:** Pydantic models for type safety

**Features:**
- Default values
- Path validation
- Environment variable override
- Multiple profiles

**Example:**
```toml
[model]
default_model = "4b"
model_path = "path/to/model.sbs"
tokenizer_path = "path/to/tokenizer.spm"

[generation]
max_tokens = 2048
temperature = 0.7
max_context = 8192

[rag]
enabled = true
redis_url = "redis://localhost:6379"
prefer_backend = "mcp"

[system]
system_prompt = "You are a helpful AI assistant."
```

### 4. Command Groups

#### Chat Commands (`gemma chat`)
- `interactive` - Full-featured chat session
- `ask` - One-shot questions
- `history` - View conversation history

**In-chat Commands:**
- `/help` - Show commands
- `/clear` - Clear history
- `/save` - Save conversation
- `/history` - View messages
- `/settings` - Show config
- `/stats` - Memory stats
- `/quit` - Exit

#### Memory Commands (`gemma memory`)
- `recall` - Semantic memory retrieval
- `store` - Store content in memory
- `search` - Advanced search with scoring
- `stats` - Usage statistics
- `ingest` - Document ingestion
- `cleanup` - Remove expired entries
- `health` - System health check

#### Model Commands (`gemma model`)
- `list` - List available models
- `info` - Model information
- `benchmark` - Performance testing

#### Config Commands (`gemma config`)
- `show` - Display configuration
- `set` - Update values
- `validate` - Check validity
- `init` - Create default config

#### MCP Commands (`gemma mcp`)
- `status` - Connection status
- `list-tools` - Available tools
- `call` - Invoke MCP tool

### 5. Error Handling

**Features:**
- User-friendly error messages
- Suggestions for fixes
- Debug mode with stack traces
- Graceful degradation
- Exit codes for automation

**Example:**
```python
try:
    result = await operation()
except FileNotFoundError:
    console.print("[red]Model file not found[/red]")
    console.print("[dim]Use: gemma model list[/dim]")
    raise click.Abort()
```

### 6. Async Support

**Custom Click Group:**
```python
class AsyncioGroup(click.Group):
    """Supports async command handlers."""
    def invoke(self, ctx):
        result = super().invoke(ctx)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
```

**Usage:**
```python
@chat_group.command()
@click.pass_context
async def interactive(ctx):
    """Async command handler."""
    await rag_manager.initialize()
    await gemma.generate_response(prompt)
```

### 7. Context Passing

**Pattern:**
```python
@click.pass_context
def command(ctx: click.Context):
    debug = ctx.obj["DEBUG"]
    config = ctx.obj["CONFIG_PATH"]
    console = ctx.obj["CONSOLE"]
```

**Global Options:**
- `--debug` - Enable debug mode
- `--config` - Configuration file path
- `--profile` - Performance profile

**Environment Variables:**
```bash
export GEMMA_DEBUG=1
export GEMMA_CONFIG=/path/to/config.toml
export GEMMA_PROFILE=quality
```

## Migration Guide

### From Old CLI to New CLI

**Old (monolithic):**
```bash
python gemma-cli.py --model path --tokenizer path --enable-rag
# Then use /commands in interactive mode
```

**New (Click-based):**
```bash
# Interactive mode
gemma chat interactive --model 4b --enable-rag

# Direct commands
gemma chat ask "Question"
gemma memory stats
gemma model list
```

### Key Differences

1. **Command Structure:**
   - Old: Single entry point, in-session commands
   - New: Subcommands, direct CLI access

2. **Configuration:**
   - Old: CLI arguments only
   - New: Config file + CLI args + env vars

3. **Output:**
   - Old: Basic colored text
   - New: Rich tables, panels, progress bars

4. **Help System:**
   - Old: `/help` in session
   - New: `--help` on any command

## Installation

### 1. Install Dependencies

```bash
cd C:/codedev/llm/stats
uv pip install -e .
```

This installs:
- Click framework
- Rich for formatting
- Pydantic for validation
- All existing dependencies

### 2. Create Configuration

```bash
gemma config init
```

Edit `config/config.toml` with your paths.

### 3. Verify Installation

```bash
gemma --version
gemma --help
gemma health
```

## Usage Examples

### Interactive Chat

```bash
# Start chat with RAG enabled
gemma chat interactive --enable-rag

# Custom settings
gemma chat interactive \
  --model 4b \
  --temperature 0.8 \
  --max-tokens 2048 \
  --profile quality
```

### Memory Operations

```bash
# Store important information
gemma memory store "Machine learning is..." \
  --tier long_term \
  --importance 0.9 \
  --tags ml --tags concepts

# Recall similar memories
gemma memory recall "deep learning" \
  --tier long_term \
  --limit 10

# Ingest document
gemma memory ingest paper.pdf \
  --tier semantic \
  --importance 0.8 \
  --tags research

# View statistics
gemma memory stats
gemma memory stats --json > stats.json
```

### Model Management

```bash
# List available models
gemma model list

# Get model info
gemma model info path/to/model.sbs

# Benchmark performance
gemma model benchmark model.sbs \
  --iterations 5 \
  --prompt "Test prompt"
```

### Configuration

```bash
# View current config
gemma config show
gemma config show --format json
gemma config show --format toml

# Validate config
gemma config validate

# Update setting
gemma config set model.temperature 0.8
```

### System Health

```bash
# Comprehensive health check
gemma health

# System information
gemma info
gemma info --format json
```

## Shell Completion

### Bash

```bash
eval "$(gemma completion bash)"
```

Add to `~/.bashrc`:
```bash
eval "$(_GEMMA_COMPLETE=bash_source gemma)"
```

### Zsh

```zsh
eval "$(gemma completion zsh)"
```

Add to `~/.zshrc`:
```zsh
eval "$(_GEMMA_COMPLETE=zsh_source gemma)"
```

### Fish

```fish
gemma completion fish | source
```

Add to `~/.config/fish/config.fish`:
```fish
eval (env _GEMMA_COMPLETE=fish_source gemma)
```

## Testing

### Unit Tests

```bash
uv run pytest tests/test_cli_commands.py -v
```

### Integration Tests

```bash
# Test full workflow
gemma chat ask "Test question"
gemma memory store "Test data" --tier working
gemma memory recall "Test" --limit 1
gemma health
```

### Performance Tests

```bash
gemma model benchmark model.sbs --iterations 10
```

## Troubleshooting

### Import Errors

```bash
# Reinstall in development mode
uv pip install -e .
```

### Configuration Issues

```bash
# Validate config
gemma config validate

# Create new config
gemma config init --force
```

### Memory System Issues

```bash
# Check Redis connection
redis-cli ping

# Check memory health
gemma memory health

# View backend stats
gemma memory stats
```

### Model Loading Issues

```bash
# Verify model exists
gemma model list

# Check model info
gemma model info path/to/model.sbs

# Test with debug mode
gemma --debug chat ask "Test"
```

## Development

### Adding New Commands

1. Create command in `src/gemma_cli/commands/<category>.py`
2. Register in command group
3. Add tests in `tests/`
4. Update documentation

**Example:**
```python
# src/gemma_cli/commands/custom.py
import click
from rich.console import Console

console = Console()

@click.group(name="custom")
def custom_group():
    """Custom commands."""
    pass

@custom_group.command()
@click.argument("name")
async def greet(name: str):
    """Greet someone."""
    console.print(f"[green]Hello, {name}![/green]")

# Register in cli.py
from gemma_cli.commands.custom import custom_group
cli.add_command(custom_group, name="custom")
```

### Adding New Options

1. Add to Click decorator
2. Update configuration schema
3. Document in help text
4. Add validation

**Example:**
```python
@chat_group.command()
@click.option(
    "--new-option",
    type=str,
    default="default",
    envvar="GEMMA_NEW_OPTION",
    help="Description of new option",
)
def command(new_option: str):
    """Command with new option."""
    pass
```

## Best Practices

1. **Always use Click decorators** for argument parsing
2. **Pass context** for shared state
3. **Use Rich Console** for all output
4. **Handle errors gracefully** with user-friendly messages
5. **Provide examples** in help text
6. **Support both** interactive and scripting use cases
7. **Validate input** early with type hints
8. **Document commands** with comprehensive docstrings
9. **Test thoroughly** with various inputs
10. **Follow Click conventions** for consistency

## Future Enhancements

1. **Interactive TUI** - Full terminal UI with curses
2. **Plugins System** - Extensible command plugins
3. **History Search** - Fuzzy search through conversations
4. **Templates** - Prompt templates and presets
5. **Streaming UI** - Real-time token display
6. **Export Formats** - PDF, HTML, Markdown
7. **Multi-session** - Parallel chat sessions
8. **Voice I/O** - Speech-to-text integration

## References

- **Click Documentation:** https://click.palletsprojects.com/
- **Rich Documentation:** https://rich.readthedocs.io/
- **Pydantic Documentation:** https://docs.pydantic.dev/
- **Async Click Guide:** https://github.com/pallets/click/issues/85

## Summary

The refactored Gemma CLI provides:

✅ Modern Click-based command structure
✅ Rich formatted output and UI
✅ TOML configuration with validation
✅ Async/await support throughout
✅ Comprehensive error handling
✅ Shell completion support
✅ Modular, extensible architecture
✅ Full integration with existing codebase
✅ Production-ready code quality

All existing functionality from the monolithic `gemma-cli.py` has been preserved and enhanced with better UX, maintainability, and extensibility.
