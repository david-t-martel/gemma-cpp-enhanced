# Onboarding System Quick Start

## Installation

```bash
# Navigate to project directory
cd C:/codedev/llm/gemma

# Install dependencies (if not already installed)
uv pip install -e .

# Install psutil specifically (new dependency)
uv pip install psutil

# Verify installation
uv run python -c "from src.gemma_cli.onboarding import OnboardingWizard; print('✓ Import successful')"
```

## Usage

### First-Time Setup

```bash
# Automatic on first run
uv run gemma-cli chat

# Or explicit initialization
uv run gemma-cli init
```

### Health Checks

```bash
# Basic health checks
uv run gemma-cli health

# Detailed diagnostics
uv run gemma-cli health --verbose
```

### Tutorial

```bash
# Full interactive tutorial
uv run gemma-cli tutorial

# Quick-start guide only
uv run gemma-cli tutorial --quick
```

### Configuration Management

```bash
# Show current configuration
uv run gemma-cli config --show

# Edit configuration
uv run gemma-cli config --edit

# Validate configuration
uv run gemma-cli config --validate
```

### Reset/Reconfigure

```bash
# Reconfigure (keeps data)
uv run gemma-cli init --force

# Reset to defaults
uv run gemma-cli reset

# Full reset (deletes everything)
uv run gemma-cli reset --full

# Reset but keep model paths
uv run gemma-cli reset --keep-models
```

## Testing

```bash
# Run onboarding tests
uv run pytest tests/test_onboarding.py -v

# Run with coverage
uv run pytest tests/test_onboarding.py --cov=src/gemma_cli/onboarding --cov-report=term-missing

# Run specific test
uv run pytest tests/test_onboarding.py::TestWizard -v
```

## Project Structure

```
src/gemma_cli/
├── onboarding/           # Onboarding system
│   ├── __init__.py       # Exports
│   ├── wizard.py         # Main wizard (556 lines)
│   ├── checks.py         # Health checks (332 lines)
│   ├── templates.py      # Config templates (309 lines)
│   └── tutorial.py       # Interactive tutorial (474 lines)
│
├── commands/
│   └── setup.py          # CLI commands (373 lines)
│
└── cli.py                # Main entry point (273 lines)

config/
└── config.example.toml   # Example configuration

docs/
└── ONBOARDING.md         # Full documentation

tests/
└── test_onboarding.py    # Test suite (300 lines)
```

## Configuration Templates

### Minimal (Lightweight)
- Basic CPU inference
- Minimal memory
- No background tasks
- MCP disabled

```bash
# Auto-selected for systems with <8GB RAM
```

### Developer (Default)
- Full features
- MCP + RAG enabled
- Monitoring
- Background tasks

```bash
# Recommended for most users
```

### Performance (Optimized)
- Maximum throughput
- Large connection pools
- Aggressive caching
- Minimal UI overhead

```bash
# For production/batch processing
```

## Quick Examples

### Example 1: First-Time User

```bash
$ uv run gemma-cli chat
No configuration found. Running first-time setup...

Welcome to Gemma CLI!
[Interactive wizard walks through 6 steps]

Step 1/6: System Health Check
✓ Python 3.11.5
✓ 12.3 GB available
✓ 45.2 GB free
✓ Redis connected

Step 2/6: Model Selection
Found 2 model(s):
1. gemma-2b-it.sbs (2.5 GB)
2. gemma-4b-it-sfp.sbs (4.8 GB)
Select model: 1

[... continues through steps 3-6 ...]

Configuration saved!

Would you like to run the interactive tutorial? y

[Tutorial starts]
```

### Example 2: Health Check

```bash
$ uv run gemma-cli health

Running Health Checks...

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                      ┃ Status   ┃ Details             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Python Version             │ ✓ PASS   │ Python 3.11.5       │
│ Available Memory           │ ✓ PASS   │ 12.3 GB available   │
│ Disk Space                 │ ✓ PASS   │ 45.2 GB free        │
│ Redis Connection           │ ✓ PASS   │ Connected to ...    │
│ sentence-transformers      │ ✓ PASS   │ Installed           │
└────────────────────────────┴──────────┴─────────────────────┘

All checks passed! ✓
```

### Example 3: Reconfiguration

```bash
$ uv run gemma-cli init --force

Found existing configuration at ~/.gemma_cli/config.toml
Do you want to reconfigure? y

[Wizard runs again with current values as defaults]

Configuration updated!
```

## Dependencies

### Required
- Python >= 3.10
- click >= 8.1.7
- rich >= 13.7.0
- prompt-toolkit >= 3.0.43
- toml >= 0.10.2
- pydantic >= 2.5.0
- psutil >= 5.9.0 ← **NEW**

### Optional (for full features)
- redis >= 5.0.0
- sentence-transformers >= 2.2.0
- colorama >= 0.4.6

## Troubleshooting

### Import Errors

```bash
# If imports fail, ensure you're in the project root
cd C:/codedev/llm/gemma

# And using uv run
uv run python -m gemma_cli.cli --help
```

### Redis Not Available

The wizard will gracefully continue without Redis. RAG features will be disabled but basic chat works fine.

```bash
# Start Redis (if installed)
redis-server

# Or continue without Redis
# (wizard will detect and skip Redis setup)
```

### Model Not Found

Provide the full path to your .sbs model file:

```
C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs
```

The wizard validates the path before continuing.

### Low Memory Warning

Use a smaller model or reduce context length:

```toml
[conversation]
max_context_length = 4096  # Reduce from 8192
```

## Next Steps

1. **Run the wizard**: `uv run gemma-cli init`
2. **Complete tutorial**: `uv run gemma-cli tutorial`
3. **Start chatting**: `uv run gemma-cli chat`
4. **Check health**: `uv run gemma-cli health`

## Documentation

- **Full Guide**: `docs/ONBOARDING.md`
- **Implementation**: `ONBOARDING_IMPLEMENTATION.md`
- **Example Config**: `config/config.example.toml`

## Support

If you encounter issues:

1. Run health checks: `uv run gemma-cli health --verbose`
2. Check configuration: `uv run gemma-cli config --validate`
3. Reset if needed: `uv run gemma-cli reset --full`
4. Review documentation: `docs/ONBOARDING.md`

## Features Summary

✅ **Interactive Setup Wizard**
- 6-step guided configuration
- Auto-detection of models
- Real-time validation
- Beautiful Rich UI

✅ **Health Checks**
- System requirements
- Redis connection
- Model files
- Dependencies

✅ **Configuration Templates**
- Minimal, Developer, Performance
- Customizable
- Pre-validated

✅ **Interactive Tutorial**
- 4 comprehensive lessons
- Quick-start option
- Markdown formatting

✅ **CLI Commands**
- init, health, tutorial, reset, config
- Rich console output
- Error recovery

✅ **Production Ready**
- Comprehensive error handling
- Graceful degradation
- Full test coverage
- Type hints throughout
