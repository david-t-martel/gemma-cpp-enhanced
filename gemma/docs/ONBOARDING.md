# Gemma CLI Onboarding System

## Overview

The onboarding system provides a comprehensive first-run experience for gemma-cli, guiding users through configuration, validation, and learning.

## Features

### 1. Interactive Setup Wizard (`wizard.py`)

The main onboarding wizard walks users through:

- **System Health Checks** - Validates Python version, memory, disk space, and dependencies
- **Model Selection** - Auto-detects available models or guides manual entry
- **Redis Configuration** - Tests Redis connection and configures fallback
- **Performance Profile** - Selects from minimal, developer, or performance templates
- **UI Preferences** - Customizes theme, stats display, and interface options
- **Optional Features** - Enables/disables MCP, RAG, and monitoring

```bash
# Run setup wizard
gemma-cli init

# Force reconfiguration
gemma-cli init --force

# Custom config path
gemma-cli init --config-path /path/to/config.toml
```

### 2. Health Checks (`checks.py`)

Comprehensive environment validation:

- **System Requirements** - Python >= 3.10, memory >= 4GB, disk >= 10GB
- **Redis Connection** - Tests connectivity on common ports
- **Model Files** - Validates .sbs model files and tokenizers
- **Dependencies** - Checks for optional libraries
- **Environment Variables** - Validates GEMMA_* environment variables

```bash
# Run health checks
gemma-cli health

# Detailed diagnostics
gemma-cli health --verbose
```

### 3. Configuration Templates (`templates.py`)

Three pre-defined templates for common use cases:

#### Minimal Setup
- Basic CPU-only inference
- Minimal memory usage
- No background tasks
- Ideal for: Testing, low-resource systems

#### Developer Setup (Recommended)
- Full feature set enabled
- MCP and RAG integration
- Performance monitoring
- Background memory consolidation
- Ideal for: Development, power users

#### Performance Optimized
- Maximum throughput
- Larger connection pools
- Aggressive caching
- Minimal UI overhead
- Ideal for: Production, batch processing

### 4. Interactive Tutorial (`tutorial.py`)

Step-by-step lessons covering:

1. **Basic Chat** - Commands, context awareness, interrupting generation
2. **Memory System** - 5-tier architecture, storing/recalling/searching memories
3. **MCP Tools** - Integration with external tools and services
4. **Advanced Features** - Session management, configuration tuning, monitoring

```bash
# Run full tutorial
gemma-cli tutorial

# Quick-start guide only
gemma-cli tutorial --quick
```

### 5. Setup Commands (`setup.py`)

CLI commands for managing configuration:

```bash
# Initialize (first-run setup)
gemma-cli init

# Health checks
gemma-cli health
gemma-cli health --verbose

# Tutorial
gemma-cli tutorial
gemma-cli tutorial --quick

# Configuration management
gemma-cli config --show       # Display current config
gemma-cli config --edit       # Open in text editor
gemma-cli config --validate   # Validate syntax

# Reset configuration
gemma-cli reset               # Reset to defaults
gemma-cli reset --full        # Delete all data
gemma-cli reset --keep-models # Preserve model paths
```

## Architecture

### Wizard Flow

```
Welcome Screen
    ↓
System Health Checks
    ↓
Model Selection
    ├─ Auto-detect models
    └─ Manual path entry
    ↓
Redis Configuration
    ├─ Test localhost:6379
    └─ Manual configuration
    ↓
Performance Profile
    ├─ Minimal
    ├─ Developer (default)
    └─ Performance
    ↓
UI Preferences
    ├─ Theme selection
    ├─ Stats display
    └─ Color scheme
    ↓
Optional Features
    ├─ MCP integration
    ├─ RAG context
    └─ Monitoring
    ↓
Test Configuration
    ├─ Model loading
    └─ Redis connection
    ↓
Save Configuration
    ↓
Tutorial (optional)
```

### File Structure

```
src/gemma_cli/onboarding/
├── __init__.py       - Package exports
├── wizard.py         - Main onboarding wizard
├── checks.py         - Environment validation
├── templates.py      - Configuration templates
└── tutorial.py       - Interactive tutorial

src/gemma_cli/commands/
└── setup.py          - CLI commands (init, health, etc.)
```

## Configuration Output

The wizard generates `~/.gemma_cli/config.toml` with:

```toml
[gemma]
default_model = "..."
default_tokenizer = "..."
executable = "..."

[redis]
host = "localhost"
port = 6379
# ... redis configuration

[memory]
# ... 5-tier memory settings

[mcp]
enabled = true
# ... MCP configuration

[ui]
theme = "default"
show_memory_stats = true
# ... UI preferences

# ... additional sections
```

## Integration with Main CLI

The main CLI (`cli.py`) automatically detects first-run:

```python
@click.group()
def cli(ctx, config):
    # Check for first run
    if not Path(config).exists():
        console.print("No configuration found. Running first-time setup...")
        wizard = OnboardingWizard()
        asyncio.run(wizard.run())
```

## Validation

### System Requirements

| Check | Minimum | Recommended |
|-------|---------|-------------|
| Python | 3.10 | 3.11+ |
| Memory | 4GB | 8GB+ |
| Disk Space | 10GB | 20GB+ |
| Redis | Optional | Recommended |

### Model Files

The wizard validates:
- `.sbs` model files exist and are readable
- Tokenizer `.spm` files are present (optional for single-file models)
- File sizes are reasonable (> 100MB for models)

### Redis Connection

Tests connectivity to:
- `localhost:6379` (default)
- `localhost:6380` (alternative)
- Custom host:port (user-specified)

Provides diagnostics and suggestions if connection fails.

## Error Handling

### Graceful Degradation

- **Redis unavailable** → Continues without RAG features
- **Model not found** → Prompts for manual path entry
- **Low memory** → Warns but allows continuation
- **Missing dependencies** → Shows installation instructions

### Recovery Options

- **Health check failed** → Suggests fixes and allows retry
- **Invalid config** → Offers to reset or edit manually
- **Connection timeout** → Increases timeout or skips

## Customization

### Adding New Templates

Edit `templates.py`:

```python
TEMPLATES["custom"] = {
    "name": "Custom Setup",
    "description": "Your custom configuration",
    "config": {
        # ... configuration
    }
}
```

### Adding Health Checks

Edit `checks.py`:

```python
async def check_custom_requirement() -> tuple[bool, str]:
    # Your validation logic
    return success, message

# Add to check_system_requirements()
checks.append(("Custom Check", *await check_custom_requirement()))
```

### Customizing Tutorial

Edit `tutorial.py`:

```python
async def _lesson_custom(self) -> None:
    markdown_content = """
    # Your Lesson Title
    ...
    """
    console.print(Markdown(markdown_content))
```

## Testing

### Manual Testing

```bash
# Test wizard
uv run python -m gemma_cli.onboarding.wizard

# Test health checks
uv run python -c "
import asyncio
from gemma_cli.onboarding.checks import check_system_requirements
asyncio.run(check_system_requirements())
"

# Test templates
uv run python -c "
from gemma_cli.onboarding.templates import get_template
print(get_template('developer'))
"
```

### Automated Testing

```bash
# Run pytest tests
uv run pytest tests/test_onboarding.py -v

# Test coverage
uv run pytest tests/test_onboarding.py --cov=src/gemma_cli/onboarding
```

## Best Practices

### For Users

1. **Run health checks first** - Identify issues before configuration
2. **Use developer template** - Good balance for most users
3. **Complete tutorial** - Learn all features efficiently
4. **Reconfigure as needed** - Use `gemma-cli init --force` anytime

### For Developers

1. **Keep wizard concise** - Each step under 2 minutes
2. **Provide clear feedback** - Rich panels, progress bars
3. **Validate early** - Test connections before saving
4. **Graceful degradation** - Continue despite non-critical failures

## Troubleshooting

### Common Issues

**Issue**: "Redis connection failed"
**Solution**:
1. Check if Redis is running: `redis-cli ping`
2. Install Redis: https://redis.io/docs/install/
3. Continue without Redis (RAG features disabled)

**Issue**: "Model not found"
**Solution**:
1. Download from Kaggle: https://www.kaggle.com/models/google/gemma-2/
2. Extract to `C:/codedev/llm/.models/`
3. Provide absolute path in wizard

**Issue**: "Low memory warning"
**Solution**:
1. Close unnecessary applications
2. Use smaller model (2B instead of 4B)
3. Reduce `max_context_length` in config

**Issue**: "Python version too old"
**Solution**:
1. Install Python 3.11+: https://www.python.org/downloads/
2. Use `uv` for environment management: `uv venv --python 3.11`

## Future Enhancements

- [ ] GPU backend detection and configuration
- [ ] Automatic model download from Kaggle/HuggingFace
- [ ] Profile import/export
- [ ] Configuration migration from older versions
- [ ] Interactive benchmarking during setup
- [ ] Cloud provider integration (AWS, GCP, Azure)

## References

- [Gemma Models](https://www.kaggle.com/models/google/gemma-2/)
- [Redis Documentation](https://redis.io/docs/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/)
