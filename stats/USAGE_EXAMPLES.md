# Gemma CLI Model Commands - Usage Examples

## Quick Start

### Check Your Hardware
```bash
# See what your system can handle
gemma model hardware

# Output shows:
# - CPU cores and model
# - RAM capacity and usage
# - GPU availability
# - Recommended models (⭐)
# - Recommended profile
```

### List Available Models
```bash
# Show built-in presets
gemma model list

# Discover models in filesystem
gemma model list --show-discovered

# Custom path
gemma model list --path /path/to/models
```

### Get Model Details
```bash
# Full model information
gemma model info gemma2-2b-it

# Shows:
# - Size, format, quality
# - Hardware requirements
# - File paths and validation
# - Readiness status
```

### Set Default Model
```bash
# Use specific model
gemma model use gemma2-2b-it

# With custom config
gemma model use gemma3-4b-it-sfp --config custom.toml
```

## Profile Management

### List Profiles
```bash
# Show all available profiles
gemma model profile list

# Output shows ⭐ for recommended profile
```

### Profile Details
```bash
# See all parameters
gemma model profile info balanced
gemma model profile info fast
gemma model profile info quality
```

### Set Active Profile
```bash
# Use balanced profile (default)
gemma model profile use balanced

# Use fast for quick iteration
gemma model profile use fast

# Use quality for important work
gemma model profile use quality
```

### Create Custom Profile
```bash
# Basic custom profile
gemma model profile create dev --max-tokens 512 --temperature 0.7

# Precise factual responses
gemma model profile create research \
  --max-tokens 1024 \
  --temperature 0.3 \
  --top-p 0.85 \
  --top-k 20

# Creative writing
gemma model profile create creative \
  --max-tokens 2048 \
  --temperature 1.2 \
  --top-p 0.95
```

## Advanced Usage

### Detect Models
```bash
# Scan default location
gemma model detect

# Scan custom directory
gemma model detect --path /custom/models

# Auto-configure (coming soon)
gemma model detect --auto-configure
```

### Validate Model
```bash
# Check if model is ready
gemma model validate gemma2-2b-it

# Validates:
# - File existence
# - Read permissions
# - RAM requirements
# - Overall readiness
```

## Workflow Examples

### First-Time Setup
```bash
# 1. Check hardware
gemma model hardware

# 2. List available models
gemma model list

# 3. Set default model (use recommended)
gemma model use gemma2-2b-it

# 4. Set performance profile
gemma model profile use balanced

# 5. Validate setup
gemma model validate gemma2-2b-it
```

### Switching Models
```bash
# Fast iteration during development
gemma model use gemma2-2b-it
gemma model profile use fast

# Production-quality responses
gemma model use gemma3-4b-it-sfp
gemma model profile use quality

# Complex reasoning tasks
gemma model use gemma2-9b-it
gemma model profile use quality
```

### Custom Profiles for Different Tasks

```bash
# Code generation
gemma model profile create code \
  --max-tokens 2048 \
  --temperature 0.4 \
  --top-p 0.9

# Documentation writing
gemma model profile create docs \
  --max-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.95

# Brainstorming
gemma model profile create brainstorm \
  --max-tokens 1024 \
  --temperature 1.3 \
  --top-p 0.98
```

## Troubleshooting

### Model Not Found
```bash
# Check if model files exist
gemma model list

# Scan filesystem
gemma model detect

# Validate specific model
gemma model validate gemma2-2b-it
```

### Hardware Concerns
```bash
# Check system capabilities
gemma model hardware

# Shows recommended models based on:
# - Available RAM
# - CPU cores
# - GPU availability
```

### Config Issues
```bash
# Models and profiles save to config/config.toml
# View current config:
cat config/config.toml

# Reset by deleting config
rm config/config.toml

# Set up again
gemma model use gemma2-2b-it
gemma model profile use balanced
```

## Integration with Chat

### Use Selected Model/Profile
```bash
# After setting model and profile
gemma model use gemma2-2b-it
gemma model profile use balanced

# Start chat (will use configured settings)
gemma chat interactive

# Override profile for single session
gemma chat ask "Question" --profile fast
```

## Tips and Best Practices

### Model Selection
- **gemma2-2b-it**: Development, testing, quick experiments
- **gemma3-4b-it-sfp**: Production, balanced workloads
- **gemma2-9b-it**: Complex reasoning, high-quality output

### Profile Selection
- **fast**: API testing, rapid iteration (512 tokens)
- **balanced**: General use (2048 tokens)
- **quality**: Important responses (4096 tokens)
- **creative**: Writing, brainstorming (temp 1.2)
- **precise**: Facts, technical content (temp 0.3)

### Hardware Guidelines
- **8 GB RAM**: Use 2B model with fast profile
- **16 GB RAM**: Use 4B model with balanced profile
- **32+ GB RAM**: Use 9B model with quality profile

### Config Management
```bash
# Use different configs for different projects
gemma model use gemma2-2b-it --config project1/config.toml
gemma model use gemma3-4b-it-sfp --config project2/config.toml

# Set via environment variable
export GEMMA_CONFIG=custom.toml
gemma model use gemma2-2b-it
```

## Command Reference

### Model Commands
| Command | Description |
|---------|-------------|
| `list` | Show available models with status |
| `info` | Detailed model information |
| `use` | Set default model |
| `detect` | Scan filesystem for models |
| `validate` | Check model readiness |
| `hardware` | System info and recommendations |

### Profile Commands
| Command | Description |
|---------|-------------|
| `list` | Show available profiles |
| `info` | Detailed profile parameters |
| `use` | Set active profile |
| `create` | Create custom profile |

### Common Options
| Option | Description |
|--------|-------------|
| `--config PATH` | Custom config file |
| `--path PATH` | Custom models directory |
| `--show-discovered` | Auto-detect models |
| `--show-recommendations` | Hardware-based suggestions |

## Example Output

### Model List
```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Name              ┃ Size  ┃ Format      ┃ Quality  ┃ Speed   ┃ Use Case              ┃ Status        ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ gemma2-2b-it ⭐   │ 2.5GB │ SFP (8-bit) │ Good     │ ~45tok/s│ Fast iteration        │ ✓ Available   │
│ gemma3-4b-it-sfp  │ 4.8GB │ SFP (8-bit) │ High     │ ~25tok/s│ Balanced quality      │ ✓ Available   │
│ gemma2-9b-it      │ 9.2GB │ SFP (8-bit) │ Very High│ ~12tok/s│ Complex reasoning     │ ✗ Not Found   │
└───────────────────┴───────┴─────────────┴──────────┴─────────┴───────────────────────┴───────────────┘
```

### Profile List
```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name       ┃ Max Tokens ┃ Temp ┃ Top-P ┃ Context ┃ Description                   ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ fast       │ 512        │ 0.7  │ 0.90  │ 2048    │ Quick responses, lower quality│
│ balanced⭐ │ 2048       │ 0.8  │ 0.95  │ 4096    │ Balance between speed/quality │
│ quality    │ 4096       │ 0.9  │ 0.98  │ 8192    │ Highest quality responses     │
│ creative   │ 2048       │ 1.2  │ 0.95  │ 4096    │ Creative writing              │
│ precise    │ 1024       │ 0.3  │ 0.85  │ 2048    │ Factual, deterministic        │
└────────────┴────────────┴──────┴───────┴─────────┴───────────────────────────────┘
```

### Hardware Info
```
╭─────────────────────── Hardware Information ───────────────────────╮
│ Platform: Windows (AMD64)                                          │
│ CPU: AMD Ryzen 9 5950X                                            │
│   Physical Cores: 16                                               │
│   Logical Cores: 32                                                │
│                                                                    │
│ RAM:                                                               │
│   Total: 64.0 GB                                                   │
│   Available: 48.2 GB                                               │
│   Usage: 24.7%                                                     │
│                                                                    │
│ GPU: Not detected (CPU-only mode)                                 │
╰────────────────────────────────────────────────────────────────────╯

╭──────────────────────── Recommendations ───────────────────────────╮
│ Recommended Models:                                                │
│   ✓ gemma2-2b-it (2.5 GB)                                         │
│   ✓ gemma3-4b-it-sfp (4.8 GB)                                     │
│   ✓ gemma2-9b-it (9.2 GB)                                         │
│                                                                    │
│ Recommended Profile:                                               │
│   quality - Highest quality responses, slower                      │
╰────────────────────────────────────────────────────────────────────╯
```

## Help Commands

Each command has detailed help:
```bash
gemma model --help
gemma model list --help
gemma model info --help
gemma model profile --help
gemma model profile create --help
```
