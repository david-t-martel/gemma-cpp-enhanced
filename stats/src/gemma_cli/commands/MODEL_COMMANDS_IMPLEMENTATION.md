# Model Management Commands Implementation

## Overview

Comprehensive model and profile management system for Gemma CLI, implemented in `src/gemma_cli/commands/model.py` (1048 lines).

## Architecture

### Data Models

1. **ModelPreset**: Model configuration with metadata
   - Size, format, quality ratings
   - Estimated performance (tokens/sec)
   - Hardware requirements
   - File paths (model + tokenizer)

2. **PerformanceProfile**: Generation parameter presets
   - Temperature, top-p, top-k settings
   - Token limits and context length
   - Use case descriptions

3. **HardwareInfo**: System capabilities detection
   - CPU/RAM specifications
   - GPU detection (via GPUtil)
   - Platform information

## Built-in Presets

### Models

- **gemma2-2b-it**: 2.5 GB, SFP format, ~45 tok/s
  - Fast iteration, testing, light tasks
  - Min 8 GB RAM

- **gemma3-4b-it-sfp**: 4.8 GB, SFP format, ~25 tok/s
  - Balanced quality and speed
  - Min 12 GB RAM

- **gemma2-9b-it**: 9.2 GB, SFP format, ~12 tok/s
  - High-quality responses, complex reasoning
  - Min 16 GB RAM

### Profiles

- **fast**: Quick responses (512 tokens, temp 0.7)
- **balanced**: Default quality/speed (2048 tokens, temp 0.8)
- **quality**: Highest quality (4096 tokens, temp 0.9)
- **creative**: Brainstorming (temp 1.2)
- **precise**: Factual/deterministic (temp 0.3)

## Commands

### Model Commands

#### `gemma model list`
- Shows built-in presets with availability status
- Hardware-based recommendations (⭐ marker)
- Optional filesystem discovery (`--show-discovered`)
- Custom path support (`--path`)

#### `gemma model info <model_name>`
- Detailed model specifications
- Hardware requirement validation
- File existence checks
- Overall readiness status

#### `gemma model use <model_name>`
- Set default model in config.toml
- Validates file existence
- Updates model and tokenizer paths
- Creates config if missing

#### `gemma model detect`
- Scans for .sbs files recursively
- Finds matching .spm tokenizers
- Detects format from filename (SFP/BF16/NUQ)
- Optional auto-configuration (placeholder)

#### `gemma model validate <model_name>`
- File existence checks
- Read permission tests
- RAM requirement validation
- Comprehensive status table

#### `gemma model hardware`
- CPU/RAM/GPU detection
- Model recommendations based on RAM
- Profile recommendations based on specs
- Availability status for each model

### Profile Commands

#### `gemma model profile list`
- All built-in profiles
- Hardware-based recommendation (⭐ marker)
- Parameter summary table

#### `gemma model profile info <profile_name>`
- Full parameter details
- Context configuration
- Use case description

#### `gemma model profile use <profile_name>`
- Updates config.toml [generation] section
- Saves all profile parameters
- Creates config if missing

#### `gemma model profile create <name>`
- Custom profile creation
- All parameters configurable via options
- Saves to [custom_profiles] section
- Displays created profile details

## Integration

### Config File Structure

```toml
[model]
default_model = "gemma2-2b-it"
model_path = "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs"
tokenizer_path = "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"

[generation]
max_tokens = 2048
temperature = 0.8
top_p = 0.95
top_k = 50
repetition_penalty = 1.15
max_context = 4096

[custom_profiles.myprofile]
max_tokens = 1024
temperature = 0.7
top_p = 0.9
top_k = 40
repetition_penalty = 1.1
context_length = 4096
```

### Dependencies

- **Click**: CLI framework
- **Rich**: Terminal UI (tables, panels, colors)
- **psutil**: Hardware detection (CPU, RAM)
- **toml**: Config file I/O
- **GPUtil** (optional): GPU detection

### UI Components

All commands use Rich for consistent formatting:
- **Tables**: Multi-column data display
- **Panels**: Grouped information sections
- **Colors**: Status indicators (green/red/yellow)
- **Symbols**: ✓/✗ for validation, ⭐ for recommendations

## Hardware Detection

### CPU/RAM
- Physical and logical core counts
- Total and available RAM
- Usage percentage calculation

### GPU (Optional)
- Requires GPUtil package
- Detects first GPU only
- Memory capacity reporting
- Graceful fallback if unavailable

### Recommendations
- **Models**: Based on RAM capacity
  - 8+ GB → gemma2-2b-it
  - 12+ GB → gemma3-4b-it-sfp
  - 16+ GB → gemma2-9b-it

- **Profiles**: Based on RAM + CPU cores
  - 32+ GB + 8+ cores → quality
  - 16+ GB → balanced
  - < 16 GB → fast

## Testing

### Manual Testing

```bash
# List all models
uv run gemma model list

# Check hardware
uv run gemma model hardware

# Validate specific model
uv run gemma model validate gemma2-2b-it

# Set default model
uv run gemma model use gemma2-2b-it

# List profiles
uv run gemma model profile list

# Set profile
uv run gemma model profile use balanced

# Create custom profile
uv run gemma model profile create dev --max-tokens 512 --temperature 0.5
```

### Import Test

```python
from src.gemma_cli.commands.model import model_group
# Should list: list, info, use, detect, validate, hardware, profile
```

## Future Enhancements

### Planned Features

1. **Auto-configuration**: `--auto-configure` in detect command
2. **Model download**: Integration with Kaggle API
3. **Benchmark integration**: Performance testing
4. **Profile analytics**: Token usage tracking
5. **Model comparison**: Side-by-side specs
6. **Custom model registration**: Add user models to presets
7. **Multi-GPU support**: Detect all available GPUs
8. **VRAM detection**: More accurate GPU memory

### Potential Improvements

- Model file integrity checks (checksum validation)
- Automatic model recommendations on first run
- Profile auto-tuning based on benchmarks
- Model format conversion utilities
- Quantization quality comparison
- Memory usage prediction
- Token throughput benchmarking
- Context length optimization

## Error Handling

All commands include:
- File existence validation
- Config file creation if missing
- Helpful error messages with hints
- Graceful degradation (e.g., GPU detection)
- Click.Abort() on critical failures

## Performance

- Lightweight imports (only load GPUtil when needed)
- Fast filesystem scans (Path.rglob)
- Efficient config I/O (TOML format)
- No expensive operations in list commands

## Code Quality

- **Type hints**: All functions annotated
- **Docstrings**: Comprehensive documentation
- **Dataclasses**: Clean data models
- **Constants**: Centralized preset definitions
- **DRY principles**: Shared utility functions
- **Rich UI**: Consistent formatting

## Lines of Code

- Total: **1048 lines**
- Model commands: ~350 lines
- Profile commands: ~350 lines
- Hardware command: ~150 lines
- Data models + utilities: ~200 lines

## Summary

Complete implementation of Phase 4.4 requirements:
- ✅ Model commands (list, info, use, detect, validate)
- ✅ Profile commands (list, info, use, create)
- ✅ Hardware detection with recommendations
- ✅ Rich UI formatting throughout
- ✅ Config file management
- ✅ Error handling and validation
- ✅ Extensible architecture for future enhancements
