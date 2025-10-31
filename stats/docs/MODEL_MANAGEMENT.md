# Model Management System Documentation

## Overview

The model management system provides comprehensive functionality for managing Gemma models, performance profiles, and hardware detection with automatic recommendations.

## Architecture

### Components

1. **ModelManager** - Model preset management and validation
2. **ProfileManager** - Performance profile configuration
3. **HardwareDetector** - System capability detection and recommendations

### File Structure

```
src/gemma_cli/config/
├── __init__.py           # Package exports
└── models.py            # Main implementation (880+ lines)

config/
├── config.toml          # Standard config
└── config_enhanced.toml # Enhanced config with presets and profiles
```

## ModelManager

### Purpose
Manages model presets with metadata, validation, and auto-discovery.

### Key Methods

```python
# Load model presets from config
model_mgr = ModelManager(config_path)

# List all configured models
models = model_mgr.list_models()

# Get specific model
model = model_mgr.get_model("gemma2-2b-it")

# Validate model files exist
is_valid, errors = model_mgr.validate_model(model)

# Auto-detect models in standard locations
detected = model_mgr.detect_models()

# Get detailed model information
info = model_mgr.get_model_info(model)

# Set default model
model_mgr.set_default_model("gemma3-4b-it-sfp")

# Display models in rich table
model_mgr.display_models_table()
```

### Configuration Format

```toml
[model_presets.gemma2-2b-it]
weights = "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs"
tokenizer = "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"
format = "sfp"
size_gb = 2.5
avg_tokens_per_sec = 45
quality = "medium"
use_case = "Fast iteration, testing, lightweight tasks"
context_length = 8192
min_ram_gb = 4
```

## ProfileManager

### Purpose
Manages performance profiles for inference parameter tuning.

### Key Methods

```python
# Load profiles from config
profile_mgr = ProfileManager(config_path)

# List all profiles
profiles = profile_mgr.list_profiles()

# Get specific profile
profile = profile_mgr.get_profile("balanced")

# Create custom profile
profile_mgr.create_profile(
    name="my_profile",
    max_tokens=1024,
    temperature=0.8,
    description="My custom profile"
)

# Update existing profile
profile_mgr.update_profile("my_profile", temperature=0.9)

# Delete profile
profile_mgr.delete_profile("my_profile")

# Get hardware-based recommendation
recommended = profile_mgr.recommend_profile(hw_info)

# Display profiles in rich table
profile_mgr.display_profiles_table()
```

### Configuration Format

```toml
[performance_profiles.balanced]
max_tokens = 1024
temperature = 0.7
top_p = 0.95
top_k = 50
description = "Balance between speed and quality for general use"
use_case = "general"
```

## HardwareDetector

### Purpose
Detects system capabilities and provides model/settings recommendations.

### Key Methods

```python
# Initialize detector
detector = HardwareDetector()

# Get CPU information
cpu_info = detector.detect_cpu()
# Returns: {'physical_cores': 16, 'logical_cores': 22, 'frequency_mhz': 1400.0, ...}

# Get memory information
mem_info = detector.detect_memory()
# Returns: {'total_gb': 63.5, 'available_gb': 33.6, 'percent_used': 47.1, ...}

# Check for GPU
has_gpu, gpu_info = detector.detect_gpu()
# Returns: (False, "Intel GPU detected (no acceleration)")

# Get comprehensive hardware info
hw_info = detector.get_hardware_info()
# Returns: HardwareInfo(cpu_cores=16, ram_total_gb=63.5, ...)

# Recommend model based on hardware
recommended_model = detector.recommend_model(model_mgr, hw_info)

# Recommend optimal settings
settings = detector.recommend_settings(hw_info)
# Returns: {'max_tokens': 2048, 'temperature': 0.7, 'context_length': 8192, ...}

# Display hardware info table
detector.display_hardware_info(hw_info)
```

## Data Models

### ModelPreset

```python
class ModelPreset(BaseModel):
    name: str                    # Unique identifier
    weights: str                 # Path to .sbs file
    tokenizer: str               # Path to .spm file
    format: str                  # sfp, bf16, f32, nuq
    size_gb: float               # Model size in GB
    avg_tokens_per_sec: int      # Expected inference speed
    quality: str                 # high, medium, fast
    use_case: str                # Recommended use case
    context_length: int          # Max context (default: 8192)
    min_ram_gb: int              # Minimum RAM requirement
```

### PerformanceProfile

```python
class PerformanceProfile(BaseModel):
    name: str                    # Profile identifier
    max_tokens: int              # Maximum tokens to generate
    temperature: float           # Sampling temperature (0.0-2.0)
    top_p: float                 # Nucleus sampling (0.0-1.0)
    top_k: int                   # Top-K sampling parameter
    description: str             # Profile description
    use_case: str                # Recommended use case
```

### HardwareInfo

```python
class HardwareInfo(BaseModel):
    cpu_cores: int               # Physical CPU cores
    cpu_logical: int             # Logical cores (with HT)
    cpu_freq_mhz: float          # Current CPU frequency
    ram_total_gb: float          # Total system RAM
    ram_available_gb: float      # Available RAM
    has_gpu: bool                # GPU detected
    gpu_info: Optional[str]      # GPU details
    os_system: str               # Operating system
    os_release: str              # OS version
```

## Usage Examples

### Basic Model Management

```python
from pathlib import Path
from src.gemma_cli.config.models import ModelManager

config = Path("config/config_enhanced.toml")
mgr = ModelManager(config)

# List all models
for model in mgr.list_models():
    print(f"{model.name}: {model.size_gb}GB {model.format}")

# Validate model files
model = mgr.get_model("gemma2-2b-it")
is_valid, errors = mgr.validate_model(model)
if not is_valid:
    print(f"Validation errors: {errors}")
```

### Profile Management

```python
from src.gemma_cli.config.models import ProfileManager

mgr = ProfileManager(Path("config/config_enhanced.toml"))

# Create custom profile for code generation
mgr.create_profile(
    name="code_gen",
    max_tokens=2048,
    temperature=0.2,  # Low temp for deterministic code
    top_p=0.85,
    top_k=20,
    description="Code generation profile",
    use_case="coding"
)

# Use profile
profile = mgr.get_profile("code_gen")
print(f"Temperature: {profile.temperature}")
```

### Hardware Detection and Recommendations

```python
from src.gemma_cli.config.models import (
    HardwareDetector,
    ModelManager,
    ProfileManager,
)

detector = HardwareDetector()
hw_info = detector.get_hardware_info()

# Display hardware capabilities
detector.display_hardware_info(hw_info)

# Get recommendations
model_mgr = ModelManager(config)
recommended_model = detector.recommend_model(model_mgr, hw_info)
print(f"Recommended: {recommended_model.name}")

# Get optimal settings
settings = detector.recommend_settings(hw_info)
print(f"Recommended max_tokens: {settings['max_tokens']}")
```

### Auto-Discovery

```python
# Scan for models in standard locations
mgr = ModelManager(config)
detected = mgr.detect_models()

for weights, tokenizer in detected:
    print(f"Found: {weights.name}")
    print(f"  Tokenizer: {tokenizer.name}")
```

## Recommendation Logic

### Model Recommendations

Based on available RAM:
- **< 4GB**: Recommend smallest model or warn insufficient
- **4-8GB**: Recommend 2B models
- **8-16GB**: Recommend 4B models
- **> 16GB**: Recommend 9B+ models (quality priority)

Factors considered:
1. Available RAM vs model requirements
2. Quality tier (prefer high quality if RAM permits)
3. Inference speed (prefer faster if low RAM)

### Profile Recommendations

Based on system capabilities:
- **High-end** (32GB+ RAM, 8+ cores): `quality` profile
- **Mid-range** (16GB+ RAM): `balanced` profile
- **Low-end** (< 16GB RAM): `fast` profile

### Settings Recommendations

Automatically adjusts:
- **max_tokens**: 512 (low RAM) → 2048 (high RAM)
- **context_length**: 2048 → 8192 based on RAM
- **num_threads**: CPU cores - 1 or CPU cores - 2
- **batch_size**: 1 (default) or 2 (8+ cores)

## Integration with CLI

The model management system integrates with existing CLI commands:

```bash
# List models (uses ModelManager)
gemma model list

# Show model info (uses ModelManager.get_model_info)
gemma model info gemma2-2b-it

# Validate model (uses ModelManager.validate_model)
gemma model validate gemma2-2b-it

# Hardware info (uses HardwareDetector)
gemma model hardware

# List profiles (uses ProfileManager)
gemma profile list

# Show profile details (uses ProfileManager.get_profile)
gemma profile info balanced
```

## Performance Characteristics

### Memory Usage
- **ModelManager**: O(n) where n = number of model presets
- **ProfileManager**: O(n) where n = number of profiles
- **HardwareDetector**: Caches hardware info after first detection

### Speed
- Config loading: < 100ms (TOML parsing)
- Model validation: < 50ms per model (file existence checks)
- Hardware detection: < 200ms (first call, then cached)
- Model discovery: Depends on filesystem size (typically < 1s)

## Error Handling

All components use comprehensive error handling:

```python
# Validation errors are caught and reported
try:
    preset = ModelPreset(**data)
except ValidationError as e:
    console.print(f"[red]Invalid preset: {e}[/red]")

# File operations handle missing files gracefully
if not weights_path.exists():
    errors.append(f"Weights file not found: {weights_path}")

# GPU detection handles missing libraries
try:
    import torch
    has_gpu = torch.cuda.is_available()
except ImportError:
    has_gpu = False
```

## Testing

Run the test suite:

```bash
cd /c/codedev/llm/stats
uv run python test_model_management.py
```

Expected output:
```
Model Management System Test Suite

Testing HardwareDetector
OK CPU detection: 16 cores
OK Memory detection: 63.5 GB total
OK GPU detection: Not found
...

All tests passed!
```

## Dependencies

Required packages:
- `pydantic >= 2.0.0` - Data validation
- `psutil >= 5.9.0` - System information
- `rich >= 13.5.0` - Terminal output
- `toml >= 0.10.0` - TOML reading
- `tomli_w >= 1.0.0` - TOML writing

Optional packages:
- `torch` - GPU detection (CUDA)
- `GPUtil` - GPU information

## Future Enhancements

Planned improvements:
1. **Model versioning** - Track model versions and updates
2. **Profile inheritance** - Derive profiles from base templates
3. **Cloud model registry** - Download models from cloud storage
4. **Benchmark integration** - Measure actual inference speed
5. **A/B testing** - Compare model performance
6. **Profile optimization** - Auto-tune profiles based on usage
7. **Multi-GPU support** - Detect and recommend GPU configurations
8. **Remote model validation** - Verify model integrity with checksums

## Troubleshooting

### Common Issues

**No models found:**
- Check `model_presets` section exists in config.toml
- Verify model paths are correct (use absolute paths on Windows)
- Run `model_mgr.detect_models()` to find existing models

**Validation fails:**
- Ensure .sbs and .spm files exist at specified paths
- Check file sizes match declared size_gb (within 10%)
- Verify file permissions allow reading

**GPU not detected:**
- Install PyTorch with CUDA support for NVIDIA GPUs
- Install `GPUtil` package: `uv pip install GPUtil`
- Check GPU drivers are installed correctly

**TOML write errors:**
- Ensure `tomli_w` package is installed
- Check write permissions on config file
- Validate TOML structure before writing

## API Reference

Full API documentation available in source docstrings:

```python
# View class documentation
help(ModelManager)
help(ProfileManager)
help(HardwareDetector)

# View method documentation
help(ModelManager.validate_model)
help(ProfileManager.recommend_profile)
help(HardwareDetector.detect_gpu)
```

## License

Part of the gemma-cli project. See main project LICENSE for details.
