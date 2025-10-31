# Quick Start: Model Management System

## Installation

```bash
cd /c/codedev/llm/stats

# Install dependencies
uv pip install toml tomli_w

# Or sync all dependencies
uv sync
```

## Basic Usage

### 1. Import the Managers

```python
from pathlib import Path
from src.gemma_cli.config.models import (
    ModelManager,
    ProfileManager,
    HardwareDetector,
)

config = Path("config/config_enhanced.toml")
```

### 2. Model Management

```python
# Initialize
model_mgr = ModelManager(config)

# List all models
models = model_mgr.list_models()
for m in models:
    print(f"{m.name}: {m.size_gb}GB {m.format}")

# Get specific model
model = model_mgr.get_model("gemma2-2b-it")

# Validate model files
is_valid, errors = model_mgr.validate_model(model)
if not is_valid:
    print("Errors:", errors)

# Display table
model_mgr.display_models_table()
```

### 3. Profile Management

```python
# Initialize
profile_mgr = ProfileManager(config)

# List profiles
for p in profile_mgr.list_profiles():
    print(f"{p.name}: temp={p.temperature}, max_tokens={p.max_tokens}")

# Get profile
profile = profile_mgr.get_profile("balanced")

# Create custom profile
profile_mgr.create_profile(
    name="my_profile",
    max_tokens=1024,
    temperature=0.8,
    description="My custom settings"
)

# Display table
profile_mgr.display_profiles_table()
```

### 4. Hardware Detection

```python
# Initialize
detector = HardwareDetector()

# Get hardware info
hw_info = detector.get_hardware_info()
print(f"CPU: {hw_info.cpu_cores} cores")
print(f"RAM: {hw_info.ram_available_gb:.1f} GB available")
print(f"GPU: {hw_info.gpu_info or 'None'}")

# Display table
detector.display_hardware_info()
```

### 5. Get Recommendations

```python
# Recommend model
model = detector.recommend_model(model_mgr, hw_info)
print(f"Recommended model: {model.name}")

# Recommend profile
profile = profile_mgr.recommend_profile(hw_info)
print(f"Recommended profile: {profile.name}")

# Recommend settings
settings = detector.recommend_settings(hw_info)
print(f"Max tokens: {settings['max_tokens']}")
print(f"Context length: {settings['context_length']}")
```

## Configuration

### config_enhanced.toml

```toml
[model_presets.my_model]
weights = "C:/path/to/model.sbs"
tokenizer = "C:/path/to/tokenizer.spm"
format = "sfp"
size_gb = 2.5
avg_tokens_per_sec = 45
quality = "high"
use_case = "General purpose"
context_length = 8192
min_ram_gb = 4

[performance_profiles.my_profile]
max_tokens = 1024
temperature = 0.7
top_p = 0.95
top_k = 50
description = "Custom profile"
use_case = "general"
```

## Testing

```bash
# Run test suite
uv run python test_model_management.py

# Type checking
uv run mypy src/gemma_cli/config/models.py --ignore-missing-imports
```

## Common Tasks

### Auto-Discover Models

```python
detected = model_mgr.detect_models()
for weights, tokenizer in detected:
    print(f"Found: {weights.name}")
```

### Get Model Details

```python
info = model_mgr.get_model_info(model)
print(f"Size: {info['size_gb']:.2f} GB")
print(f"Modified: {info['weights_modified']}")
```

### Set Default Model

```python
model_mgr.set_default_model("gemma3-4b-it-sfp")
```

### Update Profile

```python
profile_mgr.update_profile("my_profile", temperature=0.9)
```

## Full Example

```python
from pathlib import Path
from src.gemma_cli.config.models import (
    ModelManager,
    ProfileManager,
    HardwareDetector,
)

# Initialize
config = Path("config/config_enhanced.toml")
model_mgr = ModelManager(config)
profile_mgr = ProfileManager(config)
detector = HardwareDetector()

# Get hardware info
hw_info = detector.get_hardware_info()
detector.display_hardware_info(hw_info)

# Get recommendations
recommended_model = detector.recommend_model(model_mgr, hw_info)
recommended_profile = profile_mgr.recommend_profile(hw_info)
settings = detector.recommend_settings(hw_info)

print(f"\n=== Recommendations ===")
print(f"Model: {recommended_model.name}")
print(f"Profile: {recommended_profile.name}")
print(f"Settings: {settings}")

# Validate recommended model
is_valid, errors = model_mgr.validate_model(recommended_model)
if is_valid:
    print("\n✓ Model validated successfully")
else:
    print(f"\n✗ Validation errors: {errors}")

# Display all models and profiles
model_mgr.display_models_table()
profile_mgr.display_profiles_table()
```

## Troubleshooting

### Import Error
```bash
# Install missing dependencies
uv pip install toml tomli_w psutil rich pydantic
```

### No Models Found
```python
# Check config has model_presets section
# Or run auto-discovery
detected = model_mgr.detect_models()
```

### Validation Fails
```python
# Get detailed errors
is_valid, errors = model_mgr.validate_model(model)
print(errors)  # List of validation issues
```

## Next Steps

- Read full documentation: `docs/MODEL_MANAGEMENT.md`
- Review implementation: `src/gemma_cli/config/models.py`
- Check example config: `config/config_enhanced.toml`
- Run tests: `test_model_management.py`
