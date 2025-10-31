# Gemma.cpp Deployment Guide

## Quick Start

### 1. Download Pre-built Binary

If available, download the latest `gemma.exe` from releases. Otherwise, build from source:

```batch
:: Open "Developer Command Prompt for VS 2022"
cd C:\codedev\llm\gemma
cmake --preset windows-release
cmake --build --preset windows-release -j 10
```

Binary location: `build\bin\Release\gemma.exe`

### 2. Download Model Files

Download Gemma model weights from [Kaggle](https://www.kaggle.com/models/google/gemma-2/gemmaCpp):

**Recommended starter model**: Gemma 2B Instruction Tuned (SFP format)
- File: `gemma2-2b-it-sfp.sbs` (~2.5GB)
- Tokenizer: `tokenizer.spm` (~4MB)

Extract to a models directory (e.g., `C:\models\gemma\`)

### 3. Create Configuration File

Copy `gemma.config.toml` to one of these locations:
- Same directory as `gemma.exe` (highest priority)
- `%USERPROFILE%\.gemma\config.toml`
- `C:\ProgramData\gemma\config.toml`

Edit the `[model]` section with your model paths:

```toml
[model]
weights = "C:/models/gemma/gemma2-2b-it-sfp.sbs"
tokenizer = "C:/models/gemma/tokenizer.spm"
```

### 4. Run Gemma

#### Basic Usage (No Session)

```batch
gemma.exe
```

This will use settings from config file and start interactive mode.

#### With Session Management

```batch
:: Start new session with auto-save
gemma.exe --session my_chat --save_on_exit

:: Resume existing session
gemma.exe --session my_chat --load_session
```

#### Command-line Override

```batch
:: Override config file settings
gemma.exe ^
  --weights C:\models\gemma\gemma3-4b-it-sfp.sbs ^
  --tokenizer C:\models\gemma\tokenizer.spm ^
  --temperature 0.9 ^
  --max_generated_tokens 4096
```

## Session Management

### Interactive Commands

While running in session mode, use these commands:

| Command | Description |
|---------|-------------|
| `%q` | Quit session |
| `%c` | Clear/reset session history |
| `%s [filename]` | Save session (default: session_<id>.json) |
| `%l [filename]` | Load session from file |
| `%h [N]` | Show last N conversation messages (default: 10) |
| `%i` | Show session statistics (turns, tokens, cache size) |
| `%m` | List all managed sessions |

### Session Files

Sessions are saved as JSON files in the current directory:
- Format: `session_<id>.json`
- Contains: conversation history, timestamps, token counts
- Human-readable and editable

**Example session file**:
```json
{
  "session_id": "my_chat",
  "conversation_length": 10,
  "messages": [...]
}
```

## Configuration Profiles

Use pre-defined profiles for common use cases:

```batch
:: Fast responses (lower quality)
gemma.exe --profile fast

:: Balanced (default)
gemma.exe --profile balanced

:: Best quality (slower)
gemma.exe --profile quality

:: Creative writing
gemma.exe --profile creative

:: Code generation
gemma.exe --profile coding
```

## Advanced Configuration

### Sampling Parameters

Edit `gemma.config.toml` to customize generation behavior:

```toml
[inference]
temperature = 0.7        # Randomness (0.0-2.0, default: 0.7)
top_k = 40              # Sample from top K tokens (default: 40)
min_p = 0.05            # Min-P sampling threshold
typical_p = 0.95        # Typical-P sampling
```

### DRY (Don't Repeat Yourself) Penalty

Prevent repetitive output:

```toml
[inference]
dry_multiplier = 0.8    # Enable DRY penalty
dry_base = 1.75         # Exponential base
dry_allowed_length = 2  # Repetition length before penalty
```

### Context Window

Adjust context size based on available RAM:

```toml
[inference]
seq_len = 4096          # 8GB RAM: use 4096
                        # 16GB RAM: use 8192
                        # 32GB RAM: use 16384
```

## Deployment Scenarios

### Single User Desktop

1. Install to `C:\Program Files\Gemma\`
2. Place config at `%USERPROFILE%\.gemma\config.toml`
3. Create Start Menu shortcut
4. Optional: Add to PATH

### Shared Server

1. Install to `/opt/gemma/` or `C:\Program Files\Gemma\`
2. Global config at `/etc/gemma/config.toml` or `C:\ProgramData\gemma\config.toml`
3. Users override with `~/.gemma/config.toml`
4. Shared model directory with read-only permissions

### Development Environment

1. Keep in source directory: `C:\codedev\llm\gemma\build\bin\Release\`
2. Use local `gemma.config.toml` in source root
3. Multiple model presets in config
4. Session files in project directory

## Performance Tuning

### Model Selection

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Gemma 2B | 2.5GB | ~45 tok/s | Good | Testing, quick responses |
| Gemma 4B SFP | 4.8GB | ~30 tok/s | Better | Balanced, general purpose |
| Gemma 9B | 9GB | ~15 tok/s | Best | High-quality output |

**Note**: SFP format models run ~2x faster than standard formats.

### Hardware Recommendations

| RAM | Max seq_len | Recommended Model |
|-----|-------------|-------------------|
| 8GB | 4096 | Gemma 2B |
| 16GB | 8192 | Gemma 4B |
| 32GB+ | 16384 | Gemma 9B |

### Build Optimizations

For best performance, build with Intel oneAPI:

```batch
.\build_oneapi.ps1 -Config perfpack -Jobs 10
```

Expected improvements vs MSVC:
- 15-35% faster inference
- Better SIMD utilization (AVX2/AVX-512)
- Optimized math libraries (MKL, IPP)

## Troubleshooting

### Model Loading Errors

**Error**: `Failed to load model`
- Verify file paths in config are correct
- Check file exists: `dir C:\models\gemma\*.sbs`
- Ensure you have read permissions

**Error**: `Out of memory`
- Reduce `seq_len` in config
- Use smaller model (2B instead of 4B)
- Close other applications

### Session Issues

**Error**: `Failed to save session`
- Check write permissions in current directory
- Verify disk space available
- Try specifying absolute path: `%s C:\sessions\my_chat.json`

**Error**: `Session not found`
- Check session ID matches: `%m` to list sessions
- Verify JSON file exists in current directory
- Use `%l <full_path>` to load from specific location

### Performance Issues

**Slow first response**:
- Normal - model loading and auto-tuning on first run
- Second/third responses will be faster

**Consistently slow**:
- Check `verbosity` level (set to 0 for best performance)
- Reduce `seq_len` if using large context
- Use SFP format models
- Build with oneAPI optimizations

## Command-Line Reference

### Required Arguments

If config file not found, these are required:

```batch
--weights <path>     Path to .sbs model file
--tokenizer <path>   Path to .spm tokenizer file
```

### Optional Arguments

```batch
--session <id>              Session ID (enables session management)
--load_session             Load existing session
--save_on_exit             Auto-save session on exit

--temperature <float>      Sampling temperature (default: 0.7)
--top_k <int>              Top-K sampling (default: 40)
--max_generated_tokens <int>  Max tokens to generate (default: 2048)
--seq_len <int>            Context window size (default: 4096)

--verbosity <0-2>          Output verbosity (0=quiet, 1=normal, 2=debug)
--profile <name>           Use predefined profile (fast/balanced/quality/creative/coding)

--prompt <text>            Non-interactive: generate single response
--prompt_file <path>       Non-interactive: read prompt from file
```

### Examples

```batch
:: Interactive with session
gemma.exe --session work

:: Single response (non-interactive)
gemma.exe --prompt "Explain quantum computing"

:: High-quality output
gemma.exe --temperature 0.9 --top_k 60 --max_generated_tokens 4096

:: Fast iteration
gemma.exe --profile fast --verbosity 0
```

## System Requirements

### Minimum

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **CPU**: x86_64 with AVX2 support
- **RAM**: 8GB
- **Disk**: 10GB (for models)

### Recommended

- **OS**: Windows 11, Linux (Ubuntu 22.04+)
- **CPU**: x86_64 with AVX-512 support
- **RAM**: 16GB+
- **Disk**: 20GB SSD
- **Build**: Intel oneAPI optimized

### Verified Configurations

- ✅ Windows 11 + Intel Core i7-12700K + 32GB RAM
- ✅ Ubuntu 22.04 + AMD Ryzen 9 5950X + 64GB RAM
- ✅ Windows 10 + Intel Xeon W-2295 + 128GB RAM

## Support and Updates

- **Documentation**: `C:\codedev\llm\gemma\CLAUDE.md`
- **Issues**: GitHub repository
- **Model Downloads**: [Kaggle Gemma Models](https://www.kaggle.com/models/google/gemma-2/gemmaCpp)

## License

Gemma.cpp is licensed under Apache 2.0. Model weights have separate licensing terms - see Kaggle for details.
