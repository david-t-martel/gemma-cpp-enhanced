# Rust Extensions Build Setup

## Fixed Python Library Linking Issue

The Rust extensions build was failing with:
```
LINK : fatal error LNK1181: cannot open input file 'python311.lib'
```

### Root Cause
The build script was hardcoded to link against `python311.lib`, but the UV Python installation uses Python 3.13, which provides `python313.lib`.

### Solution
Updated `build.rs` to:

1. **Dynamically detect Python version** - Uses environment variables and Python executable to determine the correct version
2. **Auto-discover UV Python library paths** - Searches common UV installation directories
3. **Set proper library search paths** - Configures Cargo to find the correct Python libraries

### Key Changes in build.rs
- Added `get_python_version()` function to detect Python version dynamically
- Added `get_uv_python_lib_path()` function to find UV Python library directories
- Modified Windows linking logic to use detected version and paths

### Build Commands

#### Basic build (recommended):
```bash
cd rust_extensions
uv run maturin develop --release
```

#### From parent directory:
```bash
uv run maturin develop --manifest-path rust_extensions/Cargo.toml --release
```

#### With explicit environment variables (if needed):
```bash
$env:PYTHON_LIB_PATH="C:\Users\david\AppData\Roaming\uv\python\cpython-3.13.3-windows-x86_64-none\libs"
$env:PYO3_PYTHON_VERSION="3.13"
uv run maturin develop --release
```

### Verification
After successful build, test the extension:
```python
import gemma_extensions
print(gemma_extensions.get_build_info())
print(gemma_extensions.check_simd_support())
```

### Dependencies
- `uv` for Python environment management
- `maturin` for Python-Rust builds (automatically installed via uv)
- Rust toolchain with MSVC support
- Visual Studio Build Tools (for linking)

### UV Python Location
The build script automatically detects UV Python installations at:
- Windows: `%USERPROFILE%\AppData\Roaming\uv\python\`
- Linux/Mac: `~/.local/share/uv/python/`
- System: `/opt/uv/python/`

The specific Python version directory contains the `libs/` folder with the required `.lib` files for linking.
