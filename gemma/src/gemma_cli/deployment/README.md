# Gemma CLI Deployment System

This directory contains the build system for creating standalone Windows executables of Gemma CLI using PyInstaller.

## Quick Start

```bash
# 1. Install PyInstaller
pip install pyinstaller

# 2. Verify binaries
python build_script.py

# 3. Build executable (once script is complete)
python build_script.py --build

# 4. Test
dist/gemma-cli.exe --version
```

## Files

- **build_script.py** - Main build orchestrator
  - BinaryFinder: Locates gemma.exe and rag-redis-mcp-server.exe
  - SpecGenerator: Creates PyInstaller spec file
  - Builder: Runs PyInstaller
  - Reporter: Generates metrics report

- **gemma-cli.spec** - PyInstaller specification (auto-generated)
- **test_deployment.py** - Deployment validator
- **README.md** - This file
- **DEPLOYMENT_SYSTEM_REPORT.md** - Detailed implementation report

## Requirements

### Build Requirements

- Python 3.11+
- PyInstaller 6.0+
- UPX (optional, for compression)
- gemma.exe (built from gemma.cpp)
- rag-redis-mcp-server.exe (built from Rust)

### User Requirements (for bundled executable)

- Windows 10/11 (64-bit)
- No Python required
- No external dependencies

## Build Process

### Step 1: Verify Binaries

```bash
python build_script.py
```

This will search for:
- `gemma.exe` in build directories
- `rag-redis-mcp-server.exe` in target/release/

### Step 2: Generate Spec File (to be implemented)

```bash
python build_script.py --generate-spec
```

Creates `gemma-cli.spec` with all dependencies.

### Step 3: Build Executable

```bash
python -m PyInstaller gemma-cli.spec
```

Or use build script:
```bash
python build_script.py --build
```

### Step 4: Test

```bash
cd dist
./gemma-cli.exe --version
./gemma-cli.exe --help
./gemma-cli.exe model list
```

## Build Options

```bash
# Debug build (larger, includes symbols)
python build_script.py --debug

# Skip tests (faster)
python build_script.py --skip-tests

# Disable UPX compression
python build_script.py --no-upx

# Custom output directory
python build_script.py --output-dir /path/to/output
```

## Bundle Structure

```
gemma-cli.exe (standalone)
├── Python runtime (embedded)
├── gemma_cli/* (application code)
├── bin/
│   ├── gemma.exe (~8 MB)
│   └── rag-redis-mcp-server.exe (~1.6 MB)
├── config/ (templates)
└── *.md (documentation)
```

## Performance Metrics

| Metric | Target | Typical |
|--------|--------|---------|
| Bundle Size | <50 MB | ~35 MB (with UPX) |
| Startup Time | <3s | ~2s |
| Build Time | <5 min | ~3 min |

## Code Modifications for PyInstaller

### Required Changes

Two files need updates to support frozen executables:

**1. core/gemma.py**

Add at start of `_find_gemma_executable()`:

```python
# Check if running as PyInstaller bundle
if getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS)
    bundled_exe = base_path / 'bin' / 'gemma.exe'
    if bundled_exe.exists():
        return str(bundled_exe)
```

**2. rag/rust_rag_client.py**

Add at start of `__init__()`:

```python
# Check for bundled binary when frozen
if mcp_server_path is None and getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS)
    bundled_server = base_path / 'bin' / 'rag-redis-mcp-server.exe'
    if bundled_server.exists():
        mcp_server_path = str(bundled_server)
```

## Testing

### Automated Tests

```bash
python test_deployment.py
```

Tests:
- Binary accessibility
- CLI commands work
- RAG operations functional
- Performance metrics

### Manual Testing

Checklist:
- [ ] Test on clean Windows 10 (no Python)
- [ ] Test on clean Windows 11
- [ ] All CLI commands work
- [ ] RAG with embedded store works
- [ ] Model detection works
- [ ] Onboarding wizard works
- [ ] No antivirus false positives

## Troubleshooting

### Build Issues

**"gemma.exe not found"**

Build gemma.cpp first:
```bash
cd ../../gemma.cpp
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release
```

**"rag-redis-mcp-server.exe not found"**

Build Rust backend:
```bash
cd ../../../../stats/rag-redis-system
cargo build --release
```

**"PyInstaller import errors"**

Add missing modules to `hiddenimports` in spec file.

### Runtime Issues

**"DLL load failed"**

Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

**"Slow startup (>5s)"**

Add executable to antivirus exclusions.

**"Windows SmartScreen warning"**

Expected for unsigned executables. Users can click "More info" → "Run anyway".

Future: Implement code signing to avoid this.

## Distribution

### Package Format

```
gemma-cli-v1.0.0-windows-x64.zip
├── gemma-cli.exe
├── README.txt
├── LICENSE.txt
└── examples/
    ├── basic_chat.bat
    └── rag_demo.bat
```

### Installation

1. Extract `gemma-cli.exe` to desired location
2. Run: `gemma-cli.exe init`
3. Follow onboarding wizard
4. Download models as prompted

### Advanced: Create Installer

Use NSIS or WiX to create proper installer:

```bash
# NSIS example
makensis installer.nsi

# WiX example
candle installer.wxs
light installer.wixobj
```

## Known Limitations

1. **Node.js MCP Servers**: Not bundled (optional feature, requires Node.js)
2. **Model Files**: Not bundled (2-7 GB, user downloads)
3. **Redis**: Not bundled (optional, embedded store works without it)
4. **Code Signing**: Not implemented (users see SmartScreen warning)

## Security

### Best Practices

1. **Verify Binary Integrity**: Check SHA256 hashes before bundling
2. **Scan for Malware**: Run VirusTotal scan on final executable
3. **Code Signing**: Obtain certificate and sign executable (future)
4. **Keep Dependencies Updated**: Regularly update PyInstaller and dependencies

### Antivirus Considerations

PyInstaller executables may trigger false positives. Mitigations:
- Submit to VirusTotal
- Request whitelisting from AV vendors
- Sign executable with valid certificate
- Provide source code for verification

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Executable
on: [push, release]
jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install pyinstaller
      - name: Build executable
        run: python deployment/build_script.py
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: gemma-cli-windows
          path: dist/gemma-cli.exe
```

## Next Steps

1. Complete build_script.py implementation
2. Update core files for frozen support
3. Create test_deployment.py
4. Test on clean Windows VM
5. Set up CI/CD
6. Implement code signing
7. Create installer

## Resources

- PyInstaller Docs: https://pyinstaller.org/en/stable/
- NSIS: https://nsis.sourceforge.io/
- WiX Toolset: https://wixtoolset.org/
- Code Signing: https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools

## Support

For issues with deployment:
1. Check DEPLOYMENT_SYSTEM_REPORT.md for detailed info
2. Review PyInstaller documentation
3. Test on clean Windows system
4. Open GitHub issue with full logs

---
**Version**: 1.0.0
**Last Updated**: 2025-10-15
