# Required Code Modifications for PyInstaller Support

This document details the exact code changes needed to make Gemma CLI work as a bundled executable.

## Overview

When PyInstaller bundles an application, it extracts files to a temporary directory (`sys._MEIPASS`) at runtime. We need to update binary discovery logic to check this location first.

## Detection Pattern

```python
import sys
from pathlib import Path

# Check if running as PyInstaller bundle
if getattr(sys, 'frozen', False):
    # Running as bundled executable
    base_path = Path(sys._MEIPASS)
    bundled_binary = base_path / 'bin' / 'binary.exe'
    if bundled_binary.exists():
        return str(bundled_binary)
else:
    # Running as normal Python script
    # Use existing discovery logic
```

## File 1: core/gemma.py

### Location
`C:\codedev\llm\gemma\src\gemma_cli\core\gemma.py`

### Method to Modify
`_find_gemma_executable(self)` (around line 101)

### Current Code (excerpt)
```python
def _find_gemma_executable(self) -> str:
    """
    Find gemma executable in standard and configured locations.
    ...
    """
    exe_name = "gemma.exe" if os.name == "nt" else "gemma"
    
    # 1. Check environment variable
    if gemma_path := os.environ.get("GEMMA_EXECUTABLE"):
        if Path(gemma_path).exists():
            return gemma_path

    # TODO: [Deployment] Integrate uvx binary wrapper for gemma.exe execution.
    # TODO: [Executable Discovery] Enhance _find_gemma_executable to check for bundled gemma.exe

    # 2. Search common build directories
    search_paths = [
        Path.cwd() / "build" / "Release" / exe_name,
        # ... more paths
    ]
```

### Required Change

Add frozen detection at the very start (after exe_name assignment):

```python
def _find_gemma_executable(self) -> str:
    """
    Find gemma executable in standard and configured locations.
    
    Search Order:
    0.  PyInstaller bundled binary (if frozen)
    1.  `GEMMA_EXECUTABLE` environment variable.
    2.  Pre-defined common build directories.
    3.  System's PATH.
    ...
    """
    exe_name = "gemma.exe" if os.name == "nt" else "gemma"
    
    # 0. Check for PyInstaller bundled binary FIRST
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
        bundled_exe = base_path / 'bin' / exe_name
        if bundled_exe.exists():
            logger.info(f"Using bundled gemma executable: {bundled_exe}")
            return str(bundled_exe)
    
    # 1. Check environment variable
    if gemma_path := os.environ.get("GEMMA_EXECUTABLE"):
        if Path(gemma_path).exists():
            return gemma_path

    # Rest of existing logic unchanged...
```

### Import Required

Ensure `sys` is imported at the top of the file:
```python
import sys  # Should already be there
```

## File 2: rag/rust_rag_client.py

### Location
`C:\codedev\llm\gemma\src\gemma_cli\rag\rust_rag_client.py`

### Method to Modify
`__init__(self, mcp_server_path, ...)` (around line 81)

### Current Code (excerpt)
```python
def __init__(
    self,
    mcp_server_path: Optional[str] = None,
    startup_timeout: int = 30,
    request_timeout: int = 60,
    max_retries: int = 3,
    retry_delay: float = 0.5,
):
    """Initialize Rust RAG client."""
    
    # If no path provided, search default locations
    if mcp_server_path is None:
        mcp_server_path = self._find_server_binary()
    
    self.mcp_server_path = Path(mcp_server_path)
    # ...
```

### Required Change

Add frozen detection before calling `_find_server_binary()`:

```python
def __init__(
    self,
    mcp_server_path: Optional[str] = None,
    startup_timeout: int = 30,
    request_timeout: int = 60,
    max_retries: int = 3,
    retry_delay: float = 0.5,
):
    """Initialize Rust RAG client with PyInstaller support."""
    
    # If no path provided, check for bundled binary first
    if mcp_server_path is None:
        # Check for PyInstaller bundled binary
        if getattr(sys, 'frozen', False):
            base_path = Path(sys._MEIPASS)
            bundled_server = base_path / 'bin' / 'rag-redis-mcp-server.exe'
            if bundled_server.exists():
                logger.info(f"Using bundled RAG server: {bundled_server}")
                mcp_server_path = str(bundled_server)
            else:
                # Fallback to default search
                mcp_server_path = self._find_server_binary()
        else:
            # Not frozen, use normal discovery
            mcp_server_path = self._find_server_binary()
    
    self.mcp_server_path = Path(mcp_server_path)
    # Rest unchanged...
```

### Import Required

Ensure `sys` is imported at the top of the file:
```python
import sys  # Should already be there
```

## Testing the Changes

### During Development (Not Frozen)

Behavior should be identical:
```bash
# Test normal execution still works
python -m gemma_cli.cli --version
python -m gemma_cli.cli model list
```

### After PyInstaller Build (Frozen)

Binary discovery should find bundled executables:
```bash
# Test bundled executable
dist/gemma-cli.exe --version
dist/gemma-cli.exe model list

# Binaries should be found in _MEIPASS/bin/
```

## Verification Checklist

- [ ] Added sys import to both files
- [ ] Added frozen detection at start of _find_gemma_executable()
- [ ] Added frozen detection in RustRagClient.__init__()
- [ ] Tested normal execution still works
- [ ] Added logging statements for debugging
- [ ] No breaking changes to existing API

## Debug Logging

Add these debug statements to verify it works:

```python
# In core/gemma.py
if getattr(sys, 'frozen', False):
    logger.debug(f"Running as frozen executable, _MEIPASS={sys._MEIPASS}")
else:
    logger.debug("Running as normal Python script")

# In rag/rust_rag_client.py  
if getattr(sys, 'frozen', False):
    logger.debug(f"Frozen mode, checking: {base_path / 'bin'}")
```

## Rollback Plan

If issues arise, the changes are minimal and can be reverted:

1. Remove the `if getattr(sys, 'frozen', False):` blocks
2. Existing logic remains unchanged
3. No API changes, so no other code affected

## Additional Notes

### Why This Works

- `sys.frozen` is set by PyInstaller to True when running as bundled executable
- `sys._MEIPASS` is the temporary extraction directory (e.g., `C:\Users\user\AppData\Local\Temp\_MEIxxxxxx\`)
- Bundled files maintain directory structure: `_MEIPASS/bin/gemma.exe`

### Performance Impact

- Negligible: One additional `Path.exists()` check
- Only runs once at initialization
- No performance impact on normal execution

### Compatibility

- Works with Python 3.11+
- Compatible with PyInstaller 5.0+
- No impact on non-Windows platforms (sys.frozen not set)

## Next Steps After Code Changes

1. Test normal execution still works
2. Build with PyInstaller
3. Test bundled executable
4. Verify both binaries are accessible
5. Run full test suite

---
**Last Updated**: 2025-10-15
**Status**: Ready for implementation
