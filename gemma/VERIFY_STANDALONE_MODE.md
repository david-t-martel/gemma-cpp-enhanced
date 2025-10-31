# Quick Verification Guide: Standalone Mode Without Redis

## TL;DR

The application now works **without Redis by default**. Here's how to verify:

## Quick Verification Commands

### 1. Check Default Configuration (2 seconds)
```bash
cd /c/codedev/llm/gemma
python -c "from src.gemma_cli.config.settings import RedisConfig; print(f'Standalone mode: {RedisConfig().enable_fallback}')"
```

Expected output: `Standalone mode: True`

### 2. Verify Embedded Store Exists (5 seconds)
```bash
# Run application briefly, then check
ls -lh ~/.gemma_cli/embedded_store.json
```

Expected: File exists (created on first RAG operation)

### 3. Test Without Redis Running (30 seconds)
```bash
# Make sure Redis is NOT running
pgrep redis || echo "Redis not running - good!"

# Try to use RAG features
cd /c/codedev/llm/gemma
uv run python src/gemma_cli/test_embedded_store.py
```

Expected: All tests pass without Redis

### 4. Test Onboarding (2 minutes)
```bash
cd /c/codedev/llm/gemma

# Reset configuration to simulate fresh install
rm ~/.gemma_cli/config.toml

# Run onboarding
uv run python -m gemma_cli.cli init

# Check Step 3 output
# Should say "Memory Storage Configuration"
# Should show Redis as "optional"
# Should default to "Using embedded file-based storage"
```

Expected: Onboarding completes without requiring Redis

## Detailed Verification

### A. Configuration Files

**1. Check settings.py defaults**:
```bash
cd /c/codedev/llm/gemma
grep -A 2 "enable_fallback" src/gemma_cli/config/settings.py
```

Expected output:
```python
enable_fallback: bool = True  # Default: Use embedded store (standalone mode)
```

**2. Check hybrid_rag.py defaults**:
```bash
grep -A 3 "def __init__" src/gemma_cli/rag/hybrid_rag.py | head -4
```

Expected output includes:
```python
def __init__(self, use_embedded_store: bool = True) -> None:
```

**3. Check python_backend.py defaults**:
```bash
grep -A 5 "def __init__" src/gemma_cli/rag/python_backend.py | head -6
```

Expected output includes:
```python
use_embedded_store: bool = True,  # Default: Use embedded store (standalone)
```

### B. Runtime Verification

**1. Test embedded store directly**:
```python
# Save as test_direct.py
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gemma_cli.rag.embedded_vector_store import EmbeddedVectorStore
from gemma_cli.rag.hybrid_rag import StoreMemoryParams

async def test():
    store = EmbeddedVectorStore()
    await store.initialize()

    params = StoreMemoryParams(
        content="Test memory without Redis",
        memory_type="semantic",
        importance=0.8
    )

    memory_id = await store.store_memory(params)
    print(f"✓ Stored memory without Redis: {memory_id[:8]}...")

    await store.close()
    print(f"✓ Persisted to: {store.STORE_FILE}")

asyncio.run(test())
```

Run with: `python test_direct.py`

**2. Test via HybridRAGManager**:
```python
# Save as test_manager.py
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gemma_cli.rag.hybrid_rag import HybridRAGManager, StoreMemoryParams

async def test():
    # Default should use embedded store
    manager = HybridRAGManager()  # No args = embedded store
    await manager.initialize()

    params = StoreMemoryParams(
        content="Testing HybridRAGManager without Redis",
        memory_type="long_term",
        importance=0.9
    )

    memory_id = await manager.store_memory(params)
    print(f"✓ HybridRAGManager works without Redis: {memory_id[:8]}...")

    await manager.close()

asyncio.run(test())
```

Run with: `python test_manager.py`

### C. Application-Level Verification

**1. Start application without Redis**:
```bash
# Ensure Redis is not running
sudo systemctl stop redis 2>/dev/null || pkill redis 2>/dev/null

# Start Gemma CLI
cd /c/codedev/llm/gemma
uv run python -m gemma_cli.cli chat
```

Expected: Application starts successfully

**2. Test RAG commands** (in application):
```
# In Gemma CLI chat interface:
> /rag store "Test memory" --type semantic
> /rag search "Test"
> /rag stats
```

Expected: All commands work without Redis

### D. Documentation Verification

**1. Check updated files**:
```bash
cd /c/codedev/llm/gemma

# Configuration documentation
grep -i "standalone" src/gemma_cli/config/settings.py

# Embedded store documentation
head -30 src/gemma_cli/rag/embedded_vector_store.py

# Onboarding documentation
grep -i "Memory Storage Configuration" src/gemma_cli/onboarding/wizard.py
```

**2. Check new documentation**:
```bash
# Standalone operation guide
cat src/gemma_cli/STANDALONE_OPERATION.md | head -50

# Configuration changes summary
cat CONFIGURATION_CHANGES_SUMMARY.md | head -30
```

## Success Criteria

✅ **Configuration**: `enable_fallback=True` is default
✅ **Code**: `use_embedded_store=True` is default in RAG classes
✅ **Documentation**: Clearly states Redis is optional
✅ **Onboarding**: Shows Redis as optional, embedded as default
✅ **Functional**: Application works without Redis running
✅ **Data**: Embedded store file created at `~/.gemma_cli/embedded_store.json`

## Troubleshooting

### Issue: "Redis connection failed"
**Cause**: Application is trying to connect to Redis
**Solution**: Check `~/.gemma_cli/config.toml` has `enable_fallback = true`

### Issue: "ModuleNotFoundError: No module named 'gemma_cli'"
**Cause**: Python path not set correctly
**Solution**: Use `uv run` or set `PYTHONPATH=/c/codedev/llm/gemma/src`

### Issue: "Embedded store file not created"
**Cause**: No RAG operations performed yet
**Solution**: Store a memory first: `/rag store "test"`

### Issue: "Tests fail with import errors"
**Cause**: Dependencies not installed
**Solution**: Run `uv sync` or `pip install -e .` first

## What Changed

### Before This Update
- ❌ `enable_fallback=True` but **not used as default**
- ❌ `use_embedded_store=False` default in code
- ❌ Redis appeared required in onboarding
- ❌ No documentation about standalone mode

### After This Update
- ✅ `enable_fallback=True` **is the default**
- ✅ `use_embedded_store=True` default in code
- ✅ Redis shown as optional in onboarding
- ✅ Comprehensive documentation for standalone mode

## One-Line Summary

**Before**: "Redis is required (with fallback)"
**After**: "Standalone by default (Redis optional)"

## Files Modified

1. `src/gemma_cli/config/settings.py` - Documented default
2. `src/gemma_cli/rag/hybrid_rag.py` - Changed default to `use_embedded_store=True`
3. `src/gemma_cli/rag/python_backend.py` - Changed default to `use_embedded_store=True`
4. `src/gemma_cli/rag/embedded_vector_store.py` - Enhanced documentation
5. `src/gemma_cli/onboarding/wizard.py` - Updated onboarding flow

## Files Created

1. `src/gemma_cli/test_embedded_store.py` - Test suite
2. `src/gemma_cli/STANDALONE_OPERATION.md` - User documentation
3. `CONFIGURATION_CHANGES_SUMMARY.md` - Developer documentation
4. `VERIFY_STANDALONE_MODE.md` - This file

## Next Steps After Verification

1. ✅ Run test suite: `uv run python src/gemma_cli/test_embedded_store.py`
2. ✅ Test onboarding: `uv run python -m gemma_cli.cli init --force`
3. ✅ Test application: `uv run python -m gemma_cli.cli chat`
4. 📝 Update main README.md with standalone operation info
5. 📝 Add migration guide for Redis users
6. 🧪 Performance testing: embedded vs Redis benchmarks

## Questions?

**Q: Does this break existing Redis users?**
A: No! Existing configs with `enable_fallback=false` continue to work.

**Q: Can I switch between backends?**
A: Yes! Just change `enable_fallback` in config and restart.

**Q: Is embedded store production-ready?**
A: It's suitable for single-user deployments and datasets <10K memories. Use Redis for production with high availability needs.

**Q: How do I verify I'm using embedded store?**
A: Check if `~/.gemma_cli/embedded_store.json` exists and grows when you store memories.

**Q: Where can I learn more?**
A: Read `src/gemma_cli/STANDALONE_OPERATION.md` for comprehensive guide.
