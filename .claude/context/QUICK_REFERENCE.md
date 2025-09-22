# LLM Ecosystem - Quick Reference

## 🚀 Quick Start Commands

### Build Everything
```bash
# From project root (C:\codedev\llm)
uv pip install -r requirements.txt
cd rag-redis && cargo build
cd ../gemma && cmake . && make
```

### Run Tests
```bash
uv run pytest stats/tests/ -v
cd rag-redis && cargo test
```

### Start Services
```bash
redis-server --port 6380
uv run python stats/server.py
```

## 📁 Key Directories

| Component | Path | Purpose |
|-----------|------|---------|
| RAG/Redis | `/rag-redis/` | Rust-based RAG system |
| Python Framework | `/stats/` | Agent framework |
| Gemma Integration | `/gemma/` | C++ model inference |
| Context | `/.claude/context/` | Project documentation |

## 🔧 Current Issues

1. **Rust Build**: 90% complete, needs more RAM
2. **Gemma.exe**: Windows compatibility issue
3. **Memory**: Full model won't load
4. **Redis**: Connection not implemented

## ✅ Recent Fixes

- Python syntax in `tools.py` ✓
- RAG reorganization complete ✓
- MCP configs updated ✓
- 67% memory reduction ✓

## 🎯 Next Steps

1. Complete Rust build with adequate resources
2. Fix gemma.exe for Windows
3. Implement Redis connection
4. Complete FFI integration

## 🛠️ Debug Commands

```bash
# Check Python imports
uv run python -c "import stats.tools"

# Verify Rust workspace
cd rag-redis && cargo check

# Test Redis
redis-cli -p 6380 ping

# Check memory
wmic process get Name,WorkingSetSize
```

## 📊 Performance Metrics

- Memory: 67% reduction achieved
- Tokenization: 10x faster with Rust
- Target coverage: 85% minimum
- Redis port: 6380 (Windows-friendly)

## 🔑 Key Patterns

### Python
```python
# Always use uv
uv run python script.py
uv pip install package

# Async pattern
async def process():
    return await operation()
```

### Rust
```rust
// Result types
fn process() -> Result<Data, Error>

// SIMD optimizations
use std::simd::*;
```

## 👥 Agent Roles

| Agent | Contribution |
|-------|-------------|
| architect-reviewer | Codebase analysis |
| rust-pro | Cargo workspace |
| python-pro | Syntax fixes |
| debugger | Path resolution |
| test-automator | Verification |

## 🚨 Critical Rules

1. **NEVER** use bare `python` or `pip`
2. **ALWAYS** use `uv run python` and `uv pip`
3. **SEARCH** before creating new files
4. **UPDATE** existing files, don't duplicate
5. **TEST** everything before claiming success

---
*Quick Reference v1.0 - Updated 2025-09-19*