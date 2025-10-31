# Gemma CLI Development - Onboarding Document
*Generated: 2025-10-15*
*Task ID: gemma-cli-development*

## Executive Summary

This document provides comprehensive onboarding for the Gemma CLI project - a Python-based terminal interface for Google's Gemma LLM models. The project has recently undergone critical security fixes and architectural simplifications.

**Current State**: Secure, functional, standalone (no Redis required), well-documented

**Project Location**: `C:\codedev\llm\gemma\src\gemma_cli\`

**Active Serena Project**: `gemma` (C++ focused, at parent directory level)

---

## Project Context

### What is Gemma CLI?

A feature-rich CLI that wraps the native C++ `gemma.exe` inference engine with:
- Interactive chat interface using Click framework
- RAG (Retrieval-Augmented Generation) system with pluggable backends
- 5-tier memory system (working/short-term/long-term/episodic/semantic)
- Configuration management with Pydantic
- Rich terminal UI with syntax highlighting

### Recent Major Changes (January 2025)

1. **Security**: Fixed critical path traversal vulnerability (CWE-22)
2. **Functionality**: Fixed syntax errors in ingest command
3. **Architecture**: Changed defaults to use embedded vector store (no Redis required)
4. **Documentation**: Created comprehensive docs and CLAUDE.md

---

## Project Structure

```
gemma_cli/
├── cli.py                          # Main CLI entry point (Click-based)
├── CLAUDE.md                       # Development guide for Claude Code
├── GEMINI_TODO.md                  # Prioritized roadmap
├── IMPLEMENTATION_SUMMARY.md       # Recent fixes documentation
├── SECURITY_FIX_PATH_TRAVERSAL.md  # Security vulnerability details
├── STANDALONE_OPERATION.md         # User guide for standalone mode
│
├── core/                           # Core inference logic
│   ├── gemma.py                   # GemmaInterface - subprocess management
│   ├── conversation.py            # Multi-turn conversation tracking
│   └── enums.py                   # Type definitions
│
├── rag/                           # RAG system (pluggable backends)
│   ├── hybrid_rag.py             # High-level RAG API with Pydantic models
│   ├── python_backend.py         # Backend selection (Redis vs Embedded)
│   ├── embedded_vector_store.py  # File-based store (DEFAULT)
│   ├── memory.py                 # Memory tier abstractions
│   └── optimizations.py          # Batch embedding (Phase 2 - incomplete)
│
├── config/                        # Configuration management
│   ├── settings.py               # Pydantic models + secure expand_path()
│   ├── models.py                 # Model preset management
│   └── prompts.py                # System prompts
│
├── ui/                           # Terminal UI components
│   ├── console.py               # Rich console (global singleton - to refactor)
│   ├── formatters.py            # Message formatting
│   ├── components.py            # UI widgets (progress bars, panels)
│   ├── theme.py                 # Color schemes
│   └── widgets.py               # Custom UI elements
│
├── commands/                     # CLI command implementations
│   ├── model.py                 # Model management commands
│   ├── rag_commands.py          # RAG/memory commands
│   └── setup.py                 # Setup/configuration commands
│
├── onboarding/                   # First-run setup
│   ├── wizard.py                # Interactive setup wizard
│   ├── checks.py                # System checks
│   ├── templates.py             # Config templates
│   └── tutorial.py              # Usage tutorials
│
├── mcp/                          # Model Context Protocol (INCOMPLETE)
│   ├── client.py                # MCP client implementation
│   ├── config_loader.py         # MCP server configuration
│   ├── example_usage.py         # Usage examples
│   └── README.md                # Comprehensive docs (but not integrated)
│
├── deployment/                   # Deployment tools (INCOMPLETE)
│   ├── build_script.py          # PyInstaller packaging (stub)
│   └── uvx_wrapper.py           # Execution wrapper (stub)
│
└── tests/                        # Test suite (MINIMAL)
    └── logs/                     # Test logs only

test_embedded_store.py            # Comprehensive embedded store tests
```

---

## Architecture Deep Dive

### 1. Execution Flow

```
User Input
    ↓
cli.py (Click command dispatcher)
    ↓
    ├─> chat command → core/gemma.py → subprocess: gemma.exe
    │                      ↓
    │                  Streaming tokens via stdout
    │
    ├─> ingest command → rag/hybrid_rag.py → python_backend.py
    │                                              ↓
    │                                    embedded_vector_store.py
    │                                    (stores in ~/.gemma_cli/embedded_store.json)
    │
    └─> config/model commands → config/settings.py
                                     ↓
                              Pydantic validation
```

### 2. RAG System Architecture

**Key Insight**: The RAG system is designed as **pluggable backends** with embedded store as default.

```python
# User calls
HybridRAGManager(use_embedded_store=True)  # Default: standalone
    ↓
PythonRAGBackend(use_embedded_store=True)
    ↓
    ├─> EmbeddedVectorStore (DEFAULT)
    │   • File: ~/.gemma_cli/embedded_store.json
    │   • Simple keyword search
    │   • Good for <10K documents
    │   • Zero external dependencies
    │
    └─> RedisVectorStore (OPTIONAL)
        • Requires Redis server
        • Advanced vector search
        • Production-grade scaling
```

**Memory Tiers** (5-tier system):
1. **Working**: 15 items, 15min TTL (immediate context)
2. **Short-term**: 100 items, 1hr TTL (recent interactions)
3. **Long-term**: 10K items, 30d TTL (consolidated knowledge)
4. **Episodic**: 5K items, 7d TTL (event sequences)
5. **Semantic**: 50K items, permanent (concept relationships)

### 3. Configuration System

**All configuration uses Pydantic models** in `config/settings.py`:

```python
AppConfig
├─> GemmaConfig          # Model paths, executable location
├─> RedisConfig          # Redis settings (enable_fallback=True by default)
├─> MemoryConfig         # Memory tier TTLs and capacities
├─> EmbeddingConfig      # Embedding model settings
├─> VectorStoreConfig    # Vector search parameters
├─> DocumentConfig       # Document ingestion settings
└─> MCPConfig            # MCP server configurations (not used yet)
```

**Configuration File**: `~/.gemma_cli/config.toml` (auto-created by `init` command)

### 4. Security Architecture

**Critical Pattern**: ALL user-provided paths MUST use `expand_path()` function:

```python
from config.settings import expand_path

# ✅ CORRECT - 5-layer validation
safe_path = expand_path(user_input)

# ❌ NEVER DO THIS - vulnerable to path traversal
unsafe = Path(os.path.expandvars(user_input))
```

**expand_path() Security Layers**:
1. Pre-validation: Check raw input for `..`, `%2e%2e`, `%252e%252e`
2. Expansion: Apply `os.path.expanduser()` and `os.path.expandvars()`
3. Post-validation: Re-check expanded path for `..`
4. Resolution: Convert to canonical absolute path with `path.resolve()`
5. Allowlist: Verify path is within allowed directories

**Allowed Directories** (default):
- `Path.home()` (user home directory)
- `Path.cwd()` (current working directory)
- `Path("/c/codedev/llm")` (project root)

---

## Critical Development Patterns

### Pattern 1: Adding RAG Operations

**ALWAYS use Pydantic params objects**:

```python
from rag.hybrid_rag import IngestDocumentParams

# ✅ CORRECT
params = IngestDocumentParams(
    file_path=str(document.absolute()),
    memory_type="long_term",
    chunk_size=512,
)
result = await rag_manager.ingest_document(params=params)

# ❌ WRONG - causes SyntaxError
result = await rag_manager.ingest_document(document, "long_term", 512)
```

**Available Param Models**:
- `RecallMemoriesParams` - Retrieve memories by query
- `StoreMemoryParams` - Store new memory entry
- `IngestDocumentParams` - Ingest document chunks
- `SearchParams` - Search by content and importance

### Pattern 2: CLI Command Structure

**Standard async command pattern**:

```python
@cli.command()
@click.option("--arg", help="Description")
@click.pass_context
def my_command(ctx, arg):
    """Command docstring shown in help."""
    # Parse arguments, validate
    asyncio.run(_run_my_command(ctx, arg))

async def _run_my_command(ctx, arg):
    """Async implementation."""
    # Use async/await for I/O operations
    async with aiofiles.open(file_path) as f:
        content = await f.read()
    # Process...
```

### Pattern 3: Gemma Executable Discovery

**GemmaInterface auto-discovers gemma.exe**:

Search order:
1. `GEMMA_EXECUTABLE` environment variable
2. Common build directories:
   - `../build/Release/gemma.exe`
   - `../../gemma.cpp/build/Release/gemma.exe`
3. System PATH

**Override**:
```bash
export GEMMA_EXECUTABLE=/custom/path/to/gemma.exe
```

---

## Development Commands

### Essential Commands

```bash
# Install dependencies (ALWAYS use uv)
uv pip install -e .

# Run CLI
uv run python -m gemma_cli.cli [command]

# First-time setup
uv run python -m gemma_cli.cli init

# Chat with model
uv run python -m gemma_cli.cli chat \
  --model /path/to/model.sbs \
  --tokenizer /path/to/tokenizer.spm \
  --enable-rag

# Ingest documents for RAG
uv run python -m gemma_cli.cli ingest \
  /path/to/document.txt \
  --tier long_term \
  --chunk-size 512

# Model management
uv run python -m gemma_cli.cli model detect
uv run python -m gemma_cli.cli model list

# Configuration
uv run python -m gemma_cli.cli config show
```

### Testing

```bash
# Run embedded store tests
uv run python test_embedded_store.py

# Verify standalone operation
pkill redis 2>/dev/null || echo "Redis not running"
uv run python -c "from config.settings import RedisConfig; print(f'Standalone: {RedisConfig().enable_fallback}')"

# Code quality
uv run ruff check . --fix
uv run ruff format .
uv run mypy . --ignore-missing-imports
```

---

## Known Issues & Limitations

### Current Limitations

1. **Test Coverage**: Only one test file (`test_embedded_store.py`)
   - No integration tests for core chat functionality
   - No tests for CLI commands
   - No tests for RAG system beyond embedded store

2. **MCP Integration**: Comprehensive code exists but NOT integrated
   - `mcp/` module is complete but unused in CLI
   - Needs decision: integrate fully or remove

3. **Model Management**: Detection works but can't persist
   - `model detect` finds models but doesn't save to config
   - `model download` is stubbed out

4. **Deployment**: No packaging/distribution
   - `deployment/` contains stubs only
   - No PyInstaller configuration
   - No standalone executable

5. **UI Console**: Global singleton pattern
   - `ui/console.py` uses global `Console` instance
   - Makes testing difficult
   - Needs refactoring to dependency injection

6. **Embedded Store**: Simple implementation
   - Linear search only (no vector similarity)
   - Single-process limitation
   - Keyword-based search only
   - Good for <10K documents

### Future Enhancements (from GEMINI_TODO.md)

**Short-term** (1-2 weeks):
- Simplify model configuration
- Remove MCP stubs or integrate fully
- Refactor console singleton
- Implement model config persistence

**Medium-term** (1-2 months):
- Complete model download command
- Create deployment system (PyInstaller)
- Optional Redis integration
- Comprehensive test suite

**Long-term** (3+ months):
- Rust RAG backend via MCP (high-performance)
- SQLite-VSS for better embedded search
- Advanced multi-tier memory management
- Production-grade deployment

---

## File-by-File Quick Reference

### Core Files (Must Understand)

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `cli.py` | Main entry point | `cli()` (Click group), `chat()`, `ingest()` |
| `core/gemma.py` | Subprocess management | `GemmaInterface`, `generate_response()` |
| `rag/hybrid_rag.py` | RAG high-level API | `HybridRAGManager`, Pydantic param models |
| `rag/python_backend.py` | RAG backend selection | `PythonRAGBackend.__init__()` |
| `rag/embedded_vector_store.py` | Default storage | `EmbeddedVectorStore` |
| `config/settings.py` | Configuration + security | `expand_path()`, Pydantic models |

### Secondary Files (Important)

| File | Purpose | Notes |
|------|---------|-------|
| `onboarding/wizard.py` | First-run setup | Detects Redis, creates config |
| `ui/console.py` | Rich console | Global singleton (to refactor) |
| `commands/model.py` | Model management | `detect()` works, `download()` stub |
| `mcp/client.py` | MCP implementation | Complete but not integrated |

### Documentation Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Development guide for Claude Code |
| `GEMINI_TODO.md` | Prioritized roadmap with tasks |
| `IMPLEMENTATION_SUMMARY.md` | Recent fixes summary |
| `SECURITY_FIX_PATH_TRAVERSAL.md` | Security vulnerability details |
| `STANDALONE_OPERATION.md` | User guide for standalone mode |

---

## Integration Points

### Python ↔ C++ (gemma.exe)

**Interface**: `core/gemma.py::GemmaInterface`

**Communication**:
- Subprocess via `asyncio.create_subprocess_exec()`
- Streaming tokens via stdout (8KB buffer)
- Input validation (max 50KB prompt, 10MB response)

**Security**:
- Forbidden characters: `\x00` (null bytes), `\x1b` (escape sequences)
- Path normalization before passing to subprocess

### RAG ↔ Storage

**Default Path**: `~/.gemma_cli/embedded_store.json`

**Backend Selection**:
```python
if use_embedded_store:
    self.embedded_store = EmbeddedVectorStore()
else:
    self.redis_pool = ConnectionPool(...)
```

**Data Format** (embedded store):
```json
{
  "memories": [
    {
      "id": "unique-id",
      "content": "text",
      "memory_type": "long_term",
      "importance": 0.8,
      "timestamp": "ISO-8601",
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

### CLI ↔ Configuration

**Configuration Flow**:
1. User runs `gemma-cli init` (first time)
2. `onboarding/wizard.py` creates `~/.gemma_cli/config.toml`
3. `config/settings.py` loads and validates with Pydantic
4. CLI commands access via `ctx.obj["config"]`

---

## Development Workflow

### Adding a New CLI Command

1. **Define command** in `cli.py` or `commands/*.py`:
```python
@cli.command()
@click.option("--option", help="Description")
def my_command(option):
    """Command help text."""
    asyncio.run(_run_my_command(option))
```

2. **Create async handler**:
```python
async def _run_my_command(option):
    # Implementation
    pass
```

3. **Add tests** (create if needed):
```python
# tests/test_my_command.py
import pytest

@pytest.mark.asyncio
async def test_my_command():
    # Test implementation
    pass
```

4. **Update documentation**:
- Add to `CLAUDE.md` if significant
- Update `GEMINI_TODO.md` if part of roadmap

### Modifying RAG System

1. **Understand current backend**:
   - Embedded store is default
   - Redis is optional upgrade

2. **For new storage backend**:
   - Create class in `rag/` implementing storage interface
   - Update `PythonRAGBackend.__init__()` conditional
   - Add config option in `settings.py`

3. **For new memory operations**:
   - Add Pydantic params model in `hybrid_rag.py`
   - Implement method in `PythonRAGBackend`
   - Add to `HybridRAGManager` API

### Fixing Security Issues

1. **Path operations**: ALWAYS use `expand_path()`
2. **User input**: Validate with Pydantic models
3. **Subprocess**: Use security constants from `GemmaInterface`
4. **File operations**: Check file size limits from `DocumentConfig`

---

## Troubleshooting Guide

### "Gemma executable not found"

**Cause**: `gemma.exe` not in expected locations

**Solution**:
```bash
# Option 1: Set environment variable
export GEMMA_EXECUTABLE=/path/to/gemma.exe

# Option 2: Build gemma.cpp
cd ../../gemma.cpp
cmake -B build && cmake --build build --config Release

# Option 3: Check search paths
uv run python -c "from core.gemma import GemmaInterface; GemmaInterface._find_gemma_executable()"
```

### "Path traversal detected"

**Cause**: Attempting to access files outside allowed directories

**Solution**:
- Use absolute paths within allowed directories
- Check `expand_path()` allowed_dirs list in `config/settings.py`
- Modify config to add custom allowed directories if needed

### "RAG ingest fails with TypeError"

**Cause**: Missing `params` argument to `ingest_document()`

**Solution**:
```python
# ✅ Correct way
from rag.hybrid_rag import IngestDocumentParams
params = IngestDocumentParams(file_path=path, memory_type=tier, chunk_size=size)
result = await rag_manager.ingest_document(params=params)
```

### Tests not running

**Cause**: Missing test framework dependencies

**Solution**:
```bash
uv pip install pytest pytest-asyncio
uv run pytest -v
```

---

## Next Development Tasks

Based on `GEMINI_TODO.md` and current todo list:

### Priority 1 (Immediate)

1. ✅ **Complete onboarding documentation** (this document)
2. **Integrate rag-redis Rust backend as MCP server**
   - Location: `C:\codedev\llm\rag-redis\`
   - Create Python client in `rag/rust_rag_client.py`
   - Update CLI to optionally use Rust backend
3. **Simplify model configuration**
   - Remove complex preset/profile logic
   - Prioritize `--model` CLI argument
   - Implement config persistence for detected models

### Priority 2 (Short-term)

4. **Remove/disable incomplete MCP features**
   - Decision needed: integrate or remove `mcp/` module
   - If removing: delete `mcp/` directory
   - If integrating: wire up to CLI commands
5. **Refactor ui/console.py global singleton**
   - Change to dependency injection pattern
   - Pass console instance from `cli.py` main
   - Update all components using console

### Priority 3 (Medium-term)

6. **Complete model download command**
7. **Build deployment system (PyInstaller)**
8. **Add comprehensive test suite**

---

## Resources

### Internal Documentation
- `CLAUDE.md` - Development guide
- `GEMINI_TODO.md` - Prioritized roadmap
- `IMPLEMENTATION_SUMMARY.md` - Recent changes
- `SECURITY_FIX_PATH_TRAVERSAL.md` - Security details
- `STANDALONE_OPERATION.md` - User guide

### External References
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Rich Documentation](https://rich.readthedocs.io/)

### Related Projects
- **gemma.cpp**: C++ inference engine at `../../gemma.cpp/`
- **rag-redis**: Rust RAG backend at `C:\codedev\llm\rag-redis\`

---

## Questions to Ask Before Starting

1. **What is the current priority?**
   - Rust RAG integration?
   - Model config simplification?
   - MCP integration decision?

2. **What is the target for completion?**
   - MVP functionality?
   - Production deployment?
   - Feature-complete?

3. **Are there any constraints?**
   - Redis availability?
   - Rust toolchain available?
   - External dependencies allowed?

4. **What is the testing strategy?**
   - Unit tests required?
   - Integration tests needed?
   - Manual testing acceptable?

---

## Onboarding Checklist

- [x] Understand project structure and module organization
- [x] Review recent security fixes and architectural changes
- [x] Understand RAG system with pluggable backends
- [x] Understand configuration system with Pydantic
- [x] Review critical development patterns (RAG params, path security)
- [x] Understand CLI command structure and async patterns
- [x] Review known limitations and future roadmap
- [x] Identify next priority tasks from todo list
- [x] Document everything in this onboarding file

**Status**: ✅ **Onboarding Complete** - Ready to proceed with development tasks

---

*This onboarding document should be updated as the project evolves.*
