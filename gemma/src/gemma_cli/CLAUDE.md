# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gemma CLI is a Python-based terminal interface for Google's Gemma LLM models. It wraps the C++ `gemma.exe` inference engine with a feature-rich CLI, RAG (Retrieval-Augmented Generation) memory system, and optional Redis integration. The project prioritizes standalone operation without external dependencies while offering advanced features for power users.

**Current Status**: Core functionality is stable and secure. Recent critical fixes (January 2025) addressed security vulnerabilities, syntax errors, and dependency issues. See `IMPLEMENTATION_SUMMARY.md` for details.

## Build and Development Commands

### Environment Setup

**CRITICAL**: Always use `uv` for Python operations, never bare `python` or `pip`:

```bash
# Install dependencies
uv pip install -e .

# Run CLI
uv run python -m gemma_cli.cli [command]

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_specific.py::TestClass::test_method -vvs

# Code quality
uv run ruff check src/gemma_cli --fix
uv run ruff format src/gemma_cli
uv run mypy src/gemma_cli --ignore-missing-imports
```

### CLI Commands

```bash
# First-time setup (interactive wizard)
uv run python -m gemma_cli.cli init

# Chat with model (default command)
uv run python -m gemma_cli.cli chat --model /path/to/model.sbs --tokenizer /path/to/tokenizer.spm

# Enable RAG for context enhancement
uv run python -m gemma_cli.cli chat --enable-rag

# Ingest documents for RAG
uv run python -m gemma_cli.cli ingest /path/to/document.txt --tier long_term

# Model management
uv run python -m gemma_cli.cli model detect  # Find models on system
uv run python -m gemma_cli.cli model list    # Show configured models

# Configuration
uv run python -m gemma_cli.cli config show   # Display current config
uv run python -m gemma_cli.cli config edit   # Edit config interactively
```

### Testing Standalone Operation

```bash
# Verify works without Redis
pkill redis 2>/dev/null || echo "Redis not running"
uv run python src/gemma_cli/test_embedded_store.py

# Quick configuration check
uv run python -c "from src.gemma_cli.config.settings import RedisConfig; print(f'Standalone: {RedisConfig().enable_fallback}')"
```

## High-Level Architecture

### Core Flow: User → CLI → RAG → Gemma.exe

```
┌─────────┐
│  User   │
└────┬────┘
     │
┌────▼────────────────────────────────────────────────────┐
│  cli.py (Click-based command dispatcher)                │
│  • Entry point for all commands                         │
│  • Context setup, first-run detection                   │
│  • Command routing to specialized modules                │
└────┬────────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────────┐
│  core/gemma.py (GemmaInterface)                         │
│  • Subprocess management for gemma.exe                  │
│  • Streaming token generation via stdout                │
│  • Security: input validation, size limits              │
└────┬────────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────────┐
│  gemma.exe (C++ inference engine)                       │
│  • Native Windows binary                                │
│  • Loads .sbs model weights                             │
│  • Generates tokens via SIMD-optimized ops              │
└─────────────────────────────────────────────────────────┘
```

### RAG System Architecture (Pluggable Backends)

The RAG system uses a **Strategy Pattern** with pluggable storage backends:

```
HybridRAGManager (rag/hybrid_rag.py)
        │
        ├─> PythonRAGBackend (rag/python_backend.py)
        │         │
        │         ├─> EmbeddedVectorStore (default) [rag/embedded_vector_store.py]
        │         │   • File-based (JSON), no dependencies
        │         │   • Simple keyword search
        │         │   • Ideal for <10K documents
        │         │
        │         └─> RedisVectorStore (optional) [rag/python_backend.py]
        │             • High-performance vector search
        │             • Requires external Redis server
        │             • Recommended for >10K documents
        │
        └─> (Future) RustRAGClient [rag/rust_rag_client.py]
            • SIMD-optimized Rust backend via MCP
            • 5-tier memory system
            • Advanced semantic search
```

**Key Architectural Decision**: The embedded store is **default** (standalone operation), Redis is an **optional upgrade** for scale.

### Configuration System (Pydantic Models)

Located in `config/settings.py`, using strong typing with Pydantic:

```python
AppConfig
├─> ModelConfig      # Model paths, parameters
├─> RedisConfig      # enable_fallback=True (default), connection settings
├─> RAGConfig        # Chunking, embedding, retrieval settings
├─> UIConfig         # Theme, display preferences
└─> MonitoringConfig # Performance tracking (future)
```

**Security Note**: All path expansion **must** use `expand_path()` function (defense-in-depth validation against path traversal attacks).

### Command Structure (Click Groups)

```
cli.py
├─> chat                    # Primary: interactive conversation
├─> ingest                  # RAG: document ingestion
├─> model (group)
│   ├─> detect             # Auto-discover models
│   ├─> list               # Show configured models
│   └─> download           # (Planned) Fetch models
├─> profile (group)
│   ├─> create             # Save model presets
│   └─> use                # Switch active profile
├─> config (group)
│   ├─> show               # Display current config
│   └─> edit               # Interactive editor
└─> init                   # First-run setup wizard
```

## Critical Implementation Details

### Security: Path Validation

**All user-provided paths must be validated through `config/settings.py::expand_path()`:**

```python
from config.settings import expand_path

# ✅ CORRECT
safe_path = expand_path(user_input)

# ❌ NEVER DO THIS
unsafe_path = Path(os.path.expandvars(user_input))  # Vulnerable!
```

The `expand_path()` function implements 5 layers of validation:
1. Pre-expansion check for `..` and URL-encoded variants
2. Environment variable/tilde expansion
3. Post-expansion re-validation
4. Path resolution (canonical form)
5. Allowlist enforcement (home, cwd, project directories)

### RAG: Document Ingestion Pattern

When calling RAG manager methods, **always use Pydantic params objects**:

```python
from rag.hybrid_rag import IngestDocumentParams

# ✅ CORRECT
params = IngestDocumentParams(
    file_path=str(document.absolute()),
    memory_type=tier,  # working/short_term/long_term/episodic/semantic
    chunk_size=chunk_size,
)
result = await rag_manager.ingest_document(params=params)

# ❌ WRONG - will cause SyntaxError
result = await rag_manager.ingest_document(document, tier, chunk_size)
```

### Gemma.exe: Finding the Executable

The `GemmaInterface` auto-discovers the binary in this order:
1. `GEMMA_EXECUTABLE` environment variable
2. Common build directories: `../build/Release/`, `../../gemma.cpp/build/Release/`
3. System PATH

**Override via environment**:
```bash
export GEMMA_EXECUTABLE=/custom/path/to/gemma.exe
```

### Console Dependency Injection (NEW)

The CLI uses **dependency injection** for console instances to improve testability and avoid global state:

```python
# ✅ RECOMMENDED: Use console from Click context
from gemma_cli.ui.console import create_console

@cli.command()
@click.pass_context
def my_command(ctx: click.Context):
    console = ctx.obj["console"]  # Get injected console
    console.print("[green]Success![/green]")

# For widgets/classes, accept console as parameter
from rich.console import Console

class MyWidget:
    def __init__(self, console: Console | None = None):
        from gemma_cli.ui.console import create_console
        self.console = console or create_console()  # Fallback for compatibility

    def display(self):
        self.console.print("Widget output")

# Creating instances
console = create_console()
widget = MyWidget(console=console)
wizard = OnboardingWizard(console=console)
```

**Key Points**:
- CLI creates console in `cli()` group and injects into `ctx.obj["console"]`
- Commands retrieve console from context
- Widgets accept optional console parameter with fallback
- Use `create_console()` factory, **NOT** `get_console()` (deprecated)
- See `CONSOLE_DI_REFACTOR_SUMMARY.md` for complete documentation

### Async Patterns

The CLI uses `asyncio` for subprocess communication and RAG operations:

```python
# Entry point pattern (in cli.py commands)
@cli.command()
@click.pass_context
def my_command(ctx):
    # ... click argument parsing ...
    asyncio.run(_run_my_command(ctx, args))

async def _run_my_command(ctx, args):
    # Get console from context
    console = ctx.obj["console"]

    # Async implementation here
    async with aiofiles.open(...) as f:
        content = await f.read()
```

## Module Responsibilities

### `core/`
- **`gemma.py`**: Subprocess management, token streaming, input validation
- **`conversation.py`**: Multi-turn conversation management, context tracking

### `rag/`
- **`hybrid_rag.py`**: High-level RAG API, backend selection
- **`python_backend.py`**: Backend logic, storage selection (Redis vs Embedded)
- **`embedded_vector_store.py`**: Standalone file-based storage (default)
- **`memory.py`**: Memory tier abstraction
- **`optimizations.py`**: Batch embedding, consolidation (Phase 2 feature)

### `config/`
- **`settings.py`**: Pydantic models, configuration loading, **secure path expansion**
- **`models.py`**: Model preset management (to be simplified)

### `ui/`
- **`console.py`**: Rich console setup (currently global singleton - planned refactor)
- **`formatters.py`**: Message formatting, syntax highlighting
- **`components.py`**: Reusable UI components (progress bars, panels)

### `onboarding/`
- **`wizard.py`**: Interactive first-run setup, model discovery, config generation

### `mcp/` (Incomplete - Planned for Removal/Refactor)
- Model Context Protocol integration (currently non-functional stubs)
- Decision pending: implement fully or remove in favor of simpler tool system

## Important Development Patterns

### Adding a New CLI Command

1. Define command in `cli.py` or appropriate `commands/*.py` file
2. Use Click decorators for arguments/options
3. Create async handler function (`_run_*` naming convention)
4. Call via `asyncio.run()` from Click command
5. Update tests in `tests/commands/test_*.py`

### Extending RAG with New Storage Backend

1. Create new class implementing storage interface (see `EmbeddedVectorStore` as template)
2. Update `PythonRAGBackend.__init__` conditional logic
3. Add configuration option in `config/settings.py::RedisConfig`
4. Document in `STANDALONE_OPERATION.md`

### Modifying Configuration Schema

1. Update Pydantic models in `config/settings.py`
2. Increment `CONFIG_VERSION` constant
3. Add migration logic in `config/settings.py::load_config()`
4. Update onboarding wizard questions in `onboarding/wizard.py`
5. Regenerate config with `uv run python -m gemma_cli.cli init --reset`

## Known Limitations & Future Work

### Current Limitations
- **Embedded vector store**: Linear search, single-process, <10K docs recommended
- **Model management**: Detection works, but can't persist to config yet
- **MCP integration**: Non-functional stubs (removal pending)
- **Deployment**: No packaging/distribution system (PyInstaller planned)

### Roadmap (see `GEMINI_TODO.md`)
- **Short-term**: Simplify model config, remove MCP stubs, refactor console singleton
- **Medium-term**: Model download, deployment (standalone .exe), optional Rust RAG backend
- **Long-term**: Advanced multi-tier memory, production-grade RAG with SQLite-VSS

## Configuration Files

- **`~/.gemma_cli/config.toml`**: User configuration (auto-created by init)
- **`~/.gemma_cli/embedded_store.json`**: RAG vector store (when using embedded backend)
- **`~/.gemma_cli/conversations/`**: Saved conversation history (future)

## Environment Variables

- **`GEMMA_EXECUTABLE`**: Override gemma.exe location
- **`GEMMA_LOG_LEVEL`**: Set logging level (DEBUG, INFO, WARNING, ERROR)
- **`REDIS_URL`**: Redis connection string (optional, for advanced RAG)

## Troubleshooting

### "Gemma executable not found"
1. Check `GEMMA_EXECUTABLE` env var
2. Build gemma.cpp: `cd ../../gemma.cpp && cmake -B build && cmake --build build --config Release`
3. Verify binary exists: `ls ../../gemma.cpp/build/Release/gemma.exe`

### "Path traversal detected"
- Attempting to access files outside allowed directories (home, project root, cwd)
- Solution: Use absolute paths or ensure files are in valid locations

### "RAG ingest fails with TypeError"
- Missing `params` argument to `ingest_document()`
- Solution: Create `IngestDocumentParams` object first (see "RAG: Document Ingestion Pattern" above)

## Related Documentation

- **`IMPLEMENTATION_SUMMARY.md`**: Recent fixes and current project state
- **`GEMINI_TODO.md`**: Prioritized roadmap and known issues
- **`STANDALONE_OPERATION.md`**: User guide for Redis-free operation
- **`SECURITY_FIX_PATH_TRAVERSAL.md`**: Security vulnerability details and fix
- **`RAG_REDIS_INTEGRATION_PLAN.md`**: Future Rust backend integration strategy
