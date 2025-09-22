# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive LLM development ecosystem combining high-performance C++ inference (gemma.cpp), Python-based AI agent framework with RAG capabilities, and Rust-powered system tools. The project features multi-runtime integration across Windows native, WSL, and cross-platform targets.

## Build and Development Commands

### C++ Gemma Engine (gemma/)

```bash
# Windows (Visual Studio 2022 - recommended)
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4

# WSL/Linux
cmake --preset make && cmake --build --preset make -j $(nproc)

# Run benchmark
./build/single_benchmark --weights /c/codedev/llm/.models/gemma2-2b-it-sfp.sbs --tokenizer /c/codedev/llm/.models/tokenizer.spm

# Convert to single-file format
./build/migrate_weights --tokenizer tokenizer.spm --weights input.sbs --output_weights output-single.sbs
```

### Python Stats/RAG System (stats/)

```bash
# Environment setup (ALWAYS use uv)
uv venv && uv sync --all-groups
uv run python -m src.gcp.gemma_download --auto  # Download models first

# Run agent
uv run python main.py --lightweight  # Fast mode
uv run python main.py --enable-planning --enable-reflection  # Full features

# Start HTTP server
uv run python -m src.server.main

# Testing
uv run pytest tests/ -v --cov=src --cov-report=term-missing
uv run pytest tests/test_react_agent.py::TestClass::test_method -vvs

# Code quality
uv run ruff check src --fix
uv run ruff format src
uv run mypy src --ignore-missing-imports
```

### Rust Components

```bash
# RAG-Redis System (stats/rag-redis-system/)
cd rag-redis-system && cargo build --release
cargo test --workspace
./target/release/rag-redis-cli --config config.toml

# Rust Extensions (stats/rust_extensions/)
uv run maturin develop --release

# MCP Filesystem Server (T:/projects/rust-commander/)
cd rust-mcp-filesystem-main && cargo build --release
./target/release/rust-mcp-filesystem --allow-write --allowed-dirs /path

# Tauri Terminal App (T:/projects/rust-term/)
npm install && npm run tauri:dev
npm run tauri:build
```

## High-Level Architecture

### Multi-Project Structure

```
C:\codedev\llm\
├── gemma/                 # C++ Gemma inference engine
│   ├── gemma.cpp/         # Core implementation
│   ├── CMakeLists.txt     # Multi-backend build config
│   └── examples/          # Usage samples
│
├── stats/                 # Python AI agent framework
│   ├── src/agent/         # ReAct agent implementation
│   ├── rag-redis-system/  # Rust RAG backend
│   ├── rust_extensions/   # Performance extensions
│   └── mcp-servers/       # MCP protocol servers
│
├── .models/               # Model weights storage
│   ├── *.sbs files        # Compressed weights
│   └── tokenizer.spm      # Tokenizer files
│
└── llm.code-workspace     # VS Code multi-root workspace

T:\projects\
├── rust-commander/        # MCP filesystem tools
└── rust-term/             # Tauri desktop terminal
```

### Core System Flow

1. **Model Loading**: gemma.cpp loads weights from `.models/` → Python agent wraps inference
2. **Agent Pipeline**: User query → ReAct reasoning → Tool execution → RAG enhancement → Response
3. **Memory System**: Redis-backed multi-tier (Working → Short-term → Long-term → Episodic → Semantic)
4. **MCP Integration**: Agent tools communicate via MCP protocol for filesystem, memory, and external services

### Key Integration Points

- **Python ↔ C++**: `gemma-cli.py` wraps native binary, handles WSL path translation
- **Python ↔ Rust**: PyO3 bindings for tokenization, SIMD vector ops, Redis backend
- **Rust ↔ Redis**: Connection pooling, vector search, memory consolidation
- **MCP Servers**: Stdio/HTTP/WebSocket transports for tool communication

## Model Management

### Supported Models

```bash
# Download models (auto-selects based on hardware)
uv run python -m src.gcp.gemma_download --auto

# Specific models
uv run python -m src.gcp.gemma_download gemma-2b-it  # 2.5GB, fastest
uv run python -m src.gcp.gemma_download gemma-7b-it  # 8.5GB, better quality
uv run python -m src.gcp.gemma_download codegemma-2b # For code tasks

# Check cached
uv run python -m src.gcp.gemma_download --list-cached
```

### Model Files Location
- **Weights**: `/c/codedev/llm/.models/*.sbs` (SFP format preferred)
- **Tokenizers**: `/c/codedev/llm/.models/*.spm`
- **Cache**: `stats/models_cache/` for Python downloads

## RAG-Redis System

### Prerequisites
```bash
# Start Redis (REQUIRED)
redis-server

# Build RAG components
cd stats/rag-redis-system && cargo build --release --features full
```

### Memory Tier Configuration
- **Working**: 10 items, immediate context
- **Short-term**: 100 items, recent interactions
- **Long-term**: 10K items, consolidated facts
- **Episodic**: Event sequences with timestamps
- **Semantic**: Graph-based concept relationships

## Testing Strategy

### Test Suites
```bash
# Unit tests (fast)
uv run pytest tests/unit/ -v --maxfail=1

# Integration tests (requires Redis)
uv run pytest tests/integration/ -v

# C++ tests
cd gemma && ctest --test-dir build

# Rust tests
cargo test --workspace --all-features

# MCP validation
npx @modelcontextprotocol/inspector --cli --config mcp.json
```

### Performance Benchmarks
```bash
# C++ inference
./gemma/build/single_benchmark --weights model.sbs

# Python profiling
uv run python -m cProfile -s cumtime main.py --lightweight

# Rust benchmarks
cargo bench --workspace
```

## Common Development Tasks

### Adding a New Tool (Python)
```python
# In src/agent/tools.py
async def my_tool(query: str) -> str:
    """Tool description for LLM"""
    return result

TOOL_REGISTRY["my_tool"] = my_tool
```

### Adding MCP Server Capability
```rust
// In mcp-wrapper/src/tools/
pub async fn handle_my_tool(params: JsonValue) -> Result<JsonValue> {
    // Implementation
    Ok(json!({"result": "data"}))
}
```

### Debugging Model Issues
```bash
# Enable debug logging
export GEMMA_LOG_LEVEL=DEBUG

# Test model loading
uv run python -c "from src.agent.gemma_agent import load_model; load_model('gemma-2b')"

# Check C++ binary
./gemma/build/gemma --help
```

## Performance Optimization

### Current Optimizations
- **Rust tokenizer**: 10x faster than Python
- **SIMD vector ops**: 5x faster similarity search
- **Memory pooling**: 67% memory reduction
- **Highway SIMD**: Multi-target CPU optimization
- **SFP compression**: 2x speed over BF16

### Hardware Acceleration
- **CPU**: Highway SIMD with runtime ISA selection
- **Memory**: NUMA-aware allocation, memory mapping
- **Planned**: SYCL (Intel GPU), CUDA, Vulkan, Metal

## Critical Configuration Files

- **CMake**: `gemma/CMakeLists.txt` - C++ build configuration
- **Python**: `stats/pyproject.toml` - Dependencies and tools
- **Rust**: `stats/Cargo.toml`, `rag-redis-system/Cargo.toml`
- **MCP**: `stats/mcp.json` - Server configurations
- **VS Code**: `llm.code-workspace` - Multi-root workspace

## Important Notes

- **ALWAYS use `uv run`** for Python commands (ensures correct environment)
- **Download models first** before running inference (2-7GB per model)
- **Start Redis** before using RAG features
- **Use lightweight mode** for development (faster iteration)
- **Path translation**: Windows paths auto-convert to WSL format when needed
- **Griffin issues**: May have Windows linking problems, fallback available

## External Project Integration

### rust-commander (T:\projects\rust-commander)
- High-performance MCP filesystem server
- Directory permission management
- Archive operations (zip/unzip)

### rust-term (T:\projects\rust-term)
- Tauri-based WSL terminal GUI
- xterm.js frontend integration
- Pseudo-terminal management

Both projects integrate via the multi-root workspace for unified development.

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
