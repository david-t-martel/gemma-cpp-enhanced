# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local LLM CLI/HTTP chatbot framework using Google Gemma/Phi-2 models with PyTorch, featuring:
- ReAct (Reasoning and Acting) agent with planning capabilities
- Tool calling system (8+ built-in tools)
- High-performance Rust extensions via PyO3
- RAG-Redis system with multi-tier memory
- FastAPI HTTP server with WebSocket support
- SIMD-optimized vector operations

## Build and Development Commands

### Quick Start
```bash
# Download model first (REQUIRED, 2-7GB)
uv run python -m src.gcp.gemma_download --auto  # Auto-selects based on hardware

# Run interactive chat
uv run python main.py --lightweight  # Fast mode with smaller model
uv run python main.py                 # Full model

# Start HTTP server
uv run python -m src.server.main
```

### Testing
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_react_agent.py -v

# Run single test
uv run pytest tests/test_agent.py::TestAgentCore::test_tool_execution -vvs

# Fast unit tests only
uv run pytest tests/unit/ -v --maxfail=1

# Rust tests
cargo test --workspace
```

### Code Quality
```bash
# Format and lint Python
uv run ruff check src --fix
uv run ruff format src

# Type checking
uv run mypy src --ignore-missing-imports

# Rust formatting and linting
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Run all checks (recommended before commit)
make lint format type-check
```

### Building
```bash
# Build Rust extensions (required for performance features)
uv run maturin develop --release

# Build Python package
uv build

# Full build (Python + Rust)
make build-all
```

## High-Level Architecture

### Core Agent Flow
The ReAct agent follows a structured reasoning loop:

```
User Query → Planning → Thought → Action → Observation → Reflection → Response
           ↑                                                            ↓
           └────────────────── Loop if needed ─────────────────────────┘
```

**Key Components:**
- **Agent Core** (`src/agent/core.py`): Base agent with tool calling capability
- **ReAct Agent** (`src/agent/react_agent.py`): Implements reasoning loop
- **Tool Registry** (`src/agent/tools.py`): Manages tool discovery and execution
- **Planner** (`src/agent/planner.py`): Breaks complex tasks into steps

### Memory Architecture
The RAG-Redis system provides multi-tier memory:

```
Working Memory (immediate context, 10 items)
    ↓
Short-term Memory (recent interactions, 100 items)
    ↓
Long-term Memory (important facts, 10K items)
    ↓
Episodic Memory (event sequences with timestamps)
    ↓
Semantic Memory (concepts and relationships)
```

All backed by Redis with SIMD-optimized vector search.

### Server Architecture
FastAPI server with multiple interfaces:
- REST API endpoints (`/api/v1/*`)
- WebSocket for streaming (`/ws`)
- Health checks and metrics (`/health`, `/metrics`)

## Model Management

### Download Models
```bash
# Auto-select based on hardware
uv run python -m src.gcp.gemma_download --auto

# For code tasks
uv run python -m src.gcp.gemma_download --auto --prefer-code

# Specific model
uv run python -m src.gcp.gemma_download gemma-2b-it

# Check cached models
uv run python -m src.gcp.gemma_download --list-cached
```

### Model Selection Logic
- **<4GB VRAM**: gemma-2b (lightweight)
- **4-8GB VRAM**: gemma-2b or quantized gemma-7b
- **>8GB VRAM**: gemma-7b (full precision)
- **CPU only**: gemma-2b recommended

## RAG-Redis System

### Prerequisites
```bash
# Start Redis (REQUIRED)
redis-server

# Build RAG system
cd rag-redis-system && cargo build --release
```

### Using RAG in Code
```python
from src.agent.rag_integration import enhance_agent_with_rag

# Enhance agent with RAG
agent = create_react_agent(lightweight=True)
enhanced_agent = await enhance_agent_with_rag(agent)

# Agent now has access to:
# - Document storage and retrieval
# - Multi-tier memory system
# - Semantic search capabilities
```

## Common Development Tasks

### Adding a New Tool
1. Define in `src/agent/tools.py`:
```python
async def my_tool(query: str) -> str:
    """Tool description for LLM"""
    # Implementation
    return result

TOOL_REGISTRY["my_tool"] = my_tool
```

2. Test: `uv run pytest tests/test_tools.py::test_my_tool`

### Running with Different Models
```bash
# Lightweight (fast, less accurate)
uv run python main.py --lightweight

# With 8-bit quantization (saves memory)
uv run python main.py --8bit

# Specific model
uv run python main.py --model google/gemma-7b-it
```

### Debugging Issues
```bash
# Enable debug logging
export GEMMA_LOG_LEVEL=DEBUG
uv run python main.py --verbose

# Check model loading
uv run python -c "from src.agent.gemma_agent import load_model; load_model('gemma-2b')"

# Test tool execution
uv run python test_agent.py
```

## Performance Optimization

### Current Optimizations
- Rust tokenizer: 10x faster than Python
- SIMD vector ops: 5x faster similarity search
- Memory pooling: 67% memory reduction
- Lazy model loading: 75% faster startup

### Monitoring Performance
```bash
# Run benchmarks
cargo bench --workspace

# Profile Python code
uv run python -m cProfile -s cumtime main.py --lightweight

# Memory profiling
uv run python -m memory_profiler main.py
```

## Important Notes

- **Always use `uv run`**: Ensures correct Python environment
- **Download models first**: Required before running (2-7GB)
- **Start Redis**: Required for RAG features
- **Use lightweight mode**: For development/testing (faster)
- **Check TODO.md**: For known issues and roadmap
