# RAG Redis MCP Server - Python Bridge

This is a Python bridge to the high-performance Rust RAG Redis implementation, providing MCP (Model Context Protocol) compatibility.

## Overview

The Python bridge acts as an interface between MCP clients and the Rust-based RAG Redis system, providing:

- Document ingestion with vector embeddings
- Semantic search capabilities
- Hybrid search (vector + keyword)
- Memory management system
- Research capabilities with external sources
- Health monitoring and metrics

## Installation

```bash
# Install in the python-bridge directory
cd C:/codedev/llm/rag-redis/python-bridge
uv pip install -e .
```

## Usage

The server is configured to run via the MCP configuration in `C:/codedev/llm/stats/mcp.json`:

```json
{
  "mcpServers": {
    "rag-redis": {
      "command": "uv",
      "args": ["run", "python", "-m", "rag_redis_mcp.mcp_main", "--redis-url", "${env:REDIS_URL}", "--log-level", "error"],
      "cwd": "C:/codedev/llm/rag-redis/python-bridge"
    }
  }
}
```

## Environment Variables

- `REDIS_URL`: Redis connection URL (default: redis://127.0.0.1:6380)
- `RAG_DATA_DIR`: Data directory for RAG storage
- `EMBEDDING_CACHE_DIR`: Cache directory for embeddings
- `RUST_BINARY_PATH`: Path to the Rust MCP server binary

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │ ──▶│  Python Bridge   │ ──▶│  Rust Backend   │
│                 │    │                  │    │                 │
│ - Claude Code   │    │ - Tool routing   │    │ - Vector search │
│ - Other clients │    │ - JSON-RPC       │    │ - Redis ops     │
│                 │    │ - Error handling │    │ - Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Available Tools

- `ingest_document`: Ingest documents with metadata
- `search`: Semantic search across documents
- `hybrid_search`: Combined vector and keyword search
- `research`: Research with external sources
- `memory_store`: Store information in agent memory
- `memory_recall`: Recall stored memories
- `health_check`: System health monitoring

## Configuration

The system uses Redis as the vector database and supports:

- Multiple embedding models
- Configurable similarity thresholds
- Memory management with different types
- Caching for performance
- Comprehensive logging

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
ruff check . --fix
```