# RAG-Redis Project

This project provides a high-performance Retrieval-Augmented Generation (RAG) system built in Rust. It uses Redis as a backend for storing and retrieving documents and their vector embeddings.

## Features

*   **High-Performance RAG**: A SIMD-optimized vector store for fast similarity search.
*   **Redis Backend**: A scalable and persistent storage for documents and embeddings.
*   **Python Bridge**: A Python interface for interacting with the RAG system, exposed as an MCP server.
*   **Multi-tier Memory**: A sophisticated memory system with short-term, long-term, episodic, semantic, and working memory.

## Getting Started

### Prerequisites

*   Rust toolchain
*   Redis server

### Building the `rag-redis` system

To build the `rag-redis` system, run the following command from the `rag-redis` directory:

```bash
cargo build --release
```

This will create the necessary binaries in the `target/release` directory.

### Running the `rag-redis` MCP server

The `rag-redis` system is designed to be run as an MCP server, which is used by the `stats` agent. The `stats` agent is already configured to launch the MCP server automatically.

The Python bridge for the MCP server is located in the `python-bridge` directory. To install its dependencies, run:

```bash
cd C:\codedev\llm\rag-redis\python-bridge
uv pip install -e .
```

## Integration with the `stats` agent

The `stats` agent is configured to use the `rag-redis` system through the MCP server. The configuration is in the `stats/mcp.json` file.

When the `stats` agent starts, it will automatically launch the `rag-redis` MCP server. The agent can then use the RAG system for document ingestion, search, and memory management.

## Documentation for LLM Agents

This `README.md` file, along with the other documentation in this workspace, is intended to be used by LLM agents to understand, build, and debug the system.
