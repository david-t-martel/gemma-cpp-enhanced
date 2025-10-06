# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-Redis is a high-performance Retrieval-Augmented Generation (RAG) system combining Rust's performance with Redis-backed multi-tier memory management. The system features SIMD-optimized vector search, intelligent document processing, and a Model Context Protocol (MCP) bridge for integration with AI assistants.

## Build and Development Commands

### Rust Build Commands

```bash
# Standard build
cargo build --release

# Build with all optimizations
cargo build --release --features full

# Build Python extensions
cd rust-rag-extensions && maturin develop --release

# Build MCP server binary
cd rag-redis-system/mcp-server && cargo build --release

# Memory-optimized build for Windows
cargo build --profile release-memory-optimized

# Run benchmarks
cargo bench --workspace

# Run all tests
cargo test --workspace --all-features

# Format code
cargo fmt --all

# Lint code
cargo clippy --all-targets --all-features -- -D warnings
```

### Native Rust MCP Server Commands

```bash
# Build MCP server (from workspace root)
cargo build --release --package rag-redis-mcp-server --bin mcp-server

# Build MCP server (standalone)
cd rag-redis-system/mcp-server && cargo build --release

# Run MCP server
./target/release/mcp-server --redis-url redis://127.0.0.1:6380

# Test MCP server
cargo test --bin mcp-server

# Validate MCP configuration
npx @modelcontextprotocol/inspector --cli
```

### Binary Locations

All compiled binaries are output to the shared target directory:

```bash
# Windows
C:\Users\david\.cargo\shared-target\release\mcp-server.exe
C:\Users\david\.cargo\shared-target\release\rag-cli.exe

# Linux/WSL
~/.cargo/shared-target/release/mcp-server
~/.cargo/shared-target/release/rag-cli
```

### Deployment Commands

```bash
# Docker deployment (recommended for production)
docker-compose up -d

# Docker with monitoring stack
docker-compose --profile monitoring up -d

# Linux systemd deployment
sudo ./deploy/scripts/deploy-linux.sh --install-service --start-service

# Windows service deployment
.\deploy\scripts\deploy-windows.ps1 -InstallService -StartService

# Environment configuration
./deploy/scripts/setup-environment.sh setup production

# Health check
./deploy/scripts/health-check.sh
```

### Redis Operations

```bash
# Start Redis (Windows port 6380, WSL/Linux port 6379)
redis-server --port 6380

# Check Redis connection
redis-cli -p 6380 ping

# Monitor Redis operations
redis-cli -p 6380 monitor
```

## High-Level Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     MCP Interface Layer                   │
│               Native Rust MCP Server                     │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  Core RAG System (Rust)                  │
├──────────────────────────────────────────────────────────┤
│  • RagSystem orchestrator (lib.rs)                       │
│  • Multi-tier memory management (memory.rs - 1,313 LOC)  │
│  • SIMD vector operations (vector_store.rs)              │
│  • Document processing pipeline (document.rs)            │
│  • Embedding providers (embedding.rs)                    │
│  • Research integration (research.rs)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                    Redis Backend                         │
├──────────────────────────────────────────────────────────┤
│  • Connection pooling (bb8-redis)                        │
│  • 5-tier memory system                                  │
│  • Binary serialization (bincode)                        │
│  • Automatic compression (>1KB threshold)                │
│  • Fallback to in-memory storage                         │
└──────────────────────────────────────────────────────────┘
```

### Memory Tier System

| Tier | Capacity | TTL | Storage | Use Case |
|------|----------|-----|---------|----------|
| Working | 100 items | 15 min | Local cache | Immediate context |
| ShortTerm | 1,000 items | 1 hour | Redis | Recent interactions |
| LongTerm | 10,000 items | 30 days | Redis | Consolidated knowledge |
| Episodic | 5,000 items | 7 days | Redis | Temporal sequences |
| Semantic | 50,000 items | No TTL | Redis | Knowledge graph |

### Workspace Structure

```
rag-redis/
├── rag-redis-system/        # Core RAG implementation
│   ├── src/                 # Main library code
│   ├── mcp-server/          # MCP server binary
│   ├── benches/             # Performance benchmarks
│   └── tests/               # Integration tests
│
├── rag-binaries/            # CLI and server binaries
│   └── src/bin/
│       ├── cli.rs           # Command-line interface
│       └── server.rs        # HTTP server
│
├── rust-rag-extensions/     # Python bindings (PyO3)
│   └── src/lib.rs           # C library exports
│
├── .archive/                # Archived code (deprecated Python bridge)
│   └── python-mcp-bridge/   # Archived Python MCP implementation
│
├── models/                  # Embedding models
│   └── all-MiniLM-L6-v2/    # Default embedding model
│
└── Cargo.toml               # Workspace configuration
```

## Key Dependencies and Features

### Core Rust Dependencies
- **tokio**: Async runtime with full features
- **redis + bb8-redis**: Async Redis with connection pooling
- **simsimd**: SIMD-optimized vector operations (5x speedup)
- **hnsw**: Hierarchical Navigable Small World indexing
- **candle-core**: Rust ML framework for local embeddings
- **parking_lot**: Fast synchronization primitives
- **dashmap**: Concurrent HashMap operations

### Optional Features (Cargo features)
- `full`: All features enabled
- `simsimd`: SIMD optimizations (recommended)
- `faiss`: Facebook AI Similarity Search
- `tantivy`: Full-text search engine
- `pdf`: PDF document extraction
- `wonnx`: ONNX runtime support

## Performance Optimizations

### Build Profiles
```toml
# Standard release
[profile.release]
opt-level = 3
lto = true
codegen-units = 1

# Memory-optimized (Windows)
[profile.release-memory-optimized]
opt-level = "s"
strip = true
lto = "fat"

# Python extensions
[profile.release-python]
opt-level = 3
lto = "thin"
```

### SIMD Vector Operations
The system automatically detects and uses SIMD instructions when available:
- AVX2/AVX512 on x86_64
- NEON on ARM
- Automatic fallback to scalar operations

### Redis Optimizations
- Connection pooling with configurable size
- Binary serialization with bincode
- Automatic compression for content >1KB
- Batch operations for bulk inserts
- TTL-based automatic expiration

## Testing Commands

### Rust Tests
```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration_test

# Specific test
cargo test test_name --exact

# With output
cargo test -- --nocapture

# Benchmarks
cargo bench --bench memory_benchmark
```

### MCP Server Tests
```bash
# MCP server tests
cd rag-redis-system/mcp-server && cargo test

# Integration tests
cargo test --test mcp_integration

# Validate MCP protocol
npx @modelcontextprotocol/inspector --cli
```

## Common Development Tasks

### Adding a New MCP Tool

**Rust-native MCP**:
```rust
// In rag-redis-system/mcp-server/src/tools/
pub async fn handle_new_feature(params: Value) -> Result<Value> {
    let system = RagSystem::new(config).await?;
    // Implementation
    Ok(json!({"result": data}))
}
```

### Adding Memory Operations
```rust
// In rag-redis-system/src/memory.rs
pub async fn custom_memory_operation(&self, key: &str) -> Result<Memory> {
    let memory = Memory {
        id: Uuid::new_v4().to_string(),
        content: content.to_string(),
        memory_type: MemoryType::Custom,
        importance: 0.5,
        timestamp: Utc::now(),
        // ...
    };
    self.store_memory(&memory).await?;
    Ok(memory)
}
```

### Debugging Redis Issues
```bash
# Check Redis connection
redis-cli -p 6380 ping

# Monitor operations
redis-cli -p 6380 monitor

# Check memory usage
redis-cli -p 6380 info memory

# Clear all data (CAUTION)
redis-cli -p 6380 flushdb
```

## Environment Variables

```bash
# Redis configuration
REDIS_URL=redis://127.0.0.1:6380
REDIS_MAX_CONNECTIONS=10
REDIS_CONNECTION_TIMEOUT=5

# Data directories
RAG_DATA_DIR=C:/codedev/llm/rag-redis/data/rag
EMBEDDING_CACHE_DIR=C:/codedev/llm/rag-redis/cache/embeddings
LOG_DIR=C:/codedev/llm/rag-redis/logs

# Performance tuning
VECTOR_BATCH_SIZE=100
MEMORY_CONSOLIDATION_INTERVAL=300
EMBEDDING_MODEL=all-MiniLM-L6-v2

# MCP server configuration
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8080
```

## MCP Tools Available

| Tool | Description | Parameters |
|------|-------------|------------|
| `ingest_document` | Ingest documents with metadata | content, metadata, model |
| `search` | Semantic vector search | query, limit, threshold |
| `hybrid_search` | Combined vector + keyword search | query, limit, keyword_weight |
| `research` | Web research integration | query, sources, max_results |
| `memory_store` | Store in agent memory | content, memory_type, importance |
| `memory_recall` | Recall from memory | query, memory_type, limit |
| `health_check` | System health status | include_metrics |

## Troubleshooting

### Redis Connection Issues
```bash
# Windows: Use port 6380
redis-server --port 6380

# WSL/Linux: Default port 6379
redis-server

# Check if Redis is running
netstat -an | findstr 6380  # Windows
netstat -an | grep 6379      # Linux
```

### Build Issues
```bash
# Clean build
cargo clean && cargo build --release

# Update dependencies
cargo update

# Check for missing features
cargo build --all-features
```

### Memory Issues
```bash
# Use memory-optimized profile
cargo build --profile release-memory-optimized

# Reduce batch sizes
export VECTOR_BATCH_SIZE=50

# Clear Redis cache
redis-cli -p 6380 flushdb
```

## Performance Benchmarks

Expected performance metrics:
- Vector search: ~1ms for 10K vectors (SIMD)
- Document ingestion: ~100ms per document
- Memory operations: ~5ms per operation
- Redis operations: ~2ms roundtrip
- Embedding generation: ~50ms (local model)

## Native Rust MCP Server

The system uses a native Rust MCP server implementation:
1. Direct MCP protocol implementation in Rust using rust-mcp-schema
2. Single binary deployment with no dependencies
3. Benefits: Lower latency, reduced memory usage, type-safe protocol handling
4. Archived Python bridge: Previously used Python bridge (now archived in .archive/python-mcp-bridge/)

## Deployment Framework

### Quick Deployment

```bash
# Production deployment with Docker
docker-compose -f docker-compose.yml up -d

# Development deployment
cargo build --release && ./target/release/mcp-server

# Windows service installation
powershell -ExecutionPolicy Bypass .\deploy\scripts\deploy-windows.ps1 -InstallService

# Linux systemd installation
sudo ./deploy/scripts/deploy-linux.sh --install-service --start-service
```

### Deployment Files Structure

```
deploy/
├── docker/
│   ├── Dockerfile           # Multi-stage Docker build
│   └── docker-compose.yml   # Full stack deployment
├── systemd/
│   └── rag-redis-mcp-server.service  # Linux service
├── windows/
│   └── install-service.ps1  # Windows service installer
├── scripts/
│   ├── deploy-linux.sh      # Linux deployment automation
│   ├── deploy-windows.ps1   # Windows deployment automation
│   ├── health-check.sh      # Health monitoring
│   └── setup-environment.sh # Environment management
├── config/
│   └── environments/        # Per-environment configs
└── monitoring/
    ├── prometheus.yml       # Metrics collection
    └── alert_rules.yml      # Alert definitions
```

## Rust Build Best Practices

### Optimized Build Profiles

```toml
# Cargo.toml profiles
[profile.release]
opt-level = 3          # Maximum optimization
lto = true            # Link-time optimization
codegen-units = 1     # Single codegen unit for best optimization
strip = true          # Strip symbols for smaller binary

[profile.release-memory-optimized]
opt-level = "s"       # Optimize for size
strip = true
lto = "fat"
panic = "abort"       # Smaller panic handler

[profile.dev-fast]
opt-level = 1         # Some optimization for dev
debug = 1             # Minimal debug info
```

### Build Caching with sccache

```bash
# Install sccache
cargo install sccache

# Configure for Rust builds
export RUSTC_WRAPPER=sccache
export SCCACHE_DIR=~/.cache/sccache

# Build with caching
cargo build --release

# Check cache stats
sccache --show-stats
```

### Cross-Platform Building

```bash
# Windows target from WSL/Linux
cargo build --release --target x86_64-pc-windows-msvc

# Linux target from Windows
cargo build --release --target x86_64-unknown-linux-gnu

# Static linking for portability
RUSTFLAGS='-C target-feature=+crt-static' cargo build --release
```

## Important Notes

- **ALWAYS use `cargo build --release`** for production builds
- **Binary locations**: Check `~/.cargo/shared-target/release/` or `C:\Users\david\.cargo\shared-target\release\`
- **Use native Rust MCP server** for best performance (no Python bridge)
- **Start Redis** before any RAG operations (port 6380 on Windows, 6379 on Linux)
- **SIMD features** automatically detected at runtime for optimal performance
- **Fallback to in-memory** when Redis unavailable (configurable via REDIS_ENABLE_FALLBACK)
- **Binary serialization** with bincode for performance
- **Compression threshold** at 1KB for storage efficiency
- **Health checks** available at `/health` endpoint or via `health_check` MCP tool
- **Use Docker deployment** for production environments for consistency

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
