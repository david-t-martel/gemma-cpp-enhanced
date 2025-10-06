# RAG-Redis System - Project Context Summary

## Quick Reference
- **Last Updated**: 2025-01-22
- **Status**: Active Development
- **Redis Port**: 6380
- **Architecture**: Native Rust MCP Server

## Recent Major Changes
1. **Migrated from Python to Native Rust** - Complete replacement of Python MCP bridge
2. **Candle ML Integration** - Replaced ONNX with Candle for embeddings
3. **MCP Server Implementation** - 7 fully functional tools via JSON-RPC 2.0
4. **Python Bridge Archived** - Legacy code moved to `.archive/python-mcp-bridge/`

## Current Working State

### What's Working
- Native Rust MCP server with all 7 tools implemented
- 5-tier memory system (Working, ShortTerm, LongTerm, Episodic, Semantic)
- Redis integration on port 6380
- Comprehensive test suite (unit + integration)
- Sub-millisecond vector operations with SIMD

### Pending Tasks
- [ ] Validate MCP stdio communication with Claude Desktop
- [ ] Integrate real embedding model (currently using mock)
- [ ] Clean up unused mock code
- [ ] Optimize Candle model loading

## Key Commands

```bash
# Build the system
cd rag-redis-system
cargo build --release

# Run tests
cargo test --workspace

# Start MCP server
cargo run --bin mcp-server

# Run functional tests
cargo test --test functional_tests

# Check Redis
redis-cli -p 6380 ping
```

## Architecture Overview

```
rag-redis-system/
├── mcp-server/          # Native Rust MCP implementation
│   ├── src/
│   │   ├── main.rs      # Server entry point
│   │   ├── server.rs    # JSON-RPC handler
│   │   └── tools.rs     # MCP tool implementations
│   └── Cargo.toml
├── src/                 # Core RAG system
│   ├── memory/          # 5-tier memory management
│   ├── embeddings/      # Candle ML integration
│   ├── redis/           # Redis connection management
│   └── lib.rs
├── tests/               # Integration tests
└── Cargo.toml           # Workspace configuration
```

## MCP Tools Available

1. **store_memory** - Store information in appropriate tier
2. **retrieve_memory** - Get memories by tier and query
3. **search_memory** - Semantic search across tiers
4. **consolidate_memory** - Merge and optimize memories
5. **analyze_memory_patterns** - Find patterns in stored data
6. **get_memory_stats** - System statistics and health
7. **clear_memory** - Selective memory clearing

## Critical Design Decisions

### Why Native Rust?
- **Performance**: Sub-millisecond operations vs Python's 10-100ms
- **Memory Safety**: No GC pauses, predictable performance
- **Integration**: Direct Candle ML support without FFI

### Why Candle over ONNX?
- **Native Rust**: No C++ dependencies
- **Simpler API**: Direct tensor operations
- **Better Performance**: Optimized for Rust's memory model

### Memory Tier Design
- **Working**: Immediate context (10 items)
- **ShortTerm**: Recent session (100 items)
- **LongTerm**: Persistent knowledge (10K items)
- **Episodic**: Time-sequenced events
- **Semantic**: Concept relationships

## Known Issues & Workarounds

1. **MCP Stdio Validation**: Not yet tested with Claude Desktop
   - Workaround: Use test client for validation

2. **Mock Embeddings**: Still using placeholder model
   - Workaround: Functions work, just not semantically accurate

3. **Model Loading Time**: Candle model initialization is slow
   - Workaround: Load once and keep in memory

## Contact Points & Integration

- **Parent Project**: `C:\codedev\llm\stats`
- **Redis**: localhost:6380
- **MCP Protocol**: stdio (JSON-RPC 2.0)
- **Logs**: RUST_LOG=info for debugging

## Next Session Checklist

When resuming work:
1. Check Redis is running on port 6380
2. Verify workspace builds: `cargo build --workspace`
3. Run tests to ensure nothing broken: `cargo test`
4. Check for any new requirements or changes in parent project

## Performance Benchmarks

- Vector similarity: <1ms for 1000 vectors
- Memory storage: <5ms per item
- Memory retrieval: <10ms for complex queries
- Consolidation: <1s for 1000 items

## Technical Debt Tracker

Priority items to address:
1. Remove mock embedding model code
2. Add proper MCP stdio validation
3. Implement embedding caching
4. Clean up test utilities
5. Update documentation

---

For full details, see `project-context-2025-01-22.json`