# Context Restoration Guide

## Quick Context Restoration Command

To restore full project context in a new session, share this with Claude:

```
I'm working on the RAG-Redis system, a high-performance Rust-based RAG system with multi-tier memory management.

Key context:
- Native Rust MCP server (replaced Python bridge)
- Candle ML for embeddings (replaced ONNX)
- Redis on port 6380
- 5-tier memory system
- Located at: C:\codedev\llm\rag-redis

Please load the project context from:
- .claude/context/project-context-2025-01-22.json (full context)
- .claude/context/PROJECT_SUMMARY.md (quick reference)

Current focus: [INSERT YOUR CURRENT TASK HERE]
```

## Key Context Points for Different Tasks

### For MCP Development
- Server location: `rag-redis-system/mcp-server/`
- Protocol: JSON-RPC 2.0 over stdio
- 7 implemented tools
- Pending: Claude Desktop validation

### For Memory System Work
- 5 tiers: Working, ShortTerm, LongTerm, Episodic, Semantic
- Redis backend on port 6380
- Sub-millisecond vector operations

### For Testing
- Run: `cargo test --workspace`
- Functional tests: `cargo test --test functional_tests`
- Mock embeddings currently in use

### For Performance Optimization
- Current: <1ms vector ops, <10ms retrieval
- Target: <100ms embeddings, <1s consolidation
- SIMD optimizations enabled

## Session State Markers

Mark your session state to help future context restoration:

- [ ] Redis running on 6380
- [ ] Workspace builds successfully
- [ ] All tests passing
- [ ] MCP server validated
- [ ] Real embeddings integrated

## Last Known Good State

- **Date**: 2025-01-22
- **Commit**: [Not tracked - add when available]
- **Tests**: All passing
- **Redis**: Port 6380
- **Issues**: MCP stdio validation pending

## Integration Points to Remember

1. **Parent Project**: C:\codedev\llm\stats
2. **Redis**: localhost:6380
3. **MCP Config**: Will be added to Claude Desktop settings
4. **Logs**: Use RUST_LOG=info for debugging

## Common Commands Reference

```bash
# Build everything
cd C:\codedev\llm\rag-redis\rag-redis-system
cargo build --release --workspace

# Test everything
cargo test --workspace

# Run MCP server
cargo run --bin mcp-server

# Check Redis
redis-cli -p 6380 ping

# View logs
RUST_LOG=info cargo run --bin mcp-server

# Clean build
cargo clean && cargo build --release
```

## Files to Check When Resuming

1. `Cargo.toml` - Workspace configuration
2. `mcp-server/src/tools.rs` - MCP tool implementations
3. `src/memory/mod.rs` - Memory tier logic
4. `tests/functional_tests.rs` - Integration test status

---

Remember: This is a native Rust implementation. The Python bridge has been archived.
Performance is the priority - all operations should be sub-second.