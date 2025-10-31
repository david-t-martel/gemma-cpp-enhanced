# Hybrid RAG Backend Adapter

A production-ready, intelligent RAG (Retrieval-Augmented Generation) backend adapter with automatic backend selection and fallback support.

## Features

### 🚀 Intelligent Backend Selection
- **Priority 1: MCP Client** - Best performance via Rust backend over network
- **Priority 2: FFI Bindings** - Direct Rust integration via PyO3 (zero-copy)
- **Priority 3: Python Fallback** - Pure Python implementation (always available)

### 🎯 Unified Interface
All backends implement a consistent `RAGBackend` protocol with:
- `store_memory()` - Store memories with importance and tags
- `recall_memories()` - Query-based memory retrieval
- `search_memories()` - Similarity-scored search
- `ingest_document()` - Document chunking and ingestion
- `get_memory_stats()` - Backend statistics
- `cleanup_expired()` - Memory management

### 📊 Performance Monitoring
- Per-backend statistics tracking
- Latency measurements
- Success rate monitoring
- Health checking endpoints

### 🔧 Configurable Fallback
- Automatic backend detection
- Graceful degradation
- Manual backend preference
- Connection pooling (MCP/Redis)

## Quick Start

### Basic Usage

```python
from src.gemma_cli.rag import HybridRAGManager, MemoryType

# Initialize with automatic backend selection
manager = HybridRAGManager()
await manager.initialize()

# Store a memory
memory_id = await manager.store_memory(
    "Important information",
    memory_type=MemoryType.LONG_TERM,
    importance=0.8,
    tags=["important", "project"]
)

# Recall memories
memories = await manager.recall_memories("information", limit=5)
for memory in memories:
    print(f"{memory.content} (importance: {memory.importance})")

# Cleanup
await manager.close()
```

### Document Ingestion

```python
from pathlib import Path
from src.gemma_cli.rag import DocumentMetadata

doc_id = await manager.ingest_document(
    Path("document.txt"),
    metadata=DocumentMetadata(
        title="Project Documentation",
        source="docs",
        doc_type="markdown",
        tags=["docs"],
        importance=0.9
    ),
    chunk_size=512
)
```

### Backend Preference

```python
from src.gemma_cli.rag import BackendType

# Prefer MCP backend, fall back if unavailable
manager = HybridRAGManager(
    prefer_backend=BackendType.MCP,
    mcp_host="localhost",
    mcp_port=8765
)
await manager.initialize()
```

## Memory Tiers

The adapter supports five memory tiers matching cognitive architecture:

1. **WORKING** - Immediate context (high priority, short retention)
2. **SHORT_TERM** - Recent interactions (medium priority, hours)
3. **LONG_TERM** - Consolidated facts (high priority, persistent)
4. **EPISODIC** - Event sequences with timestamps
5. **SEMANTIC** - Concept relationships and knowledge graphs

```python
await manager.store_memory(
    "Current task context",
    memory_type=MemoryType.WORKING,
    importance=0.9
)
```

## Backend Details

### MCP Backend (Priority 1)

**Advantages:**
- Best performance (Rust implementation)
- Process isolation
- Network-based scalability
- Multiple concurrent clients

**Requirements:**
- MCP server running (default: localhost:8765)
- `aiohttp` for HTTP client

**Configuration:**
```python
manager = HybridRAGManager(
    mcp_host="localhost",
    mcp_port=8765,
)
```

### FFI Backend (Priority 2)

**Advantages:**
- Direct Rust integration
- Zero-copy data transfer
- Best latency for local operations
- No network overhead

**Requirements:**
- `rag-redis-system` Rust library compiled with PyO3
- Cargo workspace properly configured

**Build FFI Module:**
```bash
cd rag-redis-system
cargo build --release --features pyo3
```

### Python Backend (Priority 3)

**Advantages:**
- Always available (no external dependencies)
- Pure Python implementation
- Graceful degradation
- Redis optional (in-memory fallback)

**Requirements:**
- None (minimal) or `redis` for persistence

**Configuration:**
```python
manager = HybridRAGManager(
    redis_url="redis://localhost:6379"
)
```

## Performance Monitoring

### Get Backend Statistics

```python
stats = manager.get_backend_stats()
for backend, stat in stats.items():
    print(f"{backend.value}:")
    print(f"  Successful calls: {stat['successful_calls']}")
    print(f"  Avg latency: {stat['avg_latency_ms']:.2f}ms")
    print(f"  Success rate: {stat['success_rate']:.1%}")
```

### Health Check

```python
health = await manager.health_check()
print(f"Status: {health['status']}")
print(f"Active backend: {health['active_backend']}")
print(f"Avg latency: {health['performance']['avg_latency_ms']:.2f}ms")
```

## Error Handling

All backends provide consistent error handling:

```python
try:
    memory_id = await manager.store_memory("content")
except RuntimeError as e:
    # Backend not initialized or operation failed
    logger.error(f"Storage failed: {e}")
except FileNotFoundError as e:
    # Document ingestion file not found
    logger.error(f"File error: {e}")
```

The manager automatically tracks failures and maintains statistics.

## Testing

Run the comprehensive test suite:

```bash
# All tests
uv run pytest tests/test_rag_adapter.py -v

# Specific test class
uv run pytest tests/test_rag_adapter.py::TestHybridRAGManager -v

# With coverage
uv run pytest tests/test_rag_adapter.py --cov=src.gemma_cli.rag
```

## Examples

Run the example script:

```bash
uv run python src/gemma_cli/rag/example.py
```

This demonstrates:
- Basic operations
- Document ingestion
- Health monitoring
- Backend preference
- Memory tiers

## Architecture

```
HybridRAGManager
    │
    ├─> MCPRAGBackend (Priority 1)
    │   └─> HTTP Client → MCP Server (Rust)
    │
    ├─> FFIRAGBackend (Priority 2)
    │   └─> PyO3 FFI → rag-redis-system (Rust)
    │
    └─> PythonRAGBackend (Priority 3)
        └─> Redis Client → Redis Server
            └─> In-memory fallback (no Redis)
```

## Configuration via Settings

Integrate with project settings:

```python
from src.shared.config.settings import get_settings

settings = get_settings()

manager = HybridRAGManager(
    mcp_host=settings.redis.host,
    mcp_port=8765,
    redis_url=f"redis://{settings.redis.host}:{settings.redis.port}"
)
```

## Best Practices

### 1. Always Initialize
```python
manager = HybridRAGManager()
if await manager.initialize():
    # Use manager
    pass
else:
    logger.error("Failed to initialize RAG manager")
```

### 2. Use Context Managers
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def rag_manager():
    manager = HybridRAGManager()
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.close()

async with rag_manager() as manager:
    await manager.store_memory("content")
```

### 3. Monitor Performance
```python
# Periodically check health
health = await manager.health_check()
if health["status"] != "healthy":
    logger.warning(f"RAG system unhealthy: {health}")
```

### 4. Tag Memories
```python
await manager.store_memory(
    content="...",
    tags=["project:rag", "type:docs", "importance:high"]
)
```

### 5. Use Appropriate Memory Tiers
- **WORKING**: Current conversation context
- **SHORT_TERM**: Recent session data
- **LONG_TERM**: User preferences, facts
- **EPISODIC**: Event history
- **SEMANTIC**: Knowledge relationships

## Troubleshooting

### MCP Backend Not Available
```
Failed to initialize MCP backend: Connection refused
```
**Solution:** Ensure MCP server is running on the specified port.

### FFI Module Not Found
```
Failed to initialize FFI backend: No module named 'rag_redis_system'
```
**Solution:** Build and install the Rust FFI module with PyO3.

### Redis Connection Failed
```
Python backend using in-memory storage (no Redis)
```
**Solution:** This is non-fatal. Redis is optional, in-memory fallback is used.

## Contributing

When adding new backends:

1. Implement the `RAGBackend` protocol
2. Add to `BackendType` enum
3. Update `HybridRAGManager._try_backend()`
4. Add tests to `test_rag_adapter.py`
5. Update documentation

## License

Part of the Gemma LLM project.

## See Also

- [RAG-Redis System](../../rag-redis-system/README.md) - Rust backend
- [MCP Tools](../infrastructure/tools/rag_tools.py) - MCP tool implementations
- [Settings](../shared/config/settings.py) - Configuration management
