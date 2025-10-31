# Standalone Operation Without Redis

## Overview

As of this update, Gemma CLI is configured to run in **standalone mode by default**, using an embedded file-based vector store for RAG memory operations. This means you can use the application without installing or running Redis.

## Configuration Changes

### Default Settings

The following default configuration enables standalone operation:

```python
# config/settings.py - RedisConfig
enable_fallback: bool = True  # Default: Use embedded store (standalone mode)
```

When `enable_fallback=True` (the default), the application uses `EmbeddedVectorStore` instead of Redis.

### Memory Storage Backend

**Embedded Store (Default)**:
- File-based persistence: `~/.gemma_cli/embedded_store.json`
- No external dependencies required
- Automatic initialization
- Suitable for local development and single-user deployments

**Redis Backend (Optional)**:
- For distributed access or large datasets
- Set `redis.enable_fallback=False` in config
- Requires Redis server running

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│         Gemma CLI Application           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        HybridRAGManager                 │
│  (use_embedded_store=True by default)  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        PythonRAGBackend                 │
│  (Routes to appropriate backend)        │
└─────────┬───────────────────────────────┘
          │
          ├── enable_fallback=True ──► EmbeddedVectorStore
          │                            (JSON file persistence)
          │
          └── enable_fallback=False ─► Redis
                                       (Requires Redis server)
```

### Code Flow

1. **Configuration Loading** (`config/settings.py`):
   ```python
   class RedisConfig(BaseModel):
       enable_fallback: bool = True  # Default
   ```

2. **RAG Manager Initialization** (`cli.py`):
   ```python
   rag_manager = HybridRAGManager(
       use_embedded_store=settings.redis.enable_fallback
   )
   ```

3. **Backend Selection** (`rag/python_backend.py`):
   ```python
   def __init__(self, use_embedded_store: bool = True):
       if self.use_embedded_store:
           self.embedded_store = EmbeddedVectorStore()
       else:
           # Initialize Redis connection
   ```

## File Locations

### Embedded Store Data
- **Storage File**: `~/.gemma_cli/embedded_store.json`
- **Format**: JSON with memory entries
- **Auto-created**: On first run

### Configuration
- **Config File**: `~/.gemma_cli/config.toml` or `config/config.toml`
- **Relevant Setting**:
  ```toml
  [redis]
  enable_fallback = true  # Use embedded store
  ```

## Onboarding Experience

The onboarding wizard has been updated to reflect standalone operation:

### Step 3: Memory Storage Configuration

The wizard now:
1. **Checks for Redis** (optional, non-blocking)
2. **Defaults to embedded storage** if Redis not available
3. **Offers manual Redis configuration** for advanced users
4. **Never fails** if Redis is unavailable

Example output:
```
╭─ Memory Storage Configuration ─────────────────────╮
│                                                     │
│ The 5-tier memory system and RAG capabilities      │
│ can use either:                                     │
│   • Embedded storage (default) - File-based,       │
│     no setup required                               │
│   • Redis (optional) - For distributed access      │
│     or large datasets                               │
│                                                     │
│ Checking for Redis (optional)...                   │
│ ✗ Redis not available                              │
│ ✓ Using embedded file-based storage                │
│   (recommended for local use)                       │
╰─────────────────────────────────────────────────────╯
```

## Verification

To verify the application works without Redis:

### Method 1: Run Test Script
```bash
cd /c/codedev/llm/gemma
uv run python src/gemma_cli/test_embedded_store.py
```

### Method 2: Manual Testing
```bash
# Start the application (no Redis required)
cd /c/codedev/llm/gemma
uv run python -m gemma_cli.cli chat

# In the application, test RAG commands:
# - Store a memory
# - Search memories
# - Check memory stats
```

### Method 3: Check Configuration
```bash
# Verify default configuration
cd /c/codedev/llm/gemma
uv run python -c "
from gemma_cli.config.settings import load_config
settings = load_config()
print(f'enable_fallback: {settings.redis.enable_fallback}')
print('Expected: True (standalone mode)')
"
```

## Switching to Redis Backend

If you want to use Redis instead of the embedded store:

### Option 1: Configuration File
Edit `~/.gemma_cli/config.toml`:
```toml
[redis]
enable_fallback = false  # Use Redis
host = "localhost"
port = 6379
db = 0
```

### Option 2: During Onboarding
When running `gemma-cli init`, choose "Use Redis for memory storage?" when prompted.

### Option 3: Manual Configuration
Run the wizard with:
```bash
gemma-cli init --force
```

Then select Redis when prompted in Step 3.

## Features & Limitations

### Embedded Store Features
✅ **Supported**:
- All 5-tier memory operations (working, short-term, long-term, episodic, semantic)
- Store and recall memories
- Search by content and importance
- Document ingestion with chunking
- Memory statistics
- Persistent storage (JSON)
- Automatic initialization
- No external dependencies

⚠️ **Limitations**:
- Keyword-based search (not true semantic similarity without embedding model)
- Performance scales linearly with dataset size
- Single-process only (no distributed access)
- No built-in replication or backup

### When to Use Redis Instead
Consider Redis for:
- **Large datasets**: >10,000 memories
- **Production deployments**: With backup and replication
- **Distributed access**: Multiple processes/servers
- **Advanced features**: Redis Search, vector similarity
- **Performance**: Sub-millisecond lookups

## Migration Path

### From Redis to Embedded Store
1. Export Redis data:
   ```bash
   redis-cli --scan --pattern "gemma:mem:*" | \
     xargs redis-cli MGET > redis_backup.json
   ```

2. Enable fallback in config:
   ```toml
   [redis]
   enable_fallback = true
   ```

3. Application will use embedded store on next run

### From Embedded Store to Redis
1. Backup embedded store:
   ```bash
   cp ~/.gemma_cli/embedded_store.json ~/.gemma_cli/embedded_store.backup.json
   ```

2. Start Redis server

3. Disable fallback in config:
   ```toml
   [redis]
   enable_fallback = false
   ```

4. Import data (manual script needed)

## Troubleshooting

### Q: How do I know if I'm using embedded store?
**A**: Check the logs or config:
```bash
# Check config
cat ~/.gemma_cli/config.toml | grep enable_fallback

# Check if embedded store file exists
ls -lh ~/.gemma_cli/embedded_store.json
```

### Q: Can I use both Redis and embedded store?
**A**: No, only one backend is active at a time, controlled by `enable_fallback`.

### Q: What happens if Redis is unavailable when enable_fallback=False?
**A**: The application will fail to initialize RAG features. Set `enable_fallback=True` to use embedded store as fallback.

### Q: How do I clear the embedded store?
**A**: Delete the storage file:
```bash
rm ~/.gemma_cli/embedded_store.json
```

### Q: Is the embedded store production-ready?
**A**: It's suitable for:
- Local development
- Single-user deployments
- Datasets <10,000 memories
- Non-critical applications

For production with high availability needs, use Redis.

## Summary

| Feature | Embedded Store | Redis |
|---------|---------------|-------|
| **Default** | ✅ Yes | No |
| **External Dependencies** | None | Redis server |
| **Setup Required** | No | Yes |
| **Persistence** | JSON file | Redis DB |
| **Performance** | Good (<10K entries) | Excellent |
| **Distributed Access** | No | Yes |
| **Backup/Replication** | Manual | Built-in |
| **Semantic Search** | Basic (keyword) | Advanced |
| **Best For** | Local dev, single user | Production, teams |

**Recommendation**: Use embedded store (default) for local development and testing. Consider Redis for production deployments with high availability or large dataset requirements.
