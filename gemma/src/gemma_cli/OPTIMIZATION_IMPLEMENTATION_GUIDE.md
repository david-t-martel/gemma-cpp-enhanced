# Gemma CLI Performance Optimization Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the performance optimizations identified in `PERFORMANCE_ANALYSIS.md`. The optimizations are ordered by priority and impact.

## Phase 1: Quick Wins (1 Day)

### 1. Implement Lazy Imports in CLI

**File:** `cli.py`

Replace eager imports with lazy loading:

```python
# OLD (current):
from .commands.setup import setup_group
from .commands.model import model
from .core.gemma import GemmaRuntimeParams

# NEW (optimized):
def chat(ctx, ...):
    # Import only when command is used
    from .config.settings import load_config
    from .core.gemma import GemmaInterface
```

**Impact:** -200ms startup time

### 2. Enable Configuration Caching

**File:** `config/settings.py` → Use `config/optimized_settings.py`

Replace the load_config function:

```python
# In cli.py and other modules:
# from .config.settings import load_config
from .config.optimized_settings import load_config  # Use cached version
```

The optimized version includes:
- LRU caching with mtime checking
- 5-minute TTL cache
- Lazy loading patterns

**Impact:** -95% config load time (100ms → 5ms)

### 3. Increase Buffer Size

**File:** `core/gemma.py` → Use `core/optimized_gemma.py`

Change buffer size from 8KB to 64KB:

```python
# OLD:
BUFFER_SIZE = 8192  # 8KB

# NEW:
BUFFER_SIZE = 65536  # 64KB
```

**Impact:** -10% token generation time

## Phase 2: Medium-Term Optimizations (1 Week)

### 4. Deploy Indexed RAG Store

**File:** `rag/embedded_vector_store.py` → Use `rag/optimized_embedded_store.py`

Update `rag/hybrid_rag.py` to use the optimized store:

```python
# In hybrid_rag.py
if use_embedded_store:
    from .optimized_embedded_store import OptimizedEmbeddedVectorStore
    self.backend.embedded_store = OptimizedEmbeddedVectorStore()
```

Features implemented:
- Inverted index for O(log n) search
- Async write batching
- Query result caching
- Smart document chunking

**Impact:** 10x faster search, 80% faster writes

### 5. Enable Performance Monitoring

Add profiling to key functions:

```python
from utils.profiler import PerformanceMonitor

@PerformanceMonitor.track("operation_name")
async def my_function():
    # ... function body ...

# Get performance report
report = PerformanceMonitor.report()
```

**Impact:** Visibility into bottlenecks

## Phase 3: Integration Steps

### Step 1: Test Individual Components

```bash
# Test optimized config loading
python -c "from config.optimized_settings import load_config; import time; start=time.time(); load_config(); print(f'Load time: {(time.time()-start)*1000:.1f}ms')"

# Test optimized RAG store
python -c "import asyncio; from rag.optimized_embedded_store import OptimizedEmbeddedVectorStore; store = OptimizedEmbeddedVectorStore(); asyncio.run(store.initialize())"
```

### Step 2: Update Imports

Create a migration script to update imports:

```python
# migration.py
import os
import re

replacements = [
    (r'from \.config\.settings import load_config',
     'from .config.optimized_settings import load_config'),
    (r'from \.rag\.embedded_vector_store import EmbeddedVectorStore',
     'from .rag.optimized_embedded_store import OptimizedEmbeddedVectorStore'),
    (r'from \.core\.gemma import GemmaInterface',
     'from .core.optimized_gemma import GemmaInterface'),
]

for root, dirs, files in os.walk('src/gemma_cli'):
    for file in files:
        if file.endswith('.py'):
            # Apply replacements
            ...
```

### Step 3: Add Feature Flags

Allow gradual rollout with feature flags:

```python
# config/settings.py
class OptimizationConfig(BaseModel):
    use_lazy_imports: bool = True
    use_config_cache: bool = True
    use_indexed_rag: bool = True
    use_large_buffers: bool = True
    enable_profiling: bool = False

# In code:
if settings.optimizations.use_config_cache:
    from .config.optimized_settings import load_config
else:
    from .config.settings import load_config
```

## Testing & Validation

### Performance Benchmarks

Create benchmark script:

```python
# benchmarks/test_optimizations.py
import asyncio
import time
from pathlib import Path

async def benchmark_rag_search():
    """Benchmark RAG search performance."""
    from rag.optimized_embedded_store import OptimizedEmbeddedVectorStore

    store = OptimizedEmbeddedVectorStore()
    await store.initialize()

    # Add test data
    for i in range(1000):
        await store.store_memory(StoreMemoryParams(
            content=f"Test document {i} with content",
            memory_type="working",
            importance=0.5
        ))

    # Benchmark search
    start = time.perf_counter()
    for _ in range(100):
        await store.recall_memories(RecallMemoriesParams(
            query="test document",
            limit=10
        ))
    elapsed = time.perf_counter() - start

    print(f"100 searches took: {elapsed:.2f}s")
    print(f"Average search time: {elapsed/100*1000:.1f}ms")

asyncio.run(benchmark_rag_search())
```

### Expected Results

After implementing all optimizations:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Startup time | 500ms | 250ms | -50% |
| Config load (cached) | 100ms | 5ms | -95% |
| RAG search (1000 items) | 100ms | 10ms | -90% |
| RAG write | 50ms | 10ms | -80% |
| Token generation | 100ms/token | 90ms/token | -10% |
| Memory usage | 150MB | 120MB | -20% |

## Rollback Plan

If issues occur, rollback is simple:

1. **Revert imports:** Change back to original modules
2. **Clear caches:** Delete `~/.gemma_cli/*_cache.json`
3. **Reset config:** Remove optimization flags

```bash
# Rollback script
git checkout -- src/gemma_cli/cli.py
git checkout -- src/gemma_cli/config/settings.py
git checkout -- src/gemma_cli/rag/hybrid_rag.py
rm -rf ~/.gemma_cli/*_cache.json
```

## Monitoring in Production

Add metrics collection:

```python
# In main chat loop
if settings.monitoring.enabled:
    from utils.profiler import PerformanceMonitor
    PerformanceMonitor.enable()

    # At shutdown
    report = PerformanceMonitor.report()
    with open("performance_metrics.json", "w") as f:
        json.dump(report, f)
```

## Next Steps

1. **Immediate:** Implement Phase 1 optimizations
2. **Week 1:** Deploy indexed RAG and monitoring
3. **Month 1:** Design process pooling architecture
4. **Month 2:** Implement model preloading service

## Summary

The optimization package provides:

1. **`utils/profiler.py`** - Performance monitoring tools
2. **`config/optimized_settings.py`** - Cached configuration loading
3. **`rag/optimized_embedded_store.py`** - Indexed RAG with batching
4. **`core/optimized_gemma.py`** - Optimized subprocess I/O

These can be adopted incrementally with minimal risk and provide immediate performance improvements of 50-90% across critical operations.