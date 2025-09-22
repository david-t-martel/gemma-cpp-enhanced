# Gemma Rust Extensions

High-performance Rust extensions for Gemma chatbot operations using PyO3.

## Overview

This crate provides optimized implementations of core operations that benefit from Rust's performance characteristics:

- **Fast Tokenization** (`tokenizer.rs`) - BPE tokenization with SIMD optimizations
- **Tensor Operations** (`tensor_ops.rs`) - Matrix operations, attention, and activations
- **High-Performance Caching** (`cache.rs`) - Thread-safe caching with TTL support

## Features

- ✅ PyO3 with abi3 for Python 3.11+
- ✅ SIMD optimizations (AVX2, NEON) where available
- ✅ Zero-copy operations
- ✅ Memory-safe implementations
- ✅ Async/await support
- ✅ Comprehensive benchmarking
- ✅ Thread-safe concurrent operations

## Performance Optimizations

### SIMD Support
- **x86_64**: AVX2, SSE4.2 for vectorized operations
- **aarch64**: NEON for ARM optimization
- **Fallback**: Scalar implementations for unsupported platforms

### Memory Management
- Aligned memory allocation for SIMD operations
- Smart caching with LRU eviction
- Memory usage tracking and limits
- Zero-copy string operations where possible

### Concurrency
- Lock-free data structures using DashMap
- Rayon for data parallelism
- Parking lot for efficient synchronization
- Async support with Tokio integration

## Building

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin for Python integration
pip install maturin

# Install development dependencies
cargo install criterion
```

### Development Build
```bash
# Build in development mode
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build Python extension
maturin develop
```

### Production Build
```bash
# Build optimized release
cargo build --release

# Build Python wheel
maturin build --release
```

## Usage

### Python Integration

```python
import gemma_extensions

# Initialize extensions
gemma_extensions.warmup()

# Check capabilities
print(f"SIMD support: {gemma_extensions.check_simd_support()}")
print(f"Build info: {gemma_extensions.get_build_info()}")
```

### Tokenization

```python
from gemma_extensions.tokenizer import FastTokenizer, TokenizerConfig

# Create tokenizer
config = TokenizerConfig()
config.vocab_size = 32000
config.use_simd = True

tokenizer = FastTokenizer(config)

# Encode text
tokens = tokenizer.encode("Hello, world!")
print(f"Tokens: {tokens}")

# Batch operations
texts = ["Hello", "world", "test"]
batch_tokens = gemma_extensions.tokenizer.batch_encode(tokenizer, texts)

# Get statistics
stats = tokenizer.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate_percent']}%")
```

### Tensor Operations

```python
from gemma_extensions.tensor_ops import TensorOperations

# Create tensor operations
tensor_ops = TensorOperations()

# SIMD operations
a = [1.0, 2.0, 3.0, 4.0]
b = [2.0, 3.0, 4.0, 5.0]
dot_product = gemma_extensions.tensor_ops.simd_dot_product(a, b)

# Matrix multiplication
result = tensor_ops.matmul(a, b, 2, 2, 2)  # 2x2 matrices

# Activation functions
input_data = [-1.0, 0.0, 1.0, 2.0]
gelu_output = tensor_ops.gelu(input_data)
silu_output = tensor_ops.silu(input_data)

# Attention mechanism
seq_len, hidden_dim = 128, 512
query = [0.1] * (seq_len * hidden_dim)
key = [0.2] * (seq_len * hidden_dim)
value = [0.3] * (seq_len * hidden_dim)

attention_output = tensor_ops.scaled_dot_product_attention(
    query, key, value, seq_len, hidden_dim
)
```

### Caching

```python
from gemma_extensions.cache import LRUCache, CacheManager

# Create LRU cache
cache = LRUCache(capacity=1000)

# Basic operations
cache.put("key1", "value1")
value = cache.get("key1")  # Returns "value1"

# TTL support
cache = LRUCache.with_ttl(capacity=1000, ttl_seconds=300)
cache.put_with_ttl("temp_key", "temp_value", ttl_seconds=60)

# Cache manager for multiple caches
manager = CacheManager(default_capacity=500)
user_cache = manager.get_cache("users")
session_cache = manager.get_cache("sessions")

# Statistics
stats = cache.stats()
print(f"Hit rate: {stats.hit_rate():.2f}%")
print(f"Memory usage: {stats.memory_usage} bytes")
```

### Async Operations

```python
import asyncio
from gemma_extensions.cache import LRUCache

async def async_cache_operations():
    cache = LRUCache(1000)

    # Async put/get
    await cache.put_async("async_key", "async_value")
    value = await cache.get_async("async_key")

    return value

# Run async operations
result = asyncio.run(async_cache_operations())
```

## Benchmarking

Run comprehensive benchmarks:

```bash
# All benchmarks
cargo bench

# Specific benchmarks
cargo bench tokenizer
cargo bench tensor
cargo bench cache

# With profiling
cargo bench --features=debug
```

### Performance Results

Typical performance improvements over pure Python:

- **Tokenization**: 10-50x faster depending on text size
- **Tensor Operations**: 5-20x faster with SIMD
- **Caching**: 3-10x faster for concurrent access
- **Memory Usage**: 40-60% reduction

## Configuration

### Cargo Features

```toml
[features]
default = ["simd", "parallel"]
simd = []                    # Enable SIMD optimizations
parallel = ["rayon"]         # Enable parallel processing
huggingface = ["hf-hub"]     # HuggingFace integration
candle = ["candle-core"]     # Candle ML framework
debug = []                   # Debug symbols and logging
```

### Environment Variables

```bash
# Force disable SIMD
export GEMMA_DISABLE_SIMD=1

# Set thread pool size
export RAYON_NUM_THREADS=8

# Enable debug logging
export RUST_LOG=gemma_extensions=debug
```

## Development

### Code Structure

```
src/
├── lib.rs          # Main module and Python bindings
├── tokenizer.rs    # Fast tokenization with SIMD
├── tensor_ops.rs   # Optimized tensor operations
├── cache.rs        # High-performance caching
├── error.rs        # Error handling and types
└── utils.rs        # Utility functions and helpers

benches/
├── tokenizer_bench.rs  # Tokenization benchmarks
├── tensor_bench.rs     # Tensor operation benchmarks
└── cache_bench.rs      # Caching benchmarks
```

### Testing

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration

# Property-based testing
cargo test --features=proptest

# Memory safety checks
cargo miri test
```

### Profiling

```bash
# CPU profiling
cargo bench --bench tensor_bench -- --profile-time=10

# Memory profiling
valgrind --tool=massif target/release/benchmarks

# Flame graph generation
cargo flamegraph --bench tensor_bench
```

## Contributing

1. Follow Rust best practices and clippy lints
2. Add comprehensive tests for new features
3. Update benchmarks for performance-critical code
4. Document public APIs with examples
5. Ensure SIMD fallbacks work correctly

### Code Style

```bash
# Format code
cargo fmt

# Lint code
cargo clippy -- -D warnings

# Check documentation
cargo doc --no-deps --open
```

## License

MIT License - see LICENSE file for details.

## Changelog

### v0.1.0
- Initial release
- Basic tokenization support
- Core tensor operations
- LRU caching implementation
- SIMD optimizations for x86_64 and aarch64
