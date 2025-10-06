# Candle-Based Embedding System

This guide explains how to use the new Candle-based embedding system in rag-redis-system, which replaces the previous ONNX implementation with a native Rust ML framework.

## Overview

The rag-redis-system now supports local embedding generation using Candle, a Rust-native machine learning framework. This provides:

- **Native Rust performance**: No external dependencies or Python runtime required
- **Multi-model support**: DistilBERT and other transformer models via Candle
- **Hardware acceleration**: CPU and GPU support with automatic device selection
- **Memory efficiency**: Optimized tensor operations and memory pooling
- **Type safety**: Full Rust type checking and memory safety

## Features

### Supported Models

- **DistilBERT**: Lightweight BERT variant optimized for inference
- **Custom models**: Load any Candle-compatible transformer model
- **Multiple formats**: SafeTensors (.safetensors) model weights

### Configuration

```rust
use rag_redis_system::config::{EmbeddingConfig, EmbeddingProvider};
use std::time::Duration;

let config = EmbeddingConfig {
    provider: EmbeddingProvider::Candle,
    model: "distilbert-base-uncased".to_string(),
    dimension: 768,
    batch_size: 32,
    cache_embeddings: true,
    cache_ttl: Duration::from_secs(3600),
};
```

### Environment Variables

- `CANDLE_MODEL_PATH`: Path to model weights (.safetensors file)
- `CANDLE_TOKENIZER_PATH`: Path to tokenizer file (optional, auto-downloads if not set)

Example:
```bash
export CANDLE_MODEL_PATH="./models/distilbert-base-uncased.safetensors"
export CANDLE_TOKENIZER_PATH="./models/tokenizer.json"
```

## Usage

### Basic Embedding Generation

```rust
use rag_redis_system::embedding::{EmbeddingService, EmbeddingFactory};

// Create embedding service
let service = EmbeddingFactory::create(&config).await?;

// Generate single embedding
let text = "Hello, world!";
let embedding = service.embed_text(text).await?;
println!("Embedding dimension: {}", embedding.len());

// Generate batch embeddings
let texts = vec!["Text 1".to_string(), "Text 2".to_string()];
let embeddings = service.embed_batch(&texts).await?;
```

### Advanced Configuration

```rust
use rag_redis_system::config::EmbeddingProvider;

// Use specific Candle provider
let config = EmbeddingConfig {
    provider: EmbeddingProvider::Candle,
    model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
    dimension: 384, // all-MiniLM-L6-v2 dimension
    batch_size: 64,
    cache_embeddings: true,
    cache_ttl: Duration::from_secs(7200),
};

// Fallback configuration
let config = EmbeddingConfig {
    provider: EmbeddingProvider::Local, // Tries Candle first, falls back to external
    // ... other settings
};
```

## Building with Candle Support

### Prerequisites

1. **Rust toolchain**: 1.75.0 or later
2. **System dependencies**: None (Candle is pure Rust)
3. **Optional**: CUDA for GPU acceleration

### Build Commands

```bash
# Enable Candle support
cargo build --features gpu

# Include in full feature set
cargo build --features full

# Check compilation
cargo check --features gpu
```

### Available Features

- `gpu`: Enables Candle with all ML dependencies
- `full`: Includes gpu + all other features
- `default`: Basic functionality without Candle

## Model Management

### Downloading Models

Models can be downloaded from Hugging Face Hub automatically:

```rust
// Automatic download (when tokenizer not provided locally)
let service = CandleEmbeddingService::new(
    Path::new("./models/model.safetensors"),
    None, // Will download tokenizer from HF Hub
    768,
).await?;
```

### Model Formats

- **SafeTensors**: Recommended format (.safetensors)
- **Tokenizers**: JSON format from Hugging Face tokenizers

### Directory Structure

```
models/
├── distilbert-base-uncased.safetensors
├── tokenizer.json
└── config.json (optional)
```

## Performance Optimization

### Hardware Acceleration

Candle automatically selects the best available device:

1. **CUDA GPU**: If available and enabled
2. **CPU**: Fallback with optimized SIMD operations

### Memory Management

- **Tensor sharing**: Models share weights efficiently
- **Batch processing**: Optimized for multiple texts
- **Memory pooling**: Reuses allocated tensors

### Benchmarking

```bash
# Run embedding benchmarks
cargo bench --features gpu vector_search

# Profile memory usage
cargo run --features gpu --example candle_embedding_demo
```

## Error Handling

The system provides graceful fallbacks:

1. **Candle unavailable**: Falls back to external HTTP service
2. **Model not found**: Downloads from Hugging Face Hub
3. **GPU unavailable**: Uses CPU with SIMD optimizations
4. **OOM errors**: Reduces batch size automatically

## Migration from ONNX

### Dependencies Removed

- `ort` (ONNX Runtime)
- `wonnx` (WebAssembly ONNX)

### Dependencies Added

- `candle-core`: Core tensor operations
- `candle-nn`: Neural network primitives
- `candle-transformers`: Pre-built transformer models
- `tokenizers`: HuggingFace tokenizer library
- `hf-hub`: Model downloading

### Configuration Changes

```rust
// Old ONNX configuration
EmbeddingProvider::ONNX

// New Candle configuration
EmbeddingProvider::Candle
```

## Examples

### Basic Usage

```bash
# Run the demo
cargo run --features gpu --example candle_embedding_demo
```

### Integration Example

```rust
use rag_redis_system::{
    embedding::{EmbeddingService, EmbeddingFactory},
    config::{EmbeddingConfig, EmbeddingProvider},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::Candle,
        model: "distilbert-base-uncased".to_string(),
        dimension: 768,
        batch_size: 32,
        cache_embeddings: true,
        cache_ttl: std::time::Duration::from_secs(3600),
    };

    let service = EmbeddingFactory::create(&config).await?;

    let text = "RAG systems with Rust and Redis";
    let embedding = service.embed_text(text).await?;

    println!("Generated embedding with {} dimensions", embedding.len());
    Ok(())
}
```

## Troubleshooting

### Common Issues

1. **Compilation errors**: Ensure `gpu` feature is enabled
2. **Model not found**: Check `CANDLE_MODEL_PATH` environment variable
3. **GPU issues**: Candle will fallback to CPU automatically
4. **Memory issues**: Reduce batch size or use smaller models

### Debug Mode

```bash
# Enable debug logging
RUST_LOG=debug cargo run --features gpu --example candle_embedding_demo
```

### Verification

```bash
# Test that Candle integration works
cargo test --features gpu embedding::tests
```

## Future Enhancements

- **Additional models**: BERT, RoBERTa, T5 support
- **Quantization**: INT8/FP16 model compression
- **Distributed inference**: Multi-GPU support
- **Custom architectures**: User-defined model loading
- **ONNX compatibility**: Optional ONNX model import

## Contributing

When contributing to the embedding system:

1. **Test both CPU and GPU paths**
2. **Verify memory usage with large batches**
3. **Include benchmark comparisons**
4. **Document new model support**
5. **Maintain backward compatibility**

## License

This implementation follows the same MIT license as the parent project.