//! High-performance Rust extensions for Gemma chatbot operations
//!
//! This crate provides optimized implementations of core operations that benefit
//! from Rust's performance characteristics:
//! - Fast tokenization with SIMD optimizations
//! - Zero-copy tensor operations
//! - High-performance caching with concurrent access
//! - Async I/O operations with Python integration

#![allow(dead_code)]
#![allow(unused_imports)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::Once;
use tracing::{info, warn};

// Module declarations
pub mod error;
pub mod tokenizer;
pub mod utils;
// TODO: Re-enable after fixing parallel compilation issues
// pub mod document_processor;
// pub mod tensor_ops;
// pub mod cache;

// Gemma.cpp FFI module (optional)
#[cfg(feature = "gemma-cpp")]
pub mod gemma_cpp;

// RAG system modules (commented out for minimal build)
// pub mod redis_manager;
// pub mod vector_store;
// pub mod document_pipeline;
// pub mod research_client;
// pub mod ffi;

// Re-exports for Python bindings
pub use error::{GemmaError, GemmaResult};
pub use tokenizer::{FastTokenizer, TokenizationResult, TokenizerConfig};
// TODO: Re-enable after fixing compilation issues
// pub use document_processor::{DocumentProcessor, DocumentConfig, DocumentFormat,
//                             DocumentMetadata, ProcessingResult, process_document,
//                             process_documents_batch, detect_format};
// TODO: Re-enable after fixing parallel compilation issues
// pub use tensor_ops::{TensorConfig, TensorOperations, optimize_attention_weights, batch_matmul};
// pub use cache::{LRUCache, CacheManager, CacheStats};

// RAG system re-exports (commented out for minimal build)
// pub use redis_manager::{RedisManager, RedisConfig, DocumentMetadata};
// pub use vector_store::{VectorStore, VectorStoreConfig, DistanceMetric, SearchResult};
// pub use document_pipeline::{DocumentPipeline, ChunkingConfig, EmbeddingConfig, DocumentFormat};
// pub use research_client::{ResearchClient, ResearchConfig, ResearchQuery, ResearchResponse};

static INIT: Once = Once::new();

/// Initialize the Rust extension module
fn init_tracing() {
    INIT.call_once(|| {
        tracing_subscriber::fmt::init();
        info!("Gemma Rust extensions initialized");
    });
}

/// Get version information about the extension
#[pyfunction]
fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get build information
#[pyfunction]
fn get_build_info() -> PyResult<std::collections::HashMap<String, String>> {
    let mut info = std::collections::HashMap::new();
    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("target".to_string(), env!("TARGET").to_string());
    info.insert(
        "profile".to_string(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_string(),
    );
    info.insert(
        "features".to_string(),
        format!("{:?}", get_enabled_features()),
    );
    Ok(info)
}

/// Get list of enabled features
fn get_enabled_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(feature = "simd")]
    features.push("simd");

    #[cfg(feature = "parallel")]
    features.push("parallel");

    #[cfg(feature = "huggingface")]
    features.push("huggingface");

    #[cfg(feature = "candle")]
    features.push("candle");

    #[cfg(feature = "debug")]
    features.push("debug");

    #[cfg(feature = "redis-backend")]
    features.push("redis-backend");

    #[cfg(feature = "faiss-backend")]
    features.push("faiss-backend");

    #[cfg(feature = "transformers")]
    features.push("transformers");

    #[cfg(feature = "full-rag")]
    features.push("full-rag");

    features
}

/// Check if SIMD is available and working
#[pyfunction]
fn check_simd_support() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(feature = "simd"))]
    {
        false
    }
}

/// Warm up the extension by running some basic operations
#[pyfunction]
fn warmup() -> PyResult<String> {
    init_tracing();

    // Warm up tokenizer
    let tokenizer = tokenizer::FastTokenizer::new(tokenizer::TokenizerConfig::default())?;
    let _ = tokenizer.encode("Hello, world!");

    // // Warm up cache
    // let cache = cache::LRUCache::new(100)?;
    // cache.put("test".to_string(), "value".to_string())?;
    // let _ = cache.get("test")?;

    // // Warm up tensor operations
    // let data = vec![1.0f32, 2.0, 3.0, 4.0];
    // let _ = tensor_ops::simd_dot_product(&data, &data);

    info!("Warmup completed successfully");
    Ok("Warmup completed".to_string())
}

/// Benchmark core operations
#[pyfunction]
fn benchmark_operations() -> PyResult<std::collections::HashMap<String, f64>> {
    use std::time::Instant;
    let mut results = std::collections::HashMap::new();

    // TODO: Re-enable after fixing dependencies
    // // Benchmark tokenization
    // let start = Instant::now();
    // let mut tokenizer = tokenizer::FastTokenizer::new(
    //     tokenizer::TokenizerConfig::default()
    // )?;
    // for _ in 0..1000 {
    //     let _ = tokenizer.encode("This is a test sentence for benchmarking tokenization performance.");
    // }
    // let tokenization_time = start.elapsed().as_secs_f64();
    // results.insert("tokenization_1000_ops".to_string(), tokenization_time);

    // // Benchmark tensor operations
    // let data1 = vec![1.0f32; 1000];
    // let data2 = vec![2.0f32; 1000];
    // let start = Instant::now();
    // for _ in 0..1000 {
    //     let _ = tensor_ops::simd_dot_product(&data1, &data2);
    // }
    // let tensor_time = start.elapsed().as_secs_f64();
    // results.insert("tensor_ops_1000_dots".to_string(), tensor_time);

    // // Benchmark cache operations
    // let cache = cache::LRUCache::new(1000)?;
    // let start = Instant::now();
    // for i in 0..1000 {
    //     cache.put(format!("key_{}", i), format!("value_{}", i))?;
    // }
    // for i in 0..1000 {
    //     let _ = cache.get(&format!("key_{}", i))?;
    // }
    // let cache_time = start.elapsed().as_secs_f64();
    // results.insert("cache_1000_put_get".to_string(), cache_time);

    // Placeholder benchmark
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = "basic_operation".to_string();
    }
    let basic_time = start.elapsed().as_secs_f64();
    results.insert("basic_ops_1000".to_string(), basic_time);

    Ok(results)
}

/// Python module definition
#[pymodule]
fn _gemma_extensions(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing
    init_tracing();

    // Add version and build info
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(check_simd_support, m)?)?;
    m.add_function(wrap_pyfunction!(warmup, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_operations, m)?)?;

    // Add tokenizer classes and functions
    m.add_class::<tokenizer::FastTokenizer>()?;
    m.add_class::<tokenizer::TokenizerConfig>()?;
    m.add_class::<tokenizer::TokenizationResult>()?;
    m.add_function(wrap_pyfunction!(tokenizer::create_bpe_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::batch_encode, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::batch_decode, m)?)?;

    // TODO: Re-enable after fixing compilation issues
    // Add document processor classes and functions
    // m.add_class::<document_processor::DocumentProcessor>()?;
    // m.add_class::<document_processor::DocumentConfig>()?;
    // m.add_class::<document_processor::DocumentFormat>()?;
    // m.add_class::<document_processor::DocumentMetadata>()?;
    // m.add_class::<document_processor::ProcessingResult>()?;
    // m.add_function(wrap_pyfunction!(document_processor::process_document, m)?)?;
    // m.add_function(wrap_pyfunction!(document_processor::process_documents_batch, m)?)?;
    // m.add_function(wrap_pyfunction!(document_processor::detect_format, m)?)?;

    // // Add tensor operation classes and functions
    // m.add_class::<tensor_ops::TensorOperations>()?;
    // m.add_class::<tensor_ops::TensorConfig>()?;
    // m.add_function(wrap_pyfunction!(tensor_ops::simd_dot_product, m)?)?;
    // m.add_function(wrap_pyfunction!(tensor_ops::simd_vector_add, m)?)?;
    // m.add_function(wrap_pyfunction!(tensor_ops::simd_softmax, m)?)?;
    // m.add_function(wrap_pyfunction!(tensor_ops::optimize_attention_weights, m)?)?;
    // m.add_function(wrap_pyfunction!(tensor_ops::batch_matmul, m)?)?;
    // m.add_function(wrap_pyfunction!(tensor_ops::fast_layer_norm, m)?)?;

    // // Add cache classes and functions
    // m.add_class::<cache::LRUCache>()?;
    // m.add_class::<cache::CacheManager>()?;
    // m.add_class::<cache::CacheStats>()?;

    // Add async support
    // TODO: Fix pyo3-asyncio initialization after core build works
    // pyo3_asyncio::tokio::init(tokio::runtime::Builder::new_current_thread());

    // Add submodules
    let tokenizer_module = PyModule::new(py, "tokenizer")?;
    tokenizer::register_module(py, &tokenizer_module)?;
    m.add_submodule(&tokenizer_module)?;

    // TODO: Re-enable after fixing compilation issues
    // let document_module = PyModule::new(py, "document_processor")?;
    // document_processor::register_module(py, document_module)?;
    // m.add_submodule(document_module)?;

    // let tensor_module = PyModule::new(py, "tensor_ops")?;
    // tensor_ops::register_module(py, tensor_module)?;
    // m.add_submodule(tensor_module)?;

    // let cache_module = PyModule::new(py, "cache")?;
    // cache::register_module(py, cache_module)?;
    // m.add_submodule(cache_module)?;

    // Register gemma.cpp module if feature is enabled
    #[cfg(feature = "gemma-cpp")]
    {
        gemma_cpp::register_gemma_cpp(py, m)?;
        info!("Gemma.cpp FFI module registered");
    }

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS"))?;
    m.add("__doc__", env!("CARGO_PKG_DESCRIPTION"))?;

    info!("Gemma Extensions module loaded successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = get_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_build_info() {
        let info = get_build_info().unwrap();
        assert!(info.contains_key("version"));
        assert!(info.contains_key("target"));
        assert!(info.contains_key("profile"));
    }

    #[test]
    fn test_features() {
        let features = get_enabled_features();
        assert!(!features.is_empty());
    }
}
