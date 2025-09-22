//! High-performance inference engine for Gemma models
//!
//! This crate provides optimized implementations for neural network inference
//! with focus on performance, memory efficiency, and cross-platform support.
//!
//! # Features
//!
//! - **SIMD optimizations**: Leverage AVX2, AVX-512, and NEON for maximum performance
//! - **Memory pools**: Pre-allocated memory management to avoid allocation overhead
//! - **Lock-free data structures**: Concurrent execution without blocking
//! - **Cross-platform**: Supports x86_64, ARM64, and WASM targets
//! - **Multiple backends**: Candle, ONNX Runtime, and custom implementations
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Tokenizer     │    │  Tensor Ops     │    │  Memory Pool    │
//! │   - BPE/SentP   │    │  - SIMD ops     │    │  - Pre-alloc    │
//! │   - Parallel    │    │  - GPU accel    │    │  - Zero-copy    │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          └───────────────────────┼───────────────────────┘
//!                                  │
//!                    ┌─────────────────┐
//!                    │ Inference Engine │
//!                    │ - Model loading  │
//!                    │ - Forward pass   │
//!                    │ - Batching       │
//!                    └─────────────────┘
//! ```

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use tracing::{info, debug, warn, error, instrument};

pub mod error;
pub mod memory;
pub mod tensor;
pub mod tokenizer;
pub mod model;
pub mod batch;
pub mod simd;
pub mod attention;
pub mod inference;
pub mod config;
pub mod cache;

// Re-exports for public API
pub use error::{InferenceError, InferenceResult};
pub use memory::{MemoryPool, TensorArena, MemoryConfig};
pub use tensor::{Tensor, TensorOps, DataType, TensorShape};
pub use tokenizer::{TokenizerEngine, TokenizerConfig, EncodingResult};
pub use model::{ModelLoader, ModelConfig, ModelBackend};
pub use batch::{BatchProcessor, BatchConfig, BatchRequest};
pub use inference::{InferenceEngine, InferenceConfig, InferenceRequest, InferenceResponse};
pub use config::{EngineConfig, OptimizationLevel, DeviceConfig};
pub use cache::{KVCache, CacheConfig, AttentionCache};

// Global engine registry for sharing models across threads
static ENGINE_REGISTRY: Lazy<Arc<DashMap<String, Arc<InferenceEngine>>>> =
    Lazy::new(|| Arc::new(DashMap::new()));

/// Runtime feature detection and capabilities
#[derive(Debug, Clone)]
pub struct RuntimeCapabilities {
    pub simd_support: SimdCapabilities,
    pub gpu_support: GpuCapabilities,
    pub memory_info: MemoryInfo,
}

/// SIMD instruction set support
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub sse4_1: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

/// GPU acceleration support
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub cuda_available: bool,
    pub cuda_version: Option<String>,
    pub gpu_memory: u64,
    pub compute_capability: Option<(u32, u32)>,
}

/// System memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_memory: u64,
    pub available_memory: u64,
    pub page_size: u64,
}

impl RuntimeCapabilities {
    /// Detect runtime capabilities of the current system
    pub fn detect() -> Self {
        Self {
            simd_support: SimdCapabilities::detect(),
            gpu_support: GpuCapabilities::detect(),
            memory_info: MemoryInfo::detect(),
        }
    }
}

impl SimdCapabilities {
    /// Detect available SIMD instruction sets
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                sse4_1: false,
                avx: false,
                avx2: false,
                avx512f: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse2: false,
                sse4_1: false,
                avx: false,
                avx2: false,
                avx512f: false,
                neon: false,
            }
        }
    }

    /// Get the best available SIMD level
    pub fn best_simd_level(&self) -> SimdLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512f { SimdLevel::Avx512 }
            else if self.avx2 { SimdLevel::Avx2 }
            else if self.avx { SimdLevel::Avx }
            else if self.sse4_1 { SimdLevel::Sse41 }
            else if self.sse2 { SimdLevel::Sse2 }
            else { SimdLevel::Scalar }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.neon { SimdLevel::Neon }
            else { SimdLevel::Scalar }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    }
}

/// SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    Sse2,
    Sse41,
    Avx,
    Avx2,
    Avx512,
    Neon,
}

impl GpuCapabilities {
    /// Detect GPU capabilities
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            // CUDA detection would go here
            Self {
                cuda_available: false, // Placeholder
                cuda_version: None,
                gpu_memory: 0,
                compute_capability: None,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self {
                cuda_available: false,
                cuda_version: None,
                gpu_memory: 0,
                compute_capability: None,
            }
        }
    }
}

impl MemoryInfo {
    /// Detect system memory information
    pub fn detect() -> Self {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {
                total_memory: 2u64 << 30, // Assume 2GB limit for WASM
                available_memory: 1u64 << 30,
                page_size: 65536, // 64KB WASM page size
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos", target_arch = "wasm32")))]
        {
            Self {
                total_memory: 8u64 << 30, // Default 8GB
                available_memory: 4u64 << 30,
                page_size: 4096,
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;

        let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo.lines() {
            if let Some(kb_str) = line.strip_prefix("MemTotal:") {
                if let Some(kb) = kb_str.trim().split_whitespace().next() {
                    total = kb.parse().unwrap_or(0) * 1024;
                }
            } else if let Some(kb_str) = line.strip_prefix("MemAvailable:") {
                if let Some(kb) = kb_str.trim().split_whitespace().next() {
                    available = kb.parse().unwrap_or(0) * 1024;
                }
            }
        }

        Self {
            total_memory: total,
            available_memory: available,
            page_size: 4096,
        }
    }

    #[cfg(target_os = "windows")]
    fn detect_windows() -> Self {
        // Windows API calls would go here
        Self {
            total_memory: 16u64 << 30, // Placeholder 16GB
            available_memory: 8u64 << 30,
            page_size: 4096,
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_macos() -> Self {
        // macOS sysctl calls would go here
        Self {
            total_memory: 16u64 << 30, // Placeholder 16GB
            available_memory: 8u64 << 30,
            page_size: 4096,
        }
    }
}

/// Initialize the inference engine with configuration
#[instrument(skip(config))]
pub fn initialize_engine(name: &str, config: EngineConfig) -> InferenceResult<Arc<InferenceEngine>> {
    info!("Initializing inference engine: {}", name);

    // Check if engine already exists
    if let Some(engine) = ENGINE_REGISTRY.get(name) {
        debug!("Reusing existing inference engine: {}", name);
        return Ok(engine.clone());
    }

    // Create new engine
    let engine = Arc::new(InferenceEngine::new(config)?);
    ENGINE_REGISTRY.insert(name.to_string(), engine.clone());

    info!("Successfully initialized inference engine: {}", name);
    Ok(engine)
}

/// Get an existing inference engine by name
pub fn get_engine(name: &str) -> Option<Arc<InferenceEngine>> {
    ENGINE_REGISTRY.get(name).map(|entry| entry.clone())
}

/// Get runtime capabilities of the current system
pub fn get_runtime_capabilities() -> RuntimeCapabilities {
    RuntimeCapabilities::detect()
}

/// Warm up the inference engine by running basic operations
#[instrument]
pub async fn warmup_engine(engine: &InferenceEngine) -> InferenceResult<()> {
    info!("Warming up inference engine");

    // Warm up memory pools
    engine.memory_pool().warmup().await?;

    // Warm up tokenizer
    let sample_text = "Hello, world! This is a warmup sentence.";
    let _tokens = engine.tokenizer().encode(sample_text).await?;

    // Warm up tensor operations with small dummy computation
    let dummy_tensor = engine.memory_pool().allocate_tensor(&[1, 512, 768])?;
    let _result = engine.compute_attention(&dummy_tensor, &dummy_tensor, &dummy_tensor).await?;

    info!("Inference engine warmup completed");
    Ok(())
}

/// Shutdown and cleanup all engines
pub fn shutdown_engines() {
    info!("Shutting down all inference engines");
    ENGINE_REGISTRY.clear();
    info!("All inference engines shut down");
}

/// Get performance statistics for all engines
pub fn get_engine_stats() -> Vec<(String, serde_json::Value)> {
    ENGINE_REGISTRY
        .iter()
        .map(|entry| {
            let name = entry.key().clone();
            let stats = entry.value().get_statistics();
            (name, stats)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        let level = caps.best_simd_level();

        // Should at least support scalar operations
        assert!(level >= SimdLevel::Scalar);

        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 should support at least SSE2
            assert!(caps.sse2);
        }
    }

    #[test]
    fn test_memory_detection() {
        let memory = MemoryInfo::detect();

        // Should have some reasonable memory values
        assert!(memory.total_memory > 0);
        assert!(memory.available_memory > 0);
        assert!(memory.page_size > 0);

        // Available shouldn't exceed total
        assert!(memory.available_memory <= memory.total_memory);
    }

    #[tokio::test]
    async fn test_engine_registry() {
        let config = EngineConfig::default();
        let engine1 = initialize_engine("test1", config.clone()).unwrap();
        let engine2 = get_engine("test1").unwrap();

        assert!(Arc::ptr_eq(&engine1, &engine2));

        // Cleanup
        shutdown_engines();
    }
}
