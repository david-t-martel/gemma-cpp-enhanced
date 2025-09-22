//! Configuration types for the inference engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Device configuration (CPU/GPU)
    pub device: DeviceConfig,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Batching configuration
    pub batching: BatchConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            memory: MemoryConfig::default(),
            device: DeviceConfig::default(),
            optimization: OptimizationConfig::default(),
            batching: BatchConfig::default(),
        }
    }
}

/// Model-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model file
    pub path: PathBuf,
    /// Model format (safetensors, pytorch, onnx, etc.)
    pub format: ModelFormat,
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Model precision (f32, f16, int8, etc.)
    pub precision: ModelPrecision,
    /// Context length
    pub context_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("model.bin"),
            format: ModelFormat::SafeTensors,
            architecture: ModelArchitecture::Gemma,
            precision: ModelPrecision::F32,
            context_length: 2048,
            vocab_size: 32000,
        }
    }
}

/// Supported model formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelFormat {
    SafeTensors,
    PyTorch,
    Onnx,
    TensorFlow,
    Custom,
}

/// Supported model architectures
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Gemma,
    Llama,
    Mistral,
    Phi,
    Custom,
}

/// Model precision settings
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelPrecision {
    F32,
    F16,
    BF16,
    Int8,
    Int4,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Pool size in bytes
    pub pool_size: usize,
    /// Enable memory mapping for large models
    pub use_mmap: bool,
    /// Pre-allocate tensor arena
    pub preallocate_arena: bool,
    /// Arena size in bytes
    pub arena_size: usize,
    /// Enable garbage collection
    pub enable_gc: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: 2usize << 30, // 2GB
            use_mmap: true,
            preallocate_arena: true,
            arena_size: 512usize << 20, // 512MB
            enable_gc: true,
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Primary device type
    pub device_type: DeviceType,
    /// Device ID (for multi-GPU systems)
    pub device_id: Option<u32>,
    /// Number of CPU threads to use
    pub cpu_threads: Option<usize>,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU memory limit in bytes
    pub gpu_memory_limit: Option<usize>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: None,
            cpu_threads: None, // Use all available
            enable_gpu: false,
            gpu_memory_limit: None,
        }
    }
}

/// Device types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    OpenCL,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization level
    pub level: OptimizationLevel,
    /// Enable SIMD operations
    pub enable_simd: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Enable kernel fusion
    pub enable_fusion: bool,
    /// Enable operator caching
    pub enable_caching: bool,
    /// JIT compilation settings
    pub jit_config: JitConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Balanced,
            enable_simd: true,
            enable_parallel: true,
            enable_fusion: true,
            enable_caching: true,
            jit_config: JitConfig::default(),
        }
    }
}

/// Optimization levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,      // No optimizations
    Balanced,   // Balanced speed/memory
    Speed,      // Optimize for speed
    Memory,     // Optimize for memory usage
    Aggressive, // All optimizations enabled
}

/// JIT compilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitConfig {
    /// Enable JIT compilation
    pub enabled: bool,
    /// Optimization level for JIT
    pub optimization_level: u8,
    /// Enable loop unrolling
    pub enable_unrolling: bool,
    /// Enable vectorization
    pub enable_vectorization: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_level: 2,
            enable_unrolling: true,
            enable_vectorization: true,
        }
    }
}

/// Batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Dynamic batching enabled
    pub dynamic_batching: bool,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Padding strategy for variable-length sequences
    pub padding_strategy: PaddingStrategy,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            dynamic_batching: true,
            batch_timeout_ms: 100,
            padding_strategy: PaddingStrategy::Right,
        }
    }
}

/// Padding strategies for batching
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    Left,
    Right,
    None,
}

impl EngineConfig {
    /// Create configuration from a server config (placeholder)
    pub fn from_server_config(_server_config: &crate::ServerConfig) -> Self {
        Self::default()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), crate::InferenceError> {
        // Validate model path exists
        if !self.model.path.exists() && !self.model.path.to_string_lossy().starts_with("http") {
            return Err(crate::InferenceError::configuration(
                format!("Model path does not exist: {:?}", self.model.path)
            ));
        }

        // Validate memory settings
        if self.memory.pool_size < (64usize << 20) {
            return Err(crate::InferenceError::configuration(
                "Pool size must be at least 64MB"
            ));
        }

        if self.memory.arena_size > self.memory.pool_size {
            return Err(crate::InferenceError::configuration(
                "Arena size cannot exceed pool size"
            ));
        }

        // Validate batch settings
        if self.batching.max_batch_size == 0 {
            return Err(crate::InferenceError::configuration(
                "Batch size must be greater than 0"
            ));
        }

        Ok(())
    }

    /// Create a configuration optimized for speed
    pub fn for_speed() -> Self {
        Self {
            optimization: OptimizationConfig {
                level: OptimizationLevel::Speed,
                ..Default::default()
            },
            device: DeviceConfig {
                enable_gpu: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a configuration optimized for memory usage
    pub fn for_memory() -> Self {
        Self {
            optimization: OptimizationConfig {
                level: OptimizationLevel::Memory,
                ..Default::default()
            },
            memory: MemoryConfig {
                pool_size: 512usize << 20, // 512MB
                arena_size: 128usize << 20, // 128MB
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

// Placeholder for ServerConfig - would be imported from server crate
pub struct ServerConfig;
