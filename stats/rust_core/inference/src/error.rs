//! Error types for the inference engine

use thiserror::Error;

/// Result type for inference operations
pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

/// Comprehensive error type for inference operations
#[derive(Error, Debug, Clone)]
pub enum InferenceError {
    #[error("Model loading error: {message}")]
    ModelLoad { message: String },

    #[error("Tokenization error: {message}")]
    Tokenization { message: String },

    #[error("Inference runtime error: {message}")]
    Runtime { message: String },

    #[error("Memory allocation error: {message}")]
    Memory { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("GPU error: {message}")]
    Gpu { message: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid tensor shape: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>
    },

    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },

    #[error("Timeout error: operation took longer than {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
}

impl InferenceError {
    /// Create a model loading error
    pub fn model_load(message: impl Into<String>) -> Self {
        Self::ModelLoad { message: message.into() }
    }

    /// Create a tokenization error
    pub fn tokenization(message: impl Into<String>) -> Self {
        Self::Tokenization { message: message.into() }
    }

    /// Create a runtime error
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime { message: message.into() }
    }

    /// Create a memory error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory { message: message.into() }
    }

    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration { message: message.into() }
    }

    /// Create a GPU error
    pub fn gpu(message: impl Into<String>) -> Self {
        Self::Gpu { message: message.into() }
    }

    /// Create an unsupported operation error
    pub fn unsupported(operation: impl Into<String>) -> Self {
        Self::UnsupportedOperation { operation: operation.into() }
    }

    /// Create a timeout error
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            InferenceError::ModelLoad { .. } => false,
            InferenceError::Configuration { .. } => false,
            InferenceError::UnsupportedOperation { .. } => false,
            InferenceError::Timeout { .. } => true,
            InferenceError::Runtime { .. } => true,
            InferenceError::Memory { .. } => true,
            InferenceError::Gpu { .. } => true,
            InferenceError::Tokenization { .. } => true,
            InferenceError::Io(_) => true,
            InferenceError::Serialization(_) => true,
            InferenceError::InvalidShape { .. } => false,
        }
    }
}

// Implement conversion from anyhow::Error for broader compatibility
impl From<anyhow::Error> for InferenceError {
    fn from(err: anyhow::Error) -> Self {
        InferenceError::Runtime {
            message: err.to_string()
        }
    }
}
