//! Error handling for Gemma extensions
//!
//! Provides comprehensive error types and conversion utilities for the extension modules.

use pyo3::exceptions::{PyIOError, PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Result type alias for Gemma operations
pub type GemmaResult<T> = Result<T, GemmaError>;

/// Main error type for Gemma extensions
#[derive(Error, Debug)]
pub enum GemmaError {
    #[error("Tokenizer error: {message}")]
    TokenizerError { message: String },

    #[error("Tensor operation error: {message}")]
    TensorError { message: String },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid configuration: {message}")]
    ConfigError { message: String },

    #[error("Memory allocation error: {message}")]
    MemoryError { message: String },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Index out of bounds: index {index} >= length {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    #[error("Async operation error: {message}")]
    AsyncError { message: String },

    #[error("SIMD not supported on this platform")]
    SimdNotSupported,

    #[error("Invalid UTF-8 sequence")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("General error: {message}")]
    General { message: String },

    // RAG system specific errors
    #[error("Redis connection error: {0}")]
    RedisConnection(String),

    #[error("Redis operation error: {0}")]
    Redis(String),

    #[error("Vector store operation error: {0}")]
    VectorStore(String),

    #[error("Invalid vector dimension: expected {expected}, got {actual}")]
    InvalidVectorDimension { expected: usize, actual: usize },

    #[error("Document parsing error: {0}")]
    DocumentParsing(String),

    #[error("HTTP request error: {0}")]
    HttpRequest(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Blocked domain: {0}")]
    BlockedDomain(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl GemmaError {
    /// Create a new tokenizer error
    pub fn tokenizer<S: Into<String>>(message: S) -> Self {
        Self::TokenizerError {
            message: message.into(),
        }
    }

    /// Create a new tensor error
    pub fn tensor<S: Into<String>>(message: S) -> Self {
        Self::TensorError {
            message: message.into(),
        }
    }

    /// Create a new cache error
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::CacheError {
            message: message.into(),
        }
    }

    /// Create a new config error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Create a new memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// Create a new async error
    pub fn async_op<S: Into<String>>(message: S) -> Self {
        Self::AsyncError {
            message: message.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(index: usize, length: usize) -> Self {
        Self::IndexOutOfBounds { index, length }
    }

    /// Create a general error
    pub fn general<S: Into<String>>(message: S) -> Self {
        Self::General {
            message: message.into(),
        }
    }
}

/// Convert Rust errors to Python exceptions
impl From<GemmaError> for PyErr {
    fn from(error: GemmaError) -> PyErr {
        match error {
            GemmaError::TokenizerError { message } => {
                PyValueError::new_err(format!("Tokenizer error: {}", message))
            }
            GemmaError::TensorError { message } => {
                PyValueError::new_err(format!("Tensor operation error: {}", message))
            }
            GemmaError::CacheError { message } => {
                PyRuntimeError::new_err(format!("Cache error: {}", message))
            }
            GemmaError::IoError(e) => PyIOError::new_err(format!("I/O error: {}", e)),
            GemmaError::SerializationError(e) => {
                PyValueError::new_err(format!("Serialization error: {}", e))
            }
            GemmaError::ConfigError { message } => {
                PyValueError::new_err(format!("Configuration error: {}", message))
            }
            GemmaError::MemoryError { message } => {
                PyMemoryError::new_err(format!("Memory error: {}", message))
            }
            GemmaError::DimensionMismatch { expected, actual } => PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                expected, actual
            )),
            GemmaError::IndexOutOfBounds { index, length } => {
                PyValueError::new_err(format!("Index out of bounds: {} >= {}", index, length))
            }
            GemmaError::AsyncError { message } => {
                PyRuntimeError::new_err(format!("Async operation error: {}", message))
            }
            GemmaError::SimdNotSupported => {
                PyRuntimeError::new_err("SIMD operations not supported on this platform")
            }
            GemmaError::Utf8Error(e) => {
                PyValueError::new_err(format!("Invalid UTF-8 sequence: {}", e))
            }
            GemmaError::General { message } => PyRuntimeError::new_err(message),

            // RAG system specific error conversions
            GemmaError::RedisConnection(msg) => {
                PyRuntimeError::new_err(format!("Redis connection error: {}", msg))
            }
            GemmaError::Redis(msg) => PyRuntimeError::new_err(format!("Redis error: {}", msg)),
            GemmaError::VectorStore(msg) => {
                PyRuntimeError::new_err(format!("Vector store error: {}", msg))
            }
            GemmaError::InvalidVectorDimension { expected, actual } => {
                PyValueError::new_err(format!(
                    "Invalid vector dimension: expected {}, got {}",
                    expected, actual
                ))
            }
            GemmaError::DocumentParsing(msg) => {
                PyValueError::new_err(format!("Document parsing error: {}", msg))
            }
            GemmaError::HttpRequest(msg) => {
                PyRuntimeError::new_err(format!("HTTP request error: {}", msg))
            }
            GemmaError::Network(msg) => PyRuntimeError::new_err(format!("Network error: {}", msg)),
            GemmaError::BlockedDomain(domain) => {
                PyValueError::new_err(format!("Blocked domain: {}", domain))
            }
            GemmaError::NotImplemented(msg) => {
                PyRuntimeError::new_err(format!("Not implemented: {}", msg))
            }
            GemmaError::InvalidArgument(msg) => {
                PyValueError::new_err(format!("Invalid argument: {}", msg))
            }
        }
    }
}

/// Error handling utilities
pub mod utils {
    use super::*;

    /// Convert a generic error to GemmaError
    pub fn to_gemma_error<E: std::fmt::Display>(error: E) -> GemmaError {
        GemmaError::general(error.to_string())
    }

    /// Convert a result with generic error to GemmaResult
    pub fn to_gemma_result<T, E: std::fmt::Display>(result: Result<T, E>) -> GemmaResult<T> {
        result.map_err(to_gemma_error)
    }

    /// Chain errors with context
    pub fn with_context<T, E: std::fmt::Display>(
        result: Result<T, E>,
        context: &str,
    ) -> GemmaResult<T> {
        result.map_err(|e| GemmaError::general(format!("{}: {}", context, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = GemmaError::tokenizer("test message");
        assert!(matches!(error, GemmaError::TokenizerError { .. }));
        assert_eq!(error.to_string(), "Tokenizer error: test message");
    }

    #[test]
    fn test_dimension_mismatch() {
        let error = GemmaError::dimension_mismatch(10, 5);
        assert!(matches!(error, GemmaError::DimensionMismatch { .. }));
        assert_eq!(error.to_string(), "Dimension mismatch: expected 10, got 5");
    }

    #[test]
    fn test_python_conversion() {
        let error = GemmaError::tokenizer("test");
        let py_err: PyErr = error.into();
        // Test just confirms error conversion works - can't test Python instance without GIL
        assert!(format!("{}", py_err).contains("Tokenizer error"));
    }
}
