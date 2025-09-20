use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Redis error: {0}")]
    Redis(String),

    #[error("Vector store error: {0}")]
    VectorStore(String),

    #[error("Document processing error: {0}")]
    DocumentProcessing(String),

    #[error("Research error: {0}")]
    Research(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Network error: {0}")]
    Network(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("FFI error: {0}")]
    Ffi(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<redis::RedisError> for Error {
    fn from(e: redis::RedisError) -> Self {
        Error::Redis(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(e: bincode::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Network(e.to_string())
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(e: ndarray::ShapeError) -> Self {
        Error::VectorStore(e.to_string())
    }
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Api(e.to_string())
    }
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Error::Unknown(e.to_string())
    }
}

#[cfg(feature = "metrics")]
impl From<prometheus::Error> for Error {
    fn from(e: prometheus::Error) -> Self {
        Error::Config(format!("Prometheus error: {}", e))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Success = 0,
    InvalidInput = 1,
    NotFound = 2,
    Redis = 3,
    VectorStore = 4,
    DocumentProcessing = 5,
    Research = 6,
    Memory = 7,
    Config = 8,
    Io = 9,
    Serialization = 10,
    RateLimitExceeded = 11,
    Network = 12,
    Api = 13,
    Ffi = 14,
    Unknown = 999,
}

impl From<&Error> for ErrorCode {
    fn from(error: &Error) -> Self {
        match error {
            Error::InvalidInput(_) => ErrorCode::InvalidInput,
            Error::NotFound(_) => ErrorCode::NotFound,
            Error::Redis(_) => ErrorCode::Redis,
            Error::VectorStore(_) => ErrorCode::VectorStore,
            Error::DocumentProcessing(_) => ErrorCode::DocumentProcessing,
            Error::Research(_) => ErrorCode::Research,
            Error::Memory(_) => ErrorCode::Memory,
            Error::Config(_) => ErrorCode::Config,
            Error::Io(_) => ErrorCode::Io,
            Error::Serialization(_) => ErrorCode::Serialization,
            Error::RateLimitExceeded => ErrorCode::RateLimitExceeded,
            Error::Network(_) => ErrorCode::Network,
            Error::Api(_) => ErrorCode::Api,
            Error::Ffi(_) => ErrorCode::Ffi,
            _ => ErrorCode::Unknown,
        }
    }
}
