use thiserror::Error;

#[derive(Error, Debug)]
pub enum GemmaError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[error("Document parsing error: {0}")]
    DocumentParsing(String),
    #[error("Redis error: {0}")]
    Redis(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

pub type GemmaResult<T> = Result<T, GemmaError>;
