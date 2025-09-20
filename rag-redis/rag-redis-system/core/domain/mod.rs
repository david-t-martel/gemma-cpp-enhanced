// Core domain entities - Pure business logic with no external dependencies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// Domain entities
pub mod document;
pub mod chunk;
pub mod embedding;
pub mod memory;

pub use document::{Document, DocumentId, DocumentMetadata};
pub use chunk::{Chunk, ChunkId, ChunkMetadata};
pub use embedding::{Embedding, EmbeddingVector};
pub use memory::{Memory, MemoryId, MemoryType};

// Value objects
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score(f32);

impl Score {
    pub fn new(value: f32) -> Result<Self, DomainError> {
        if value < 0.0 || value > 1.0 {
            return Err(DomainError::InvalidScore(value));
        }
        Ok(Score(value))
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub score: Score,
    pub metadata: HashMap<String, serde_json::Value>,
}

// Domain errors
#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("Invalid score value: {0}. Must be between 0.0 and 1.0")]
    InvalidScore(f32),

    #[error("Invalid document: {0}")]
    InvalidDocument(String),

    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(usize),

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },
}

// Domain events (for event sourcing if needed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainEvent {
    DocumentIngested {
        document_id: DocumentId,
        timestamp: DateTime<Utc>,
    },
    ChunkCreated {
        chunk_id: ChunkId,
        document_id: DocumentId,
        timestamp: DateTime<Utc>,
    },
    MemoryStored {
        memory_id: MemoryId,
        memory_type: MemoryType,
        timestamp: DateTime<Utc>,
    },
}
