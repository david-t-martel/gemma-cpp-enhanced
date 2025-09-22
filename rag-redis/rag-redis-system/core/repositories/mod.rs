// Repository traits - Ports for data persistence
// These are abstractions that the domain uses, implemented by infrastructure

use async_trait::async_trait;
use crate::core::domain::{
    Document, DocumentId, Chunk, ChunkId,
    Memory, MemoryId, MemoryType, SearchResult
};

pub type Result<T> = std::result::Result<T, RepositoryError>;

#[derive(Debug, thiserror::Error)]
pub enum RepositoryError {
    #[error("Entity not found: {0}")]
    NotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),
}

/// Repository for document persistence
#[async_trait]
pub trait DocumentRepository: Send + Sync {
    /// Save a document
    async fn save(&self, document: &Document) -> Result<()>;

    /// Find a document by ID
    async fn find_by_id(&self, id: &DocumentId) -> Result<Option<Document>>;

    /// Delete a document
    async fn delete(&self, id: &DocumentId) -> Result<()>;

    /// List all documents with pagination
    async fn list(&self, offset: usize, limit: usize) -> Result<Vec<Document>>;

    /// Count total documents
    async fn count(&self) -> Result<usize>;
}

/// Repository for chunk persistence
#[async_trait]
pub trait ChunkRepository: Send + Sync {
    /// Save a chunk
    async fn save(&self, chunk: &Chunk) -> Result<()>;

    /// Save multiple chunks in batch
    async fn save_batch(&self, chunks: &[Chunk]) -> Result<()>;

    /// Find a chunk by ID
    async fn find_by_id(&self, id: &ChunkId) -> Result<Option<Chunk>>;

    /// Find chunks by document ID
    async fn find_by_document(&self, doc_id: &DocumentId) -> Result<Vec<Chunk>>;

    /// Delete chunks by document ID
    async fn delete_by_document(&self, doc_id: &DocumentId) -> Result<()>;
}

/// Repository for vector operations
#[async_trait]
pub trait VectorRepository: Send + Sync {
    /// Index vectors with their IDs
    async fn index(&self, vectors: Vec<(String, Vec<f32>)>) -> Result<()>;

    /// Search for similar vectors
    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        threshold: Option<f32>
    ) -> Result<Vec<SearchResult>>;

    /// Remove a vector by ID
    async fn remove(&self, id: &str) -> Result<()>;

    /// Update a vector
    async fn update(&self, id: &str, vector: &[f32]) -> Result<()>;

    /// Get vector dimension
    fn dimension(&self) -> usize;
}

/// Repository for memory persistence
#[async_trait]
pub trait MemoryRepository: Send + Sync {
    /// Store a memory
    async fn save(&self, memory: &Memory) -> Result<()>;

    /// Find a memory by ID
    async fn find_by_id(&self, id: &MemoryId) -> Result<Option<Memory>>;

    /// Find memories by type
    async fn find_by_type(&self, memory_type: MemoryType, limit: usize) -> Result<Vec<Memory>>;

    /// Search memories by content
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<Memory>>;

    /// Delete old memories based on type and age
    async fn cleanup(&self, memory_type: MemoryType, max_age_days: u32) -> Result<usize>;
}

/// Repository for embedding cache
#[async_trait]
pub trait EmbeddingCacheRepository: Send + Sync {
    /// Get cached embedding
    async fn get(&self, text_hash: &str) -> Result<Option<Vec<f32>>>;

    /// Store embedding in cache
    async fn set(&self, text_hash: &str, embedding: &[f32]) -> Result<()>;

    /// Check if embedding exists
    async fn exists(&self, text_hash: &str) -> Result<bool>;

    /// Clear old cache entries
    async fn cleanup(&self, max_age_hours: u32) -> Result<usize>;
}
