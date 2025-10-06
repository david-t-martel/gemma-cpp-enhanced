// Application services - Use case implementations
// These orchestrate domain logic and repository operations

pub mod document_service;
pub mod search_service;
pub mod memory_service;
pub mod research_service;

pub use document_service::DocumentService;
pub use search_service::SearchService;
pub use memory_service::MemoryService;
pub use research_service::ResearchService;

use std::sync::Arc;

/// Service registry for dependency injection
pub struct ServiceRegistry {
    pub document_service: Arc<dyn DocumentServiceTrait>,
    pub search_service: Arc<dyn SearchServiceTrait>,
    pub memory_service: Arc<dyn MemoryServiceTrait>,
    pub research_service: Arc<dyn ResearchServiceTrait>,
}

/// Common service error type
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("Repository error: {0}")]
    Repository(#[from] crate::core::repositories::RepositoryError),

    #[error("Domain error: {0}")]
    Domain(#[from] crate::core::domain::DomainError),

    #[error("External service error: {0}")]
    External(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Processing error: {0}")]
    Processing(String),
}

pub type Result<T> = std::result::Result<T, ServiceError>;

// Service traits for abstraction
use async_trait::async_trait;
use crate::core::domain::{Document, DocumentId, SearchResult, Memory, MemoryType};

#[async_trait]
pub trait DocumentServiceTrait: Send + Sync {
    async fn ingest(&self, content: String, metadata: Option<serde_json::Value>) -> Result<DocumentId>;
    async fn get(&self, id: &DocumentId) -> Result<Option<Document>>;
    async fn delete(&self, id: &DocumentId) -> Result<()>;
}

#[async_trait]
pub trait SearchServiceTrait: Send + Sync {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>>;
    async fn search_with_filters(&self, query: &str, filters: serde_json::Value, limit: usize) -> Result<Vec<SearchResult>>;
}

#[async_trait]
pub trait MemoryServiceTrait: Send + Sync {
    async fn store(&self, content: String, memory_type: MemoryType, importance: f32) -> Result<String>;
    async fn recall(&self, query: &str, memory_type: Option<MemoryType>, limit: usize) -> Result<Vec<Memory>>;
    async fn forget(&self, memory_id: &str) -> Result<()>;
}

#[async_trait]
pub trait ResearchServiceTrait: Send + Sync {
    async fn research(&self, query: &str, sources: Vec<String>) -> Result<Vec<SearchResult>>;
    async fn combine_results(&self, local: Vec<SearchResult>, external: Vec<SearchResult>) -> Result<Vec<SearchResult>>;
}
