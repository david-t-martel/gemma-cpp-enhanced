pub mod config;
pub mod document;
pub mod embedding;
pub mod error;
pub mod memory;
pub mod metrics;
pub mod redis_backend;
pub mod research;
pub mod vector_store;

// Memory optimization modules
pub mod memory_dashboard;
pub mod memory_pool;
pub mod memory_profiler;
pub mod smart_cache;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(test)]
pub mod performance_test;

pub use config::{Config, EmbeddingProvider};
pub use document::{Document, DocumentPipeline};
pub use embedding::EmbeddingModel;
pub use error::{Error, Result};
pub use memory::{MemoryManager, MemoryType};
pub use redis_backend::{RedisClient, RedisHealthClient, RedisManager};
pub use research::{ResearchClient, SearchResult};
pub use vector_store::{VectorIndex, VectorStore};

use embedding::{EmbeddingFactory, EmbeddingService};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct RagSystem {
    pub config: Arc<Config>,
    redis_manager: Arc<RedisManager>,
    vector_store: Arc<RwLock<VectorStore>>,
    document_pipeline: Arc<DocumentPipeline>,
    memory_manager: Arc<MemoryManager>,
    research_client: Arc<ResearchClient>,
    embedding_service: Arc<Box<dyn EmbeddingService>>,
    metrics: Arc<metrics::MetricsCollector>,
}

impl RagSystem {
    pub async fn new(config: Config) -> Result<Self> {
        let config = Arc::new(config);

        let redis_manager = Arc::new(RedisManager::new(&config.redis).await?);

        let vector_store = Arc::new(RwLock::new(VectorStore::new(config.vector_store.clone())?));

        let document_pipeline = Arc::new(DocumentPipeline::new(config.document.clone()));

        let memory_manager =
            MemoryManager::new(redis_manager.clone(), config.memory.clone().into()).await?;

        let research_client = Arc::new(ResearchClient::new(config.research.clone())?);

        let embedding_service = Arc::new(EmbeddingFactory::create(&config.embedding).await?);

        #[cfg(feature = "metrics")]
        let metrics = Arc::new(metrics::MetricsCollector::new(&config.metrics)?);

        #[cfg(not(feature = "metrics"))]
        let metrics = Arc::new(metrics::MetricsCollector::new(&())?);

        Ok(Self {
            config,
            redis_manager,
            vector_store,
            document_pipeline,
            memory_manager,
            research_client,
            embedding_service,
            metrics,
        })
    }

    pub async fn ingest_document(
        &self,
        content: &str,
        metadata: serde_json::Value,
    ) -> Result<String> {
        let document = self.document_pipeline.process(content, metadata).await?;
        let doc_id = document.id.clone();

        let chunks = self.document_pipeline.chunk_document(&document)?;

        for chunk in chunks {
            let embedding = self.get_embedding(&chunk.text).await?;

            self.vector_store.write().await.add_vector(
                &chunk.id,
                &embedding,
                chunk.metadata.clone(),
            )?;

            self.redis_manager.store_chunk(&chunk).await?;
        }

        self.redis_manager.store_document(&document).await?;

        Ok(doc_id)
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.get_embedding(query).await?;

        let vector_results =
            self.vector_store
                .read()
                .await
                .search(&query_embedding, limit * 2, None)?;

        let mut results = Vec::new();
        for (id, score, metadata) in vector_results {
            if let Ok(chunk) = self.redis_manager.get_chunk(&id).await {
                results.push(SearchResult {
                    id,
                    text: chunk.text,
                    score,
                    metadata,
                });
            }
        }

        results.truncate(limit);
        Ok(results)
    }

    pub async fn research(&self, query: &str, sources: Vec<String>) -> Result<Vec<SearchResult>> {
        let local_results = self.search(query, 5).await?;

        let web_results = if !sources.is_empty() {
            self.research_client.search_web(query, sources).await?
        } else {
            Vec::new()
        };

        let mut combined_results = local_results;
        combined_results.extend(web_results);

        combined_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        combined_results.truncate(10);

        Ok(combined_results)
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding_service
            .embed_text(text)
            .await
            .map_err(|e| Error::Api(format!("Embedding service error: {}", e)))
    }

    /// Public getter for config field
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    /// Public getter for redis_manager field
    pub fn redis_manager(&self) -> &Arc<RedisManager> {
        &self.redis_manager
    }

    /// Handle MCP ingest_document request
    pub async fn handle_ingest_document(
        &self,
        content: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let metadata = metadata.unwrap_or_else(|| serde_json::json!({}));
        self.ingest_document(&content, metadata).await
    }

    /// Handle MCP search request
    pub async fn handle_search(
        &self,
        query: String,
        limit: usize,
        _threshold: Option<f32>,
    ) -> Result<Vec<serde_json::Value>> {
        let results = self.search(&query, limit).await?;

        let mut formatted_results = Vec::new();
        for result in results {
            formatted_results.push(serde_json::json!({
                "id": result.id,
                "content": result.text,
                "score": result.score,
                "metadata": result.metadata
            }));
        }

        Ok(formatted_results)
    }

    /// Handle MCP store_memory request
    pub async fn handle_store_memory(
        &self,
        content: String,
        memory_type: String,
        importance: f32,
    ) -> Result<String> {
        use crate::memory::MemoryType;

        let mem_type = match memory_type.as_str() {
            "short_term" => MemoryType::ShortTerm,
            "long_term" => MemoryType::LongTerm,
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            "working" => MemoryType::Working,
            _ => return Err(Error::Api(format!("Invalid memory type: {}", memory_type))),
        };

        let memory_id = self
            .memory_manager
            .store(content, mem_type, importance)
            .await?;
        Ok(memory_id)
    }

    /// Handle MCP recall_memory request
    pub async fn handle_recall_memory(
        &self,
        query: String,
        memory_type: Option<String>,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        use crate::memory::MemoryType;

        let mem_type = memory_type.and_then(|t| match t.as_str() {
            "short_term" => Some(MemoryType::ShortTerm),
            "long_term" => Some(MemoryType::LongTerm),
            "episodic" => Some(MemoryType::Episodic),
            "semantic" => Some(MemoryType::Semantic),
            "working" => Some(MemoryType::Working),
            _ => None,
        });

        let memories = self.memory_manager.recall(&query, mem_type, limit).await?;

        let mut formatted_memories = Vec::new();
        for memory in memories {
            formatted_memories.push(serde_json::json!({
                "id": memory.id,
                "content": memory.content,
                "memory_type": format!("{:?}", memory.memory_type),
                "importance": memory.importance_score,
                "created_at": memory.created_at.to_rfc3339()
            }));
        }

        Ok(formatted_memories)
    }

    /// Get metrics collector for monitoring
    pub fn metrics(&self) -> &Arc<metrics::MetricsCollector> {
        &self.metrics
    }
}
