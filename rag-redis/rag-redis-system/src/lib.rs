pub mod agent_memory;
pub mod config;
pub mod document;
pub mod embedding;
pub mod error;
pub mod memory;
pub mod memory_archive;
pub mod memory_retrieval;
pub mod metrics;
pub mod project_context;
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

pub use agent_memory::{AgentMemorySystem, AgentType, ContextHint, AgentMemory, MemoryTemplate, MemoryDigest, AgentMemoryConfig};
pub use config::{Config, EmbeddingProvider};
pub use document::{Document, DocumentPipeline};
pub use embedding::EmbeddingModel;
pub use error::{Error, Result};
pub use memory::{MemoryManager, MemoryType};
pub use project_context::{
    ProjectContextManager, ProjectSnapshot, SaveOptions, LoadOptions, ContextDiff,
    ProjectFiles, ProjectConfiguration, ProjectMemories, ProjectVectors, ProjectMetadata,
    SnapshotType, SessionType, FileContent, MemoryStatistics, ComplexityMetrics, ActivityMetrics,
};
pub use redis_backend::{RedisClient, RedisHealthClient, RedisManager};
pub use research::{ResearchClient, SearchResult};
pub use vector_store::{VectorIndex, VectorStore};

use embedding::{EmbeddingFactory, EmbeddingService};
use std::sync::Arc;
use tokio::sync::{RwLock, OnceCell};
use std::sync::OnceLock;

pub struct RagSystem {
    pub config: Arc<Config>,
    // Core component - initialized immediately
    redis_manager: Arc<RedisManager>,

    // Lazy components - initialized on first use
    vector_store: OnceCell<Arc<RwLock<VectorStore>>>,
    document_pipeline: OnceLock<Arc<DocumentPipeline>>,
    memory_manager: OnceCell<Arc<MemoryManager>>,
    research_client: OnceLock<Arc<ResearchClient>>,
    embedding_service: OnceCell<Arc<Box<dyn EmbeddingService>>>,
    metrics: OnceLock<Arc<metrics::MetricsCollector>>,
    project_context_manager: OnceCell<Arc<project_context::ProjectContextManager>>,
}

impl RagSystem {
    /// Fast startup - only initialize critical path (Redis connection)
    pub async fn new(config: Config) -> Result<Self> {
        let config = Arc::new(config);

        // Only initialize Redis connection immediately (critical for MCP health checks)
        let redis_manager = Arc::new(RedisManager::new(&config.redis).await?);

        tracing::info!("RAG system initialized in fast-start mode");

        Ok(Self {
            config,
            redis_manager,
            vector_store: OnceCell::new(),
            document_pipeline: OnceLock::new(),
            memory_manager: OnceCell::new(),
            research_client: OnceLock::new(),
            embedding_service: OnceCell::new(),
            metrics: OnceLock::new(),
            project_context_manager: OnceCell::new(),
        })
    }

    /// Warm up all components for better first-request performance
    pub async fn warm_up(&self) -> Result<()> {
        tracing::info!("Warming up RAG system components");

        // Pre-initialize all components in parallel
        tokio::try_join!(
            async { self.vector_store().await.map(|_| ()) },
            async { self.embedding_service().await.map(|_| ()) },
            async { self.memory_manager().await.map(|_| ()) }
        )?;

        tracing::info!("RAG system warm-up completed");
        Ok(())
    }

    /// Lazy getter for vector store
    async fn vector_store(&self) -> Result<&Arc<RwLock<VectorStore>>> {
        self.vector_store.get_or_try_init(|| async {
            tracing::debug!("Lazy initializing vector store");
            let store = VectorStore::new(self.config.vector_store.clone())?;
            Ok(Arc::new(RwLock::new(store)))
        }).await
    }

    /// Lazy getter for document pipeline
    fn document_pipeline(&self) -> &Arc<DocumentPipeline> {
        self.document_pipeline.get_or_init(|| {
            tracing::debug!("Lazy initializing document pipeline");
            Arc::new(DocumentPipeline::new(self.config.document.clone()))
        })
    }

    /// Lazy getter for memory manager
    async fn memory_manager(&self) -> Result<&Arc<MemoryManager>> {
        self.memory_manager.get_or_try_init(|| async {
            tracing::debug!("Lazy initializing memory manager");
            MemoryManager::new(
                self.redis_manager.clone(),
                self.config.memory.clone().into()
            ).await
        }).await
    }

    /// Lazy getter for research client
    fn research_client(&self) -> Result<&Arc<ResearchClient>> {
        self.research_client.get_or_init(|| {
            tracing::debug!("Lazy initializing research client");
            Arc::new(ResearchClient::new(self.config.research.clone())
                .expect("Failed to create research client"))
        });
        Ok(self.research_client.get().unwrap())
    }

    /// Lazy getter for embedding service
    async fn embedding_service(&self) -> Result<&Arc<Box<dyn EmbeddingService>>> {
        self.embedding_service.get_or_try_init(|| async {
            tracing::debug!("Lazy initializing embedding service");
            let service = EmbeddingFactory::create(&self.config.embedding).await?;
            Ok(Arc::new(service))
        }).await
    }

    /// Lazy getter for metrics collector
    fn metrics(&self) -> &Arc<metrics::MetricsCollector> {
        self.metrics.get_or_init(|| {
            tracing::debug!("Lazy initializing metrics collector");

            #[cfg(feature = "metrics")]
            let collector = metrics::MetricsCollector::new(&self.config.metrics)
                .expect("Failed to create metrics collector");

            #[cfg(not(feature = "metrics"))]
            let collector = metrics::MetricsCollector::new(&())
                .expect("Failed to create metrics collector");

            Arc::new(collector)
        })
    }

    /// Lazy getter for project context manager
    async fn project_context_manager(&self) -> Result<&Arc<project_context::ProjectContextManager>> {
        self.project_context_manager.get_or_try_init(|| async {
            tracing::debug!("Lazy initializing project context manager");
            let manager = project_context::ProjectContextManager::new(self.redis_manager.clone()).await?;

            // Connect with memory manager if available
            if let Ok(memory_manager) = self.memory_manager().await {
                Ok(Arc::new(manager.with_memory_manager(memory_manager.clone())))
            } else {
                Ok(Arc::new(manager))
            }
        }).await
    }

    pub async fn ingest_document(
        &self,
        content: &str,
        metadata: serde_json::Value,
    ) -> Result<String> {
        let document = self.document_pipeline().process(content, metadata).await?;
        let doc_id = document.id.clone();

        let chunks = self.document_pipeline().chunk_document(&document)?;

        for chunk in chunks {
            let embedding = self.get_embedding(&chunk.text).await?;

            self.vector_store().await?
                .write().await.add_vector(
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
            self.vector_store().await?
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
            self.research_client()?.search_web(query, sources).await?
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
        self.embedding_service().await?
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
            .memory_manager().await?
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

        let memories = self.memory_manager().await?.recall(&query, mem_type, limit).await?;

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
    pub fn metrics_collector(&self) -> &Arc<metrics::MetricsCollector> {
        self.metrics()
    }

    /// Handle MCP save_project_context request
    pub async fn handle_save_project_context(
        &self,
        project_id: String,
        project_root: Option<String>,
        options: Option<serde_json::Value>,
    ) -> Result<String> {
        let project_root = if let Some(root) = project_root {
            std::path::PathBuf::from(root)
        } else {
            std::env::current_dir()
                .map_err(|e| Error::Api(format!("Failed to get current directory: {}", e)))?
        };

        let save_options = if let Some(opts) = options {
            serde_json::from_value(opts)
                .unwrap_or_else(|_| project_context::SaveOptions::default())
        } else {
            project_context::SaveOptions::default()
        };

        self.project_context_manager().await?
            .save_project_context(&project_id, &project_root, save_options)
            .await
    }

    /// Handle MCP load_project_context request
    pub async fn handle_load_project_context(
        &self,
        project_id: String,
        snapshot_id: Option<String>,
    ) -> Result<serde_json::Value> {
        let snapshot = self.project_context_manager().await?
            .load_project_context(&project_id, snapshot_id.as_deref())
            .await?;

        serde_json::to_value(snapshot)
            .map_err(|e| Error::Api(format!("Failed to serialize snapshot: {}", e)))
    }

    /// Handle MCP quick_save_session request
    pub async fn handle_quick_save_session(
        &self,
        project_id: String,
        description: String,
    ) -> Result<String> {
        self.project_context_manager().await?
            .quick_save_session(&project_id, &description)
            .await
    }

    /// Handle MCP quick_load_session request
    pub async fn handle_quick_load_session(
        &self,
        project_id: String,
        session_id: String,
    ) -> Result<serde_json::Value> {
        let snapshot = self.project_context_manager().await?
            .quick_load_session(&project_id, &session_id)
            .await?;

        serde_json::to_value(snapshot)
            .map_err(|e| Error::Api(format!("Failed to serialize snapshot: {}", e)))
    }

    /// Handle MCP diff_contexts request
    pub async fn handle_diff_contexts(
        &self,
        project_id: String,
        from_version: String,
        to_version: String,
    ) -> Result<serde_json::Value> {
        let diff = self.project_context_manager().await?
            .diff_contexts(&project_id, &from_version, &to_version)
            .await?;

        serde_json::to_value(diff)
            .map_err(|e| Error::Api(format!("Failed to serialize diff: {}", e)))
    }

    /// Handle MCP list_project_snapshots request
    pub async fn handle_list_project_snapshots(
        &self,
        project_id: String,
        limit: Option<usize>,
    ) -> Result<Vec<serde_json::Value>> {
        let snapshots = self.project_context_manager().await?
            .list_project_snapshots(&project_id, limit)
            .await?;

        let mut formatted_snapshots = Vec::new();
        for snapshot in snapshots {
            formatted_snapshots.push(serde_json::json!({
                "id": snapshot.id,
                "project_id": snapshot.project_id,
                "version": snapshot.version,
                "name": snapshot.name,
                "description": snapshot.description,
                "created_at": snapshot.created_at.to_rfc3339(),
                "created_by": snapshot.created_by,
                "snapshot_type": format!("{:?}", snapshot.snapshot_metadata.snapshot_type),
                "storage_size": snapshot.snapshot_metadata.storage_size,
                "compression_ratio": snapshot.snapshot_metadata.compression_ratio,
                "file_count": snapshot.project_files.file_count,
                "total_size": snapshot.project_files.total_size,
                "programming_languages": snapshot.project_metadata.programming_languages,
                "frameworks": snapshot.project_metadata.frameworks,
            }));
        }

        Ok(formatted_snapshots)
    }

    /// Handle MCP get_project_statistics request
    pub async fn handle_get_project_statistics(
        &self,
        project_id: String,
    ) -> Result<serde_json::Value> {
        let stats = self.project_context_manager().await?
            .get_project_statistics(&project_id)
            .await?;

        serde_json::to_value(stats)
            .map_err(|e| Error::Api(format!("Failed to serialize statistics: {}", e)))
    }

    /// Handle MCP cleanup_old_snapshots request
    pub async fn handle_cleanup_old_snapshots(
        &self,
        project_id: String,
    ) -> Result<usize> {
        self.project_context_manager().await?
            .cleanup_old_snapshots(&project_id)
            .await
    }
}
