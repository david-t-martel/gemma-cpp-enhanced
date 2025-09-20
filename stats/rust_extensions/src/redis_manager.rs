//! Redis memory management module for RAG system
//!
//! This module provides high-performance Redis integration for:
//! - Document storage and retrieval
//! - Vector embedding caching
//! - Search result caching
//! - Session management
//! - Connection pooling

use crate::error::{GemmaError, GemmaResult};
use bb8::{Pool, PooledConnection};
use bb8_redis::RedisConnectionManager;
use futures::future::try_join_all;
use redis::{aio::ConnectionLike, AsyncCommands, RedisResult};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Configuration for Redis connection and behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL (e.g., "redis://localhost:6379")
    pub url: String,
    /// Maximum number of connections in the pool
    pub max_connections: u32,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Default expiration time for cached items in seconds
    pub default_ttl: u64,
    /// Prefix for all Redis keys to avoid collisions
    pub key_prefix: String,
    /// Enable Redis pipelining for batch operations
    pub enable_pipelining: bool,
    /// Maximum retry attempts for failed operations
    pub max_retries: u32,
}

impl Default for RedisConfig {
    fn default() -> Self {
        // Auto-detect environment and use appropriate Redis configuration
        let default_url = if std::env::var("DOCKER_CONTAINER").is_ok() {
            "redis://redis:6379".to_string()
        } else if cfg!(target_os = "windows") {
            "redis://127.0.0.1:6380".to_string()
        } else {
            "redis://127.0.0.1:6379".to_string()
        };

        Self {
            url: std::env::var("REDIS_URL").unwrap_or(default_url),
            max_connections: std::env::var("REDIS_MAX_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(20),
            connection_timeout_ms: std::env::var("REDIS_CONNECTION_TIMEOUT")
                .ok()
                .and_then(|s| s.parse::<u64>().ok().map(|t| t * 1000))
                .unwrap_or(5000),
            default_ttl: std::env::var("REDIS_DEFAULT_TTL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3600), // 1 hour
            key_prefix: std::env::var("REDIS_KEY_PREFIX")
                .unwrap_or_else(|_| "rag:".to_string()),
            enable_pipelining: std::env::var("REDIS_ENABLE_PIPELINING")
                .ok()
                .map(|s| s.to_lowercase() != "false")
                .unwrap_or(true),
            max_retries: std::env::var("REDIS_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
        }
    }
}

/// Statistics for Redis operations
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RedisStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_operations: u64,
    pub failed_operations: u64,
    pub average_response_time_ms: f64,
    pub active_connections: u32,
}

impl RedisStats {
    pub fn hit_rate(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.total_operations - self.failed_operations) as f64 / self.total_operations as f64
        }
    }
}

/// Document metadata stored alongside document content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub content_type: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub chunk_count: usize,
    pub embedding_model: Option<String>,
    pub tags: Vec<String>,
    pub language: Option<String>,
}

impl DocumentMetadata {
    pub fn new(id: String, content_type: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id,
            title: None,
            url: None,
            content_type,
            created_at: timestamp,
            updated_at: timestamp,
            chunk_count: 0,
            embedding_model: None,
            tags: Vec::new(),
            language: None,
        }
    }
}

/// Redis-backed memory manager for RAG system
pub struct RedisManager {
    pool: Pool<RedisConnectionManager>,
    config: RedisConfig,
    stats: Arc<RwLock<RedisStats>>,
}

impl RedisManager {
    /// Create a new Redis manager with the given configuration
    pub async fn new(config: RedisConfig) -> GemmaResult<Self> {
        let manager = RedisConnectionManager::new(config.url.as_str())
            .map_err(|e| GemmaError::RedisConnection(e.to_string()))?;

        let pool = Pool::builder()
            .max_size(config.max_connections)
            .connection_timeout(Duration::from_millis(config.connection_timeout_ms))
            .build(manager)
            .await
            .map_err(|e| GemmaError::RedisConnection(e.to_string()))?;

        let stats = Arc::new(RwLock::new(RedisStats::default()));

        info!(
            "Redis manager initialized with {} max connections",
            config.max_connections
        );

        Ok(Self {
            pool,
            config,
            stats,
        })
    }

    /// Get a connection from the pool
    async fn get_connection(
        &self,
    ) -> GemmaResult<PooledConnection<'_, RedisConnectionManager>> {
        self.pool
            .get()
            .await
            .map_err(|e| GemmaError::RedisConnection(e.to_string()))
    }

    /// Generate a prefixed key
    fn make_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }

    /// Store a document with metadata
    pub async fn store_document(
        &self,
        content: &str,
        metadata: &DocumentMetadata,
    ) -> GemmaResult<()> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let doc_key = self.make_key(&format!("doc:{}", metadata.id));
        let meta_key = self.make_key(&format!("meta:{}", metadata.id));

        // Use a pipeline for atomic storage
        let mut pipe = redis::pipe();
        pipe.set_ex(&doc_key, content, self.config.default_ttl)
            .set_ex(
                &meta_key,
                serde_json::to_string(metadata)
                    .map_err(|e| GemmaError::Serialization(e.to_string()))?,
                self.config.default_ttl,
            )
            .sadd(self.make_key("docs"), &metadata.id);

        pipe.query_async(&mut *conn)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, true).await;
        debug!("Stored document: {}", metadata.id);
        Ok(())
    }

    /// Retrieve a document by ID
    pub async fn get_document(&self, id: &str) -> GemmaResult<Option<(String, DocumentMetadata)>> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let doc_key = self.make_key(&format!("doc:{}", id));
        let meta_key = self.make_key(&format!("meta:{}", id));

        // Use a pipeline for efficient retrieval
        let mut pipe = redis::pipe();
        pipe.get(&doc_key).get(&meta_key);

        let (content, metadata_str): (Option<String>, Option<String>) = pipe
            .query_async(&mut *conn)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, content.is_some()).await;

        match (content, metadata_str) {
            (Some(content), Some(metadata_str)) => {
                let metadata: DocumentMetadata = serde_json::from_str(&metadata_str)
                    .map_err(|e| GemmaError::Serialization(e.to_string()))?;
                Ok(Some((content, metadata)))
            }
            _ => Ok(None),
        }
    }

    /// Store vector embeddings for a document chunk
    pub async fn store_embedding(
        &self,
        doc_id: &str,
        chunk_id: usize,
        embedding: &[f32],
        text: &str,
    ) -> GemmaResult<()> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let embedding_key = self.make_key(&format!("emb:{}:{}", doc_id, chunk_id));
        let text_key = self.make_key(&format!("chunk:{}:{}", doc_id, chunk_id));

        // Serialize embedding as bytes for efficient storage
        let embedding_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut pipe = redis::pipe();
        pipe.set_ex(&embedding_key, &embedding_bytes, self.config.default_ttl)
            .set_ex(&text_key, text, self.config.default_ttl)
            .sadd(self.make_key(&format!("chunks:{}", doc_id)), chunk_id);

        pipe.query_async(&mut *conn)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, true).await;
        debug!("Stored embedding for {}:{}", doc_id, chunk_id);
        Ok(())
    }

    /// Retrieve vector embedding for a document chunk
    pub async fn get_embedding(
        &self,
        doc_id: &str,
        chunk_id: usize,
    ) -> GemmaResult<Option<(Vec<f32>, String)>> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let embedding_key = self.make_key(&format!("emb:{}:{}", doc_id, chunk_id));
        let text_key = self.make_key(&format!("chunk:{}:{}", doc_id, chunk_id));

        let mut pipe = redis::pipe();
        pipe.get(&embedding_key).get(&text_key);

        let (embedding_bytes, text): (Option<Vec<u8>>, Option<String>) = pipe
            .query_async(&mut *conn)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, embedding_bytes.is_some()).await;

        match (embedding_bytes, text) {
            (Some(bytes), Some(text)) => {
                // Deserialize bytes back to f32 vector
                let embedding: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Some((embedding, text)))
            }
            _ => Ok(None),
        }
    }

    /// Batch store multiple embeddings
    pub async fn batch_store_embeddings(
        &self,
        doc_id: &str,
        embeddings: &[(usize, Vec<f32>, String)],
    ) -> GemmaResult<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let mut pipe = redis::pipe();

        for (chunk_id, embedding, text) in embeddings {
            let embedding_key = self.make_key(&format!("emb:{}:{}", doc_id, chunk_id));
            let text_key = self.make_key(&format!("chunk:{}:{}", doc_id, chunk_id));

            let embedding_bytes: Vec<u8> = embedding
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            pipe.set_ex(&embedding_key, &embedding_bytes, self.config.default_ttl)
                .set_ex(&text_key, text, self.config.default_ttl)
                .sadd(self.make_key(&format!("chunks:{}", doc_id)), chunk_id);
        }

        pipe.query_async(&mut *conn)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, true).await;
        debug!("Batch stored {} embeddings for {}", embeddings.len(), doc_id);
        Ok(())
    }

    /// Get all chunk IDs for a document
    pub async fn get_document_chunks(&self, doc_id: &str) -> GemmaResult<Vec<usize>> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let chunks_key = self.make_key(&format!("chunks:{}", doc_id));
        let chunk_ids: Vec<usize> = conn
            .smembers(&chunks_key)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, !chunk_ids.is_empty()).await;
        Ok(chunk_ids)
    }

    /// Store search results cache
    pub async fn cache_search_results(
        &self,
        query_hash: &str,
        results: &[(String, f32)],
        ttl_seconds: u64,
    ) -> GemmaResult<()> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let cache_key = self.make_key(&format!("search:{}", query_hash));
        let serialized = serde_json::to_string(results)
            .map_err(|e| GemmaError::Serialization(e.to_string()))?;

        conn.set_ex(&cache_key, serialized, ttl_seconds)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, true).await;
        debug!("Cached search results for query: {}", query_hash);
        Ok(())
    }

    /// Retrieve cached search results
    pub async fn get_cached_search_results(
        &self,
        query_hash: &str,
    ) -> GemmaResult<Option<Vec<(String, f32)>>> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let cache_key = self.make_key(&format!("search:{}", query_hash));
        let cached: Option<String> = conn
            .get(&cache_key)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, cached.is_some()).await;

        match cached {
            Some(serialized) => {
                let results = serde_json::from_str(&serialized)
                    .map_err(|e| GemmaError::Serialization(e.to_string()))?;
                Ok(Some(results))
            }
            None => Ok(None),
        }
    }

    /// Delete a document and all its associated data
    pub async fn delete_document(&self, doc_id: &str) -> GemmaResult<bool> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        // Get all chunk IDs first
        let chunk_ids = self.get_document_chunks(doc_id).await?;

        let mut pipe = redis::pipe();

        // Delete document and metadata
        pipe.del(self.make_key(&format!("doc:{}", doc_id)))
            .del(self.make_key(&format!("meta:{}", doc_id)))
            .srem(self.make_key("docs"), doc_id)
            .del(self.make_key(&format!("chunks:{}", doc_id)));

        // Delete all embeddings and chunks
        for chunk_id in chunk_ids {
            pipe.del(self.make_key(&format!("emb:{}:{}", doc_id, chunk_id)))
                .del(self.make_key(&format!("chunk:{}:{}", doc_id, chunk_id)));
        }

        let deleted_count: usize = pipe
            .query_async(&mut *conn)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, deleted_count > 0).await;
        debug!("Deleted document {} (deleted {} keys)", doc_id, deleted_count);
        Ok(deleted_count > 0)
    }

    /// List all document IDs
    pub async fn list_documents(&self) -> GemmaResult<Vec<String>> {
        let start_time = std::time::Instant::now();
        let mut conn = self.get_connection().await?;

        let docs_key = self.make_key("docs");
        let doc_ids: Vec<String> = conn
            .smembers(&docs_key)
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        self.update_stats(start_time, !doc_ids.is_empty()).await;
        Ok(doc_ids)
    }

    /// Health check for Redis connection
    pub async fn health_check(&self) -> GemmaResult<bool> {
        let mut conn = self.get_connection().await?;
        let response: String = conn
            .ping()
            .await
            .map_err(|e| GemmaError::Redis(e.to_string()))?;

        Ok(response == "PONG")
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> RedisStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = RedisStats::default();
    }

    /// Update statistics after an operation
    async fn update_stats(&self, start_time: std::time::Instant, success: bool) {
        let mut stats = self.stats.write().await;
        let duration = start_time.elapsed().as_millis() as f64;

        stats.total_operations += 1;
        if success {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
            stats.failed_operations += 1;
        }

        // Update moving average of response time
        let weight = 1.0 / stats.total_operations as f64;
        stats.average_response_time_ms =
            stats.average_response_time_ms * (1.0 - weight) + duration * weight;

        // Update active connections (approximation)
        stats.active_connections = self.pool.state().connections;
    }
}

/// Utility function to generate a hash for search queries
pub fn hash_query(query: &str, filters: &[String]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    filters.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    fn test_config() -> RedisConfig {
        RedisConfig {
            url: "redis://localhost:6379".to_string(),
            max_connections: 5,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_redis_config() {
        let config = RedisConfig::default();
        assert_eq!(config.max_connections, 20);
        assert_eq!(config.default_ttl, 3600);
    }

    #[tokio::test]
    async fn test_document_metadata() {
        let meta = DocumentMetadata::new("test-doc".to_string(), "text/plain".to_string());
        assert_eq!(meta.id, "test-doc");
        assert_eq!(meta.content_type, "text/plain");
        assert_eq!(meta.chunk_count, 0);
    }

    #[tokio::test]
    async fn test_redis_stats() {
        let mut stats = RedisStats::default();
        stats.cache_hits = 8;
        stats.cache_misses = 2;
        stats.total_operations = 10;
        stats.failed_operations = 1;

        assert_eq!(stats.hit_rate(), 0.8);
        assert_eq!(stats.success_rate(), 0.9);
    }

    #[tokio::test]
    async fn test_hash_query() {
        let hash1 = hash_query("test query", &["filter1".to_string()]);
        let hash2 = hash_query("test query", &["filter1".to_string()]);
        let hash3 = hash_query("different query", &["filter1".to_string()]);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    // Integration tests would require a Redis instance
    #[tokio::test]
    #[ignore]
    async fn test_redis_manager_integration() {
        let config = test_config();
        let manager = RedisManager::new(config).await;
        assert!(manager.is_ok());

        if let Ok(manager) = manager {
            assert!(manager.health_check().await.is_ok());
        }
    }
}
