use crate::{
    config::RedisConfig,
    error::{Error, Result},
};
use bb8::{Pool, PooledConnection};
use bb8_redis::RedisConnectionManager;
use parking_lot::RwLock as SyncRwLock;
use redis::{aio::ConnectionManager, AsyncCommands, RedisResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, warn};

/// Simple in-memory fallback store used when Redis is unavailable.
#[derive(Default, Clone)]
struct FallbackStore {
    inner: Arc<SyncRwLock<HashMap<String, Vec<u8>>>>,
}

impl FallbackStore {
    fn set(&self, key: &str, value: Vec<u8>) {
        self.inner.write().insert(key.to_string(), value);
    }
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.inner.read().get(key).cloned()
    }
    fn del(&self, key: &str) {
        self.inner.write().remove(key);
    }
    fn scan_prefix(&self, prefix: &str) -> Vec<String> {
        self.inner
            .read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect()
    }
}

// RedisConfig is now defined in config.rs

pub struct RedisClient {
    connection: ConnectionManager,
}

impl RedisClient {
    pub async fn new(config: &RedisConfig) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| Error::Redis(format!("Failed to create Redis client: {}", e)))?;

        let connection = ConnectionManager::new(client)
            .await
            .map_err(|e| Error::Redis(format!("Failed to connect to Redis: {}", e)))?;

        Ok(Self { connection })
    }

    pub async fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>> {
        self.connection
            .get(key)
            .await
            .map_err(|e| Error::Redis(format!("Failed to get key {}: {}", key, e)))
    }

    pub async fn set(&mut self, key: &str, value: &[u8]) -> Result<()> {
        self.connection
            .set(key, value)
            .await
            .map_err(|e| Error::Redis(format!("Failed to set key {}: {}", key, e)))
    }

    pub async fn set_ex(&mut self, key: &str, value: &[u8], ttl: Duration) -> Result<()> {
        self.connection
            .set_ex(key, value, ttl.as_secs())
            .await
            .map_err(|e| Error::Redis(format!("Failed to set key {} with TTL: {}", key, e)))
    }

    pub async fn delete(&mut self, key: &str) -> Result<()> {
        let _: RedisResult<()> = self.connection.del(key).await;
        Ok(())
    }

    pub async fn exists(&mut self, key: &str) -> Result<bool> {
        self.connection
            .exists(key)
            .await
            .map_err(|e| Error::Redis(format!("Failed to check existence of key {}: {}", key, e)))
    }

    pub async fn expire(&mut self, key: &str, ttl: Duration) -> Result<()> {
        self.connection
            .expire(key, ttl.as_secs() as i64)
            .await
            .map_err(|e| Error::Redis(format!("Failed to set expiration for key {}: {}", key, e)))
    }

    pub async fn lpush(&mut self, key: &str, value: &[u8]) -> Result<()> {
        self.connection
            .lpush(key, value)
            .await
            .map_err(|e| Error::Redis(format!("Failed to lpush to key {}: {}", key, e)))
    }

    pub async fn lrange(&mut self, key: &str, start: isize, stop: isize) -> Result<Vec<Vec<u8>>> {
        self.connection
            .lrange(key, start, stop)
            .await
            .map_err(|e| Error::Redis(format!("Failed to lrange key {}: {}", key, e)))
    }

    pub async fn hset(&mut self, key: &str, field: &str, value: &[u8]) -> Result<()> {
        self.connection
            .hset(key, field, value)
            .await
            .map_err(|e| Error::Redis(format!("Failed to hset {}:{}: {}", key, field, e)))
    }

    pub async fn hget(&mut self, key: &str, field: &str) -> Result<Option<Vec<u8>>> {
        self.connection
            .hget(key, field)
            .await
            .map_err(|e| Error::Redis(format!("Failed to hget {}:{}: {}", key, field, e)))
    }

    pub async fn hgetall(&mut self, key: &str) -> Result<HashMap<String, Vec<u8>>> {
        self.connection
            .hgetall(key)
            .await
            .map_err(|e| Error::Redis(format!("Failed to hgetall {}: {}", key, e)))
    }

    pub async fn zadd(&mut self, key: &str, member: &str, score: f64) -> Result<()> {
        self.connection
            .zadd(key, member, score)
            .await
            .map_err(|e| Error::Redis(format!("Failed to zadd to {}: {}", key, e)))
    }

    pub async fn zrange_withscores(
        &mut self,
        key: &str,
        start: isize,
        stop: isize,
    ) -> Result<Vec<(String, f64)>> {
        self.connection
            .zrange_withscores(key, start, stop)
            .await
            .map_err(|e| Error::Redis(format!("Failed to zrange {}: {}", key, e)))
    }

    pub async fn ping(&mut self) -> Result<()> {
        let _: String = redis::cmd("PING")
            .query_async::<String>(&mut self.connection)
            .await
            .map_err(|e| Error::Redis(format!("Ping failed: {}", e)))?;
        Ok(())
    }
}

pub struct RedisManager {
    pool: Pool<RedisConnectionManager>,
    config: RedisConfig,
    stats: Arc<RwLock<RedisStats>>,
    fallback: Option<FallbackStore>,
}

#[derive(Debug, Default)]
struct RedisStats {
    total_operations: u64,
    failed_operations: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl RedisManager {
    pub async fn new(config: &RedisConfig) -> Result<Self> {
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..=config.max_retries {
            match RedisConnectionManager::new(config.url.as_str()) {
                Ok(manager) => {
                    match Pool::builder()
                        .max_size(config.pool_size)
                        .connection_timeout(config.connection_timeout)
                        .build(manager)
                        .await
                    {
                        Ok(pool) => {
                            return Ok(Self {
                                pool,
                                config: config.clone(),
                                stats: Arc::new(RwLock::new(RedisStats::default())),
                                fallback: None,
                            });
                        }
                        Err(e) => {
                            last_err = Some(anyhow::anyhow!(e));
                            warn!(
                                "Redis connection pool attempt {} failed: {}",
                                attempt + 1,
                                last_err.as_ref().unwrap()
                            );
                        }
                    }
                }
                Err(e) => {
                    last_err = Some(anyhow::anyhow!(e));
                    warn!(
                        "Redis manager creation attempt {} failed: {}",
                        attempt + 1,
                        last_err.as_ref().unwrap()
                    );
                }
            }
            tokio::time::sleep(config.retry_delay).await;
        }

        if config.enable_fallback {
            warn!(
                "Falling back to in-memory store (Redis unreachable: {:?})",
                last_err
            );
            // Create a dummy pool by attempting again; if it still fails we return a minimal pool using an unreachable URL
            let manager = RedisConnectionManager::new("redis://127.0.0.1:6379")
                .map_err(|e| Error::Redis(format!("Failed to create fallback manager: {}", e)))?;
            let pool = Pool::builder()
                .max_size(1)
                .connection_timeout(Duration::from_secs(1))
                .build(manager)
                .await
                .map_err(|e| Error::Redis(format!("Failed to create fallback pool: {}", e)))?;

            return Ok(Self {
                pool,
                config: config.clone(),
                stats: Arc::new(RwLock::new(RedisStats::default())),
                fallback: Some(FallbackStore::default()),
            });
        }

        Err(Error::Redis(format!(
            "Failed to connect to Redis after retries: {:?}",
            last_err
        )))
    }

    async fn get_connection(&self) -> Result<PooledConnection<'_, RedisConnectionManager>> {
        if self.fallback.is_some() {
            // In fallback mode we still return a pooled connection (may be broken) but most operations short-circuit.
            return self
                .pool
                .get()
                .await
                .map_err(|e| Error::Redis(format!("Fallback pool error: {}", e)));
        }
        self.pool
            .get()
            .await
            .map_err(|e| Error::Redis(format!("Failed to get connection from pool: {}", e)))
    }

    pub async fn store_document(&self, document: &crate::Document) -> Result<()> {
        let key = format!("doc:{}", document.id);
        let value = serde_json::to_vec(document)?;
        if let Some(fallback) = &self.fallback {
            fallback.set(&key, value);
        } else {
            let mut conn = self.get_connection().await?;
            let _: () = conn
                .set_ex(&key, &value, self.config.command_timeout.as_secs())
                .await
                .map_err(|e| Error::Redis(format!("Failed to store document: {}", e)))?;
        }

        self.update_stats(true).await;
        Ok(())
    }

    pub async fn get_document(&self, id: &str) -> Result<Option<crate::document::Document>> {
        let key = format!("doc:{}", id);
        let value: Option<Vec<u8>> = if let Some(fallback) = &self.fallback {
            fallback.get(&key)
        } else {
            let mut conn = self.get_connection().await?;
            conn.get(&key)
                .await
                .map_err(|e| Error::Redis(format!("Failed to get document: {}", e)))?
        };

        match value {
            Some(data) => {
                self.update_stats(true).await;
                let document = serde_json::from_slice(&data)?;
                Ok(Some(document))
            }
            None => {
                self.update_stats(false).await;
                Ok(None)
            }
        }
    }

    pub async fn store_chunk(&self, chunk: &crate::document::DocumentChunk) -> Result<()> {
        let key = format!("chunk:{}", chunk.id);
        let value = serde_json::to_vec(chunk)?;
        if let Some(fallback) = &self.fallback {
            fallback.set(&key, value);
            let doc_chunks_key = format!("doc:{}:chunks", chunk.document_id);
            // Maintain a simple list by concatenating ids with a separator
            let list_key = format!("{}_list", doc_chunks_key);
            let mut existing = fallback.get(&list_key).unwrap_or_default();
            existing.extend_from_slice(chunk.id.as_bytes());
            existing.push(b'\n');
            fallback.set(&list_key, existing);
        } else {
            let mut conn = self.get_connection().await?;
            let _: () = conn
                .set(&key, &value)
                .await
                .map_err(|e| Error::Redis(format!("Failed to store chunk: {}", e)))?;
            let doc_chunks_key = format!("doc:{}:chunks", chunk.document_id);
            let _: () = conn
                .lpush(&doc_chunks_key, chunk.id.as_bytes())
                .await
                .map_err(|e| {
                    Error::Redis(format!("Failed to add chunk to document list: {}", e))
                })?;
        }

        self.update_stats(true).await;
        Ok(())
    }

    pub async fn get_chunk(&self, id: &str) -> Result<crate::document::DocumentChunk> {
        let key = format!("chunk:{}", id);
        let value: Option<Vec<u8>> = if let Some(fallback) = &self.fallback {
            fallback.get(&key)
        } else {
            let mut conn = self.get_connection().await?;
            conn.get(&key)
                .await
                .map_err(|e| Error::Redis(format!("Failed to get chunk: {}", e)))?
        };

        match value {
            Some(data) => {
                self.update_stats(true).await;
                let chunk = serde_json::from_slice(&data)?;
                Ok(chunk)
            }
            None => {
                self.update_stats(false).await;
                Err(Error::NotFound(format!("Chunk {} not found", id)))
            }
        }
    }

    pub async fn store_embedding(&self, id: &str, embedding: &[f32]) -> Result<()> {
        let key = format!("embedding:{}", id);
        let value = bincode::serialize(embedding)?;
        if let Some(fallback) = &self.fallback {
            fallback.set(&key, value);
        } else {
            let mut conn = self.get_connection().await?;
            let _: () = conn
                .set(&key, &value)
                .await
                .map_err(|e| Error::Redis(format!("Failed to store embedding: {}", e)))?;
        }

        self.update_stats(true).await;
        Ok(())
    }

    pub async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let key = format!("embedding:{}", id);
        let value: Option<Vec<u8>> = if let Some(fallback) = &self.fallback {
            fallback.get(&key)
        } else {
            let mut conn = self.get_connection().await?;
            conn.get(&key)
                .await
                .map_err(|e| Error::Redis(format!("Failed to get embedding: {}", e)))?
        };

        match value {
            Some(data) => {
                self.update_stats(true).await;
                let embedding = bincode::deserialize(&data)?;
                Ok(Some(embedding))
            }
            None => {
                self.update_stats(false).await;
                Ok(None)
            }
        }
    }

    pub async fn cache_search_result(
        &self,
        query_hash: &str,
        results: &[crate::research::SearchResult],
        ttl: Duration,
    ) -> Result<()> {
        let key = format!("search:{}", query_hash);
        let value = serde_json::to_vec(results)?;
        if let Some(fallback) = &self.fallback {
            fallback.set(&key, value);
        } else {
            let mut conn = self.get_connection().await?;
            let _: () = conn
                .set_ex(&key, &value, ttl.as_secs())
                .await
                .map_err(|e| Error::Redis(format!("Failed to cache search results: {}", e)))?;
        }

        self.update_stats(true).await;
        Ok(())
    }

    pub async fn get_cached_search(
        &self,
        query_hash: &str,
    ) -> Result<Option<Vec<crate::research::SearchResult>>> {
        let key = format!("search:{}", query_hash);
        let value: Option<Vec<u8>> = if let Some(fallback) = &self.fallback {
            fallback.get(&key)
        } else {
            let mut conn = self.get_connection().await?;
            conn.get(&key)
                .await
                .map_err(|e| Error::Redis(format!("Failed to get cached search: {}", e)))?
        };

        match value {
            Some(data) => {
                self.update_stats(true).await;
                let results = serde_json::from_slice(&data)?;
                Ok(Some(results))
            }
            None => {
                self.update_stats(false).await;
                Ok(None)
            }
        }
    }

    pub async fn health_check(&self) -> Result<bool> {
        if self.fallback.is_some() {
            return Ok(true); // In-memory always "healthy"
        }
        match self.get_connection().await {
            Ok(mut conn) => match redis::cmd("PING").query_async::<String>(&mut *conn).await {
                Ok(_) => Ok(true),
                Err(e) => {
                    warn!("Redis health check failed: {}", e);
                    Ok(false)
                }
            },
            Err(e) => {
                error!("Failed to get Redis connection for health check: {}", e);
                Ok(false)
            }
        }
    }

    async fn update_stats(&self, cache_hit: bool) {
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }

    pub async fn get_stats(&self) -> HashMap<String, u64> {
        let stats = self.stats.read().await;
        let mut result = HashMap::new();
        result.insert("total_operations".to_string(), stats.total_operations);
        result.insert("failed_operations".to_string(), stats.failed_operations);
        result.insert("cache_hits".to_string(), stats.cache_hits);
        result.insert("cache_misses".to_string(), stats.cache_misses);

        if stats.total_operations > 0 {
            let hit_rate = (stats.cache_hits as f64 / stats.total_operations as f64 * 100.0) as u64;
            result.insert("cache_hit_rate".to_string(), hit_rate);
        }

        result
    }

    /// Store raw data with key
    pub async fn set_raw(&self, key: &str, value: &[u8]) -> Result<()> {
        if let Some(fallback) = &self.fallback {
            fallback.set(key, value.to_vec());
            return Ok(());
        }
        let mut conn = self.get_connection().await?;
        let _: () = conn
            .set(key, value)
            .await
            .map_err(|e| Error::Redis(format!("Failed to set key {}: {}", key, e)))?;
        self.update_stats(true).await;
        Ok(())
    }

    /// Store raw data with key and TTL
    pub async fn set_raw_ex(&self, key: &str, value: &[u8], ttl_seconds: u64) -> Result<()> {
        if let Some(fallback) = &self.fallback {
            fallback.set(key, value.to_vec());
            return Ok(());
        }
        let mut conn = self.get_connection().await?;
        let _: () = conn
            .set_ex(key, value, ttl_seconds)
            .await
            .map_err(|e| Error::Redis(format!("Failed to set key {} with TTL: {}", key, e)))?;
        self.update_stats(true).await;
        Ok(())
    }

    /// Get raw data by key
    pub async fn get_raw(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if let Some(fallback) = &self.fallback {
            return Ok(fallback.get(key));
        }
        let mut conn = self.get_connection().await?;
        let result = conn
            .get(key)
            .await
            .map_err(|e| Error::Redis(format!("Failed to get key {}: {}", key, e)))?;

        match result {
            Some(_) => self.update_stats(true).await,
            None => self.update_stats(false).await,
        }

        Ok(result)
    }

    /// Delete key
    pub async fn delete_raw(&self, key: &str) -> Result<()> {
        if let Some(fallback) = &self.fallback {
            fallback.del(key);
            return Ok(());
        }
        let mut conn = self.get_connection().await?;
        let _: redis::RedisResult<()> = conn.del(key).await;
        Ok(())
    }

    /// Get a client that can be used for simple operations
    pub fn redis_client(&self) -> RedisHealthClient {
        RedisHealthClient {
            pool: self.pool.clone(),
        }
    }

    /// Get a pooled connection from the Redis connection pool
    pub async fn get_pooled_connection(
        &self,
    ) -> Result<PooledConnection<'_, RedisConnectionManager>> {
        self.get_connection().await
    }

    /// Scan keys matching a prefix (used by memory manager) with fallback support.
    pub async fn scan_pattern(&self, prefix: &str) -> Result<Vec<String>> {
        if let Some(fallback) = &self.fallback {
            return Ok(fallback.scan_prefix(prefix));
        }
        let mut cursor = 0u64;
        let mut keys_acc = Vec::new();
        let mut conn = self.get_connection().await?;
        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(format!("{}*", prefix))
                .arg("COUNT")
                .arg(100)
                .query_async(&mut *conn)
                .await?;
            keys_acc.extend(keys);
            cursor = new_cursor;
            if cursor == 0 {
                break;
            }
        }
        Ok(keys_acc)
    }
}

/// Simple wrapper for health checks
pub struct RedisHealthClient {
    pool: Pool<RedisConnectionManager>,
}

impl RedisHealthClient {
    pub async fn ping(&self) -> Result<()> {
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| Error::Redis(format!("Failed to get connection from pool: {}", e)))?;

        let _: String = redis::cmd("PING")
            .query_async::<String>(&mut *conn)
            .await
            .map_err(|e| Error::Redis(format!("Ping failed: {}", e)))?;
        Ok(())
    }
}
