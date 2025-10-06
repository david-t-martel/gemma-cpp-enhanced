//! Enhanced Memory Retrieval System with Smart Caching and Query Optimization
//!
//! This module provides intelligent memory retrieval with multi-level caching,
//! query optimization, and parallel search across memory tiers.

use crate::{
    error::Result,
    memory::{MemoryEntry, MemoryType, MemoryFilter, MemoryManager, MemorySearchResult},
    memory_archive::{MemoryArchiveManager, ArchiveLevel},
    vector_store::VectorStore,
};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock as AsyncRwLock, Semaphore};
use tracing::{debug, info, warn};

/// Retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Maximum cache size (entries)
    pub cache_size: usize,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Enable parallel search
    pub parallel_search: bool,

    /// Maximum parallel queries
    pub max_parallel_queries: usize,

    /// Query result limit
    pub default_limit: usize,

    /// Prefetch related memories
    pub prefetch_related: bool,

    /// Prefetch depth
    pub prefetch_depth: usize,

    /// Enable query optimization
    pub optimize_queries: bool,

    /// Minimum similarity threshold
    pub similarity_threshold: f32,

    /// Enable adaptive caching
    pub adaptive_caching: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            cache_size: 10000,
            cache_ttl_seconds: 300,
            parallel_search: true,
            max_parallel_queries: 10,
            default_limit: 20,
            prefetch_related: true,
            prefetch_depth: 2,
            optimize_queries: true,
            similarity_threshold: 0.5,
            adaptive_caching: true,
        }
    }
}

/// Query type for optimized retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    Exact { id: String },
    Semantic { query: String, embedding: Option<Vec<f32>> },
    Pattern { pattern: String },
    Temporal { start: chrono::DateTime<chrono::Utc>, end: chrono::DateTime<chrono::Utc> },
    Combined { queries: Vec<QueryType> },
    ProjectContext { project_id: String },
}

/// Query result with metadata
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub entry: MemoryEntry,
    pub score: f32,
    pub source: RetrievalSource,
    pub retrieval_time_ms: u64,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetrievalSource {
    Cache,
    Working,
    ShortTerm,
    LongTerm,
    Episodic,
    Semantic,
    Archive(ArchiveLevel),
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    entry: MemoryEntry,
    inserted_at: Instant,
    access_count: u64,
    last_accessed: Instant,
}

impl CacheEntry {
    fn new(entry: MemoryEntry) -> Self {
        let now = Instant::now();
        Self {
            entry,
            inserted_at: now,
            access_count: 1,
            last_accessed: now,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.inserted_at.elapsed() > ttl
    }

    fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }
}

/// Multi-level cache for fast retrieval
struct MultiLevelCache {
    l1_cache: Arc<DashMap<String, CacheEntry>>,  // Hot cache
    l2_cache: Arc<DashMap<String, CacheEntry>>,  // Warm cache
    l3_cache: Arc<DashMap<String, CacheEntry>>,  // Cold cache
    access_history: Arc<RwLock<VecDeque<String>>>,
    config: RetrievalConfig,
}

impl MultiLevelCache {
    fn new(config: RetrievalConfig) -> Self {
        Self {
            l1_cache: Arc::new(DashMap::new()),
            l2_cache: Arc::new(DashMap::new()),
            l3_cache: Arc::new(DashMap::new()),
            access_history: Arc::new(RwLock::new(VecDeque::new())),
            config,
        }
    }

    fn get(&self, key: &str) -> Option<MemoryEntry> {
        // Check L1 (hot)
        if let Some(mut entry) = self.l1_cache.get_mut(key) {
            entry.access();
            return Some(entry.entry.clone());
        }

        // Check L2 (warm) and promote to L1
        if let Some(entry) = self.l2_cache.remove(key) {
            let mut cache_entry = entry.1;
            cache_entry.access();
            let result = cache_entry.entry.clone();
            self.l1_cache.insert(key.to_string(), cache_entry);
            self.manage_cache_size();
            return Some(result);
        }

        // Check L3 (cold) and promote to L2
        if let Some(entry) = self.l3_cache.remove(key) {
            let mut cache_entry = entry.1;
            cache_entry.access();
            let result = cache_entry.entry.clone();
            self.l2_cache.insert(key.to_string(), cache_entry);
            return Some(result);
        }

        None
    }

    fn insert(&self, key: String, entry: MemoryEntry) {
        let cache_entry = CacheEntry::new(entry);

        // Add to L1
        self.l1_cache.insert(key.clone(), cache_entry);

        // Track access
        let mut history = self.access_history.write();
        history.push_back(key);

        self.manage_cache_size();
    }

    fn manage_cache_size(&self) {
        let l1_size = self.config.cache_size / 4;
        let l2_size = self.config.cache_size / 4;
        let l3_size = self.config.cache_size / 2;

        // Manage L1 -> L2 promotion
        if self.l1_cache.len() > l1_size {
            let evict_count = self.l1_cache.len() - l1_size;
            let mut entries: Vec<_> = self.l1_cache
                .iter()
                .map(|e| (e.key().clone(), e.value().last_accessed))
                .collect();

            entries.sort_by_key(|e| e.1);

            for (key, _) in entries.iter().take(evict_count) {
                if let Some(entry) = self.l1_cache.remove(key) {
                    self.l2_cache.insert(key.clone(), entry.1);
                }
            }
        }

        // Manage L2 -> L3 promotion
        if self.l2_cache.len() > l2_size {
            let evict_count = self.l2_cache.len() - l2_size;
            let mut entries: Vec<_> = self.l2_cache
                .iter()
                .map(|e| (e.key().clone(), e.value().last_accessed))
                .collect();

            entries.sort_by_key(|e| e.1);

            for (key, _) in entries.iter().take(evict_count) {
                if let Some(entry) = self.l2_cache.remove(key) {
                    self.l3_cache.insert(key.clone(), entry.1);
                }
            }
        }

        // Evict from L3 if needed
        if self.l3_cache.len() > l3_size {
            let evict_count = self.l3_cache.len() - l3_size;
            let mut entries: Vec<_> = self.l3_cache
                .iter()
                .map(|e| (e.key().clone(), e.value().last_accessed))
                .collect();

            entries.sort_by_key(|e| e.1);

            for (key, _) in entries.iter().take(evict_count) {
                self.l3_cache.remove(key);
            }
        }
    }

    fn clear_expired(&self) {
        let ttl = Duration::from_secs(self.config.cache_ttl_seconds);

        // Clear expired from all levels
        self.l1_cache.retain(|_, v| !v.is_expired(ttl));
        self.l2_cache.retain(|_, v| !v.is_expired(ttl));
        self.l3_cache.retain(|_, v| !v.is_expired(ttl));
    }
}

/// Query optimizer for intelligent retrieval
struct QueryOptimizer {
    query_history: Arc<DashMap<String, Vec<String>>>,
    pattern_cache: Arc<DashMap<String, Vec<String>>>,
}

impl QueryOptimizer {
    fn new() -> Self {
        Self {
            query_history: Arc::new(DashMap::new()),
            pattern_cache: Arc::new(DashMap::new()),
        }
    }

    fn optimize_query(&self, query: &QueryType) -> QueryType {
        match query {
            QueryType::Semantic { query: q, embedding: _ } => {
                // Check if we've seen similar queries
                if let Some(cached) = self.pattern_cache.get(q) {
                    // Return cached pattern results
                    QueryType::Combined {
                        queries: cached
                            .iter()
                            .map(|id| QueryType::Exact { id: id.clone() })
                            .collect(),
                    }
                } else {
                    query.clone()
                }
            }
            QueryType::Combined { queries } => {
                // Deduplicate and optimize sub-queries
                let mut optimized = Vec::new();
                let mut seen = HashSet::new();

                for q in queries {
                    if let QueryType::Exact { id } = q {
                        if seen.insert(id.clone()) {
                            optimized.push(q.clone());
                        }
                    } else {
                        optimized.push(self.optimize_query(q));
                    }
                }

                QueryType::Combined { queries: optimized }
            }
            _ => query.clone(),
        }
    }

    fn record_results(&self, query: String, results: Vec<String>) {
        self.query_history
            .entry(query.clone())
            .or_insert_with(Vec::new)
            .extend(results.clone());

        // Cache pattern for future use
        if results.len() > 5 {
            self.pattern_cache.insert(query, results.into_iter().take(10).collect());
        }
    }
}

/// Enhanced Memory Retrieval Manager
pub struct MemoryRetrievalManager {
    config: RetrievalConfig,
    memory_manager: Arc<MemoryManager>,
    archive_manager: Arc<MemoryArchiveManager>,
    vector_store: Arc<AsyncRwLock<VectorStore>>,
    cache: Arc<MultiLevelCache>,
    query_optimizer: Arc<QueryOptimizer>,
    search_semaphore: Arc<Semaphore>,

    // Metrics
    metrics: Arc<RwLock<RetrievalMetrics>>,
}

#[derive(Debug, Default, Clone)]
struct RetrievalMetrics {
    total_queries: u64,
    cache_hits: u64,
    cache_misses: u64,
    avg_retrieval_time_ms: f64,
    queries_by_type: HashMap<String, u64>,
}

impl MemoryRetrievalManager {
    pub fn new(
        config: RetrievalConfig,
        memory_manager: Arc<MemoryManager>,
        archive_manager: Arc<MemoryArchiveManager>,
        vector_store: Arc<AsyncRwLock<VectorStore>>,
    ) -> Self {
        let max_parallel = config.max_parallel_queries;

        Self {
            cache: Arc::new(MultiLevelCache::new(config.clone())),
            config,
            memory_manager,
            archive_manager,
            vector_store,
            query_optimizer: Arc::new(QueryOptimizer::new()),
            search_semaphore: Arc::new(Semaphore::new(max_parallel)),
            metrics: Arc::new(RwLock::new(RetrievalMetrics::default())),
        }
    }

    /// Retrieve memory by query
    pub async fn retrieve(&self, query: QueryType) -> Result<Vec<RetrievalResult>> {
        let start = Instant::now();

        // Optimize query if enabled
        let optimized_query = if self.config.optimize_queries {
            self.query_optimizer.optimize_query(&query)
        } else {
            query.clone()
        };

        let results = match optimized_query {
            QueryType::Exact { id } => {
                self.retrieve_exact(&id).await?
            }
            QueryType::Semantic { query: q, embedding } => {
                self.retrieve_semantic(&q, embedding).await?
            }
            QueryType::Pattern { pattern } => {
                self.retrieve_pattern(&pattern).await?
            }
            QueryType::Temporal { start, end } => {
                self.retrieve_temporal(start, end).await?
            }
            QueryType::Combined { queries } => {
                Box::pin(self.retrieve_combined(queries)).await?
            }
            QueryType::ProjectContext { project_id } => {
                self.retrieve_project_context(&project_id).await?
            }
        };

        // Update metrics
        let elapsed = start.elapsed().as_millis() as u64;
        self.update_metrics(&query, results.len(), elapsed);

        // Prefetch related if enabled
        if self.config.prefetch_related && !results.is_empty() {
            self.prefetch_related(&results).await;
        }

        Ok(results)
    }

    async fn retrieve_exact(&self, id: &str) -> Result<Vec<RetrievalResult>> {
        let start = Instant::now();

        // Check cache first
        if let Some(entry) = self.cache.get(id) {
            return Ok(vec![RetrievalResult {
                entry,
                score: 1.0,
                source: RetrievalSource::Cache,
                retrieval_time_ms: start.elapsed().as_millis() as u64,
                cache_hit: true,
            }]);
        }

        // Try memory manager
        if let Ok(Some(entry)) = self.memory_manager.get(id).await {
            self.cache.insert(id.to_string(), entry.clone());

            let source = match entry.memory_type {
                MemoryType::Working => RetrievalSource::Working,
                MemoryType::ShortTerm => RetrievalSource::ShortTerm,
                MemoryType::LongTerm => RetrievalSource::LongTerm,
                MemoryType::Episodic => RetrievalSource::Episodic,
                MemoryType::Semantic => RetrievalSource::Semantic,
            };

            return Ok(vec![RetrievalResult {
                entry,
                score: 1.0,
                source,
                retrieval_time_ms: start.elapsed().as_millis() as u64,
                cache_hit: false,
            }]);
        }

        // Try archive
        if let Some(entry) = self.archive_manager.retrieve_memory(id).await? {
            self.cache.insert(id.to_string(), entry.clone());

            return Ok(vec![RetrievalResult {
                entry,
                score: 1.0,
                source: RetrievalSource::Archive(ArchiveLevel::Cold),
                retrieval_time_ms: start.elapsed().as_millis() as u64,
                cache_hit: false,
            }]);
        }

        Ok(Vec::new())
    }

    async fn retrieve_semantic(&self, query: &str, embedding: Option<Vec<f32>>) -> Result<Vec<RetrievalResult>> {
        let start = Instant::now();
        let _permit = self.search_semaphore.acquire().await?;

        // Generate embedding if not provided
        let query_embedding = if let Some(emb) = embedding {
            emb
        } else {
            // This would call your embedding service
            vec![0.0; 768] // Placeholder
        };

        // Search vector store
        let vector_results = self.vector_store
            .read()
            .await
            .search(&query_embedding, self.config.default_limit, None)?;

        let mut results = Vec::new();

        // Retrieve full entries
        for (id, score, _metadata) in vector_results {
            if score >= self.config.similarity_threshold {
                if let Ok(entries) = self.retrieve_exact(&id).await {
                    for mut entry in entries {
                        entry.score = score;
                        results.push(entry);
                    }
                }
            }
        }

        // Record for optimization
        if self.config.optimize_queries {
            let result_ids: Vec<String> = results.iter().map(|r| r.entry.id.clone()).collect();
            self.query_optimizer.record_results(query.to_string(), result_ids);
        }

        Ok(results)
    }

    async fn retrieve_pattern(&self, pattern: &str) -> Result<Vec<RetrievalResult>> {
        let start = Instant::now();

        let filter = MemoryFilter {
            content_pattern: Some(pattern.to_string()),
            limit: Some(self.config.default_limit),
            sort_by_importance: true,
            ..Default::default()
        };

        let search_results = self.memory_manager.search(&filter).await?;

        let results: Vec<RetrievalResult> = search_results
            .into_iter()
            .map(|sr| RetrievalResult {
                entry: sr.entry,
                score: sr.score as f32,
                source: RetrievalSource::ShortTerm,
                retrieval_time_ms: start.elapsed().as_millis() as u64,
                cache_hit: false,
            })
            .collect();

        Ok(results)
    }

    async fn retrieve_temporal(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<RetrievalResult>> {
        let start_time = Instant::now();

        let filter = MemoryFilter {
            max_age: Some(std::time::Duration::from_secs(
                (end - start).num_seconds() as u64
            )),
            limit: Some(self.config.default_limit),
            sort_by_recency: true,
            ..Default::default()
        };

        let search_results = self.memory_manager.search(&filter).await?;

        let results: Vec<RetrievalResult> = search_results
            .into_iter()
            .filter(|sr| sr.entry.created_at >= start && sr.entry.created_at <= end)
            .map(|sr| RetrievalResult {
                entry: sr.entry,
                score: sr.score as f32,
                source: RetrievalSource::Episodic,
                retrieval_time_ms: start_time.elapsed().as_millis() as u64,
                cache_hit: false,
            })
            .collect();

        Ok(results)
    }

    async fn retrieve_combined(&self, queries: Vec<QueryType>) -> Result<Vec<RetrievalResult>> {
        // For now, always use sequential search to avoid Send issues
        // TODO: Fix Send trait implementation for parallel search
        let mut all_results = Vec::new();
        for query in queries {
            let results = Box::pin(self.retrieve(query)).await?;
            all_results.extend(results);
        }
        Ok(self.deduplicate_results(all_results))
    }

    async fn retrieve_project_context(&self, project_id: &str) -> Result<Vec<RetrievalResult>> {
        let start = Instant::now();

        if let Some(context) = self.archive_manager.get_project_context(project_id).await? {
            let mut results = Vec::new();

            for memory_id in &context.important_memories {
                if let Ok(entries) = self.retrieve_exact(memory_id).await {
                    results.extend(entries);
                }
            }

            return Ok(results);
        }

        Ok(Vec::new())
    }

    fn deduplicate_results(&self, results: Vec<RetrievalResult>) -> Vec<RetrievalResult> {
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();

        for result in results {
            if seen.insert(result.entry.id.clone()) {
                deduped.push(result);
            }
        }

        // Sort by score
        deduped.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Limit results
        deduped.truncate(self.config.default_limit);

        deduped
    }

    async fn prefetch_related(&self, results: &[RetrievalResult]) {
        if self.config.prefetch_depth == 0 {
            return;
        }

        let mut to_prefetch = HashSet::new();

        for result in results {
            // Extract related IDs from metadata
            if let Some(related) = result.entry.metadata.get("related_ids") {
                if let Some(ids) = related.as_array() {
                    for id in ids {
                        if let Some(id_str) = id.as_str() {
                            to_prefetch.insert(id_str.to_string());
                        }
                    }
                }
            }
        }

        // Prefetch in background
        for id in to_prefetch.into_iter().take(10) {
            let self_clone = Arc::new(self.clone());
            tokio::spawn(async move {
                let _ = self_clone.retrieve_exact(&id).await;
            });
        }
    }

    fn update_metrics(&self, query: &QueryType, _result_count: usize, elapsed_ms: u64) {
        let mut metrics = self.metrics.write();

        metrics.total_queries += 1;

        let query_type = match query {
            QueryType::Exact { .. } => "exact",
            QueryType::Semantic { .. } => "semantic",
            QueryType::Pattern { .. } => "pattern",
            QueryType::Temporal { .. } => "temporal",
            QueryType::Combined { .. } => "combined",
            QueryType::ProjectContext { .. } => "project",
        };

        *metrics.queries_by_type.entry(query_type.to_string()).or_insert(0) += 1;

        // Update average retrieval time
        let total_time = metrics.avg_retrieval_time_ms * (metrics.total_queries - 1) as f64;
        metrics.avg_retrieval_time_ms = (total_time + elapsed_ms as f64) / metrics.total_queries as f64;
    }

    /// Get retrieval metrics
    pub fn get_metrics(&self) -> RetrievalMetrics {
        self.metrics.read().clone()
    }

    /// Clear all caches
    pub fn clear_cache(&self) {
        self.cache.l1_cache.clear();
        self.cache.l2_cache.clear();
        self.cache.l3_cache.clear();
    }

    /// Clear expired cache entries
    pub fn cleanup_cache(&self) {
        self.cache.clear_expired();
    }
}

// Implement Clone for the manager
impl Clone for MemoryRetrievalManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            memory_manager: self.memory_manager.clone(),
            archive_manager: self.archive_manager.clone(),
            vector_store: self.vector_store.clone(),
            cache: self.cache.clone(),
            query_optimizer: self.query_optimizer.clone(),
            search_semaphore: Arc::new(Semaphore::new(self.config.max_parallel_queries)),
            metrics: Arc::new(RwLock::new(RetrievalMetrics::default())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_operations() {
        let config = RetrievalConfig::default();
        let cache = MultiLevelCache::new(config);

        let entry = MemoryEntry::new(
            crate::memory::MemoryContent::Text("Test".to_string()),
            MemoryType::ShortTerm,
            0.5,
            None,
        );

        cache.insert("test_id".to_string(), entry.clone());

        assert!(cache.get("test_id").is_some());
        assert!(cache.get("non_existent").is_none());
    }

    #[test]
    fn test_query_optimization() {
        let optimizer = QueryOptimizer::new();

        let query = QueryType::Semantic {
            query: "test query".to_string(),
            embedding: None,
        };

        let optimized = optimizer.optimize_query(&query);

        match optimized {
            QueryType::Semantic { .. } => assert!(true),
            _ => assert!(false),
        }
    }
}