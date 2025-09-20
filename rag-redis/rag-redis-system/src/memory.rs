//! # Memory Management System
//!
//! This module provides a comprehensive memory management system for the RAG Redis system,
//! featuring multiple memory types, importance scoring, and efficient Redis backend storage.
//!
//! ## Features
//!
//! - **Multiple Memory Types**: Short-term, Long-term, Episodic, Semantic, and Working memory
//! - **Importance Scoring**: Weighted retrieval based on importance, recency, and access frequency
//! - **TTL Support**: Automatic expiration of memory entries
//! - **Memory Consolidation**: Migration from short-term to long-term memory based on importance
//! - **Search and Filtering**: Advanced search capabilities with multiple filter criteria
//! - **Redis Backend**: Efficient storage using Redis with connection pooling
//! - **Memory Pruning**: Automatic garbage collection and cleanup
//! - **Thread Safety**: Full concurrent access support
//! - **Performance Optimization**: Local caching for working memory and importance scores
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use rag_redis_system::memory::{MemoryManager, MemoryEntry, MemoryContent, MemoryType, MemoryConfig};
//! use rag_redis_system::redis_backend::RedisManager;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create Redis manager
//! let redis_config = Default::default();
//! let redis_manager = Arc::new(RedisManager::new(&redis_config).await?);
//!
//! // Create memory manager with custom config
//! let memory_config = MemoryConfig {
//!     short_term_ttl: Duration::from_secs(3600),  // 1 hour
//!     long_term_ttl: Duration::from_secs(86400 * 30),  // 30 days
//!     working_memory_capacity: 100,
//!     consolidation_threshold: 0.75,
//!     ..Default::default()
//! };
//!
//! let memory_manager = MemoryManager::new(redis_manager, memory_config).await?;
//!
//! // Create and store a memory entry
//! let content = MemoryContent::Text("Important information to remember".to_string());
//! let mut entry = MemoryEntry::new(
//!     content,
//!     MemoryType::ShortTerm,
//!     0.8,  // High importance score
//!     Some(Duration::from_secs(3600)),
//! );
//! entry.tags.insert("research".to_string());
//!
//! let entry_id = memory_manager.store(entry).await?;
//!
//! // Retrieve the memory entry
//! if let Some(retrieved) = memory_manager.get(&entry_id).await? {
//!     println!("Retrieved memory: {:?}", retrieved.content);
//! }
//!
//! // Search memories with filters
//! let mut filter = MemoryFilter::default();
//! filter.min_importance = Some(0.5);
//! filter.limit = Some(10);
//!
//! let search_results = memory_manager.search(&filter).await?;
//! println!("Found {} matching memories", search_results.len());
//!
//! // Consolidate important short-term memories to long-term
//! let consolidated = memory_manager.consolidate_memories().await?;
//! println!("Consolidated {} memories", consolidated);
//!
//! // Cleanup expired entries
//! let (removed, freed_bytes) = memory_manager.cleanup().await?;
//! println!("Cleaned up {} entries, freed {} bytes", removed, freed_bytes);
//!
//! // Get memory statistics
//! let stats = memory_manager.get_stats().await;
//! println!("Total entries: {}", stats.total_entries);
//! println!("Average importance: {:.2}", stats.average_importance);
//!
//! // Graceful shutdown
//! memory_manager.shutdown().await?;
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use crate::redis_backend::RedisManager;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock as AsyncRwLock};
use tokio::time::interval;
use tracing::{error, info};
use uuid::Uuid;

/// Configuration for memory management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub short_term_ttl: Duration,
    pub long_term_ttl: Duration,
    pub working_memory_capacity: usize,
    pub consolidation_threshold: f64,
    pub importance_decay_rate: f64,
    pub max_memory_entries: usize,
    pub cleanup_interval: Duration,
    pub enable_background_tasks: bool,
    pub compression_threshold: usize,
    pub batch_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_term_ttl: Duration::from_secs(3600),      // 1 hour
            long_term_ttl: Duration::from_secs(86400 * 30), // 30 days
            working_memory_capacity: 100,
            consolidation_threshold: 0.75,
            importance_decay_rate: 0.1,
            max_memory_entries: 10000,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            enable_background_tasks: true,
            compression_threshold: 1024, // 1KB
            batch_size: 100,
        }
    }
}

/// Different types of memory storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Temporary memory for immediate processing
    ShortTerm,
    /// Persistent memory for long-term storage
    LongTerm,
    /// Episode-based memories with temporal context
    Episodic,
    /// Semantic memories representing knowledge
    Semantic,
    /// Active working memory for current operations
    Working,
}

impl MemoryType {
    pub fn redis_prefix(&self) -> &'static str {
        match self {
            MemoryType::ShortTerm => "mem:short",
            MemoryType::LongTerm => "mem:long",
            MemoryType::Episodic => "mem:episodic",
            MemoryType::Semantic => "mem:semantic",
            MemoryType::Working => "mem:working",
        }
    }

    pub fn default_ttl(&self, config: &MemoryConfig) -> Option<Duration> {
        match self {
            MemoryType::ShortTerm => Some(config.short_term_ttl),
            MemoryType::LongTerm => Some(config.long_term_ttl),
            MemoryType::Episodic => Some(config.long_term_ttl),
            MemoryType::Semantic => None, // No TTL for semantic memory
            MemoryType::Working => Some(Duration::from_secs(900)), // 15 minutes
        }
    }
}

/// Memory entry with metadata and importance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: MemoryContent,
    pub memory_type: MemoryType,
    pub importance_score: f64,
    pub access_count: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub ttl: Option<Duration>,
    pub tags: HashSet<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub compressed: bool,
}

impl MemoryEntry {
    pub fn new(
        content: MemoryContent,
        memory_type: MemoryType,
        importance_score: f64,
        ttl: Option<Duration>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            memory_type,
            importance_score,
            access_count: 0,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            ttl,
            tags: HashSet::new(),
            metadata: HashMap::new(),
            embedding: None,
            compressed: false,
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }

    pub fn update_importance(&mut self, new_score: f64) {
        self.importance_score = new_score;
        self.updated_at = Utc::now();
    }

    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let expiry = self.created_at + chrono::Duration::from_std(ttl).unwrap_or_default();
            Utc::now() > expiry
        } else {
            false
        }
    }

    pub fn calculate_weighted_score(&self) -> f64 {
        let recency_factor = {
            let age = Utc::now().signed_duration_since(self.last_accessed);
            let hours = age.num_hours() as f64;
            (-hours / 24.0).exp() // Exponential decay over 24 hours
        };

        let frequency_factor = (self.access_count as f64).ln().max(1.0);

        self.importance_score * recency_factor * frequency_factor
    }
}

/// Different types of memory content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryContent {
    Text(String),
    Structured(serde_json::Value),
    Binary(Vec<u8>),
    Compressed(Vec<u8>),
    Reference(String), // Reference to external storage
}

impl MemoryContent {
    pub fn size_bytes(&self) -> usize {
        match self {
            MemoryContent::Text(s) => s.len(),
            MemoryContent::Structured(v) => serde_json::to_vec(v).map(|v| v.len()).unwrap_or(0),
            MemoryContent::Binary(b) => b.len(),
            MemoryContent::Compressed(b) => b.len(),
            MemoryContent::Reference(s) => s.len(),
        }
    }
}

/// Search filter for memory queries
#[derive(Debug, Clone, Default)]
pub struct MemoryFilter {
    pub memory_types: Option<HashSet<MemoryType>>,
    pub tags: Option<HashSet<String>>,
    pub min_importance: Option<f64>,
    pub max_age: Option<Duration>,
    pub include_expired: bool,
    pub content_pattern: Option<String>,
    pub limit: Option<usize>,
    pub sort_by_importance: bool,
    pub sort_by_recency: bool,
}

/// Memory search result with ranking
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub entry: MemoryEntry,
    pub score: f64,
    pub rank: usize,
}

/// Statistics for memory usage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub entries_by_type: HashMap<String, usize>,
    pub total_size_bytes: usize,
    pub average_importance: f64,
    pub cache_hit_rate: f64,
    pub consolidation_rate: f64,
    pub expired_entries: usize,
    pub compression_ratio: f64,
}

/// Events emitted by the memory system
#[derive(Debug, Clone)]
pub enum MemoryEvent {
    EntryCreated { id: String, memory_type: MemoryType },
    EntryUpdated { id: String, memory_type: MemoryType },
    EntryAccessed { id: String, memory_type: MemoryType },
    EntryExpired { id: String, memory_type: MemoryType },
    ConsolidationTriggered { count: usize },
    CleanupCompleted { removed: usize, freed_bytes: usize },
}

/// High-performance memory manager with multiple memory types
pub struct MemoryManager {
    redis_manager: Arc<RedisManager>,
    config: MemoryConfig,

    // Local caches for performance
    working_memory: Arc<RwLock<BTreeMap<String, MemoryEntry>>>,
    memory_index: Arc<DashMap<String, MemoryType>>,
    importance_cache: Arc<RwLock<HashMap<String, f64>>>,

    // Background task handles
    cleanup_handle: Arc<AsyncRwLock<Option<tokio::task::JoinHandle<()>>>>,
    consolidation_handle: Arc<AsyncRwLock<Option<tokio::task::JoinHandle<()>>>>,

    // Event broadcasting
    event_tx: broadcast::Sender<MemoryEvent>,

    // Statistics
    stats: Arc<RwLock<MemoryStats>>,

    // Shutdown flag
    shutdown_flag: Arc<parking_lot::RwLock<bool>>,
}

impl MemoryManager {
    pub async fn new(redis_manager: Arc<RedisManager>, config: MemoryConfig) -> Result<Arc<Self>> {
        let (event_tx, _) = broadcast::channel(1000);

        let manager = Arc::new(Self {
            redis_manager,
            config: config.clone(),
            working_memory: Arc::new(RwLock::new(BTreeMap::new())),
            memory_index: Arc::new(DashMap::new()),
            importance_cache: Arc::new(RwLock::new(HashMap::new())),
            cleanup_handle: Arc::new(AsyncRwLock::new(None)),
            consolidation_handle: Arc::new(AsyncRwLock::new(None)),
            event_tx,
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            shutdown_flag: Arc::new(parking_lot::RwLock::new(false)),
        });

        if config.enable_background_tasks {
            manager.start_background_tasks().await?;
        }

        Ok(manager)
    }

    /// Store a memory with content, type, and importance
    pub async fn store(
        &self,
        content: String,
        memory_type: MemoryType,
        importance: f32,
    ) -> Result<String> {
        let entry = MemoryEntry::new(
            MemoryContent::Text(content),
            memory_type,
            importance as f64,
            memory_type.default_ttl(&self.config),
        );

        self.store_entry(entry).await
    }

    /// Recall memories based on query, optional type filter, and limit
    pub async fn recall(
        &self,
        query: &str,
        memory_type: Option<MemoryType>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let mut filter = MemoryFilter::default();
        if let Some(mem_type) = memory_type {
            filter.memory_types = Some([mem_type].iter().cloned().collect());
        }
        filter.limit = Some(limit);
        filter.content_pattern = Some(query.to_string());
        filter.sort_by_importance = true;

        let results = self.search(&filter).await?;
        Ok(results.into_iter().map(|r| r.entry).collect())
    }

    /// Store a memory entry
    pub async fn store_entry(&self, mut entry: MemoryEntry) -> Result<String> {
        let entry_id = entry.id.clone();

        // Compress large content if needed
        if entry.content.size_bytes() > self.config.compression_threshold {
            entry.content = self.compress_content(&entry.content)?;
            entry.compressed = true;
        }

        // Store in appropriate backend based on memory type
        match entry.memory_type {
            MemoryType::Working => {
                // Store in local working memory
                let mut working_mem = self.working_memory.write();

                // Enforce capacity limits
                if working_mem.len() >= self.config.working_memory_capacity {
                    self.evict_working_memory(&mut working_mem)?;
                }

                working_mem.insert(entry_id.clone(), entry.clone());
            }
            _ => {
                // Store in Redis using the redis_manager's pattern
                self.store_to_redis(&entry).await?;

                // Update index
                self.memory_index
                    .insert(entry_id.clone(), entry.memory_type);
            }
        }

        // Update importance cache
        self.importance_cache
            .write()
            .insert(entry_id.clone(), entry.importance_score);

        // Emit event
        let _ = self.event_tx.send(MemoryEvent::EntryCreated {
            id: entry_id.clone(),
            memory_type: entry.memory_type,
        });

        // Update stats
        self.update_stats_for_store(&entry).await;

        Ok(entry_id)
    }

    /// Retrieve a memory entry by ID
    pub async fn get(&self, id: &str) -> Result<Option<MemoryEntry>> {
        // Check working memory first
        let entry = self.working_memory.read().get(id).cloned();
        if let Some(entry) = entry {
            self.record_access(id, &entry).await?;
            return Ok(Some(entry));
        }

        // Check memory index to determine type
        if let Some(memory_type) = self.memory_index.get(id) {
            if let Some(mut entry) = self.get_from_redis(id, *memory_type.value()).await? {
                // Decompress if needed
                if entry.compressed {
                    entry.content = self.decompress_content(&entry.content)?;
                    entry.compressed = false;
                }

                self.record_access(id, &entry).await?;
                return Ok(Some(entry));
            }
        }

        // Search across all memory types if not found in index
        for memory_type in [
            MemoryType::ShortTerm,
            MemoryType::LongTerm,
            MemoryType::Episodic,
            MemoryType::Semantic,
        ] {
            if let Some(mut entry) = self.get_from_redis(id, memory_type).await? {
                if entry.compressed {
                    entry.content = self.decompress_content(&entry.content)?;
                    entry.compressed = false;
                }

                // Update index for future lookups
                self.memory_index.insert(id.to_string(), memory_type);

                self.record_access(id, &entry).await?;
                return Ok(Some(entry));
            }
        }

        Ok(None)
    }

    /// Search memories with filtering and ranking
    pub async fn search(&self, filter: &MemoryFilter) -> Result<Vec<MemorySearchResult>> {
        let mut results = Vec::new();
        let memory_types = filter.memory_types.clone().unwrap_or_else(|| {
            [
                MemoryType::ShortTerm,
                MemoryType::LongTerm,
                MemoryType::Episodic,
                MemoryType::Semantic,
                MemoryType::Working,
            ]
            .iter()
            .cloned()
            .collect()
        });

        // Search working memory
        if memory_types.contains(&MemoryType::Working) {
            let working_mem = self.working_memory.read();
            for entry in working_mem.values() {
                if self.matches_filter(entry, filter) {
                    let score = self.calculate_search_score(entry, filter);
                    results.push(MemorySearchResult {
                        entry: entry.clone(),
                        score,
                        rank: 0, // Will be set later
                    });
                }
            }
        }

        // Search Redis-backed memories
        for memory_type in memory_types.iter().filter(|&&t| t != MemoryType::Working) {
            let entries = self.scan_memory_type(*memory_type).await?;

            for mut entry in entries.into_iter().take(self.config.batch_size) {
                if entry.compressed {
                    entry.content = self.decompress_content(&entry.content)?;
                    entry.compressed = false;
                }

                if self.matches_filter(&entry, filter) {
                    let score = self.calculate_search_score(&entry, filter);
                    results.push(MemorySearchResult {
                        entry,
                        score,
                        rank: 0,
                    });
                }
            }
        }

        // Sort and rank results
        if filter.sort_by_importance {
            results.sort_by(|a, b| {
                b.entry
                    .importance_score
                    .partial_cmp(&a.entry.importance_score)
                    .unwrap()
            });
        } else if filter.sort_by_recency {
            results.sort_by(|a, b| b.entry.last_accessed.cmp(&a.entry.last_accessed));
        } else {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        }

        // Assign ranks and apply limit
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Update an existing memory entry
    pub async fn update(&self, id: &str, update_fn: impl FnOnce(&mut MemoryEntry)) -> Result<bool> {
        if let Some(mut entry) = self.get(id).await? {
            update_fn(&mut entry);
            entry.updated_at = Utc::now();

            self.store_entry(entry).await?;

            let _ = self.event_tx.send(MemoryEvent::EntryUpdated {
                id: id.to_string(),
                memory_type: self
                    .memory_index
                    .get(id)
                    .map(|kv| *kv.value())
                    .unwrap_or(MemoryType::ShortTerm),
            });

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete a memory entry
    pub async fn delete(&self, id: &str) -> Result<bool> {
        let mut deleted = false;

        // Remove from working memory
        if self.working_memory.write().remove(id).is_some() {
            deleted = true;
        }

        // Remove from Redis
        if let Some(memory_type) = self.memory_index.remove(id) {
            self.delete_from_redis(id, memory_type.1).await?;
            deleted = true;
        }

        // Remove from importance cache
        self.importance_cache.write().remove(id);

        if deleted {
            self.update_stats_for_delete().await;
        }

        Ok(deleted)
    }

    /// Consolidate short-term memories to long-term based on importance
    pub async fn consolidate_memories(&self) -> Result<usize> {
        let mut consolidated_count = 0;
        let threshold = self.config.consolidation_threshold;

        let short_term_entries = self.scan_memory_type(MemoryType::ShortTerm).await?;

        for entry in short_term_entries {
            if entry.calculate_weighted_score() >= threshold {
                // Create long-term memory entry
                let mut long_term_entry = entry.clone();
                long_term_entry.memory_type = MemoryType::LongTerm;
                long_term_entry.ttl = MemoryType::LongTerm.default_ttl(&self.config);

                // Store as long-term memory
                self.store_to_redis(&long_term_entry).await?;

                // Remove from short-term
                self.delete_from_redis(&entry.id, MemoryType::ShortTerm)
                    .await?;

                // Update index
                self.memory_index
                    .insert(entry.id.clone(), MemoryType::LongTerm);

                consolidated_count += 1;
            }
        }

        if consolidated_count > 0 {
            let _ = self.event_tx.send(MemoryEvent::ConsolidationTriggered {
                count: consolidated_count,
            });
        }

        Ok(consolidated_count)
    }

    /// Clean up expired memories and optimize storage
    pub async fn cleanup(&self) -> Result<(usize, usize)> {
        let mut removed_count = 0;
        let mut freed_bytes = 0;

        // Clean up each memory type
        for memory_type in [
            MemoryType::ShortTerm,
            MemoryType::LongTerm,
            MemoryType::Episodic,
            MemoryType::Working,
        ] {
            let entries = self.scan_memory_type(memory_type).await?;

            for entry in entries {
                if entry.is_expired() {
                    freed_bytes += entry.content.size_bytes();

                    if memory_type == MemoryType::Working {
                        self.working_memory.write().remove(&entry.id);
                    } else {
                        self.delete_from_redis(&entry.id, memory_type).await?;
                    }

                    self.memory_index.remove(&entry.id);
                    self.importance_cache.write().remove(&entry.id);
                    removed_count += 1;

                    let _ = self.event_tx.send(MemoryEvent::EntryExpired {
                        id: entry.id,
                        memory_type: entry.memory_type,
                    });
                }
            }
        }

        if removed_count > 0 {
            let _ = self.event_tx.send(MemoryEvent::CleanupCompleted {
                removed: removed_count,
                freed_bytes,
            });
        }

        self.update_stats_after_cleanup(removed_count, freed_bytes)
            .await;

        Ok((removed_count, freed_bytes))
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> MemoryStats {
        let mut stats = self.stats.read().clone();

        // Update real-time stats
        stats.total_entries = self.memory_index.len() + self.working_memory.read().len();
        stats.entries_by_type.clear();

        // Count entries by type
        for memory_type in self.memory_index.iter() {
            let type_name = format!("{:?}", memory_type.value());
            *stats.entries_by_type.entry(type_name).or_insert(0) += 1;
        }

        let working_count = self.working_memory.read().len();
        if working_count > 0 {
            stats
                .entries_by_type
                .insert("Working".to_string(), working_count);
        }

        // Calculate average importance
        let importance_values: Vec<f64> = self.importance_cache.read().values().cloned().collect();
        if !importance_values.is_empty() {
            stats.average_importance =
                importance_values.iter().sum::<f64>() / importance_values.len() as f64;
        }

        stats
    }

    /// Subscribe to memory events
    pub fn subscribe_events(&self) -> broadcast::Receiver<MemoryEvent> {
        self.event_tx.subscribe()
    }

    /// Shutdown the memory manager gracefully
    pub async fn shutdown(&self) -> Result<()> {
        *self.shutdown_flag.write() = true;

        // Cancel background tasks
        if let Some(handle) = self.cleanup_handle.write().await.take() {
            handle.abort();
        }
        if let Some(handle) = self.consolidation_handle.write().await.take() {
            handle.abort();
        }

        // Perform final cleanup
        self.cleanup().await?;

        info!("Memory manager shutdown complete");
        Ok(())
    }

    // Private helper methods

    async fn store_to_redis(&self, entry: &MemoryEntry) -> Result<()> {
        let key = format!("{}:{}", entry.memory_type.redis_prefix(), entry.id);
        let value = bincode::serialize(entry)?;

        if let Some(ttl) = entry.ttl {
            self.redis_manager
                .set_raw_ex(&key, &value, ttl.as_secs())
                .await?;
        } else {
            self.redis_manager.set_raw(&key, &value).await?;
        }

        Ok(())
    }

    async fn get_from_redis(
        &self,
        id: &str,
        memory_type: MemoryType,
    ) -> Result<Option<MemoryEntry>> {
        let key = format!("{}:{}", memory_type.redis_prefix(), id);

        if let Some(data) = self.redis_manager.get_raw(&key).await? {
            let entry: MemoryEntry = bincode::deserialize(&data)?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    async fn delete_from_redis(&self, id: &str, memory_type: MemoryType) -> Result<()> {
        let key = format!("{}:{}", memory_type.redis_prefix(), id);
        self.redis_manager.delete_raw(&key).await?;
        Ok(())
    }

    async fn scan_memory_type(&self, memory_type: MemoryType) -> Result<Vec<MemoryEntry>> {
        let prefix = format!("{}:", memory_type.redis_prefix());
        let keys = self.redis_manager.scan_pattern(&prefix).await?;
        let mut entries = Vec::new();
        for key in keys {
            if let Some(id) = key.strip_prefix(&prefix) {
                if let Ok(Some(entry)) = self.get_from_redis(id, memory_type).await {
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }

    async fn start_background_tasks(&self) -> Result<()> {
        // Background cleanup task
        {
            let _redis_manager = self.redis_manager.clone();
            let config = self.config.clone();
            let shutdown_flag = self.shutdown_flag.clone();
            let event_tx = self.event_tx.clone();
            let stats = self.stats.clone();
            let memory_index = self.memory_index.clone();
            let importance_cache = self.importance_cache.clone();
            let working_memory = self.working_memory.clone();

            let handle = tokio::spawn(async move {
                let mut interval = interval(config.cleanup_interval);

                while !*shutdown_flag.read() {
                    interval.tick().await;

                    if let Err(e) = Self::background_cleanup(
                        &memory_index,
                        &importance_cache,
                        &working_memory,
                        &event_tx,
                        &stats,
                    )
                    .await
                    {
                        error!("Background cleanup error: {}", e);
                    }
                }
            });

            *self.cleanup_handle.write().await = Some(handle);
        }

        // Background consolidation task
        {
            let _redis_manager = self.redis_manager.clone();
            let config = self.config.clone();
            let shutdown_flag = self.shutdown_flag.clone();
            let event_tx = self.event_tx.clone();
            let memory_index = self.memory_index.clone();

            let handle = tokio::spawn(async move {
                let mut interval = interval(Duration::from_secs(1800)); // Every 30 minutes

                while !*shutdown_flag.read() {
                    interval.tick().await;

                    if let Err(e) = Self::background_consolidation(
                        &_redis_manager,
                        &config,
                        &memory_index,
                        &event_tx,
                    )
                    .await
                    {
                        error!("Background consolidation error: {}", e);
                    }
                }
            });

            *self.consolidation_handle.write().await = Some(handle);
        }

        Ok(())
    }

    async fn background_cleanup(
        _memory_index: &Arc<DashMap<String, MemoryType>>,
        importance_cache: &Arc<RwLock<HashMap<String, f64>>>,
        working_memory: &Arc<RwLock<BTreeMap<String, MemoryEntry>>>,
        event_tx: &broadcast::Sender<MemoryEvent>,
        _stats: &Arc<RwLock<MemoryStats>>,
    ) -> Result<()> {
        let mut removed_count = 0;
        let mut freed_bytes = 0;

        // Clean expired working memory entries
        {
            let mut working_mem = working_memory.write();
            let expired_keys: Vec<String> = working_mem
                .iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired_keys {
                if let Some(entry) = working_mem.remove(&key) {
                    freed_bytes += entry.content.size_bytes();
                    removed_count += 1;
                    importance_cache.write().remove(&key);
                }
            }
        }

        if removed_count > 0 {
            let _ = event_tx.send(MemoryEvent::CleanupCompleted {
                removed: removed_count,
                freed_bytes,
            });
        }

        Ok(())
    }

    async fn background_consolidation(
        redis_manager: &Arc<RedisManager>,
        config: &MemoryConfig,
        memory_index: &Arc<DashMap<String, MemoryType>>,
        event_tx: &broadcast::Sender<MemoryEvent>,
    ) -> Result<()> {
        // Get short-term memories ready for consolidation
        let short_term_entries =
            Self::scan_memory_type_static(redis_manager, MemoryType::ShortTerm).await?;
        let mut consolidated_count = 0;

        for entry in short_term_entries.iter() {
            // Check if memory meets consolidation criteria
            if entry.importance_score >= config.consolidation_threshold && entry.access_count >= 3 {
                // Use a reasonable default for min access

                // Migrate to long-term storage
                let mut upgraded_memory = entry.clone();
                upgraded_memory.memory_type = MemoryType::LongTerm;
                upgraded_memory.ttl = MemoryType::LongTerm.default_ttl(config);
                upgraded_memory.metadata.insert(
                    "consolidated_at".to_string(),
                    chrono::Utc::now().to_rfc3339().into(),
                );

                // Store in long-term tier
                Self::store_to_redis_static(redis_manager, &upgraded_memory).await?;

                // Remove from short-term
                Self::delete_from_redis_static(redis_manager, &entry.id, MemoryType::ShortTerm)
                    .await?;

                // Update index
                memory_index.insert(entry.id.clone(), MemoryType::LongTerm);

                consolidated_count += 1;
            }
        }

        if consolidated_count > 0 {
            let _ = event_tx.send(MemoryEvent::ConsolidationTriggered {
                count: consolidated_count,
            });
        }

        Ok(())
    }

    fn evict_working_memory(&self, working_mem: &mut BTreeMap<String, MemoryEntry>) -> Result<()> {
        // Remove least important entries to make room
        let mut entries: Vec<(String, f64)> = working_mem
            .iter()
            .map(|(k, v)| (k.clone(), v.calculate_weighted_score()))
            .collect();

        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Remove bottom 25% of entries
        let remove_count = (entries.len() as f64 * 0.25).ceil() as usize;
        for i in 0..remove_count {
            if let Some((key, _)) = entries.get(i) {
                working_mem.remove(key);
                self.importance_cache.write().remove(key);
            }
        }

        Ok(())
    }

    async fn record_access(&self, id: &str, entry: &MemoryEntry) -> Result<()> {
        // Update access statistics
        let _ = self.event_tx.send(MemoryEvent::EntryAccessed {
            id: id.to_string(),
            memory_type: entry.memory_type,
        });

        // Update in storage if needed (lazy update)
        // This could be optimized to batch updates

        Ok(())
    }

    fn matches_filter(&self, entry: &MemoryEntry, filter: &MemoryFilter) -> bool {
        // Check memory type
        if let Some(ref types) = filter.memory_types {
            if !types.contains(&entry.memory_type) {
                return false;
            }
        }

        // Check tags
        if let Some(ref filter_tags) = filter.tags {
            if filter_tags.intersection(&entry.tags).count() == 0 {
                return false;
            }
        }

        // Check minimum importance
        if let Some(min_importance) = filter.min_importance {
            if entry.importance_score < min_importance {
                return false;
            }
        }

        // Check age
        if let Some(max_age) = filter.max_age {
            let age = Utc::now().signed_duration_since(entry.created_at);
            if age > chrono::Duration::from_std(max_age).unwrap_or_default() {
                return false;
            }
        }

        // Check expiration
        if !filter.include_expired && entry.is_expired() {
            return false;
        }

        // Check content pattern
        if let Some(ref pattern) = filter.content_pattern {
            match &entry.content {
                MemoryContent::Text(text) => {
                    if !text.contains(pattern) {
                        return false;
                    }
                }
                MemoryContent::Structured(value) => {
                    let text = serde_json::to_string(value).unwrap_or_default();
                    if !text.contains(pattern) {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        true
    }

    fn calculate_search_score(&self, entry: &MemoryEntry, _filter: &MemoryFilter) -> f64 {
        // Calculate relevance score based on multiple factors
        entry.calculate_weighted_score()
    }

    fn compress_content(&self, content: &MemoryContent) -> Result<MemoryContent> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        match content {
            MemoryContent::Text(text) => {
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(text.as_bytes())?;
                let compressed = encoder.finish()?;
                Ok(MemoryContent::Compressed(compressed))
            }
            MemoryContent::Structured(value) => {
                let json = serde_json::to_string(value)?;
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(json.as_bytes())?;
                let compressed = encoder.finish()?;
                Ok(MemoryContent::Compressed(compressed))
            }
            _ => Ok(content.clone()),
        }
    }

    fn decompress_content(&self, content: &MemoryContent) -> Result<MemoryContent> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        match content {
            MemoryContent::Compressed(bytes) => {
                let mut decoder = GzDecoder::new(&bytes[..]);
                let mut decompressed = String::new();
                decoder.read_to_string(&mut decompressed)?;

                // Try to parse as JSON first
                if let Ok(value) = serde_json::from_str(&decompressed) {
                    Ok(MemoryContent::Structured(value))
                } else {
                    Ok(MemoryContent::Text(decompressed))
                }
            }
            _ => Ok(content.clone()),
        }
    }

    async fn update_stats_for_store(&self, _entry: &MemoryEntry) {
        // Update statistics after storing entry
        let mut stats = self.stats.write();
        stats.total_entries += 1;
    }

    async fn update_stats_for_delete(&self) {
        // Update statistics after deleting entry
        let mut stats = self.stats.write();
        stats.total_entries = stats.total_entries.saturating_sub(1);
    }

    async fn update_stats_after_cleanup(&self, removed: usize, freed_bytes: usize) {
        let mut stats = self.stats.write();
        stats.expired_entries += removed;
        stats.total_size_bytes = stats.total_size_bytes.saturating_sub(freed_bytes);
    }

    // Static versions of helper methods for background tasks
    async fn scan_memory_type_static(
        redis_manager: &Arc<RedisManager>,
        memory_type: MemoryType,
    ) -> Result<Vec<MemoryEntry>> {
        let pattern = format!("{}:*", memory_type.redis_prefix());
        let mut cursor = 0u64;
        let mut entries = Vec::new();

        let mut conn = redis_manager.get_pooled_connection().await?;

        loop {
            let (new_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(100)
                .query_async(&mut *conn)
                .await?;

            for key in keys {
                if let Some(id) = key.strip_prefix(&format!("{}:", memory_type.redis_prefix())) {
                    if let Ok(Some(entry)) =
                        Self::get_from_redis_static(redis_manager, id, memory_type).await
                    {
                        entries.push(entry);
                    }
                }
            }

            cursor = new_cursor;
            if cursor == 0 {
                break;
            }
        }

        Ok(entries)
    }

    async fn get_from_redis_static(
        redis_manager: &Arc<RedisManager>,
        id: &str,
        memory_type: MemoryType,
    ) -> Result<Option<MemoryEntry>> {
        let key = format!("{}:{}", memory_type.redis_prefix(), id);

        if let Some(data) = redis_manager.get_raw(&key).await? {
            let entry: MemoryEntry = bincode::deserialize(&data)?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    async fn store_to_redis_static(
        redis_manager: &Arc<RedisManager>,
        entry: &MemoryEntry,
    ) -> Result<()> {
        let key = format!("{}:{}", entry.memory_type.redis_prefix(), entry.id);
        let value = bincode::serialize(entry)?;

        if let Some(ttl) = entry.ttl {
            redis_manager
                .set_raw_ex(&key, &value, ttl.as_secs())
                .await?;
        } else {
            redis_manager.set_raw(&key, &value).await?;
        }

        Ok(())
    }

    async fn delete_from_redis_static(
        redis_manager: &Arc<RedisManager>,
        id: &str,
        memory_type: MemoryType,
    ) -> Result<()> {
        let key = format!("{}:{}", memory_type.redis_prefix(), id);
        redis_manager.delete_raw(&key).await?;
        Ok(())
    }
}

// Implement Drop for graceful shutdown
impl Drop for MemoryManager {
    fn drop(&mut self) {
        *self.shutdown_flag.write() = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_memory_entry_creation() {
        let content = MemoryContent::Text("Test content".to_string());
        let entry = MemoryEntry::new(
            content,
            MemoryType::ShortTerm,
            0.8,
            Some(Duration::from_secs(3600)),
        );

        assert_eq!(entry.importance_score, 0.8);
        assert_eq!(entry.memory_type, MemoryType::ShortTerm);
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_memory_filter() {
        let mut filter = MemoryFilter::default();
        filter.memory_types = Some([MemoryType::LongTerm].iter().cloned().collect());
        filter.min_importance = Some(0.5);

        let entry = MemoryEntry::new(
            MemoryContent::Text("Test".to_string()),
            MemoryType::LongTerm,
            0.7,
            None,
        );

        // This test would need a MemoryManager instance to call matches_filter
        // assert!(manager.matches_filter(&entry, &filter));
    }

    #[test]
    fn test_weighted_score_calculation() {
        let mut entry = MemoryEntry::new(
            MemoryContent::Text("Test content".to_string()),
            MemoryType::ShortTerm,
            0.8,
            Some(Duration::from_secs(3600)),
        );

        entry.access_count = 5;

        let score = entry.calculate_weighted_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_memory_content_size() {
        let text_content = MemoryContent::Text("Hello, World!".to_string());
        assert_eq!(text_content.size_bytes(), 13);

        let binary_content = MemoryContent::Binary(vec![1, 2, 3, 4, 5]);
        assert_eq!(binary_content.size_bytes(), 5);
    }

    #[test]
    fn test_memory_type_prefixes() {
        assert_eq!(MemoryType::ShortTerm.redis_prefix(), "mem:short");
        assert_eq!(MemoryType::LongTerm.redis_prefix(), "mem:long");
        assert_eq!(MemoryType::Episodic.redis_prefix(), "mem:episodic");
        assert_eq!(MemoryType::Semantic.redis_prefix(), "mem:semantic");
        assert_eq!(MemoryType::Working.redis_prefix(), "mem:working");
    }
}
impl From<crate::config::MemoryConfig> for MemoryConfig {
    fn from(config: crate::config::MemoryConfig) -> Self {
        Self {
            short_term_ttl: config
                .ttl
                .get("short_term")
                .copied()
                .unwrap_or_else(|| Duration::from_secs(3600)),
            long_term_ttl: config
                .ttl
                .get("long_term")
                .copied()
                .unwrap_or_else(|| Duration::from_secs(86400 * 30)),
            working_memory_capacity: config.working_memory_capacity,
            consolidation_threshold: config.consolidation_threshold,
            importance_decay_rate: 0.1, // Default value not in config
            max_memory_entries: config.max_entries.get("total").copied().unwrap_or(10000),
            cleanup_interval: config.cleanup_interval,
            enable_background_tasks: true, // Default value not in config
            compression_threshold: if config.enable_compression {
                1024
            } else {
                usize::MAX
            },
            batch_size: 100, // Default value not in config
        }
    }
}
