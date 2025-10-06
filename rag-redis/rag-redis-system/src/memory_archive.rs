//! Enhanced Memory Archival and Retrieval System
//!
//! This module implements automatic memory consolidation, hierarchical storage,
//! compression, importance scoring with decay, and semantic indexing for the
//! 5-tier memory system.

use crate::{
    error::Result,
    memory::{MemoryEntry, MemoryType, MemoryContent, MemoryManager},
    redis_backend::RedisManager,
    vector_store::VectorStore,
};
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn, debug};

/// Archive configuration with enhanced parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveConfig {
    /// Access pattern tracking window (hours)
    pub access_pattern_window: u32,

    /// Minimum access count for hot data
    pub hot_data_threshold: u32,

    /// Compression threshold in bytes
    pub compression_threshold: usize,

    /// Maximum archive depth (levels)
    pub max_archive_depth: usize,

    /// Importance decay rate per day
    pub importance_decay_daily: f64,

    /// Minimum importance for archival
    pub min_archive_importance: f64,

    /// Batch size for archival operations
    pub archive_batch_size: usize,

    /// Enable automatic consolidation
    pub auto_consolidation: bool,

    /// Consolidation interval (seconds)
    pub consolidation_interval: u64,

    /// Project context retention days
    pub project_context_retention_days: u32,

    /// Semantic index update interval (seconds)
    pub semantic_index_interval: u64,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            access_pattern_window: 24,
            hot_data_threshold: 5,
            compression_threshold: 1024,
            max_archive_depth: 3,
            importance_decay_daily: 0.05,
            min_archive_importance: 0.1,
            archive_batch_size: 100,
            auto_consolidation: true,
            consolidation_interval: 3600,
            project_context_retention_days: 90,
            semantic_index_interval: 1800,
        }
    }
}

/// Access pattern tracking for intelligent archival
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub memory_id: String,
    pub access_times: Vec<DateTime<Utc>>,
    pub total_accesses: u64,
    pub last_access: DateTime<Utc>,
    pub access_frequency: f64,
    pub is_hot: bool,
}

impl AccessPattern {
    pub fn new(memory_id: String) -> Self {
        Self {
            memory_id,
            access_times: Vec::new(),
            total_accesses: 0,
            last_access: Utc::now(),
            access_frequency: 0.0,
            is_hot: false,
        }
    }

    pub fn record_access(&mut self) {
        let now = Utc::now();
        self.access_times.push(now);
        self.total_accesses += 1;
        self.last_access = now;
        self.update_frequency();
    }

    fn update_frequency(&mut self) {
        if self.access_times.len() < 2 {
            self.access_frequency = 0.0;
            return;
        }

        // Calculate access frequency in the last window
        let window_start = Utc::now() - Duration::hours(24);
        let recent_accesses = self.access_times
            .iter()
            .filter(|t| **t > window_start)
            .count() as f64;

        self.access_frequency = recent_accesses / 24.0; // Accesses per hour
    }

    pub fn should_archive(&self, threshold: u32) -> bool {
        self.total_accesses < threshold as u64 && !self.is_hot
    }
}

/// Hierarchical memory level for tiered storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArchiveLevel {
    Active = 0,     // In active memory tiers
    Warm = 1,       // Recently accessed, uncompressed
    Cool = 2,       // Compressed, indexed
    Cold = 3,       // Highly compressed, archived
}

/// Memory importance scorer with temporal decay
#[derive(Debug, Clone)]
pub struct ImportanceScorer {
    decay_rate: f64,
    access_weight: f64,
    recency_weight: f64,
    semantic_weight: f64,
    context_weight: f64,
}

impl ImportanceScorer {
    pub fn new(config: &ArchiveConfig) -> Self {
        Self {
            decay_rate: config.importance_decay_daily,
            access_weight: 0.3,
            recency_weight: 0.2,
            semantic_weight: 0.25,
            context_weight: 0.25,
        }
    }

    pub fn calculate_score(
        &self,
        entry: &MemoryEntry,
        access_pattern: &AccessPattern,
        semantic_relevance: f64,
        context_relevance: f64,
    ) -> f64 {
        // Time-based decay
        let age_days = (Utc::now() - entry.created_at).num_days() as f64;
        let time_decay = (-self.decay_rate * age_days).exp();

        // Access frequency score
        let access_score = (access_pattern.total_accesses as f64).ln().max(1.0) / 10.0;

        // Recency score
        let recency_hours = (Utc::now() - access_pattern.last_access).num_hours() as f64;
        let recency_score = (-recency_hours / 168.0).exp(); // Decay over a week

        // Weighted combination
        let score = entry.importance_score * time_decay
            + access_score * self.access_weight
            + recency_score * self.recency_weight
            + semantic_relevance * self.semantic_weight
            + context_relevance * self.context_weight;

        score.min(1.0).max(0.0)
    }
}

/// Semantic index for fast retrieval
#[derive(Debug)]
pub struct SemanticIndex {
    vector_store: Arc<RwLock<VectorStore>>,
    concept_graph: Arc<DashMap<String, HashSet<String>>>,
    term_index: Arc<DashMap<String, HashSet<String>>>,
    cluster_map: Arc<DashMap<String, usize>>,
}

impl SemanticIndex {
    pub async fn new(vector_store: Arc<RwLock<VectorStore>>) -> Self {
        Self {
            vector_store,
            concept_graph: Arc::new(DashMap::new()),
            term_index: Arc::new(DashMap::new()),
            cluster_map: Arc::new(DashMap::new()),
        }
    }

    pub async fn index_memory(&self, entry: &MemoryEntry) -> Result<()> {
        // Extract terms and concepts
        let terms = self.extract_terms(&entry.content);
        let concepts = self.extract_concepts(&terms);

        // Update term index
        for term in &terms {
            self.term_index
                .entry(term.clone())
                .or_insert_with(HashSet::new)
                .insert(entry.id.clone());
        }

        // Update concept graph
        for concept in &concepts {
            self.concept_graph
                .entry(concept.clone())
                .or_insert_with(HashSet::new)
                .insert(entry.id.clone());
        }

        // Add to vector store if embedding exists
        if let Some(embedding) = &entry.embedding {
            let metadata = serde_json::json!({
                "memory_id": entry.id,
                "memory_type": format!("{:?}", entry.memory_type),
                "importance": entry.importance_score,
            });

            self.vector_store.write().add_vector(
                &entry.id,
                embedding,
                metadata,
            )?;
        }

        Ok(())
    }

    fn extract_terms(&self, content: &MemoryContent) -> Vec<String> {
        match content {
            MemoryContent::Text(text) => {
                // Simple tokenization - can be enhanced with NLP
                text.to_lowercase()
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .map(|w| w.to_string())
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    fn extract_concepts(&self, terms: &[String]) -> Vec<String> {
        // Extract high-level concepts from terms
        // This is a simplified implementation - could use NLP/ML
        terms.iter()
            .filter(|t| t.len() > 5)
            .take(5)
            .cloned()
            .collect()
    }

    pub async fn search_similar(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<String>> {
        let results = self.vector_store.read().search(
            query_embedding,
            limit,
            None,
        )?;

        Ok(results.into_iter().map(|(id, _, _)| id).collect())
    }
}

/// Project context storage for domain-specific memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    pub project_id: String,
    pub domain: String,
    pub key_concepts: HashSet<String>,
    pub important_memories: Vec<String>,
    pub relationships: HashMap<String, Vec<String>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ProjectContext {
    pub fn new(project_id: String, domain: String) -> Self {
        let now = Utc::now();
        Self {
            project_id,
            domain,
            key_concepts: HashSet::new(),
            important_memories: Vec::new(),
            relationships: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn add_memory(&mut self, memory_id: String, concepts: Vec<String>) {
        self.important_memories.push(memory_id.clone());
        for concept in concepts {
            self.key_concepts.insert(concept.clone());
            self.relationships
                .entry(concept)
                .or_insert_with(Vec::new)
                .push(memory_id.clone());
        }
        self.updated_at = Utc::now();
    }
}

/// Enhanced Memory Archive Manager
pub struct MemoryArchiveManager {
    config: ArchiveConfig,
    redis_manager: Arc<RedisManager>,
    memory_manager: Arc<MemoryManager>,
    semantic_index: Arc<SemanticIndex>,
    importance_scorer: ImportanceScorer,

    // Tracking structures
    access_patterns: Arc<DashMap<String, AccessPattern>>,
    archive_levels: Arc<DashMap<String, ArchiveLevel>>,
    project_contexts: Arc<DashMap<String, ProjectContext>>,

    // Performance optimization
    compression_cache: Arc<DashMap<String, Vec<u8>>>,
    consolidation_semaphore: Arc<Semaphore>,

    // Background task handles
    consolidation_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    index_update_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl MemoryArchiveManager {
    pub async fn new(
        config: ArchiveConfig,
        redis_manager: Arc<RedisManager>,
        memory_manager: Arc<MemoryManager>,
        vector_store: Arc<RwLock<VectorStore>>,
    ) -> Result<Arc<Self>> {
        let semantic_index = Arc::new(SemanticIndex::new(vector_store).await);
        let importance_scorer = ImportanceScorer::new(&config);

        let manager = Arc::new(Self {
            config: config.clone(),
            redis_manager,
            memory_manager,
            semantic_index,
            importance_scorer,
            access_patterns: Arc::new(DashMap::new()),
            archive_levels: Arc::new(DashMap::new()),
            project_contexts: Arc::new(DashMap::new()),
            compression_cache: Arc::new(DashMap::new()),
            consolidation_semaphore: Arc::new(Semaphore::new(1)),
            consolidation_handle: Arc::new(RwLock::new(None)),
            index_update_handle: Arc::new(RwLock::new(None)),
        });

        if config.auto_consolidation {
            manager.start_background_tasks().await?;
        }

        Ok(manager)
    }

    /// Track memory access for pattern analysis
    pub async fn track_access(&self, memory_id: &str) -> Result<()> {
        let mut pattern = self.access_patterns
            .entry(memory_id.to_string())
            .or_insert_with(|| AccessPattern::new(memory_id.to_string()));

        pattern.record_access();

        // Update hot data status
        if pattern.total_accesses >= self.config.hot_data_threshold as u64 {
            pattern.is_hot = true;
            self.promote_to_active(memory_id).await?;
        }

        Ok(())
    }

    /// Consolidate memories based on access patterns
    pub async fn consolidate_memories(&self) -> Result<usize> {
        let _permit = self.consolidation_semaphore.acquire().await?;
        info!("Starting memory consolidation");

        let mut consolidated = 0;
        let mut batch = Vec::new();

        // Collect memories for consolidation
        for pattern_ref in self.access_patterns.iter() {
            let pattern = pattern_ref.value();

            if pattern.should_archive(self.config.hot_data_threshold) {
                batch.push(pattern.memory_id.clone());

                if batch.len() >= self.config.archive_batch_size {
                    consolidated += self.process_consolidation_batch(&batch).await?;
                    batch.clear();
                }
            }
        }

        // Process remaining batch
        if !batch.is_empty() {
            consolidated += self.process_consolidation_batch(&batch).await?;
        }

        info!("Consolidated {} memories", consolidated);
        Ok(consolidated)
    }

    async fn process_consolidation_batch(&self, memory_ids: &[String]) -> Result<usize> {
        let mut consolidated = 0;

        for memory_id in memory_ids {
            if let Ok(Some(entry)) = self.memory_manager.get(memory_id).await {
                let level = self.determine_archive_level(&entry, memory_id).await?;

                match level {
                    ArchiveLevel::Cool | ArchiveLevel::Cold => {
                        self.archive_memory(entry, level).await?;
                        consolidated += 1;
                    }
                    ArchiveLevel::Warm => {
                        self.compress_memory(&entry).await?;
                        consolidated += 1;
                    }
                    _ => {}
                }
            }
        }

        Ok(consolidated)
    }

    async fn determine_archive_level(&self, entry: &MemoryEntry, memory_id: &str) -> Result<ArchiveLevel> {
        let pattern = self.access_patterns
            .get(memory_id)
            .map(|p| p.clone())
            .unwrap_or_else(|| AccessPattern::new(memory_id.to_string()));

        let importance = self.calculate_importance(entry, &pattern).await?;

        if importance >= 0.7 || pattern.is_hot {
            Ok(ArchiveLevel::Active)
        } else if importance >= 0.4 {
            Ok(ArchiveLevel::Warm)
        } else if importance >= self.config.min_archive_importance {
            Ok(ArchiveLevel::Cool)
        } else {
            Ok(ArchiveLevel::Cold)
        }
    }

    async fn calculate_importance(&self, entry: &MemoryEntry, pattern: &AccessPattern) -> Result<f64> {
        // Get semantic relevance from current context
        let semantic_relevance = self.calculate_semantic_relevance(entry).await?;

        // Get project context relevance
        let context_relevance = self.calculate_context_relevance(entry).await?;

        Ok(self.importance_scorer.calculate_score(
            entry,
            pattern,
            semantic_relevance,
            context_relevance,
        ))
    }

    async fn calculate_semantic_relevance(&self, entry: &MemoryEntry) -> Result<f64> {
        // Simplified semantic relevance - can be enhanced with embeddings
        if entry.tags.contains("important") || entry.tags.contains("core") {
            Ok(0.8)
        } else if entry.tags.contains("reference") {
            Ok(0.5)
        } else {
            Ok(0.3)
        }
    }

    async fn calculate_context_relevance(&self, entry: &MemoryEntry) -> Result<f64> {
        // Check if memory is part of any active project context
        for context_ref in self.project_contexts.iter() {
            let context = context_ref.value();
            if context.important_memories.contains(&entry.id) {
                return Ok(0.9);
            }
        }
        Ok(0.2)
    }

    /// Archive memory to appropriate storage tier
    async fn archive_memory(&self, mut entry: MemoryEntry, level: ArchiveLevel) -> Result<()> {
        debug!("Archiving memory {} to level {:?}", entry.id, level);

        // Compress content based on level
        if level >= ArchiveLevel::Cool {
            entry.content = self.compress_content(&entry.content, level).await?;
            entry.compressed = true;
        }

        // Update metadata
        entry.metadata.insert(
            "archive_level".to_string(),
            serde_json::Value::String(format!("{:?}", level)),
        );
        entry.metadata.insert(
            "archived_at".to_string(),
            serde_json::Value::String(Utc::now().to_rfc3339()),
        );

        // Store in appropriate tier
        let archive_key = format!("archive:{}:{}", level as u8, entry.id);
        let serialized = bincode::serialize(&entry)?;

        self.redis_manager.set_raw(&archive_key, &serialized).await?;

        // Update tracking
        self.archive_levels.insert(entry.id.clone(), level);

        // Remove from active memory if cold
        if level == ArchiveLevel::Cold {
            self.memory_manager.delete(&entry.id).await?;
        }

        Ok(())
    }

    async fn compress_content(&self, content: &MemoryContent, level: ArchiveLevel) -> Result<MemoryContent> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let compression_level = match level {
            ArchiveLevel::Cool => Compression::fast(),
            ArchiveLevel::Cold => Compression::best(),
            _ => Compression::default(),
        };

        match content {
            MemoryContent::Text(text) => {
                let mut encoder = GzEncoder::new(Vec::new(), compression_level);
                encoder.write_all(text.as_bytes())?;
                let compressed = encoder.finish()?;
                Ok(MemoryContent::Compressed(compressed))
            }
            MemoryContent::Structured(value) => {
                let json = serde_json::to_string(value)?;
                let mut encoder = GzEncoder::new(Vec::new(), compression_level);
                encoder.write_all(json.as_bytes())?;
                let compressed = encoder.finish()?;
                Ok(MemoryContent::Compressed(compressed))
            }
            _ => Ok(content.clone()),
        }
    }

    async fn compress_memory(&self, entry: &MemoryEntry) -> Result<()> {
        if entry.content.size_bytes() > self.config.compression_threshold {
            let compressed = self.compress_content(&entry.content, ArchiveLevel::Warm).await?;

            // Cache compressed version
            if let MemoryContent::Compressed(data) = &compressed {
                self.compression_cache.insert(entry.id.clone(), data.clone());
            }

            // Update in memory manager
            self.memory_manager.update(&entry.id, |e| {
                e.content = compressed.clone();
                e.compressed = true;
            }).await?;
        }

        Ok(())
    }

    async fn promote_to_active(&self, memory_id: &str) -> Result<()> {
        if let Some(level) = self.archive_levels.get(memory_id) {
            if *level != ArchiveLevel::Active {
                // Retrieve from archive
                let archive_key = format!("archive:{}:{}", *level as u8, memory_id);

                if let Some(data) = self.redis_manager.get_raw(&archive_key).await? {
                    let mut entry: MemoryEntry = bincode::deserialize(&data)?;

                    // Decompress if needed
                    if entry.compressed {
                        entry.content = self.decompress_content(&entry.content).await?;
                        entry.compressed = false;
                    }

                    // Restore to active memory
                    self.memory_manager.store_entry(entry).await?;

                    // Update level
                    self.archive_levels.insert(memory_id.to_string(), ArchiveLevel::Active);

                    // Remove from archive
                    self.redis_manager.delete_raw(&archive_key).await?;
                }
            }
        }

        Ok(())
    }

    async fn decompress_content(&self, content: &MemoryContent) -> Result<MemoryContent> {
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

    /// Retrieve memory with intelligent caching and decompression
    pub async fn retrieve_memory(&self, memory_id: &str) -> Result<Option<MemoryEntry>> {
        // Track access
        self.track_access(memory_id).await?;

        // Check active memory first
        if let Ok(Some(entry)) = self.memory_manager.get(memory_id).await {
            return Ok(Some(entry));
        }

        // Check archive levels
        if let Some(level) = self.archive_levels.get(memory_id) {
            let archive_key = format!("archive:{}:{}", *level as u8, memory_id);

            if let Some(data) = self.redis_manager.get_raw(&archive_key).await? {
                let mut entry: MemoryEntry = bincode::deserialize(&data)?;

                // Decompress if needed
                if entry.compressed {
                    // Check compression cache first
                    if let Some(cached) = self.compression_cache.get(memory_id) {
                        entry.content = MemoryContent::Compressed(cached.clone());
                    }
                    entry.content = self.decompress_content(&entry.content).await?;
                }

                return Ok(Some(entry));
            }
        }

        Ok(None)
    }

    /// Store project context
    pub async fn store_project_context(&self, context: ProjectContext) -> Result<()> {
        let project_id = context.project_id.clone();
        self.project_contexts.insert(project_id.clone(), context.clone());

        // Persist to Redis
        let key = format!("context:project:{}", project_id);
        let serialized = serde_json::to_vec(&context)?;

        let ttl = std::time::Duration::from_secs(
            self.config.project_context_retention_days as u64 * 86400
        );

        self.redis_manager.set_raw_ex(&key, &serialized, ttl.as_secs()).await?;

        Ok(())
    }

    /// Retrieve project context
    pub async fn get_project_context(&self, project_id: &str) -> Result<Option<ProjectContext>> {
        // Check cache first
        if let Some(context) = self.project_contexts.get(project_id) {
            return Ok(Some(context.clone()));
        }

        // Load from Redis
        let key = format!("context:project:{}", project_id);
        if let Some(data) = self.redis_manager.get_raw(&key).await? {
            let context: ProjectContext = serde_json::from_slice(&data)?;
            self.project_contexts.insert(project_id.to_string(), context.clone());
            return Ok(Some(context));
        }

        Ok(None)
    }

    /// Update semantic index
    pub async fn update_semantic_index(&self) -> Result<()> {
        info!("Updating semantic index");

        // Index all active memories
        let filter = crate::memory::MemoryFilter {
            memory_types: Some([
                MemoryType::Working,
                MemoryType::ShortTerm,
                MemoryType::LongTerm,
                MemoryType::Semantic,
            ].iter().cloned().collect()),
            ..Default::default()
        };

        let results = self.memory_manager.search(&filter).await?;

        for result in results {
            self.semantic_index.index_memory(&result.entry).await?;
        }

        info!("Semantic index updated");
        Ok(())
    }

    async fn start_background_tasks(&self) -> Result<()> {
        // Consolidation task
        {
            let manager = Arc::downgrade(&Arc::new(self.clone()));
            let interval_secs = self.config.consolidation_interval;

            let handle = tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    std::time::Duration::from_secs(interval_secs)
                );

                loop {
                    interval.tick().await;

                    if let Some(mgr) = manager.upgrade() {
                        if let Err(e) = mgr.consolidate_memories().await {
                            warn!("Consolidation error: {}", e);
                        }
                    } else {
                        break;
                    }
                }
            });

            *self.consolidation_handle.write() = Some(handle);
        }

        // Semantic index update task
        {
            let manager = Arc::downgrade(&Arc::new(self.clone()));
            let interval_secs = self.config.semantic_index_interval;

            let handle = tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    std::time::Duration::from_secs(interval_secs)
                );

                loop {
                    interval.tick().await;

                    if let Some(mgr) = manager.upgrade() {
                        if let Err(e) = mgr.update_semantic_index().await {
                            warn!("Index update error: {}", e);
                        }
                    } else {
                        break;
                    }
                }
            });

            *self.index_update_handle.write() = Some(handle);
        }

        Ok(())
    }
}

// Implement Clone manually to handle the Arc fields
impl Clone for MemoryArchiveManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            redis_manager: self.redis_manager.clone(),
            memory_manager: self.memory_manager.clone(),
            semantic_index: self.semantic_index.clone(),
            importance_scorer: self.importance_scorer.clone(),
            access_patterns: self.access_patterns.clone(),
            archive_levels: self.archive_levels.clone(),
            project_contexts: self.project_contexts.clone(),
            compression_cache: self.compression_cache.clone(),
            consolidation_semaphore: self.consolidation_semaphore.clone(),
            consolidation_handle: Arc::new(RwLock::new(None)),
            index_update_handle: Arc::new(RwLock::new(None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_tracking() {
        let mut pattern = AccessPattern::new("test_memory".to_string());

        pattern.record_access();
        assert_eq!(pattern.total_accesses, 1);

        pattern.record_access();
        assert_eq!(pattern.total_accesses, 2);

        assert!(!pattern.is_hot);
    }

    #[test]
    fn test_importance_decay() {
        let config = ArchiveConfig::default();
        let scorer = ImportanceScorer::new(&config);

        let entry = MemoryEntry::new(
            MemoryContent::Text("Test".to_string()),
            MemoryType::ShortTerm,
            0.8,
            None,
        );

        let pattern = AccessPattern::new("test".to_string());

        let score = scorer.calculate_score(&entry, &pattern, 0.5, 0.5);
        assert!(score > 0.0 && score <= 1.0);
    }
}