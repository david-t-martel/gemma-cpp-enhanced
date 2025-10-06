//! # Project Context Storage and Retrieval System
//!
//! This module provides comprehensive project state management for RAG-Redis, including:
//! - Complete project state storage (files, configurations, memories, metadata)
//! - Versioning and snapshots for context tracking
//! - Quick save/load mechanisms for LLM sessions
//! - Project-specific memory isolation
//! - Context compression and deduplication
//! - Context diff system for change tracking
//!
//! ## Features
//!
//! - **Full State Capture**: Files, configurations, memories, vector embeddings, and metadata
//! - **Versioning**: Snapshot-based versioning with semantic version support
//! - **Memory Isolation**: Project-specific memory spaces with cross-project sharing
//! - **Compression**: Automatic compression and deduplication for large contexts
//! - **Diff Tracking**: Git-like diff system for tracking changes between contexts
//! - **Quick Save/Load**: Fast session persistence for LLM continuity
//! - **Metadata Rich**: Extensive metadata tracking for project analytics
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use rag_redis_system::project_context::{ProjectContextManager, ProjectSnapshot, SaveOptions};
//! use rag_redis_system::redis_backend::RedisManager;
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let redis_manager = Arc::new(RedisManager::new(&Default::default()).await?);
//! let context_manager = ProjectContextManager::new(redis_manager).await?;
//!
//! // Save complete project state
//! let project_root = PathBuf::from("C:/myproject");
//! let save_options = SaveOptions {
//!     include_files: true,
//!     include_memories: true,
//!     include_vectors: true,
//!     compress: true,
//!     deduplicate: true,
//!     ..Default::default()
//! };
//!
//! let snapshot_id = context_manager
//!     .save_project_context("my-project", &project_root, save_options)
//!     .await?;
//!
//! // Load project state
//! let loaded_context = context_manager
//!     .load_project_context("my-project", Some(&snapshot_id))
//!     .await?;
//!
//! // Quick session save/load
//! let session_id = context_manager
//!     .quick_save_session("my-project", "working on feature X")
//!     .await?;
//!
//! context_manager
//!     .quick_load_session("my-project", &session_id)
//!     .await?;
//!
//! // Track changes
//! let diff = context_manager
//!     .diff_contexts("my-project", "v1.0.0", "v1.1.0")
//!     .await?;
//!
//! println!("Changes: {} files modified, {} memories added",
//!          diff.file_changes.len(), diff.memory_changes.added.len());
//! # Ok(())
//! # }
//! ```

use crate::error::{Error, Result};
use crate::memory::{MemoryEntry, MemoryManager, MemoryType};
use crate::redis_backend::RedisManager;
use chrono::{DateTime, Utc};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use uuid::Uuid;

/// Project context manager for comprehensive state management
pub struct ProjectContextManager {
    redis_manager: Arc<RedisManager>,
    memory_manager: Option<Arc<MemoryManager>>,
    compression_threshold: usize,
    max_snapshot_age: chrono::Duration,
}

/// Complete project snapshot with all state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSnapshot {
    pub id: String,
    pub project_id: String,
    pub version: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub parent_snapshot: Option<String>,
    pub tags: HashSet<String>,

    // Core project data
    pub project_files: ProjectFiles,
    pub project_configuration: ProjectConfiguration,
    pub project_memories: ProjectMemories,
    pub project_vectors: ProjectVectors,
    pub project_metadata: ProjectMetadata,

    // Snapshot metadata
    pub snapshot_metadata: SnapshotMetadata,
}

/// Project files with content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFiles {
    pub root_path: PathBuf,
    pub files: BTreeMap<PathBuf, FileEntry>,
    pub total_size: u64,
    pub file_count: usize,
    pub last_modified: DateTime<Utc>,
}

/// Individual file entry with content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub content: FileContent,
    pub metadata: FileMetadata,
    pub hash: String,
    pub size: u64,
    pub modified: DateTime<Utc>,
    pub created: DateTime<Utc>,
    pub permissions: Option<String>,
}

/// File content storage (supports compression and references)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileContent {
    /// Direct text content
    Text(String),
    /// Compressed content
    Compressed(Vec<u8>),
    /// Reference to external storage (for large files)
    Reference(String),
    /// Binary content
    Binary(Vec<u8>),
    /// Deduplicated content (hash reference)
    Deduplicated(String),
}

impl FileContent {
    pub fn size_bytes(&self) -> usize {
        match self {
            FileContent::Text(s) => s.len(),
            FileContent::Binary(b) => b.len(),
            FileContent::Compressed(b) => b.len(),
            FileContent::Reference(s) => s.len(),
            FileContent::Deduplicated(s) => s.len(), // Just the hash size
        }
    }
}

/// File metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub file_type: String,
    pub encoding: Option<String>,
    pub language: Option<String>,
    pub line_count: Option<usize>,
    pub is_generated: bool,
    pub is_config: bool,
    pub is_test: bool,
    pub tags: HashSet<String>,
    pub custom: HashMap<String, serde_json::Value>,
}

/// Project configuration snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfiguration {
    pub config_files: BTreeMap<String, serde_json::Value>,
    pub environment_variables: HashMap<String, String>,
    pub build_configuration: Option<serde_json::Value>,
    pub dependencies: Option<serde_json::Value>,
    pub rag_configuration: Option<serde_json::Value>,
    pub custom_configs: HashMap<String, serde_json::Value>,
}

/// Project memories with isolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMemories {
    pub project_memory_space: String,
    pub memories_by_type: HashMap<String, Vec<MemoryEntry>>,
    pub memory_statistics: MemoryStatistics,
    pub cross_project_references: Vec<String>,
    pub memory_tags: HashSet<String>,
}

/// Project vectors and embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectVectors {
    pub document_embeddings: HashMap<String, DocumentEmbedding>,
    pub code_embeddings: HashMap<String, CodeEmbedding>,
    pub vector_index_metadata: VectorIndexMetadata,
    pub embedding_model_info: EmbeddingModelInfo,
}

/// Project metadata and analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub project_name: String,
    pub project_type: Option<String>,
    pub programming_languages: HashSet<String>,
    pub frameworks: HashSet<String>,
    pub complexity_metrics: ComplexityMetrics,
    pub activity_metrics: ActivityMetrics,
    pub session_history: Vec<SessionRecord>,
    pub custom_metadata: HashMap<String, serde_json::Value>,
}

/// Snapshot-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub compression_ratio: f64,
    pub deduplication_savings: u64,
    pub storage_size: u64,
    pub creation_duration: std::time::Duration,
    pub validation_hash: String,
    pub storage_backend: String,
    pub snapshot_type: SnapshotType,
    pub quality_score: f64,
}

/// Types of snapshots
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SnapshotType {
    /// Full project snapshot
    Full,
    /// Incremental changes only
    Incremental,
    /// Quick session save
    Session,
    /// Milestone/release snapshot
    Milestone,
    /// Backup snapshot
    Backup,
}

/// Document embedding with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEmbedding {
    pub document_id: String,
    pub embedding: Vec<f32>,
    pub chunk_embeddings: Vec<ChunkEmbedding>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Code-specific embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEmbedding {
    pub file_path: PathBuf,
    pub function_embeddings: HashMap<String, Vec<f32>>,
    pub class_embeddings: HashMap<String, Vec<f32>>,
    pub semantic_embeddings: Vec<f32>,
    pub syntax_tree_hash: String,
}

/// Chunk embedding for documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEmbedding {
    pub chunk_id: String,
    pub embedding: Vec<f32>,
    pub start_offset: usize,
    pub end_offset: usize,
    pub content_hash: String,
}

/// Vector index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexMetadata {
    pub index_type: String,
    pub dimension: usize,
    pub total_vectors: usize,
    pub index_parameters: HashMap<String, serde_json::Value>,
    pub performance_metrics: IndexPerformanceMetrics,
}

/// Embedding model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    pub model_name: String,
    pub model_version: String,
    pub dimension: usize,
    pub max_sequence_length: usize,
    pub normalization: bool,
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    pub total_memories: usize,
    pub memories_by_type: HashMap<String, usize>,
    pub average_importance: f64,
    pub memory_size_bytes: u64,
    pub oldest_memory: Option<DateTime<Utc>>,
    pub newest_memory: Option<DateTime<Utc>>,
}

/// Project complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: Option<u32>,
    pub lines_of_code: u32,
    pub file_count: u32,
    pub dependency_count: u32,
    pub test_coverage: Option<f64>,
    pub code_duplication: Option<f64>,
    pub technical_debt_ratio: Option<f64>,
}

/// Project activity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityMetrics {
    pub total_sessions: u32,
    pub total_session_time: std::time::Duration,
    pub files_modified: u32,
    pub memories_created: u32,
    pub searches_performed: u32,
    pub last_activity: DateTime<Utc>,
    pub activity_heatmap: HashMap<String, u32>, // Date -> activity count
}

/// Session record for tracking LLM sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub session_type: SessionType,
    pub description: Option<String>,
    pub files_touched: Vec<PathBuf>,
    pub memories_created: Vec<String>,
    pub context_switches: u32,
    pub performance_metrics: SessionPerformanceMetrics,
}

/// Types of LLM sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionType {
    Development,
    Research,
    Documentation,
    Debugging,
    Refactoring,
    Testing,
    Planning,
    Review,
}

/// Session performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPerformanceMetrics {
    pub response_times: Vec<std::time::Duration>,
    pub memory_usage: Vec<u64>,
    pub context_load_times: Vec<std::time::Duration>,
    pub search_performance: Vec<SearchPerformance>,
}

/// Search performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformance {
    pub query_time: std::time::Duration,
    pub results_count: usize,
    pub relevance_score: f64,
    pub cache_hit: bool,
}

/// Index performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPerformanceMetrics {
    pub average_query_time: std::time::Duration,
    pub index_size_bytes: u64,
    pub rebuild_frequency: std::time::Duration,
    pub cache_hit_rate: f64,
}

/// Options for saving project context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveOptions {
    pub include_files: bool,
    pub include_memories: bool,
    pub include_vectors: bool,
    pub include_config: bool,
    pub compress: bool,
    pub deduplicate: bool,
    pub file_patterns: Option<Vec<String>>,
    pub exclude_patterns: Option<Vec<String>>,
    pub max_file_size: Option<u64>,
    pub snapshot_type: SnapshotType,
    pub description: Option<String>,
    pub tags: Vec<String>,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            include_files: true,
            include_memories: true,
            include_vectors: true,
            include_config: true,
            compress: true,
            deduplicate: true,
            file_patterns: None,
            exclude_patterns: Some(vec![
                ".git/**".to_string(),
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "*.tmp".to_string(),
                "*.log".to_string(),
            ]),
            max_file_size: Some(10 * 1024 * 1024), // 10MB
            snapshot_type: SnapshotType::Full,
            description: None,
            tags: Vec::new(),
        }
    }
}

/// Options for loading project context
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub load_files: bool,
    pub load_memories: bool,
    pub load_vectors: bool,
    pub load_config: bool,
    pub target_directory: Option<PathBuf>,
    pub overwrite_existing: bool,
    pub create_backup: bool,
    pub verify_integrity: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            load_files: true,
            load_memories: true,
            load_vectors: true,
            load_config: true,
            target_directory: None,
            overwrite_existing: false,
            create_backup: true,
            verify_integrity: true,
        }
    }
}

/// Context diff for tracking changes between snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDiff {
    pub from_snapshot: String,
    pub to_snapshot: String,
    pub generated_at: DateTime<Utc>,

    pub file_changes: FileChanges,
    pub memory_changes: MemoryChanges,
    pub config_changes: ConfigChanges,
    pub vector_changes: VectorChanges,
    pub metadata_changes: MetadataChanges,

    pub summary: DiffSummary,
}

/// File changes in diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChanges {
    pub added: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    pub modified: Vec<FileModification>,
    pub renamed: Vec<FileRename>,
}

/// Memory changes in diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChanges {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<MemoryModification>,
    pub type_changes: HashMap<String, (String, String)>, // ID -> (old_type, new_type)
}

/// Configuration changes in diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChanges {
    pub added_configs: HashMap<String, serde_json::Value>,
    pub removed_configs: Vec<String>,
    pub modified_configs: HashMap<String, ConfigModification>,
    pub environment_changes: HashMap<String, EnvironmentChange>,
}

/// Vector changes in diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorChanges {
    pub added_documents: Vec<String>,
    pub removed_documents: Vec<String>,
    pub updated_embeddings: Vec<String>,
    pub index_changes: Vec<IndexChange>,
}

/// Metadata changes in diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataChanges {
    pub complexity_delta: ComplexityDelta,
    pub activity_delta: ActivityDelta,
    pub session_changes: Vec<SessionRecord>,
    pub custom_metadata_changes: HashMap<String, MetadataChange>,
}

/// File modification details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileModification {
    pub path: PathBuf,
    pub changes: Vec<LineChange>,
    pub size_delta: i64,
    pub hash_before: String,
    pub hash_after: String,
}

/// File rename details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRename {
    pub old_path: PathBuf,
    pub new_path: PathBuf,
    pub similarity_score: f64,
}

/// Line-level changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineChange {
    pub line_number: usize,
    pub change_type: ChangeType,
    pub old_content: Option<String>,
    pub new_content: Option<String>,
}

/// Type of change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Added,
    Removed,
    Modified,
}

/// Memory modification details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryModification {
    pub memory_id: String,
    pub field_changes: HashMap<String, FieldChange>,
    pub importance_delta: f64,
    pub content_diff: Option<String>,
}

/// Field change details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldChange {
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
}

/// Configuration modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigModification {
    pub config_key: String,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
    pub change_impact: ChangeImpact,
}

/// Environment variable change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentChange {
    pub variable: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub change_type: ChangeType,
}

/// Index change details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexChange {
    pub index_type: String,
    pub change_description: String,
    pub performance_impact: f64,
}

/// Complexity metrics delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityDelta {
    pub cyclomatic_complexity_delta: Option<i32>,
    pub lines_of_code_delta: i32,
    pub file_count_delta: i32,
    pub dependency_count_delta: i32,
    pub test_coverage_delta: Option<f64>,
}

/// Activity metrics delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityDelta {
    pub session_count_delta: i32,
    pub session_time_delta: std::time::Duration,
    pub files_modified_delta: i32,
    pub memories_created_delta: i32,
}

/// Metadata change details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataChange {
    pub key: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
    pub change_type: ChangeType,
}

/// Change impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeImpact {
    Low,
    Medium,
    High,
    Critical,
}

/// Diff summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub total_changes: usize,
    pub files_affected: usize,
    pub memories_affected: usize,
    pub configs_affected: usize,
    pub overall_impact: ChangeImpact,
    pub change_categories: HashMap<String, usize>,
    pub estimated_review_time: std::time::Duration,
}

impl ProjectContextManager {
    /// Create a new project context manager
    pub async fn new(redis_manager: Arc<RedisManager>) -> Result<Self> {
        Ok(Self {
            redis_manager,
            memory_manager: None,
            compression_threshold: 1024, // 1KB
            max_snapshot_age: chrono::Duration::days(90),
        })
    }

    /// Set the memory manager for memory operations
    pub fn with_memory_manager(mut self, memory_manager: Arc<MemoryManager>) -> Self {
        self.memory_manager = Some(memory_manager);
        self
    }

    /// Save complete project context with all specified components
    pub async fn save_project_context(
        &self,
        project_id: &str,
        project_root: &Path,
        options: SaveOptions,
    ) -> Result<String> {
        let snapshot_id = Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();

        tracing::info!("Starting project context save for project: {}", project_id);

        let mut snapshot = ProjectSnapshot {
            id: snapshot_id.clone(),
            project_id: project_id.to_string(),
            version: self.generate_version(project_id).await?,
            name: None,
            description: options.description.clone(),
            created_at: Utc::now(),
            created_by: std::env::var("USER").ok().or_else(|| std::env::var("USERNAME").ok()),
            parent_snapshot: self.get_latest_snapshot_id(project_id).await.ok(),
            tags: options.tags.iter().cloned().collect(),
            project_files: ProjectFiles {
                root_path: project_root.to_path_buf(),
                files: BTreeMap::new(),
                total_size: 0,
                file_count: 0,
                last_modified: Utc::now(),
            },
            project_configuration: ProjectConfiguration {
                config_files: BTreeMap::new(),
                environment_variables: HashMap::new(),
                build_configuration: None,
                dependencies: None,
                rag_configuration: None,
                custom_configs: HashMap::new(),
            },
            project_memories: ProjectMemories {
                project_memory_space: format!("project:{}", project_id),
                memories_by_type: HashMap::new(),
                memory_statistics: MemoryStatistics {
                    total_memories: 0,
                    memories_by_type: HashMap::new(),
                    average_importance: 0.0,
                    memory_size_bytes: 0,
                    oldest_memory: None,
                    newest_memory: None,
                },
                cross_project_references: Vec::new(),
                memory_tags: HashSet::new(),
            },
            project_vectors: ProjectVectors {
                document_embeddings: HashMap::new(),
                code_embeddings: HashMap::new(),
                vector_index_metadata: VectorIndexMetadata {
                    index_type: "hnsw".to_string(),
                    dimension: 384,
                    total_vectors: 0,
                    index_parameters: HashMap::new(),
                    performance_metrics: IndexPerformanceMetrics {
                        average_query_time: std::time::Duration::from_millis(1),
                        index_size_bytes: 0,
                        rebuild_frequency: std::time::Duration::from_secs(24 * 60 * 60),
                        cache_hit_rate: 0.0,
                    },
                },
                embedding_model_info: EmbeddingModelInfo {
                    model_name: "all-MiniLM-L6-v2".to_string(),
                    model_version: "1.0".to_string(),
                    dimension: 384,
                    max_sequence_length: 512,
                    normalization: true,
                },
            },
            project_metadata: ProjectMetadata {
                project_name: project_id.to_string(),
                project_type: None,
                programming_languages: HashSet::new(),
                frameworks: HashSet::new(),
                complexity_metrics: ComplexityMetrics {
                    cyclomatic_complexity: None,
                    lines_of_code: 0,
                    file_count: 0,
                    dependency_count: 0,
                    test_coverage: None,
                    code_duplication: None,
                    technical_debt_ratio: None,
                },
                activity_metrics: ActivityMetrics {
                    total_sessions: 0,
                    total_session_time: std::time::Duration::from_secs(0),
                    files_modified: 0,
                    memories_created: 0,
                    searches_performed: 0,
                    last_activity: Utc::now(),
                    activity_heatmap: HashMap::new(),
                },
                session_history: Vec::new(),
                custom_metadata: HashMap::new(),
            },
            snapshot_metadata: SnapshotMetadata {
                compression_ratio: 1.0,
                deduplication_savings: 0,
                storage_size: 0,
                creation_duration: std::time::Duration::from_secs(0),
                validation_hash: String::new(),
                storage_backend: "redis".to_string(),
                snapshot_type: options.snapshot_type.clone(),
                quality_score: 1.0,
            },
        };

        // Collect project files if enabled
        if options.include_files {
            snapshot.project_files = self.collect_project_files(
                project_root,
                &options.file_patterns,
                &options.exclude_patterns,
                options.max_file_size,
            ).await?;
        }

        // Collect project configuration if enabled
        if options.include_config {
            snapshot.project_configuration = self.collect_project_configuration(project_root).await?;
        }

        // Collect project memories if enabled
        if options.include_memories {
            snapshot.project_memories = self.collect_project_memories(project_id).await?;
        }

        // Collect project vectors if enabled
        if options.include_vectors {
            snapshot.project_vectors = self.collect_project_vectors(project_id).await?;
        }

        // Collect project metadata
        snapshot.project_metadata = self.collect_project_metadata(project_root, project_id).await?;

        // Apply compression and deduplication if enabled
        if options.compress || options.deduplicate {
            snapshot = self.optimize_snapshot(snapshot, options.compress, options.deduplicate).await?;
        }

        // Generate validation hash
        snapshot.snapshot_metadata.validation_hash = self.generate_validation_hash(&snapshot)?;
        snapshot.snapshot_metadata.creation_duration = start_time.elapsed();

        // Store the snapshot
        self.store_snapshot(&snapshot).await?;

        // Update project index
        self.update_project_index(project_id, &snapshot).await?;

        tracing::info!("Project context saved: {} ({})", snapshot_id, snapshot.version);
        Ok(snapshot_id)
    }

    /// Load complete project context
    pub async fn load_project_context(
        &self,
        project_id: &str,
        snapshot_id: Option<&str>,
    ) -> Result<ProjectSnapshot> {
        let snapshot_id = if let Some(id) = snapshot_id {
            id.to_string()
        } else {
            self.get_latest_snapshot_id(project_id).await?
        };

        tracing::info!("Loading project context: {} ({})", project_id, snapshot_id);

        let snapshot = self.load_snapshot(&snapshot_id).await?;

        // Verify integrity
        let computed_hash = self.generate_validation_hash(&snapshot)?;
        if computed_hash != snapshot.snapshot_metadata.validation_hash {
            return Err(Error::Api("Snapshot integrity check failed".to_string()));
        }

        tracing::info!("Project context loaded successfully");
        Ok(snapshot)
    }

    /// Quick save current session state
    pub async fn quick_save_session(
        &self,
        project_id: &str,
        description: &str,
    ) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();

        // Get current working directory as project root
        let current_dir = std::env::current_dir()
            .map_err(|e| Error::Api(format!("Failed to get current directory: {}", e)))?;

        let options = SaveOptions {
            snapshot_type: SnapshotType::Session,
            description: Some(description.to_string()),
            include_vectors: false, // Skip vectors for quick saves
            compress: true,
            ..Default::default()
        };

        self.save_project_context(project_id, &current_dir, options).await
    }

    /// Quick load session state
    pub async fn quick_load_session(
        &self,
        project_id: &str,
        session_id: &str,
    ) -> Result<ProjectSnapshot> {
        self.load_project_context(project_id, Some(session_id)).await
    }

    /// Generate diff between two project contexts
    pub async fn diff_contexts(
        &self,
        project_id: &str,
        from_version: &str,
        to_version: &str,
    ) -> Result<ContextDiff> {
        let from_snapshot = self.get_snapshot_by_version(project_id, from_version).await?;
        let to_snapshot = self.get_snapshot_by_version(project_id, to_version).await?;

        tracing::info!("Generating diff: {} -> {}", from_version, to_version);

        let file_changes = self.diff_files(&from_snapshot.project_files, &to_snapshot.project_files)?;
        let memory_changes = self.diff_memories(&from_snapshot.project_memories, &to_snapshot.project_memories)?;
        let config_changes = self.diff_configs(&from_snapshot.project_configuration, &to_snapshot.project_configuration)?;
        let vector_changes = self.diff_vectors(&from_snapshot.project_vectors, &to_snapshot.project_vectors)?;
        let metadata_changes = self.diff_metadata(&from_snapshot.project_metadata, &to_snapshot.project_metadata)?;

        let total_changes = file_changes.added.len() + file_changes.removed.len() + file_changes.modified.len()
            + memory_changes.added.len() + memory_changes.removed.len() + memory_changes.modified.len()
            + config_changes.added_configs.len() + config_changes.removed_configs.len() + config_changes.modified_configs.len();

        let overall_impact = if total_changes > 100 {
            ChangeImpact::Critical
        } else if total_changes > 50 {
            ChangeImpact::High
        } else if total_changes > 10 {
            ChangeImpact::Medium
        } else {
            ChangeImpact::Low
        };

        let summary = DiffSummary {
            total_changes,
            files_affected: file_changes.added.len() + file_changes.removed.len() + file_changes.modified.len(),
            memories_affected: memory_changes.added.len() + memory_changes.removed.len() + memory_changes.modified.len(),
            configs_affected: config_changes.added_configs.len() + config_changes.removed_configs.len() + config_changes.modified_configs.len(),
            overall_impact,
            change_categories: HashMap::new(),
            estimated_review_time: std::time::Duration::from_secs((total_changes * 30) as u64), // 30 seconds per change
        };

        Ok(ContextDiff {
            from_snapshot: from_snapshot.id,
            to_snapshot: to_snapshot.id,
            generated_at: Utc::now(),
            file_changes,
            memory_changes,
            config_changes,
            vector_changes,
            metadata_changes,
            summary,
        })
    }

    /// List all snapshots for a project
    pub async fn list_project_snapshots(
        &self,
        project_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ProjectSnapshot>> {
        let key = format!("project_context:{}:snapshots", project_id);
        let snapshot_ids = self.redis_manager.redis_zrevrange(&key, 0, limit.unwrap_or(50) as isize - 1).await?;

        let mut snapshots = Vec::new();
        for snapshot_id in snapshot_ids {
            if let Ok(snapshot) = self.load_snapshot(&snapshot_id).await {
                snapshots.push(snapshot);
            }
        }

        Ok(snapshots)
    }

    /// Delete old snapshots based on retention policy
    pub async fn cleanup_old_snapshots(&self, project_id: &str) -> Result<usize> {
        let cutoff_time = Utc::now() - self.max_snapshot_age;
        let snapshots = self.list_project_snapshots(project_id, None).await?;

        let mut deleted_count = 0;
        for snapshot in snapshots {
            if snapshot.created_at < cutoff_time && snapshot.snapshot_metadata.snapshot_type != SnapshotType::Milestone {
                self.delete_snapshot(&snapshot.id).await?;
                deleted_count += 1;
            }
        }

        tracing::info!("Cleaned up {} old snapshots for project {}", deleted_count, project_id);
        Ok(deleted_count)
    }

    /// Get project statistics across all snapshots
    pub async fn get_project_statistics(&self, project_id: &str) -> Result<ProjectStatistics> {
        let snapshots = self.list_project_snapshots(project_id, None).await?;

        let mut stats = ProjectStatistics {
            project_id: project_id.to_string(),
            total_snapshots: snapshots.len(),
            total_storage_size: 0,
            oldest_snapshot: None,
            newest_snapshot: None,
            snapshot_types: HashMap::new(),
            storage_trend: Vec::new(),
            complexity_trend: Vec::new(),
            activity_trend: Vec::new(),
        };

        for snapshot in &snapshots {
            stats.total_storage_size += snapshot.snapshot_metadata.storage_size;

            let snapshot_type_str = format!("{:?}", snapshot.snapshot_metadata.snapshot_type);
            *stats.snapshot_types.entry(snapshot_type_str).or_insert(0) += 1;

            if stats.oldest_snapshot.is_none() || snapshot.created_at < stats.oldest_snapshot.unwrap() {
                stats.oldest_snapshot = Some(snapshot.created_at);
            }

            if stats.newest_snapshot.is_none() || snapshot.created_at > stats.newest_snapshot.unwrap() {
                stats.newest_snapshot = Some(snapshot.created_at);
            }
        }

        Ok(stats)
    }

    // Private implementation methods

    async fn collect_project_files(
        &self,
        project_root: &Path,
        include_patterns: &Option<Vec<String>>,
        exclude_patterns: &Option<Vec<String>>,
        max_file_size: Option<u64>,
    ) -> Result<ProjectFiles> {
        let mut files = BTreeMap::new();
        let mut total_size = 0u64;
        let mut file_count = 0usize;
        let mut last_modified = DateTime::<Utc>::MIN_UTC;

        let walker = walkdir::WalkDir::new(project_root)
            .follow_links(false)
            .max_depth(10);

        for entry in walker {
            let entry = entry.map_err(|e| Error::Api(format!("Failed to walk directory: {}", e)))?;

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let relative_path = path.strip_prefix(project_root)
                .map_err(|e| Error::Api(format!("Failed to get relative path: {}", e)))?;

            // Apply filters
            if let Some(exclude) = exclude_patterns {
                if self.matches_patterns(relative_path, exclude) {
                    continue;
                }
            }

            if let Some(include) = include_patterns {
                if !self.matches_patterns(relative_path, include) {
                    continue;
                }
            }

            let metadata = entry.metadata()
                .map_err(|e| Error::Api(format!("Failed to get file metadata: {}", e)))?;

            let file_size = metadata.len();
            if let Some(max_size) = max_file_size {
                if file_size > max_size {
                    tracing::warn!("Skipping large file: {} ({} bytes)", path.display(), file_size);
                    continue;
                }
            }

            // Read file content
            let content = match fs::read_to_string(path).await {
                Ok(text) => {
                    if self.is_binary_content(&text) {
                        FileContent::Binary(fs::read(path).await
                            .map_err(|e| Error::Api(format!("Failed to read binary file: {}", e)))?)
                    } else {
                        FileContent::Text(text)
                    }
                }
                Err(_) => {
                    // Try as binary
                    FileContent::Binary(fs::read(path).await
                        .map_err(|e| Error::Api(format!("Failed to read file: {}", e)))?)
                }
            };

            // Generate content hash
            let hash = self.generate_content_hash(&content)?;

            // Create file metadata
            let file_metadata = FileMetadata {
                file_type: self.detect_file_type(path),
                encoding: Some("utf-8".to_string()),
                language: self.detect_language(path),
                line_count: match &content {
                    FileContent::Text(text) => Some(text.lines().count()),
                    _ => None,
                },
                is_generated: self.is_generated_file(path),
                is_config: self.is_config_file(path),
                is_test: self.is_test_file(path),
                tags: HashSet::new(),
                custom: HashMap::new(),
            };

            let file_entry = FileEntry {
                path: relative_path.to_path_buf(),
                content,
                metadata: file_metadata,
                hash,
                size: file_size,
                modified: DateTime::from_timestamp(
                    metadata.modified()
                        .map_err(|e| Error::Api(format!("Failed to get modified time: {}", e)))?
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|e| Error::Api(format!("Invalid modified time: {}", e)))?
                        .as_secs() as i64,
                    0
                ).unwrap_or_else(|| Utc::now()),
                created: DateTime::from_timestamp(
                    metadata.created().ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or_else(|| Utc::now().timestamp()),
                    0
                ).unwrap_or_else(|| Utc::now()),
                permissions: None,
            };

            if file_entry.modified > last_modified {
                last_modified = file_entry.modified;
            }

            total_size += file_size;
            file_count += 1;
            files.insert(relative_path.to_path_buf(), file_entry);
        }

        Ok(ProjectFiles {
            root_path: project_root.to_path_buf(),
            files,
            total_size,
            file_count,
            last_modified,
        })
    }

    async fn collect_project_configuration(&self, project_root: &Path) -> Result<ProjectConfiguration> {
        let mut config_files = BTreeMap::new();
        let mut environment_variables = HashMap::new();

        // Common configuration files
        let config_file_names = vec![
            "package.json", "Cargo.toml", "pyproject.toml", "requirements.txt",
            "composer.json", "pom.xml", "build.gradle", "CMakeLists.txt",
            ".env", ".env.local", ".env.production", ".gitignore",
            "tsconfig.json", "webpack.config.js", "vite.config.js",
            "Dockerfile", "docker-compose.yml", ".dockerignore",
        ];

        for file_name in config_file_names {
            let config_path = project_root.join(file_name);
            if config_path.exists() {
                if let Ok(content) = fs::read_to_string(&config_path).await {
                    let config_value = if file_name.ends_with(".json") {
                        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::Value::String(content))
                    } else {
                        serde_json::Value::String(content)
                    };
                    config_files.insert(file_name.to_string(), config_value);
                }
            }
        }

        // Collect environment variables
        for (key, value) in std::env::vars() {
            if key.starts_with("PROJECT_") || key.starts_with("APP_") || key.starts_with("NODE_") {
                environment_variables.insert(key, value);
            }
        }

        Ok(ProjectConfiguration {
            config_files,
            environment_variables,
            build_configuration: None,
            dependencies: None,
            rag_configuration: None,
            custom_configs: HashMap::new(),
        })
    }

    async fn collect_project_memories(&self, project_id: &str) -> Result<ProjectMemories> {
        if let Some(memory_manager) = &self.memory_manager {
            // Collect memories from all types
            let mut memories_by_type = HashMap::new();
            let mut total_memories = 0;
            let mut total_size = 0u64;
            let mut oldest_memory = None;
            let mut newest_memory = None;

            for memory_type in [
                MemoryType::ShortTerm,
                MemoryType::LongTerm,
                MemoryType::Episodic,
                MemoryType::Semantic,
                MemoryType::Working,
            ] {
                let memories = memory_manager.recall("", Some(memory_type), 1000).await?;
                let type_name = format!("{:?}", memory_type);

                for memory in &memories {
                    total_size += memory.content.size_bytes() as u64;
                    total_memories += 1;

                    if oldest_memory.is_none() || memory.created_at < oldest_memory.unwrap() {
                        oldest_memory = Some(memory.created_at);
                    }
                    if newest_memory.is_none() || memory.created_at > newest_memory.unwrap() {
                        newest_memory = Some(memory.created_at);
                    }
                }

                memories_by_type.insert(type_name, memories);
            }

            let memory_statistics = MemoryStatistics {
                total_memories,
                memories_by_type: memories_by_type.iter()
                    .map(|(k, v)| (k.clone(), v.len()))
                    .collect(),
                average_importance: if total_memories > 0 {
                    memories_by_type.values()
                        .flatten()
                        .map(|m| m.importance_score)
                        .sum::<f64>() / total_memories as f64
                } else {
                    0.0
                },
                memory_size_bytes: total_size,
                oldest_memory,
                newest_memory,
            };

            Ok(ProjectMemories {
                project_memory_space: format!("project:{}", project_id),
                memories_by_type,
                memory_statistics,
                cross_project_references: Vec::new(),
                memory_tags: HashSet::new(),
            })
        } else {
            Ok(ProjectMemories {
                project_memory_space: format!("project:{}", project_id),
                memories_by_type: HashMap::new(),
                memory_statistics: MemoryStatistics {
                    total_memories: 0,
                    memories_by_type: HashMap::new(),
                    average_importance: 0.0,
                    memory_size_bytes: 0,
                    oldest_memory: None,
                    newest_memory: None,
                },
                cross_project_references: Vec::new(),
                memory_tags: HashSet::new(),
            })
        }
    }

    async fn collect_project_vectors(&self, _project_id: &str) -> Result<ProjectVectors> {
        // TODO: Integrate with vector store to collect embeddings
        Ok(ProjectVectors {
            document_embeddings: HashMap::new(),
            code_embeddings: HashMap::new(),
            vector_index_metadata: VectorIndexMetadata {
                index_type: "hnsw".to_string(),
                dimension: 384,
                total_vectors: 0,
                index_parameters: HashMap::new(),
                performance_metrics: IndexPerformanceMetrics {
                    average_query_time: std::time::Duration::from_millis(1),
                    index_size_bytes: 0,
                    rebuild_frequency: std::time::Duration::from_secs(24 * 60 * 60),
                    cache_hit_rate: 0.0,
                },
            },
            embedding_model_info: EmbeddingModelInfo {
                model_name: "all-MiniLM-L6-v2".to_string(),
                model_version: "1.0".to_string(),
                dimension: 384,
                max_sequence_length: 512,
                normalization: true,
            },
        })
    }

    async fn collect_project_metadata(&self, project_root: &Path, project_id: &str) -> Result<ProjectMetadata> {
        let mut programming_languages = HashSet::new();
        let mut frameworks = HashSet::new();
        let mut lines_of_code = 0u32;
        let mut file_count = 0u32;

        // Analyze project structure
        let walker = walkdir::WalkDir::new(project_root).max_depth(5);
        for entry in walker {
            if let Ok(entry) = entry {
                if entry.file_type().is_file() {
                    file_count += 1;
                    let path = entry.path();

                    // Detect programming language
                    if let Some(language) = self.detect_language(path) {
                        programming_languages.insert(language);
                    }

                    // Count lines for text files
                    if let Ok(content) = std::fs::read_to_string(path) {
                        lines_of_code += content.lines().count() as u32;
                    }
                }
            }
        }

        // Detect frameworks from config files
        if project_root.join("package.json").exists() {
            frameworks.insert("Node.js".to_string());
        }
        if project_root.join("Cargo.toml").exists() {
            frameworks.insert("Rust".to_string());
        }
        if project_root.join("pyproject.toml").exists() || project_root.join("requirements.txt").exists() {
            frameworks.insert("Python".to_string());
        }

        Ok(ProjectMetadata {
            project_name: project_id.to_string(),
            project_type: self.detect_project_type(project_root),
            programming_languages,
            frameworks,
            complexity_metrics: ComplexityMetrics {
                cyclomatic_complexity: None,
                lines_of_code,
                file_count,
                dependency_count: 0,
                test_coverage: None,
                code_duplication: None,
                technical_debt_ratio: None,
            },
            activity_metrics: ActivityMetrics {
                total_sessions: 0,
                total_session_time: std::time::Duration::from_secs(0),
                files_modified: 0,
                memories_created: 0,
                searches_performed: 0,
                last_activity: Utc::now(),
                activity_heatmap: HashMap::new(),
            },
            session_history: Vec::new(),
            custom_metadata: HashMap::new(),
        })
    }

    async fn optimize_snapshot(
        &self,
        mut snapshot: ProjectSnapshot,
        compress: bool,
        deduplicate: bool,
    ) -> Result<ProjectSnapshot> {
        let original_size = self.estimate_snapshot_size(&snapshot)?;
        let mut deduplication_savings = 0u64;

        if deduplicate {
            // Deduplicate file contents
            let mut content_hashes: HashMap<String, String> = HashMap::new();
            for (path, file_entry) in &mut snapshot.project_files.files {
                if let Some(existing_hash) = content_hashes.get(&file_entry.hash) {
                    let original_size = file_entry.content.size_bytes() as u64;
                    file_entry.content = FileContent::Deduplicated(existing_hash.clone());
                    deduplication_savings += original_size;
                } else {
                    content_hashes.insert(file_entry.hash.clone(), file_entry.hash.clone());
                }
            }
        }

        if compress {
            // Compress large file contents
            for (_, file_entry) in &mut snapshot.project_files.files {
                if file_entry.content.size_bytes() > self.compression_threshold {
                    match &file_entry.content {
                        FileContent::Text(text) => {
                            let compressed = self.compress_data(text.as_bytes())?;
                            file_entry.content = FileContent::Compressed(compressed);
                        }
                        FileContent::Binary(data) => {
                            let compressed = self.compress_data(data)?;
                            file_entry.content = FileContent::Compressed(compressed);
                        }
                        _ => {}
                    }
                }
            }
        }

        let final_size = self.estimate_snapshot_size(&snapshot)?;
        let compression_ratio = if original_size > 0 {
            final_size as f64 / original_size as f64
        } else {
            1.0
        };

        snapshot.snapshot_metadata.compression_ratio = compression_ratio;
        snapshot.snapshot_metadata.deduplication_savings = deduplication_savings;
        snapshot.snapshot_metadata.storage_size = final_size;

        Ok(snapshot)
    }

    async fn store_snapshot(&self, snapshot: &ProjectSnapshot) -> Result<()> {
        let key = format!("project_context:snapshot:{}", snapshot.id);
        let data = serde_json::to_vec(snapshot)
            .map_err(|e| Error::Api(format!("Failed to serialize snapshot: {}", e)))?;

        self.redis_manager.redis_set(&key, &data).await?;

        // Add to project snapshot index
        let index_key = format!("project_context:{}:snapshots", snapshot.project_id);
        self.redis_manager.redis_zadd(&index_key, &snapshot.id, snapshot.created_at.timestamp() as f64).await?;

        Ok(())
    }

    async fn load_snapshot(&self, snapshot_id: &str) -> Result<ProjectSnapshot> {
        let key = format!("project_context:snapshot:{}", snapshot_id);
        let data = self.redis_manager.redis_get(&key).await?
            .ok_or_else(|| Error::Api("Snapshot not found".to_string()))?;

        serde_json::from_slice(&data)
            .map_err(|e| Error::Api(format!("Failed to deserialize snapshot: {}", e)))
    }

    async fn delete_snapshot(&self, snapshot_id: &str) -> Result<()> {
        let key = format!("project_context:snapshot:{}", snapshot_id);
        self.redis_manager.redis_del(&key).await?;

        Ok(())
    }

    async fn get_latest_snapshot_id(&self, project_id: &str) -> Result<String> {
        let key = format!("project_context:{}:snapshots", project_id);
        let snapshot_ids = self.redis_manager.redis_zrevrange(&key, 0, 0).await?;

        snapshot_ids.into_iter().next()
            .ok_or_else(|| Error::Api("No snapshots found".to_string()))
    }

    async fn get_snapshot_by_version(&self, project_id: &str, version: &str) -> Result<ProjectSnapshot> {
        let snapshots = self.list_project_snapshots(project_id, None).await?;
        snapshots.into_iter()
            .find(|s| s.version == version)
            .ok_or_else(|| Error::Api(format!("Snapshot version {} not found", version)))
    }

    async fn generate_version(&self, project_id: &str) -> Result<String> {
        let snapshots = self.list_project_snapshots(project_id, Some(1)).await?;
        if snapshots.is_empty() {
            Ok("1.0.0".to_string())
        } else {
            // Simple version increment logic
            let latest = &snapshots[0];
            let parts: Vec<&str> = latest.version.split('.').collect();
            if parts.len() == 3 {
                if let Ok(patch) = parts[2].parse::<u32>() {
                    return Ok(format!("{}.{}.{}", parts[0], parts[1], patch + 1));
                }
            }
            Ok(format!("{}.1", latest.version))
        }
    }

    async fn update_project_index(&self, project_id: &str, snapshot: &ProjectSnapshot) -> Result<()> {
        let key = format!("project_context:{}:metadata", project_id);
        let metadata = serde_json::json!({
            "latest_snapshot": snapshot.id,
            "latest_version": snapshot.version,
            "last_updated": snapshot.created_at,
            "total_files": snapshot.project_files.file_count,
            "total_size": snapshot.project_files.total_size,
            "project_type": snapshot.project_metadata.project_type,
            "programming_languages": snapshot.project_metadata.programming_languages,
        });

        let metadata_data = serde_json::to_vec(&metadata)
            .map_err(|e| Error::Api(format!("Failed to serialize metadata: {}", e)))?;
        self.redis_manager.redis_set(&key, &metadata_data).await?;

        Ok(())
    }

    // Helper methods for file analysis and processing

    fn matches_patterns(&self, path: &Path, patterns: &[String]) -> bool {
        patterns.iter().any(|pattern| {
            if let Ok(glob) = glob::Pattern::new(pattern) {
                glob.matches_path(path)
            } else {
                false
            }
        })
    }

    fn is_binary_content(&self, content: &str) -> bool {
        content.chars().any(|c| c == '\0' || (c as u32) < 32 && c != '\t' && c != '\n' && c != '\r')
    }

    fn detect_file_type(&self, path: &Path) -> String {
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            match extension.to_lowercase().as_str() {
                "rs" => "rust".to_string(),
                "py" => "python".to_string(),
                "js" | "jsx" => "javascript".to_string(),
                "ts" | "tsx" => "typescript".to_string(),
                "json" => "json".to_string(),
                "toml" => "toml".to_string(),
                "yaml" | "yml" => "yaml".to_string(),
                "md" => "markdown".to_string(),
                "txt" => "text".to_string(),
                _ => "unknown".to_string(),
            }
        } else {
            "unknown".to_string()
        }
    }

    fn detect_language(&self, path: &Path) -> Option<String> {
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            match extension.to_lowercase().as_str() {
                "rs" => Some("Rust".to_string()),
                "py" => Some("Python".to_string()),
                "js" | "jsx" => Some("JavaScript".to_string()),
                "ts" | "tsx" => Some("TypeScript".to_string()),
                "c" => Some("C".to_string()),
                "cpp" | "cc" | "cxx" => Some("C++".to_string()),
                "java" => Some("Java".to_string()),
                "go" => Some("Go".to_string()),
                "php" => Some("PHP".to_string()),
                "rb" => Some("Ruby".to_string()),
                "swift" => Some("Swift".to_string()),
                "kt" => Some("Kotlin".to_string()),
                _ => None,
            }
        } else {
            None
        }
    }

    fn is_generated_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        path_str.contains("generated") ||
        path_str.contains("build") ||
        path_str.contains("dist") ||
        path_str.contains("target") ||
        path.file_name().and_then(|n| n.to_str()).map_or(false, |name| {
            name.starts_with("_") || name.ends_with(".generated.rs")
        })
    }

    fn is_config_file(&self, path: &Path) -> bool {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            matches!(name,
                "package.json" | "Cargo.toml" | "pyproject.toml" | "requirements.txt" |
                "composer.json" | "pom.xml" | "build.gradle" | "CMakeLists.txt" |
                ".env" | ".env.local" | ".env.production" | ".gitignore" |
                "tsconfig.json" | "webpack.config.js" | "vite.config.js" |
                "Dockerfile" | "docker-compose.yml" | ".dockerignore"
            )
        } else {
            false
        }
    }

    fn is_test_file(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy().to_lowercase();
        path_str.contains("test") ||
        path_str.contains("spec") ||
        path_str.contains("__tests__")
    }

    fn detect_project_type(&self, project_root: &Path) -> Option<String> {
        if project_root.join("Cargo.toml").exists() {
            Some("Rust".to_string())
        } else if project_root.join("package.json").exists() {
            Some("Node.js".to_string())
        } else if project_root.join("pyproject.toml").exists() || project_root.join("requirements.txt").exists() {
            Some("Python".to_string())
        } else if project_root.join("pom.xml").exists() {
            Some("Java".to_string())
        } else if project_root.join("CMakeLists.txt").exists() {
            Some("C/C++".to_string())
        } else {
            None
        }
    }

    fn generate_content_hash(&self, content: &FileContent) -> Result<String> {
        let mut hasher = Sha256::new();
        match content {
            FileContent::Text(text) => hasher.update(text.as_bytes()),
            FileContent::Binary(data) => hasher.update(data),
            FileContent::Compressed(data) => hasher.update(data),
            FileContent::Reference(reference) => hasher.update(reference.as_bytes()),
            FileContent::Deduplicated(hash) => return Ok(hash.clone()),
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn generate_validation_hash(&self, snapshot: &ProjectSnapshot) -> Result<String> {
        let mut hasher = Sha256::new();

        // Hash core snapshot data (excluding metadata that changes)
        hasher.update(snapshot.id.as_bytes());
        hasher.update(snapshot.project_id.as_bytes());
        hasher.update(snapshot.version.as_bytes());

        // Hash file contents
        for (path, file_entry) in &snapshot.project_files.files {
            hasher.update(path.to_string_lossy().as_bytes());
            hasher.update(&file_entry.hash.as_bytes());
        }

        // Hash configuration
        if let Ok(config_data) = serde_json::to_vec(&snapshot.project_configuration) {
            hasher.update(&config_data);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| Error::Api(format!("Compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| Error::Api(format!("Compression finalization failed: {}", e)))
    }

    fn decompress_data(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| Error::Api(format!("Decompression failed: {}", e)))?;
        Ok(decompressed)
    }

    fn estimate_snapshot_size(&self, snapshot: &ProjectSnapshot) -> Result<u64> {
        serde_json::to_vec(snapshot)
            .map(|v| v.len() as u64)
            .map_err(|e| Error::Api(format!("Failed to estimate size: {}", e)))
    }

    // Diff implementation methods

    fn diff_files(&self, from: &ProjectFiles, to: &ProjectFiles) -> Result<FileChanges> {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let renamed = Vec::new();

        // Find added and modified files
        for (path, to_file) in &to.files {
            if let Some(from_file) = from.files.get(path) {
                if from_file.hash != to_file.hash {
                    modified.push(FileModification {
                        path: path.clone(),
                        changes: Vec::new(), // TODO: Implement line-by-line diff
                        size_delta: to_file.size as i64 - from_file.size as i64,
                        hash_before: from_file.hash.clone(),
                        hash_after: to_file.hash.clone(),
                    });
                }
            } else {
                added.push(path.clone());
            }
        }

        // Find removed files
        for path in from.files.keys() {
            if !to.files.contains_key(path) {
                removed.push(path.clone());
            }
        }

        Ok(FileChanges {
            added,
            removed,
            modified,
            renamed,
        })
    }

    fn diff_memories(&self, from: &ProjectMemories, to: &ProjectMemories) -> Result<MemoryChanges> {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut type_changes = HashMap::new();

        // Create lookup maps for easier comparison
        let from_memories: HashMap<String, &MemoryEntry> = from.memories_by_type
            .values()
            .flatten()
            .map(|m| (m.id.clone(), m))
            .collect();

        let to_memories: HashMap<String, &MemoryEntry> = to.memories_by_type
            .values()
            .flatten()
            .map(|m| (m.id.clone(), m))
            .collect();

        for (id, to_memory) in &to_memories {
            if let Some(from_memory) = from_memories.get(id) {
                if from_memory.content.size_bytes() != to_memory.content.size_bytes() ||
                   from_memory.importance_score != to_memory.importance_score {
                    modified.push(MemoryModification {
                        memory_id: id.clone(),
                        field_changes: HashMap::new(),
                        importance_delta: to_memory.importance_score - from_memory.importance_score,
                        content_diff: None,
                    });
                }

                if format!("{:?}", from_memory.memory_type) != format!("{:?}", to_memory.memory_type) {
                    type_changes.insert(
                        id.clone(),
                        (format!("{:?}", from_memory.memory_type), format!("{:?}", to_memory.memory_type))
                    );
                }
            } else {
                added.push(id.clone());
            }
        }

        for id in from_memories.keys() {
            if !to_memories.contains_key(id) {
                removed.push(id.clone());
            }
        }

        Ok(MemoryChanges {
            added,
            removed,
            modified,
            type_changes,
        })
    }

    fn diff_configs(&self, from: &ProjectConfiguration, to: &ProjectConfiguration) -> Result<ConfigChanges> {
        let mut added_configs = HashMap::new();
        let mut removed_configs = Vec::new();
        let mut modified_configs = HashMap::new();
        let mut environment_changes = HashMap::new();

        // Compare config files
        for (key, to_value) in &to.config_files {
            if let Some(from_value) = from.config_files.get(key) {
                if from_value != to_value {
                    modified_configs.insert(key.clone(), ConfigModification {
                        config_key: key.clone(),
                        old_value: from_value.clone(),
                        new_value: to_value.clone(),
                        change_impact: ChangeImpact::Medium,
                    });
                }
            } else {
                added_configs.insert(key.clone(), to_value.clone());
            }
        }

        for key in from.config_files.keys() {
            if !to.config_files.contains_key(key) {
                removed_configs.push(key.clone());
            }
        }

        // Compare environment variables
        for (var, to_value) in &to.environment_variables {
            if let Some(from_value) = from.environment_variables.get(var) {
                if from_value != to_value {
                    environment_changes.insert(var.clone(), EnvironmentChange {
                        variable: var.clone(),
                        old_value: Some(from_value.clone()),
                        new_value: Some(to_value.clone()),
                        change_type: ChangeType::Modified,
                    });
                }
            } else {
                environment_changes.insert(var.clone(), EnvironmentChange {
                    variable: var.clone(),
                    old_value: None,
                    new_value: Some(to_value.clone()),
                    change_type: ChangeType::Added,
                });
            }
        }

        Ok(ConfigChanges {
            added_configs,
            removed_configs,
            modified_configs,
            environment_changes,
        })
    }

    fn diff_vectors(&self, _from: &ProjectVectors, _to: &ProjectVectors) -> Result<VectorChanges> {
        // TODO: Implement vector diff logic
        Ok(VectorChanges {
            added_documents: Vec::new(),
            removed_documents: Vec::new(),
            updated_embeddings: Vec::new(),
            index_changes: Vec::new(),
        })
    }

    fn diff_metadata(&self, from: &ProjectMetadata, to: &ProjectMetadata) -> Result<MetadataChanges> {
        let complexity_delta = ComplexityDelta {
            cyclomatic_complexity_delta: match (from.complexity_metrics.cyclomatic_complexity, to.complexity_metrics.cyclomatic_complexity) {
                (Some(from_cc), Some(to_cc)) => Some(to_cc as i32 - from_cc as i32),
                _ => None,
            },
            lines_of_code_delta: to.complexity_metrics.lines_of_code as i32 - from.complexity_metrics.lines_of_code as i32,
            file_count_delta: to.complexity_metrics.file_count as i32 - from.complexity_metrics.file_count as i32,
            dependency_count_delta: to.complexity_metrics.dependency_count as i32 - from.complexity_metrics.dependency_count as i32,
            test_coverage_delta: match (from.complexity_metrics.test_coverage, to.complexity_metrics.test_coverage) {
                (Some(from_tc), Some(to_tc)) => Some(to_tc - from_tc),
                _ => None,
            },
        };

        let activity_delta = ActivityDelta {
            session_count_delta: to.activity_metrics.total_sessions as i32 - from.activity_metrics.total_sessions as i32,
            session_time_delta: to.activity_metrics.total_session_time.saturating_sub(from.activity_metrics.total_session_time),
            files_modified_delta: to.activity_metrics.files_modified as i32 - from.activity_metrics.files_modified as i32,
            memories_created_delta: to.activity_metrics.memories_created as i32 - from.activity_metrics.memories_created as i32,
        };

        Ok(MetadataChanges {
            complexity_delta,
            activity_delta,
            session_changes: to.session_history.clone(),
            custom_metadata_changes: HashMap::new(),
        })
    }
}

/// Project statistics across all snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatistics {
    pub project_id: String,
    pub total_snapshots: usize,
    pub total_storage_size: u64,
    pub oldest_snapshot: Option<DateTime<Utc>>,
    pub newest_snapshot: Option<DateTime<Utc>>,
    pub snapshot_types: HashMap<String, usize>,
    pub storage_trend: Vec<(DateTime<Utc>, u64)>,
    pub complexity_trend: Vec<(DateTime<Utc>, u32)>,
    pub activity_trend: Vec<(DateTime<Utc>, u32)>,
}

// Additional imports needed
use walkdir;
use glob;

#[cfg(test)]
mod tests;