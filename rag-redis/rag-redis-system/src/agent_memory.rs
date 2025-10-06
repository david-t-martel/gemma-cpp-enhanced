//! # Agent Memory Embedding System
//!
//! This module provides a sophisticated memory management system specifically designed
//! for LLM agents (Claude, Gemini, Gemma, etc.). It implements context hints, memory
//! prompts, semantic embeddings, and agent-specific retrieval interfaces.
//!
//! ## Features
//!
//! - **Context Hints**: Provides contextual clues for agents to understand memory relevance
//! - **Memory Prompts**: Generates agent-specific prompts based on memory content
//! - **Semantic Embeddings**: Creates and manages embeddings for agent memory retrieval
//! - **Agent Templates**: Customized memory formats for different LLM types
//! - **Memory Digests**: Summarizes relevant memories for efficient agent consumption
//! - **Multi-tier Caching**: Optimized retrieval with local and distributed caching
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use rag_redis_system::agent_memory::{AgentMemorySystem, AgentType, ContextHint};
//! use rag_redis_system::redis_backend::RedisManager;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let redis_manager = Arc::new(RedisManager::new(&Default::default()).await?);
//! let agent_memory = AgentMemorySystem::new(redis_manager, Default::default()).await?;
//!
//! // Store contextualized memory for Claude
//! let memory_id = agent_memory.store_contextualized(
//!     "User preferences include dark mode and compact UI",
//!     AgentType::Claude,
//!     vec![ContextHint::UserPreference, ContextHint::InterfaceSettings],
//!     0.8,
//! ).await?;
//!
//! // Retrieve memories with agent-specific formatting
//! let memories = agent_memory.retrieve_for_agent(
//!     "What are the user's UI preferences?",
//!     AgentType::Claude,
//!     5,
//! ).await?;
//!
//! // Generate memory digest for agent context
//! let digest = agent_memory.generate_digest(
//!     AgentType::Claude,
//!     Some("UI customization"),
//!     10,
//! ).await?;
//! # Ok(())
//! # }
//! ```

use crate::embedding::EmbeddingService;
use crate::error::{Error, Result};
use crate::memory::MemoryManager;
use crate::redis_backend::RedisManager;
use crate::vector_store::VectorStore;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{error, info};
use uuid::Uuid;

/// Supported LLM agent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    Claude,
    Gemini,
    Gemma,
    GPT4,
    Llama,
    Custom(u8), // Support up to 256 custom agent types
}

impl AgentType {
    pub fn name(&self) -> &str {
        match self {
            AgentType::Claude => "Claude",
            AgentType::Gemini => "Gemini",
            AgentType::Gemma => "Gemma",
            AgentType::GPT4 => "GPT-4",
            AgentType::Llama => "Llama",
            AgentType::Custom(_) => "Custom",
        }
    }

    pub fn context_window_size(&self) -> usize {
        match self {
            AgentType::Claude => 200_000, // Claude 3 context window
            AgentType::Gemini => 1_000_000, // Gemini 1.5 Pro
            AgentType::GPT4 => 128_000, // GPT-4 Turbo
            AgentType::Gemma => 8_192, // Gemma default
            AgentType::Llama => 32_768, // Llama 2
            AgentType::Custom(_) => 16_384, // Conservative default
        }
    }

    pub fn optimal_chunk_size(&self) -> usize {
        match self {
            AgentType::Claude => 2048,
            AgentType::Gemini => 4096,
            AgentType::GPT4 => 2048,
            AgentType::Gemma => 512,
            AgentType::Llama => 1024,
            AgentType::Custom(_) => 1024,
        }
    }
}

/// Context hints for memory relevance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextHint {
    UserPreference,
    SystemConfiguration,
    ConversationHistory,
    TaskContext,
    DomainKnowledge,
    PersonalInformation,
    TemporalContext,
    SpatialContext,
    EmotionalContext,
    InterfaceSettings,
    LearningOutcome,
    ErrorCorrection,
    PerformanceMetric,
    SecurityContext,
    Custom(String),
}

impl ContextHint {
    pub fn weight(&self) -> f32 {
        match self {
            ContextHint::UserPreference => 0.9,
            ContextHint::PersonalInformation => 0.85,
            ContextHint::TaskContext => 0.8,
            ContextHint::ConversationHistory => 0.75,
            ContextHint::DomainKnowledge => 0.7,
            ContextHint::SystemConfiguration => 0.65,
            ContextHint::TemporalContext => 0.6,
            ContextHint::LearningOutcome => 0.6,
            ContextHint::ErrorCorrection => 0.55,
            ContextHint::EmotionalContext => 0.5,
            ContextHint::InterfaceSettings => 0.45,
            ContextHint::SpatialContext => 0.4,
            ContextHint::PerformanceMetric => 0.35,
            ContextHint::SecurityContext => 0.3,
            ContextHint::Custom(_) => 0.5,
        }
    }

    pub fn to_prompt_prefix(&self) -> &str {
        match self {
            ContextHint::UserPreference => "User preference: ",
            ContextHint::SystemConfiguration => "System configuration: ",
            ContextHint::ConversationHistory => "Previous conversation: ",
            ContextHint::TaskContext => "Current task: ",
            ContextHint::DomainKnowledge => "Domain knowledge: ",
            ContextHint::PersonalInformation => "Personal context: ",
            ContextHint::TemporalContext => "Temporal context: ",
            ContextHint::SpatialContext => "Spatial context: ",
            ContextHint::EmotionalContext => "Emotional context: ",
            ContextHint::InterfaceSettings => "Interface setting: ",
            ContextHint::LearningOutcome => "Learned: ",
            ContextHint::ErrorCorrection => "Correction: ",
            ContextHint::PerformanceMetric => "Performance metric: ",
            ContextHint::SecurityContext => "Security context: ",
            ContextHint::Custom(s) => s,
        }
    }
}

/// Agent-specific memory entry with embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemory {
    pub id: String,
    pub content: String,
    pub agent_type: AgentType,
    pub context_hints: Vec<ContextHint>,
    pub embedding: Option<Vec<f32>>,
    pub importance_score: f32,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub related_memories: Vec<String>, // IDs of related memories
    pub summary: Option<String>,
}

impl AgentMemory {
    pub fn new(
        content: String,
        agent_type: AgentType,
        context_hints: Vec<ContextHint>,
        importance: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            agent_type,
            context_hints,
            embedding: None,
            importance_score: importance,
            access_count: 0,
            last_accessed: now,
            created_at: now,
            metadata: HashMap::new(),
            related_memories: Vec::new(),
            summary: None,
        }
    }

    pub fn calculate_relevance_score(&self, query_hints: &[ContextHint], recency_weight: f32) -> f32 {
        // Calculate hint overlap score
        let hint_score: f32 = self.context_hints.iter()
            .filter(|h| query_hints.contains(h))
            .map(|h| h.weight())
            .sum::<f32>()
            / (self.context_hints.len().max(1) as f32);

        // Calculate recency score (exponential decay)
        let age = Utc::now().signed_duration_since(self.last_accessed);
        let age_hours = age.num_hours() as f32;
        let recency_score = (-age_hours / 24.0).exp(); // Exponential decay over days

        // Calculate access frequency score
        let frequency_score = (self.access_count as f32).log2().max(1.0) / 10.0;

        // Weighted combination
        let base_score = (hint_score * 0.4)
            + (self.importance_score * 0.3)
            + (recency_score * recency_weight)
            + (frequency_score * 0.1);

        base_score.min(1.0).max(0.0)
    }
}

/// Memory template for different agent types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTemplate {
    pub agent_type: AgentType,
    pub format_style: FormatStyle,
    pub include_metadata: bool,
    pub include_timestamps: bool,
    pub include_confidence: bool,
    pub max_context_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormatStyle {
    Markdown,
    JSON,
    XML,
    PlainText,
    Structured,
}

impl MemoryTemplate {
    pub fn for_agent(agent_type: AgentType) -> Self {
        match agent_type {
            AgentType::Claude => Self {
                agent_type,
                format_style: FormatStyle::Markdown,
                include_metadata: true,
                include_timestamps: true,
                include_confidence: true,
                max_context_length: 4096,
            },
            AgentType::Gemini => Self {
                agent_type,
                format_style: FormatStyle::Structured,
                include_metadata: true,
                include_timestamps: false,
                include_confidence: true,
                max_context_length: 8192,
            },
            AgentType::GPT4 => Self {
                agent_type,
                format_style: FormatStyle::JSON,
                include_metadata: false,
                include_timestamps: false,
                include_confidence: true,
                max_context_length: 4096,
            },
            AgentType::Gemma => Self {
                agent_type,
                format_style: FormatStyle::PlainText,
                include_metadata: false,
                include_timestamps: false,
                include_confidence: false,
                max_context_length: 2048,
            },
            AgentType::Llama => Self {
                agent_type,
                format_style: FormatStyle::Markdown,
                include_metadata: false,
                include_timestamps: false,
                include_confidence: true,
                max_context_length: 2048,
            },
            AgentType::Custom(_) => Self {
                agent_type,
                format_style: FormatStyle::PlainText,
                include_metadata: false,
                include_timestamps: false,
                include_confidence: false,
                max_context_length: 1024,
            },
        }
    }

    pub fn format_memory(&self, memory: &AgentMemory) -> String {
        match self.format_style {
            FormatStyle::Markdown => self.format_as_markdown(memory),
            FormatStyle::JSON => self.format_as_json(memory),
            FormatStyle::XML => self.format_as_xml(memory),
            FormatStyle::PlainText => self.format_as_plain(memory),
            FormatStyle::Structured => self.format_as_structured(memory),
        }
    }

    fn format_as_markdown(&self, memory: &AgentMemory) -> String {
        let mut output = String::new();

        // Add context hints as tags
        if !memory.context_hints.is_empty() {
            output.push_str("**Context**: ");
            for hint in &memory.context_hints {
                output.push_str(&format!("`{}` ", hint.to_prompt_prefix().trim_end()));
            }
            output.push_str("\n\n");
        }

        // Add main content
        output.push_str(&memory.content);

        // Add metadata if configured
        if self.include_metadata && !memory.metadata.is_empty() {
            output.push_str("\n\n---\n");
            output.push_str("**Metadata**:\n");
            for (key, value) in &memory.metadata {
                output.push_str(&format!("- {}: {}\n", key, value));
            }
        }

        // Add confidence/importance
        if self.include_confidence {
            output.push_str(&format!("\n*Confidence: {:.0}%*", memory.importance_score * 100.0));
        }

        // Add timestamp
        if self.include_timestamps {
            output.push_str(&format!(
                "\n*Last accessed: {}*",
                memory.last_accessed.format("%Y-%m-%d %H:%M UTC")
            ));
        }

        output
    }

    fn format_as_json(&self, memory: &AgentMemory) -> String {
        let mut json_obj = serde_json::json!({
            "content": memory.content,
            "hints": memory.context_hints.iter()
                .map(|h| h.to_prompt_prefix())
                .collect::<Vec<_>>(),
        });

        if self.include_confidence {
            json_obj["confidence"] = serde_json::json!(memory.importance_score);
        }

        if self.include_timestamps {
            json_obj["last_accessed"] = serde_json::json!(memory.last_accessed.to_rfc3339());
        }

        if self.include_metadata {
            json_obj["metadata"] = serde_json::json!(memory.metadata);
        }

        serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| memory.content.clone())
    }

    fn format_as_xml(&self, memory: &AgentMemory) -> String {
        let mut output = String::from("<memory>\n");

        output.push_str(&format!("  <content>{}</content>\n",
            xmltree::escape(&memory.content)));

        if !memory.context_hints.is_empty() {
            output.push_str("  <hints>\n");
            for hint in &memory.context_hints {
                output.push_str(&format!("    <hint>{}</hint>\n",
                    xmltree::escape(hint.to_prompt_prefix())));
            }
            output.push_str("  </hints>\n");
        }

        if self.include_confidence {
            output.push_str(&format!("  <confidence>{:.2}</confidence>\n",
                memory.importance_score));
        }

        output.push_str("</memory>");
        output
    }

    fn format_as_plain(&self, memory: &AgentMemory) -> String {
        let mut output = String::new();

        // Add context prefix if hints exist
        if !memory.context_hints.is_empty() {
            output.push_str(&format!(
                "[{}] ",
                memory.context_hints[0].to_prompt_prefix().trim_end()
            ));
        }

        output.push_str(&memory.content);
        output
    }

    fn format_as_structured(&self, memory: &AgentMemory) -> String {
        let mut output = String::new();

        // Header
        output.push_str("═══ Memory Entry ═══\n\n");

        // Context
        if !memory.context_hints.is_empty() {
            output.push_str("Context Types:\n");
            for hint in &memory.context_hints {
                output.push_str(&format!("  • {}\n", hint.to_prompt_prefix().trim_end()));
            }
            output.push_str("\n");
        }

        // Content
        output.push_str("Content:\n");
        output.push_str(&format!("  {}\n", memory.content));

        // Footer
        if self.include_confidence {
            output.push_str(&format!("\nRelevance: {:.0}%\n", memory.importance_score * 100.0));
        }

        output.push_str("═══════════════════\n");
        output
    }
}

/// Memory digest for agent consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDigest {
    pub agent_type: AgentType,
    pub topic: Option<String>,
    pub summary: String,
    pub key_points: Vec<String>,
    pub relevant_memories: Vec<AgentMemory>,
    pub total_memories_considered: usize,
    pub confidence_score: f32,
    pub generated_at: DateTime<Utc>,
}

/// Configuration for the agent memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemoryConfig {
    pub embedding_cache_size: usize,
    pub context_window_buffer: f32, // Percentage of context window to use
    pub similarity_threshold: f32,
    pub max_related_memories: usize,
    pub digest_summary_length: usize,
    pub auto_consolidate: bool,
    pub consolidation_interval: Duration,
}

impl Default for AgentMemoryConfig {
    fn default() -> Self {
        Self {
            embedding_cache_size: 10000,
            context_window_buffer: 0.8, // Use 80% of context window
            similarity_threshold: 0.7,
            max_related_memories: 5,
            digest_summary_length: 500,
            auto_consolidate: true,
            consolidation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Main agent memory system
pub struct AgentMemorySystem {
    redis_manager: Arc<RedisManager>,
    memory_manager: Arc<MemoryManager>,
    embedding_service: Arc<Box<dyn EmbeddingService>>,
    vector_store: Arc<AsyncRwLock<VectorStore>>,
    config: AgentMemoryConfig,

    // Caches
    embedding_cache: Arc<DashMap<String, Vec<f32>>>,
    template_cache: Arc<DashMap<AgentType, MemoryTemplate>>,
    digest_cache: Arc<RwLock<HashMap<String, (MemoryDigest, DateTime<Utc>)>>>,

    // Statistics
    stats: Arc<RwLock<AgentMemoryStats>>,
}

#[derive(Debug, Default, Clone)]
struct AgentMemoryStats {
    total_stores: u64,
    total_retrievals: u64,
    cache_hits: u64,
    cache_misses: u64,
    digest_generations: u64,
    consolidations: u64,
}

impl AgentMemorySystem {
    pub async fn new(
        redis_manager: Arc<RedisManager>,
        config: AgentMemoryConfig,
    ) -> Result<Self> {
        // Initialize memory manager
        let memory_config = crate::memory::MemoryConfig::default();
        let memory_manager = MemoryManager::new(redis_manager.clone(), memory_config).await?;

        // Initialize embedding service
        let embedding_config = crate::config::EmbeddingConfig::default();
        let embedding_service = Arc::new(
            crate::embedding::EmbeddingFactory::create(&embedding_config).await?
        );

        // Initialize vector store
        let vector_config = crate::config::VectorStoreConfig::default();
        let vector_store = Arc::new(AsyncRwLock::new(
            VectorStore::new(vector_config)?
        ));

        // Initialize template cache with defaults
        let template_cache = Arc::new(DashMap::new());
        for agent_type in [
            AgentType::Claude,
            AgentType::Gemini,
            AgentType::GPT4,
            AgentType::Gemma,
            AgentType::Llama,
        ] {
            template_cache.insert(agent_type, MemoryTemplate::for_agent(agent_type));
        }

        let system = Self {
            redis_manager,
            memory_manager,
            embedding_service,
            vector_store,
            config,
            embedding_cache: Arc::new(DashMap::new()),
            template_cache,
            digest_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AgentMemoryStats::default())),
        };

        // Start background consolidation if enabled
        if system.config.auto_consolidate {
            system.start_consolidation_task();
        }

        Ok(system)
    }

    /// Store contextualized memory with embeddings
    pub async fn store_contextualized(
        &self,
        content: String,
        agent_type: AgentType,
        context_hints: Vec<ContextHint>,
        importance: f32,
    ) -> Result<String> {
        let mut memory = AgentMemory::new(content.clone(), agent_type, context_hints, importance);

        // Generate embedding
        let embedding = self.get_or_compute_embedding(&content).await?;
        memory.embedding = Some(embedding.clone());

        // Find related memories
        let related = self.find_related_memories(&embedding, self.config.max_related_memories).await?;
        memory.related_memories = related.iter().map(|m| m.id.clone()).collect();

        // Generate summary if content is long
        if content.len() > 500 {
            memory.summary = Some(self.generate_summary(&content, 100));
        }

        // Store in Redis
        let serialized = bincode::serialize(&memory)
            .map_err(|e| Error::Serialization(format!("Failed to serialize agent memory: {}", e)))?;

        let key = format!("agent_memory:{}:{}", agent_type.name(), memory.id);
        self.redis_manager
            .set_raw(&key, &serialized)
            .await?;

        // Store in vector store
        self.vector_store.write().await
            .add_vector(&memory.id, &embedding, serde_json::to_value(memory.metadata.clone()).unwrap_or_default())?;

        // Update stats
        self.stats.write().total_stores += 1;

        info!("Stored agent memory {} for {}", memory.id, agent_type.name());
        Ok(memory.id)
    }

    /// Retrieve memories formatted for specific agent
    pub async fn retrieve_for_agent(
        &self,
        query: &str,
        agent_type: AgentType,
        limit: usize,
    ) -> Result<Vec<String>> {
        let query_embedding = self.get_or_compute_embedding(query).await?;

        // Search vector store
        let results = self.vector_store.read().await
            .search(&query_embedding, limit * 2, None)?;

        let mut agent_memories = Vec::new();

        for (id, score, _metadata) in results {
            if let Ok(memory) = self.get_agent_memory(&id).await {
                if memory.agent_type == agent_type || self.is_compatible(memory.agent_type, agent_type) {
                    agent_memories.push((memory, score));
                }
            }
        }

        // Sort by relevance
        agent_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        agent_memories.truncate(limit);

        // Format memories using agent template
        let template = self.get_or_create_template(agent_type);
        let formatted: Vec<String> = agent_memories
            .into_iter()
            .map(|(mut memory, _)| {
                memory.access_count += 1;
                memory.last_accessed = Utc::now();
                let _ = self.update_agent_memory(&memory); // Fire and forget
                template.format_memory(&memory)
            })
            .collect();

        // Update stats
        self.stats.write().total_retrievals += 1;

        Ok(formatted)
    }

    /// Generate a memory digest for agent context
    pub async fn generate_digest(
        &self,
        agent_type: AgentType,
        topic: Option<String>,
        max_memories: usize,
    ) -> Result<MemoryDigest> {
        // Check cache
        if let Some(topic_str) = &topic {
            let cache_key = format!("{}:{}", agent_type.name(), topic_str);
            let cache = self.digest_cache.read();
            if let Some((digest, generated_at)) = cache.get(&cache_key) {
                let age = Utc::now().signed_duration_since(*generated_at);
                if age.num_minutes() < 30 {
                    self.stats.write().cache_hits += 1;
                    return Ok(digest.clone());
                }
            }
        }

        self.stats.write().cache_misses += 1;

        // Retrieve relevant memories
        let query = topic.as_deref().unwrap_or("general context");
        let query_embedding = self.get_or_compute_embedding(query).await?;

        let results = self.vector_store.read().await
            .search(&query_embedding, max_memories * 3, None)?;

        let mut relevant_memories = Vec::new();
        let total_considered = results.len();

        for (id, score, _) in results {
            if let Ok(memory) = self.get_agent_memory(&id).await {
                if memory.agent_type == agent_type || self.is_compatible(memory.agent_type, agent_type) {
                    relevant_memories.push((memory, score));
                }
            }
        }

        // Sort and limit
        relevant_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        relevant_memories.truncate(max_memories);

        // Generate key points
        let key_points = self.extract_key_points(&relevant_memories);

        // Generate summary
        let summary = self.generate_digest_summary(&relevant_memories, &topic);

        // Calculate confidence
        let confidence_score = if relevant_memories.is_empty() {
            0.0
        } else {
            relevant_memories.iter().map(|(_, s)| s).sum::<f32>() / relevant_memories.len() as f32
        };

        let digest = MemoryDigest {
            agent_type,
            topic: topic.clone(),
            summary,
            key_points,
            relevant_memories: relevant_memories.into_iter().map(|(m, _)| m).collect(),
            total_memories_considered: total_considered,
            confidence_score,
            generated_at: Utc::now(),
        };

        // Cache the digest
        if let Some(topic_str) = &topic {
            let cache_key = format!("{}:{}", agent_type.name(), topic_str);
            self.digest_cache.write().insert(cache_key, (digest.clone(), Utc::now()));
        }

        // Update stats
        self.stats.write().digest_generations += 1;

        Ok(digest)
    }

    /// Create memory prompts for agent interaction
    pub async fn create_memory_prompt(
        &self,
        agent_type: AgentType,
        context_hints: Vec<ContextHint>,
        max_tokens: usize,
    ) -> Result<String> {
        let mut prompt = String::new();
        let template = self.get_or_create_template(agent_type);

        // Add system context
        prompt.push_str("Based on the following context from memory:\n\n");

        // Retrieve memories based on context hints
        let mut all_memories = Vec::new();

        for hint in &context_hints {
            let query = hint.to_prompt_prefix();
            let memories = self.retrieve_for_agent(query, agent_type, 3).await?;

            for memory_str in memories {
                all_memories.push((hint.clone(), memory_str));
            }
        }

        // Build prompt with token limit
        let mut current_tokens = 0;
        let avg_tokens_per_char = 0.25; // Rough estimate

        for (hint, memory_str) in all_memories {
            let section = format!("{}:\n{}\n\n", hint.to_prompt_prefix(), memory_str);
            let estimated_tokens = (section.len() as f32 * avg_tokens_per_char) as usize;

            if current_tokens + estimated_tokens > max_tokens {
                break;
            }

            prompt.push_str(&section);
            current_tokens += estimated_tokens;
        }

        // Add guidance for agent
        match agent_type {
            AgentType::Claude => {
                prompt.push_str("\nConsider this context when formulating your response, but focus on being helpful, harmless, and honest.");
            }
            AgentType::Gemini => {
                prompt.push_str("\nUse this context to provide accurate and comprehensive responses.");
            }
            AgentType::GPT4 => {
                prompt.push_str("\nIncorporate this contextual information naturally into your response.");
            }
            AgentType::Gemma => {
                prompt.push_str("\nUse this context for your response.");
            }
            _ => {
                prompt.push_str("\nConsider this context in your response.");
            }
        }

        Ok(prompt)
    }

    // Helper methods

    async fn get_or_compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache
        if let Some(embedding) = self.embedding_cache.get(text) {
            return Ok(embedding.clone());
        }

        // Compute embedding
        let embedding = self.embedding_service.embed_text(text).await
            .map_err(|e| Error::Api(format!("Failed to generate embedding: {}", e)))?;

        // Cache if under limit
        if self.embedding_cache.len() < self.config.embedding_cache_size {
            self.embedding_cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    async fn get_agent_memory(&self, id: &str) -> Result<AgentMemory> {
        // Try all agent type prefixes
        for agent_type in [
            AgentType::Claude,
            AgentType::Gemini,
            AgentType::GPT4,
            AgentType::Gemma,
            AgentType::Llama,
        ] {
            let key = format!("agent_memory:{}:{}", agent_type.name(), id);
            if let Ok(data) = self.redis_manager.get_raw(&key).await {
                if let Some(data_vec) = data {
                    if let Ok(memory) = bincode::deserialize::<AgentMemory>(&data_vec) {
                        return Ok(memory);
                    }
                }
            }
        }

        Err(Error::NotFound(format!("Agent memory {} not found", id)))
    }

    async fn update_agent_memory(&self, memory: &AgentMemory) -> Result<()> {
        let serialized = bincode::serialize(memory)
            .map_err(|e| Error::Serialization(format!("Failed to serialize agent memory: {}", e)))?;

        let key = format!("agent_memory:{}:{}", memory.agent_type.name(), memory.id);
        self.redis_manager
            .set_raw(&key, &serialized)
            .await
    }

    async fn find_related_memories(&self, embedding: &[f32], limit: usize) -> Result<Vec<AgentMemory>> {
        let results = self.vector_store.read().await
            .search(embedding, limit, None)?;

        let mut memories = Vec::new();
        for (id, _, _) in results {
            if let Ok(memory) = self.get_agent_memory(&id).await {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    fn get_or_create_template(&self, agent_type: AgentType) -> MemoryTemplate {
        self.template_cache
            .entry(agent_type)
            .or_insert_with(|| MemoryTemplate::for_agent(agent_type))
            .clone()
    }

    fn is_compatible(&self, memory_type: AgentType, query_type: AgentType) -> bool {
        // Define compatibility rules
        match (memory_type, query_type) {
            // Same type is always compatible
            (a, b) if a == b => true,
            // Claude and GPT4 are somewhat compatible
            (AgentType::Claude, AgentType::GPT4) | (AgentType::GPT4, AgentType::Claude) => true,
            // Gemini and Gemma share some compatibility
            (AgentType::Gemini, AgentType::Gemma) | (AgentType::Gemma, AgentType::Gemini) => true,
            // Custom types are compatible with themselves
            (AgentType::Custom(a), AgentType::Custom(b)) if a == b => true,
            _ => false,
        }
    }

    fn generate_summary(&self, content: &str, max_length: usize) -> String {
        // Simple extractive summarization
        let sentences: Vec<&str> = content.split(". ").collect();
        let mut summary = String::new();

        for sentence in sentences {
            if summary.len() + sentence.len() > max_length {
                break;
            }
            summary.push_str(sentence);
            summary.push_str(". ");
        }

        if summary.is_empty() && !content.is_empty() {
            summary = content.chars().take(max_length).collect();
            summary.push_str("...");
        }

        summary
    }

    fn extract_key_points(&self, memories: &[(AgentMemory, f32)]) -> Vec<String> {
        let mut key_points = Vec::new();

        for (memory, _) in memories.iter().take(5) {
            // Extract from summary if available
            if let Some(summary) = &memory.summary {
                key_points.push(summary.clone());
            } else {
                // Extract first sentence or up to 100 chars
                let point = memory.content
                    .split(". ")
                    .next()
                    .unwrap_or(&memory.content)
                    .chars()
                    .take(100)
                    .collect::<String>();
                key_points.push(point);
            }
        }

        key_points
    }

    fn generate_digest_summary(&self, memories: &[(AgentMemory, f32)], topic: &Option<String>) -> String {
        let mut summary = String::new();

        if let Some(topic_str) = topic {
            summary.push_str(&format!("Memory digest for topic '{}': ", topic_str));
        } else {
            summary.push_str("General memory digest: ");
        }

        if memories.is_empty() {
            summary.push_str("No relevant memories found.");
        } else {
            summary.push_str(&format!(
                "Found {} relevant memories with average confidence {:.0}%. ",
                memories.len(),
                memories.iter().map(|(_, s)| s).sum::<f32>() / memories.len() as f32 * 100.0
            ));

            // Add top context types
            let mut context_counts = HashMap::new();
            for (memory, _) in memories {
                for hint in &memory.context_hints {
                    *context_counts.entry(hint.clone()).or_insert(0) += 1;
                }
            }

            if !context_counts.is_empty() {
                let mut contexts: Vec<_> = context_counts.into_iter().collect();
                contexts.sort_by_key(|(_, count)| -(*count as i32));

                summary.push_str("Primary contexts: ");
                for (hint, _) in contexts.iter().take(3) {
                    summary.push_str(&format!("{}, ", hint.to_prompt_prefix().trim_end()));
                }
                summary.truncate(summary.len() - 2); // Remove last ", "
                summary.push('.');
            }
        }

        summary
    }

    fn start_consolidation_task(&self) {
        let memory_manager = self.memory_manager.clone();
        let stats = self.stats.clone();
        let interval = self.config.consolidation_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;

                match memory_manager.consolidate_memories().await {
                    Ok(count) => {
                        if count > 0 {
                            info!("Consolidated {} memories", count);
                            stats.write().consolidations += 1;
                        }
                    }
                    Err(e) => {
                        error!("Memory consolidation failed: {}", e);
                    }
                }
            }
        });
    }

    /// Get system statistics
    pub fn get_stats(&self) -> AgentMemoryStats {
        self.stats.read().clone()
    }

    /// Clear caches
    pub fn clear_caches(&self) {
        self.embedding_cache.clear();
        self.digest_cache.write().clear();
        info!("Cleared agent memory caches");
    }
}

// Helper module for xmltree escape function simulation
mod xmltree {
    pub fn escape(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '<' => "&lt;".to_string(),
                '>' => "&gt;".to_string(),
                '&' => "&amp;".to_string(),
                '"' => "&quot;".to_string(),
                '\'' => "&apos;".to_string(),
                c => c.to_string(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_hint_weights() {
        assert!(ContextHint::UserPreference.weight() > ContextHint::PerformanceMetric.weight());
        assert!(ContextHint::PersonalInformation.weight() > ContextHint::SpatialContext.weight());
    }

    #[test]
    fn test_agent_type_context_windows() {
        assert_eq!(AgentType::Claude.context_window_size(), 200_000);
        assert_eq!(AgentType::Gemini.context_window_size(), 1_000_000);
        assert!(AgentType::Gemma.context_window_size() < AgentType::GPT4.context_window_size());
    }

    #[test]
    fn test_memory_template_creation() {
        let claude_template = MemoryTemplate::for_agent(AgentType::Claude);
        assert!(claude_template.include_metadata);
        assert!(claude_template.include_timestamps);

        let gemma_template = MemoryTemplate::for_agent(AgentType::Gemma);
        assert!(!gemma_template.include_metadata);
        assert!(!gemma_template.include_timestamps);
    }

    #[tokio::test]
    async fn test_agent_memory_creation() {
        let memory = AgentMemory::new(
            "Test content".to_string(),
            AgentType::Claude,
            vec![ContextHint::UserPreference],
            0.8,
        );

        assert_eq!(memory.agent_type, AgentType::Claude);
        assert_eq!(memory.importance_score, 0.8);
        assert_eq!(memory.access_count, 0);
    }

    #[test]
    fn test_relevance_score_calculation() {
        let memory = AgentMemory::new(
            "Test".to_string(),
            AgentType::Claude,
            vec![ContextHint::UserPreference, ContextHint::TaskContext],
            0.7,
        );

        let query_hints = vec![ContextHint::UserPreference];
        let score = memory.calculate_relevance_score(&query_hints, 0.2);

        assert!(score > 0.0);
        assert!(score <= 1.0);
    }
}