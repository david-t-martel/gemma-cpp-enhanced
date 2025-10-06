//! Example demonstrating the Agent Memory Embedding System
//!
//! This example shows how to:
//! - Store contextualized memories for different agent types
//! - Retrieve memories with agent-specific formatting
//! - Generate memory digests and prompts
//! - Work with context hints and embeddings

use rag_redis_system::{
    agent_memory::{
        AgentMemorySystem, AgentType, ContextHint, AgentMemoryConfig,
    },
    redis_backend::{RedisManager, RedisConfig},
    Result,
};
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting Agent Memory System demo");

    // Configure Redis connection
    let redis_config = RedisConfig {
        url: "redis://127.0.0.1:6380".to_string(), // Windows port
        max_connections: 10,
        connection_timeout: std::time::Duration::from_secs(5),
        operation_timeout: std::time::Duration::from_secs(2),
        retry_count: 3,
        retry_delay: std::time::Duration::from_millis(100),
    };

    // Initialize Redis manager
    let redis_manager = Arc::new(RedisManager::new(&redis_config).await?);

    // Configure agent memory system
    let agent_config = AgentMemoryConfig {
        embedding_cache_size: 1000,
        context_window_buffer: 0.8,
        similarity_threshold: 0.7,
        max_related_memories: 5,
        digest_summary_length: 500,
        auto_consolidate: true,
        consolidation_interval: std::time::Duration::from_secs(3600),
    };

    // Create agent memory system
    let agent_memory = AgentMemorySystem::new(redis_manager, agent_config).await?;
    info!("Agent memory system initialized");

    // Example 1: Store contextualized memories for Claude
    info!("\n=== Storing Claude Agent Memories ===");

    let claude_memories = vec![
        (
            "The user prefers dark mode interfaces with high contrast for better readability",
            vec![ContextHint::UserPreference, ContextHint::InterfaceSettings],
            0.9,
        ),
        (
            "User's primary programming language is Rust, with experience in Python and TypeScript",
            vec![ContextHint::DomainKnowledge, ContextHint::PersonalInformation],
            0.85,
        ),
        (
            "Previous conversation discussed implementing a RAG system with Redis backend",
            vec![ContextHint::ConversationHistory, ContextHint::TaskContext],
            0.75,
        ),
        (
            "User values code performance and memory efficiency in system design",
            vec![ContextHint::UserPreference, ContextHint::PerformanceMetric],
            0.8,
        ),
    ];

    let mut claude_memory_ids = Vec::new();
    for (content, hints, importance) in claude_memories {
        let memory_id = agent_memory.store_contextualized(
            content.to_string(),
            AgentType::Claude,
            hints,
            importance,
        ).await?;
        claude_memory_ids.push(memory_id.clone());
        info!("Stored Claude memory: {}", memory_id);
    }

    // Example 2: Store memories for Gemini
    info!("\n=== Storing Gemini Agent Memories ===");

    let gemini_memories = vec![
        (
            "System architecture uses microservices with Redis for caching and message queuing",
            vec![ContextHint::SystemConfiguration, ContextHint::DomainKnowledge],
            0.85,
        ),
        (
            "Performance benchmarks show 10x improvement with SIMD optimizations",
            vec![ContextHint::PerformanceMetric, ContextHint::LearningOutcome],
            0.7,
        ),
    ];

    for (content, hints, importance) in gemini_memories {
        let memory_id = agent_memory.store_contextualized(
            content.to_string(),
            AgentType::Gemini,
            hints,
            importance,
        ).await?;
        info!("Stored Gemini memory: {}", memory_id);
    }

    // Example 3: Store memories for Gemma (lightweight model)
    info!("\n=== Storing Gemma Agent Memories ===");

    let gemma_memory_id = agent_memory.store_contextualized(
        "Simple task: implement vector search functionality".to_string(),
        AgentType::Gemma,
        vec![ContextHint::TaskContext],
        0.6,
    ).await?;
    info!("Stored Gemma memory: {}", gemma_memory_id);

    // Wait a moment for indexing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Example 4: Retrieve memories for Claude with formatting
    info!("\n=== Retrieving Memories for Claude ===");

    let claude_query = "What are the user's preferences and technical background?";
    let claude_results = agent_memory.retrieve_for_agent(
        claude_query,
        AgentType::Claude,
        5,
    ).await?;

    info!("Query: {}", claude_query);
    info!("Retrieved {} formatted memories:", claude_results.len());
    for (i, memory) in claude_results.iter().enumerate() {
        info!("\n--- Memory {} ---\n{}", i + 1, memory);
    }

    // Example 5: Generate memory digest for Gemini
    info!("\n=== Generating Memory Digest for Gemini ===");

    let digest = agent_memory.generate_digest(
        AgentType::Gemini,
        Some("system architecture and performance".to_string()),
        10,
    ).await?;

    info!("Digest Summary: {}", digest.summary);
    info!("Key Points:");
    for point in &digest.key_points {
        info!("  • {}", point);
    }
    info!("Confidence Score: {:.0}%", digest.confidence_score * 100.0);
    info!("Total Memories Considered: {}", digest.total_memories_considered);

    // Example 6: Create memory prompt for Claude
    info!("\n=== Creating Memory Prompt for Claude ===");

    let context_hints = vec![
        ContextHint::UserPreference,
        ContextHint::DomainKnowledge,
        ContextHint::ConversationHistory,
    ];

    let prompt = agent_memory.create_memory_prompt(
        AgentType::Claude,
        context_hints,
        1000, // Max tokens
    ).await?;

    info!("Generated Prompt:\n{}", prompt);

    // Example 7: Retrieve memories for different agent types
    info!("\n=== Cross-Agent Memory Retrieval ===");

    let technical_query = "What programming languages and technologies are being used?";

    info!("\nQuerying as Claude:");
    let claude_tech = agent_memory.retrieve_for_agent(
        technical_query,
        AgentType::Claude,
        3,
    ).await?;
    info!("Found {} memories", claude_tech.len());

    info!("\nQuerying as Gemini:");
    let gemini_tech = agent_memory.retrieve_for_agent(
        technical_query,
        AgentType::Gemini,
        3,
    ).await?;
    info!("Found {} memories", gemini_tech.len());

    info!("\nQuerying as Gemma:");
    let gemma_tech = agent_memory.retrieve_for_agent(
        technical_query,
        AgentType::Gemma,
        3,
    ).await?;
    info!("Found {} memories", gemma_tech.len());

    // Example 8: Store and retrieve with custom metadata
    info!("\n=== Working with Metadata ===");

    let mut memory_with_metadata = agent_memory.store_contextualized(
        "Project deadline is December 31, 2024, with weekly progress reviews".to_string(),
        AgentType::Claude,
        vec![ContextHint::TemporalContext, ContextHint::TaskContext],
        0.95,
    ).await?;

    info!("Stored memory with high importance: {}", memory_with_metadata);

    // Example 9: Generate digest without specific topic
    info!("\n=== General Memory Digest ===");

    let general_digest = agent_memory.generate_digest(
        AgentType::Claude,
        None, // No specific topic
        10,
    ).await?;

    info!("General Digest Summary: {}", general_digest.summary);
    info!("Found {} relevant memories", general_digest.relevant_memories.len());

    // Example 10: Display statistics
    info!("\n=== Agent Memory System Statistics ===");

    let stats = agent_memory.get_stats();
    info!("Total Stores: {}", stats.total_stores);
    info!("Total Retrievals: {}", stats.total_retrievals);
    info!("Cache Hits: {}", stats.cache_hits);
    info!("Cache Misses: {}", stats.cache_misses);
    info!("Digest Generations: {}", stats.digest_generations);
    info!("Memory Consolidations: {}", stats.consolidations);

    // Example 11: Test different agent type configurations
    info!("\n=== Agent Type Configurations ===");

    for agent_type in [
        AgentType::Claude,
        AgentType::Gemini,
        AgentType::GPT4,
        AgentType::Gemma,
        AgentType::Llama,
    ] {
        info!("\n{} Configuration:", agent_type.name());
        info!("  Context Window: {} tokens", agent_type.context_window_size());
        info!("  Optimal Chunk Size: {} tokens", agent_type.optimal_chunk_size());
    }

    // Example 12: Clear caches for fresh retrieval
    info!("\n=== Cache Management ===");

    agent_memory.clear_caches();
    info!("Cleared all caches");

    // Retrieve again to test cache miss
    let _ = agent_memory.retrieve_for_agent(
        "test query",
        AgentType::Claude,
        1,
    ).await?;

    let final_stats = agent_memory.get_stats();
    info!("Cache misses after clear: {}", final_stats.cache_misses);

    info!("\n=== Demo completed successfully ===");
    Ok(())
}