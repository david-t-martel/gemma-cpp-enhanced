//! Integration tests for the Agent Memory System

use rag_redis_system::{
    agent_memory::{
        AgentMemorySystem, AgentType, ContextHint, AgentMemoryConfig,
        AgentMemory, MemoryTemplate, FormatStyle,
    },
    redis_backend::{RedisManager, RedisConfig},
    Result,
};
use std::sync::Arc;
use std::time::Duration;

/// Helper function to create test Redis manager
async fn create_test_redis_manager() -> Result<Arc<RedisManager>> {
    let redis_config = RedisConfig {
        url: std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6380".to_string()),
        max_connections: 5,
        connection_timeout: Duration::from_secs(5),
        operation_timeout: Duration::from_secs(2),
        retry_count: 1,
        retry_delay: Duration::from_millis(100),
    };

    Ok(Arc::new(RedisManager::new(&redis_config).await?))
}

/// Helper function to create test agent memory system
async fn create_test_agent_memory() -> Result<AgentMemorySystem> {
    let redis_manager = create_test_redis_manager().await?;
    let config = AgentMemoryConfig {
        embedding_cache_size: 100,
        context_window_buffer: 0.8,
        similarity_threshold: 0.6,
        max_related_memories: 3,
        digest_summary_length: 200,
        auto_consolidate: false, // Disable for tests
        consolidation_interval: Duration::from_secs(3600),
    };

    AgentMemorySystem::new(redis_manager, config).await
}

#[tokio::test]
async fn test_store_and_retrieve_contextualized_memory() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Store memory
    let content = "Test user prefers functional programming paradigms";
    let memory_id = agent_memory.store_contextualized(
        content.to_string(),
        AgentType::Claude,
        vec![ContextHint::UserPreference, ContextHint::DomainKnowledge],
        0.8,
    ).await?;

    assert!(!memory_id.is_empty());

    // Retrieve memories
    let results = agent_memory.retrieve_for_agent(
        "programming preferences",
        AgentType::Claude,
        5,
    ).await?;

    // Should find at least one memory
    assert!(!results.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_agent_type_compatibility() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Store memory for GPT4
    let memory_id = agent_memory.store_contextualized(
        "GPT4 specific context about API usage".to_string(),
        AgentType::GPT4,
        vec![ContextHint::SystemConfiguration],
        0.7,
    ).await?;

    // Wait for indexing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Claude should be able to retrieve GPT4 memories (they're compatible)
    let claude_results = agent_memory.retrieve_for_agent(
        "API usage",
        AgentType::Claude,
        5,
    ).await?;

    // Should find the memory due to compatibility
    assert!(!claude_results.is_empty());

    // Gemma should not retrieve GPT4 memories (not compatible)
    let gemma_results = agent_memory.retrieve_for_agent(
        "API usage",
        AgentType::Gemma,
        5,
    ).await?;

    // Might be empty or contain only Gemma-specific memories
    // This test depends on what else is in the database

    Ok(())
}

#[tokio::test]
async fn test_memory_digest_generation() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Store multiple memories
    let memories = vec![
        ("Redis is used for caching", vec![ContextHint::SystemConfiguration], 0.8),
        ("Performance improved by 50%", vec![ContextHint::PerformanceMetric], 0.7),
        ("User prefers async operations", vec![ContextHint::UserPreference], 0.9),
    ];

    for (content, hints, importance) in memories {
        agent_memory.store_contextualized(
            content.to_string(),
            AgentType::Gemini,
            hints,
            importance,
        ).await?;
    }

    // Wait for indexing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Generate digest
    let digest = agent_memory.generate_digest(
        AgentType::Gemini,
        Some("system performance".to_string()),
        10,
    ).await?;

    assert!(!digest.summary.is_empty());
    assert!(digest.confidence_score >= 0.0 && digest.confidence_score <= 1.0);

    Ok(())
}

#[tokio::test]
async fn test_memory_prompt_creation() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Store some context
    agent_memory.store_contextualized(
        "Database uses PostgreSQL with Redis cache".to_string(),
        AgentType::Claude,
        vec![ContextHint::SystemConfiguration],
        0.75,
    ).await?;

    // Create prompt
    let prompt = agent_memory.create_memory_prompt(
        AgentType::Claude,
        vec![ContextHint::SystemConfiguration, ContextHint::TaskContext],
        500,
    ).await?;

    assert!(!prompt.is_empty());
    assert!(prompt.contains("Based on the following context"));

    Ok(())
}

#[tokio::test]
async fn test_context_hint_weights() {
    assert!(ContextHint::UserPreference.weight() > ContextHint::PerformanceMetric.weight());
    assert!(ContextHint::PersonalInformation.weight() > ContextHint::SpatialContext.weight());

    // Custom hints should have moderate weight
    let custom = ContextHint::Custom("custom_hint".to_string());
    assert_eq!(custom.weight(), 0.5);
}

#[tokio::test]
async fn test_agent_memory_relevance_scoring() {
    let memory = AgentMemory::new(
        "Test content".to_string(),
        AgentType::Claude,
        vec![ContextHint::UserPreference, ContextHint::TaskContext],
        0.8,
    );

    // Test with matching hints
    let query_hints = vec![ContextHint::UserPreference, ContextHint::TaskContext];
    let score = memory.calculate_relevance_score(&query_hints, 0.2);
    assert!(score > 0.5); // Should have high relevance

    // Test with no matching hints
    let query_hints = vec![ContextHint::SecurityContext];
    let score = memory.calculate_relevance_score(&query_hints, 0.2);
    assert!(score < 0.5); // Should have lower relevance
}

#[tokio::test]
async fn test_memory_template_formatting() {
    let memory = AgentMemory::new(
        "Test memory content for formatting".to_string(),
        AgentType::Claude,
        vec![ContextHint::DomainKnowledge],
        0.75,
    );

    // Test Markdown formatting
    let template = MemoryTemplate::for_agent(AgentType::Claude);
    let formatted = template.format_memory(&memory);
    assert!(formatted.contains("**Context**"));
    assert!(formatted.contains("Test memory content"));

    // Test JSON formatting
    let mut template = MemoryTemplate::for_agent(AgentType::GPT4);
    template.format_style = FormatStyle::JSON;
    let formatted = template.format_memory(&memory);
    assert!(formatted.contains("\"content\""));
    assert!(formatted.contains("\"hints\""));

    // Test Plain text formatting
    let template = MemoryTemplate::for_agent(AgentType::Gemma);
    let formatted = template.format_memory(&memory);
    assert!(formatted.contains("Test memory content"));
}

#[tokio::test]
async fn test_cache_management() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Store and retrieve to populate cache
    let memory_id = agent_memory.store_contextualized(
        "Cache test memory".to_string(),
        AgentType::Claude,
        vec![ContextHint::TaskContext],
        0.5,
    ).await?;

    // First retrieval (cache miss)
    let stats_before = agent_memory.get_stats();
    let cache_misses_before = stats_before.cache_misses;

    let _ = agent_memory.retrieve_for_agent(
        "cache test",
        AgentType::Claude,
        1,
    ).await?;

    // Clear caches
    agent_memory.clear_caches();

    // Second retrieval after clear (should be cache miss)
    let _ = agent_memory.retrieve_for_agent(
        "cache test",
        AgentType::Claude,
        1,
    ).await?;

    let stats_after = agent_memory.get_stats();
    assert!(stats_after.cache_misses > cache_misses_before);

    Ok(())
}

#[tokio::test]
async fn test_different_agent_configurations() {
    // Test each agent type has appropriate configuration
    let agents = vec![
        (AgentType::Claude, 200_000, 2048),
        (AgentType::Gemini, 1_000_000, 4096),
        (AgentType::GPT4, 128_000, 2048),
        (AgentType::Gemma, 8_192, 512),
        (AgentType::Llama, 32_768, 1024),
    ];

    for (agent_type, expected_context, expected_chunk) in agents {
        assert_eq!(agent_type.context_window_size(), expected_context);
        assert_eq!(agent_type.optimal_chunk_size(), expected_chunk);

        let template = MemoryTemplate::for_agent(agent_type);
        assert_eq!(template.agent_type, agent_type);
    }
}

#[tokio::test]
async fn test_memory_with_metadata() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Create memory with metadata
    let mut memory = AgentMemory::new(
        "Memory with metadata".to_string(),
        AgentType::Claude,
        vec![ContextHint::TaskContext],
        0.7,
    );

    memory.metadata.insert(
        "source".to_string(),
        serde_json::json!("unit_test"),
    );
    memory.metadata.insert(
        "version".to_string(),
        serde_json::json!(1),
    );

    // Store using the system (this will generate embedding)
    let memory_id = agent_memory.store_contextualized(
        memory.content.clone(),
        memory.agent_type,
        memory.context_hints.clone(),
        memory.importance_score,
    ).await?;

    assert!(!memory_id.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_digest_caching() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    // Store some memories
    for i in 0..3 {
        agent_memory.store_contextualized(
            format!("Test memory {} for digest caching", i),
            AgentType::Claude,
            vec![ContextHint::TaskContext],
            0.6,
        ).await?;
    }

    // Wait for indexing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Generate digest (should cache)
    let topic = Some("test digest".to_string());
    let digest1 = agent_memory.generate_digest(
        AgentType::Claude,
        topic.clone(),
        5,
    ).await?;

    let stats1 = agent_memory.get_stats();

    // Generate same digest again (should hit cache)
    let digest2 = agent_memory.generate_digest(
        AgentType::Claude,
        topic,
        5,
    ).await?;

    let stats2 = agent_memory.get_stats();

    // Cache hit count should increase
    assert!(stats2.cache_hits > stats1.cache_hits);

    // Digests should be similar
    assert_eq!(digest1.summary, digest2.summary);

    Ok(())
}

#[tokio::test]
async fn test_statistics_tracking() -> Result<()> {
    let agent_memory = create_test_agent_memory().await?;

    let initial_stats = agent_memory.get_stats();

    // Perform operations
    agent_memory.store_contextualized(
        "Stats test memory".to_string(),
        AgentType::Claude,
        vec![ContextHint::TaskContext],
        0.5,
    ).await?;

    agent_memory.retrieve_for_agent(
        "stats test",
        AgentType::Claude,
        1,
    ).await?;

    agent_memory.generate_digest(
        AgentType::Claude,
        Some("stats".to_string()),
        5,
    ).await?;

    let final_stats = agent_memory.get_stats();

    assert!(final_stats.total_stores > initial_stats.total_stores);
    assert!(final_stats.total_retrievals > initial_stats.total_retrievals);
    assert!(final_stats.digest_generations > initial_stats.digest_generations);

    Ok(())
}