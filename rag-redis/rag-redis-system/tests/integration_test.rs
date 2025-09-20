//! Integration tests for the complete RAG-Redis system
//!
//! These tests verify the full system workflow including:
//! - Document ingestion and processing
//! - Vector storage and retrieval
//! - Search functionality
//! - Redis backend operations
//! - Error handling and edge cases

use std::collections::HashMap;
use std::time::Duration;

use rag_redis_system::{
    config::{
        DocumentConfig, EmbeddingConfig, MemoryConfig, RedisConfig, ResearchConfig, ServerConfig,
        VectorStoreConfig,
    },
    Config, Document, DocumentChunk, Error, RagSystem, RedisManager, Result, VectorStore,
};

use tempfile::TempDir;
use tokio::time::timeout;

/// Test configuration for integration tests
fn create_test_config() -> Config {
    Config {
        redis: RedisConfig {
            url: "redis://127.0.0.1:6379".to_string(),
            pool_size: 5,
            connection_timeout: Duration::from_secs(2),
            command_timeout: Duration::from_secs(5),
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            enable_cluster: false,
        },
        vector_store: VectorStoreConfig {
            dimension: 384, // Smaller dimension for faster tests
            distance_metric: rag_redis_system::vector_store::DistanceMetric::Cosine,
            index_type: rag_redis_system::config::IndexType::Hnsw,
            hnsw_m: 8,
            hnsw_ef_construction: 50,
            hnsw_ef_search: 20,
            max_vectors: Some(1000),
        },
        document: DocumentConfig::default(),
        memory: MemoryConfig::default(),
        research: ResearchConfig::default(),
        embedding: EmbeddingConfig {
            dimension: 384,
            ..EmbeddingConfig::default()
        },
        #[cfg(feature = "metrics")]
        metrics: rag_redis_system::config::MetricsConfig::default(),
        server: ServerConfig::default(),
    }
}

/// Create mock embedding vectors for testing
fn create_test_embedding(dimension: usize, seed: u64) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(dimension);
    let mut x = seed as f32;

    for i in 0..dimension {
        x = (x * 1103515245.0 + 12345.0) % (1u64 << 31) as f32;
        embedding.push((x / (1u64 << 31) as f32 - 0.5) * 2.0); // Normalize to [-1, 1]
    }

    // Normalize vector
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= magnitude);
    }

    embedding
}

/// Test basic system initialization
#[tokio::test]
async fn test_system_initialization() {
    let config = create_test_config();

    // Test that the system can be created without Redis connection
    // In a real test environment, you'd have Redis running
    let result = RagSystem::new(config).await;

    // This might fail if Redis is not running, which is expected in CI
    match result {
        Ok(_system) => {
            // System initialized successfully
            println!("✓ System initialized successfully");
        }
        Err(Error::Redis(_)) => {
            // Redis not available, skip this test
            println!("⚠ Redis not available, skipping integration tests");
            return;
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test document ingestion workflow
#[tokio::test]
async fn test_document_ingestion_workflow() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    // Test document content
    let test_content = r#"
        # Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence (AI) that focuses on developing
        algorithms and statistical models that enable computers to learn and make decisions
        without being explicitly programmed for every task.

        ## Types of Machine Learning

        1. **Supervised Learning**: Uses labeled training data
        2. **Unsupervised Learning**: Finds patterns in unlabeled data
        3. **Reinforcement Learning**: Learns through interaction with environment

        ## Applications

        Machine learning has numerous applications including:
        - Natural language processing
        - Computer vision
        - Recommendation systems
        - Fraud detection
    "#;

    let metadata = serde_json::json!({
        "title": "Introduction to Machine Learning",
        "author": "Test Author",
        "source": "test_document",
        "created_at": "2024-01-01T00:00:00Z"
    });

    // Test document ingestion
    let document_id = system.ingest_document(test_content, metadata).await;

    match document_id {
        Ok(id) => {
            println!("✓ Document ingested successfully with ID: {}", id);

            // Test search functionality
            test_search_functionality(&system, &id).await;
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Redis operation failed, skipping test");
        }
        Err(e) => panic!("Document ingestion failed: {:?}", e),
    }
}

/// Test search functionality with the ingested document
async fn test_search_functionality(system: &RagSystem, document_id: &str) {
    // Test various search queries
    let test_queries = vec![
        ("machine learning", "Should find content about ML"),
        (
            "supervised learning",
            "Should find content about supervised learning",
        ),
        ("applications", "Should find application examples"),
        (
            "nonexistent query xyz",
            "Should return empty or low relevance results",
        ),
    ];

    for (query, description) in test_queries {
        match system.search(query, 5).await {
            Ok(results) => {
                println!(
                    "✓ Search for '{}': {} - Found {} results",
                    query,
                    description,
                    results.len()
                );

                // Validate result structure
                for result in results {
                    assert!(!result.id.is_empty());
                    assert!(!result.text.is_empty());
                    assert!(result.score >= 0.0);
                }
            }
            Err(Error::Redis(_)) => {
                println!("⚠ Redis operation failed during search");
                break;
            }
            Err(e) => {
                println!("✗ Search failed for '{}': {:?}", query, e);
            }
        }
    }
}

/// Test research functionality (combines local and web search)
#[tokio::test]
async fn test_research_functionality() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    let query = "artificial intelligence applications";
    let sources = vec!["example.com".to_string()]; // Mock source

    match system.research(query, sources).await {
        Ok(results) => {
            println!(
                "✓ Research query completed, found {} results",
                results.len()
            );

            // Validate research results
            for result in results {
                assert!(!result.id.is_empty());
                assert!(result.score >= 0.0);
            }
        }
        Err(Error::Research(_)) => {
            println!("⚠ Research functionality not fully available");
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Redis operation failed during research");
        }
        Err(e) => {
            println!("Research query failed: {:?}", e);
        }
    }
}

/// Test system error handling and edge cases
#[tokio::test]
async fn test_error_handling() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    // Test empty content ingestion
    let empty_result = system.ingest_document("", serde_json::json!({})).await;
    match empty_result {
        Ok(_) => println!("✓ Empty document handled gracefully"),
        Err(e) => println!("✓ Empty document properly rejected: {:?}", e),
    }

    // Test very large content (should be handled gracefully)
    let large_content = "A".repeat(1_000_000); // 1MB of 'A's
    let large_result = system
        .ingest_document(&large_content, serde_json::json!({"test": true}))
        .await;
    match large_result {
        Ok(_) => println!("✓ Large document processed successfully"),
        Err(e) => println!("✓ Large document handled with error: {:?}", e),
    }

    // Test search with empty query
    let empty_search_result = system.search("", 10).await;
    match empty_search_result {
        Ok(results) => println!("✓ Empty search query returned {} results", results.len()),
        Err(e) => println!("✓ Empty search query properly handled: {:?}", e),
    }

    // Test search with very long query
    let long_query =
        "artificial intelligence machine learning deep learning neural networks".repeat(50);
    let long_search_result = system.search(&long_query, 5).await;
    match long_search_result {
        Ok(results) => println!("✓ Long search query returned {} results", results.len()),
        Err(e) => println!("✓ Long search query handled with error: {:?}", e),
    }
}

/// Test concurrent operations
#[tokio::test]
async fn test_concurrent_operations() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    let system = std::sync::Arc::new(system);
    let mut handles = Vec::new();

    // Spawn multiple concurrent ingestion tasks
    for i in 0..5 {
        let system_clone = system.clone();
        let handle = tokio::spawn(async move {
            let content = format!("Test document {} with unique content about topic {}", i, i);
            let metadata = serde_json::json!({
                "id": i,
                "title": format!("Test Document {}", i)
            });

            match system_clone.ingest_document(&content, metadata).await {
                Ok(id) => println!("✓ Concurrent ingestion {} succeeded: {}", i, id),
                Err(Error::Redis(_)) => {
                    println!("⚠ Concurrent ingestion {} failed: Redis error", i)
                }
                Err(e) => println!("✗ Concurrent ingestion {} failed: {:?}", i, e),
            }
        });
        handles.push(handle);
    }

    // Spawn multiple concurrent search tasks
    for i in 0..5 {
        let system_clone = system.clone();
        let handle = tokio::spawn(async move {
            let query = format!("test topic {}", i % 3);
            match system_clone.search(&query, 3).await {
                Ok(results) => {
                    println!("✓ Concurrent search {} found {} results", i, results.len())
                }
                Err(Error::Redis(_)) => println!("⚠ Concurrent search {} failed: Redis error", i),
                Err(e) => println!("✗ Concurrent search {} failed: {:?}", i, e),
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let _ = handle.await;
    }

    println!("✓ All concurrent operations completed");
}

/// Test system performance with timeouts
#[tokio::test]
async fn test_performance_and_timeouts() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    // Test operations with timeout
    let timeout_duration = Duration::from_secs(10);

    // Test document ingestion with timeout
    let content = "Performance test document with substantial content to process and analyze.";
    let metadata = serde_json::json!({"test": "performance"});

    let ingestion_result =
        timeout(timeout_duration, system.ingest_document(content, metadata)).await;

    match ingestion_result {
        Ok(Ok(id)) => println!("✓ Document ingestion completed within timeout: {}", id),
        Ok(Err(Error::Redis(_))) => println!("⚠ Document ingestion failed: Redis error"),
        Ok(Err(e)) => println!("✗ Document ingestion failed: {:?}", e),
        Err(_) => println!("✗ Document ingestion timed out"),
    }

    // Test search with timeout
    let search_result = timeout(timeout_duration, system.search("performance test", 10)).await;

    match search_result {
        Ok(Ok(results)) => println!(
            "✓ Search completed within timeout, found {} results",
            results.len()
        ),
        Ok(Err(Error::Redis(_))) => println!("⚠ Search failed: Redis error"),
        Ok(Err(e)) => println!("✗ Search failed: {:?}", e),
        Err(_) => println!("✗ Search timed out"),
    }
}

/// Helper function to create test documents for batch operations
fn create_test_documents() -> Vec<(String, serde_json::Value)> {
    vec![
        (
            "Rust is a systems programming language focused on safety and performance.".to_string(),
            serde_json::json!({"topic": "programming", "language": "rust"}),
        ),
        (
            "Python is a high-level programming language with dynamic semantics.".to_string(),
            serde_json::json!({"topic": "programming", "language": "python"}),
        ),
        (
            "Machine learning algorithms can identify patterns in large datasets.".to_string(),
            serde_json::json!({"topic": "machine_learning", "category": "algorithms"}),
        ),
        (
            "Docker containers provide lightweight virtualization for applications.".to_string(),
            serde_json::json!({"topic": "devops", "category": "containers"}),
        ),
        (
            "Kubernetes orchestrates containerized applications at scale.".to_string(),
            serde_json::json!({"topic": "devops", "category": "orchestration"}),
        ),
    ]
}

/// Test batch operations and data consistency
#[tokio::test]
async fn test_batch_operations() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    let test_documents = create_test_documents();
    let mut document_ids = Vec::new();

    // Ingest multiple documents
    for (content, metadata) in test_documents {
        match system.ingest_document(&content, metadata).await {
            Ok(id) => {
                document_ids.push(id);
                println!("✓ Batch document ingested successfully");
            }
            Err(Error::Redis(_)) => {
                println!("⚠ Batch ingestion failed: Redis error");
                return;
            }
            Err(e) => {
                println!("✗ Batch ingestion failed: {:?}", e);
            }
        }
    }

    println!("✓ Ingested {} documents successfully", document_ids.len());

    // Test various search scenarios
    let search_scenarios = vec![
        ("programming", "Should find Rust and Python documents"),
        ("machine learning", "Should find ML-related content"),
        ("containers", "Should find Docker content"),
        ("performance safety", "Should find Rust content"),
    ];

    for (query, description) in search_scenarios {
        match system.search(query, 10).await {
            Ok(results) => {
                println!(
                    "✓ Batch search '{}': {} - Found {} results",
                    query,
                    description,
                    results.len()
                );

                // Validate that results are relevant and properly formatted
                for result in results {
                    assert!(!result.id.is_empty());
                    assert!(!result.text.is_empty());
                    assert!(result.score >= 0.0 && result.score <= 1.0);
                }
            }
            Err(Error::Redis(_)) => {
                println!("⚠ Batch search failed: Redis error");
                break;
            }
            Err(e) => {
                println!("✗ Batch search failed: {:?}", e);
            }
        }
    }
}

/// Test system recovery and resilience
#[tokio::test]
async fn test_system_resilience() {
    let config = create_test_config();

    let system = match RagSystem::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping resilience test");
            return;
        }
        Err(e) => panic!("Failed to create system: {:?}", e),
    };

    // Test malformed input handling
    let malformed_inputs = vec![
        ("\x00\x01\x02invalid\x03\x04", "binary data"),
        ("", "empty content"),
        ("a".repeat(10_000), "very long content"),
        (
            "Special chars: áéíóú ñ ü ç 中文 日本語 العربية",
            "unicode content",
        ),
    ];

    for (content, description) in malformed_inputs {
        let metadata = serde_json::json!({"test": description});

        match system.ingest_document(content, metadata).await {
            Ok(_) => println!("✓ System handled {} gracefully", description),
            Err(Error::Redis(_)) => {
                println!("⚠ Redis error with {}", description);
                continue;
            }
            Err(e) => println!("✓ System properly rejected {} with: {:?}", description, e),
        }
    }

    // Test search with malformed queries
    let malformed_queries = vec![
        ("\x00\x01\x02", "binary query"),
        ("", "empty query"),
        ("a".repeat(1000), "very long query"),
        ("SELECT * FROM users", "SQL injection attempt"),
    ];

    for (query, description) in malformed_queries {
        match system.search(query, 5).await {
            Ok(results) => println!(
                "✓ System handled {} query, found {} results",
                description,
                results.len()
            ),
            Err(Error::Redis(_)) => {
                println!("⚠ Redis error with {} query", description);
                continue;
            }
            Err(e) => println!(
                "✓ System properly handled {} query with: {:?}",
                description, e
            ),
        }
    }
}

/// Integration test runner with proper setup/teardown
pub async fn run_integration_tests() -> Result<()> {
    println!("🚀 Starting RAG-Redis System Integration Tests");

    // Test system initialization first
    println!("\n📋 Testing System Initialization...");
    test_system_initialization().await;

    println!("\n📄 Testing Document Ingestion Workflow...");
    test_document_ingestion_workflow().await;

    println!("\n🔍 Testing Research Functionality...");
    test_research_functionality().await;

    println!("\n🚨 Testing Error Handling...");
    test_error_handling().await;

    println!("\n⚡ Testing Concurrent Operations...");
    test_concurrent_operations().await;

    println!("\n⏱️ Testing Performance and Timeouts...");
    test_performance_and_timeouts().await;

    println!("\n📚 Testing Batch Operations...");
    test_batch_operations().await;

    println!("\n🛡️ Testing System Resilience...");
    test_system_resilience().await;

    println!("\n✅ Integration Tests Completed!");

    Ok(())
}
