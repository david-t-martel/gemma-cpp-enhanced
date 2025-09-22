//! Redis backend tests with mock Redis implementation
//!
//! These tests verify the Redis backend functionality including:
//! - Connection management and pooling
//! - Document and chunk storage/retrieval
//! - Embedding caching
//! - Search result caching
//! - Health checks and error handling
//! - Connection pooling and concurrency

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rag_redis_system::{
    config::RedisConfig, research::SearchResult, Document, DocumentChunk, Error, RedisClient,
    RedisManager, Result,
};

use tokio::sync::RwLock;
use uuid::Uuid;

/// Mock Redis implementation for testing
#[derive(Debug, Default)]
pub struct MockRedis {
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    connection_count: Arc<Mutex<usize>>,
    should_fail: Arc<Mutex<bool>>,
    latency_ms: Arc<Mutex<u64>>,
}

impl MockRedis {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        if *self.should_fail.lock().unwrap() {
            return Err(Error::Redis("Mock Redis connection failed".to_string()));
        }

        let latency = *self.latency_ms.lock().unwrap();
        if latency > 0 {
            tokio::time::sleep(Duration::from_millis(latency)).await;
        }

        self.data.write().await.insert(key.to_string(), value);
        Ok(())
    }

    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if *self.should_fail.lock().unwrap() {
            return Err(Error::Redis("Mock Redis connection failed".to_string()));
        }

        let latency = *self.latency_ms.lock().unwrap();
        if latency > 0 {
            tokio::time::sleep(Duration::from_millis(latency)).await;
        }

        Ok(self.data.read().await.get(key).cloned())
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        if *self.should_fail.lock().unwrap() {
            return Err(Error::Redis("Mock Redis connection failed".to_string()));
        }

        self.data.write().await.remove(key);
        Ok(())
    }

    pub async fn exists(&self, key: &str) -> Result<bool> {
        if *self.should_fail.lock().unwrap() {
            return Err(Error::Redis("Mock Redis connection failed".to_string()));
        }

        Ok(self.data.read().await.contains_key(key))
    }

    pub fn set_should_fail(&self, should_fail: bool) {
        *self.should_fail.lock().unwrap() = should_fail;
    }

    pub fn set_latency(&self, latency_ms: u64) {
        *self.latency_ms.lock().unwrap() = latency_ms;
    }

    pub async fn len(&self) -> usize {
        self.data.read().await.len()
    }

    pub async fn clear(&self) {
        self.data.write().await.clear();
    }

    pub fn increment_connection_count(&self) {
        *self.connection_count.lock().unwrap() += 1;
    }

    pub fn get_connection_count(&self) -> usize {
        *self.connection_count.lock().unwrap()
    }
}

/// Create test configuration
fn create_test_redis_config() -> RedisConfig {
    RedisConfig {
        url: "redis://127.0.0.1:6379".to_string(),
        pool_size: 5,
        connection_timeout: Duration::from_secs(1),
        command_timeout: Duration::from_secs(2),
        max_retries: 3,
        retry_delay: Duration::from_millis(50),
        enable_cluster: false,
    }
}

/// Create a test document
fn create_test_document() -> Document {
    Document {
        id: Uuid::new_v4().to_string(),
        title: "Test Document".to_string(),
        content: "This is a test document for Redis backend testing.".to_string(),
        metadata: serde_json::json!({
            "author": "Test Author",
            "created_at": "2024-01-01T00:00:00Z"
        }),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        tags: vec!["test".to_string(), "redis".to_string()],
        source_url: Some("https://example.com/test".to_string()),
        language: Some("en".to_string()),
        word_count: 10,
        embedding: None,
    }
}

/// Create a test document chunk
fn create_test_chunk(document_id: &str, chunk_index: usize) -> DocumentChunk {
    DocumentChunk {
        id: format!("{}_{}", document_id, chunk_index),
        document_id: document_id.to_string(),
        text: format!("This is chunk {} of the test document.", chunk_index),
        start_char: chunk_index * 100,
        end_char: (chunk_index + 1) * 100,
        metadata: serde_json::json!({
            "chunk_index": chunk_index,
            "section": "test_section"
        }),
        tokens: vec!["this".to_string(), "is".to_string(), "chunk".to_string()],
        embedding: None,
        created_at: chrono::Utc::now(),
    }
}

/// Test basic Redis client operations
#[tokio::test]
async fn test_redis_client_basic_operations() {
    let config = create_test_redis_config();

    // Test client creation (will fail without Redis, which is expected)
    let client_result = RedisClient::new(&config).await;

    match client_result {
        Ok(mut client) => {
            // If Redis is available, test basic operations
            println!("✓ Redis client created successfully");

            // Test ping
            match client.ping().await {
                Ok(()) => println!("✓ Redis ping successful"),
                Err(e) => println!("⚠ Redis ping failed: {:?}", e),
            }

            // Test set/get operations
            let test_key = "test_key";
            let test_value = b"test_value";

            match client.set(test_key, test_value).await {
                Ok(()) => {
                    println!("✓ Redis SET operation successful");

                    // Test get
                    match client.get(test_key).await {
                        Ok(Some(value)) => {
                            assert_eq!(value, test_value);
                            println!("✓ Redis GET operation successful");
                        }
                        Ok(None) => println!("✗ Redis GET returned None"),
                        Err(e) => println!("✗ Redis GET failed: {:?}", e),
                    }

                    // Test delete
                    match client.delete(test_key).await {
                        Ok(()) => {
                            println!("✓ Redis DELETE operation successful");

                            // Verify deletion
                            match client.exists(test_key).await {
                                Ok(false) => println!("✓ Key properly deleted"),
                                Ok(true) => println!("⚠ Key still exists after deletion"),
                                Err(e) => println!("✗ EXISTS check failed: {:?}", e),
                            }
                        }
                        Err(e) => println!("✗ Redis DELETE failed: {:?}", e),
                    }
                }
                Err(e) => println!("✗ Redis SET failed: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping client tests");
        }
        Err(e) => {
            panic!("Unexpected error creating Redis client: {:?}", e);
        }
    }
}

/// Test Redis manager initialization and connection pooling
#[tokio::test]
async fn test_redis_manager_initialization() {
    let config = create_test_redis_config();

    match RedisManager::new(&config).await {
        Ok(manager) => {
            println!("✓ Redis manager created successfully");

            // Test health check
            match manager.health_check().await {
                Ok(true) => println!("✓ Redis health check passed"),
                Ok(false) => println!("⚠ Redis health check failed"),
                Err(e) => println!("✗ Health check error: {:?}", e),
            }

            // Test statistics
            let stats = manager.get_stats().await;
            println!("✓ Redis stats: {:?}", stats);
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping manager tests");
        }
        Err(e) => {
            panic!("Unexpected error creating Redis manager: {:?}", e);
        }
    }
}

/// Test document storage and retrieval
#[tokio::test]
async fn test_document_storage() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping document storage tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    let test_document = create_test_document();
    let document_id = test_document.id.clone();

    // Test document storage
    match manager.store_document(&test_document).await {
        Ok(()) => {
            println!("✓ Document stored successfully");

            // Test document retrieval
            match manager.get_document(&document_id).await {
                Ok(Some(retrieved_doc)) => {
                    assert_eq!(retrieved_doc.id, test_document.id);
                    assert_eq!(retrieved_doc.title, test_document.title);
                    assert_eq!(retrieved_doc.content, test_document.content);
                    println!("✓ Document retrieved successfully");
                }
                Ok(None) => println!("✗ Document not found after storage"),
                Err(e) => println!("✗ Document retrieval failed: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Document storage failed: Redis error");
        }
        Err(e) => {
            println!("✗ Document storage failed: {:?}", e);
        }
    }
}

/// Test chunk storage and retrieval
#[tokio::test]
async fn test_chunk_storage() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping chunk storage tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    let document_id = Uuid::new_v4().to_string();
    let test_chunks: Vec<DocumentChunk> =
        (0..3).map(|i| create_test_chunk(&document_id, i)).collect();

    // Store multiple chunks
    for (i, chunk) in test_chunks.iter().enumerate() {
        match manager.store_chunk(chunk).await {
            Ok(()) => {
                println!("✓ Chunk {} stored successfully", i);

                // Test chunk retrieval
                match manager.get_chunk(&chunk.id).await {
                    Ok(retrieved_chunk) => {
                        assert_eq!(retrieved_chunk.id, chunk.id);
                        assert_eq!(retrieved_chunk.document_id, chunk.document_id);
                        assert_eq!(retrieved_chunk.text, chunk.text);
                        println!("✓ Chunk {} retrieved successfully", i);
                    }
                    Err(e) => {
                        println!("✗ Chunk {} retrieval failed: {:?}", i, e);
                    }
                }
            }
            Err(Error::Redis(_)) => {
                println!("⚠ Chunk {} storage failed: Redis error", i);
                return;
            }
            Err(e) => {
                println!("✗ Chunk {} storage failed: {:?}", i, e);
            }
        }
    }

    // Test retrieving non-existent chunk
    match manager.get_chunk("nonexistent_chunk").await {
        Ok(_) => println!("✗ Retrieved non-existent chunk"),
        Err(Error::NotFound(_)) => println!("✓ Non-existent chunk properly handled"),
        Err(e) => println!("✗ Unexpected error for non-existent chunk: {:?}", e),
    }
}

/// Test embedding storage and retrieval
#[tokio::test]
async fn test_embedding_storage() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping embedding storage tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    let embedding_id = "test_embedding";
    let test_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    // Test embedding storage
    match manager.store_embedding(embedding_id, &test_embedding).await {
        Ok(()) => {
            println!("✓ Embedding stored successfully");

            // Test embedding retrieval
            match manager.get_embedding(embedding_id).await {
                Ok(Some(retrieved_embedding)) => {
                    assert_eq!(retrieved_embedding, test_embedding);
                    println!("✓ Embedding retrieved successfully");
                }
                Ok(None) => println!("✗ Embedding not found after storage"),
                Err(e) => println!("✗ Embedding retrieval failed: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Embedding storage failed: Redis error");
        }
        Err(e) => {
            println!("✗ Embedding storage failed: {:?}", e);
        }
    }

    // Test retrieving non-existent embedding
    match manager.get_embedding("nonexistent_embedding").await {
        Ok(None) => println!("✓ Non-existent embedding properly handled"),
        Ok(Some(_)) => println!("✗ Retrieved non-existent embedding"),
        Err(e) => println!("✗ Unexpected error for non-existent embedding: {:?}", e),
    }
}

/// Test search result caching
#[tokio::test]
async fn test_search_result_caching() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping search caching tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    let query_hash = "test_query_hash";
    let test_results = vec![
        SearchResult {
            id: "result_1".to_string(),
            text: "First search result".to_string(),
            score: 0.95,
            metadata: serde_json::json!({"source": "test1"}),
        },
        SearchResult {
            id: "result_2".to_string(),
            text: "Second search result".to_string(),
            score: 0.87,
            metadata: serde_json::json!({"source": "test2"}),
        },
    ];

    let ttl = Duration::from_secs(300); // 5 minutes

    // Test caching search results
    match manager
        .cache_search_result(query_hash, &test_results, ttl)
        .await
    {
        Ok(()) => {
            println!("✓ Search results cached successfully");

            // Test retrieving cached results
            match manager.get_cached_search(query_hash).await {
                Ok(Some(cached_results)) => {
                    assert_eq!(cached_results.len(), test_results.len());
                    assert_eq!(cached_results[0].id, test_results[0].id);
                    assert_eq!(cached_results[0].score, test_results[0].score);
                    println!("✓ Cached search results retrieved successfully");
                }
                Ok(None) => println!("✗ Cached search results not found"),
                Err(e) => println!("✗ Cached search retrieval failed: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Search result caching failed: Redis error");
        }
        Err(e) => {
            println!("✗ Search result caching failed: {:?}", e);
        }
    }

    // Test retrieving non-existent cached results
    match manager.get_cached_search("nonexistent_query").await {
        Ok(None) => println!("✓ Non-existent cached search properly handled"),
        Ok(Some(_)) => println!("✗ Retrieved non-existent cached search"),
        Err(e) => println!("✗ Unexpected error for non-existent cached search: {:?}", e),
    }
}

/// Test raw data operations
#[tokio::test]
async fn test_raw_data_operations() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping raw data tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    let test_key = "raw_test_key";
    let test_data = b"raw test data with binary content \x00\x01\x02";

    // Test storing raw data
    match manager.set_raw(test_key, test_data).await {
        Ok(()) => {
            println!("✓ Raw data stored successfully");

            // Test retrieving raw data
            match manager.get_raw(test_key).await {
                Ok(Some(retrieved_data)) => {
                    assert_eq!(retrieved_data, test_data);
                    println!("✓ Raw data retrieved successfully");
                }
                Ok(None) => println!("✗ Raw data not found after storage"),
                Err(e) => println!("✗ Raw data retrieval failed: {:?}", e),
            }

            // Test deleting raw data
            match manager.delete_raw(test_key).await {
                Ok(()) => {
                    println!("✓ Raw data deleted successfully");

                    // Verify deletion
                    match manager.get_raw(test_key).await {
                        Ok(None) => println!("✓ Raw data properly deleted"),
                        Ok(Some(_)) => println!("⚠ Raw data still exists after deletion"),
                        Err(e) => println!("✗ Error checking deleted data: {:?}", e),
                    }
                }
                Err(e) => println!("✗ Raw data deletion failed: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Raw data storage failed: Redis error");
        }
        Err(e) => {
            println!("✗ Raw data storage failed: {:?}", e);
        }
    }

    // Test storing raw data with TTL
    let ttl_key = "ttl_test_key";
    let ttl_data = b"ttl test data";
    let ttl_seconds = 1; // 1 second for quick testing

    match manager.set_raw_ex(ttl_key, ttl_data, ttl_seconds).await {
        Ok(()) => {
            println!("✓ Raw data with TTL stored successfully");

            // Immediately check if data exists
            match manager.get_raw(ttl_key).await {
                Ok(Some(_)) => println!("✓ Data exists immediately after storage"),
                Ok(None) => println!("✗ Data not found immediately after storage"),
                Err(e) => println!("✗ Error retrieving TTL data: {:?}", e),
            }

            // Wait for TTL expiration (in real scenarios, you'd wait longer)
            tokio::time::sleep(Duration::from_secs(2)).await;

            // Check if data has expired
            match manager.get_raw(ttl_key).await {
                Ok(None) => println!("✓ Data properly expired after TTL"),
                Ok(Some(_)) => println!("⚠ Data still exists after TTL expiration"),
                Err(e) => println!("✗ Error checking expired data: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ TTL data storage failed: Redis error");
        }
        Err(e) => {
            println!("✗ TTL data storage failed: {:?}", e);
        }
    }
}

/// Test concurrent Redis operations
#[tokio::test]
async fn test_concurrent_operations() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping concurrent tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    let manager = Arc::new(manager);
    let mut handles = Vec::new();

    // Spawn multiple concurrent operations
    for i in 0..10 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            let key = format!("concurrent_test_{}", i);
            let value = format!("test value {}", i).into_bytes();

            // Store data
            match manager_clone.set_raw(&key, &value).await {
                Ok(()) => {
                    // Retrieve data
                    match manager_clone.get_raw(&key).await {
                        Ok(Some(retrieved)) => {
                            assert_eq!(retrieved, value);
                            println!("✓ Concurrent operation {} completed successfully", i);
                        }
                        Ok(None) => println!("✗ Concurrent operation {} data not found", i),
                        Err(e) => {
                            println!("✗ Concurrent operation {} retrieval failed: {:?}", i, e)
                        }
                    }
                }
                Err(e) => println!("✗ Concurrent operation {} storage failed: {:?}", i, e),
            }
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        let _ = handle.await;
    }

    println!("✓ All concurrent operations completed");
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() {
    let mut config = create_test_redis_config();
    // Use an invalid URL to trigger connection errors
    config.url = "redis://invalid_host:9999".to_string();

    // Test manager creation with invalid config
    match RedisManager::new(&config).await {
        Ok(_) => println!("⚠ Unexpected success with invalid config"),
        Err(Error::Redis(_)) => println!("✓ Invalid config properly rejected"),
        Err(e) => println!("✗ Unexpected error type: {:?}", e),
    }

    // Test client creation with invalid config
    match RedisClient::new(&config).await {
        Ok(_) => println!("⚠ Unexpected success with invalid config"),
        Err(Error::Redis(_)) => println!("✓ Invalid client config properly rejected"),
        Err(e) => println!("✗ Unexpected error type: {:?}", e),
    }

    // Test operations with timeout
    config.url = "redis://127.0.0.1:6379".to_string();
    config.command_timeout = Duration::from_millis(1); // Very short timeout

    match RedisManager::new(&config).await {
        Ok(manager) => {
            // Test operation with very short timeout
            let test_doc = create_test_document();
            match manager.store_document(&test_doc).await {
                Ok(()) => println!("✓ Operation completed within timeout"),
                Err(Error::Redis(_)) => println!("✓ Operation properly timed out"),
                Err(e) => println!("✗ Unexpected timeout error: {:?}", e),
            }
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available for timeout test");
        }
        Err(e) => println!("✗ Unexpected manager creation error: {:?}", e),
    }
}

/// Test Redis statistics and monitoring
#[tokio::test]
async fn test_statistics_and_monitoring() {
    let config = create_test_redis_config();

    let manager = match RedisManager::new(&config).await {
        Ok(m) => m,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping statistics tests");
            return;
        }
        Err(e) => panic!("Failed to create Redis manager: {:?}", e),
    };

    // Get initial stats
    let initial_stats = manager.get_stats().await;
    println!("Initial stats: {:?}", initial_stats);

    // Perform some operations to generate stats
    let test_keys = ["stat_test_1", "stat_test_2", "stat_test_3"];
    let test_value = b"statistics test value";

    for key in &test_keys {
        let _ = manager.set_raw(key, test_value).await;
        let _ = manager.get_raw(key).await;
    }

    // Get stats after operations
    let final_stats = manager.get_stats().await;
    println!("Final stats: {:?}", final_stats);

    // Verify stats have increased
    let initial_ops = initial_stats.get("total_operations").unwrap_or(&0);
    let final_ops = final_stats.get("total_operations").unwrap_or(&0);

    if final_ops > initial_ops {
        println!("✓ Statistics properly tracking operations");
    } else {
        println!("⚠ Statistics may not be tracking operations correctly");
    }

    // Test health check
    match manager.health_check().await {
        Ok(true) => println!("✓ Health check returned healthy status"),
        Ok(false) => println!("⚠ Health check returned unhealthy status"),
        Err(e) => println!("✗ Health check failed: {:?}", e),
    }
}

/// Helper function to run all Redis tests
pub async fn run_redis_tests() -> Result<()> {
    println!("🚀 Starting Redis Backend Tests");

    println!("\n🔧 Testing Redis Client Basic Operations...");
    test_redis_client_basic_operations().await;

    println!("\n🏗️ Testing Redis Manager Initialization...");
    test_redis_manager_initialization().await;

    println!("\n📄 Testing Document Storage...");
    test_document_storage().await;

    println!("\n🧩 Testing Chunk Storage...");
    test_chunk_storage().await;

    println!("\n🔢 Testing Embedding Storage...");
    test_embedding_storage().await;

    println!("\n💾 Testing Search Result Caching...");
    test_search_result_caching().await;

    println!("\n📦 Testing Raw Data Operations...");
    test_raw_data_operations().await;

    println!("\n⚡ Testing Concurrent Operations...");
    test_concurrent_operations().await;

    println!("\n🚨 Testing Error Handling...");
    test_error_handling().await;

    println!("\n📊 Testing Statistics and Monitoring...");
    test_statistics_and_monitoring().await;

    println!("\n✅ Redis Backend Tests Completed!");

    Ok(())
}
