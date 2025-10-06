/// Integration tests for RAG-Redis MCP Server
/// These tests require Redis to be running and perform real operations
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;

/// Simple Redis health check
async fn check_redis_health() -> bool {
    // Try to connect to Redis using redis-cli
    let output = Command::new("redis-cli")
        .arg("ping")
        .output();

    match output {
        Ok(output) => {
            let response = String::from_utf8_lossy(&output.stdout);
            response.trim() == "PONG"
        }
        Err(_) => false,
    }
}

/// Start Redis if not running
async fn ensure_redis_running() -> Result<(), String> {
    if check_redis_health().await {
        println!("✅ Redis is already running");
        return Ok(());
    }

    println!("🚀 Starting Redis server...");

    // Try to start Redis
    let _process = Command::new("redis-server")
        .arg("--daemonize")
        .arg("yes")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start Redis: {}. Please ensure Redis is installed.", e))?;

    // Wait for Redis to start
    for _ in 0..30 {
        if check_redis_health().await {
            println!("✅ Redis started successfully");
            return Ok(());
        }
        sleep(Duration::from_millis(100)).await;
    }

    Err("Redis failed to start within timeout".to_string())
}

#[tokio::test]
async fn test_redis_connection() {
    ensure_redis_running().await.expect("Failed to start Redis");
    assert!(check_redis_health().await, "Redis should be running");
}

#[tokio::test]
async fn test_mcp_server_creation() {
    // This test verifies that we can create the MCP server components
    // without actually starting the full server

    ensure_redis_running().await.expect("Failed to start Redis");

    // Test that we can create a mock RAG system
    let config = rag_redis_mcp_server::mock_rag::Config {
        redis_url: "redis://localhost:6379/9".to_string(),
        embedding_model: "test-model".to_string(),
        chunk_size: 256,
    };

    let rag_system = rag_redis_mcp_server::mock_rag::RagSystem::new(config).await;
    assert!(rag_system.is_ok(), "Should be able to create RAG system");
}

#[tokio::test]
async fn test_document_operations() {
    ensure_redis_running().await.expect("Failed to start Redis");

    let config = rag_redis_mcp_server::mock_rag::Config {
        redis_url: "redis://localhost:6379/9".to_string(),
        embedding_model: "test-model".to_string(),
        chunk_size: 256,
    };

    let rag_system = rag_redis_mcp_server::mock_rag::RagSystem::new(config).await
        .expect("Failed to create RAG system");

    // Test document ingestion
    let test_content = "This is a test document about Rust programming language features.";
    let metadata = serde_json::json!({
        "title": "Test Document",
        "category": "test"
    });

    let doc_id = rag_system.ingest_document(test_content, metadata).await
        .expect("Failed to ingest document");

    assert!(!doc_id.is_empty(), "Document ID should not be empty");

    // Test document retrieval
    let retrieved_doc = rag_system.get_document(&doc_id).await
        .expect("Failed to retrieve document");

    assert!(retrieved_doc.is_some(), "Should be able to retrieve document");
    let doc = retrieved_doc.unwrap();
    assert_eq!(doc.content, test_content);

    // Test search
    let search_results = rag_system.search("Rust programming", 5).await
        .expect("Failed to search documents");

    assert!(!search_results.is_empty(), "Search should return results");

    // Test document listing
    let documents = rag_system.list_documents(10, 0).await
        .expect("Failed to list documents");

    assert!(!documents.is_empty(), "Should have at least one document");

    // Test document deletion
    let deleted = rag_system.delete_document(&doc_id).await
        .expect("Failed to delete document");

    assert!(deleted, "Document should be deleted successfully");
}

#[tokio::test]
async fn test_mcp_handler_integration() {
    ensure_redis_running().await.expect("Failed to start Redis");

    let config = rag_redis_mcp_server::mock_rag::Config {
        redis_url: "redis://localhost:6379/9".to_string(),
        embedding_model: "test-model".to_string(),
        chunk_size: 256,
    };

    let handler = rag_redis_mcp_server::handlers::McpHandler::new(config).await
        .expect("Failed to create MCP handler");

    // Test initialization
    let init_request = rag_redis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::Value::Number(1.into())),
        method: "initialize".to_string(),
        params: Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "integration-test",
                "version": "1.0.0"
            }
        })),
    };

    let response = handler.handle_request(init_request).await;
    assert!(response.error.is_none(), "Initialization should succeed");

    // Test tools listing
    let tools_request = rag_redis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::Value::Number(2.into())),
        method: "tools/list".to_string(),
        params: None,
    };

    let tools_response = handler.handle_request(tools_request).await;
    assert!(tools_response.error.is_none(), "Tools list should succeed");

    let tools = tools_response.result
        .and_then(|r| r.get("tools").cloned())
        .and_then(|t| t.as_array().cloned())
        .expect("Should have tools array");

    assert!(!tools.is_empty(), "Should have available tools");

    // Verify specific tools exist
    let tool_names: Vec<String> = tools.iter()
        .filter_map(|tool| tool.get("name"))
        .filter_map(|name| name.as_str())
        .map(|s| s.to_string())
        .collect();

    assert!(tool_names.contains(&"ingest_document".to_string()));
    assert!(tool_names.contains(&"search_documents".to_string()));
    assert!(tool_names.contains(&"health_check".to_string()));
}

/// Performance test to ensure operations complete within reasonable time
#[tokio::test]
async fn test_performance_benchmarks() {
    ensure_redis_running().await.expect("Failed to start Redis");

    let config = rag_redis_mcp_server::mock_rag::Config {
        redis_url: "redis://localhost:6379/9".to_string(),
        embedding_model: "test-model".to_string(),
        chunk_size: 256,
    };

    let rag_system = rag_redis_mcp_server::mock_rag::RagSystem::new(config).await
        .expect("Failed to create RAG system");

    let start = std::time::Instant::now();

    // Ingest multiple documents
    for i in 0..10 {
        let content = format!("Test document {} with content about topic {}", i, i % 3);
        let metadata = serde_json::json!({
            "title": format!("Document {}", i),
            "topic": i % 3
        });

        rag_system.ingest_document(&content, metadata).await
            .expect("Failed to ingest document");
    }

    let ingestion_time = start.elapsed();
    println!("📊 Ingested 10 documents in {:?}", ingestion_time);

    // Test search performance
    let search_start = std::time::Instant::now();
    let results = rag_system.search("Test document", 5).await
        .expect("Failed to search documents");
    let search_time = search_start.elapsed();

    println!("📊 Search completed in {:?}, found {} results", search_time, results.len());

    // Performance assertions
    assert!(ingestion_time < Duration::from_secs(10), "Ingestion should complete within 10 seconds");
    assert!(search_time < Duration::from_secs(5), "Search should complete within 5 seconds");
    assert!(!results.is_empty(), "Search should return results");
}

/// Test cleanup and system state
#[tokio::test]
async fn test_system_cleanup() {
    ensure_redis_running().await.expect("Failed to start Redis");

    let config = rag_redis_mcp_server::mock_rag::Config {
        redis_url: "redis://localhost:6379/9".to_string(),
        embedding_model: "test-model".to_string(),
        chunk_size: 256,
    };

    let rag_system = rag_redis_mcp_server::mock_rag::RagSystem::new(config).await
        .expect("Failed to create RAG system");

    // Add some test data
    let doc_id = rag_system.ingest_document("Test cleanup content", serde_json::json!({})).await
        .expect("Failed to ingest document");

    // Verify data exists
    let doc = rag_system.get_document(&doc_id).await
        .expect("Failed to get document");
    assert!(doc.is_some());

    // Test cleanup
    rag_system.clear_all_documents().await
        .expect("Failed to clear documents");

    // Verify cleanup worked
    let doc_after_cleanup = rag_system.get_document(&doc_id).await
        .expect("Failed to check document after cleanup");
    assert!(doc_after_cleanup.is_none(), "Document should be deleted after cleanup");

    let documents_list = rag_system.list_documents(100, 0).await
        .expect("Failed to list documents after cleanup");
    assert!(documents_list.is_empty(), "Document list should be empty after cleanup");
}