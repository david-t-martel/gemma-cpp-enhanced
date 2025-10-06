use rag_redis_mcp_server::{McpHandler, JsonRpcRequest, ToolCallRequest};
use serde_json::{json, Value};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create MCP handler
    let config = rag_redis_mcp_server::mock_rag::Config::default();
    let handler = McpHandler::new(config).await?;

    // Initialize the server first
    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {},
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })),
    };

    let response = handler.handle_request(init_request).await;
    println!("Initialize response: {}", serde_json::to_string_pretty(&response)?);

    // Test ingesting a document
    let mut tool_args = HashMap::new();
    tool_args.insert("content".to_string(), json!("This is a test document about machine learning and AI."));
    tool_args.insert("metadata".to_string(), json!({
        "title": "Test Document",
        "category": "technology"
    }));

    let ingest_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "ingest_document",
            "arguments": tool_args
        })),
    };

    let response = handler.handle_request(ingest_request).await;
    println!("Ingest response: {}", serde_json::to_string_pretty(&response)?);

    // Test searching documents
    let mut search_args = HashMap::new();
    search_args.insert("query".to_string(), json!("machine learning"));
    search_args.insert("limit".to_string(), json!(5));

    let search_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "search_documents",
            "arguments": search_args
        })),
    };

    let response = handler.handle_request(search_request).await;
    println!("Search response: {}", serde_json::to_string_pretty(&response)?);

    // Test health check
    let health_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(4)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "health_check",
            "arguments": {}
        })),
    };

    let response = handler.handle_request(health_request).await;
    println!("Health check response: {}", serde_json::to_string_pretty(&response)?);

    Ok(())
}
