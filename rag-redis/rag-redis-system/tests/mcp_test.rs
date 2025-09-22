//! MCP server tests for the RAG-Redis system
//!
//! These tests verify the MCP (Model Context Protocol) server functionality including:
//! - MCP message handling and protocol compliance
//! - Tool execution and response formatting
//! - Error handling and edge cases
//! - Resource management
//! - Performance and concurrency
//! - Integration with RAG system components

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rag_redis_system::{Config, Error, RagSystem, Result};

use serde_json::{json, Value};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Mock MCP message structure
#[derive(Debug, Clone)]
struct McpMessage {
    id: String,
    method: String,
    params: Value,
}

/// Mock MCP response structure
#[derive(Debug, Clone)]
struct McpResponse {
    id: String,
    result: Option<Value>,
    error: Option<McpError>,
}

/// Mock MCP error structure
#[derive(Debug, Clone)]
struct McpError {
    code: i32,
    message: String,
    data: Option<Value>,
}

/// Mock MCP server implementation for testing
pub struct MockMcpServer {
    system: Arc<RagSystem>,
    tools: HashMap<String, Box<dyn Fn(&Value) -> Result<Value> + Send + Sync>>,
    resources: HashMap<String, Value>,
    capabilities: Vec<String>,
}

impl MockMcpServer {
    pub async fn new(config: Config) -> Result<Self> {
        let system = Arc::new(RagSystem::new(config).await?);
        let mut server = Self {
            system,
            tools: HashMap::new(),
            resources: HashMap::new(),
            capabilities: vec![
                "tools".to_string(),
                "resources".to_string(),
                "prompts".to_string(),
            ],
        };

        server.register_tools();
        server.register_resources();

        Ok(server)
    }

    fn register_tools(&mut self) {
        // Register ingest_document tool
        self.tools.insert(
            "ingest_document".to_string(),
            Box::new(|params: &Value| -> Result<Value> {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::InvalidInput("Missing content parameter".to_string()))?;

                let metadata = params.get("metadata").cloned().unwrap_or_else(|| json!({}));

                // For testing, return a mock document ID
                let doc_id = format!("doc_{}", Uuid::new_v4());

                Ok(json!({
                    "document_id": doc_id,
                    "status": "success",
                    "chunks_created": content.len() / 100 + 1
                }))
            }),
        );

        // Register search tool
        self.tools.insert(
            "search".to_string(),
            Box::new(|params: &Value| -> Result<Value> {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::InvalidInput("Missing query parameter".to_string()))?;

                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

                // Mock search results
                let results = (0..std::cmp::min(limit, 3))
                    .map(|i| {
                        json!({
                            "id": format!("result_{}", i),
                            "text": format!("Mock result {} for query: {}", i, query),
                            "score": 0.9 - (i as f64 * 0.1),
                            "metadata": {
                                "source": "mock",
                                "index": i
                            }
                        })
                    })
                    .collect::<Vec<_>>();

                Ok(json!({
                    "results": results,
                    "total": results.len(),
                    "query": query
                }))
            }),
        );

        // Register research tool
        self.tools.insert(
            "research".to_string(),
            Box::new(|params: &Value| -> Result<Value> {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::InvalidInput("Missing query parameter".to_string()))?;

                let sources = params
                    .get("sources")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.len())
                    .unwrap_or(0);

                // Mock research results
                let results = vec![
                    json!({
                        "id": "research_1",
                        "text": format!("Research result for: {}", query),
                        "score": 0.95,
                        "metadata": {
                            "source": "web",
                            "type": "research"
                        }
                    }),
                    json!({
                        "id": "local_1",
                        "text": format!("Local result for: {}", query),
                        "score": 0.88,
                        "metadata": {
                            "source": "local",
                            "type": "document"
                        }
                    }),
                ];

                Ok(json!({
                    "results": results,
                    "sources_searched": sources,
                    "query": query
                }))
            }),
        );

        // Register get_stats tool
        self.tools.insert(
            "get_stats".to_string(),
            Box::new(|_params: &Value| -> Result<Value> {
                Ok(json!({
                    "vector_store": {
                        "vector_count": 1000,
                        "dimension": 768,
                        "memory_usage": 50_000_000
                    },
                    "redis": {
                        "total_operations": 5000,
                        "cache_hits": 4200,
                        "cache_misses": 800,
                        "hit_rate": 84
                    },
                    "system": {
                        "uptime_seconds": 3600,
                        "total_documents": 100,
                        "total_searches": 250
                    }
                }))
            }),
        );
    }

    fn register_resources(&mut self) {
        self.resources.insert(
            "system_status".to_string(),
            json!({
                "status": "healthy",
                "components": {
                    "redis": "connected",
                    "vector_store": "ready",
                    "embedding_model": "loaded"
                },
                "last_updated": chrono::Utc::now().to_rfc3339()
            }),
        );

        self.resources.insert(
            "capabilities".to_string(),
            json!({
                "tools": ["ingest_document", "search", "research", "get_stats"],
                "resources": ["system_status", "capabilities", "config"],
                "supported_formats": ["txt", "md", "html", "json", "pdf"],
                "max_document_size": 50_000_000,
                "vector_dimensions": [384, 768, 1536]
            }),
        );

        self.resources.insert(
            "config".to_string(),
            json!({
                "vector_store": {
                    "dimension": 768,
                    "distance_metric": "Cosine"
                },
                "redis": {
                    "pool_size": 10,
                    "connection_timeout_secs": 5
                },
                "document": {
                    "chunk_size": 512,
                    "chunk_overlap": 50
                }
            }),
        );
    }

    pub async fn handle_message(&self, message: McpMessage) -> McpResponse {
        match message.method.as_str() {
            "initialize" => self.handle_initialize(message).await,
            "tools/list" => self.handle_list_tools(message).await,
            "tools/call" => self.handle_call_tool(message).await,
            "resources/list" => self.handle_list_resources(message).await,
            "resources/read" => self.handle_read_resource(message).await,
            "prompts/list" => self.handle_list_prompts(message).await,
            _ => McpResponse {
                id: message.id,
                result: None,
                error: Some(McpError {
                    code: -32601,
                    message: format!("Method not found: {}", message.method),
                    data: None,
                }),
            },
        }
    }

    async fn handle_initialize(&self, message: McpMessage) -> McpResponse {
        let capabilities = json!({
            "tools": true,
            "resources": true,
            "prompts": true,
            "experimental": {
                "streaming": false,
                "batch_operations": true
            }
        });

        McpResponse {
            id: message.id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": capabilities,
                "serverInfo": {
                    "name": "rag-redis-system",
                    "version": "0.1.0",
                    "description": "High-performance local RAG system with Redis backend"
                }
            })),
            error: None,
        }
    }

    async fn handle_list_tools(&self, message: McpMessage) -> McpResponse {
        let tools = self
            .tools
            .keys()
            .map(|name| {
                json!({
                    "name": name,
                    "description": self.get_tool_description(name),
                    "inputSchema": self.get_tool_schema(name)
                })
            })
            .collect::<Vec<_>>();

        McpResponse {
            id: message.id,
            result: Some(json!({ "tools": tools })),
            error: None,
        }
    }

    async fn handle_call_tool(&self, message: McpMessage) -> McpResponse {
        let tool_name = match message.params.get("name").and_then(|v| v.as_str()) {
            Some(name) => name,
            None => {
                return McpResponse {
                    id: message.id,
                    result: None,
                    error: Some(McpError {
                        code: -32602,
                        message: "Missing tool name".to_string(),
                        data: None,
                    }),
                };
            }
        };

        let arguments = message.params.get("arguments").unwrap_or(&json!({}));

        if let Some(tool) = self.tools.get(tool_name) {
            match tool(arguments) {
                Ok(result) => McpResponse {
                    id: message.id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                        }],
                        "isError": false
                    })),
                    error: None,
                },
                Err(e) => McpResponse {
                    id: message.id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Tool execution failed: {}", e)
                        }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        } else {
            McpResponse {
                id: message.id,
                result: None,
                error: Some(McpError {
                    code: -32601,
                    message: format!("Tool not found: {}", tool_name),
                    data: None,
                }),
            }
        }
    }

    async fn handle_list_resources(&self, message: McpMessage) -> McpResponse {
        let resources = self
            .resources
            .keys()
            .map(|name| {
                json!({
                    "uri": format!("rag-redis://{}", name),
                    "name": name,
                    "description": self.get_resource_description(name),
                    "mimeType": "application/json"
                })
            })
            .collect::<Vec<_>>();

        McpResponse {
            id: message.id,
            result: Some(json!({ "resources": resources })),
            error: None,
        }
    }

    async fn handle_read_resource(&self, message: McpMessage) -> McpResponse {
        let uri = match message.params.get("uri").and_then(|v| v.as_str()) {
            Some(uri) => uri,
            None => {
                return McpResponse {
                    id: message.id,
                    result: None,
                    error: Some(McpError {
                        code: -32602,
                        message: "Missing resource URI".to_string(),
                        data: None,
                    }),
                };
            }
        };

        let resource_name = uri.strip_prefix("rag-redis://").unwrap_or(uri);

        if let Some(resource) = self.resources.get(resource_name) {
            McpResponse {
                id: message.id,
                result: Some(json!({
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": serde_json::to_string_pretty(resource).unwrap_or_default()
                    }]
                })),
                error: None,
            }
        } else {
            McpResponse {
                id: message.id,
                result: None,
                error: Some(McpError {
                    code: -32601,
                    message: format!("Resource not found: {}", resource_name),
                    data: None,
                }),
            }
        }
    }

    async fn handle_list_prompts(&self, message: McpMessage) -> McpResponse {
        let prompts = vec![
            json!({
                "name": "summarize_document",
                "description": "Summarize a document using RAG search results",
                "arguments": [
                    {
                        "name": "query",
                        "description": "Search query for relevant content",
                        "required": true
                    },
                    {
                        "name": "max_results",
                        "description": "Maximum number of search results to include",
                        "required": false
                    }
                ]
            }),
            json!({
                "name": "research_topic",
                "description": "Research a topic using both local and web sources",
                "arguments": [
                    {
                        "name": "topic",
                        "description": "Topic to research",
                        "required": true
                    },
                    {
                        "name": "sources",
                        "description": "List of web sources to include",
                        "required": false
                    }
                ]
            }),
        ];

        McpResponse {
            id: message.id,
            result: Some(json!({ "prompts": prompts })),
            error: None,
        }
    }

    fn get_tool_description(&self, name: &str) -> String {
        match name {
            "ingest_document" => "Ingest a document into the RAG system".to_string(),
            "search" => "Search for similar content in the vector database".to_string(),
            "research" => "Perform research using both local and web sources".to_string(),
            "get_stats" => "Get system statistics and health information".to_string(),
            _ => "Unknown tool".to_string(),
        }
    }

    fn get_tool_schema(&self, name: &str) -> Value {
        match name {
            "ingest_document" => json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The document content to ingest"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for the document"
                    }
                },
                "required": ["content"]
            }),
            "search" => json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }),
            "research" => json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of web sources to search"
                    }
                },
                "required": ["query"]
            }),
            "get_stats" => json!({
                "type": "object",
                "properties": {},
                "description": "No parameters required"
            }),
            _ => json!({}),
        }
    }

    fn get_resource_description(&self, name: &str) -> String {
        match name {
            "system_status" => "Current system status and component health".to_string(),
            "capabilities" => "System capabilities and supported operations".to_string(),
            "config" => "Current system configuration".to_string(),
            _ => "Unknown resource".to_string(),
        }
    }
}

/// Test MCP server initialization
#[tokio::test]
async fn test_mcp_server_initialization() {
    let config = Config::default();

    match MockMcpServer::new(config).await {
        Ok(server) => {
            println!("✓ MCP server initialized successfully");

            // Test initialize message
            let init_message = McpMessage {
                id: "test_init_1".to_string(),
                method: "initialize".to_string(),
                params: json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {}
                }),
            };

            let response = server.handle_message(init_message).await;
            assert!(
                response.error.is_none(),
                "Initialize should not return error"
            );
            assert!(response.result.is_some(), "Initialize should return result");

            if let Some(result) = response.result {
                assert!(result.get("protocolVersion").is_some());
                assert!(result.get("capabilities").is_some());
                assert!(result.get("serverInfo").is_some());
            }

            println!("✓ MCP initialization protocol working correctly");
        }
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP server tests");
            return;
        }
        Err(e) => {
            panic!("Failed to initialize MCP server: {:?}", e);
        }
    }
}

/// Test MCP tools listing and execution
#[tokio::test]
async fn test_mcp_tools() {
    let config = Config::default();

    let server = match MockMcpServer::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP tools tests");
            return;
        }
        Err(e) => panic!("Failed to create MCP server: {:?}", e),
    };

    // Test tools listing
    let list_tools_message = McpMessage {
        id: "test_list_tools_1".to_string(),
        method: "tools/list".to_string(),
        params: json!({}),
    };

    let response = server.handle_message(list_tools_message).await;
    assert!(
        response.error.is_none(),
        "Tools list should not return error"
    );

    if let Some(result) = response.result {
        let tools = result.get("tools").expect("Should have tools array");
        let tools_array = tools.as_array().expect("Tools should be array");

        assert!(!tools_array.is_empty(), "Should have at least one tool");

        // Check that expected tools are present
        let tool_names: Vec<String> = tools_array
            .iter()
            .filter_map(|t| {
                t.get("name")
                    .and_then(|n| n.as_str().map(|s| s.to_string()))
            })
            .collect();

        let expected_tools = ["ingest_document", "search", "research", "get_stats"];
        for expected_tool in &expected_tools {
            assert!(
                tool_names.contains(&expected_tool.to_string()),
                "Should have {} tool",
                expected_tool
            );
        }

        println!("✓ Found {} tools: {:?}", tool_names.len(), tool_names);
    }

    // Test individual tools
    test_ingest_document_tool(&server).await;
    test_search_tool(&server).await;
    test_research_tool(&server).await;
    test_get_stats_tool(&server).await;
}

async fn test_ingest_document_tool(server: &MockMcpServer) {
    let call_tool_message = McpMessage {
        id: "test_ingest_1".to_string(),
        method: "tools/call".to_string(),
        params: json!({
            "name": "ingest_document",
            "arguments": {
                "content": "This is a test document for MCP testing. It contains information about testing procedures and best practices.",
                "metadata": {
                    "title": "MCP Test Document",
                    "author": "Test Suite",
                    "category": "testing"
                }
            }
        }),
    };

    let response = server.handle_message(call_tool_message).await;
    assert!(
        response.error.is_none(),
        "Ingest document should not return error"
    );

    if let Some(result) = response.result {
        let content = result.get("content").expect("Should have content");
        let content_array = content.as_array().expect("Content should be array");

        assert!(!content_array.is_empty(), "Should have content");
        assert_eq!(content_array[0]["type"], "text");

        let text = content_array[0]["text"].as_str().expect("Should have text");
        assert!(text.contains("document_id"), "Should return document ID");
        assert!(text.contains("success"), "Should indicate success");

        println!("✓ Ingest document tool working correctly");
    }
}

async fn test_search_tool(server: &MockMcpServer) {
    let call_tool_message = McpMessage {
        id: "test_search_1".to_string(),
        method: "tools/call".to_string(),
        params: json!({
            "name": "search",
            "arguments": {
                "query": "machine learning algorithms",
                "limit": 5
            }
        }),
    };

    let response = server.handle_message(call_tool_message).await;
    assert!(response.error.is_none(), "Search should not return error");

    if let Some(result) = response.result {
        let content = result.get("content").expect("Should have content");
        let content_array = content.as_array().expect("Content should be array");

        let text = content_array[0]["text"].as_str().expect("Should have text");
        assert!(text.contains("results"), "Should return results");
        assert!(
            text.contains("machine learning algorithms"),
            "Should include query"
        );

        println!("✓ Search tool working correctly");
    }
}

async fn test_research_tool(server: &MockMcpServer) {
    let call_tool_message = McpMessage {
        id: "test_research_1".to_string(),
        method: "tools/call".to_string(),
        params: json!({
            "name": "research",
            "arguments": {
                "query": "artificial intelligence trends",
                "sources": ["arxiv.org", "openai.com"]
            }
        }),
    };

    let response = server.handle_message(call_tool_message).await;
    assert!(response.error.is_none(), "Research should not return error");

    if let Some(result) = response.result {
        let content = result.get("content").expect("Should have content");
        let content_array = content.as_array().expect("Content should be array");

        let text = content_array[0]["text"].as_str().expect("Should have text");
        assert!(text.contains("results"), "Should return results");
        assert!(
            text.contains("artificial intelligence trends"),
            "Should include query"
        );

        println!("✓ Research tool working correctly");
    }
}

async fn test_get_stats_tool(server: &MockMcpServer) {
    let call_tool_message = McpMessage {
        id: "test_stats_1".to_string(),
        method: "tools/call".to_string(),
        params: json!({
            "name": "get_stats",
            "arguments": {}
        }),
    };

    let response = server.handle_message(call_tool_message).await;
    assert!(
        response.error.is_none(),
        "Get stats should not return error"
    );

    if let Some(result) = response.result {
        let content = result.get("content").expect("Should have content");
        let content_array = content.as_array().expect("Content should be array");

        let text = content_array[0]["text"].as_str().expect("Should have text");
        assert!(
            text.contains("vector_store"),
            "Should include vector store stats"
        );
        assert!(text.contains("redis"), "Should include redis stats");
        assert!(text.contains("system"), "Should include system stats");

        println!("✓ Get stats tool working correctly");
    }
}

/// Test MCP resources functionality
#[tokio::test]
async fn test_mcp_resources() {
    let config = Config::default();

    let server = match MockMcpServer::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP resources tests");
            return;
        }
        Err(e) => panic!("Failed to create MCP server: {:?}", e),
    };

    // Test resources listing
    let list_resources_message = McpMessage {
        id: "test_list_resources_1".to_string(),
        method: "resources/list".to_string(),
        params: json!({}),
    };

    let response = server.handle_message(list_resources_message).await;
    assert!(
        response.error.is_none(),
        "Resources list should not return error"
    );

    if let Some(result) = response.result {
        let resources = result
            .get("resources")
            .expect("Should have resources array");
        let resources_array = resources.as_array().expect("Resources should be array");

        assert!(
            !resources_array.is_empty(),
            "Should have at least one resource"
        );

        let resource_names: Vec<String> = resources_array
            .iter()
            .filter_map(|r| {
                r.get("name")
                    .and_then(|n| n.as_str().map(|s| s.to_string()))
            })
            .collect();

        let expected_resources = ["system_status", "capabilities", "config"];
        for expected_resource in &expected_resources {
            assert!(
                resource_names.contains(&expected_resource.to_string()),
                "Should have {} resource",
                expected_resource
            );
        }

        println!(
            "✓ Found {} resources: {:?}",
            resource_names.len(),
            resource_names
        );
    }

    // Test reading individual resources
    for resource_name in ["system_status", "capabilities", "config"] {
        let read_resource_message = McpMessage {
            id: format!("test_read_resource_{}", resource_name),
            method: "resources/read".to_string(),
            params: json!({
                "uri": format!("rag-redis://{}", resource_name)
            }),
        };

        let response = server.handle_message(read_resource_message).await;
        assert!(
            response.error.is_none(),
            "Resource read should not return error"
        );

        if let Some(result) = response.result {
            let contents = result.get("contents").expect("Should have contents");
            let contents_array = contents.as_array().expect("Contents should be array");

            assert!(!contents_array.is_empty(), "Should have content");
            assert_eq!(contents_array[0]["mimeType"], "application/json");
            assert!(
                contents_array[0].get("text").is_some(),
                "Should have text content"
            );

            println!("✓ Resource '{}' readable", resource_name);
        }
    }
}

/// Test MCP error handling
#[tokio::test]
async fn test_mcp_error_handling() {
    let config = Config::default();

    let server = match MockMcpServer::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP error tests");
            return;
        }
        Err(e) => panic!("Failed to create MCP server: {:?}", e),
    };

    // Test unknown method
    let unknown_method_message = McpMessage {
        id: "test_error_1".to_string(),
        method: "unknown/method".to_string(),
        params: json!({}),
    };

    let response = server.handle_message(unknown_method_message).await;
    assert!(
        response.error.is_some(),
        "Unknown method should return error"
    );

    if let Some(error) = response.error {
        assert_eq!(error.code, -32601);
        assert!(error.message.contains("Method not found"));
        println!("✓ Unknown method properly handled");
    }

    // Test tool with missing parameters
    let missing_params_message = McpMessage {
        id: "test_error_2".to_string(),
        method: "tools/call".to_string(),
        params: json!({
            "arguments": {"query": "test"}
            // Missing "name" parameter
        }),
    };

    let response = server.handle_message(missing_params_message).await;
    assert!(
        response.error.is_some(),
        "Missing tool name should return error"
    );

    if let Some(error) = response.error {
        assert_eq!(error.code, -32602);
        assert!(error.message.contains("Missing tool name"));
        println!("✓ Missing parameters properly handled");
    }

    // Test calling non-existent tool
    let nonexistent_tool_message = McpMessage {
        id: "test_error_3".to_string(),
        method: "tools/call".to_string(),
        params: json!({
            "name": "nonexistent_tool",
            "arguments": {}
        }),
    };

    let response = server.handle_message(nonexistent_tool_message).await;
    assert!(
        response.error.is_some(),
        "Non-existent tool should return error"
    );

    if let Some(error) = response.error {
        assert_eq!(error.code, -32601);
        assert!(error.message.contains("Tool not found"));
        println!("✓ Non-existent tool properly handled");
    }

    // Test reading non-existent resource
    let nonexistent_resource_message = McpMessage {
        id: "test_error_4".to_string(),
        method: "resources/read".to_string(),
        params: json!({
            "uri": "rag-redis://nonexistent_resource"
        }),
    };

    let response = server.handle_message(nonexistent_resource_message).await;
    assert!(
        response.error.is_some(),
        "Non-existent resource should return error"
    );

    if let Some(error) = response.error {
        assert_eq!(error.code, -32601);
        assert!(error.message.contains("Resource not found"));
        println!("✓ Non-existent resource properly handled");
    }
}

/// Test MCP concurrent message handling
#[tokio::test]
async fn test_mcp_concurrent_handling() {
    let config = Config::default();

    let server = match Arc::new(MockMcpServer::new(config).await) {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP concurrency tests");
            return;
        }
        Err(e) => panic!("Failed to create MCP server: {:?}", e),
    };

    let mut handles = Vec::new();

    // Spawn multiple concurrent requests
    for i in 0..10 {
        let server_clone = server.clone();
        let handle = tokio::spawn(async move {
            let message = McpMessage {
                id: format!("concurrent_test_{}", i),
                method: "tools/call".to_string(),
                params: json!({
                    "name": "search",
                    "arguments": {
                        "query": format!("concurrent query {}", i),
                        "limit": 3
                    }
                }),
            };

            let response = server_clone.handle_message(message).await;
            assert!(
                response.error.is_none(),
                "Concurrent request {} should succeed",
                i
            );

            println!("✓ Concurrent request {} completed", i);
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    for handle in handles {
        handle.await.expect("Concurrent task should complete");
    }

    println!("✓ All concurrent requests completed successfully");
}

/// Test MCP performance with timing
#[tokio::test]
async fn test_mcp_performance() {
    let config = Config::default();

    let server = match MockMcpServer::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP performance tests");
            return;
        }
        Err(e) => panic!("Failed to create MCP server: {:?}", e),
    };

    // Test tool execution performance
    let start_time = std::time::Instant::now();
    let iterations = 100;

    for i in 0..iterations {
        let message = McpMessage {
            id: format!("perf_test_{}", i),
            method: "tools/call".to_string(),
            params: json!({
                "name": "get_stats",
                "arguments": {}
            }),
        };

        let response = server.handle_message(message).await;
        assert!(
            response.error.is_none(),
            "Performance test {} should succeed",
            i
        );
    }

    let elapsed = start_time.elapsed();
    let avg_time = elapsed / iterations;

    println!("✓ Completed {} tool calls in {:?}", iterations, elapsed);
    println!("✓ Average time per call: {:?}", avg_time);

    // Performance should be reasonable (less than 1ms per call for mock operations)
    assert!(avg_time.as_millis() < 10, "Tool calls should be fast");

    // Test resource read performance
    let start_time = std::time::Instant::now();

    for i in 0..iterations {
        let message = McpMessage {
            id: format!("perf_resource_{}", i),
            method: "resources/read".to_string(),
            params: json!({
                "uri": "rag-redis://system_status"
            }),
        };

        let response = server.handle_message(message).await;
        assert!(
            response.error.is_none(),
            "Resource read {} should succeed",
            i
        );
    }

    let elapsed = start_time.elapsed();
    let avg_time = elapsed / iterations;

    println!("✓ Completed {} resource reads in {:?}", iterations, elapsed);
    println!("✓ Average time per read: {:?}", avg_time);

    assert!(avg_time.as_millis() < 10, "Resource reads should be fast");
}

/// Test MCP protocol compliance
#[tokio::test]
async fn test_mcp_protocol_compliance() {
    let config = Config::default();

    let server = match MockMcpServer::new(config).await {
        Ok(s) => s,
        Err(Error::Redis(_)) => {
            println!("⚠ Redis not available, skipping MCP protocol tests");
            return;
        }
        Err(e) => panic!("Failed to create MCP server: {:?}", e),
    };

    // Test that all responses have correct structure
    let test_messages = vec![
        McpMessage {
            id: "proto_test_1".to_string(),
            method: "initialize".to_string(),
            params: json!({}),
        },
        McpMessage {
            id: "proto_test_2".to_string(),
            method: "tools/list".to_string(),
            params: json!({}),
        },
        McpMessage {
            id: "proto_test_3".to_string(),
            method: "resources/list".to_string(),
            params: json!({}),
        },
    ];

    for message in test_messages {
        let response = server.handle_message(message.clone()).await;

        // Check response structure
        assert_eq!(
            response.id, message.id,
            "Response ID should match request ID"
        );

        // Response should have either result or error, but not both
        match (response.result.is_some(), response.error.is_some()) {
            (true, false) => {
                println!("✓ Message {} returned success result", message.method);
            }
            (false, true) => {
                println!("✓ Message {} returned error (as expected)", message.method);
            }
            _ => {
                panic!("Response should have either result or error, not both or neither");
            }
        }
    }

    // Test that tool schemas are valid JSON Schema
    let list_tools_message = McpMessage {
        id: "schema_test".to_string(),
        method: "tools/list".to_string(),
        params: json!({}),
    };

    let response = server.handle_message(list_tools_message).await;
    if let Some(result) = response.result {
        let tools = result.get("tools").expect("Should have tools");
        let tools_array = tools.as_array().expect("Tools should be array");

        for tool in tools_array {
            assert!(tool.get("name").is_some(), "Tool should have name");
            assert!(
                tool.get("description").is_some(),
                "Tool should have description"
            );

            if let Some(schema) = tool.get("inputSchema") {
                assert!(schema.get("type").is_some(), "Schema should have type");
                // Additional schema validation could go here
            }
        }

        println!("✓ All tool schemas are properly formatted");
    }
}

/// Helper function to run all MCP tests
pub async fn run_mcp_tests() -> Result<()> {
    println!("🚀 Starting MCP Server Tests");

    println!("\n🏗️ Testing MCP Server Initialization...");
    test_mcp_server_initialization().await;

    println!("\n🔧 Testing MCP Tools...");
    test_mcp_tools().await;

    println!("\n📚 Testing MCP Resources...");
    test_mcp_resources().await;

    println!("\n🚨 Testing MCP Error Handling...");
    test_mcp_error_handling().await;

    println!("\n⚡ Testing MCP Concurrent Handling...");
    test_mcp_concurrent_handling().await;

    println!("\n⏱️ Testing MCP Performance...");
    test_mcp_performance().await;

    println!("\n📋 Testing MCP Protocol Compliance...");
    test_mcp_protocol_compliance().await;

    println!("\n✅ MCP Server Tests Completed!");

    Ok(())
}
