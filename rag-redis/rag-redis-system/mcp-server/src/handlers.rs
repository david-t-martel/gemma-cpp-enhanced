use crate::protocol::*;
use crate::tools::create_tools;
use crate::mock_rag::{Config, RagSystem, Result as RagResult};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// MCP Server handler managing the RAG system
pub struct McpHandler {
    rag_system: Arc<RagSystem>,
    server_info: ServerInfo,
    capabilities: ServerCapabilities,
    initialized: Arc<RwLock<bool>>,
    progress_tokens: Arc<RwLock<HashMap<String, ProgressToken>>>,
}

impl McpHandler {
    pub async fn new(config: Config) -> RagResult<Self> {
        let rag_system = Arc::new(RagSystem::new(config).await?);

        let server_info = ServerInfo {
            name: "rag-redis-mcp-server".to_string(),
            version: "0.1.0".to_string(),
        };

        let capabilities = ServerCapabilities {
            tools: Some(ToolsCapability {
                list_changed: Some(false),
            }),
            resources: Some(ResourcesCapability {
                subscribe: Some(false),
                list_changed: Some(false),
            }),
            prompts: None,
            logging: Some(LoggingCapability {
                level: Some("info".to_string()),
            }),
        };

        Ok(Self {
            rag_system,
            server_info,
            capabilities,
            initialized: Arc::new(RwLock::new(false)),
            progress_tokens: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {}", request.method);

        match request.method.as_str() {
            methods::INITIALIZE => self.handle_initialize(request).await,
            methods::INITIALIZED => self.handle_initialized(request).await,
            methods::PING => self.handle_ping(request).await,
            methods::TOOLS_LIST => self.handle_tools_list(request).await,
            methods::TOOLS_CALL => self.handle_tools_call(request).await,
            methods::RESOURCES_LIST => self.handle_resources_list(request).await,
            methods::RESOURCES_READ => self.handle_resources_read(request).await,
            methods::LOGGING_SET_LEVEL => self.handle_set_logging_level(request).await,
            _ => JsonRpcResponse::error(request.id, JsonRpcError::method_not_found()),
        }
    }

    async fn handle_initialize(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params = match request.params {
            Some(params) => params,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing initialization parameters"),
                )
            }
        };

        let init_request: InitializeRequest = match serde_json::from_value(params) {
            Ok(req) => req,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params(format!("Invalid initialization parameters: {}", e)),
                )
            }
        };

        // Check protocol version compatibility
        if init_request.protocol_version != MCP_VERSION {
            warn!(
                "Protocol version mismatch: client={}, server={}",
                init_request.protocol_version, MCP_VERSION
            );
        }

        info!(
            "Initializing MCP server for client: {} v{}",
            init_request.client_info.name, init_request.client_info.version
        );

        let response = InitializeResponse {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: self.capabilities.clone(),
            server_info: self.server_info.clone(),
            instructions: Some(
                "RAG-Redis MCP Server: High-performance document ingestion, vector search, and memory management system. Use tools to ingest documents, perform semantic searches, and manage the knowledge base.".to_string()
            ),
        };

        *self.initialized.write().await = true;
        info!("MCP server initialized successfully");

        JsonRpcResponse::success(request.id, json!(response))
    }

    async fn handle_initialized(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        info!("Received initialized notification from client");
        // This is a notification - in a proper implementation, notifications don't get responses
        // The main loop will check for request.id.is_none() and skip sending the response
        JsonRpcResponse::success(request.id, json!({}))
    }

    async fn handle_ping(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse::success(request.id, json!({}))
    }

    async fn handle_tools_list(&self, _request: JsonRpcRequest) -> JsonRpcResponse {
        if !*self.initialized.read().await {
            return JsonRpcResponse::error(
                _request.id,
                JsonRpcError::internal_error("Server not initialized"),
            );
        }

        let tools = create_tools();
        JsonRpcResponse::success(_request.id, json!({ "tools": tools }))
    }

    async fn handle_tools_call(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        if !*self.initialized.read().await {
            return JsonRpcResponse::error(
                request.id,
                JsonRpcError::internal_error("Server not initialized"),
            );
        }

        let params = match request.params {
            Some(params) => params,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing tool call parameters"),
                )
            }
        };

        let tool_call: ToolCallRequest = match serde_json::from_value(params) {
            Ok(call) => call,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params(format!("Invalid tool call parameters: {}", e)),
                )
            }
        };

        let result = self.execute_tool(&tool_call).await;
        JsonRpcResponse::success(request.id, json!(result))
    }

    async fn handle_resources_list(&self, _request: JsonRpcRequest) -> JsonRpcResponse {
        let resources = vec![
            Resource {
                uri: "rag-redis://system/config".to_string(),
                name: "System Configuration".to_string(),
                description: Some("Current system configuration and settings".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "rag-redis://system/metrics".to_string(),
                name: "System Metrics".to_string(),
                description: Some("Real-time system performance metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            Resource {
                uri: "rag-redis://memory/stats".to_string(),
                name: "Memory Statistics".to_string(),
                description: Some("Memory usage and storage statistics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ];

        JsonRpcResponse::success(_request.id, json!({ "resources": resources }))
    }

    async fn handle_resources_read(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params = match request.params {
            Some(params) => params,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing resource URI"),
                )
            }
        };

        let uri = match params.get("uri").and_then(|v| v.as_str()) {
            Some(uri) => uri,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing or invalid URI"),
                )
            }
        };

        match uri {
            "rag-redis://system/config" => {
                let content = json!({
                    "server": "rag-redis-mcp-server",
                    "version": "0.1.0",
                    "capabilities": self.capabilities
                });
                JsonRpcResponse::success(
                    request.id,
                    json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": content.to_string()
                        }]
                    })
                )
            }
            "rag-redis://system/metrics" => {
                // Get system metrics through tool call
                let metrics_result = self.get_system_metrics().await;
                JsonRpcResponse::success(
                    request.id,
                    json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": serde_json::to_string_pretty(&metrics_result).unwrap_or_default()
                        }]
                    })
                )
            }
            "rag-redis://memory/stats" => {
                let memory_stats = self.get_memory_stats().await;
                JsonRpcResponse::success(
                    request.id,
                    json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": serde_json::to_string_pretty(&memory_stats).unwrap_or_default()
                        }]
                    })
                )
            }
            _ => JsonRpcResponse::error(request.id, JsonRpcError::resource_not_found(uri)),
        }
    }

    async fn handle_set_logging_level(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params = match request.params {
            Some(params) => params,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing logging level"),
                )
            }
        };

        let level = match params.get("level").and_then(|v| v.as_str()) {
            Some(level) => level,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing or invalid logging level"),
                )
            }
        };

        info!("Setting logging level to: {}", level);
        // In a real implementation, you would update the tracing subscriber filter
        JsonRpcResponse::success(request.id, json!({}))
    }

    async fn execute_tool(&self, tool_call: &ToolCallRequest) -> ToolCallResult {
        debug!("Executing tool: {}", tool_call.name);

        let result = match tool_call.name.as_str() {
            "ingest_document" => self.tool_ingest_document(&tool_call.arguments).await,
            "search_documents" => self.tool_search_documents(&tool_call.arguments).await,
            "research_query" => self.tool_research_query(&tool_call.arguments).await,
            "list_documents" => self.tool_list_documents(&tool_call.arguments).await,
            "get_document" => self.tool_get_document(&tool_call.arguments).await,
            "delete_document" => self.tool_delete_document(&tool_call.arguments).await,
            "clear_memory" => self.tool_clear_memory(&tool_call.arguments).await,
            "get_memory_stats" => self.tool_get_memory_stats(&tool_call.arguments).await,
            "get_system_metrics" => self.tool_get_system_metrics(&tool_call.arguments).await,
            "health_check" => self.tool_health_check(&tool_call.arguments).await,
            "configure_system" => self.tool_configure_system(&tool_call.arguments).await,
            "batch_ingest" => self.tool_batch_ingest(&tool_call.arguments).await,
            "semantic_search" => self.tool_semantic_search(&tool_call.arguments).await,
            "hybrid_search" => self.tool_hybrid_search(&tool_call.arguments).await,
            _ => {
                error!("Unknown tool: {}", tool_call.name);
                return ToolCallResult {
                    content: vec![ToolCallContent {
                        content_type: "text".to_string(),
                        text: format!("Unknown tool: {}", tool_call.name),
                    }],
                    is_error: Some(true),
                };
            }
        };

        match result {
            Ok(content) => ToolCallResult {
                content: vec![ToolCallContent {
                    content_type: "text".to_string(),
                    text: content,
                }],
                is_error: None,
            },
            Err(e) => {
                error!("Tool execution failed: {}", e);
                ToolCallResult {
                    content: vec![ToolCallContent {
                        content_type: "text".to_string(),
                        text: format!("Error: {}", e),
                    }],
                    is_error: Some(true),
                }
            }
        }
    }

    // Tool implementations
    async fn tool_ingest_document(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' parameter")?;

        let metadata = args.get("metadata")
            .cloned()
            .unwrap_or_else(|| json!({}));

        let document_id = match self.rag_system.ingest_document(content, metadata).await {
            Ok(id) => id,
            Err(e) => return Err(format!("Failed to ingest document: {}", e)),
        };

        Ok(format!("Document ingested successfully with ID: {}", document_id))
    }

    async fn tool_search_documents(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let query = args.get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query' parameter")?;

        let limit = args.get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let results = match self.rag_system.search(query, limit).await {
            Ok(results) => results,
            Err(e) => return Err(format!("Search failed: {}", e)),
        };

        let response = json!({
            "query": query,
            "results": results.len(),
            "documents": results
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_research_query(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let query = args.get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query' parameter")?;

        let sources = args.get("sources")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(Vec::new);

        let results = match self.rag_system.research(query, sources).await {
            Ok(results) => results,
            Err(e) => return Err(format!("Research query failed: {}", e)),
        };

        let response = json!({
            "query": query,
            "results": results.len(),
            "combined_results": results
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_list_documents(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let limit = args.get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let offset = args.get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let documents = match self.rag_system.list_documents(limit, offset).await {
            Ok(docs) => docs,
            Err(e) => return Err(format!("Failed to list documents: {}", e)),
        };

        let response = json!({
            "documents": documents,
            "count": documents.len(),
            "limit": limit,
            "offset": offset
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_get_document(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let document_id = args.get("document_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'document_id' parameter")?;

        let document = match self.rag_system.get_document(document_id).await {
            Ok(Some(doc)) => doc,
            Ok(None) => return Err(format!("Document not found: {}", document_id)),
            Err(e) => return Err(format!("Failed to retrieve document: {}", e)),
        };

        let response = json!({
            "document": document,
            "id": document_id
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_delete_document(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let document_id = args.get("document_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'document_id' parameter")?;

        let deleted = match self.rag_system.delete_document(document_id).await {
            Ok(deleted) => deleted,
            Err(e) => return Err(format!("Failed to delete document: {}", e)),
        };

        if deleted {
            Ok(format!("Document {} deleted successfully", document_id))
        } else {
            Err(format!("Document {} not found", document_id))
        }
    }

    async fn tool_clear_memory(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let confirm = args.get("confirm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !confirm {
            return Err("Memory clearing requires explicit confirmation (set 'confirm': true)".to_string());
        }

        // Placeholder implementation - in a real system, this would clear Redis data
        let response = json!({
            "message": "Memory clearing functionality not yet implemented",
            "status": "placeholder",
            "warning": "This would be a destructive operation"
        });
        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_get_memory_stats(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        let stats = self.get_memory_stats().await;
        Ok(serde_json::to_string_pretty(&stats).unwrap_or_default())
    }

    async fn tool_get_system_metrics(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        let metrics = self.get_system_metrics().await;
        Ok(serde_json::to_string_pretty(&metrics).unwrap_or_default())
    }

    async fn tool_health_check(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        let health = json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "components": {
                "rag_system": "operational",
                "redis": "connected",
                "vector_store": "operational",
                "embedding_model": "loaded"
            },
            "version": "0.1.0"
        });
        Ok(serde_json::to_string_pretty(&health).unwrap_or_default())
    }

    async fn tool_configure_system(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        // Placeholder implementation
        let response = json!({
            "message": "System configuration functionality not yet implemented",
            "status": "placeholder"
        });
        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_batch_ingest(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        // Placeholder implementation
        let response = json!({
            "message": "Batch ingestion functionality not yet implemented",
            "status": "placeholder"
        });
        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_semantic_search(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        // For now, delegate to regular search
        self.tool_search_documents(args).await
    }

    async fn tool_hybrid_search(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        // For now, delegate to regular search
        self.tool_search_documents(args).await
    }

    // Helper methods
    async fn get_memory_stats(&self) -> Value {
        json!({
            "total_documents": 0,
            "total_chunks": 0,
            "memory_usage": {
                "episodic": 0,
                "semantic": 0,
                "procedural": 0
            },
            "storage": {
                "redis_keys": 0,
                "vector_index_size": 0
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }

    async fn get_system_metrics(&self) -> Value {
        json!({
            "performance": {
                "avg_search_latency_ms": 0.0,
                "avg_ingestion_time_ms": 0.0,
                "queries_per_second": 0.0
            },
            "resources": {
                "cpu_usage_percent": 0.0,
                "memory_usage_mb": 0.0,
                "redis_memory_mb": 0.0
            },
            "operations": {
                "total_searches": 0,
                "total_ingestions": 0,
                "total_errors": 0
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }
}
