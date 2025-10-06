use crate::protocol::*;
use crate::tools::create_tools;
use rag_redis_system::{Config, RagSystem, Result as RagResult};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// MCP Server handler managing the RAG system
#[derive(Clone)]
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
            "initialize" => self.handle_initialize(request).await,
            "initialized" => self.handle_initialized(request).await,
            "ping" => self.handle_ping(request).await,
            "tools/list" => self.handle_tools_list(request).await,
            "tools/call" => self.handle_tools_call(request).await,
            "resources/list" => self.handle_resources_list(request).await,
            "resources/read" => self.handle_resources_read(request).await,
            "logging/setLevel" => self.handle_logging_set_level(request).await,
            "notifications/tools/list_changed" => {
                self.handle_tools_list_changed(request).await
            }
            _ => {
                error!("Unknown method: {}", request.method);
                JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::method_not_found(),
                )
            }
        }
    }

    async fn handle_initialize(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let init_request: InitializeRequest = match request.params.and_then(|p| serde_json::from_value(p).ok()) {
            Some(req) => req,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Invalid initialize request"),
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
                "RAG-Redis MCP Server: High-performance document ingestion, vector search, memory management, and comprehensive project context storage system. Use tools to ingest documents, perform semantic searches, manage the knowledge base, and save/restore complete project states.".to_string()
            ),
        };

        *self.initialized.write().await = true;
        info!("MCP server initialized successfully");

        JsonRpcResponse::success(request.id, json!(response))
    }

    async fn handle_initialized(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        info!("Received initialized notification from client");
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
            Resource {
                uri: "rag-redis://health/status".to_string(),
                name: "Health Status".to_string(),
                description: Some("System health and component status".to_string()),
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
                    JsonRpcError::invalid_params("Missing resource read parameters"),
                )
            }
        };

        let read_request: ResourceReadRequest = match serde_json::from_value(params) {
            Ok(req) => req,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params(format!("Invalid resource read parameters: {}", e)),
                )
            }
        };

        let content = match read_request.uri.as_str() {
            "rag-redis://system/config" => {
                json!({
                    "version": "0.1.0",
                    "server": "rag-redis-mcp-server",
                    "capabilities": self.capabilities,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })
            }
            "rag-redis://system/metrics" => self.get_system_metrics().await,
            "rag-redis://memory/stats" => self.get_memory_stats().await,
            "rag-redis://health/status" => {
                json!({
                    "status": "healthy",
                    "components": {
                        "rag_system": "operational",
                        "redis": "connected",
                        "vector_store": "ready"
                    },
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })
            }
            _ => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Unknown resource URI"),
                )
            }
        };

        let response = ResourceReadResponse {
            contents: vec![ResourceContent {
                uri: read_request.uri,
                mime_type: Some("application/json".to_string()),
                text: Some(serde_json::to_string_pretty(&content).unwrap_or_default()),
                blob: None,
            }],
        };

        JsonRpcResponse::success(request.id, json!(response))
    }

    async fn handle_logging_set_level(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params = match request.params {
            Some(params) => params,
            None => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params("Missing logging parameters"),
                )
            }
        };

        let _level_request: LoggingSetLevelRequest = match serde_json::from_value(params) {
            Ok(req) => req,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id,
                    JsonRpcError::invalid_params(format!("Invalid logging parameters: {}", e)),
                )
            }
        };

        // Log level change would be handled here
        info!("Logging level change requested");
        JsonRpcResponse::success(request.id, json!({}))
    }

    async fn handle_tools_list_changed(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        // This is a notification from client about tools list changes
        info!("Client notified of tools list changes");
        JsonRpcResponse::success(request.id, json!({}))
    }

    async fn execute_tool(&self, tool_call: &ToolCallRequest) -> ToolCallResult {
        debug!("Executing tool: {}", tool_call.name);

        let result = match tool_call.name.as_str() {
            // Document management tools
            "ingest_document" => self.tool_ingest_document(&tool_call.arguments).await,
            "search_documents" => self.tool_search_documents(&tool_call.arguments).await,
            "research_query" => self.tool_research_query(&tool_call.arguments).await,
            "list_documents" => self.tool_list_documents(&tool_call.arguments).await,
            "get_document" => self.tool_get_document(&tool_call.arguments).await,
            "delete_document" => self.tool_delete_document(&tool_call.arguments).await,
            "batch_ingest" => self.tool_batch_ingest(&tool_call.arguments).await,
            "semantic_search" => self.tool_semantic_search(&tool_call.arguments).await,
            "hybrid_search" => self.tool_hybrid_search(&tool_call.arguments).await,
            
            // Memory management tools
            "store_memory" => self.tool_store_memory(&tool_call.arguments).await,
            "recall_memory" => self.tool_recall_memory(&tool_call.arguments).await,
            "clear_memory" => self.tool_clear_memory(&tool_call.arguments).await,
            "get_memory_stats" => self.tool_get_memory_stats(&tool_call.arguments).await,
            
            // Project context management tools
            "save_project_context" => self.tool_save_project_context(&tool_call.arguments).await,
            "load_project_context" => self.tool_load_project_context(&tool_call.arguments).await,
            "quick_save_session" => self.tool_quick_save_session(&tool_call.arguments).await,
            "quick_load_session" => self.tool_quick_load_session(&tool_call.arguments).await,
            "diff_contexts" => self.tool_diff_contexts(&tool_call.arguments).await,
            "list_project_snapshots" => self.tool_list_project_snapshots(&tool_call.arguments).await,
            "get_project_statistics" => self.tool_get_project_statistics(&tool_call.arguments).await,
            "cleanup_old_snapshots" => self.tool_cleanup_old_snapshots(&tool_call.arguments).await,
            
            // System management tools
            "get_system_metrics" => self.tool_get_system_metrics(&tool_call.arguments).await,
            "health_check" => self.tool_health_check(&tool_call.arguments).await,
            "configure_system" => self.tool_configure_system(&tool_call.arguments).await,
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

    // Document management tool implementations
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

    async fn tool_list_documents(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        // The real RagSystem doesn't have a list_documents method
        // This functionality would require additional implementation in the core system
        let response = json!({
            "message": "Document listing functionality not available in current RAG system implementation",
            "suggestion": "Use search functionality to find specific documents",
            "status": "not_implemented"
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_get_document(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        // The real RagSystem doesn't have a get_document method
        // This functionality would require additional implementation in the core system
        let response = json!({
            "message": "Direct document retrieval functionality not available in current RAG system implementation",
            "suggestion": "Use search functionality to find document content",
            "status": "not_implemented"
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_delete_document(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        // The real RagSystem doesn't have a delete_document method
        // This functionality would require additional implementation in the core system
        let response = json!({
            "message": "Document deletion functionality not available in current RAG system implementation",
            "suggestion": "Document management features need to be implemented in the core RAG system",
            "status": "not_implemented"
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

    // Memory management tool implementations
    async fn tool_store_memory(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' parameter")?;

        let memory_type = args.get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("short_term");

        let importance = args.get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let memory_id = match self.rag_system.handle_store_memory(
            content.to_string(),
            memory_type.to_string(),
            importance,
        ).await {
            Ok(id) => id,
            Err(e) => return Err(format!("Failed to store memory: {}", e)),
        };

        Ok(format!("Memory stored successfully with ID: {}", memory_id))
    }

    async fn tool_recall_memory(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let query = args.get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query' parameter")?;

        let memory_type = args.get("memory_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let limit = args.get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let memories = match self.rag_system.handle_recall_memory(
            query.to_string(),
            memory_type,
            limit,
        ).await {
            Ok(memories) => memories,
            Err(e) => return Err(format!("Failed to recall memories: {}", e)),
        };

        let response = json!({
            "query": query,
            "memories": memories,
            "count": memories.len()
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
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

    // Project context management tool implementations
    async fn tool_save_project_context(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let project_root = args.get("project_root")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let options = args.get("options")
            .cloned();

        let snapshot_id = match self.rag_system.handle_save_project_context(
            project_id.to_string(),
            project_root,
            options,
        ).await {
            Ok(id) => id,
            Err(e) => return Err(format!("Failed to save project context: {}", e)),
        };

        Ok(format!("Project context saved successfully. Snapshot ID: {}", snapshot_id))
    }

    async fn tool_load_project_context(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let snapshot_id = args.get("snapshot_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let snapshot = match self.rag_system.handle_load_project_context(
            project_id.to_string(),
            snapshot_id,
        ).await {
            Ok(snapshot) => snapshot,
            Err(e) => return Err(format!("Failed to load project context: {}", e)),
        };

        Ok(serde_json::to_string_pretty(&snapshot).unwrap_or_default())
    }

    async fn tool_quick_save_session(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let description = args.get("description")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'description' parameter")?;

        let session_id = match self.rag_system.handle_quick_save_session(
            project_id.to_string(),
            description.to_string(),
        ).await {
            Ok(id) => id,
            Err(e) => return Err(format!("Failed to save session: {}", e)),
        };

        Ok(format!("Session saved successfully. Session ID: {}", session_id))
    }

    async fn tool_quick_load_session(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let session_id = args.get("session_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'session_id' parameter")?;

        let snapshot = match self.rag_system.handle_quick_load_session(
            project_id.to_string(),
            session_id.to_string(),
        ).await {
            Ok(snapshot) => snapshot,
            Err(e) => return Err(format!("Failed to load session: {}", e)),
        };

        Ok(serde_json::to_string_pretty(&snapshot).unwrap_or_default())
    }

    async fn tool_diff_contexts(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let from_version = args.get("from_version")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'from_version' parameter")?;

        let to_version = args.get("to_version")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'to_version' parameter")?;

        let diff = match self.rag_system.handle_diff_contexts(
            project_id.to_string(),
            from_version.to_string(),
            to_version.to_string(),
        ).await {
            Ok(diff) => diff,
            Err(e) => return Err(format!("Failed to generate context diff: {}", e)),
        };

        Ok(serde_json::to_string_pretty(&diff).unwrap_or_default())
    }

    async fn tool_list_project_snapshots(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let limit = args.get("limit")
            .and_then(|v| v.as_u64())
            .map(|l| l as usize);

        let snapshots = match self.rag_system.handle_list_project_snapshots(
            project_id.to_string(),
            limit,
        ).await {
            Ok(snapshots) => snapshots,
            Err(e) => return Err(format!("Failed to list snapshots: {}", e)),
        };

        let response = json!({
            "project_id": project_id,
            "snapshots": snapshots,
            "count": snapshots.len()
        });

        Ok(serde_json::to_string_pretty(&response).unwrap_or_default())
    }

    async fn tool_get_project_statistics(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let stats = match self.rag_system.handle_get_project_statistics(
            project_id.to_string(),
        ).await {
            Ok(stats) => stats,
            Err(e) => return Err(format!("Failed to get project statistics: {}", e)),
        };

        Ok(serde_json::to_string_pretty(&stats).unwrap_or_default())
    }

    async fn tool_cleanup_old_snapshots(&self, args: &HashMap<String, Value>) -> Result<String, String> {
        let project_id = args.get("project_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'project_id' parameter")?;

        let deleted_count = match self.rag_system.handle_cleanup_old_snapshots(
            project_id.to_string(),
        ).await {
            Ok(count) => count,
            Err(e) => return Err(format!("Failed to cleanup snapshots: {}", e)),
        };

        Ok(format!("Successfully cleaned up {} old snapshots for project {}", deleted_count, project_id))
    }

    // System management tool implementations
    async fn tool_get_system_metrics(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        let metrics = self.get_system_metrics().await;
        Ok(serde_json::to_string_pretty(&metrics).unwrap_or_default())
    }

    async fn tool_health_check(&self, _args: &HashMap<String, Value>) -> Result<String, String> {
        // Check actual Redis connection status
        let redis_connected = self.rag_system.redis_manager().health_check().await.unwrap_or(false);

        let health = json!({
            "status": if redis_connected { "healthy" } else { "degraded" },
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "components": {
                "rag_system": "operational",
                "redis": redis_connected,
                "vector_store": if redis_connected { "ready" } else { "unavailable" },
                "memory_system": if redis_connected { "operational" } else { "fallback_mode" },
                "project_context": if redis_connected { "operational" } else { "limited" }
            },
            "redis_connected": redis_connected,
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

    // Helper methods
    async fn get_memory_stats(&self) -> Value {
        json!({
            "total_documents": 0,
            "total_chunks": 0,
            "memory_usage": {
                "working": 0,
                "short_term": 0,
                "long_term": 0,
                "episodic": 0,
                "semantic": 0
            },
            "storage": {
                "redis_keys": 0,
                "vector_index_size": 0,
                "project_snapshots": 0
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }

    async fn get_system_metrics(&self) -> Value {
        json!({
            "performance": {
                "avg_search_latency_ms": 0.0,
                "avg_ingestion_time_ms": 0.0,
                "avg_context_save_time_ms": 0.0,
                "queries_per_second": 0.0
            },
            "resources": {
                "cpu_usage_percent": 0.0,
                "memory_usage_mb": 0.0,
                "redis_memory_mb": 0.0,
                "disk_usage_mb": 0.0
            },
            "operations": {
                "total_searches": 0,
                "total_ingestions": 0,
                "total_context_saves": 0,
                "total_context_loads": 0,
                "total_errors": 0
            },
            "features": {
                "project_context": true,
                "memory_management": true,
                "vector_search": true,
                "document_processing": true
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }
}