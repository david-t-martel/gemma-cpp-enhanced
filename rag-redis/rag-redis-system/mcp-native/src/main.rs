use anyhow::{Context, Result};
use clap::Parser;
use jsonrpc_core::{IoHandler, Params, Request};
use rag_redis_system::{Config as RagConfig, RagSystem};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Parser)]
#[command(
    name = "RAG-Redis MCP Native Server",
    about = "Native Rust MCP server for RAG-Redis system",
    version = "0.1.0"
)]
struct Args {
    /// Redis URL to connect to
    #[arg(long, env = "REDIS_URL", default_value = "redis://localhost:6379")]
    redis_url: String,

    /// Embedding model to use
    #[arg(long, env = "EMBEDDING_MODEL", default_value = "sentence-transformers/all-MiniLM-L6-v2")]
    embedding_model: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "LOG_LEVEL", default_value = "info")]
    log_level: String,

    /// Enable pretty JSON output for debugging
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeRequest {
    protocol_version: String,
    capabilities: serde_json::Value,
    client_info: ClientInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClientInfo {
    name: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResponse {
    protocol_version: String,
    capabilities: ServerCapabilities,
    server_info: ServerInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ServerCapabilities {
    tools: ToolsCapability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolsCapability {
    list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ServerInfo {
    name: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Tool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolCallRequest {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolCallResponse {
    content: Vec<ContentItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentItem {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

struct McpServer {
    rag_system: Arc<RagSystem>,
    initialized: Arc<RwLock<bool>>,
}

impl McpServer {
    async fn new(args: &Args) -> Result<Self> {
        // Create RAG config with proper structure
        let mut config = RagConfig::default();
        config.redis.url = args.redis_url.clone();
        config.embedding.model = args.embedding_model.clone();

        // Initialize RAG system
        let rag_system = Arc::new(
            RagSystem::new(config)
                .await
                .context("Failed to create RAG system")?
        );

        Ok(Self {
            rag_system,
            initialized: Arc::new(RwLock::new(false)),
        })
    }

    fn create_rpc_handler(self: Arc<Self>) -> IoHandler {
        let mut handler = IoHandler::new();
        let server = self.clone();

        // Initialize method
        handler.add_method("initialize", move |params: Params| {
            let server = server.clone();
            Box::pin(async move {
                let request: InitializeRequest = params.parse()?;

                info!("Initializing MCP server with protocol version: {}", request.protocol_version);

                *server.initialized.write().await = true;

                let response = InitializeResponse {
                    protocol_version: "2024-11-05".to_string(),
                    capabilities: ServerCapabilities {
                        tools: ToolsCapability {
                            list_changed: false,
                        },
                    },
                    server_info: ServerInfo {
                        name: "rag-redis-mcp-native".to_string(),
                        version: "0.1.0".to_string(),
                    },
                };

                Ok(serde_json::to_value(response)
                    .map_err(|_| jsonrpc_core::Error::internal_error())?)
            })
        });

        let server = self.clone();

        // List tools method
        handler.add_method("tools/list", move |_params: Params| {
            let server = server.clone();
            Box::pin(async move {
                if !*server.initialized.read().await {
                    return Err(jsonrpc_core::Error::invalid_request());
                }

                let tools = vec![
                    Tool {
                        name: "rag_ingest_document".to_string(),
                        description: "Ingest a document into the RAG system".to_string(),
                        input_schema: json!({
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Document content to ingest"
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional metadata for the document"
                                }
                            },
                            "required": ["content"]
                        }),
                    },
                    Tool {
                        name: "rag_search".to_string(),
                        description: "Search for relevant documents".to_string(),
                        input_schema: json!({
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results",
                                    "default": 10
                                },
                                "threshold": {
                                    "type": "number",
                                    "description": "Similarity threshold",
                                    "default": 0.7
                                }
                            },
                            "required": ["query"]
                        }),
                    },
                    Tool {
                        name: "rag_memory_store".to_string(),
                        description: "Store information in memory".to_string(),
                        input_schema: json!({
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Content to store"
                                },
                                "memory_type": {
                                    "type": "string",
                                    "description": "Memory type (short_term, long_term, episodic, semantic, working)",
                                    "default": "short_term"
                                },
                                "importance": {
                                    "type": "number",
                                    "description": "Importance score (0-1)",
                                    "default": 0.5
                                }
                            },
                            "required": ["content"]
                        }),
                    },
                    Tool {
                        name: "rag_memory_recall".to_string(),
                        description: "Recall information from memory".to_string(),
                        input_schema: json!({
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Query for memory recall"
                                },
                                "memory_types": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Memory types to search",
                                    "default": ["all"]
                                }
                            }
                        }),
                    },
                    Tool {
                        name: "rag_research".to_string(),
                        description: "Research a topic using various sources".to_string(),
                        input_schema: json!({
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Research query"
                                },
                                "sources": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Sources to use (local, web)",
                                    "default": ["local"]
                                },
                                "combine_with_local": {
                                    "type": "boolean",
                                    "description": "Combine with local knowledge",
                                    "default": true
                                }
                            },
                            "required": ["query"]
                        }),
                    },
                ];

                Ok(json!({ "tools": tools }))
            })
        });

        let server = self.clone();

        // Tool call method
        handler.add_method("tools/call", move |params: Params| {
            let server = server.clone();
            Box::pin(async move {
                if !*server.initialized.read().await {
                    return Err(jsonrpc_core::Error::invalid_request());
                }

                let request: ToolCallRequest = params.parse()?;

                debug!("Tool call: {} with args: {}", request.name, request.arguments);

                let result = match request.name.as_str() {
                    "rag_ingest_document" => {
                        let content = request.arguments["content"]
                            .as_str()
                            .ok_or_else(|| jsonrpc_core::Error::invalid_params("content required"))?;
                        let metadata = request.arguments.get("metadata");

                        match server.rag_system.handle_ingest_document(
                            content.to_string(),
                            metadata.cloned()
                        ).await {
                            Ok(doc_id) => json!({ "document_id": doc_id }),
                            Err(e) => {
                                error!("Ingest error: {}", e);
                                json!({ "error": e.to_string() })
                            }
                        }
                    },

                    "rag_search" => {
                        let query = request.arguments["query"]
                            .as_str()
                            .ok_or_else(|| jsonrpc_core::Error::invalid_params("query required"))?;
                        let limit = request.arguments.get("limit")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(10) as usize;
                        let threshold = request.arguments.get("threshold")
                            .and_then(|v| v.as_f64())
                            .map(|v| v as f32);

                        match server.rag_system.handle_search(
                            query.to_string(),
                            limit,
                            threshold
                        ).await {
                            Ok(results) => json!(results),
                            Err(e) => {
                                error!("Search error: {}", e);
                                json!({ "error": e.to_string() })
                            }
                        }
                    },

                    "rag_memory_store" => {
                        let content = request.arguments["content"]
                            .as_str()
                            .ok_or_else(|| jsonrpc_core::Error::invalid_params("content required"))?;
                        let memory_type = request.arguments.get("memory_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("short_term");
                        let importance = request.arguments.get("importance")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.5) as f32;

                        match server.rag_system.handle_store_memory(
                            content.to_string(),
                            memory_type.to_string(),
                            importance
                        ).await {
                            Ok(memory_id) => json!({ "memory_id": memory_id }),
                            Err(e) => {
                                error!("Memory store error: {}", e);
                                json!({ "error": e.to_string() })
                            }
                        }
                    },

                    "rag_memory_recall" => {
                        let query = request.arguments.get("query")
                            .and_then(|v| v.as_str());
                        let memory_types = request.arguments.get("memory_types")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect::<Vec<_>>()
                            });

                        match server.rag_system.handle_recall_memory(
                            query.unwrap_or("").to_string(),
                            memory_types.as_ref().and_then(|types| types.first().cloned()),
                            10
                        ).await {
                            Ok(memories) => json!(memories),
                            Err(e) => {
                                error!("Memory recall error: {}", e);
                                json!({ "error": e.to_string() })
                            }
                        }
                    },

                    "rag_research" => {
                        let query = request.arguments["query"]
                            .as_str()
                            .ok_or_else(|| jsonrpc_core::Error::invalid_params("query required"))?;
                        let sources = request.arguments.get("sources")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect::<Vec<_>>()
                            });
                        let _combine_with_local = request.arguments.get("combine_with_local")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);

                        let sources = sources.unwrap_or_else(|| vec!["local".to_string()]);
                        match server.rag_system.research(
                            query,
                            sources
                        ).await {
                            Ok(results) => json!(results),
                            Err(e) => {
                                error!("Research error: {}", e);
                                json!({ "error": e.to_string() })
                            }
                        }
                    },

                    _ => {
                        warn!("Unknown tool: {}", request.name);
                        return Err(jsonrpc_core::Error::method_not_found());
                    }
                };

                let response = ToolCallResponse {
                    content: vec![ContentItem {
                        content_type: "text".to_string(),
                        text: serde_json::to_string(&result)
                            .map_err(|_| jsonrpc_core::Error::internal_error())?,
                    }],
                };

                Ok(serde_json::to_value(response)
                    .map_err(|_| jsonrpc_core::Error::internal_error())?)
            })
        });

        // Handle notifications
        handler.add_notification("notifications/initialized", |_params: Params| {
            info!("Client initialized notification received");
        });

        handler
    }

    async fn run_stdio_loop(self: Arc<Self>, pretty: bool) -> Result<()> {
        let handler = self.create_rpc_handler();

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);

        info!("MCP Native server started, waiting for requests...");

        let mut line = String::new();
        loop {
            line.clear();

            // Read a line from stdin
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    // EOF
                    info!("Received EOF, shutting down");
                    break;
                },
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    debug!("Received request: {}", trimmed);

                    // Parse and handle the JSON-RPC request
                    match serde_json::from_str::<Request>(trimmed) {
                        Ok(request) => {
                            // Handle the request
                            let response = handler.handle_rpc_request(request);

                            // Wait for the future to complete
                            if let Some(response_result) = response.await {
                                // Format response
                                let output = if pretty {
                                    serde_json::to_string_pretty(&response_result)?
                                } else {
                                    serde_json::to_string(&response_result)?
                                };

                                debug!("Sending response: {}", output);

                                // Write response
                                stdout.write_all(output.as_bytes()).await?;
                                stdout.write_all(b"\n").await?;
                                stdout.flush().await?;
                            }
                        },
                        Err(e) => {
                            error!("Failed to parse JSON-RPC request: {}", e);

                            // Send error response
                            let error_response = json!({
                                "jsonrpc": "2.0",
                                "error": {
                                    "code": -32700,
                                    "message": "Parse error",
                                    "data": e.to_string()
                                },
                                "id": null
                            });

                            let output = if pretty {
                                serde_json::to_string_pretty(&error_response)?
                            } else {
                                serde_json::to_string(&error_response)?
                            };

                            stdout.write_all(output.as_bytes()).await?;
                            stdout.write_all(b"\n").await?;
                            stdout.flush().await?;
                        }
                    }
                },
                Err(e) => {
                    error!("Error reading from stdin: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = args.log_level.parse::<tracing::Level>()
        .unwrap_or(tracing::Level::INFO);

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .with_writer(std::io::stderr)
        .compact()
        .init();

    info!("Starting RAG-Redis MCP Native Server v0.1.0");
    debug!("Redis URL: {}", args.redis_url);
    debug!("Embedding model: {}", args.embedding_model);

    // Create and run server
    let server = Arc::new(
        McpServer::new(&args)
            .await
            .context("Failed to create MCP server")?
    );

    // Run the stdio loop
    server.run_stdio_loop(args.pretty)
        .await
        .context("MCP server error")?;

    info!("MCP Native server shut down");
    Ok(())
}
