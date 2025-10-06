use anyhow::{Context, Result};
use serde_json::json;
use std::env;
use std::io;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as AsyncBufReader};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod handlers;
mod protocol;
mod tools;

use handlers::McpHandler;
use protocol::{JsonRpcRequest, JsonRpcResponse};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    // Optimized for I/O-bound MCP operations with minimal worker threads
    // Initialize logging
    init_logging()?;

    info!("Starting RAG-Redis MCP Server v0.1.0");

    // Load configuration
    let config = load_config().await?;

    // Initialize MCP handler
    let handler = McpHandler::new(config).await
        .context("Failed to initialize MCP handler")?;

    info!("MCP server initialized successfully");

    // Run the server
    run_stdio_server(handler).await?;

    info!("MCP server shutting down");
    Ok(())
}

fn init_logging() -> Result<()> {
    // Set default log level if not specified
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(io::stderr)
                .with_ansi(false)
                .compact()
        )
        .init();

    Ok(())
}

async fn load_config() -> Result<rag_redis_system::Config> {
    // Try to load config from environment or use defaults
    let config = rag_redis_system::Config::default();

    info!("Using default configuration");
    debug!("Config: Redis URL from env or default");

    Ok(config)
}

async fn run_stdio_server(handler: McpHandler) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();

    let mut reader = AsyncBufReader::new(stdin);
    let mut line = String::new();

    info!("MCP server listening on stdio");

    loop {
        line.clear();

        match reader.read_line(&mut line).await {
            Ok(0) => {
                // EOF reached
                debug!("EOF reached, shutting down");
                break;
            }
            Ok(_) => {
                let trimmed_line = line.trim();
                if trimmed_line.is_empty() {
                    continue;
                }

                debug!("Received raw input: {}", trimmed_line);

                // Parse JSON-RPC request
                let request: JsonRpcRequest = match serde_json::from_str(trimmed_line) {
                    Ok(req) => req,
                    Err(e) => {
                        error!("Failed to parse JSON-RPC request: {}", e);
                        let error_response = JsonRpcResponse::error(
                            None,
                            protocol::JsonRpcError::parse_error(),
                        );
                        send_response(&mut stdout, &error_response).await?;
                        continue;
                    }
                };

                debug!("Parsed request: method={}, id={:?}", request.method, request.id);

                // Handle the request
                let response = handler.handle_request(request.clone()).await;

                // Only send response if this is not a notification (notifications have no id)
                if request.id.is_some() {
                    send_response(&mut stdout, &response).await?;
                } else {
                    debug!("Notification processed, no response sent");
                }
            }
            Err(e) => {
                error!("Error reading from stdin: {}", e);
                break;
            }
        }
    }

    Ok(())
}

async fn send_response(
    stdout: &mut tokio::io::Stdout,
    response: &JsonRpcResponse,
) -> Result<()> {
    let json_str = serde_json::to_string(response)
        .context("Failed to serialize JSON-RPC response")?;

    debug!("Sending response: {}", json_str);

    stdout.write_all(json_str.as_bytes()).await
        .context("Failed to write response to stdout")?;
    stdout.write_all(b"\n").await
        .context("Failed to write newline to stdout")?;
    stdout.flush().await
        .context("Failed to flush stdout")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{InitializeRequest, ClientCapabilities, ClientInfo};

    #[tokio::test]
    async fn test_handler_initialization() {
        let config = rag_redis_system::Config::default();
        let handler = McpHandler::new(config).await;
        assert!(handler.is_ok());
    }

    #[tokio::test]
    async fn test_initialize_request() {
        let config = rag_redis_system::Config::default();
        let handler = McpHandler::new(config).await.unwrap();

        let init_request = InitializeRequest {
            protocol_version: protocol::MCP_VERSION.to_string(),
            capabilities: ClientCapabilities {
                roots: None,
                sampling: None,
            },
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!(init_request)),
        };

        let response = handler.handle_request(request).await;
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_tools_list() {
        let config = rag_redis_system::Config::default();
        let handler = McpHandler::new(config).await.unwrap();

        // First initialize
        let init_request = InitializeRequest {
            protocol_version: protocol::MCP_VERSION.to_string(),
            capabilities: ClientCapabilities {
                roots: None,
                sampling: None,
            },
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        };

        let init_req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!(init_request)),
        };

        handler.handle_request(init_req).await;

        // Now test tools list
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/list".to_string(),
            params: None,
        };

        let response = handler.handle_request(request).await;
        assert!(response.result.is_some());
        assert!(response.error.is_none());

        let result = response.result.unwrap();
        assert!(result.get("tools").is_some());
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert!(!tools.is_empty());
    }

    #[tokio::test]
    async fn test_ping() {
        let config = rag_redis_system::Config::default();
        let handler = McpHandler::new(config).await.unwrap();

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "ping".to_string(),
            params: None,
        };

        let response = handler.handle_request(request).await;
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_unknown_method() {
        let config = rag_redis_system::Config::default();
        let handler = McpHandler::new(config).await.unwrap();

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "unknown_method".to_string(),
            params: None,
        };

        let response = handler.handle_request(request).await;
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, protocol::error_codes::METHOD_NOT_FOUND);
    }
}