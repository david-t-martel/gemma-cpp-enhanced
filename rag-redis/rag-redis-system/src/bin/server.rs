use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::response::Html;
use axum::{
    extract::{Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use rag_redis_system::{Config, Error, RagSystem, Result};

/// Application state shared between handlers
#[derive(Clone)]
struct AppState {
    rag_system: Arc<RagSystem>,
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    version: String,
    redis_connected: bool,
    vector_store_ready: bool,
}

/// Document ingestion request
#[derive(Deserialize)]
struct IngestRequest {
    content: String,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    #[allow(dead_code)]
    source: Option<String>,
}

/// Document ingestion response
#[allow(dead_code)]
#[derive(Serialize)]
struct IngestResponse {
    document_id: String,
    chunks_processed: usize,
    message: String,
}

/// Search request
#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    #[allow(dead_code)]
    include_metadata: bool,
}

fn default_limit() -> usize {
    10
}

/// Search query parameters (for GET requests)
#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    #[allow(dead_code)]
    include_metadata: bool,
}

/// Research request
#[derive(Deserialize)]
struct ResearchRequest {
    query: String,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default = "default_limit")]
    #[allow(dead_code)]
    limit: usize,
}

/// Generic API response
#[derive(Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    fn error(message: impl Into<String>) -> Self {
        ApiResponse {
            success: false,
            data: None,
            error: Some(message.into()),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rag_redis_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting RAG-Redis Server");

    // Load configuration
    let config = match std::env::var("RAG_CONFIG_PATH") {
        Ok(path) => {
            info!("Loading configuration from: {}", path);
            Config::from_file(&path.into())?
        }
        Err(_) => {
            warn!("No RAG_CONFIG_PATH set, using default configuration");
            Config::default()
        }
    };

    // Validate configuration
    config.validate()?;

    let bind_addr = SocketAddr::from(([127, 0, 0, 1], config.server.port));
    info!("Server will bind to: {}", bind_addr);

    // Initialize RAG system
    info!("Initializing RAG system...");
    let rag_system = Arc::new(RagSystem::new(config.clone()).await?);
    info!("RAG system initialized successfully");

    // Create application state
    let app_state = AppState {
        rag_system: rag_system.clone(),
    };

    // Build the application router
    let app = create_app(app_state);

    // Create server with graceful shutdown
    info!("Starting server on {}", bind_addr);
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|e| Error::Network(format!("Failed to bind to {}: {}", bind_addr, e)))?;

    info!("RAG-Redis Server is running at http://{}", bind_addr);
    info!("Health check available at: http://{}/health", bind_addr);
    info!("API documentation available at: http://{}/", bind_addr);

    // Run the server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Error::Network(format!("Server error: {}", e)))?;

    info!("Server shutdown complete");
    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        // Root endpoint with API documentation
        .route("/", get(api_docs))
        // Health check
        .route("/health", get(health_check))
        // Document ingestion
        .route("/api/v1/documents", post(ingest_document))
        // Search endpoints
        .route("/api/v1/search", get(search_get))
        .route("/api/v1/search", post(search_post))
        // Research endpoint
        .route("/api/v1/research", post(research))
        // System status
        .route("/api/v1/status", get(system_status))
        // WebSocket endpoint for real-time updates
        .route("/ws", get(websocket_handler))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_headers(Any)
                        .allow_methods(Any),
                ),
        )
        .with_state(state)
}

async fn api_docs() -> Html<&'static str> {
    Html(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG-Redis System API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { display: inline-block; padding: 3px 8px; border-radius: 3px; font-weight: bold; color: white; }
        .get { background-color: #61affe; }
        .post { background-color: #49cc90; }
        code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
        pre { background: #f9f9f9; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>RAG-Redis System API</h1>
    <p>Welcome to the RAG-Redis System HTTP API. This system provides document ingestion, vector search, and research capabilities.</p>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/health</strong>
        <p>Health check endpoint that returns system status and connectivity information.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/v1/status</strong>
        <p>Detailed system status including memory usage, Redis connectivity, and performance metrics.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <strong>/api/v1/documents</strong>
        <p>Ingest a document into the system for indexing and future search.</p>
        <pre>{
  "content": "Your document content here...",
  "metadata": {"source": "web", "author": "John Doe"},
  "source": "optional-source-identifier"
}</pre>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/v1/search?q=query&limit=10</strong>
        <p>Search for documents using query parameters. Returns matching document chunks.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <strong>/api/v1/search</strong>
        <p>Search for documents using JSON payload. Provides more control over search parameters.</p>
        <pre>{
  "query": "your search query",
  "limit": 10,
  "include_metadata": true
}</pre>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <strong>/api/v1/research</strong>
        <p>Perform research combining local document search with optional web sources.</p>
        <pre>{
  "query": "research question",
  "sources": ["https://example.com", "https://source2.com"],
  "limit": 10
}</pre>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/ws</strong>
        <p>WebSocket endpoint for real-time updates and bidirectional communication.</p>
    </div>

    <h2>Response Format</h2>
    <p>All API endpoints return responses in the following format:</p>
    <pre>{
  "success": true,
  "data": { /* response data */ },
  "error": null,
  "timestamp": "2024-01-01T12:00:00Z"
}</pre>
</body>
</html>
    "#,
    )
}

async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    // Check Redis connectivity
    let redis_connected = state
        .rag_system
        .redis_manager()
        .redis_client()
        .ping()
        .await
        .is_ok();

    let health = HealthResponse {
        status: if redis_connected {
            "healthy"
        } else {
            "degraded"
        }
        .to_string(),
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        redis_connected,
        vector_store_ready: true, // Assume ready if we got this far
    };

    let status_code = if redis_connected {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status_code, Json(ApiResponse::success(health)))
}

async fn system_status(State(state): State<AppState>) -> impl IntoResponse {
    let redis_connected = state
        .rag_system
        .redis_manager()
        .redis_client()
        .ping()
        .await
        .is_ok();

    let status = serde_json::json!({
        "system": {
            "version": env!("CARGO_PKG_VERSION"),
            "uptime": "unknown", // Would need to track this
            "memory_usage": "unknown" // Would need system info
        },
        "components": {
            "redis": {
                "connected": redis_connected,
                "url": state.rag_system.config.redis.url
            },
            "vector_store": {
                "type": "HNSW",
                "dimension": state.rag_system.config.vector_store.dimension,
                "ready": true
            },
            "embedding": {
                "provider": format!("{:?}", state.rag_system.config.embedding.provider),
                "model": &state.rag_system.config.embedding.model,
                "dimension": state.rag_system.config.embedding.dimension
            }
        }
    });

    Json(ApiResponse::success(status))
}

async fn ingest_document(
    State(state): State<AppState>,
    Json(request): Json<IngestRequest>,
) -> impl IntoResponse {
    if request.content.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<serde_json::Value>::error(
                "Content cannot be empty",
            )),
        );
    }

    match state
        .rag_system
        .ingest_document(&request.content, request.metadata)
        .await
    {
        Ok(document_id) => {
            info!("Successfully ingested document: {}", document_id);
            let response = serde_json::json!({
                "document_id": document_id,
                "chunks_processed": 1,
                "message": "Document successfully ingested"
            });
            (StatusCode::CREATED, Json(ApiResponse::success(response)))
        }
        Err(e) => {
            error!("Failed to ingest document: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<serde_json::Value>::error(format!(
                    "Ingestion failed: {}",
                    e
                ))),
            )
        }
    }
}

async fn search_get(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    if params.q.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<serde_json::Value>::error(
                "Query parameter 'q' cannot be empty",
            )),
        );
    }

    perform_search(state, params.q, params.limit).await
}

async fn search_post(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    if request.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<serde_json::Value>::error(
                "Query cannot be empty",
            )),
        );
    }

    perform_search(state, request.query, request.limit).await
}

async fn perform_search(
    state: AppState,
    query: String,
    limit: usize,
) -> (StatusCode, Json<ApiResponse<serde_json::Value>>) {
    match state.rag_system.search(&query, limit).await {
        Ok(results) => {
            info!(
                "Search completed: {} results for query '{}'",
                results.len(),
                query
            );
            let response_data = serde_json::json!({
                "query": query,
                "results": results,
                "total_found": results.len()
            });
            (StatusCode::OK, Json(ApiResponse::success(response_data)))
        }
        Err(e) => {
            error!("Search failed for query '{}': {}", query, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<serde_json::Value>::error(format!(
                    "Search failed: {}",
                    e
                ))),
            )
        }
    }
}

async fn research(
    State(state): State<AppState>,
    Json(request): Json<ResearchRequest>,
) -> impl IntoResponse {
    if request.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<serde_json::Value>::error(
                "Query cannot be empty",
            )),
        );
    }

    match state
        .rag_system
        .research(&request.query, request.sources)
        .await
    {
        Ok(results) => {
            info!(
                "Research completed: {} results for query '{}'",
                results.len(),
                request.query
            );
            let response_data = serde_json::json!({
                "query": request.query,
                "results": results,
                "total_found": results.len()
            });
            (StatusCode::OK, Json(ApiResponse::success(response_data)))
        }
        Err(e) => {
            error!("Research failed for query '{}': {}", request.query, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<serde_json::Value>::error(format!(
                    "Research failed: {}",
                    e
                ))),
            )
        }
    }
}

async fn websocket_handler(ws: WebSocketUpgrade, State(_state): State<AppState>) -> Response {
    ws.on_upgrade(handle_websocket)
}

async fn handle_websocket(mut socket: axum::extract::ws::WebSocket) {
    use axum::extract::ws::Message;
    #[allow(unused_imports)]
    use futures::sink::SinkExt;
    #[allow(unused_imports)]
    use futures::stream::StreamExt;

    info!("WebSocket connection established");

    // Send welcome message
    if let Err(e) = socket
        .send(Message::Text(
            serde_json::json!({
                "type": "welcome",
                "message": "Connected to RAG-Redis System WebSocket",
                "timestamp": chrono::Utc::now()
            })
            .to_string(),
        ))
        .await
    {
        error!("Failed to send welcome message: {}", e);
        return;
    }

    // Handle incoming messages
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                info!("Received WebSocket message: {}", text);

                // Echo back for now - could implement real-time search, notifications, etc.
                let response = serde_json::json!({
                    "type": "echo",
                    "original_message": text,
                    "timestamp": chrono::Utc::now()
                });

                if let Err(e) = socket.send(Message::Text(response.to_string())).await {
                    error!("Failed to send WebSocket response: {}", e);
                    break;
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket connection closed by client");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    info!("WebSocket connection terminated");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down gracefully");
        },
        _ = terminate => {
            info!("Received SIGTERM, shutting down gracefully");
        },
    }
}
