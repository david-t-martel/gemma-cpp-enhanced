use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use axum::response::Html;
use serde::{Deserialize, Serialize};
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Basic error type for the server
#[derive(Debug)]
enum ServerError {
    Io(std::io::Error),
    Network(String),
    Config(String),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::Io(e) => write!(f, "IO error: {}", e),
            ServerError::Network(e) => write!(f, "Network error: {}", e),
            ServerError::Config(e) => write!(f, "Configuration error: {}", e),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<std::io::Error> for ServerError {
    fn from(e: std::io::Error) -> Self {
        ServerError::Io(e)
    }
}

type ServerResult<T> = Result<T, ServerError>;

/// Simplified server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout: Duration,
    pub enable_cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            request_timeout: Duration::from_secs(30),
            enable_cors: true,
        }
    }
}

/// Application state shared between handlers
#[derive(Clone)]
struct AppState {
    config: Arc<ServerConfig>,
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    version: String,
}

/// Document ingestion request
#[derive(Deserialize)]
struct IngestRequest {
    content: String,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    source: Option<String>,
}

/// Document ingestion response
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
    include_metadata: bool,
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

    fn error(message: String) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            error: Some(message),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[tokio::main]
async fn main() -> ServerResult<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rag_redis_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting RAG-Redis Server (Standalone Mode)");

    // Load configuration
    let config = ServerConfig::default();
    let bind_addr = SocketAddr::from(([127, 0, 0, 1], config.port));
    info!("Server will bind to: {}", bind_addr);

    // Create application state
    let app_state = AppState {
        config: Arc::new(config.clone()),
    };

    // Build the application router
    let app = create_app(app_state);

    // Create server with graceful shutdown
    info!("Starting server on {}", bind_addr);
    let listener = tokio::net::TcpListener::bind(bind_addr).await
        .map_err(|e| ServerError::Network(format!("Failed to bind to {}: {}", bind_addr, e)))?;

    info!("RAG-Redis Server is running at http://{}", bind_addr);
    info!("Health check available at: http://{}/health", bind_addr);
    info!("Note: This is standalone mode - RAG system not initialized");

    // Run the server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| ServerError::Network(format!("Server error: {}", e)))?;

    info!("Server shutdown complete");
    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        // Root endpoint with API documentation
        .route("/", get(api_docs))
        // Health check
        .route("/health", get(health_check))
        // Document ingestion (placeholder)
        .route("/api/v1/documents", post(ingest_document))
        // Search endpoints (placeholder)
        .route("/api/v1/search", get(search_get))
        .route("/api/v1/search", post(search_post))
        // System status
        .route("/api/v1/status", get(system_status))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_headers(Any)
                        .allow_methods(Any),
                )
        )
        .with_state(state)
}

async fn api_docs() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG-Redis System API (Standalone Mode)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { display: inline-block; padding: 3px 8px; border-radius: 3px; font-weight: bold; color: white; }
        .get { background-color: #61affe; }
        .post { background-color: #49cc90; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
        pre { background: #f9f9f9; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>RAG-Redis System API</h1>
    <div class="warning">
        <strong>⚠️ Standalone Mode:</strong> This server is running in standalone mode without the full RAG system.
        All endpoints provide placeholder responses for testing the server framework.
    </div>

    <p>This is a working HTTP server that exposes REST API endpoints for what would be document ingestion and search operations.</p>

    <h2>Available Endpoints</h2>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/health</strong>
        <p>Health check endpoint that returns server status.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/v1/status</strong>
        <p>System configuration and runtime status.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <strong>/api/v1/documents</strong>
        <p>Document ingestion endpoint (placeholder implementation).</p>
        <pre>{
  "content": "Your document content here...",
  "metadata": {"source": "web", "author": "John Doe"},
  "source": "optional-source-identifier"
}</pre>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/v1/search?q=query&limit=10</strong>
        <p>Search endpoint using query parameters (placeholder implementation).</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <strong>/api/v1/search</strong>
        <p>Search endpoint using JSON payload (placeholder implementation).</p>
        <pre>{
  "query": "your search query",
  "limit": 10,
  "include_metadata": true
}</pre>
    </div>

    <h2>Features</h2>
    <ul>
        <li>✅ Concurrent request handling</li>
        <li>✅ Proper error handling and responses</li>
        <li>✅ Health check endpoints</li>
        <li>✅ CORS support</li>
        <li>✅ Request timeout handling</li>
        <li>✅ Structured logging</li>
        <li>✅ Graceful shutdown</li>
        <li>⚠️ Document processing (placeholder)</li>
        <li>⚠️ Vector search (placeholder)</li>
        <li>⚠️ Redis integration (placeholder)</li>
    </ul>

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
    "#)
}

async fn health_check(State(_state): State<AppState>) -> impl IntoResponse {
    let health = HealthResponse {
        status: "healthy (standalone mode)".to_string(),
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    (StatusCode::OK, Json(ApiResponse::success(health)))
}

async fn system_status(State(state): State<AppState>) -> impl IntoResponse {
    let status = serde_json::json!({
        "system": {
            "version": env!("CARGO_PKG_VERSION"),
            "mode": "standalone",
            "message": "Server framework operational - RAG system not initialized"
        },
        "server": {
            "host": state.config.host,
            "port": state.config.port,
            "max_connections": state.config.max_connections,
            "request_timeout_secs": state.config.request_timeout.as_secs()
        },
        "features": {
            "http_server": "operational",
            "health_checks": "operational",
            "error_handling": "operational",
            "cors": if state.config.enable_cors { "enabled" } else { "disabled" },
            "rag_system": "not_initialized",
            "redis": "not_connected",
            "vector_search": "not_available"
        }
    });

    Json(ApiResponse::success(status))
}

async fn ingest_document(
    State(_state): State<AppState>,
    Json(request): Json<IngestRequest>,
) -> impl IntoResponse {
    if request.content.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::error("Content cannot be empty".to_string())),
        ).into_response();
    }

    // Placeholder implementation
    let document_id = format!("doc_{}", uuid::Uuid::new_v4());
    let response = IngestResponse {
        document_id: document_id.clone(),
        chunks_processed: 0,
        message: format!("PLACEHOLDER: Document {} would be processed (standalone mode)", document_id),
    };

    info!("Placeholder ingestion: {} chars", request.content.len());

    (StatusCode::OK, Json(ApiResponse::success(response))).into_response()
}

async fn search_get(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    if params.q.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::error("Query parameter 'q' cannot be empty".to_string())),
        ).into_response();
    }

    let (status, response) = perform_search(state, params.q, params.limit).await;
    (status, response).into_response()
}

async fn search_post(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    if request.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::error("Query cannot be empty".to_string())),
        ).into_response();
    }

    let (status, response) = perform_search(state, request.query, request.limit).await;
    (status, response).into_response()
}

async fn perform_search(
    _state: AppState,
    query: String,
    limit: usize,
) -> (StatusCode, Json<ApiResponse<serde_json::Value>>) {
    info!("Placeholder search for query: '{}', limit: {}", query, limit);

    // Placeholder response with sample structure
    let response_data = serde_json::json!({
        "query": query,
        "results": [],
        "total_found": 0,
        "search_time_ms": 1,
        "message": "PLACEHOLDER: Search functionality not implemented (standalone mode)",
        "parameters": {
            "limit": limit,
            "actual_limit": 0
        }
    });

    (StatusCode::OK, Json(ApiResponse::success(response_data)))
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
