//! High-performance Axum-based HTTP server for Gemma inference
//!
//! This server provides a REST API for neural network inference with focus on:
//! - High throughput and low latency
//! - Async request processing with batching
//! - Comprehensive monitoring and health checks
//! - Graceful shutdown and error handling
//! - Security with rate limiting and authentication

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::signal;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{info, warn, error, instrument, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use gemma_inference::{
    InferenceEngine, InferenceConfig, InferenceRequest, InferenceResponse,
    EngineConfig, get_runtime_capabilities, initialize_engine,
};

mod handlers;
mod middleware_auth;
mod metrics_collector;
mod rate_limiter;
mod websocket;
mod health;
mod config;

use handlers::*;
use config::ServerConfig;

/// Command-line arguments for the server
#[derive(Parser, Debug)]
#[command(name = "gemma-server")]
#[command(about = "High-performance inference server for Gemma models")]
#[command(version)]
pub struct Args {
    /// Server configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Server bind address
    #[arg(short, long, default_value = "127.0.0.1")]
    host: String,

    /// Server bind port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Number of worker threads
    #[arg(short, long, default_value = "0")]
    workers: usize,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Enable metrics endpoint
    #[arg(long)]
    enable_metrics: bool,

    /// Enable debug mode
    #[arg(long)]
    debug: bool,

    /// Model path or name
    #[arg(short, long)]
    model: Option<String>,
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub inference_engine: Arc<InferenceEngine>,
    pub config: Arc<ServerConfig>,
    pub metrics: Arc<metrics_collector::MetricsCollector>,
}

/// Server information response
#[derive(Serialize)]
pub struct ServerInfo {
    name: &'static str,
    version: &'static str,
    uptime: Duration,
    capabilities: serde_json::Value,
    config: ServerConfigResponse,
}

/// Server configuration for API responses
#[derive(Serialize)]
pub struct ServerConfigResponse {
    max_batch_size: usize,
    timeout_seconds: u64,
    rate_limit_enabled: bool,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
    timestamp: chrono::DateTime<chrono::Utc>,
    version: &'static str,
    checks: HealthChecks,
}

/// Individual health checks
#[derive(Serialize)]
pub struct HealthChecks {
    inference_engine: &'static str,
    memory_usage: MemoryUsage,
    gpu_status: Option<&'static str>,
}

/// Memory usage information
#[derive(Serialize)]
pub struct MemoryUsage {
    heap_used: u64,
    heap_total: u64,
    system_available: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    init_tracing(&args.log_level)?;
    info!("Starting Gemma inference server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = load_config(&args).await?;
    info!("Loaded configuration from: {}", args.config);

    // Set up tokio runtime with optimal worker count
    let worker_threads = if args.workers == 0 {
        std::thread::available_parallelism()?.get()
    } else {
        args.workers
    };

    info!("Using {} worker threads", worker_threads);

    // Initialize inference engine
    let inference_config = EngineConfig::from_server_config(&config);
    let engine = initialize_engine("main", inference_config)
        .context("Failed to initialize inference engine")?;

    // Warm up the engine
    gemma_inference::warmup_engine(&engine).await
        .context("Failed to warm up inference engine")?;

    // Initialize metrics collector
    let metrics = Arc::new(metrics_collector::MetricsCollector::new());

    // Create shared application state
    let state = AppState {
        inference_engine: engine,
        config: Arc::new(config),
        metrics,
    };

    // Build the application router
    let app = create_app(state.clone()).await?;

    // Bind to socket address
    let addr = SocketAddr::new(args.host.parse()?, args.port);
    info!("Server starting on {}", addr);

    // Create the server with graceful shutdown
    let server = axum_server::bind(addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal());

    // Start metrics server if enabled
    if args.enable_metrics || state.config.enable_metrics {
        tokio::spawn(start_metrics_server(state.metrics.clone()));
    }

    // Start the server
    info!("🚀 Gemma inference server ready!");
    server.await.context("Server error")?;

    info!("Server shutting down gracefully");
    Ok(())
}

/// Initialize structured logging
fn init_tracing(log_level: &str) -> Result<()> {
    let level = match log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("gemma_server={},tower_http=debug", level).into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    Ok(())
}

/// Load server configuration
async fn load_config(args: &Args) -> Result<ServerConfig> {
    let mut config = ServerConfig::default();

    // Try to load from config file
    if std::path::Path::new(&args.config).exists() {
        let config_str = tokio::fs::read_to_string(&args.config).await
            .with_context(|| format!("Failed to read config file: {}", args.config))?;

        config = toml::from_str(&config_str)
            .with_context(|| format!("Failed to parse config file: {}", args.config))?;
    }

    // Override with command-line arguments
    config.server.host = args.host.clone();
    config.server.port = args.port;

    if let Some(model) = &args.model {
        config.model.path = model.clone();
    }

    Ok(config)
}

/// Create the main application router
async fn create_app(state: AppState) -> Result<Router> {
    let cors = CorsLayer::new()
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers(Any)
        .allow_origin(Any);

    let middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(Duration::from_secs(state.config.server.timeout_seconds)))
        .layer(cors)
        .layer(CompressionLayer::new())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limiter::rate_limit_middleware,
        ));

    let app = Router::new()
        // Health and info endpoints
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/live", get(liveness_check))
        .route("/info", get(server_info))
        .route("/capabilities", get(capabilities))

        // Core inference endpoints
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/batch", post(batch_inference))

        // Model management
        .route("/v1/models", get(list_models))
        .route("/v1/models/:model_id", get(get_model_info))

        // Streaming endpoints
        .route("/v1/completions/stream", post(stream_completions))
        .route("/v1/chat/completions/stream", post(stream_chat_completions))

        // WebSocket endpoint for real-time inference
        .route("/ws/inference", get(websocket::websocket_handler))

        // Administrative endpoints
        .route("/admin/stats", get(get_stats))
        .route("/admin/config", get(get_config))
        .route("/admin/warmup", post(warmup_handler))

        .layer(middleware_stack)
        .with_state(state);

    Ok(app)
}

/// Handle server info requests
#[instrument(skip(state))]
async fn server_info(State(state): State<AppState>) -> Json<ServerInfo> {
    let uptime = Duration::from_secs(0); // TODO: Track actual uptime
    let capabilities = serde_json::to_value(get_runtime_capabilities()).unwrap_or_default();

    Json(ServerInfo {
        name: "gemma-server",
        version: env!("CARGO_PKG_VERSION"),
        uptime,
        capabilities,
        config: ServerConfigResponse {
            max_batch_size: state.config.inference.max_batch_size,
            timeout_seconds: state.config.server.timeout_seconds,
            rate_limit_enabled: state.config.security.rate_limit_enabled,
        },
    })
}

/// Handle capabilities requests
#[instrument]
async fn capabilities() -> Json<serde_json::Value> {
    Json(serde_json::to_value(get_runtime_capabilities()).unwrap_or_default())
}

/// Start the metrics server on a separate port
async fn start_metrics_server(metrics: Arc<metrics_collector::MetricsCollector>) {
    let metrics_router = Router::new()
        .route("/metrics", get(move || async move {
            metrics.export_prometheus()
        }));

    let addr = SocketAddr::from(([127, 0, 0, 1], 9090));
    info!("Metrics server starting on {}", addr);

    if let Err(e) = axum_server::bind(addr)
        .serve(metrics_router.into_make_service())
        .await
    {
        error!("Metrics server error: {}", e);
    }
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
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

    // Perform cleanup
    gemma_inference::shutdown_engines();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::util::ServiceExt;
    use http_body_util::BodyExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let config = ServerConfig::default();
        let engine_config = EngineConfig::from_server_config(&config);
        let engine = initialize_engine("test", engine_config).unwrap();
        let metrics = Arc::new(metrics_collector::MetricsCollector::new());

        let state = AppState {
            inference_engine: engine,
            config: Arc::new(config),
            metrics,
        };

        let app = create_app(state).await.unwrap();

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "healthy");
    }

    #[tokio::test]
    async fn test_info_endpoint() {
        let config = ServerConfig::default();
        let engine_config = EngineConfig::from_server_config(&config);
        let engine = initialize_engine("test", engine_config).unwrap();
        let metrics = Arc::new(metrics_collector::MetricsCollector::new());

        let state = AppState {
            inference_engine: engine,
            config: Arc::new(config),
            metrics,
        };

        let app = create_app(state).await.unwrap();

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/info")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let info: ServerInfo = serde_json::from_slice(&body).unwrap();
        assert_eq!(info.name, "gemma-server");
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
    }
}
