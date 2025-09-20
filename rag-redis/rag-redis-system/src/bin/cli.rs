use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use rag_redis_system::{Config, Error, RagSystem, Result};

/// RAG-Redis System Command Line Interface
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, env = "RAG_CONFIG_PATH")]
    config: Option<PathBuf>,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Output format (json, pretty, plain)
    #[arg(long, default_value = "pretty")]
    output: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone)]
enum OutputFormat {
    Json,
    Pretty,
    Plain,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "pretty" => Ok(OutputFormat::Pretty),
            "plain" => Ok(OutputFormat::Plain),
            _ => Err(format!("Unknown output format: {}", s)),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Document ingestion commands
    Ingest(IngestArgs),
    /// Search commands
    Search(SearchArgs),
    /// Research commands
    Research(ResearchArgs),
    /// System status and health checks
    Status(StatusArgs),
    /// Configuration management
    Config(ConfigArgs),
}

#[derive(Args)]
struct IngestArgs {
    /// Content to ingest (or file path with --file)
    content: Option<String>,

    /// Read content from file
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Additional metadata as JSON
    #[arg(short, long)]
    metadata: Option<String>,

    /// Source identifier
    #[arg(short, long)]
    source: Option<String>,

    /// Batch ingest from directory
    #[arg(long)]
    batch_dir: Option<PathBuf>,

    /// File extensions to include in batch (comma-separated)
    #[arg(long, default_value = "txt,md,html,json")]
    extensions: String,
}

#[derive(Args)]
struct SearchArgs {
    /// Search query
    query: String,

    /// Maximum number of results
    #[arg(short, long, default_value = "10")]
    limit: usize,

    /// Include metadata in results
    #[arg(long)]
    include_metadata: bool,

    /// Minimum similarity score (0.0-1.0)
    #[arg(long)]
    min_score: Option<f32>,
}

#[derive(Args)]
struct ResearchArgs {
    /// Research query
    query: String,

    /// Web sources to search (URLs)
    #[arg(short, long)]
    sources: Vec<String>,

    /// Maximum number of results
    #[arg(short, long, default_value = "10")]
    limit: usize,

    /// Save results to file
    #[arg(long)]
    save: Option<PathBuf>,
}

#[derive(Args)]
struct StatusArgs {
    /// Show detailed system information
    #[arg(long)]
    detailed: bool,

    /// Check specific component (redis, vector_store, embedding)
    #[arg(long)]
    component: Option<String>,

    /// Continuous monitoring (refresh interval in seconds)
    #[arg(long)]
    watch: Option<u64>,
}

#[derive(Args)]
struct ConfigArgs {
    #[command(subcommand)]
    action: ConfigAction,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Validate configuration
    Validate,
    /// Generate default configuration file
    Generate {
        /// Output file path
        #[arg(short, long, default_value = "rag-config.json")]
        output: PathBuf,
    },
    /// Test configuration components
    Test {
        /// Component to test (redis, embedding, all)
        #[arg(default_value = "all")]
        component: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("rag_redis_cli={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = load_config(cli.config.as_deref()).await?;

    // Handle commands that don't need RagSystem initialization
    if let Commands::Config(config_args) = &cli.command {
        return handle_config_command(config_args, &config, &cli.output).await;
    }

    // Initialize RagSystem for other commands
    let rag_system = Arc::new(RagSystem::new(config).await?);
    info!("RAG system initialized successfully");

    // Execute command
    let result = match cli.command {
        Commands::Ingest(args) => handle_ingest(rag_system, args).await,
        Commands::Search(args) => handle_search(rag_system, args).await,
        Commands::Research(args) => handle_research(rag_system, args).await,
        Commands::Status(args) => handle_status(rag_system, args).await,
        Commands::Config(_) => unreachable!(), // Handled above
    };

    match result {
        Ok(output) => {
            print_output(&output, &cli.output);
            Ok(())
        }
        Err(e) => {
            error!("Command failed: {}", e);
            std::process::exit(1);
        }
    }
}

async fn load_config(config_path: Option<&std::path::Path>) -> Result<Config> {
    match config_path {
        Some(path) => {
            info!("Loading configuration from: {}", path.display());
            Config::from_file(&path.to_path_buf())
        }
        None => {
            if let Ok(path) = std::env::var("RAG_CONFIG_PATH") {
                info!("Loading configuration from environment: {}", path);
                Config::from_file(&PathBuf::from(path))
            } else {
                warn!("No configuration file specified, using defaults");
                Ok(Config::default())
            }
        }
    }
}

async fn handle_config_command(
    args: &ConfigArgs,
    config: &Config,
    output_format: &OutputFormat,
) -> Result<()> {
    match &args.action {
        ConfigAction::Show => {
            let output = json!({
                "config": config,
                "loaded_from": "default_or_file"
            });
            print_output(&output, output_format);
        }
        ConfigAction::Validate => match config.validate() {
            Ok(()) => {
                let output = json!({
                    "status": "valid",
                    "message": "Configuration is valid"
                });
                print_output(&output, output_format);
            }
            Err(e) => {
                let output = json!({
                    "status": "invalid",
                    "error": e.to_string()
                });
                print_output(&output, output_format);
                return Err(e);
            }
        },
        ConfigAction::Generate {
            output: output_path,
        } => {
            let default_config = Config::default();
            let config_json = serde_json::to_string_pretty(&default_config)?;
            std::fs::write(output_path, config_json)?;

            let output = json!({
                "status": "generated",
                "file": output_path.to_string_lossy(),
                "message": "Default configuration file generated"
            });
            print_output(&output, output_format);
        }
        ConfigAction::Test { component } => {
            let results = test_configuration(config, component).await;
            print_output(&results, output_format);

            // Exit with error if any tests failed
            if let Some(overall_success) = results.get("success").and_then(|v| v.as_bool()) {
                if !overall_success {
                    return Err(Error::Config("Configuration test failed".to_string()));
                }
            }
        }
    }
    Ok(())
}

async fn test_configuration(config: &Config, component: &str) -> Value {
    let mut results = serde_json::Map::new();
    let mut overall_success = true;

    if component == "all" || component == "redis" {
        info!("Testing Redis connection...");
        match rag_redis_system::RedisManager::new(&config.redis).await {
            Ok(redis_manager) => match redis_manager.redis_client().ping().await {
                Ok(_) => {
                    results.insert(
                        "redis".to_string(),
                        json!({
                            "status": "success",
                            "message": "Redis connection successful"
                        }),
                    );
                }
                Err(e) => {
                    overall_success = false;
                    results.insert(
                        "redis".to_string(),
                        json!({
                            "status": "error",
                            "message": format!("Redis ping failed: {}", e)
                        }),
                    );
                }
            },
            Err(e) => {
                overall_success = false;
                results.insert(
                    "redis".to_string(),
                    json!({
                        "status": "error",
                        "message": format!("Redis connection failed: {}", e)
                    }),
                );
            }
        }
    }

    if component == "all" || component == "embedding" {
        info!("Testing embedding service...");
        // This would require initializing the embedding service
        results.insert(
            "embedding".to_string(),
            json!({
                "status": "skipped",
                "message": "Embedding service test not implemented"
            }),
        );
    }

    results.insert("success".to_string(), json!(overall_success));
    results.insert("timestamp".to_string(), json!(chrono::Utc::now()));

    json!(results)
}

async fn handle_ingest(rag_system: Arc<RagSystem>, args: IngestArgs) -> Result<Value> {
    if let Some(batch_dir) = args.batch_dir {
        return handle_batch_ingest(rag_system, batch_dir, &args.extensions).await;
    }

    let content = if let Some(file_path) = args.file {
        info!("Reading content from file: {}", file_path.display());
        std::fs::read_to_string(file_path)?
    } else if let Some(content) = args.content {
        content
    } else {
        return Err(Error::InvalidInput(
            "Must provide either content or --file".to_string(),
        ));
    };

    let metadata = if let Some(metadata_str) = args.metadata {
        serde_json::from_str(&metadata_str)
            .map_err(|e| Error::InvalidInput(format!("Invalid metadata JSON: {}", e)))?
    } else {
        json!({})
    };

    if let Some(source) = args.source {
        let mut meta_obj = metadata
            .as_object()
            .unwrap_or(&serde_json::Map::new())
            .clone();
        meta_obj.insert("source".to_string(), json!(source));
        let _metadata = json!(meta_obj);
    }

    info!("Ingesting document...");
    let document_id = rag_system
        .ingest_document(&content, metadata.clone())
        .await?;

    Ok(json!({
        "status": "success",
        "document_id": document_id,
        "content_length": content.len(),
        "metadata": metadata,
        "timestamp": chrono::Utc::now()
    }))
}

async fn handle_batch_ingest(
    rag_system: Arc<RagSystem>,
    dir_path: PathBuf,
    extensions: &str,
) -> Result<Value> {
    let exts: Vec<&str> = extensions.split(',').map(|s| s.trim()).collect();

    info!(
        "Starting batch ingestion from directory: {}",
        dir_path.display()
    );

    let mut results = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;

    for entry in std::fs::read_dir(&dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        if let Some(ext) = path.extension() {
            if !exts.contains(&ext.to_string_lossy().as_ref()) {
                continue;
            }
        }

        info!("Processing file: {}", path.display());

        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let metadata = json!({
                    "source": "batch_ingest",
                    "file_path": path.to_string_lossy(),
                    "file_name": path.file_name().unwrap_or_default().to_string_lossy()
                });

                match rag_system.ingest_document(&content, metadata.clone()).await {
                    Ok(document_id) => {
                        success_count += 1;
                        results.push(json!({
                            "file": path.to_string_lossy(),
                            "status": "success",
                            "document_id": document_id
                        }));
                    }
                    Err(e) => {
                        error_count += 1;
                        error!("Failed to ingest {}: {}", path.display(), e);
                        results.push(json!({
                            "file": path.to_string_lossy(),
                            "status": "error",
                            "error": e.to_string()
                        }));
                    }
                }
            }
            Err(e) => {
                error_count += 1;
                error!("Failed to read {}: {}", path.display(), e);
                results.push(json!({
                    "file": path.to_string_lossy(),
                    "status": "error",
                    "error": format!("Failed to read file: {}", e)
                }));
            }
        }
    }

    Ok(json!({
        "status": "completed",
        "directory": dir_path.to_string_lossy(),
        "total_processed": results.len(),
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
        "timestamp": chrono::Utc::now()
    }))
}

async fn handle_search(rag_system: Arc<RagSystem>, args: SearchArgs) -> Result<Value> {
    info!("Searching for: '{}'", args.query);

    let mut results = rag_system.search(&args.query, args.limit).await?;

    // Apply minimum score filter if specified
    if let Some(min_score) = args.min_score {
        results.retain(|r| r.score >= min_score);
    }

    Ok(json!({
        "query": args.query,
        "total_found": results.len(),
        "results": results,
        "parameters": {
            "limit": args.limit,
            "include_metadata": args.include_metadata,
            "min_score": args.min_score
        },
        "timestamp": chrono::Utc::now()
    }))
}

async fn handle_research(rag_system: Arc<RagSystem>, args: ResearchArgs) -> Result<Value> {
    info!("Researching: '{}'", args.query);

    let results = rag_system
        .research(&args.query, args.sources.clone())
        .await?;

    let output = json!({
        "query": args.query,
        "total_found": results.len(),
        "results": results,
        "parameters": {
            "sources": args.sources,
            "limit": args.limit
        },
        "timestamp": chrono::Utc::now()
    });

    // Save results if requested
    if let Some(save_path) = args.save {
        let formatted_output = serde_json::to_string_pretty(&output)?;
        std::fs::write(&save_path, formatted_output)?;
        info!("Results saved to: {}", save_path.display());
    }

    Ok(output)
}

async fn handle_status(rag_system: Arc<RagSystem>, args: StatusArgs) -> Result<Value> {
    // Simplified: ignore continuous watch mode in this build to avoid recursive async.

    let redis_connected = rag_system
        .redis_manager()
        .redis_client()
        .ping()
        .await
        .is_ok();

    let mut status = json!({
        "system": {
            "version": env!("CARGO_PKG_VERSION"),
            "status": if redis_connected { "healthy" } else { "degraded" }
        },
        "components": {
            "redis": {
                "connected": redis_connected,
                "url": rag_system.config.redis.url
            },
            "vector_store": {
                "ready": true,
                "dimension": rag_system.config.vector_store.dimension
            }
        },
        "timestamp": chrono::Utc::now()
    });

    if args.detailed {
        let detailed_info = json!({
            "configuration": {
                "redis_pool_size": rag_system.config.redis.pool_size,
                "vector_store_type": format!("{:?}", rag_system.config.vector_store.index_type),
                "embedding_provider": format!("{:?}", rag_system.config.embedding.provider),
                "embedding_model": rag_system.config.embedding.model,
                "server_config": {
                    "host": rag_system.config.server.host,
                    "port": rag_system.config.server.port
                }
            }
        });

        if let Some(obj) = status.as_object_mut() {
            obj.insert("detailed".to_string(), detailed_info);
        }
    }

    if let Some(component) = args.component {
        // Filter to show only specific component
        match component.as_str() {
            "redis" => {
                return Ok(json!({
                    "component": "redis",
                    "status": status["components"]["redis"].clone(),
                    "timestamp": chrono::Utc::now()
                }));
            }
            "vector_store" => {
                return Ok(json!({
                    "component": "vector_store",
                    "status": status["components"]["vector_store"].clone(),
                    "timestamp": chrono::Utc::now()
                }));
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "Unknown component: {}",
                    component
                )));
            }
        }
    }

    Ok(status)
}

// Continuous watch mode removed for simplified Windows build; could be reintroduced with a non-recursive task loop.

fn print_output(output: &Value, format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string(output).unwrap_or_default());
        }
        OutputFormat::Pretty => {
            println!(
                "{}",
                serde_json::to_string_pretty(output).unwrap_or_default()
            );
        }
        OutputFormat::Plain => {
            // Simple plain text output for common cases
            if let Some(obj) = output.as_object() {
                for (key, value) in obj {
                    match value {
                        Value::String(s) => println!("{}: {}", key, s),
                        Value::Number(n) => println!("{}: {}", key, n),
                        Value::Bool(b) => println!("{}: {}", key, b),
                        _ => println!("{}: {}", key, value),
                    }
                }
            } else {
                println!("{}", output);
            }
        }
    }
}
