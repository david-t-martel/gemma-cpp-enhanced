use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub redis: RedisConfig,
    pub vector_store: VectorStoreConfig,
    pub document: DocumentConfig,
    pub memory: MemoryConfig,
    pub research: ResearchConfig,
    pub embedding: EmbeddingConfig,
    #[cfg(feature = "metrics")]
    pub metrics: MetricsConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: u32,
    pub connection_timeout: Duration,
    pub command_timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub enable_cluster: bool,
    /// If true, automatically falls back to an in-memory store when Redis
    /// cannot be reached at startup. This is helpful on Windows where a
    /// local Redis service may not yet be initialized or when running
    /// in ephemeral CI environments.
    #[serde(default = "default_enable_fallback")]
    pub enable_fallback: bool,
}

impl Default for RedisConfig {
    fn default() -> Self {
        // Auto-detect environment and use appropriate default port
        let default_url = if std::env::var("DOCKER_CONTAINER").is_ok()
            || std::path::Path::new("/.dockerenv").exists()
        {
            // Docker environment - use standard Redis port with service name
            "redis://redis:6379".to_string()
        } else if cfg!(target_os = "windows") {
            // Windows environment - use port 6380 to avoid conflicts
            "redis://127.0.0.1:6380".to_string()
        } else {
            // Linux/Unix environment - use standard port
            "redis://127.0.0.1:6379".to_string()
        };

        // Override with environment variable if set
        let url = std::env::var("REDIS_URL").unwrap_or(default_url);

        Self {
            url,
            pool_size: std::env::var("REDIS_POOL_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            connection_timeout: Duration::from_secs(
                std::env::var("REDIS_CONNECTION_TIMEOUT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5),
            ),
            command_timeout: Duration::from_secs(
                std::env::var("REDIS_COMMAND_TIMEOUT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10),
            ),
            max_retries: std::env::var("REDIS_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            retry_delay: Duration::from_millis(
                std::env::var("REDIS_RETRY_DELAY_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(100),
            ),
            enable_cluster: std::env::var("REDIS_ENABLE_CLUSTER")
                .ok()
                .map(|s| s.to_lowercase() == "true")
                .unwrap_or(false),
            enable_fallback: std::env::var("REDIS_ENABLE_FALLBACK")
                .ok()
                .map(|s| s.to_lowercase() != "false")
                .unwrap_or(true),
        }
    }
}

fn default_enable_fallback() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub index_type: IndexType,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub max_vectors: Option<usize>,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            distance_metric: DistanceMetric::Cosine,
            index_type: IndexType::Hnsw,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            max_vectors: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    Hnsw,
    #[cfg(feature = "faiss")]
    Faiss,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentConfig {
    pub chunking: ChunkingConfig,
    pub supported_formats: Vec<String>,
    pub max_file_size: usize,
    pub preprocessing: PreprocessingConfig,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            chunking: ChunkingConfig::default(),
            supported_formats: vec![
                "txt".to_string(),
                "md".to_string(),
                "html".to_string(),
                "json".to_string(),
                "pdf".to_string(),
            ],
            max_file_size: 50 * 1024 * 1024, // 50MB
            preprocessing: PreprocessingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub method: ChunkingMethod,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub separator_priority: Vec<String>,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            method: ChunkingMethod::TokenBased,
            chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 100,
            max_chunk_size: 1000,
            separator_priority: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "! ".to_string(),
                "? ".to_string(),
                "; ".to_string(),
                ", ".to_string(),
                " ".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingMethod {
    TokenBased,
    CharacterBased,
    Semantic,
    Sliding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub clean_whitespace: bool,
    pub normalize_unicode: bool,
    pub remove_html_tags: bool,
    pub extract_metadata: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            clean_whitespace: true,
            normalize_unicode: true,
            remove_html_tags: true,
            extract_metadata: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub ttl: HashMap<String, Duration>,
    pub max_entries: HashMap<String, usize>,
    pub consolidation_threshold: f64,
    pub cleanup_interval: Duration,
    pub working_memory_capacity: usize,
    pub enable_compression: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let mut ttl = HashMap::new();
        ttl.insert("short_term".to_string(), Duration::from_secs(3600));
        ttl.insert("long_term".to_string(), Duration::from_secs(86400 * 30));
        ttl.insert("episodic".to_string(), Duration::from_secs(86400 * 7));
        ttl.insert("working".to_string(), Duration::from_secs(900));

        let mut max_entries = HashMap::new();
        max_entries.insert("short_term".to_string(), 1000);
        max_entries.insert("long_term".to_string(), 10000);
        max_entries.insert("episodic".to_string(), 5000);
        max_entries.insert("semantic".to_string(), 50000);
        max_entries.insert("working".to_string(), 100);

        Self {
            ttl,
            max_entries,
            consolidation_threshold: 0.75,
            cleanup_interval: Duration::from_secs(300),
            working_memory_capacity: 100,
            enable_compression: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    pub max_concurrent_requests: Option<usize>,
    pub request_timeout_secs: Option<u64>,
    pub rate_limit_per_minute: Option<u32>,
    pub blocked_domains: Vec<String>,
    pub allowed_domains: Option<Vec<String>>,
    pub apis: HashMap<String, ApiConfig>,
    pub content_selectors: ContentSelectors,
    pub quality_weights: QualityWeights,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: Some(20),
            request_timeout_secs: Some(30),
            rate_limit_per_minute: Some(60),
            blocked_domains: vec![],
            allowed_domains: None,
            apis: HashMap::new(),
            content_selectors: ContentSelectors::default(),
            quality_weights: QualityWeights::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSelectors {
    pub title: Vec<String>,
    pub content: Vec<String>,
    pub date: Vec<String>,
    pub author: Vec<String>,
}

impl Default for ContentSelectors {
    fn default() -> Self {
        Self {
            title: vec!["h1".to_string(), "title".to_string(), ".title".to_string()],
            content: vec![
                "article".to_string(),
                "main".to_string(),
                ".content".to_string(),
            ],
            date: vec![
                "time".to_string(),
                ".date".to_string(),
                ".published".to_string(),
            ],
            author: vec![
                ".author".to_string(),
                ".by".to_string(),
                "[rel='author']".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityWeights {
    pub content_length: f64,
    pub readability: f64,
    pub structure: f64,
    pub freshness: f64,
}

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            content_length: 0.3,
            readability: 0.3,
            structure: 0.2,
            freshness: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    pub model: String,
    pub dimension: usize,
    pub batch_size: usize,
    pub cache_embeddings: bool,
    pub cache_ttl: Duration,
    #[cfg(feature = "onnx")]
    pub onnx_model_path: Option<PathBuf>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Local,
            model: "all-MiniLM-L6-v2".to_string(),
            dimension: 768,
            batch_size: 32,
            cache_embeddings: true,
            cache_ttl: Duration::from_secs(3600),
            #[cfg(feature = "onnx")]
            onnx_model_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    Local,
    OpenAI,
    Cohere,
    HuggingFace,
    #[cfg(feature = "onnx")]
    ONNX,
    #[cfg(feature = "gpu")]
    Candle,
    Custom(String),
}

#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enable: bool,
    pub prometheus_port: u16,
    pub export_interval: Duration,
    pub retention_period: Duration,
    pub labels: HashMap<String, String>,
}

#[cfg(feature = "metrics")]
impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable: true,
            prometheus_port: 9090,
            export_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(86400),
            labels: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout: Duration,
    pub enable_cors: bool,
    pub auth_enabled: bool,
    pub auth_token: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            request_timeout: Duration::from_secs(30),
            enable_cors: true,
            auth_enabled: false,
            auth_token: None,
        }
    }
}

impl Config {
    pub fn from_file(path: &PathBuf) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)
            .map_err(|e| crate::error::Error::Config(format!("Failed to parse config: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.vector_store.dimension == 0 {
            return Err(crate::error::Error::Config(
                "Vector dimension must be > 0".to_string(),
            ));
        }

        if self.document.chunking.chunk_size <= self.document.chunking.chunk_overlap {
            return Err(crate::error::Error::Config(
                "Chunk size must be greater than overlap".to_string(),
            ));
        }

        if self.server.port == 0 {
            return Err(crate::error::Error::Config(
                "Server port must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}
