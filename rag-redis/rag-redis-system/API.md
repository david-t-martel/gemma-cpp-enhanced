# RAG-Redis System API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Rust API](#rust-api)
3. [C API](#c-api)
4. [C++ API](#cpp-api)
5. [REST API](#rest-api)
6. [MCP Interface](#mcp-interface)
7. [Error Handling](#error-handling)
8. [Type Definitions](#type-definitions)
9. [Examples](#examples)

## Overview

The RAG-Redis System provides multiple API interfaces to accommodate different use cases and programming languages:

- **Rust API**: Native async/await interface with full type safety
- **C API**: Low-level FFI bindings for C integration
- **C++ API**: Modern C++ wrapper with RAII and exceptions
- **REST API**: HTTP/JSON interface for web services
- **MCP Interface**: Model Context Protocol for LLM integration

## Rust API

### Core Module: `rag_redis_system`

#### `RagSystem`

The main entry point for the RAG system.

```rust
pub struct RagSystem {
    // Private fields
}

impl RagSystem {
    /// Creates a new RAG system with the given configuration
    pub async fn new(config: Config) -> Result<Self>

    /// Creates a new RAG system from a configuration file
    pub async fn from_file(path: &Path) -> Result<Self>

    /// Ingests a document into the system
    pub async fn ingest_document(
        &self,
        content: &str,
        metadata: serde_json::Value
    ) -> Result<String>

    /// Searches for relevant documents
    pub async fn search(
        &self,
        query: &str,
        limit: usize
    ) -> Result<Vec<SearchResult>>

    /// Performs research with web search
    pub async fn research(
        &self,
        query: &str,
        sources: Vec<String>
    ) -> Result<Vec<SearchResult>>

    /// Updates system configuration
    pub async fn update_config(&self, config: Config) -> Result<()>

    /// Gets system statistics
    pub async fn get_stats(&self) -> Result<SystemStats>

    /// Performs health check
    pub async fn health_check(&self) -> Result<HealthStatus>

    /// Shuts down the system gracefully
    pub async fn shutdown(self) -> Result<()>
}
```

### Module: `config`

#### Configuration Types

```rust
/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub redis: RedisConfig,
    pub vector_store: VectorStoreConfig,
    pub document: DocumentConfig,
    pub memory: MemoryConfig,
    pub research: ResearchConfig,
    pub embedding: EmbeddingConfig,
    pub server: ServerConfig,
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: u32,
    pub connection_timeout: Duration,
    pub command_timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub enable_cluster: bool,
}

/// Vector store configuration
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

/// Document processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_chunks_per_doc: usize,
    pub supported_formats: Vec<String>,
    pub extract_metadata: bool,
}
```

### Module: `document`

#### Document Processing

```rust
/// Represents a document in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: serde_json::Value,
    pub chunks: Vec<DocumentChunk>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Represents a chunk of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub text: String,
    pub position: usize,
    pub metadata: serde_json::Value,
    pub embedding: Option<Vec<f32>>,
}

/// Document processing pipeline
pub struct DocumentPipeline {
    // Private fields
}

impl DocumentPipeline {
    /// Creates a new document pipeline
    pub fn new(config: DocumentConfig) -> Self

    /// Processes a document
    pub async fn process(
        &self,
        content: &str,
        metadata: serde_json::Value
    ) -> Result<Document>

    /// Chunks a document
    pub fn chunk_document(&self, doc: &Document) -> Result<Vec<DocumentChunk>>

    /// Extracts metadata from content
    pub fn extract_metadata(&self, content: &str) -> Result<serde_json::Value>

    /// Validates document format
    pub fn validate_format(&self, content: &[u8]) -> Result<DocumentFormat>
}
```

### Module: `vector_store`

#### Vector Operations

```rust
/// Vector store for similarity search
pub struct VectorStore {
    // Private fields
}

impl VectorStore {
    /// Creates a new vector store
    pub fn new(config: VectorStoreConfig) -> Result<Self>

    /// Adds a vector to the store
    pub fn add_vector(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: serde_json::Value
    ) -> Result<()>

    /// Searches for similar vectors
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<MetadataFilter>
    ) -> Result<Vec<(String, f32, serde_json::Value)>>

    /// Removes a vector from the store
    pub fn remove_vector(&mut self, id: &str) -> Result<()>

    /// Updates vector metadata
    pub fn update_metadata(
        &mut self,
        id: &str,
        metadata: serde_json::Value
    ) -> Result<()>

    /// Gets vector by ID
    pub fn get_vector(&self, id: &str) -> Result<Option<(Vec<f32>, serde_json::Value)>>

    /// Persists the index to disk
    pub async fn persist(&self, path: &Path) -> Result<()>

    /// Loads the index from disk
    pub async fn load(path: &Path) -> Result<Self>

    /// Gets store statistics
    pub fn stats(&self) -> VectorStoreStats
}

/// Metadata filter for vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    Contains,
    GreaterThan,
    LessThan,
    In,
    NotIn,
}
```

### Module: `embedding`

#### Embedding Generation

```rust
/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generates embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generates embeddings for multiple texts
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Gets the embedding dimension
    fn dimension(&self) -> usize;

    /// Gets maximum token limit
    fn max_tokens(&self) -> usize;
}

/// Embedding model types
pub enum EmbeddingModel {
    OpenAI(OpenAIProvider),
    Cohere(CohereProvider),
    Local(LocalProvider),
    Custom(Box<dyn EmbeddingProvider>),
}

/// OpenAI embedding provider
pub struct OpenAIProvider {
    api_key: String,
    model: String,
    dimension: usize,
}

/// Local embedding provider (ONNX/Candle)
pub struct LocalProvider {
    model_path: PathBuf,
    device: Device,
    dimension: usize,
}
```

### Module: `redis_backend`

#### Redis Operations

```rust
/// Redis client wrapper
pub struct RedisClient {
    // Private fields
}

impl RedisClient {
    /// Creates a new Redis client
    pub async fn new(config: &RedisConfig) -> Result<Self>

    /// Basic key-value operations
    pub async fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>>
    pub async fn set(&mut self, key: &str, value: &[u8]) -> Result<()>
    pub async fn set_ex(&mut self, key: &str, value: &[u8], ttl: Duration) -> Result<()>
    pub async fn delete(&mut self, key: &str) -> Result<()>
    pub async fn exists(&mut self, key: &str) -> Result<bool>
    pub async fn expire(&mut self, key: &str, ttl: Duration) -> Result<()>

    /// List operations
    pub async fn lpush(&mut self, key: &str, value: &[u8]) -> Result<()>
    pub async fn rpush(&mut self, key: &str, value: &[u8]) -> Result<()>
    pub async fn lrange(&mut self, key: &str, start: isize, stop: isize) -> Result<Vec<Vec<u8>>>
    pub async fn llen(&mut self, key: &str) -> Result<usize>

    /// Hash operations
    pub async fn hset(&mut self, key: &str, field: &str, value: &[u8]) -> Result<()>
    pub async fn hget(&mut self, key: &str, field: &str) -> Result<Option<Vec<u8>>>
    pub async fn hgetall(&mut self, key: &str) -> Result<HashMap<String, Vec<u8>>>
    pub async fn hdel(&mut self, key: &str, field: &str) -> Result<()>

    /// Set operations
    pub async fn sadd(&mut self, key: &str, member: &[u8]) -> Result<()>
    pub async fn srem(&mut self, key: &str, member: &[u8]) -> Result<()>
    pub async fn smembers(&mut self, key: &str) -> Result<Vec<Vec<u8>>>
    pub async fn sismember(&mut self, key: &str, member: &[u8]) -> Result<bool>
}

/// Redis manager for high-level operations
pub struct RedisManager {
    // Private fields
}

impl RedisManager {
    /// Creates a new Redis manager
    pub async fn new(config: &RedisConfig) -> Result<Self>

    /// Stores a document
    pub async fn store_document(&self, doc: &Document) -> Result<()>

    /// Retrieves a document
    pub async fn get_document(&self, id: &str) -> Result<Option<Document>>

    /// Stores a document chunk
    pub async fn store_chunk(&self, chunk: &DocumentChunk) -> Result<()>

    /// Retrieves a document chunk
    pub async fn get_chunk(&self, id: &str) -> Result<Option<DocumentChunk>>

    /// Lists all documents
    pub async fn list_documents(&self) -> Result<Vec<String>>

    /// Deletes a document and its chunks
    pub async fn delete_document(&self, id: &str) -> Result<()>
}
```

## C API

### System Management

```c
// System lifecycle
RagSystem* rag_system_new(const char* config_json);
RagSystem* rag_system_new_from_file(const char* config_path);
void rag_system_free(RagSystem* system);

// Document operations
const char* rag_system_ingest(
    RagSystem* system,
    const char* content,
    const char* metadata_json
);

// Search operations
SearchResults* rag_system_search(
    RagSystem* system,
    const char* query,
    size_t limit
);

SearchResults* rag_system_research(
    RagSystem* system,
    const char* query,
    const char** sources,
    size_t source_count
);

// Results management
void rag_search_results_free(SearchResults* results);

// Error handling
RagError rag_last_error(void);
const char* rag_error_message(RagError error);
void rag_clear_error(void);

// Configuration
int rag_system_update_config(
    RagSystem* system,
    const char* config_json
);

// Statistics
SystemStats* rag_system_get_stats(RagSystem* system);
void rag_stats_free(SystemStats* stats);
```

### Data Structures

```c
// Search result structure
typedef struct {
    char* id;
    char* text;
    float score;
    char* metadata_json;
} SearchResult;

// Search results container
typedef struct {
    SearchResult* items;
    size_t count;
} SearchResults;

// System statistics
typedef struct {
    size_t total_documents;
    size_t total_chunks;
    size_t total_vectors;
    double avg_search_time_ms;
    size_t cache_hits;
    size_t cache_misses;
} SystemStats;

// Error codes
typedef enum {
    RAG_ERROR_NONE = 0,
    RAG_ERROR_INVALID_CONFIG = 1,
    RAG_ERROR_REDIS_CONNECTION = 2,
    RAG_ERROR_VECTOR_DIMENSION = 3,
    RAG_ERROR_DOCUMENT_PARSE = 4,
    RAG_ERROR_EMBEDDING_FAILED = 5,
    RAG_ERROR_SEARCH_FAILED = 6,
    RAG_ERROR_OUT_OF_MEMORY = 7,
    RAG_ERROR_INVALID_ARGUMENT = 8,
    RAG_ERROR_INTERNAL = 9,
} RagError;
```

## C++ API

### System Class

```cpp
namespace rag {

class System {
public:
    // Constructors
    explicit System(const std::string& config_json);
    explicit System(const std::filesystem::path& config_path);

    // Move semantics
    System(System&& other) noexcept;
    System& operator=(System&& other) noexcept;

    // Delete copy semantics
    System(const System&) = delete;
    System& operator=(const System&) = delete;

    // Destructor
    ~System();

    // Document operations
    std::string ingestDocument(
        const std::string& content,
        const nlohmann::json& metadata = {}
    );

    // Search operations
    std::vector<SearchResult> search(
        const std::string& query,
        size_t limit = 10
    );

    std::vector<SearchResult> research(
        const std::string& query,
        const std::vector<std::string>& sources = {}
    );

    // Configuration
    void updateConfig(const nlohmann::json& config);
    nlohmann::json getConfig() const;

    // Statistics
    SystemStats getStats() const;

    // Health check
    HealthStatus healthCheck() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

// Search result structure
struct SearchResult {
    std::string id;
    std::string text;
    float score;
    nlohmann::json metadata;

    // Comparison operators
    bool operator<(const SearchResult& other) const {
        return score > other.score; // Higher scores first
    }
};

// System statistics
struct SystemStats {
    size_t totalDocuments;
    size_t totalChunks;
    size_t totalVectors;
    std::chrono::milliseconds avgSearchTime;
    size_t cacheHits;
    size_t cacheMisses;

    // Utility methods
    double cacheHitRate() const {
        auto total = cacheHits + cacheMisses;
        return total > 0 ? static_cast<double>(cacheHits) / total : 0.0;
    }
};

// Error handling
class Error : public std::runtime_error {
public:
    explicit Error(ErrorCode code, const std::string& message);
    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

enum class ErrorCode {
    None = 0,
    InvalidConfig = 1,
    RedisConnection = 2,
    VectorDimension = 3,
    DocumentParse = 4,
    EmbeddingFailed = 5,
    SearchFailed = 6,
    OutOfMemory = 7,
    InvalidArgument = 8,
    Internal = 9,
};

} // namespace rag
```

### RAII Wrappers

```cpp
namespace rag {

// Automatic resource management for C API
template<typename T, void(*Deleter)(T*)>
class UniqueHandle {
public:
    explicit UniqueHandle(T* handle = nullptr) : handle_(handle) {}
    ~UniqueHandle() { if (handle_) Deleter(handle_); }

    // Move semantics
    UniqueHandle(UniqueHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) Deleter(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    // Delete copy semantics
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    // Access
    T* get() const noexcept { return handle_; }
    T* release() noexcept { return std::exchange(handle_, nullptr); }
    void reset(T* handle = nullptr) {
        if (handle_) Deleter(handle_);
        handle_ = handle;
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    T* handle_;
};

// Type aliases for common handles
using SystemHandle = UniqueHandle<RagSystem, rag_system_free>;
using ResultsHandle = UniqueHandle<SearchResults, rag_search_results_free>;
using StatsHandle = UniqueHandle<SystemStats, rag_stats_free>;

} // namespace rag
```

## REST API

### Endpoints

#### System Management

**GET /health**
```http
Response: 200 OK
{
    "status": "healthy",
    "version": "0.1.0",
    "uptime_seconds": 3600,
    "redis_connected": true,
    "vector_index_ready": true
}
```

**GET /stats**
```http
Response: 200 OK
{
    "total_documents": 1000,
    "total_chunks": 5000,
    "total_vectors": 5000,
    "avg_search_time_ms": 25.5,
    "cache_hits": 450,
    "cache_misses": 50
}
```

#### Document Operations

**POST /documents**
```http
Content-Type: application/json

{
    "content": "Document content here...",
    "metadata": {
        "source": "manual",
        "author": "John Doe",
        "tags": ["technology", "ai"]
    }
}

Response: 201 Created
{
    "id": "doc_abc123",
    "chunks": 10,
    "status": "indexed"
}
```

**GET /documents/{id}**
```http
Response: 200 OK
{
    "id": "doc_abc123",
    "content": "Original content...",
    "metadata": {...},
    "chunks": [...],
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
}
```

**DELETE /documents/{id}**
```http
Response: 204 No Content
```

#### Search Operations

**POST /search**
```http
Content-Type: application/json

{
    "query": "What is RAG?",
    "limit": 10,
    "filter": {
        "field": "source",
        "operator": "equals",
        "value": "documentation"
    }
}

Response: 200 OK
{
    "results": [
        {
            "id": "chunk_xyz789",
            "text": "RAG stands for Retrieval-Augmented Generation...",
            "score": 0.95,
            "metadata": {...},
            "document_id": "doc_abc123"
        }
    ],
    "total": 10,
    "search_time_ms": 25
}
```

**POST /research**
```http
Content-Type: application/json

{
    "query": "Latest developments in RAG systems",
    "sources": ["arxiv", "github", "papers"],
    "limit": 20
}

Response: 200 OK
{
    "results": [...],
    "sources_searched": 3,
    "total_results": 20
}
```

## MCP Interface

### Tool Definitions

```json
{
    "tools": [
        {
            "name": "rag_ingest",
            "description": "Ingest a document into the RAG system",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Document content"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Document metadata"
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "rag_search",
            "description": "Search for relevant documents",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "rag_research",
            "description": "Research with web sources",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research query"
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Web sources to search"
                    }
                },
                "required": ["query"]
            }
        }
    ]
}
```

### Resource Definitions

```json
{
    "resources": [
        {
            "uri": "rag://documents",
            "name": "Documents",
            "description": "All indexed documents",
            "mimeType": "application/json"
        },
        {
            "uri": "rag://documents/{id}",
            "name": "Document",
            "description": "Specific document by ID",
            "mimeType": "application/json"
        },
        {
            "uri": "rag://stats",
            "name": "System Statistics",
            "description": "Current system statistics",
            "mimeType": "application/json"
        }
    ]
}
```

## Error Handling

### Rust Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Redis error: {0}")]
    Redis(String),

    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Document parsing error: {0}")]
    DocumentParse(String),

    #[error("Embedding generation failed: {0}")]
    EmbeddingFailed(String),

    #[error("Search failed: {0}")]
    SearchFailed(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

### Error Response Format

```json
{
    "error": {
        "code": "VECTOR_DIMENSION_MISMATCH",
        "message": "Vector dimension mismatch: expected 768, got 1024",
        "details": {
            "expected": 768,
            "actual": 1024
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }
}
```

## Type Definitions

### Common Types

```rust
/// Search result with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    pub document_id: Option<String>,
    pub chunk_position: Option<usize>,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub redis_connected: bool,
    pub vector_index_ready: bool,
    pub embedding_provider_ready: bool,
    pub errors: Vec<String>,
}

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// Index types for vector storage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    Hnsw,
    #[cfg(feature = "faiss")]
    Faiss,
}

/// Document formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Html,
    Pdf,
    Json,
}

/// Memory types for caching
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryType {
    ShortTerm,  // L1 cache
    LongTerm,   // L2 cache
    Persistent, // L3 storage
}
```

## Examples

### Rust Example: Document Processing Pipeline

```rust
use rag_redis_system::{RagSystem, Config, SearchResult};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize system
    let mut config = Config::default();
    config.redis.url = "redis://localhost:6379".to_string();
    config.vector_store.dimension = 768;

    let rag = RagSystem::new(config).await?;

    // Ingest multiple documents
    let documents = vec![
        ("Introduction to RAG", json!({"category": "tutorial"})),
        ("Advanced RAG Techniques", json!({"category": "advanced"})),
        ("RAG Performance Optimization", json!({"category": "performance"})),
    ];

    for (content, metadata) in documents {
        let doc_id = rag.ingest_document(content, metadata).await?;
        println!("Ingested document: {}", doc_id);
    }

    // Perform semantic search
    let results = rag.search("How to optimize RAG performance?", 5).await?;

    for result in results {
        println!("Score: {:.3}, Text: {}...",
                 result.score,
                 &result.text[..100.min(result.text.len())]);
    }

    // Research with web sources
    let research_results = rag.research(
        "Latest RAG architectures 2024",
        vec!["arxiv.org".to_string(), "github.com".to_string()]
    ).await?;

    println!("Found {} research results", research_results.len());

    Ok(())
}
```

### C++ Example: Error Handling and RAII

```cpp
#include <rag_redis.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Initialize with configuration file
        rag::System system("config.json");

        // Ingest documents with metadata
        std::vector<std::pair<std::string, nlohmann::json>> documents = {
            {"Document 1 content", {{"source", "manual"}}},
            {"Document 2 content", {{"source", "api"}}},
        };

        for (const auto& [content, metadata] : documents) {
            auto id = system.ingestDocument(content, metadata);
            std::cout << "Ingested: " << id << std::endl;
        }

        // Search with error handling
        try {
            auto results = system.search("query text", 10);

            for (const auto& result : results) {
                std::cout << "Score: " << result.score
                          << ", ID: " << result.id << std::endl;
            }
        } catch (const rag::Error& e) {
            if (e.code() == rag::ErrorCode::SearchFailed) {
                std::cerr << "Search failed: " << e.what() << std::endl;
                // Handle search failure
            } else {
                throw; // Re-throw other errors
            }
        }

        // Get and display statistics
        auto stats = system.getStats();
        std::cout << "Total documents: " << stats.totalDocuments << std::endl;
        std::cout << "Cache hit rate: " << stats.cacheHitRate() << std::endl;

    } catch (const rag::Error& e) {
        std::cerr << "RAG Error [" << static_cast<int>(e.code())
                  << "]: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
```

### C Example: Basic Usage

```c
#include <rag_redis.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize system
    RagSystem* system = rag_system_new_from_file("config.json");
    if (!system) {
        RagError error = rag_last_error();
        fprintf(stderr, "Failed to initialize: %s\n",
                rag_error_message(error));
        return 1;
    }

    // Ingest a document
    const char* doc_id = rag_system_ingest(
        system,
        "This is a sample document about RAG systems.",
        "{\"source\": \"example\"}"
    );

    if (!doc_id) {
        fprintf(stderr, "Ingestion failed: %s\n",
                rag_error_message(rag_last_error()));
        rag_system_free(system);
        return 1;
    }

    printf("Document ingested with ID: %s\n", doc_id);

    // Perform search
    SearchResults* results = rag_system_search(system, "RAG systems", 5);
    if (!results) {
        fprintf(stderr, "Search failed: %s\n",
                rag_error_message(rag_last_error()));
        rag_system_free(system);
        return 1;
    }

    // Process results
    for (size_t i = 0; i < results->count; i++) {
        printf("Result %zu:\n", i + 1);
        printf("  Score: %.3f\n", results->items[i].score);
        printf("  Text: %.100s...\n", results->items[i].text);
    }

    // Clean up
    rag_search_results_free(results);
    rag_system_free(system);

    return 0;
}
```

## Best Practices

1. **Error Handling**: Always check return values and handle errors appropriately
2. **Resource Management**: Use RAII in C++ or explicit cleanup in C
3. **Configuration**: Validate configuration before initializing the system
4. **Batching**: Use batch operations for multiple documents/queries
5. **Caching**: Leverage the built-in caching for frequently accessed data
6. **Monitoring**: Regularly check system statistics and health status
7. **Threading**: The system is thread-safe; use from multiple threads as needed
8. **Memory**: Monitor memory usage, especially with large vector indices

## Performance Tips

1. **Batch Processing**: Process multiple documents/queries together
2. **Connection Pooling**: Configure appropriate pool sizes for Redis
3. **Index Tuning**: Adjust HNSW parameters based on accuracy/speed requirements
4. **Caching**: Enable and configure multi-level caching
5. **Async Operations**: Use async APIs where available
6. **SIMD**: Ensure SIMD support is enabled for vector operations
7. **Compression**: Enable compression for network transfers and storage
