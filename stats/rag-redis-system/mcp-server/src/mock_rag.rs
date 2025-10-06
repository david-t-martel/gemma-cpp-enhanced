use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

/// Mock RAG system configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub redis_url: String,
    pub embedding_model: String,
    pub chunk_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            chunk_size: 512,
        }
    }
}

/// Mock RAG system errors
#[derive(Debug, Error)]
pub enum Error {
    #[error("Redis error: {0}")]
    Redis(String),
    #[error("Vector store error: {0}")]
    VectorStore(String),
    #[error("Document processing error: {0}")]
    Document(String),
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Research error: {0}")]
    Research(String),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Mock search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: Value,
}

/// Mock RAG system implementation
pub struct RagSystem {
    config: Config,
    documents: std::sync::Arc<tokio::sync::RwLock<HashMap<String, Document>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl RagSystem {
    pub async fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            documents: std::sync::Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    pub async fn ingest_document(&self, content: &str, metadata: Value) -> Result<String> {
        let doc_id = uuid::Uuid::new_v4().to_string();
        let document = Document {
            id: doc_id.clone(),
            content: content.to_string(),
            metadata,
            created_at: chrono::Utc::now(),
        };

        self.documents.write().await.insert(doc_id.clone(), document);
        tracing::info!("Ingested document with ID: {}", doc_id);

        Ok(doc_id)
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let documents = self.documents.read().await;

        // Simple mock search - find documents containing query terms
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for (id, doc) in documents.iter() {
            if doc.content.to_lowercase().contains(&query_lower) {
                results.push(SearchResult {
                    id: id.clone(),
                    text: doc.content.clone(),
                    score: 0.8, // Mock similarity score
                    metadata: doc.metadata.clone(),
                });
            }
        }

        results.truncate(limit);
        tracing::debug!("Found {} search results for query: {}", results.len(), query);

        Ok(results)
    }

    pub async fn research(&self, query: &str, sources: Vec<String>) -> Result<Vec<SearchResult>> {
        // Combine local search with mock web search
        let mut local_results = self.search(query, 5).await?;

        // Mock web search results
        if !sources.is_empty() {
            for (i, source) in sources.iter().enumerate() {
                if i >= 3 { break; } // Limit web results

                local_results.push(SearchResult {
                    id: format!("web_{}", i),
                    text: format!("Mock web result from {} about: {}", source, query),
                    score: 0.7 - (i as f32 * 0.1),
                    metadata: serde_json::json!({
                        "source": source,
                        "type": "web_result"
                    }),
                });
            }
        }

        local_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        local_results.truncate(10);

        tracing::debug!("Research found {} combined results", local_results.len());

        Ok(local_results)
    }

    pub async fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
        let documents = self.documents.read().await;
        Ok(documents.get(doc_id).cloned())
    }

    pub async fn list_documents(&self, limit: usize, offset: usize) -> Result<Vec<Document>> {
        let documents = self.documents.read().await;
        let docs: Vec<Document> = documents.values().cloned().collect();

        let start = std::cmp::min(offset, docs.len());
        let end = std::cmp::min(start + limit, docs.len());

        Ok(docs[start..end].to_vec())
    }

    pub async fn delete_document(&self, doc_id: &str) -> Result<bool> {
        let mut documents = self.documents.write().await;
        Ok(documents.remove(doc_id).is_some())
    }

    pub async fn clear_all_documents(&self) -> Result<()> {
        self.documents.write().await.clear();
        tracing::info!("Cleared all documents from system");
        Ok(())
    }

    pub async fn get_stats(&self) -> Result<SystemStats> {
        let documents = self.documents.read().await;
        Ok(SystemStats {
            total_documents: documents.len(),
            total_chunks: documents.len() * 3, // Mock chunk count
            memory_usage_mb: documents.len() as f64 * 0.1, // Mock memory usage
            avg_search_latency_ms: 15.0,
            queries_per_second: 10.0,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub memory_usage_mb: f64,
    pub avg_search_latency_ms: f64,
    pub queries_per_second: f64,
}
