//! # RAG-Redis System Demo
//!
//! This demo showcases the key features of the RAG-Redis system:
//! - Document ingestion and processing
//! - Semantic search capabilities
//! - Vector embeddings
//! - Memory management
//! - Error handling
//!
//! ## Running the Demo
//!
//! ```bash
//! # Ensure Redis server is running (optional - demo uses mock backends)
//! redis-server
//!
//! # Run the demo
//! cargo run --example demo
//! ```

use serde_json::json;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Simplified document structure for demo
#[derive(Debug, Clone)]
struct DemoDocument {
    id: String,
    title: String,
    content: String,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Document chunk for processing
#[derive(Debug, Clone)]
struct DocumentChunk {
    id: String,
    document_id: String,
    text: String,
    embedding: Vec<f32>,
    metadata: serde_json::Value,
}

/// Search result structure
#[derive(Debug, Clone)]
struct SearchResult {
    id: String,
    text: String,
    score: f32,
    metadata: serde_json::Value,
}

/// Demo RAG system implementation
struct DemoRagSystem {
    documents: HashMap<String, DemoDocument>,
    chunks: HashMap<String, DocumentChunk>,
    config: DemoConfig,
}

#[derive(Debug, Clone)]
struct DemoConfig {
    chunk_size: usize,
    chunk_overlap: usize,
    embedding_dimension: usize,
    max_search_results: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            embedding_dimension: 768,
            max_search_results: 10,
        }
    }
}

impl DemoRagSystem {
    fn new() -> Self {
        Self {
            documents: HashMap::new(),
            chunks: HashMap::new(),
            config: DemoConfig::default(),
        }
    }

    async fn ingest_document(
        &mut self,
        title: &str,
        content: &str,
        metadata: serde_json::Value,
    ) -> Result<String, String> {
        let doc_id = format!("doc-{}", uuid::Uuid::new_v4());

        let document = DemoDocument {
            id: doc_id.clone(),
            title: title.to_string(),
            content: content.to_string(),
            metadata,
            created_at: chrono::Utc::now(),
        };

        // Create chunks from document
        let chunks = self.create_chunks(&document).await?;

        // Store document and chunks
        self.documents.insert(doc_id.clone(), document);
        for chunk in chunks {
            self.chunks.insert(chunk.id.clone(), chunk);
        }

        Ok(doc_id)
    }

    async fn create_chunks(&self, document: &DemoDocument) -> Result<Vec<DocumentChunk>, String> {
        let mut chunks = Vec::new();
        let content = &document.content;
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.chunk_overlap;

        let mut start = 0;
        let mut chunk_index = 0;

        while start < content.len() {
            let end = std::cmp::min(start + chunk_size, content.len());
            let chunk_text = content[start..end].to_string();

            // Generate mock embedding
            let embedding = self.generate_embedding(&chunk_text).await?;

            let chunk = DocumentChunk {
                id: format!("{}-chunk-{}", document.id, chunk_index),
                document_id: document.id.clone(),
                text: chunk_text,
                embedding,
                metadata: json!({
                    "document_title": document.title,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end
                }),
            };

            chunks.push(chunk);

            if end == content.len() {
                break;
            }

            start = if end - start <= overlap {
                end
            } else {
                start + chunk_size - overlap
            };

            chunk_index += 1;
        }

        Ok(chunks)
    }

    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, String> {
        // Simulate embedding generation with deterministic results
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.config.embedding_dimension);
        let mut rng = hash;

        for _ in 0..self.config.embedding_dimension {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = ((rng >> 16) & 0xFFFF) as f32 / 65535.0;
            embedding.push(value * 2.0 - 1.0); // Normalize to [-1, 1]
        }

        // Normalize to unit vector for cosine similarity
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut embedding {
                *value /= norm;
            }
        }

        // Small delay to simulate processing time
        sleep(Duration::from_millis(10)).await;

        Ok(embedding)
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
        let query_embedding = self.generate_embedding(query).await?;
        let mut results = Vec::new();

        for chunk in self.chunks.values() {
            let similarity = cosine_similarity(&query_embedding, &chunk.embedding);

            results.push(SearchResult {
                id: chunk.id.clone(),
                text: chunk.text.clone(),
                score: similarity,
                metadata: chunk.metadata.clone(),
            });
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), json!(self.documents.len()));
        stats.insert("total_chunks".to_string(), json!(self.chunks.len()));
        stats.insert("chunk_size".to_string(), json!(self.config.chunk_size));
        stats.insert(
            "embedding_dimension".to_string(),
            json!(self.config.embedding_dimension),
        );

        if !self.chunks.is_empty() {
            let avg_chunk_size: f64 = self.chunks.values().map(|c| c.text.len()).sum::<usize>()
                as f64
                / self.chunks.len() as f64;
            stats.insert(
                "avg_chunk_size".to_string(),
                json!(avg_chunk_size.round() as usize),
            );
        }

        stats
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Sample documents for the demonstration
const SAMPLE_DOCUMENTS: &[(&str, &str)] = &[
    (
        "Rust Programming Language",
        r#"Rust is a systems programming language that runs blazingly fast, prevents segfaults,
        and guarantees thread safety. It accomplishes these goals by being memory safe without
        using garbage collection. Rust has great documentation, a friendly compiler with useful
        error messages, and top-notch tooling — an integrated package manager and build tool,
        smart multi-editor support with auto-completion and type inspections, an auto-formatter,
        and more. Rust is used for system programming, web backends, command-line tools, network
        services, embedded systems, cryptocurrency, virtualization and containerization, database
        engines, operating systems, browser components, machine learning, and more."#,
    ),
    (
        "Vector Databases",
        r#"Vector databases are specialized database systems designed to store, index, and query
        high-dimensional vector data efficiently. They are particularly useful for applications
        involving machine learning, artificial intelligence, and similarity search. Vector databases
        use various indexing algorithms like HNSW (Hierarchical Navigable Small World), IVF
        (Inverted File), or LSH (Locality-Sensitive Hashing) to enable fast approximate nearest
        neighbor search. Popular vector databases include Pinecone, Weaviate, Milvus, and Qdrant.
        They are commonly used for semantic search, recommendation systems, image similarity,
        natural language processing, and retrieval-augmented generation (RAG) applications."#,
    ),
    (
        "Redis Architecture",
        r#"Redis is an open-source, in-memory data structure store used as a database, cache,
        and message broker. It supports various data structures such as strings, hashes, lists,
        sets, sorted sets with range queries, bitmaps, hyperloglogs, geospatial indexes, and streams.
        Redis has built-in replication, Lua scripting, LRU eviction, transactions, and different
        levels of on-disk persistence. It provides high availability via Redis Sentinel and automatic
        partitioning with Redis Cluster. Redis is atomic, meaning all operations are either
        completed successfully or not executed at all. It supports publish/subscribe messaging,
        blocking operations, and has excellent performance characteristics with sub-millisecond
        response times for most operations."#,
    ),
    (
        "Machine Learning Embeddings",
        r#"Embeddings are dense vector representations of data that capture semantic meaning and
        relationships. In machine learning, embeddings transform high-dimensional, sparse data
        (like words, sentences, or images) into lower-dimensional, dense vectors while preserving
        important characteristics. Word embeddings like Word2Vec, GloVe, and FastText represent
        words as vectors where semantically similar words are closer in vector space. Sentence
        and document embeddings from models like BERT, RoBERTa, and Sentence-BERT capture
        contextual meaning. These embeddings enable various applications including semantic search,
        clustering, classification, recommendation systems, and similarity matching. The quality
        of embeddings is crucial for downstream tasks, and they can be fine-tuned for specific
        domains or applications."#,
    ),
    (
        "Retrieval Augmented Generation",
        r#"Retrieval-Augmented Generation (RAG) is a natural language processing framework that
        combines the strengths of retrieval-based and generation-based approaches. RAG systems
        first retrieve relevant documents or passages from a knowledge base using semantic search,
        then use this retrieved context to generate accurate and informative responses. This approach
        addresses limitations of pure generative models, such as hallucination and lack of up-to-date
        information. RAG systems typically consist of three main components: a retriever that finds
        relevant documents, an encoder that processes the retrieved content, and a generator that
        produces the final response. Popular implementations include Facebook's RAG model, OpenAI's
        retrieval-augmented GPT models, and various open-source frameworks. RAG is widely used
        for question answering, chatbots, knowledge management, and information synthesis tasks."#,
    ),
];

const DEMO_QUERIES: &[&str] = &[
    "What is Rust programming language?",
    "How do vector databases work?",
    "Explain Redis data structures",
    "What are embeddings in machine learning?",
    "How does retrieval augmented generation work?",
    "Performance characteristics of systems",
    "Memory safety in programming",
    "Semantic search applications",
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better logging
    tracing_subscriber::fmt()
        .with_env_filter("demo=info")
        .with_target(false)
        .init();

    info!("🚀 Starting RAG-Redis System Demo");
    info!("==================================");

    // Initialize the demo system
    let mut system = DemoRagSystem::new();

    // Demo 1: System initialization
    info!("📋 Demo Overview:");
    info!("1. System Initialization & Configuration");
    info!("2. Document Ingestion & Processing");
    info!("3. Embedding Generation & Vector Storage");
    info!("4. Semantic Search & Retrieval");
    info!("5. Memory Management Demonstration");
    info!("6. Error Handling & Recovery");
    info!("7. Performance Metrics");
    println!();

    // Demo 2: Document ingestion
    info!("📄 Step 1: Document Ingestion & Processing");
    let mut document_ids = Vec::new();

    for (i, (title, content)) in SAMPLE_DOCUMENTS.iter().enumerate() {
        let metadata = json!({
            "title": title,
            "source": "demo",
            "type": "educational",
            "index": i,
            "length": content.len(),
            "ingested_at": chrono::Utc::now().to_rfc3339()
        });

        match system.ingest_document(title, content, metadata).await {
            Ok(doc_id) => {
                document_ids.push(doc_id.clone());
                info!("  ✅ Ingested: '{}' (ID: {})", title, doc_id);

                // Show chunking info
                if let Some(doc) = system.documents.get(&doc_id) {
                    let chunk_count = system
                        .chunks
                        .values()
                        .filter(|c| c.document_id == doc_id)
                        .count();
                    debug!("     Created {} chunks", chunk_count);
                }
            }
            Err(e) => {
                warn!("  ⚠️ Failed to ingest '{}': {}", title, e);
            }
        }

        sleep(Duration::from_millis(100)).await;
    }

    info!(
        "  📊 Ingestion completed: {} documents processed",
        document_ids.len()
    );
    println!();

    // Demo 3: Show system statistics
    info!("🔧 Step 2: System Configuration & Statistics");
    let stats = system.get_stats();
    for (key, value) in stats.iter() {
        info!("  {}: {}", key, value);
    }
    println!();

    // Demo 4: Search demonstrations
    info!("🔍 Step 3: Semantic Search & Retrieval");

    for (i, query) in DEMO_QUERIES.iter().enumerate() {
        info!("  🔎 Query {}: '{}'", i + 1, query);

        let search_start = Instant::now();
        match system.search(query, 3).await {
            Ok(results) => {
                let search_time = search_start.elapsed();
                info!("    ⏱️ Search completed in {:?}", search_time);
                info!("    📊 Found {} results:", results.len());

                for (j, result) in results.iter().enumerate() {
                    let preview = if result.text.len() > 150 {
                        format!("{}...", &result.text[..150])
                    } else {
                        result.text.clone()
                    };

                    info!("      {}. Score: {:.3} - {}", j + 1, result.score, preview);

                    // Show source document
                    if let Ok(meta) = serde_json::from_value::<
                        serde_json::Map<String, serde_json::Value>,
                    >(result.metadata.clone())
                    {
                        if let Some(title) = meta.get("document_title") {
                            debug!("         Source: {}", title);
                        }
                    }
                }
            }
            Err(e) => {
                error!("    ❌ Search failed: {}", e);
            }
        }

        println!();
        sleep(Duration::from_millis(300)).await;
    }

    // Demo 5: Memory management
    info!("🧠 Step 4: Memory Management Demonstration");
    info!("  📝 Memory Types Available:");
    info!(
        "    - Document Storage: {} documents in memory",
        system.documents.len()
    );
    info!(
        "    - Chunk Storage: {} chunks with embeddings",
        system.chunks.len()
    );
    info!("    - Vector Index: In-memory similarity search");
    info!("    - Embedding Cache: Deterministic mock embeddings");

    // Calculate memory usage
    let doc_memory: usize = system
        .documents
        .values()
        .map(|d| d.content.len() + d.title.len())
        .sum();
    let chunk_memory: usize = system
        .chunks
        .values()
        .map(|c| c.text.len() + c.embedding.len() * 4)
        .sum();

    info!("  📊 Estimated Memory Usage:");
    info!("    - Documents: ~{:.1} KB", doc_memory as f64 / 1024.0);
    info!(
        "    - Chunks + Embeddings: ~{:.1} KB",
        chunk_memory as f64 / 1024.0
    );
    info!(
        "    - Total: ~{:.1} KB",
        (doc_memory + chunk_memory) as f64 / 1024.0
    );
    println!();

    // Demo 6: Error handling
    info!("⚠️ Step 5: Error Handling & Recovery");

    let error_test_queries = [
        ("Empty query", ""),
        ("Very long query", &"word ".repeat(1000)),
        ("Special characters", "###@@@%%%^^^&&&***"),
        ("Unicode query", "测试中文查询 🚀"),
    ];

    for (test_name, query) in &error_test_queries {
        info!("  🧪 Testing: {}", test_name);

        match system.search(query, 3).await {
            Ok(results) => {
                info!("    ✅ Handled gracefully: {} results", results.len());
            }
            Err(e) => {
                info!("    ⚠️ Error handled: {}", e);
            }
        }
    }
    println!();

    // Demo 7: Performance metrics
    info!("📊 Step 6: Performance Metrics & Analysis");

    // Test search performance with different result limits
    let perf_query = "Rust programming language performance";
    let limits = [1, 5, 10];

    for limit in limits {
        let start = Instant::now();
        match system.search(perf_query, limit).await {
            Ok(results) => {
                let duration = start.elapsed();
                info!(
                    "  📈 Search (limit {}): {:?} for {} results",
                    limit,
                    duration,
                    results.len()
                );
            }
            Err(e) => {
                warn!("  ❌ Performance test failed: {}", e);
            }
        }
    }

    // Final statistics
    let final_stats = system.get_stats();
    info!("  🏆 Final System Statistics:");
    for (key, value) in final_stats {
        info!("    {}: {}", key, value);
    }

    println!();
    info!("✅ RAG-Redis System Demo completed successfully!");
    info!("🎉 All features demonstrated:");
    info!("   ✅ Document ingestion and chunking");
    info!("   ✅ Embedding generation (mock deterministic)");
    info!("   ✅ Vector similarity search");
    info!("   ✅ Memory management");
    info!("   ✅ Error handling and recovery");
    info!("   ✅ Performance monitoring");
    info!("   ✅ System statistics and metrics");

    Ok(())
}
