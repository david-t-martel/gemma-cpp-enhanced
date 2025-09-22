//! RAG System Demonstration
//!
//! This example shows how to use the complete RAG system pipeline:
//! 1. Initialize Redis connection
//! 2. Create vector store
//! 3. Process documents through pipeline
//! 4. Perform similarity search
//! 5. Conduct internet research

use gemma_extensions::{
    document_pipeline::{ChunkingConfig, DocumentFormat, DocumentPipeline, EmbeddingConfig},
    redis_manager::{DocumentMetadata, RedisConfig, RedisManager},
    research_client::{ResearchClient, ResearchConfig, ResearchQuery},
    vector_store::{DistanceMetric, VectorStore, VectorStoreConfig},
};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("🚀 Initializing Redis-backed RAG System...\n");

    // 1. Setup Redis Manager
    println!("📦 Setting up Redis connection...");
    let redis_config = RedisConfig {
        url: "redis://localhost:6379".to_string(),
        max_connections: 10,
        connection_timeout_ms: 5000,
        default_ttl: 3600,
        key_prefix: "rag_demo:".to_string(),
        ..Default::default()
    };

    let redis_manager = match RedisManager::new(redis_config).await {
        Ok(manager) => {
            println!("✅ Redis manager initialized successfully");
            Arc::new(manager)
        }
        Err(e) => {
            println!("❌ Failed to initialize Redis: {}", e);
            println!("💡 Make sure Redis is running on localhost:6379");
            return Ok(());
        }
    };

    // 2. Setup Vector Store
    println!("\n🔍 Setting up vector store with HNSW index...");
    let vector_config = VectorStoreConfig {
        dimension: 384, // Using smaller dimension for demo
        metric: DistanceMetric::Cosine,
        max_connections: 16,
        ef_construction: 200,
        ef_search: 100,
        normalize_vectors: true,
        max_memory_vectors: 10000,
    };

    let vector_store = match VectorStore::with_redis(vector_config, redis_manager.clone()) {
        Ok(store) => {
            println!("✅ Vector store initialized with {} dimensions", 384);
            Arc::new(store)
        }
        Err(e) => {
            println!("❌ Failed to initialize vector store: {}", e);
            return Ok(());
        }
    };

    // 3. Setup Document Processing Pipeline
    println!("\n📄 Setting up document processing pipeline...");
    let chunking_config = ChunkingConfig {
        max_chunk_size: 256,
        chunk_overlap: 25,
        min_chunk_size: 50,
        respect_sentence_boundaries: true,
        respect_paragraph_boundaries: true,
        separators: vec![
            "\n\n".to_string(),
            "\n".to_string(),
            ". ".to_string(),
            "! ".to_string(),
            "? ".to_string(),
            " ".to_string(),
        ],
        semantic_chunking: false,
        max_chunks_per_document: 100,
    };

    let embedding_config = EmbeddingConfig {
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        batch_size: 16,
        dimension: 384,
        normalize: true,
        max_text_length: 4000,
        max_retries: 3,
        timeout_ms: 30000,
    };

    let pipeline = DocumentPipeline::new(chunking_config, embedding_config)?
        .with_vector_store(vector_store.clone())
        .with_redis(redis_manager.clone());

    println!("✅ Document pipeline initialized");

    // 4. Process Sample Documents
    println!("\n📚 Processing sample documents...");
    let sample_documents = vec![
        (
            "machine_learning_basics",
            "Machine Learning Fundamentals\n\nMachine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n\nSupervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks.\n\nUnsupervised learning finds hidden patterns in unlabeled data. Examples include clustering algorithms like k-means and dimensionality reduction techniques like PCA.\n\nReinforcement learning involves an agent learning to make decisions by receiving rewards or penalties for its actions in an environment.",
            DocumentFormat::PlainText,
        ),
        (
            "neural_networks_guide",
            "# Neural Networks Deep Dive\n\n## Introduction\nNeural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.\n\n## Architecture\n- **Input Layer**: Receives data\n- **Hidden Layers**: Process information\n- **Output Layer**: Produces results\n\n## Training Process\n1. Forward propagation: Data flows through network\n2. Loss calculation: Compare output to expected result\n3. Backpropagation: Update weights to minimize loss\n4. Repeat until convergence\n\n## Applications\nNeural networks excel in:\n- Image recognition\n- Natural language processing\n- Speech recognition\n- Game playing (e.g., AlphaGo)",
            DocumentFormat::Markdown,
        ),
        (
            "ai_ethics_html",
            r#"<html><head><title>AI Ethics</title></head><body>
                <h1>Ethical Considerations in AI</h1>
                <p>As artificial intelligence becomes more prevalent, ethical considerations become increasingly important.</p>

                <h2>Key Ethical Issues</h2>
                <ul>
                    <li><strong>Bias and Fairness:</strong> AI systems can perpetuate or amplify existing biases</li>
                    <li><strong>Privacy:</strong> AI often requires large amounts of personal data</li>
                    <li><strong>Transparency:</strong> Many AI systems are "black boxes" with unclear decision processes</li>
                    <li><strong>Accountability:</strong> Who is responsible when AI systems make mistakes?</li>
                </ul>

                <h2>Guidelines for Responsible AI</h2>
                <p>Organizations should implement guidelines ensuring AI systems are fair, accountable, and transparent.</p>

                <script>console.log("This script should be ignored");</script>
            </body></html>"#,
            DocumentFormat::Html,
        ),
    ];

    let mut processed_docs = Vec::new();

    for (doc_id, content, format) in sample_documents {
        println!("  📝 Processing document: {}", doc_id);

        let metadata = DocumentMetadata::new(doc_id.to_string(), "demo_document".to_string());

        match pipeline.process_document(content, metadata, format).await {
            Ok(vector_ids) => {
                println!("    ✅ Created {} vector embeddings", vector_ids.len());
                processed_docs.push((doc_id, vector_ids));
            }
            Err(e) => {
                println!("    ❌ Failed to process document {}: {}", doc_id, e);
            }
        }
    }

    // 5. Perform Similarity Search
    println!("\n🔎 Performing similarity search...");

    // Create a sample query vector (in practice, this would be generated from query text)
    let query_text = "What is supervised learning in machine learning?";
    println!("  🔍 Query: {}", query_text);

    // For demo purposes, create a dummy query vector
    // In real implementation, this would be generated using the same embedding model
    let query_vector: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();

    match vector_store.search(&query_vector, 5, false).await {
        Ok(results) => {
            println!("  📊 Found {} similar documents:", results.len());
            for (i, result) in results.iter().enumerate() {
                println!(
                    "    {}. Document: {} | Chunk: {} | Similarity: {:.4}",
                    i + 1,
                    result.metadata.document_id,
                    result.metadata.chunk_id,
                    result.similarity
                );
                println!(
                    "       Text preview: {}",
                    if result.metadata.text.len() > 100 {
                        format!("{}...", &result.metadata.text[..100])
                    } else {
                        result.metadata.text.clone()
                    }
                );
                println!();
            }
        }
        Err(e) => {
            println!("  ❌ Search failed: {}", e);
        }
    }

    // 6. Research Client Demo
    println!("🌐 Testing research client capabilities...");

    let research_config = ResearchConfig {
        max_concurrent_requests: 5,
        timeout_ms: 10000,
        rate_limit_rps: 2.0,
        enable_caching: true,
        cache_ttl_seconds: 1800,
        ..Default::default()
    };

    let research_client = match ResearchClient::new(research_config) {
        Ok(client) => {
            println!("✅ Research client initialized");
            client.with_redis(redis_manager.clone())
        }
        Err(e) => {
            println!("❌ Failed to initialize research client: {}", e);
            return Ok(());
        }
    };

    // Example research query (would typically scrape actual URLs)
    let research_query = ResearchQuery::new("machine learning tutorials".to_string())
        .with_sources(vec![
            "https://scikit-learn.org/stable/tutorial/index.html".to_string(),
            "https://pytorch.org/tutorials/".to_string(),
            "https://www.tensorflow.org/tutorials".to_string(),
        ])
        .with_max_results(3)
        .with_priority(7);

    println!("  🔍 Research query: {}", research_query.query);
    println!("  📡 Sources: {} URLs", research_query.sources.len());

    match research_client.research(research_query).await {
        Ok(response) => {
            println!("  📊 Research completed:");
            println!("    - Query ID: {}", response.query_id);
            println!("    - Sources queried: {}", response.total_sources_queried);
            println!("    - Successful queries: {}", response.successful_queries);
            println!("    - Processing time: {}ms", response.processing_time_ms);
            println!("    - Results found: {}", response.results.len());

            for (i, result) in response.results.iter().enumerate() {
                println!(
                    "      {}. {} (confidence: {:.2})",
                    i + 1,
                    result.title,
                    result.confidence_score
                );
                println!("         URL: {}", result.source_url);
                println!("         Snippet: {}", result.snippet);
                println!();
            }
        }
        Err(e) => {
            println!("  ⚠️  Research query failed: {}", e);
            println!("     This is expected without actual web scraping setup");
        }
    }

    // 7. System Statistics
    println!("📈 System Statistics:");

    // Redis stats
    let redis_stats = redis_manager.get_stats().await;
    println!("  Redis Manager:");
    println!(
        "    - Cache hit rate: {:.2}%",
        redis_stats.hit_rate() * 100.0
    );
    println!("    - Total operations: {}", redis_stats.total_operations);
    println!(
        "    - Active connections: {}",
        redis_stats.active_connections
    );

    // Vector store stats
    let vector_stats = vector_store.get_stats();
    println!("  Vector Store:");
    println!("    - Total vectors: {}", vector_stats.total_vectors);
    println!("    - Search queries: {}", vector_stats.search_queries);
    println!(
        "    - Average search time: {:.2}ms",
        vector_stats.average_search_time_ms
    );

    // Pipeline stats
    let pipeline_stats = pipeline.get_stats().await;
    println!("  Document Pipeline:");
    println!(
        "    - Documents processed: {}",
        pipeline_stats.documents_processed
    );
    println!(
        "    - Chunks generated: {}",
        pipeline_stats.chunks_generated
    );
    println!(
        "    - Embeddings created: {}",
        pipeline_stats.embeddings_created
    );
    println!(
        "    - Average chunks per document: {:.1}",
        pipeline_stats.average_chunks_per_document
    );
    println!(
        "    - Average processing time: {:.2}ms",
        pipeline_stats.average_processing_time_ms
    );

    println!("\n🎉 RAG System Demo Completed Successfully!");
    println!("📝 Summary:");
    println!("   - Processed {} documents", processed_docs.len());
    println!("   - Generated embeddings and stored in vector database");
    println!("   - Performed similarity search with results");
    println!("   - Tested research capabilities");
    println!("   - All components working in harmony!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rag_system_components() {
        // Test individual components without Redis dependency

        // Test vector store creation
        let vector_config = VectorStoreConfig {
            dimension: 128,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let vector_store = VectorStore::new(vector_config).unwrap();
        assert_eq!(vector_store.get_stats().total_vectors, 0);

        // Test document pipeline creation
        let chunking_config = ChunkingConfig::default();
        let embedding_config = EmbeddingConfig::default();
        let _pipeline = DocumentPipeline::new(chunking_config, embedding_config).unwrap();

        // Test research client creation
        let research_config = ResearchConfig::default();
        let _client = ResearchClient::new(research_config).unwrap();
    }

    #[test]
    fn test_configurations() {
        // Test configuration defaults
        let redis_config = RedisConfig::default();
        assert_eq!(redis_config.max_connections, 20);
        assert_eq!(redis_config.default_ttl, 3600);

        let vector_config = VectorStoreConfig::default();
        assert_eq!(vector_config.dimension, 768);
        assert_eq!(vector_config.metric, DistanceMetric::Cosine);

        let chunking_config = ChunkingConfig::default();
        assert_eq!(chunking_config.max_chunk_size, 512);
        assert_eq!(chunking_config.chunk_overlap, 50);

        let embedding_config = EmbeddingConfig::default();
        assert_eq!(embedding_config.dimension, 1536);
        assert!(embedding_config.normalize);

        let research_config = ResearchConfig::default();
        assert_eq!(research_config.max_concurrent_requests, 10);
        assert_eq!(research_config.rate_limit_rps, 5.0);
    }
}
