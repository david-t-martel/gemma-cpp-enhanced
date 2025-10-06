use rag_redis_system::embedding::{EmbeddingService, EmbeddingFactory};
use rag_redis_system::config::{EmbeddingConfig, EmbeddingProvider};
use std::time::Duration;

/// Example demonstrating Candle-based embedding generation
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔥 RAG-Redis Candle Embedding Demo");

    // Configuration for Candle-based embeddings
    let embedding_config = EmbeddingConfig {
        provider: EmbeddingProvider::Candle,
        model: "distilbert-base-uncased".to_string(),
        dimension: 768,
        batch_size: 32,
        cache_embeddings: true,
        cache_ttl: Duration::from_secs(3600),
    };

    // Create embedding service
    println!("📦 Creating Candle embedding service...");
    let embedding_service = match EmbeddingFactory::create(&embedding_config).await {
        Ok(service) => {
            println!("✅ Successfully created Candle embedding service");
            service
        }
        Err(e) => {
            println!("⚠️  Failed to create Candle service, falling back to external: {}", e);
            // Fallback to external service for demo
            let fallback_config = EmbeddingConfig {
                provider: EmbeddingProvider::Custom("http://localhost:8000/embed".to_string()),
                ..embedding_config
            };
            EmbeddingFactory::create(&fallback_config).await?
        }
    };

    // Test texts for embedding generation
    let test_texts = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Machine learning with Rust and Candle is powerful".to_string(),
        "RAG systems combine retrieval and generation effectively".to_string(),
        "Vector databases enable semantic search capabilities".to_string(),
    ];

    // Generate embeddings for individual texts
    println!("\n🔢 Generating individual embeddings...");
    for (i, text) in test_texts.iter().enumerate() {
        match embedding_service.embed_text(text).await {
            Ok(embedding) => {
                println!("✅ Text {}: {} -> [{}...] (dim: {})",
                         i + 1,
                         &text[..50.min(text.len())],
                         &embedding[..3].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "),
                         embedding.len());
            }
            Err(e) => {
                println!("❌ Failed to embed text {}: {}", i + 1, e);
            }
        }
    }

    // Generate batch embeddings
    println!("\n📊 Generating batch embeddings...");
    match embedding_service.embed_batch(&test_texts).await {
        Ok(embeddings) => {
            println!("✅ Generated {} embeddings in batch", embeddings.len());
            for (i, embedding) in embeddings.iter().enumerate() {
                println!("   Batch {}: [{}...] (dim: {})",
                         i + 1,
                         &embedding[..3].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(", "),
                         embedding.len());
            }
        }
        Err(e) => {
            println!("❌ Failed to generate batch embeddings: {}", e);
        }
    }

    // Display service information
    println!("\n📋 Service Information:");
    println!("   Model: {}", embedding_service.model_name());
    println!("   Dimension: {}", embedding_service.dimension());

    println!("\n🎉 Demo completed successfully!");

    Ok(())
}