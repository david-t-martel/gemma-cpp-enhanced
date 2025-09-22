use std::time::Instant;
use rag_redis_system::{VectorStore, config::{VectorStoreConfig, DistanceMetric, IndexType}};

fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(count);
    let mut rng_state = 42u64;

    for _ in 0..count {
        let mut vector = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = ((rng_state >> 16) & 0xFFFF) as f32 / 65536.0 - 0.5;
            vector.push(random_val);
        }

        // Normalize vector
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            vector.iter_mut().for_each(|x| *x /= magnitude);
        }

        vectors.push(vector);
    }

    vectors
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RAG-Redis System Baseline Performance Test ===");

    let dimension = 384;
    let dataset_sizes = [100, 500, 1000, 2000];

    for &size in &dataset_sizes {
        println!("\n--- Testing with {} vectors ({}D) ---", size, dimension);

        // Create vector store
        let config = VectorStoreConfig {
            dimension,
            distance_metric: DistanceMetric::Cosine,
            index_type: IndexType::Flat,
            hnsw_m: 16,
            hnsw_ef_construction: 50,
            hnsw_ef_search: 20,
            max_vectors: Some(size + 100),
        };

        let store = VectorStore::new(config)?;
        let vectors = generate_test_vectors(size + 1, dimension);

        // Benchmark vector addition
        println!("Benchmarking vector addition...");
        let start = Instant::now();
        for (i, vector) in vectors[..size].iter().enumerate() {
            let id = format!("vec_{}", i);
            let metadata = serde_json::json!({
                "doc_id": format!("doc_{}", i / 10),
                "chunk_id": i,
                "category": match i % 4 {
                    0 => "tech",
                    1 => "science",
                    2 => "business",
                    _ => "general"
                }
            });
            store.add_vector(&id, vector, metadata)?;
        }
        let addition_time = start.elapsed();
        println!("  Added {} vectors in {:?} ({:.2} vectors/sec)",
                size, addition_time, size as f64 / addition_time.as_secs_f64());

        // Benchmark search
        let query_vector = &vectors[size];
        let search_limits = [1, 5, 10, 20];

        for &limit in &search_limits {
            println!("Benchmarking search (limit={})...", limit);
            let mut total_time = std::time::Duration::new(0, 0);
            let num_queries = 10;

            for _ in 0..num_queries {
                let start = Instant::now();
                let results = store.search(query_vector, limit, None)?;
                total_time += start.elapsed();

                if results.len() != limit.min(size) {
                    println!("  Warning: Expected {} results, got {}", limit.min(size), results.len());
                }
            }

            let avg_time = total_time / num_queries;
            println!("  Average search time (limit={}): {:?} ({:.2} queries/sec)",
                    limit, avg_time, 1.0 / avg_time.as_secs_f64());
        }

        // Memory usage
        let stats = store.get_stats();
        println!("Memory usage: {} bytes ({:.2} MB)",
                stats.memory_usage, stats.memory_usage as f64 / (1024.0 * 1024.0));
    }

    println!("\n=== Distance Metric Performance ===");

    // Test different distance metrics
    let metrics = [DistanceMetric::Cosine, DistanceMetric::Euclidean, DistanceMetric::DotProduct];
    let test_vectors = generate_test_vectors(2, 768);

    for metric in &metrics {
        println!("\nTesting {:?} distance:", metric);
        let config = VectorStoreConfig {
            dimension: 768,
            distance_metric: *metric,
            index_type: IndexType::Flat,
            hnsw_m: 16,
            hnsw_ef_construction: 50,
            hnsw_ef_search: 20,
            max_vectors: Some(1000),
        };

        let store = VectorStore::new(config)?;

        // Add test vectors
        store.add_vector("test1", &test_vectors[0], serde_json::json!({"test": true}))?;
        store.add_vector("test2", &test_vectors[1], serde_json::json!({"test": true}))?;

        // Benchmark distance calculation
        let mut total_time = std::time::Duration::new(0, 0);
        let num_iterations = 1000;

        for _ in 0..num_iterations {
            let start = Instant::now();
            let _results = store.search(&test_vectors[0], 2, None)?;
            total_time += start.elapsed();
        }

        let avg_time = total_time / num_iterations;
        println!("  Average distance calculation: {:?} ({:.2} ops/sec)",
                avg_time, 1.0 / avg_time.as_secs_f64());
    }

    println!("\n=== Baseline Performance Test Complete ===");
    Ok(())
}
