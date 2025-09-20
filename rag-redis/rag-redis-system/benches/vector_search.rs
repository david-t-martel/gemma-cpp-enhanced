//! Vector search benchmarks for the RAG-Redis system
//!
//! These benchmarks measure the performance of:
//! - Vector storage and indexing
//! - Distance metric calculations
//! - Search operations with various parameters
//! - SIMD optimizations vs fallbacks
//! - Memory usage and efficiency
//! - Scalability with different dataset sizes

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;
use std::time::Duration;

use rag_redis_system::{
    config::VectorStoreConfig,
    vector_store::{DistanceMetric, SearchFilter, VectorIndex, VectorMetadata},
    VectorStore,
};

/// Generate deterministic test vectors for consistent benchmarking
fn generate_benchmark_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(count);
    let mut rng_state = seed;

    for _ in 0..count {
        let mut vector = Vec::with_capacity(dimension);

        for _ in 0..dimension {
            // Linear congruential generator for reproducible results
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = ((rng_state >> 16) & 0xFFFF) as f32 / 65536.0 - 0.5;
            vector.push(random_val);
        }

        // L2 normalize vector for consistent magnitudes
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            vector.iter_mut().for_each(|x| *x /= magnitude);
        }

        vectors.push(vector);
    }

    vectors
}

/// Create a vector store with specified configuration
fn create_benchmark_store(dimension: usize, metric: DistanceMetric) -> VectorStore {
    let config = VectorStoreConfig {
        dimension,
        distance_metric: metric,
        index_type: rag_redis_system::config::IndexType::Hnsw,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 50,
        max_vectors: Some(100_000),
    };

    VectorStore::new(config).expect("Failed to create vector store")
}

/// Benchmark vector addition performance
fn bench_vector_addition(c: &mut Criterion) {
    let dimensions = [128, 384, 768, 1536];
    let vector_counts = [100, 1000, 10000];

    let mut group = c.benchmark_group("vector_addition");

    for &dimension in &dimensions {
        for &count in &vector_counts {
            let store = create_benchmark_store(dimension, DistanceMetric::Cosine);
            let vectors = generate_benchmark_vectors(count, dimension, 42);

            group.throughput(Throughput::Elements(count as u64));
            group.bench_with_input(
                BenchmarkId::new("cosine", format!("{}d_{}v", dimension, count)),
                &(store, vectors),
                |b, (store, vectors)| {
                    b.iter_batched(
                        || store.clone(),
                        |store| {
                            for (i, vector) in vectors.iter().enumerate() {
                                let id = format!("bench_vec_{}", i);
                                let metadata = serde_json::json!({
                                    "document_id": format!("doc_{}", i / 100),
                                    "chunk_id": format!("chunk_{}", i),
                                    "index": i
                                });
                                store.add_vector(&id, vector, metadata).unwrap();
                            }
                        },
                        criterion::BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Benchmark distance metric calculations
fn bench_distance_metrics(c: &mut Criterion) {
    let dimensions = [128, 384, 768];
    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ];

    let mut group = c.benchmark_group("distance_metrics");
    group.sample_size(1000);

    for &dimension in &dimensions {
        let vectors = generate_benchmark_vectors(2, dimension, 123);
        let query_vector = &vectors[0];
        let target_vector = &vectors[1];

        for &metric in &metrics {
            let config = VectorStoreConfig {
                dimension,
                distance_metric: metric,
                index_type: rag_redis_system::config::IndexType::Hnsw,
                hnsw_m: 16,
                hnsw_ef_construction: 50,
                hnsw_ef_search: 20,
                max_vectors: Some(1000),
            };

            let index = VectorIndex::new(config).expect("Failed to create index");

            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", metric), format!("{}d", dimension)),
                &(index, query_vector.clone(), target_vector.clone()),
                |b, (_index, query, target)| {
                    b.iter(|| {
                        // Direct distance calculation benchmark
                        let calculator =
                            rag_redis_system::vector_store::SimdDistanceCalculator::new(dimension);
                        match metric {
                            DistanceMetric::Cosine => {
                                calculator.cosine_similarity(query, target).unwrap()
                            }
                            DistanceMetric::Euclidean => {
                                calculator.euclidean_distance(query, target).unwrap()
                            }
                            DistanceMetric::DotProduct => {
                                calculator.dot_product(query, target).unwrap()
                            }
                            DistanceMetric::Manhattan => {
                                calculator.manhattan_distance(query, target).unwrap()
                            }
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark search performance with varying dataset sizes
fn bench_search_performance(c: &mut Criterion) {
    let dimension = 384; // Common embedding dimension
    let dataset_sizes = [100, 500, 1000, 5000, 10000];
    let search_limits = [1, 5, 10, 50];

    let mut group = c.benchmark_group("search_performance");
    group.measurement_time(Duration::from_secs(10));

    for &dataset_size in &dataset_sizes {
        for &search_limit in &search_limits {
            let store = create_benchmark_store(dimension, DistanceMetric::Cosine);
            let vectors = generate_benchmark_vectors(dataset_size + 1, dimension, 456);

            // Add vectors to store (excluding last one for querying)
            for (i, vector) in vectors[..dataset_size].iter().enumerate() {
                let id = format!("search_vec_{}", i);
                let metadata = serde_json::json!({
                    "document_id": format!("doc_{}", i / 10),
                    "chunk_id": format!("chunk_{}", i),
                    "category": match i % 4 {
                        0 => "technology",
                        1 => "science",
                        2 => "business",
                        _ => "general",
                    }
                });
                store.add_vector(&id, vector, metadata).unwrap();
            }

            let query_vector = &vectors[dataset_size]; // Use last vector as query

            group.throughput(Throughput::Elements(dataset_size as u64));
            group.bench_with_input(
                BenchmarkId::new("search", format!("{}v_limit{}", dataset_size, search_limit)),
                &(store, query_vector.clone(), search_limit),
                |b, (store, query, limit)| b.iter(|| store.search(query, *limit, None).unwrap()),
            );
        }
    }

    group.finish();
}

/// Benchmark search with filters
fn bench_filtered_search(c: &mut Criterion) {
    let dimension = 768;
    let dataset_size = 5000;
    let store = create_benchmark_store(dimension, DistanceMetric::Cosine);
    let vectors = generate_benchmark_vectors(dataset_size + 1, dimension, 789);

    // Add vectors with diverse metadata for filtering
    for (i, vector) in vectors[..dataset_size].iter().enumerate() {
        let id = format!("filtered_vec_{}", i);
        let metadata = serde_json::json!({
            "document_id": format!("doc_{}", i % 100), // 100 different documents
            "chunk_id": format!("chunk_{}", i),
            "category": match i % 5 {
                0 => "AI",
                1 => "ML",
                2 => "Data",
                3 => "Tech",
                _ => "General",
            },
            "priority": if i < dataset_size / 2 { "high" } else { "low" },
            "tags": match i % 3 {
                0 => vec!["important", "recent"],
                1 => vec!["important"],
                _ => vec!["archived"],
            }
        });
        store.add_vector(&id, vector, metadata).unwrap();
    }

    let query_vector = &vectors[dataset_size];

    let mut group = c.benchmark_group("filtered_search");

    // Benchmark unfiltered search
    group.bench_function("no_filter", |b| {
        b.iter(|| store.search(query_vector, 20, None).unwrap())
    });

    // Note: Since the current implementation doesn't expose SearchFilter directly,
    // we'll benchmark the overhead of metadata processing during search
    group.bench_function("with_metadata_processing", |b| {
        b.iter(|| {
            let results = store.search(query_vector, 50, None).unwrap();
            // Simulate filter processing
            results
                .into_iter()
                .filter(|(_, _, metadata)| {
                    metadata
                        .get("category")
                        .and_then(|v| v.as_str())
                        .map(|s| s == "AI" || s == "ML")
                        .unwrap_or(false)
                })
                .take(20)
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

/// Benchmark concurrent search operations
fn bench_concurrent_search(c: &mut Criterion) {
    let dimension = 384;
    let dataset_size = 2000;
    let store = create_benchmark_store(dimension, DistanceMetric::Cosine);
    let vectors = generate_benchmark_vectors(dataset_size + 10, dimension, 101112);

    // Add vectors to store
    for (i, vector) in vectors[..dataset_size].iter().enumerate() {
        let id = format!("concurrent_vec_{}", i);
        let metadata = serde_json::json!({
            "document_id": format!("doc_{}", i / 20),
            "chunk_id": format!("chunk_{}", i),
            "thread_group": i % 4
        });
        store.add_vector(&id, vector, metadata).unwrap();
    }

    let query_vectors: Vec<Vec<f32>> = vectors[dataset_size..].to_vec();
    let store = std::sync::Arc::new(store);

    let mut group = c.benchmark_group("concurrent_search");

    // Benchmark single-threaded search
    group.bench_function("single_thread", |b| {
        b.iter(|| {
            for query in &query_vectors {
                store.search(query, 10, None).unwrap();
            }
        })
    });

    // Benchmark multi-threaded search (using rayon for CPU parallelism)
    group.bench_function("multi_thread", |b| {
        b.iter(|| {
            query_vectors.par_iter().for_each(|query| {
                store.search(query, 10, None).unwrap();
            });
        })
    });

    group.finish();
}

/// Benchmark memory usage and efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let dimensions = [128, 384, 768];
    let vector_counts = [1000, 5000, 10000];

    let mut group = c.benchmark_group("memory_efficiency");

    for &dimension in &dimensions {
        for &count in &vector_counts {
            group.bench_with_input(
                BenchmarkId::new("memory_usage", format!("{}d_{}v", dimension, count)),
                &(dimension, count),
                |b, &(dim, cnt)| {
                    b.iter_batched(
                        || {
                            let store = create_benchmark_store(dim, DistanceMetric::Cosine);
                            let vectors = generate_benchmark_vectors(cnt, dim, 131415);
                            (store, vectors)
                        },
                        |(store, vectors)| {
                            // Add all vectors
                            for (i, vector) in vectors.iter().enumerate() {
                                let id = format!("mem_vec_{}", i);
                                let metadata = serde_json::json!({"index": i});
                                store.add_vector(&id, &vector, metadata).unwrap();
                            }

                            // Get memory stats
                            let stats = store.get_stats();
                            (stats.vector_count, stats.memory_usage)
                        },
                        criterion::BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Benchmark vector store clearing and rebuilding
fn bench_vector_store_operations(c: &mut Criterion) {
    let dimension = 512;
    let vector_count = 5000;

    let mut group = c.benchmark_group("store_operations");

    // Benchmark clearing
    group.bench_function("clear_store", |b| {
        b.iter_batched(
            || {
                let store = create_benchmark_store(dimension, DistanceMetric::Cosine);
                let vectors = generate_benchmark_vectors(vector_count, dimension, 161718);

                for (i, vector) in vectors.iter().enumerate() {
                    let id = format!("clear_vec_{}", i);
                    let metadata = serde_json::json!({"index": i});
                    store.add_vector(&id, vector, metadata).unwrap();
                }

                store
            },
            |store| store.clear().unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    // Benchmark vector removal
    group.bench_function("remove_vectors", |b| {
        b.iter_batched(
            || {
                let store = create_benchmark_store(dimension, DistanceMetric::Cosine);
                let vectors = generate_benchmark_vectors(1000, dimension, 192021);

                let ids: Vec<String> = (0..1000)
                    .map(|i| {
                        let id = format!("remove_vec_{}", i);
                        let metadata = serde_json::json!({"index": i});
                        store.add_vector(&id, &vectors[i], metadata).unwrap();
                        id
                    })
                    .collect();

                (store, ids)
            },
            |(store, ids)| {
                // Remove every other vector
                for (i, id) in ids.iter().enumerate() {
                    if i % 2 == 0 {
                        store.remove_vector(id).unwrap();
                    }
                }
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

/// Benchmark different distance metrics head-to-head
fn bench_distance_metric_comparison(c: &mut Criterion) {
    let dimension = 768;
    let dataset_size = 1000;
    let query_vector = generate_benchmark_vectors(1, dimension, 222324)[0].clone();

    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ];

    let mut group = c.benchmark_group("distance_metric_comparison");

    for &metric in &metrics {
        let store = create_benchmark_store(dimension, metric);
        let vectors = generate_benchmark_vectors(dataset_size, dimension, 252627);

        // Add vectors to store
        for (i, vector) in vectors.iter().enumerate() {
            let id = format!("comp_vec_{}", i);
            let metadata = serde_json::json!({"index": i});
            store.add_vector(&id, vector, metadata).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("search", format!("{:?}", metric)),
            &(store, query_vector.clone()),
            |b, (store, query)| b.iter(|| store.search(query, 20, None).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark SIMD vs fallback implementations
fn bench_simd_vs_fallback(c: &mut Criterion) {
    let dimensions = [128, 256, 512, 768, 1024];

    let mut group = c.benchmark_group("simd_vs_fallback");
    group.sample_size(10000);

    for &dimension in &dimensions {
        let vectors = generate_benchmark_vectors(2, dimension, 282930);
        let vec_a = &vectors[0];
        let vec_b = &vectors[1];

        // Benchmark cosine similarity (most commonly used)
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", format!("{}d", dimension)),
            &(vec_a.clone(), vec_b.clone()),
            |bench, (a, b)| {
                bench.iter(|| {
                    let calculator =
                        rag_redis_system::vector_store::SimdDistanceCalculator::new(dimension);
                    calculator.cosine_similarity(a, b).unwrap()
                })
            },
        );

        // Manual fallback implementation for comparison
        group.bench_with_input(
            BenchmarkId::new("cosine_fallback", format!("{}d", dimension)),
            &(vec_a.clone(), vec_b.clone()),
            |bench, (a, b)| {
                bench.iter(|| {
                    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                    if norm_a == 0.0 || norm_b == 0.0 {
                        0.0
                    } else {
                        dot_product / (norm_a * norm_b)
                    }
                })
            },
        );
    }

    group.finish();
}

/// Comprehensive benchmark suite
criterion_group!(
    name = vector_search_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(20))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets =
        bench_vector_addition,
        bench_distance_metrics,
        bench_search_performance,
        bench_filtered_search,
        bench_concurrent_search,
        bench_memory_efficiency,
        bench_vector_store_operations,
        bench_distance_metric_comparison,
        bench_simd_vs_fallback
);

criterion_main!(vector_search_benches);

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    /// Test that benchmark functions don't panic
    #[test]
    fn test_benchmark_vector_generation() {
        let vectors = generate_benchmark_vectors(100, 384, 42);
        assert_eq!(vectors.len(), 100);
        assert_eq!(vectors[0].len(), 384);

        // Verify vectors are normalized
        for vector in &vectors {
            let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (magnitude - 1.0).abs() < 0.001,
                "Vector should be normalized"
            );
        }
    }

    #[test]
    fn test_benchmark_store_creation() {
        let store = create_benchmark_store(256, DistanceMetric::Cosine);
        let stats = store.get_stats();
        assert_eq!(stats.dimension, 256);
        assert_eq!(stats.distance_metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_vector_operations_dont_panic() {
        let store = create_benchmark_store(128, DistanceMetric::Cosine);
        let vectors = generate_benchmark_vectors(10, 128, 12345);

        // Test adding vectors
        for (i, vector) in vectors.iter().enumerate() {
            let id = format!("test_vec_{}", i);
            let metadata = serde_json::json!({"test": true});
            store.add_vector(&id, vector, metadata).unwrap();
        }

        // Test search
        let results = store.search(&vectors[0], 5, None).unwrap();
        assert!(!results.is_empty());

        // Test clearing
        store.clear().unwrap();
        assert_eq!(store.get_stats().vector_count, 0);
    }
}
