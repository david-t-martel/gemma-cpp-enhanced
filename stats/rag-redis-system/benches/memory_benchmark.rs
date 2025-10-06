//! Memory performance benchmarks for the RAG-Redis system
//!
//! This benchmark suite measures memory usage and performance improvements
//! from the optimization strategies.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rag_redis_system::{
    memory_pool::{Arena, ObjectPool, SlabAllocator, VectorPool},
    memory_profiler::{MemoryProfiler, MemoryScope, MemorySnapshot},
    smart_cache::{CacheBuilder, CacheConfig, SmartCache},
    vector_store::{VectorIndex, VectorStore},
};
use rand::prelude::*;
use std::sync::Arc;
use std::time::Duration;

/// Benchmark vector allocation with and without pooling
fn bench_vector_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_allocation");

    // Standard allocation
    group.bench_function("standard_allocation", |b| {
        b.iter(|| {
            let mut vectors = Vec::new();
            for _ in 0..1000 {
                let vec: Vec<f32> = vec![0.0; 768]; // Common embedding dimension
                vectors.push(black_box(vec));
            }
            vectors.clear();
        });
    });

    // Pooled allocation
    let pool: VectorPool<f32> = VectorPool::new(768, 100);
    pool.prewarm(50);

    group.bench_function("pooled_allocation", |b| {
        b.iter(|| {
            let mut vectors = Vec::new();
            for _ in 0..1000 {
                let vec = pool.acquire();
                vectors.push(black_box(vec));
            }
            vectors.clear();
        });
    });

    group.finish();
}

/// Benchmark cache operations with different eviction policies
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");

    // Create test data
    let mut rng = thread_rng();
    let test_data: Vec<(String, Vec<f32>)> = (0..10000)
        .map(|i| {
            let key = format!("key_{}", i);
            let value: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
            (key, value)
        })
        .collect();

    // Standard HashMap
    group.bench_function("hashmap_cache", |b| {
        use std::collections::HashMap;
        let mut cache = HashMap::new();

        b.iter(|| {
            for (key, value) in &test_data[..1000] {
                cache.insert(key.clone(), value.clone());
            }

            for _ in 0..100 {
                let idx = rng.gen_range(0..1000);
                let _ = cache.get(&test_data[idx].0);
            }
        });
    });

    // Smart cache with ARC eviction
    let smart_cache: SmartCache<String, Vec<f32>> = CacheBuilder::new()
        .hot_capacity(100)
        .warm_capacity(500)
        .cold_capacity(1000)
        .max_memory(10 * 1024 * 1024) // 10MB
        .build();

    group.bench_function("smart_cache_arc", |b| {
        b.iter(|| {
            for (key, value) in &test_data[..1000] {
                smart_cache.insert(
                    key.clone(),
                    value.clone(),
                    value.len() * 4,
                    Some(Duration::from_secs(3600)),
                );
            }

            for _ in 0..100 {
                let idx = rng.gen_range(0..1000);
                let _ = smart_cache.get(&test_data[idx].0);
            }
        });
    });

    group.finish();
}

/// Benchmark memory fragmentation with different allocators
fn bench_memory_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_fragmentation");

    // Standard allocation (causes fragmentation)
    group.bench_function("standard_alloc_fragmentation", |b| {
        b.iter(|| {
            let mut allocations: Vec<Vec<u8>> = Vec::new();

            // Allocate various sizes
            for i in 0..1000 {
                let size = 100 + (i % 10) * 100;
                allocations.push(vec![0u8; size]);
            }

            // Deallocate every other one (creates fragmentation)
            for i in (0..allocations.len()).step_by(2).rev() {
                allocations.remove(i);
            }

            // Allocate more
            for i in 0..500 {
                let size = 200 + (i % 5) * 200;
                allocations.push(vec![0u8; size]);
            }
        });
    });

    // Slab allocator (reduces fragmentation)
    group.bench_function("slab_alloc_no_fragmentation", |b| {
        b.iter(|| {
            let mut allocator = SlabAllocator::<Vec<u8>>::new(100);
            let mut refs = Vec::new();

            // Allocate
            for i in 0..1000 {
                let size = 100 + (i % 10) * 100;
                let data = vec![0u8; size];
                refs.push(allocator.allocate(data));
            }

            // Deallocate every other one
            for i in (0..refs.len()).step_by(2).rev() {
                allocator.deallocate(refs[i]);
                refs.remove(i);
            }

            // Allocate more
            for i in 0..500 {
                let size = 200 + (i % 5) * 200;
                let data = vec![0u8; size];
                refs.push(allocator.allocate(data));
            }
        });
    });

    // Arena allocator (batch deallocation)
    group.bench_function("arena_alloc_batch", |b| {
        b.iter(|| {
            let mut arena = Arena::new(64 * 1024); // 64KB chunks
            let mut _ptrs = Vec::new();

            // Allocate various sizes
            for i in 0..1000 {
                let size = 100 + (i % 10) * 100;
                let ptr = arena.allocate(size);
                _ptrs.push(ptr);
            }

            // Reset arena (deallocates everything at once)
            arena.reset();

            // Allocate more
            for i in 0..500 {
                let size = 200 + (i % 5) * 200;
                let ptr = arena.allocate(size);
                _ptrs.push(ptr);
            }
        });
    });

    group.finish();
}

/// Benchmark vector store operations with memory optimizations
fn bench_vector_store_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_store");

    // Create test vectors
    let mut rng = thread_rng();
    let test_vectors: Vec<(String, Vec<f32>, serde_json::Value)> = (0..1000)
        .map(|i| {
            let id = format!("vec_{}", i);
            let vector: Vec<f32> = (0..768).map(|_| rng.gen()).collect();
            let metadata = serde_json::json!({
                "document_id": format!("doc_{}", i / 10),
                "chunk_id": format!("chunk_{}", i),
            });
            (id, vector, metadata)
        })
        .collect();

    // Standard vector store
    let config = rag_redis_system::config::VectorStoreConfig {
        dimension: 768,
        distance_metric: rag_redis_system::config::DistanceMetric::Cosine,
        index_type: rag_redis_system::config::IndexType::Flat,
        hnsw_m: 16,
        hnsw_ef_construction: 50,
        hnsw_ef_search: 20,
        max_vectors: Some(10000),
    };

    let store = VectorStore::new(config.clone()).unwrap();

    group.bench_function("vector_insert", |b| {
        b.iter(|| {
            for (id, vector, metadata) in &test_vectors[..100] {
                store.add_vector(id, vector, metadata.clone()).unwrap();
            }
        });
    });

    // Pre-populate for search benchmark
    for (id, vector, metadata) in &test_vectors {
        store.add_vector(id, vector, metadata.clone()).unwrap();
    }

    group.bench_function("vector_search", |b| {
        let query_vector: Vec<f32> = (0..768).map(|_| rng.gen()).collect();

        b.iter(|| {
            let results = store.search(&query_vector, 10, None).unwrap();
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark memory profiling overhead
fn bench_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_overhead");

    // Baseline without profiling
    group.bench_function("baseline_no_profiling", |b| {
        b.iter(|| {
            let mut data = Vec::new();
            for i in 0..10000 {
                data.push(vec![0u8; 1024]);
                if i % 100 == 0 {
                    data.clear();
                }
            }
        });
    });

    // With memory profiling
    let profiler = Arc::new(MemoryProfiler::new());

    group.bench_function("with_profiling", |b| {
        b.iter(|| {
            let _scope = MemoryScope::new("benchmark", profiler.clone());

            let mut data = Vec::new();
            for i in 0..10000 {
                data.push(vec![0u8; 1024]);
                if i % 100 == 0 {
                    data.clear();
                    let snapshot = MemorySnapshot::capture();
                    profiler.record_snapshot(snapshot);
                }
            }
        });
    });

    group.finish();
}

/// Benchmark compression strategies for cold storage
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    let mut rng = thread_rng();

    // Generate test data of various sizes
    let small_data: Vec<u8> = (0..1024).map(|_| rng.gen()).collect();
    let medium_data: Vec<u8> = (0..10240).map(|_| rng.gen()).collect();
    let large_data: Vec<u8> = (0..102400).map(|_| rng.gen()).collect();

    for (name, data) in &[
        ("small_1kb", &small_data),
        ("medium_10kb", &medium_data),
        ("large_100kb", &large_data),
    ] {
        group.bench_with_input(BenchmarkId::new("lz4_compress", name), data, |b, data| {
            b.iter(|| {
                let compressed = lz4::block::compress(data, None, false).unwrap();
                black_box(compressed);
            });
        });

        let compressed = lz4::block::compress(data, None, false).unwrap();

        group.bench_with_input(
            BenchmarkId::new("lz4_decompress", name),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    let decompressed = lz4::block::decompress(compressed, None).unwrap();
                    black_box(decompressed);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_allocation,
    bench_cache_operations,
    bench_memory_fragmentation,
    bench_vector_store_operations,
    bench_profiling_overhead,
    bench_compression
);

criterion_main!(benches);
