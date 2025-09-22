//! Comprehensive benchmark suite for the inference engine
//!
//! These benchmarks test the performance of critical operations including
//! SIMD operations, memory management, and tensor operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gemma_inference::*;
use std::time::Duration;

/// Benchmark SIMD operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    // Test different vector sizes
    let sizes = [64, 256, 1024, 4096, 16384];

    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark scalar dot product
        group.bench_with_input(
            BenchmarkId::new("dot_product_scalar", size),
            &size,
            |bench, &_size| {
                bench.iter(|| {
                    black_box(TensorOps::dot_product_scalar(black_box(&a), black_box(&b)))
                });
            },
        );

        // Benchmark auto-selected dot product (best SIMD available)
        group.bench_with_input(
            BenchmarkId::new("dot_product_auto", size),
            &size,
            |bench, &_size| {
                bench.iter(|| {
                    black_box(TensorOps::dot_product_auto(black_box(&a), black_box(&b)).unwrap())
                });
            },
        );

        // Benchmark vector addition
        let mut result = vec![0.0f32; size];
        group.bench_with_input(
            BenchmarkId::new("vector_add_scalar", size),
            &size,
            |bench, &_size| {
                bench.iter(|| {
                    TensorOps::vector_add_scalar(black_box(&a), black_box(&b), black_box(&mut result));
                });
            },
        );

        // Benchmark specific SIMD implementations if available
        let caps = SimdCapabilities::detect();

        #[cfg(target_arch = "x86_64")]
        {
            if caps.avx2 {
                group.bench_with_input(
                    BenchmarkId::new("dot_product_avx2", size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| unsafe {
                            black_box(TensorOps::dot_product_avx2(black_box(&a), black_box(&b)))
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("vector_add_avx2", size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| unsafe {
                            TensorOps::vector_add_avx2(black_box(&a), black_box(&b), black_box(&mut result));
                        });
                    },
                );
            }

            if caps.sse4_1 {
                group.bench_with_input(
                    BenchmarkId::new("dot_product_sse", size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| unsafe {
                            black_box(TensorOps::dot_product_sse(black_box(&a), black_box(&b)))
                        });
                    },
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if caps.neon {
                group.bench_with_input(
                    BenchmarkId::new("dot_product_neon", size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| unsafe {
                            black_box(TensorOps::dot_product_neon(black_box(&a), black_box(&b)))
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("vector_add_neon", size),
                    &size,
                    |bench, &_size| {
                        bench.iter(|| unsafe {
                            TensorOps::vector_add_neon(black_box(&a), black_box(&b), black_box(&mut result));
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark memory management operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    let config = MemoryConfig::default();
    let pool = MemoryPool::new(config);

    // Benchmark tensor allocation
    let shapes = vec![
        TensorShape::new(vec![64, 64]),
        TensorShape::new(vec![128, 128]),
        TensorShape::new(vec![256, 256]),
        TensorShape::new(vec![512, 512]),
    ];

    for shape in shapes {
        let elements = shape.total_elements();
        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(
            BenchmarkId::new("tensor_allocation", elements),
            &shape,
            |bench, shape| {
                bench.iter(|| {
                    let allocation = pool.allocate_tensor(black_box(shape)).unwrap();
                    black_box(allocation);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tensor_creation", elements),
            &shape,
            |bench, shape| {
                bench.iter(|| {
                    let tensor = Tensor::zeros(black_box(shape), DataType::F32);
                    black_box(tensor);
                });
            },
        );
    }

    // Benchmark memory block operations
    let sizes = [1024, 4096, 16384, 65536];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("memory_block_alloc", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let block = MemoryBlock::new(black_box(size), 32).unwrap();
                    black_box(block);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("memory_block_zero", size),
            &size,
            |bench, &size| {
                let mut block = MemoryBlock::new(size, 32).unwrap();
                bench.iter(|| {
                    let slice = block.as_mut_slice();
                    slice.fill(black_box(0));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tensor operations
fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");

    let shapes = vec![
        TensorShape::new(vec![100, 100]),
        TensorShape::new(vec![200, 200]),
        TensorShape::new(vec![500, 500]),
    ];

    for shape in shapes {
        let elements = shape.total_elements();
        group.throughput(Throughput::Elements(elements as u64));

        let tensor1 = Tensor::ones(&shape, DataType::F32);
        let tensor2 = Tensor::ones(&shape, DataType::F32);

        group.bench_with_input(
            BenchmarkId::new("element_wise_add", elements),
            &elements,
            |bench, &_elements| {
                bench.iter(|| {
                    let result = tensor1.element_wise_add(black_box(&tensor2)).unwrap();
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("element_wise_mul", elements),
            &elements,
            |bench, &_elements| {
                bench.iter(|| {
                    let result = tensor1.element_wise_mul(black_box(&tensor2)).unwrap();
                    black_box(result);
                });
            },
        );

        // Benchmark tensor indexing
        group.bench_with_input(
            BenchmarkId::new("tensor_indexing", elements),
            &shape,
            |bench, shape| {
                bench.iter(|| {
                    let mut sum = 0.0f32;
                    for i in 0..shape.dims()[0].min(100) {
                        for j in 0..shape.dims()[1].min(100) {
                            sum += tensor1.get_f32(&[i, j]).unwrap();
                        }
                    }
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark attention operations (critical for transformer models)
fn bench_attention_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_operations");

    let configs = vec![
        (64, 8, 512),   // Small: seq_len, heads, d_model
        (128, 12, 768), // Medium
        (256, 16, 1024), // Large
    ];

    for (seq_len, num_heads, d_model) in configs {
        let head_dim = d_model / num_heads;
        let elements = seq_len * seq_len * num_heads;

        group.throughput(Throughput::Elements(elements as u64));

        // Create attention matrices
        let q_shape = TensorShape::new(vec![seq_len, d_model]);
        let k_shape = TensorShape::new(vec![seq_len, d_model]);
        let v_shape = TensorShape::new(vec![seq_len, d_model]);

        let query = Tensor::ones(&q_shape, DataType::F32);
        let key = Tensor::ones(&k_shape, DataType::F32);
        let value = Tensor::ones(&v_shape, DataType::F32);

        group.bench_with_input(
            BenchmarkId::new("attention_computation", elements),
            &elements,
            |bench, &_elements| {
                bench.iter(|| {
                    // Simulate attention computation
                    let qk = query.matmul(black_box(&key)).unwrap();
                    let scores = qk.softmax(-1).unwrap();
                    let output = scores.matmul(black_box(&value)).unwrap();
                    black_box(output);
                });
            },
        );

        // Benchmark scaled dot-product attention specifically
        group.bench_with_input(
            BenchmarkId::new("scaled_dot_product_attention", elements),
            &(seq_len, num_heads, head_dim),
            |bench, &(seq_len, num_heads, head_dim)| {
                bench.iter(|| {
                    // Simulate the core attention mechanism
                    let scale = 1.0 / (head_dim as f32).sqrt();

                    for head in 0..num_heads {
                        let q_slice = query.get_slice(head * head_dim, (head + 1) * head_dim).unwrap();
                        let k_slice = key.get_slice(head * head_dim, (head + 1) * head_dim).unwrap();
                        let v_slice = value.get_slice(head * head_dim, (head + 1) * head_dim).unwrap();

                        let scores = q_slice.matmul(&k_slice).unwrap();
                        let scaled_scores = scores.scale(scale).unwrap();
                        let attention_weights = scaled_scores.softmax(-1).unwrap();
                        let output = attention_weights.matmul(&v_slice).unwrap();

                        black_box(output);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");

    let cache_sizes = [100, 1000, 10000];
    let key_lengths = [10, 50, 100];

    for cache_size in cache_sizes {
        for key_length in key_lengths {
            let cache = KVCache::new(cache_size);

            // Generate test data
            let keys: Vec<String> = (0..cache_size)
                .map(|i| format!("{:0width$}", i, width = key_length))
                .collect();

            let values: Vec<Vec<f32>> = (0..cache_size)
                .map(|i| vec![i as f32; 256])
                .collect();

            group.throughput(Throughput::Elements(cache_size as u64));

            // Benchmark cache insertion
            group.bench_with_input(
                BenchmarkId::new("cache_insert", format!("{}_{}",cache_size, key_length)),
                &(cache_size, key_length),
                |bench, &(_cache_size, _key_length)| {
                    bench.iter(|| {
                        let cache = KVCache::new(cache_size);
                        for (key, value) in keys.iter().zip(values.iter()) {
                            cache.insert(black_box(key.clone()), black_box(value.clone()));
                        }
                        black_box(cache);
                    });
                },
            );

            // Setup cache for retrieval benchmarks
            for (key, value) in keys.iter().zip(values.iter()) {
                cache.insert(key.clone(), value.clone());
            }

            // Benchmark cache retrieval
            group.bench_with_input(
                BenchmarkId::new("cache_get", format!("{}_{}", cache_size, key_length)),
                &(cache_size, key_length),
                |bench, &(_cache_size, _key_length)| {
                    bench.iter(|| {
                        for key in &keys {
                            let value = cache.get(black_box(key));
                            black_box(value);
                        }
                    });
                },
            );

            // Benchmark cache hit/miss ratio
            let mixed_keys: Vec<String> = (0..cache_size * 2)
                .map(|i| format!("{:0width$}", i, width = key_length))
                .collect();

            group.bench_with_input(
                BenchmarkId::new("cache_mixed_access", format!("{}_{}", cache_size, key_length)),
                &(cache_size, key_length),
                |bench, &(_cache_size, _key_length)| {
                    bench.iter(|| {
                        for key in &mixed_keys {
                            let value = cache.get(black_box(key));
                            black_box(value);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark system capability detection
fn bench_capability_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability_detection");

    group.bench_function("simd_capabilities_detect", |bench| {
        bench.iter(|| {
            let caps = SimdCapabilities::detect();
            black_box(caps);
        });
    });

    group.bench_function("memory_info_detect", |bench| {
        bench.iter(|| {
            let info = MemoryInfo::detect();
            black_box(info);
        });
    });

    group.bench_function("runtime_capabilities_detect", |bench| {
        bench.iter(|| {
            let caps = RuntimeCapabilities::detect();
            black_box(caps);
        });
    });

    group.bench_function("simd_level_determination", |bench| {
        bench.iter(|| {
            let caps = SimdCapabilities::detect();
            let level = caps.best_simd_level();
            black_box(level);
        });
    });

    group.finish();
}

/// Benchmark engine lifecycle operations
fn bench_engine_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_lifecycle");

    // Benchmark engine initialization
    group.bench_function("engine_initialization", |bench| {
        bench.iter(|| {
            let config = EngineConfig::default();
            let engine = initialize_engine("bench_engine", black_box(config)).unwrap();
            black_box(engine);
            shutdown_engines();
        });
    });

    // Benchmark engine warmup
    let config = EngineConfig::default();
    let engine = initialize_engine("warmup_bench", config).unwrap();

    group.bench_function("engine_warmup", |bench| {
        bench.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    warmup_engine(black_box(&engine)).await.unwrap();
                });
            }
            start.elapsed()
        });
    });

    // Benchmark concurrent engine access
    group.bench_function("concurrent_engine_access", |bench| {
        bench.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    std::thread::spawn(move || {
                        let name = format!("concurrent_bench_{}", i);
                        let config = EngineConfig::default();
                        initialize_engine(&name, config).unwrap()
                    })
                })
                .collect();

            let engines: Vec<_> = handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect();

            black_box(engines);
            shutdown_engines();
        });
    });

    shutdown_engines();
    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    let config = MemoryConfig::default();
    let pool = MemoryPool::new(config);

    // Benchmark allocation/deallocation cycles
    group.bench_function("alloc_dealloc_cycle", |bench| {
        bench.iter(|| {
            let shape = TensorShape::new(vec![100, 100]);
            let allocation = pool.allocate_tensor(black_box(&shape)).unwrap();
            black_box(allocation);
            // Allocation is dropped here
        });
    });

    // Benchmark fragmentation resistance
    group.bench_function("fragmentation_test", |bench| {
        bench.iter(|| {
            let mut allocations = Vec::new();

            // Allocate various sizes
            for size in [10, 50, 25, 100, 5, 75, 30].iter().cycle().take(20) {
                let shape = TensorShape::new(vec![*size, *size]);
                if let Ok(allocation) = pool.allocate_tensor(&shape) {
                    allocations.push(allocation);
                }
            }

            // Drop every other allocation
            for i in (0..allocations.len()).step_by(2) {
                if i < allocations.len() {
                    allocations.remove(i);
                }
            }

            black_box(allocations);
        });
    });

    // Benchmark large allocation handling
    group.bench_function("large_allocation", |bench| {
        bench.iter(|| {
            let shape = TensorShape::new(vec![1000, 1000]);
            let allocation = pool.allocate_tensor(black_box(&shape));
            black_box(allocation);
        });
    });

    group.finish();
}

// Configure criterion
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets =
        bench_simd_operations,
        bench_memory_operations,
        bench_tensor_operations,
        bench_attention_operations,
        bench_cache_operations,
        bench_capability_detection,
        bench_engine_lifecycle,
        bench_allocation_patterns
);

criterion_main!(benches);