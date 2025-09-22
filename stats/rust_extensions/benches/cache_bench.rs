use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gemma_extensions::cache::{CacheManager, ConcurrentCache, LRUCache};
use std::sync::Arc;
use std::thread;

fn lru_cache_benchmark(c: &mut Criterion) {
    let cache = LRUCache::new(1000).unwrap();

    // Pre-populate cache
    for i in 0..500 {
        cache
            .put(format!("key_{}", i), format!("value_{}", i))
            .unwrap();
    }

    let mut group = c.benchmark_group("lru_cache");

    group.bench_function("put", |b| {
        let mut counter = 0;
        b.iter(|| {
            let key = format!("bench_key_{}", counter);
            let value = format!("bench_value_{}", counter);
            counter += 1;
            cache.put(black_box(key), black_box(value))
        })
    });

    group.bench_function("get_hit", |b| {
        b.iter(|| {
            let key = format!("key_{}", black_box(250)); // Should exist
            cache.get(&key)
        })
    });

    group.bench_function("get_miss", |b| {
        b.iter(|| {
            let key = format!("nonexistent_{}", black_box(999999));
            cache.get(&key)
        })
    });

    group.finish();
}

fn concurrent_cache_benchmark(c: &mut Criterion) {
    let cache = ConcurrentCache::new(1000);

    // Pre-populate cache
    for i in 0..500 {
        cache
            .put(format!("key_{}", i), format!("value_{}", i))
            .unwrap();
    }

    let mut group = c.benchmark_group("concurrent_cache");

    group.bench_function("put", |b| {
        let mut counter = 0;
        b.iter(|| {
            let key = format!("bench_key_{}", counter);
            let value = format!("bench_value_{}", counter);
            counter += 1;
            cache.put(black_box(key), black_box(value))
        })
    });

    group.bench_function("get_hit", |b| {
        b.iter(|| {
            let key = format!("key_{}", black_box(250));
            cache.get(&key)
        })
    });

    group.bench_function("get_miss", |b| {
        b.iter(|| {
            let key = format!("nonexistent_{}", black_box(999999));
            cache.get(&key)
        })
    });

    group.finish();
}

fn cache_contention_benchmark(c: &mut Criterion) {
    let cache = Arc::new(ConcurrentCache::new(10000));
    let num_threads = 8;
    let operations_per_thread = 1000;

    let mut group = c.benchmark_group("cache_contention");
    group.throughput(Throughput::Elements(
        (num_threads * operations_per_thread) as u64,
    ));

    group.bench_function("concurrent_mixed_operations", |b| {
        b.iter(|| {
            let mut handles = vec![];

            for thread_id in 0..num_threads {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    for op_id in 0..operations_per_thread {
                        let key = format!("thread_{}_op_{}", thread_id, op_id);
                        let value = format!("value_{}", op_id);

                        // Mix of operations: 50% put, 40% get, 10% remove
                        match op_id % 10 {
                            0..=4 => {
                                let _ = cache_clone.put(key, value);
                            }
                            5..=8 => {
                                let _ = cache_clone.get(&key);
                            }
                            9 => {
                                let _ = cache_clone.remove(&key);
                            }
                            _ => unreachable!(),
                        }
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

fn cache_manager_benchmark(c: &mut Criterion) {
    let manager = CacheManager::new(Some(1000));

    // Create multiple caches
    for i in 0..10 {
        manager.create_cache(&format!("cache_{}", i), 100).unwrap();
    }

    let mut group = c.benchmark_group("cache_manager");

    group.bench_function("get_cache", |b| {
        b.iter(|| {
            let cache_name = format!("cache_{}", black_box(5));
            manager.get_cache(&cache_name)
        })
    });

    group.bench_function("create_cache", |b| {
        let mut counter = 1000;
        b.iter(|| {
            let cache_name = format!("dynamic_cache_{}", counter);
            counter += 1;
            manager.create_cache(&cache_name, 50)
        })
    });

    group.bench_function("list_caches", |b| b.iter(|| manager.list_caches()));

    group.bench_function("get_all_stats", |b| b.iter(|| manager.get_all_stats()));

    group.finish();
}

fn cache_memory_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_memory");

    let small_cache = LRUCache::new(100).unwrap();
    let large_cache = LRUCache::new(10000).unwrap();

    // Benchmark memory usage patterns
    group.bench_function("small_cache_fill", |b| {
        b.iter(|| {
            let cache = LRUCache::new(100).unwrap();
            for i in 0..200 {
                let key = format!("key_{}", i);
                let value = "x".repeat(100); // 100 byte values
                let _ = cache.put(key, value);
            }
            black_box(cache);
        })
    });

    group.bench_function("large_cache_fill", |b| {
        b.iter(|| {
            let cache = LRUCache::new(1000).unwrap();
            for i in 0..2000 {
                let key = format!("key_{}", i);
                let value = "x".repeat(1000); // 1KB values
                let _ = cache.put(key, value);
            }
            black_box(cache);
        })
    });

    group.bench_function("memory_usage_tracking", |b| {
        b.iter(|| {
            let memory_usage = large_cache.memory_usage();
            black_box(memory_usage);
        })
    });

    group.finish();
}

fn cache_ttl_benchmark(c: &mut Criterion) {
    let cache = LRUCache::with_ttl(1000, 60).unwrap(); // 60 second TTL

    // Pre-populate with items
    for i in 0..500 {
        cache
            .put(format!("key_{}", i), format!("value_{}", i))
            .unwrap();
    }

    let mut group = c.benchmark_group("cache_ttl");

    group.bench_function("put_with_ttl", |b| {
        let mut counter = 1000;
        b.iter(|| {
            let key = format!("ttl_key_{}", counter);
            let value = format!("ttl_value_{}", counter);
            counter += 1;
            cache.put_with_ttl(black_box(key), black_box(value), Some(30))
        })
    });

    group.bench_function("cleanup_expired", |b| b.iter(|| cache.cleanup_expired()));

    group.finish();
}

criterion_group!(
    benches,
    lru_cache_benchmark,
    concurrent_cache_benchmark,
    cache_contention_benchmark,
    cache_manager_benchmark,
    cache_memory_benchmark,
    cache_ttl_benchmark
);
criterion_main!(benches);
