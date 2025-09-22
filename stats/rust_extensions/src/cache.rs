//! High-performance caching layer for Gemma operations
//!
//! This module provides thread-safe, high-performance caching implementations
//! optimized for AI workloads with features like TTL, memory management,
//! and async support.

use crate::error::{GemmaError, GemmaResult};
use pyo3::prelude::*;
// TODO: Re-enable when pyo3-asyncio is updated for PyO3 0.24+
// use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::{RwLock, Mutex};
use dashmap::DashMap;
use lru::LruCache;
use tokio::time::sleep;
use serde::{Deserialize, Serialize};
use fnv::FnvBuildHasher;
use ahash::AHasher;
use std::hash::{BuildHasher, Hasher};

/// Fast hash builder using AHash
type FastHasher = ahash::RandomState;

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
}

impl<V> CacheEntry<V> {
    fn new(value: V, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Cache statistics
#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    #[pyo3(get)]
    pub hits: u64,
    #[pyo3(get)]
    pub misses: u64,
    #[pyo3(get)]
    pub evictions: u64,
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub capacity: usize,
    #[pyo3(get)]
    pub memory_usage: usize,
}

#[pymethods]
impl CacheStats {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get cache hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            (self.hits as f64) / (total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get cache miss rate as percentage
    pub fn miss_rate(&self) -> f64 {
        100.0 - self.hit_rate()
    }

    /// Get average memory usage per entry
    pub fn avg_entry_size(&self) -> f64 {
        if self.size > 0 {
            self.memory_usage as f64 / self.size as f64
        } else {
            0.0
        }
    }

    fn to_dict(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("hits".to_string(), self.hits as f64);
        stats.insert("misses".to_string(), self.misses as f64);
        stats.insert("evictions".to_string(), self.evictions as f64);
        stats.insert("size".to_string(), self.size as f64);
        stats.insert("capacity".to_string(), self.capacity as f64);
        stats.insert("memory_usage".to_string(), self.memory_usage as f64);
        stats.insert("hit_rate".to_string(), self.hit_rate());
        stats.insert("miss_rate".to_string(), self.miss_rate());
        stats.insert("avg_entry_size".to_string(), self.avg_entry_size());
        stats
    }
}

/// Thread-safe LRU cache with TTL support
#[pyclass]
#[derive(Clone)]
pub struct LRUCache {
    cache: Arc<Mutex<LruCache<String, CacheEntry<String>, FastHasher>>>,
    stats: Arc<RwLock<CacheStats>>,
    max_memory: Option<usize>,
    current_memory: Arc<AtomicUsize>,
    default_ttl: Option<Duration>,
}

#[pymethods]
impl LRUCache {
    #[new]
    pub fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(GemmaError::config("Cache capacity must be greater than 0").into());
        }

        Ok(Self {
            cache: Arc::new(Mutex::new(LruCache::with_hasher(
                capacity.try_into().unwrap(),
                FastHasher::default(),
            ))),
            stats: Arc::new(RwLock::new(CacheStats {
                capacity,
                ..Default::default()
            })),
            max_memory: None,
            current_memory: Arc::new(AtomicUsize::new(0)),
            default_ttl: None,
        })
    }

    /// Create cache with memory limit
    #[staticmethod]
    pub fn with_memory_limit(capacity: usize, max_memory_mb: usize) -> PyResult<Self> {
        let mut cache = Self::new(capacity)?;
        cache.max_memory = Some(max_memory_mb * 1024 * 1024);
        Ok(cache)
    }

    /// Create cache with default TTL
    #[staticmethod]
    pub fn with_ttl(capacity: usize, ttl_seconds: u64) -> PyResult<Self> {
        let mut cache = Self::new(capacity)?;
        cache.default_ttl = Some(Duration::from_secs(ttl_seconds));
        Ok(cache)
    }

    /// Put a value in the cache
    pub fn put(&self, key: String, value: String) -> PyResult<()> {
        self.put_with_ttl(key, value, None)
    }

    /// Put a value with specific TTL
    pub fn put_with_ttl(&self, key: String, value: String, ttl_seconds: Option<u64>) -> PyResult<()> {
        let ttl = ttl_seconds
            .map(Duration::from_secs)
            .or(self.default_ttl);

        let entry = CacheEntry::new(value.clone(), ttl);
        let entry_size = key.len() + value.len() + std::mem::size_of::<CacheEntry<String>>();

        // Check memory limit
        if let Some(max_mem) = self.max_memory {
            let current_mem = self.current_memory.load(Ordering::Relaxed);
            if current_mem + entry_size > max_mem {
                return Err(GemmaError::memory("Cache memory limit exceeded").into());
            }
        }

        let mut cache = self.cache.lock();
        let mut stats = self.stats.write();

        // Remove old entry if exists
        if let Some(old_entry) = cache.get(&key) {
            let old_size = key.len() + old_entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
            self.current_memory.fetch_sub(old_size, Ordering::Relaxed);
            stats.size -= 1;
            stats.memory_usage -= old_size;
        }

        // Add new entry
        if let Some(evicted) = cache.push(key.clone(), entry) {
            let evicted_size = evicted.0.len() + evicted.1.value.len() + std::mem::size_of::<CacheEntry<String>>();
            self.current_memory.fetch_sub(evicted_size, Ordering::Relaxed);
            stats.evictions += 1;
            stats.memory_usage -= evicted_size;
        } else {
            stats.size += 1;
        }

        self.current_memory.fetch_add(entry_size, Ordering::Relaxed);
        stats.memory_usage += entry_size;

        Ok(())
    }

    /// Get a value from the cache
    pub fn get(&self, key: &str) -> PyResult<Option<String>> {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.write();

        if let Some(entry) = cache.get_mut(key) {
            if entry.is_expired() {
                // Remove expired entry
                let entry_size = key.len() + entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
                cache.pop(key);
                self.current_memory.fetch_sub(entry_size, Ordering::Relaxed);
                stats.size -= 1;
                stats.memory_usage -= entry_size;
                stats.misses += 1;
                Ok(None)
            } else {
                entry.touch();
                stats.hits += 1;
                Ok(Some(entry.value.clone()))
            }
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    /// Remove a key from the cache
    pub fn remove(&self, key: &str) -> PyResult<bool> {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.write();

        if let Some(entry) = cache.pop(key) {
            let entry_size = key.len() + entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
            self.current_memory.fetch_sub(entry_size, Ordering::Relaxed);
            stats.size -= 1;
            stats.memory_usage -= entry_size;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if key exists in cache
    pub fn contains(&self, key: &str) -> bool {
        let cache = self.cache.lock();
        if let Some(entry) = cache.peek(key) {
            !entry.is_expired()
        } else {
            false
        }
    }

    /// Clear all entries from cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.write();

        cache.clear();
        self.current_memory.store(0, Ordering::Relaxed);
        stats.size = 0;
        stats.memory_usage = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    /// Clean expired entries
    pub fn cleanup_expired(&self) -> usize {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.write();
        let mut cleaned = 0;
        let mut keys_to_remove = Vec::new();

        // Collect expired keys
        for (key, entry) in cache.iter() {
            if entry.is_expired() {
                keys_to_remove.push(key.clone());
            }
        }

        // Remove expired entries
        for key in keys_to_remove {
            if let Some(entry) = cache.pop(&key) {
                let entry_size = key.len() + entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
                self.current_memory.fetch_sub(entry_size, Ordering::Relaxed);
                stats.size -= 1;
                stats.memory_usage -= entry_size;
                cleaned += 1;
            }
        }

        cleaned
    }

    /// Get current size
    pub fn size(&self) -> usize {
        self.stats.read().size
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.stats.read().capacity
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed)
    }
}

/// High-performance concurrent cache using DashMap
#[pyclass]
pub struct ConcurrentCache {
    cache: Arc<DashMap<String, CacheEntry<String>, FastHasher>>,
    stats: Arc<RwLock<CacheStats>>,
    max_size: usize,
    max_memory: Option<usize>,
    current_memory: Arc<AtomicUsize>,
    default_ttl: Option<Duration>,
}

#[pymethods]
impl ConcurrentCache {
    #[new]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(DashMap::with_hasher(FastHasher::default())),
            stats: Arc::new(RwLock::new(CacheStats {
                capacity: max_size,
                ..Default::default()
            })),
            max_size,
            max_memory: None,
            current_memory: Arc::new(AtomicUsize::new(0)),
            default_ttl: None,
        }
    }

    /// Put a value in the cache
    pub fn put(&self, key: String, value: String) -> PyResult<()> {
        if self.cache.len() >= self.max_size {
            return Err(GemmaError::cache("Cache is full").into());
        }

        let entry = CacheEntry::new(value.clone(), self.default_ttl);
        let entry_size = key.len() + value.len() + std::mem::size_of::<CacheEntry<String>>();

        // Check memory limit
        if let Some(max_mem) = self.max_memory {
            let current_mem = self.current_memory.load(Ordering::Relaxed);
            if current_mem + entry_size > max_mem {
                return Err(GemmaError::memory("Cache memory limit exceeded").into());
            }
        }

        self.cache.insert(key, entry);
        self.current_memory.fetch_add(entry_size, Ordering::Relaxed);

        let mut stats = self.stats.write();
        stats.size = self.cache.len();
        stats.memory_usage = self.current_memory.load(Ordering::Relaxed);

        Ok(())
    }

    /// Get a value from the cache
    pub fn get(&self, key: &str) -> PyResult<Option<String>> {
        let mut stats = self.stats.write();

        if let Some(mut entry) = self.cache.get_mut(key) {
            if entry.is_expired() {
                drop(entry);
                let entry_size = key.len() + std::mem::size_of::<CacheEntry<String>>();
                if let Some((_, old_entry)) = self.cache.remove(key) {
                    let actual_size = key.len() + old_entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
                    self.current_memory.fetch_sub(actual_size, Ordering::Relaxed);
                    stats.memory_usage = self.current_memory.load(Ordering::Relaxed);
                }
                stats.size = self.cache.len();
                stats.misses += 1;
                Ok(None)
            } else {
                entry.touch();
                stats.hits += 1;
                Ok(Some(entry.value.clone()))
            }
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }

    /// Remove a key from the cache
    pub fn remove(&self, key: &str) -> PyResult<bool> {
        if let Some((_, entry)) = self.cache.remove(key) {
            let entry_size = key.len() + entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
            self.current_memory.fetch_sub(entry_size, Ordering::Relaxed);

            let mut stats = self.stats.write();
            stats.size = self.cache.len();
            stats.memory_usage = self.current_memory.load(Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let mut stats = self.stats.write();
        stats.size = self.cache.len();
        stats.memory_usage = self.current_memory.load(Ordering::Relaxed);
        stats.clone()
    }
}

/// Cache manager for managing multiple cache instances
#[pyclass]
pub struct CacheManager {
    caches: Arc<DashMap<String, Arc<LRUCache>>>,
    default_capacity: usize,
    stats: Arc<RwLock<HashMap<String, CacheStats>>>,
}

#[pymethods]
impl CacheManager {
    #[new]
    pub fn new(default_capacity: Option<usize>) -> Self {
        Self {
            caches: Arc::new(DashMap::new()),
            default_capacity: default_capacity.unwrap_or(1000),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a cache
    pub fn get_cache(&self, name: &str) -> PyResult<Py<LRUCache>> {
        Python::with_gil(|py| {
            if let Some(cache) = self.caches.get(name) {
                Ok(Py::new(py, (**cache).clone())?)
            } else {
                let cache = Arc::new(LRUCache::new(self.default_capacity)?);
                self.caches.insert(name.to_string(), cache.clone());
                Ok(Py::new(py, (**cache).clone())?)
            }
        })
    }

    /// Create a cache with specific capacity
    pub fn create_cache(&self, name: &str, capacity: usize) -> PyResult<()> {
        let cache = Arc::new(LRUCache::new(capacity)?);
        self.caches.insert(name.to_string(), cache);
        Ok(())
    }

    /// Remove a cache
    pub fn remove_cache(&self, name: &str) -> bool {
        self.caches.remove(name).is_some()
    }

    /// List all cache names
    pub fn list_caches(&self) -> Vec<String> {
        self.caches.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get combined statistics for all caches
    pub fn get_all_stats(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut all_stats = HashMap::new();
        for entry in self.caches.iter() {
            let cache_name = entry.key();
            let cache = entry.value();
            all_stats.insert(cache_name.clone(), cache.stats().to_dict());
        }
        all_stats
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        for entry in self.caches.iter() {
            entry.value().clear();
        }
    }

    /// Cleanup expired entries in all caches
    pub fn cleanup_all_expired(&self) -> HashMap<String, usize> {
        let mut results = HashMap::new();
        for entry in self.caches.iter() {
            let cache_name = entry.key();
            let cache = entry.value();
            let cleaned = cache.cleanup_expired();
            results.insert(cache_name.clone(), cleaned);
        }
        results
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(&self, interval_seconds: u64) -> PyResult<()> {
        let caches = self.caches.clone();
        let interval = Duration::from_secs(interval_seconds);

        tokio::spawn(async move {
            loop {
                sleep(interval).await;

                for entry in caches.iter() {
                    entry.value().cleanup_expired();
                }
            }
        });

        Ok(())
    }
}

// TODO: Move async methods to the main pymethods block
/*
impl LRUCache {
    /// Async put operation
    pub fn put_async<'p>(&self, py: Python<'p>, key: String, value: String) -> PyResult<&'p PyAny> {
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        let current_memory = Arc::clone(&self.current_memory);
        let max_memory = self.max_memory;
        let default_ttl = self.default_ttl;

        future_into_py(py, async move {
            let entry = CacheEntry::new(value.clone(), default_ttl);
            let entry_size = key.len() + value.len() + std::mem::size_of::<CacheEntry<String>>();

            // Check memory limit
            if let Some(max_mem) = max_memory {
                let current_mem = current_memory.load(Ordering::Relaxed);
                if current_mem + entry_size > max_mem {
                    return Err(GemmaError::memory("Cache memory limit exceeded"));
                }
            }

            let mut cache_guard = cache.lock();
            let mut stats_guard = stats.write();

            // Remove old entry if exists
            if let Some(old_entry) = cache_guard.get(&key) {
                let old_size = key.len() + old_entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
                current_memory.fetch_sub(old_size, Ordering::Relaxed);
                stats_guard.size -= 1;
                stats_guard.memory_usage -= old_size;
            }

            // Add new entry
            if let Some(evicted) = cache_guard.push(key.clone(), entry) {
                let evicted_size = evicted.0.len() + evicted.1.value.len() + std::mem::size_of::<CacheEntry<String>>();
                current_memory.fetch_sub(evicted_size, Ordering::Relaxed);
                stats_guard.evictions += 1;
                stats_guard.memory_usage -= evicted_size;
            } else {
                stats_guard.size += 1;
            }

            current_memory.fetch_add(entry_size, Ordering::Relaxed);
            stats_guard.memory_usage += entry_size;

            Ok(())
        })
    }

    /// Async get operation
    pub fn get_async<'p>(&self, py: Python<'p>, key: String) -> PyResult<&'p PyAny> {
        let cache = Arc::clone(&self.cache);
        let stats = Arc::clone(&self.stats);
        let current_memory = Arc::clone(&self.current_memory);

        future_into_py(py, async move {
            let mut cache_guard = cache.lock();
            let mut stats_guard = stats.write();

            if let Some(entry) = cache_guard.get_mut(&key) {
                if entry.is_expired() {
                    let entry_size = key.len() + entry.value.len() + std::mem::size_of::<CacheEntry<String>>();
                    cache_guard.pop(&key);
                    current_memory.fetch_sub(entry_size, Ordering::Relaxed);
                    stats_guard.size -= 1;
                    stats_guard.memory_usage -= entry_size;
                    stats_guard.misses += 1;
                    Ok(None::<String>)
                } else {
                    entry.touch();
                    stats_guard.hits += 1;
                    Ok(Some(entry.value.clone()))
                }
            } else {
                stats_guard.misses += 1;
                Ok(None::<String>)
            }
        })
    }
}
*/

/// Register the cache module with Python
pub fn register_module(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<LRUCache>()?;
    module.add_class::<ConcurrentCache>()?;
    module.add_class::<CacheManager>()?;
    module.add_class::<CacheStats>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_lru_cache_basic_operations() {
        let cache = LRUCache::new(2).unwrap();

        // Test put and get
        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        assert_eq!(cache.get("key1").unwrap(), Some("value1".to_string()));

        // Test miss
        assert_eq!(cache.get("nonexistent").unwrap(), None);

        // Test eviction
        cache.put("key2".to_string(), "value2".to_string()).unwrap();
        cache.put("key3".to_string(), "value3".to_string()).unwrap();

        // key1 should be evicted
        assert_eq!(cache.get("key1").unwrap(), None);
        assert_eq!(cache.get("key2").unwrap(), Some("value2".to_string()));
        assert_eq!(cache.get("key3").unwrap(), Some("value3".to_string()));
    }

    #[test]
    fn test_cache_stats() {
        let cache = LRUCache::new(10).unwrap();

        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        let _ = cache.get("key1").unwrap();
        let _ = cache.get("nonexistent").unwrap();

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
        assert!(stats.hit_rate() > 0.0);
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = LRUCache::with_ttl(10, 1).unwrap(); // 1 second TTL

        cache.put("key1".to_string(), "value1".to_string()).unwrap();
        assert_eq!(cache.get("key1").unwrap(), Some("value1".to_string()));

        // Sleep to let TTL expire
        thread::sleep(Duration::from_millis(1100));
        assert_eq!(cache.get("key1").unwrap(), None);
    }

    #[test]
    fn test_concurrent_cache() {
        let cache = Arc::new(ConcurrentCache::new(100));
        let mut handles = vec![];

        // Test concurrent writes
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let key = format!("key_{}_{}", i, j);
                    let value = format!("value_{}_{}", i, j);
                    cache_clone.put(key, value).unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.size, 100);
    }

    #[test]
    fn test_cache_manager() {
        let manager = CacheManager::new(Some(10));

        // Create and access caches
        manager.create_cache("cache1", 5).unwrap();
        manager.create_cache("cache2", 15).unwrap();

        let cache_names = manager.list_caches();
        assert!(cache_names.contains(&"cache1".to_string()));
        assert!(cache_names.contains(&"cache2".to_string()));

        // Test removal
        assert!(manager.remove_cache("cache1"));
        assert!(!manager.remove_cache("nonexistent"));
    }

    #[test]
    fn test_memory_limit() {
        let cache = LRUCache::with_memory_limit(2, 1).unwrap(); // 1MB limit

        // This should work
        cache.put("small".to_string(), "value".to_string()).unwrap();

        // This should fail due to memory limit
        let large_value = "x".repeat(2 * 1024 * 1024); // 2MB
        let result = cache.put("large".to_string(), large_value);
        assert!(result.is_err());
    }
}
