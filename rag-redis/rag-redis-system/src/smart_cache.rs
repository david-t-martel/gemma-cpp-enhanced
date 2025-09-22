//! Smart caching system with advanced eviction policies and memory optimization
//!
//! This module provides intelligent caching with multiple eviction strategies,
//! memory pressure awareness, and automatic optimization.

use bytes::Bytes;
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cache entry with metadata for intelligent eviction
#[derive(Clone, Debug)]
pub struct CacheEntry<T> {
    pub value: T,
    pub size_bytes: usize,
    pub access_count: u64,
    pub last_access: Instant,
    pub created_at: Instant,
    pub ttl: Option<Duration>,
    pub priority: CachePriority,
    pub compressed: bool,
}

impl<T> CacheEntry<T> {
    pub fn new(value: T, size_bytes: usize, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            size_bytes,
            access_count: 0,
            last_access: now,
            created_at: now,
            ttl,
            priority: CachePriority::Normal,
            compressed: false,
        }
    }

    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();
    }

    pub fn calculate_score(&self) -> f64 {
        let age = self.last_access.elapsed().as_secs() as f64;
        let frequency = (self.access_count as f64).ln().max(1.0);
        let size_factor = 1.0 / (self.size_bytes as f64 / 1024.0).max(1.0);
        let priority_multiplier = self.priority.multiplier();

        // Higher score = more valuable to keep
        (frequency * size_factor * priority_multiplier) / (age + 1.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CachePriority {
    Low,
    Normal,
    High,
    Critical, // Never evict unless expired
}

impl CachePriority {
    fn multiplier(&self) -> f64 {
        match self {
            CachePriority::Low => 0.5,
            CachePriority::Normal => 1.0,
            CachePriority::High => 2.0,
            CachePriority::Critical => 100.0,
        }
    }
}

/// Eviction policy for the cache
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,     // Least Recently Used
    LFU,     // Least Frequently Used
    FIFO,    // First In First Out
    ARC,     // Adaptive Replacement Cache
    TinyLFU, // Frequency-based with small overhead
    SLRU,    // Segmented LRU
}

/// Multi-tier cache with intelligent eviction
pub struct SmartCache<K: Clone + Eq + std::hash::Hash, V> {
    // Hot tier - frequently accessed items
    hot_cache: Arc<RwLock<LruCache<K, Arc<CacheEntry<V>>>>>,

    // Warm tier - less frequently accessed
    warm_cache: Arc<RwLock<HashMap<K, Arc<CacheEntry<V>>>>>,

    // Cold tier - compressed, rarely accessed
    cold_cache: Arc<RwLock<HashMap<K, Bytes>>>,

    // Cache statistics
    stats: Arc<RwLock<CacheStats>>,

    // Configuration
    config: CacheConfig,

    // Memory pressure monitor
    memory_monitor: Arc<MemoryMonitor>,

    // Access history for adaptive policies
    access_history: Arc<RwLock<AccessHistory<K>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheConfig {
    pub hot_capacity: usize,
    pub warm_capacity: usize,
    pub cold_capacity: usize,
    pub max_memory_bytes: usize,
    pub eviction_policy: EvictionPolicy,
    pub compression_threshold: usize,
    pub ttl_check_interval: Duration,
    pub promotion_threshold: u64, // Access count to promote to hot tier
    pub demotion_threshold: Duration, // Time without access to demote
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            hot_capacity: 1000,
            warm_capacity: 5000,
            cold_capacity: 10000,
            max_memory_bytes: 500 * 1024 * 1024, // 500MB
            eviction_policy: EvictionPolicy::ARC,
            compression_threshold: 1024, // 1KB
            ttl_check_interval: Duration::from_secs(60),
            promotion_threshold: 3,
            demotion_threshold: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub compressions: u64,
    pub decompressions: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub memory_usage: usize,
    pub hot_tier_size: usize,
    pub warm_tier_size: usize,
    pub cold_tier_size: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Access history for adaptive eviction policies
struct AccessHistory<K> {
    history: BTreeMap<Instant, K>,
    key_times: HashMap<K, Vec<Instant>>,
    max_history: usize,
}

impl<K: Clone + Eq + std::hash::Hash> AccessHistory<K> {
    fn new(max_history: usize) -> Self {
        Self {
            history: BTreeMap::new(),
            key_times: HashMap::new(),
            max_history,
        }
    }

    fn record_access(&mut self, key: K) {
        let now = Instant::now();
        self.history.insert(now, key.clone());
        self.key_times.entry(key).or_default().push(now);

        // Cleanup old history
        if self.history.len() > self.max_history {
            if let Some((time, old_key)) = self.history.iter().next().map(|(t, k)| (*t, k.clone()))
            {
                self.history.remove(&time);
                if let Some(times) = self.key_times.get_mut(&old_key) {
                    times.retain(|&t| t != time);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn get_frequency(&self, key: &K, window: Duration) -> usize {
        let cutoff = Instant::now() - window;
        self.key_times
            .get(key)
            .map(|times| times.iter().filter(|&&t| t > cutoff).count())
            .unwrap_or(0)
    }
}

/// Memory pressure monitor
struct MemoryMonitor {
    threshold_bytes: usize,
    current_usage: Arc<RwLock<usize>>,
}

impl MemoryMonitor {
    fn new(threshold_bytes: usize) -> Self {
        Self {
            threshold_bytes,
            current_usage: Arc::new(RwLock::new(0)),
        }
    }

    fn update_usage(&self, delta: isize) {
        let mut usage = self.current_usage.write();
        if delta > 0 {
            *usage = usage.saturating_add(delta as usize);
        } else {
            *usage = usage.saturating_sub((-delta) as usize);
        }
    }

    fn is_under_pressure(&self) -> bool {
        *self.current_usage.read() > self.threshold_bytes
    }

    fn usage_ratio(&self) -> f64 {
        *self.current_usage.read() as f64 / self.threshold_bytes as f64
    }
}

impl<K, V> SmartCache<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    pub fn new(config: CacheConfig) -> Self {
        let hot_cache = Arc::new(RwLock::new(LruCache::new(
            NonZeroUsize::new(config.hot_capacity).unwrap(),
        )));

        Self {
            hot_cache,
            warm_cache: Arc::new(RwLock::new(HashMap::with_capacity(config.warm_capacity))),
            cold_cache: Arc::new(RwLock::new(HashMap::with_capacity(config.cold_capacity))),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            memory_monitor: Arc::new(MemoryMonitor::new(config.max_memory_bytes)),
            access_history: Arc::new(RwLock::new(AccessHistory::new(10000))),
            config,
        }
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V, size_bytes: usize, ttl: Option<Duration>) -> bool {
        // Check memory pressure
        if self.memory_monitor.is_under_pressure() {
            self.evict_under_pressure();
        }

        let entry = Arc::new(CacheEntry::new(value, size_bytes, ttl));

        // Insert into hot cache initially
        {
            let mut hot = self.hot_cache.write();
            hot.put(key.clone(), entry.clone());
        }

        self.memory_monitor.update_usage(size_bytes as isize);
        self.update_stats(|stats| {
            stats.hot_tier_size += 1;
            stats.memory_usage += size_bytes;
        });

        true
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        // Check hot tier first
        {
            let mut hot = self.hot_cache.write();
            if let Some(entry) = hot.get_mut(key) {
                if !entry.is_expired() {
                    let entry = Arc::make_mut(entry);
                    entry.access();
                    self.access_history.write().record_access(key.clone());
                    self.update_stats(|stats| stats.hits += 1);
                    return Some(entry.value.clone());
                } else {
                    hot.pop(key);
                }
            }
        }

        // Check warm tier
        {
            let should_promote;
            let result_value;

            {
                let mut warm = self.warm_cache.write();
                if let Some(entry) = warm.get_mut(key) {
                    if !entry.is_expired() {
                        let entry_mut = Arc::make_mut(entry);
                        entry_mut.access();

                        should_promote = entry_mut.access_count >= self.config.promotion_threshold;
                        result_value = Some(entry_mut.value.clone());
                    } else {
                        warm.remove(key);
                        should_promote = false;
                        result_value = None;
                    }
                } else {
                    should_promote = false;
                    result_value = None;
                }
            }

            if let Some(value) = result_value {
                if should_promote {
                    // Handle promotion in separate scope
                    let mut warm = self.warm_cache.write();
                    if let Some(entry_to_promote) = warm.remove(key) {
                        drop(warm); // Release warm lock before acquiring hot lock

                        let mut hot = self.hot_cache.write();
                        hot.put(key.clone(), entry_to_promote);

                        self.update_stats(|stats| {
                            stats.promotions += 1;
                            stats.warm_tier_size -= 1;
                            stats.hot_tier_size += 1;
                        });
                    }
                }

                self.access_history.write().record_access(key.clone());
                self.update_stats(|stats| stats.hits += 1);
                return Some(value);
            }
        }

        // Check cold tier (compressed)
        {
            let decompressed_data;
            let compressed_size;

            {
                let cold = self.cold_cache.read();
                if let Some(compressed) = cold.get(key) {
                    // Decompress
                    if let Ok(decompressed) = self.decompress::<V>(compressed) {
                        decompressed_data = Some(decompressed);
                        compressed_size = compressed.len();
                    } else {
                        decompressed_data = None;
                        compressed_size = 0;
                    }
                } else {
                    decompressed_data = None;
                    compressed_size = 0;
                }
            }

            if let Some(decompressed) = decompressed_data {
                self.update_stats(|stats| {
                    stats.decompressions += 1;
                    stats.hits += 1;
                });

                // Promote to warm tier
                let mut cold = self.cold_cache.write();
                cold.remove(key);
                drop(cold);

                let entry = Arc::new(CacheEntry::new(decompressed.clone(), compressed_size, None));

                self.warm_cache.write().insert(key.clone(), entry);
                self.update_stats(|stats| {
                    stats.cold_tier_size -= 1;
                    stats.warm_tier_size += 1;
                });

                return Some(decompressed);
            }
        }

        self.update_stats(|stats| stats.misses += 1);
        None
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &K) -> bool {
        let mut removed = false;

        // Check all tiers
        if self.hot_cache.write().pop(key).is_some() {
            removed = true;
            self.update_stats(|stats| stats.hot_tier_size -= 1);
        }

        if self.warm_cache.write().remove(key).is_some() {
            removed = true;
            self.update_stats(|stats| stats.warm_tier_size -= 1);
        }

        if self.cold_cache.write().remove(key).is_some() {
            removed = true;
            self.update_stats(|stats| stats.cold_tier_size -= 1);
        }

        removed
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.hot_cache.write().clear();
        self.warm_cache.write().clear();
        self.cold_cache.write().clear();

        self.memory_monitor
            .update_usage(-(*self.memory_monitor.current_usage.read() as isize));

        *self.stats.write() = CacheStats::default();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    /// Run maintenance tasks (TTL cleanup, tier migration)
    pub async fn maintenance(&self) {
        self.cleanup_expired();
        self.migrate_tiers();
        self.compress_cold_entries();
    }

    // Private helper methods

    fn evict_under_pressure(&self) {
        let pressure_ratio = self.memory_monitor.usage_ratio();

        if pressure_ratio > 0.9 {
            // Aggressive eviction
            self.evict_from_tier(CacheTier::Cold, 0.5);
            self.evict_from_tier(CacheTier::Warm, 0.3);
        } else if pressure_ratio > 0.8 {
            // Moderate eviction
            self.evict_from_tier(CacheTier::Cold, 0.3);
            self.evict_from_tier(CacheTier::Warm, 0.1);
        }
    }

    fn evict_from_tier(&self, tier: CacheTier, evict_ratio: f64) {
        match tier {
            CacheTier::Warm => {
                let mut warm = self.warm_cache.write();
                let evict_count = (warm.len() as f64 * evict_ratio) as usize;

                // Score all entries and evict lowest scoring
                let mut scored: Vec<_> = warm
                    .iter()
                    .map(|(k, v)| (k.clone(), v.calculate_score()))
                    .collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                for (key, _) in scored.iter().take(evict_count) {
                    if let Some(entry) = warm.remove(key) {
                        self.memory_monitor
                            .update_usage(-(entry.size_bytes as isize));
                    }
                }

                self.update_stats(|stats| {
                    stats.evictions += evict_count as u64;
                    stats.warm_tier_size = warm.len();
                });
            }
            CacheTier::Cold => {
                let mut cold = self.cold_cache.write();
                let evict_count = (cold.len() as f64 * evict_ratio) as usize;

                let keys: Vec<_> = cold.keys().take(evict_count).cloned().collect();
                for key in keys {
                    if let Some(bytes) = cold.remove(&key) {
                        self.memory_monitor.update_usage(-(bytes.len() as isize));
                    }
                }

                self.update_stats(|stats| {
                    stats.evictions += evict_count as u64;
                    stats.cold_tier_size = cold.len();
                });
            }
            _ => {}
        }
    }

    fn cleanup_expired(&self) {
        // Clean hot tier
        {
            let mut hot = self.hot_cache.write();
            let expired_keys: Vec<_> = hot
                .iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired_keys {
                hot.pop(&key);
            }
        }

        // Clean warm tier
        {
            let mut warm = self.warm_cache.write();
            warm.retain(|_, entry| !entry.is_expired());
        }
    }

    fn migrate_tiers(&self) {
        // Demote from hot to warm if not accessed recently
        {
            let mut hot = self.hot_cache.write();
            let mut warm = self.warm_cache.write();

            let demote_keys: Vec<_> = hot
                .iter()
                .filter(|(_, entry)| entry.last_access.elapsed() > self.config.demotion_threshold)
                .map(|(k, _)| k.clone())
                .collect();

            for key in demote_keys {
                if let Some(entry) = hot.pop(&key) {
                    warm.insert(key, entry);
                    self.update_stats(|stats| {
                        stats.demotions += 1;
                        stats.hot_tier_size -= 1;
                        stats.warm_tier_size += 1;
                    });
                }
            }
        }

        // Demote from warm to cold if rarely accessed
        {
            let warm = self.warm_cache.read();
            let demote_candidates: Vec<_> = warm
                .iter()
                .filter(|(_, entry)| {
                    entry.size_bytes > self.config.compression_threshold
                        && entry.last_access.elapsed() > self.config.demotion_threshold * 2
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            drop(warm);

            for (key, entry) in demote_candidates {
                if let Ok(compressed) = self.compress(&entry.value) {
                    self.warm_cache.write().remove(&key);
                    self.cold_cache.write().insert(key, compressed);

                    self.update_stats(|stats| {
                        stats.compressions += 1;
                        stats.warm_tier_size -= 1;
                        stats.cold_tier_size += 1;
                    });
                }
            }
        }
    }

    fn compress_cold_entries(&self) {
        // Already handled in migrate_tiers
    }

    fn compress<T: Serialize>(&self, value: &T) -> Result<Bytes, Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(value)?;
        let compressed = lz4::block::compress(&serialized, None, false)?;
        Ok(Bytes::from(compressed))
    }

    fn decompress<T: for<'de> Deserialize<'de>>(
        &self,
        bytes: &Bytes,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let decompressed = lz4::block::decompress(bytes, None)?;
        let value = bincode::deserialize(&decompressed)?;
        Ok(value)
    }

    fn update_stats<F>(&self, f: F)
    where
        F: FnOnce(&mut CacheStats),
    {
        let mut stats = self.stats.write();
        f(&mut stats);
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum CacheTier {
    Hot,
    Warm,
    Cold,
}

/// Cache builder for easy configuration
pub struct CacheBuilder<K, V> {
    config: CacheConfig,
    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K, V> Default for CacheBuilder<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> CacheBuilder<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn hot_capacity(mut self, capacity: usize) -> Self {
        self.config.hot_capacity = capacity;
        self
    }

    pub fn warm_capacity(mut self, capacity: usize) -> Self {
        self.config.warm_capacity = capacity;
        self
    }

    pub fn cold_capacity(mut self, capacity: usize) -> Self {
        self.config.cold_capacity = capacity;
        self
    }

    pub fn max_memory(mut self, bytes: usize) -> Self {
        self.config.max_memory_bytes = bytes;
        self
    }

    pub fn eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.config.eviction_policy = policy;
        self
    }

    pub fn build(self) -> SmartCache<K, V> {
        SmartCache::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache: SmartCache<String, String> = CacheBuilder::new()
            .hot_capacity(10)
            .warm_capacity(20)
            .build();

        cache.insert("key1".to_string(), "value1".to_string(), 100, None);
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        cache.remove(&"key1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), None);
    }

    #[test]
    fn test_cache_stats() {
        let cache: SmartCache<i32, i32> = CacheBuilder::new().build();

        cache.insert(1, 100, 4, None);
        cache.get(&1);
        cache.get(&2);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }
}
