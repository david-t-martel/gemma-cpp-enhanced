//! KV cache and attention caching for inference

use crate::error::{InferenceError, InferenceResult};
use crate::tensor::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// KV cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub max_sequence_length: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_sequence_length: 2048,
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
        }
    }
}

/// Key-Value cache for attention layers
pub struct KVCache {
    config: CacheConfig,
    cache_entries: RwLock<HashMap<String, CacheEntry>>,
    usage_stats: RwLock<CacheStats>,
}

/// Single cache entry for a sequence
struct CacheEntry {
    key_cache: Vec<Tensor>,   // Per layer
    value_cache: Vec<Tensor>, // Per layer
    sequence_length: usize,
    last_accessed: std::time::Instant,
}

/// Cache statistics
#[derive(Debug, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    memory_usage: usize,
}

/// Attention cache for storing attention weights
pub struct AttentionCache {
    cache: RwLock<HashMap<String, AttentionEntry>>,
}

struct AttentionEntry {
    attention_weights: Tensor,
    timestamp: std::time::Instant,
}

impl KVCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache_entries: RwLock::new(HashMap::new()),
            usage_stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Get cached key-value tensors for a sequence
    pub fn get(&self, sequence_id: &str) -> Option<(Vec<Tensor>, Vec<Tensor>)> {
        let mut cache = self.cache_entries.write();
        let mut stats = self.usage_stats.write();

        if let Some(entry) = cache.get_mut(sequence_id) {
            entry.last_accessed = std::time::Instant::now();
            stats.hits += 1;
            Some((entry.key_cache.clone(), entry.value_cache.clone()))
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Store key-value tensors for a sequence
    pub fn put(&self, sequence_id: String, key_cache: Vec<Tensor>, value_cache: Vec<Tensor>) -> InferenceResult<()> {
        let mut cache = self.cache_entries.write();
        let mut stats = self.usage_stats.write();

        // Check if we need to evict entries
        if cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache, &mut stats)?;
        }

        let sequence_length = key_cache.first()
            .map(|t| t.shape().dims[1])
            .unwrap_or(0);

        let entry = CacheEntry {
            key_cache,
            value_cache,
            sequence_length,
            last_accessed: std::time::Instant::now(),
        };

        cache.insert(sequence_id, entry);
        Ok(())
    }

    /// Evict least recently used entry
    fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>, stats: &mut CacheStats) -> InferenceResult<()> {
        let oldest_key = cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            cache.remove(&key);
            stats.evictions += 1;
        }

        Ok(())
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut cache = self.cache_entries.write();
        cache.clear();

        let mut stats = self.usage_stats.write();
        *stats = CacheStats::default();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStatistics {
        let stats = self.usage_stats.read();
        CacheStatistics {
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            hit_rate: if stats.hits + stats.misses > 0 {
                stats.hits as f64 / (stats.hits + stats.misses) as f64
            } else {
                0.0
            },
            memory_usage: stats.memory_usage,
        }
    }
}

impl AttentionCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_attention(&self, key: &str) -> Option<Tensor> {
        let cache = self.cache.read();
        cache.get(key).map(|entry| entry.attention_weights.clone())
    }

    pub fn store_attention(&self, key: String, attention_weights: Tensor) {
        let mut cache = self.cache.write();
        cache.insert(key, AttentionEntry {
            attention_weights,
            timestamp: std::time::Instant::now(),
        });
    }

    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }
}

/// Public cache statistics
#[derive(Debug)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::TensorShape;
    use crate::tensor::{Tensor, DataType};

    #[test]
    fn test_kv_cache_creation() {
        let config = CacheConfig::default();
        let cache = KVCache::new(config);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let config = CacheConfig::default();
        let cache = KVCache::new(config);

        let result = cache.get("non_existent");
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_attention_cache() {
        let cache = AttentionCache::new();
        let tensor = Tensor::new(TensorShape::new(vec![1, 8, 64, 64]), DataType::F32);

        cache.store_attention("test_key".to_string(), tensor.clone());
        let retrieved = cache.get_attention("test_key");

        assert!(retrieved.is_some());
    }
}
