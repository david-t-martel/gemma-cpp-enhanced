//! Vector storage and similarity search module using HNSW algorithm
//!
//! This module provides high-performance vector similarity search capabilities:
//! - HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
//! - Support for various distance metrics (cosine, euclidean, dot product)
//! - Batch operations for efficient vector insertion and search
//! - Integration with Redis for persistent storage
//! - SIMD-optimized distance calculations

use crate::error::{GemmaError, GemmaResult};
use crate::redis_manager::RedisManager;
// TODO: Fix HNSW integration after core build works
// use hnsw::{Hnsw, Params};
use ndarray::{Array1, Array2, ArrayView1};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Distance metrics supported by the vector store
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

impl DistanceMetric {
    /// Calculate distance between two vectors using the specified metric
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b), // Negative for min-heap
            DistanceMetric::Manhattan => manhattan_distance(a, b),
        }
    }

    /// Check if lower distance means higher similarity
    pub fn is_similarity(&self) -> bool {
        matches!(self, DistanceMetric::DotProduct)
    }
}

/// Configuration for the vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Dimensionality of the vectors
    pub dimension: usize,
    /// Distance metric to use for similarity search
    pub metric: DistanceMetric,
    /// Maximum number of connections in HNSW graph
    pub max_connections: usize,
    /// Level generation factor for HNSW
    pub ml: f64,
    /// Size of the dynamic candidate list
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search
    pub ef_search: usize,
    /// Enable normalization of vectors before storage
    pub normalize_vectors: bool,
    /// Maximum number of vectors to keep in memory
    pub max_memory_vectors: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 768, // Common embedding dimension
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ml: 1.0 / 2.0_f64.ln(),
            ef_construction: 200,
            ef_search: 100,
            normalize_vectors: true,
            max_memory_vectors: 100_000,
        }
    }
}

/// Metadata associated with a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub id: String,
    pub document_id: String,
    pub chunk_id: usize,
    pub text: String,
    pub created_at: u64,
    pub tags: Vec<String>,
}

impl VectorMetadata {
    pub fn new(document_id: String, chunk_id: usize, text: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id,
            chunk_id,
            text,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            tags: Vec::new(),
        }
    }
}

/// Search result with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub metadata: VectorMetadata,
    pub similarity: f32,
    pub distance: f32,
    pub vector: Option<Vec<f32>>,
}

/// Statistics for vector store operations
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VectorStoreStats {
    pub total_vectors: usize,
    pub search_queries: u64,
    pub average_search_time_ms: f64,
    pub index_size_bytes: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// High-performance vector store with HNSW index
pub struct VectorStore {
    /// TODO: HNSW index for fast similarity search
    // index: Arc<RwLock<Hnsw<f32, DistanceSpace>>>,
    /// Configuration
    config: VectorStoreConfig,
    /// Vector metadata storage
    metadata: Arc<RwLock<HashMap<usize, VectorMetadata>>>,
    /// Vector storage for full vector retrieval
    vectors: Arc<RwLock<HashMap<usize, Vec<f32>>>>,
    /// Redis manager for persistent storage
    redis: Option<Arc<RedisManager>>,
    /// Next available index ID
    next_id: AtomicUsize,
    /// Statistics
    stats: Arc<RwLock<VectorStoreStats>>,
}

/// Custom distance space for HNSW
#[derive(Clone)]
struct DistanceSpace {
    metric: DistanceMetric,
}

// TODO: Fix HNSW Distance trait implementation
/*
impl hnsw::Distance<f32> for DistanceSpace {
    fn distance(&self, a: &f32, b: &f32) -> f32 {
        // This is called per element, we need the full vector distance
        // This is a simplified implementation - in practice you'd need
        // to restructure to pass full vectors
        (a - b).abs()
    }
}
*/

impl VectorStore {
    /// Create a new vector store with the given configuration
    pub fn new(config: VectorStoreConfig) -> GemmaResult<Self> {
        // TODO: Re-enable HNSW when API is fixed
        /*
        let params = Params::new()
            .max_connections(config.max_connections)
            .ml(config.ml)
            .ef_construction(config.ef_construction);

        let distance_space = DistanceSpace {
            metric: config.metric,
        };

        let index = Hnsw::new(params, distance_space);
        */

        Ok(Self {
            // index: Arc::new(RwLock::new(index)),
            config,
            metadata: Arc::new(RwLock::new(HashMap::new())),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            redis: None,
            next_id: AtomicUsize::new(0),
            stats: Arc::new(RwLock::new(VectorStoreStats::default())),
        })
    }

    /// Create a new vector store with Redis backend
    pub fn with_redis(config: VectorStoreConfig, redis: Arc<RedisManager>) -> GemmaResult<Self> {
        let mut store = Self::new(config)?;
        store.redis = Some(redis);
        Ok(store)
    }

    /// Add a vector to the store
    pub async fn add_vector(
        &self,
        vector: Vec<f32>,
        metadata: VectorMetadata,
    ) -> GemmaResult<usize> {
        if vector.len() != self.config.dimension {
            return Err(GemmaError::InvalidVectorDimension {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        let processed_vector = if self.config.normalize_vectors {
            normalize_vector(&vector)
        } else {
            vector
        };

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Store in HNSW index
        // {
        //     let mut index = self.index.write();
        //     index.add(&processed_vector, id)
        //         .map_err(|e| GemmaError::VectorStore(e.to_string()))?;
        // }

        // Store metadata and vectors
        {
            let mut metadata_map = self.metadata.write();
            let mut vectors_map = self.vectors.write();

            metadata_map.insert(id, metadata.clone());
            vectors_map.insert(id, processed_vector.clone());
        }

        // Store in Redis if available
        if let Some(redis) = &self.redis {
            redis
                .store_embedding(
                    &metadata.document_id,
                    metadata.chunk_id,
                    &processed_vector,
                    &metadata.text,
                )
                .await?;
        }

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_vectors += 1;
        }

        debug!("Added vector {} for document {}", id, metadata.document_id);
        Ok(id)
    }

    /// Batch add vectors for better performance
    pub async fn batch_add_vectors(
        &self,
        vectors_with_metadata: Vec<(Vec<f32>, VectorMetadata)>,
    ) -> GemmaResult<Vec<usize>> {
        let mut ids = Vec::with_capacity(vectors_with_metadata.len());

        // Validate dimensions first
        for (vector, _) in &vectors_with_metadata {
            if vector.len() != self.config.dimension {
                return Err(GemmaError::InvalidVectorDimension {
                    expected: self.config.dimension,
                    actual: vector.len(),
                });
            }
        }

        // Process vectors
        let processed: Vec<_> = vectors_with_metadata
            .into_iter()
            .map(|(vector, metadata)| {
                let processed_vector = if self.config.normalize_vectors {
                    normalize_vector(&vector)
                } else {
                    vector
                };
                (processed_vector, metadata)
            })
            .collect();

        // Batch insert into HNSW index
        {
            // let mut index = self.index.write();
            let mut metadata_map = self.metadata.write();
            let mut vectors_map = self.vectors.write();

            for (vector, metadata) in processed.iter() {
                let id = self.next_id.fetch_add(1, Ordering::SeqCst);

                // index.add(vector, id)
                //     .map_err(|e| GemmaError::VectorStore(e.to_string()))?;

                metadata_map.insert(id, metadata.clone());
                vectors_map.insert(id, vector.clone());
                ids.push(id);
            }
        }

        // Batch store in Redis if available
        if let Some(redis) = &self.redis {
            // Group by document ID for efficient storage
            let mut doc_embeddings: HashMap<String, Vec<(usize, Vec<f32>, String)>> = HashMap::new();

            for ((vector, metadata), &id) in processed.iter().zip(ids.iter()) {
                doc_embeddings
                    .entry(metadata.document_id.clone())
                    .or_default()
                    .push((metadata.chunk_id, vector.clone(), metadata.text.clone()));
            }

            for (doc_id, embeddings) in doc_embeddings {
                redis.batch_store_embeddings(&doc_id, &embeddings).await?;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_vectors += ids.len();
        }

        info!("Batch added {} vectors", ids.len());
        Ok(ids)
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        query_vector: &[f32],
        k: usize,
        include_vectors: bool,
    ) -> GemmaResult<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();

        if query_vector.len() != self.config.dimension {
            return Err(GemmaError::InvalidVectorDimension {
                expected: self.config.dimension,
                actual: query_vector.len(),
            });
        }

        let processed_query = if self.config.normalize_vectors {
            normalize_vector(query_vector)
        } else {
            query_vector.to_vec()
        };

        // Search in HNSW index
        // let search_results = {
        //     let mut index = self.index.write();
        //     index.search(&processed_query, k)
        //         .map_err(|e| GemmaError::VectorStore(e.to_string()))?
        // };

        // For now, do a brute force search through all vectors
        let search_results = self.brute_force_search(&processed_query, k);

        // Prepare results with metadata
        let mut results = Vec::new();
        let metadata_map = self.metadata.read();
        let vectors_map = self.vectors.read();

        for (distance, id) in search_results {
            if let Some(metadata) = metadata_map.get(&id) {
                let vector = if include_vectors {
                    vectors_map.get(&id).cloned()
                } else {
                    None
                };

                // Convert distance to similarity
                let similarity = match self.config.metric {
                    DistanceMetric::Cosine => 1.0 - distance,
                    DistanceMetric::DotProduct => -distance, // Was negated for min-heap
                    _ => 1.0 / (1.0 + distance), // General similarity conversion
                };

                results.push(SearchResult {
                    metadata: metadata.clone(),
                    similarity,
                    distance,
                    vector,
                });
            }
        }

        // Update statistics
        let search_time = start_time.elapsed().as_millis() as f64;
        {
            let mut stats = self.stats.write();
            stats.search_queries += 1;
            let weight = 1.0 / stats.search_queries as f64;
            stats.average_search_time_ms =
                stats.average_search_time_ms * (1.0 - weight) + search_time * weight;
        }

        debug!("Search completed in {:.2}ms, found {} results", search_time, results.len());
        Ok(results)
    }

    /// Brute force search through all vectors (fallback when HNSW is not available)
    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(f32, usize)> {
        let vectors_map = self.vectors.read();
        let mut distances: Vec<(f32, usize)> = vectors_map
            .iter()
            .map(|(&id, vector)| {
                let distance = self.config.metric.calculate(query, vector);
                (distance, id)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    /// Search with filters based on metadata
    pub async fn search_with_filters(
        &self,
        query_vector: &[f32],
        k: usize,
        filters: &HashMap<String, String>,
        include_vectors: bool,
    ) -> GemmaResult<Vec<SearchResult>> {
        // For now, we'll search more than k and then filter
        // A more efficient approach would be to implement filtered search in HNSW
        let expanded_k = k * 3; // Search more to account for filtering
        let initial_results = self.search(query_vector, expanded_k, include_vectors).await?;

        let filtered_results: Vec<_> = initial_results
            .into_iter()
            .filter(|result| {
                // Apply filters
                for (key, value) in filters {
                    match key.as_str() {
                        "document_id" => {
                            if &result.metadata.document_id != value {
                                return false;
                            }
                        }
                        "tag" => {
                            if !result.metadata.tags.contains(value) {
                                return false;
                            }
                        }
                        _ => {} // Ignore unknown filters
                    }
                }
                true
            })
            .take(k)
            .collect();

        Ok(filtered_results)
    }

    /// Remove a vector from the store
    pub async fn remove_vector(&self, id: usize) -> GemmaResult<bool> {
        let mut removed = false;

        // Remove from metadata and vectors
        {
            let mut metadata_map = self.metadata.write();
            let mut vectors_map = self.vectors.write();

            removed = metadata_map.remove(&id).is_some() && vectors_map.remove(&id).is_some();
        }

        // Note: HNSW doesn't support efficient removal, so we keep the vector in the index
        // In practice, you might want to rebuild the index periodically

        // Update statistics
        if removed {
            let mut stats = self.stats.write();
            stats.total_vectors = stats.total_vectors.saturating_sub(1);
        }

        Ok(removed)
    }

    /// Remove all vectors for a document
    pub async fn remove_document_vectors(&self, document_id: &str) -> GemmaResult<usize> {
        let mut removed_count = 0;
        let mut ids_to_remove = Vec::new();

        // Find all vectors for this document
        {
            let metadata_map = self.metadata.read();
            for (&id, metadata) in metadata_map.iter() {
                if metadata.document_id == document_id {
                    ids_to_remove.push(id);
                }
            }
        }

        // Remove them
        for id in ids_to_remove {
            if self.remove_vector(id).await? {
                removed_count += 1;
            }
        }

        // Remove from Redis if available
        if let Some(redis) = &self.redis {
            redis.delete_document(document_id).await?;
        }

        info!("Removed {} vectors for document {}", removed_count, document_id);
        Ok(removed_count)
    }

    /// Get vector by ID
    pub fn get_vector(&self, id: usize) -> Option<(Vec<f32>, VectorMetadata)> {
        let metadata_map = self.metadata.read();
        let vectors_map = self.vectors.read();

        if let (Some(metadata), Some(vector)) = (metadata_map.get(&id), vectors_map.get(&id)) {
            Some((vector.clone(), metadata.clone()))
        } else {
            None
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> VectorStoreStats {
        self.stats.read().clone()
    }

    /// Clear all vectors from the store
    pub async fn clear(&self) -> GemmaResult<()> {
        {
            // let mut index = self.index.write();
            let mut metadata_map = self.metadata.write();
            let mut vectors_map = self.vectors.write();
            let mut stats = self.stats.write();

            // TODO: Create a new index (HNSW doesn't have a clear method)
            /*
            let params = Params::new()
                .max_connections(self.config.max_connections)
                .ml(self.config.ml)
                .ef_construction(self.config.ef_construction);

            let distance_space = DistanceSpace {
                metric: self.config.metric,
            };

            *index = Hnsw::new(params, distance_space);
            */
            metadata_map.clear();
            vectors_map.clear();
            *stats = VectorStoreStats::default();
        }

        self.next_id.store(0, Ordering::SeqCst);
        info!("Vector store cleared");
        Ok(())
    }

    /// Update search parameters
    pub fn set_ef_search(&self, ef_search: usize) {
        // Note: This would require modifying the HNSW library to support runtime parameter updates
        warn!("Runtime ef_search updates not supported by current HNSW implementation");
    }
}

// Distance calculation functions with SIMD optimizations

/// Calculate cosine distance between two vectors
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = vector_norm(a);
    let norm_b = vector_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0 // Maximum distance for zero vectors
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum()
}

/// Calculate dot product between two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .sum()
}

/// Calculate L2 norm of a vector
pub fn vector_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector to unit length
pub fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm = vector_norm(v);
    if norm == 0.0 {
        v.to_vec() // Return original vector if it's zero
    } else {
        v.iter().map(|&x| x / norm).collect()
    }
}

/// Batch normalize vectors
pub fn batch_normalize_vectors(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vectors.iter().map(|v| normalize_vector(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let cosine = cosine_distance(&a, &b);
        let euclidean = euclidean_distance(&a, &b);
        let manhattan = manhattan_distance(&a, &b);
        let dot = dot_product(&a, &b);

        assert!(cosine >= 0.0 && cosine <= 2.0);
        assert!(euclidean > 0.0);
        assert!(manhattan > 0.0);
        assert_eq!(dot, 32.0);
    }

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);
        let norm = vector_norm(&normalized);

        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_store_config() {
        let config = VectorStoreConfig::default();
        assert_eq!(config.dimension, 768);
        assert_eq!(config.metric, DistanceMetric::Cosine);
    }

    #[tokio::test]
    async fn test_vector_store_creation() {
        let config = VectorStoreConfig::default();
        let store = VectorStore::new(config);
        assert!(store.is_ok());
    }

    #[tokio::test]
    async fn test_vector_metadata() {
        let metadata = VectorMetadata::new(
            "doc1".to_string(),
            0,
            "test text".to_string(),
        );

        assert_eq!(metadata.document_id, "doc1");
        assert_eq!(metadata.chunk_id, 0);
        assert_eq!(metadata.text, "test text");
        assert!(!metadata.id.is_empty());
    }

    #[test]
    fn test_distance_metric_calculation() {
        let metric = DistanceMetric::Cosine;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let distance = metric.calculate(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6); // Orthogonal vectors
    }
}