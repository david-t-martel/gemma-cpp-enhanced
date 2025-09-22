//! Optimized high-performance vector store implementation
//!
//! This module provides significant performance improvements over the standard implementation:
//! - Advanced SIMD optimizations with runtime CPU feature detection
//! - Memory pooling for embedding operations to reduce allocations
//! - Batch processing with parallel computation using Rayon
//! - Zero-copy operations where possible
//! - Optimized data structures for better cache locality
//! - Hardware-accelerated distance calculations

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
    mem::MaybeUninit,
    alloc::{alloc, dealloc, Layout},
    ptr,
};

use parking_lot::{RwLock, Mutex};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use rayon::prelude::*;

#[cfg(feature = "simsimd")]
use simsimd::SpatialSimilarity;

// Re-export for convenience
pub use hnsw::{Hnsw, Searcher};

/// Vector store specific errors
#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    #[error("Invalid distance metric: {metric}")]
    InvalidDistanceMetric { metric: String },

    #[error("Index build failed: {reason}")]
    IndexBuildFailed { reason: String },

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("HNSW error: {0}")]
    HnswError(String),

    #[error("SIMD operation not available on this platform")]
    SimdNotAvailable,

    #[error("Memory pool error: {0}")]
    MemoryPoolError(String),
}

pub type Result<T> = std::result::Result<T, VectorStoreError>;

// Re-export DistanceMetric from config to avoid duplication
pub use crate::config::DistanceMetric;

/// CPU features detected at runtime
#[derive(Debug, Clone)]
struct CpuFeatures {
    has_avx: bool,
    has_avx2: bool,
    has_avx512f: bool,
    has_fma: bool,
    has_sse2: bool,
    has_sse41: bool,
    has_neon: bool, // For ARM
}

impl CpuFeatures {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx: is_x86_feature_detected!("avx"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
                has_sse2: is_x86_feature_detected!("sse2"),
                has_sse41: is_x86_feature_detected!("sse4.1"),
                has_neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx: false,
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
                has_sse2: false,
                has_sse41: false,
                has_neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_avx: false,
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
                has_sse2: false,
                has_sse41: false,
                has_neon: false,
            }
        }
    }

    fn best_simd_width(&self) -> usize {
        if self.has_avx512f {
            512 / 32 // 16 f32s
        } else if self.has_avx2 || self.has_avx {
            256 / 32 // 8 f32s
        } else if self.has_sse2 {
            128 / 32 // 4 f32s
        } else if self.has_neon {
            128 / 32 // 4 f32s
        } else {
            1 // Scalar fallback
        }
    }
}

/// Memory pool for efficient vector allocation and reuse
#[derive(Debug)]
struct VectorMemoryPool {
    /// Pre-allocated vector buffers organized by size
    pools: Vec<Mutex<Vec<Vec<f32>>>>,
    /// Pool statistics
    stats: RwLock<MemoryPoolStats>,
    /// Maximum dimension supported
    max_dimension: usize,
    /// Pool sizes (powers of 2)
    pool_sizes: Vec<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct MemoryPoolStats {
    total_allocations: usize,
    total_deallocations: usize,
    pool_hits: usize,
    pool_misses: usize,
    peak_memory_usage: usize,
    current_memory_usage: usize,
}

impl VectorMemoryPool {
    fn new(max_dimension: usize) -> Self {
        // Create pools for common vector sizes (powers of 2 up to max_dimension)
        let mut pool_sizes = Vec::new();
        let mut size = 64; // Start with 64-dimensional vectors
        while size <= max_dimension {
            pool_sizes.push(size);
            size *= 2;
        }
        if pool_sizes.last() != Some(&max_dimension) {
            pool_sizes.push(max_dimension);
        }

        let pools = pool_sizes.iter()
            .map(|_| Mutex::new(Vec::with_capacity(1000)))
            .collect();

        Self {
            pools,
            stats: RwLock::new(MemoryPoolStats::default()),
            max_dimension,
            pool_sizes,
        }
    }

    fn get_vector(&self, dimension: usize) -> Vec<f32> {
        // Find the appropriate pool
        if let Some((pool_idx, &pool_size)) = self.pool_sizes.iter()
            .enumerate()
            .find(|(_, &size)| size >= dimension) {

            let mut pool = self.pools[pool_idx].lock();
            if let Some(mut vec) = pool.pop() {
                vec.clear();
                vec.resize(dimension, 0.0);

                // Update stats
                let mut stats = self.stats.write();
                stats.pool_hits += 1;
                stats.current_memory_usage -= pool_size * 4; // 4 bytes per f32

                return vec;
            }
        }

        // Pool miss - allocate new vector
        let mut stats = self.stats.write();
        stats.pool_misses += 1;
        stats.total_allocations += 1;
        stats.current_memory_usage += dimension * 4;
        stats.peak_memory_usage = stats.peak_memory_usage.max(stats.current_memory_usage);

        vec![0.0; dimension]
    }

    fn return_vector(&self, mut vec: Vec<f32>) {
        let dimension = vec.capacity();

        // Find the appropriate pool
        if let Some((pool_idx, &pool_size)) = self.pool_sizes.iter()
            .enumerate()
            .find(|(_, &size)| *size >= dimension) {

            // Only keep the vector if the pool isn't too full
            let mut pool = self.pools[pool_idx].lock();
            if pool.len() < pool.capacity() {
                vec.clear();
                pool.push(vec);

                // Update stats
                let mut stats = self.stats.write();
                stats.current_memory_usage += pool_size * 4;
                return;
            }
        }

        // Drop the vector (will be deallocated)
        let mut stats = self.stats.write();
        stats.total_deallocations += 1;
        stats.current_memory_usage -= dimension * 4;
    }

    fn get_stats(&self) -> MemoryPoolStats {
        self.stats.read().clone()
    }

    fn clear(&self) {
        for pool in &self.pools {
            pool.lock().clear();
        }
        *self.stats.write() = MemoryPoolStats::default();
    }
}

/// Vector metadata for filtering and additional information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Document ID this vector belongs to
    pub document_id: String,
    /// Chunk ID within the document
    pub chunk_id: String,
    /// Additional metadata as key-value pairs
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when vector was added
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl VectorMetadata {
    pub fn new(document_id: String, chunk_id: String) -> Self {
        Self {
            document_id,
            chunk_id,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            tags: Vec::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Search filter for metadata-based filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilter {
    /// Filter by document IDs
    pub document_ids: Option<HashSet<String>>,
    /// Filter by tags (must contain all specified tags)
    pub tags: Option<Vec<String>>,
    /// Filter by metadata key-value pairs
    pub metadata_filters: Option<HashMap<String, serde_json::Value>>,
    /// Time range filter
    pub time_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
}

impl SearchFilter {
    pub fn new() -> Self {
        Self {
            document_ids: None,
            tags: None,
            metadata_filters: None,
            time_range: None,
        }
    }

    pub fn with_document_ids(mut self, ids: HashSet<String>) -> Self {
        self.document_ids = Some(ids);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn with_metadata(mut self, filters: HashMap<String, serde_json::Value>) -> Self {
        self.metadata_filters = Some(filters);
        self
    }

    pub fn matches(&self, metadata: &VectorMetadata) -> bool {
        // Check document IDs
        if let Some(ref doc_ids) = self.document_ids {
            if !doc_ids.contains(&metadata.document_id) {
                return false;
            }
        }

        // Check tags (must contain all specified tags)
        if let Some(ref filter_tags) = self.tags {
            for tag in filter_tags {
                if !metadata.tags.contains(tag) {
                    return false;
                }
            }
        }

        // Check metadata filters
        if let Some(ref meta_filters) = self.metadata_filters {
            for (key, value) in meta_filters {
                if let Some(meta_value) = metadata.metadata.get(key) {
                    if meta_value != value {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        // Check time range
        if let Some((start, end)) = self.time_range {
            if metadata.timestamp < start || metadata.timestamp > end {
                return false;
            }
        }

        true
    }
}

impl Default for SearchFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Search result with similarity score and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score (higher is more similar)
    pub score: f32,
    /// Vector metadata
    pub metadata: VectorMetadata,
}

/// Statistics about the vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of vectors in the index
    pub vector_count: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Index dimension
    pub dimension: usize,
    /// Distance metric used
    pub distance_metric: DistanceMetric,
    /// Average search time in microseconds
    pub avg_search_time_us: u64,
    /// Total number of searches performed
    pub total_searches: u64,
    /// Index build time in milliseconds
    pub build_time_ms: u64,
    /// Memory pool statistics
    pub memory_pool_stats: MemoryPoolStats,
    /// SIMD acceleration info
    pub simd_acceleration: String,
}

/// Advanced SIMD-optimized distance calculations with runtime CPU feature detection
struct OptimizedSimdCalculator {
    dimension: usize,
    cpu_features: CpuFeatures,
    simd_width: usize,
    memory_pool: Arc<VectorMemoryPool>,
}

impl OptimizedSimdCalculator {
    fn new(dimension: usize, memory_pool: Arc<VectorMemoryPool>) -> Self {
        let cpu_features = CpuFeatures::detect();
        let simd_width = cpu_features.best_simd_width();

        Self {
            dimension,
            cpu_features,
            simd_width,
            memory_pool,
        }
    }

    /// Calculate cosine similarity using the best available SIMD instructions
    #[inline]
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: a.len().min(b.len()),
            });
        }

        // Try external SIMD library first
        #[cfg(feature = "simsimd")]
        {
            if let Some(similarity) = SpatialSimilarity::cosine(a, b) {
                return Ok(similarity as f32);
            }
        }

        // Use optimized platform-specific implementations
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 && self.dimension >= 8 {
                return unsafe { self.cosine_similarity_avx2(a, b) };
            } else if self.cpu_features.has_sse2 && self.dimension >= 4 {
                return unsafe { self.cosine_similarity_sse2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_features.has_neon && self.dimension >= 4 {
                return unsafe { self.cosine_similarity_neon(a, b) };
            }
        }

        // Fallback to optimized scalar implementation
        self.cosine_similarity_scalar_optimized(a, b)
    }

    /// AVX2-optimized cosine similarity calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn cosine_similarity_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;

            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

            // Dot product accumulation
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);

            // Norm accumulation
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum of AVX2 registers
        let dot_result = self.horizontal_sum_avx2(dot_sum);
        let norm_a_result = self.horizontal_sum_avx2(norm_a_sum);
        let norm_b_result = self.horizontal_sum_avx2(norm_b_sum);

        // Handle remaining elements
        let remainder = len % 8;
        let (mut dot_final, mut norm_a_final, mut norm_b_final) = (dot_result, norm_a_result, norm_b_result);

        if remainder > 0 {
            let start = chunks * 8;
            for i in start..len {
                let ai = a[i];
                let bi = b[i];
                dot_final += ai * bi;
                norm_a_final += ai * ai;
                norm_b_final += bi * bi;
            }
        }

        let norm_a_sqrt = norm_a_final.sqrt();
        let norm_b_sqrt = norm_b_final.sqrt();

        if norm_a_sqrt == 0.0 || norm_b_sqrt == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_final / (norm_a_sqrt * norm_b_sqrt))
        }
    }

    /// SSE2-optimized cosine similarity calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn cosine_similarity_sse2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut dot_sum = _mm_setzero_ps();
        let mut norm_a_sum = _mm_setzero_ps();
        let mut norm_b_sum = _mm_setzero_ps();

        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm_loadu_ps(b.as_ptr().add(offset));

            // Dot product and norm accumulation
            dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
            norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
            norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
        }

        // Horizontal sum of SSE registers
        let dot_result = self.horizontal_sum_sse2(dot_sum);
        let norm_a_result = self.horizontal_sum_sse2(norm_a_sum);
        let norm_b_result = self.horizontal_sum_sse2(norm_b_sum);

        // Handle remaining elements
        let remainder = len % 4;
        let (mut dot_final, mut norm_a_final, mut norm_b_final) = (dot_result, norm_a_result, norm_b_result);

        if remainder > 0 {
            let start = chunks * 4;
            for i in start..len {
                let ai = a[i];
                let bi = b[i];
                dot_final += ai * bi;
                norm_a_final += ai * ai;
                norm_b_final += bi * bi;
            }
        }

        let norm_a_sqrt = norm_a_final.sqrt();
        let norm_b_sqrt = norm_b_final.sqrt();

        if norm_a_sqrt == 0.0 || norm_b_sqrt == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_final / (norm_a_sqrt * norm_b_sqrt))
        }
    }

    /// NEON-optimized cosine similarity calculation for ARM
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn cosine_similarity_neon(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::aarch64::*;

        let len = a.len();
        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));

            // Dot product and norm accumulation
            dot_sum = vmlaq_f32(dot_sum, va, vb);
            norm_a_sum = vmlaq_f32(norm_a_sum, va, va);
            norm_b_sum = vmlaq_f32(norm_b_sum, vb, vb);
        }

        // Horizontal sum of NEON registers
        let dot_result = vaddvq_f32(dot_sum);
        let norm_a_result = vaddvq_f32(norm_a_sum);
        let norm_b_result = vaddvq_f32(norm_b_sum);

        // Handle remaining elements
        let remainder = len % 4;
        let (mut dot_final, mut norm_a_final, mut norm_b_final) = (dot_result, norm_a_result, norm_b_result);

        if remainder > 0 {
            let start = chunks * 4;
            for i in start..len {
                let ai = a[i];
                let bi = b[i];
                dot_final += ai * bi;
                norm_a_final += ai * ai;
                norm_b_final += bi * bi;
            }
        }

        let norm_a_sqrt = norm_a_final.sqrt();
        let norm_b_sqrt = norm_b_final.sqrt();

        if norm_a_sqrt == 0.0 || norm_b_sqrt == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_final / (norm_a_sqrt * norm_b_sqrt))
        }
    }

    /// Optimized scalar cosine similarity calculation with loop unrolling
    fn cosine_similarity_scalar_optimized(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // Unroll loop by 4 for better performance
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;

            // Manual loop unrolling
            let a0 = a[base];
            let b0 = b[base];
            let a1 = a[base + 1];
            let b1 = b[base + 1];
            let a2 = a[base + 2];
            let b2 = b[base + 2];
            let a3 = a[base + 3];
            let b3 = b[base + 3];

            dot_product += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
            norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        }

        // Handle remainder
        let start = chunks * 4;
        for i in start..(start + remainder) {
            let ai = a[i];
            let bi = b[i];
            dot_product += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let norm_a_sqrt = norm_a.sqrt();
        let norm_b_sqrt = norm_b.sqrt();

        if norm_a_sqrt == 0.0 || norm_b_sqrt == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a_sqrt * norm_b_sqrt))
        }
    }

    /// Calculate Euclidean distance using SIMD optimizations
    #[inline]
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: a.len().min(b.len()),
            });
        }

        // Try external SIMD library first
        #[cfg(feature = "simsimd")]
        {
            if let Some(distance) = SpatialSimilarity::sqeuclidean(a, b) {
                return Ok((distance as f32).sqrt());
            }
        }

        // Use optimized platform-specific implementations
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 && self.dimension >= 8 {
                return unsafe { self.euclidean_distance_avx2(a, b) };
            } else if self.cpu_features.has_sse2 && self.dimension >= 4 {
                return unsafe { self.euclidean_distance_sse2(a, b) };
            }
        }

        // Fallback to optimized scalar implementation
        self.euclidean_distance_scalar_optimized(a, b)
    }

    /// AVX2-optimized Euclidean distance calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn euclidean_distance_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut sum = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;

            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm256_sub_ps(va, vb);

            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut result = self.horizontal_sum_avx2(sum);

        // Handle remaining elements
        let remainder = len % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..len {
                let diff = a[i] - b[i];
                result += diff * diff;
            }
        }

        Ok(result.sqrt())
    }

    /// SSE2-optimized Euclidean distance calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn euclidean_distance_sse2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut sum = _mm_setzero_ps();

        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm_sub_ps(va, vb);

            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        let mut result = self.horizontal_sum_sse2(sum);

        // Handle remaining elements
        let remainder = len % 4;
        if remainder > 0 {
            let start = chunks * 4;
            for i in start..len {
                let diff = a[i] - b[i];
                result += diff * diff;
            }
        }

        Ok(result.sqrt())
    }

    /// Optimized scalar Euclidean distance calculation
    fn euclidean_distance_scalar_optimized(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let mut sum_sq = 0.0f32;

        // Unroll loop by 4
        let chunks = len / 4;
        for i in 0..chunks {
            let base = i * 4;

            let d0 = a[base] - b[base];
            let d1 = a[base + 1] - b[base + 1];
            let d2 = a[base + 2] - b[base + 2];
            let d3 = a[base + 3] - b[base + 3];

            sum_sq += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        // Handle remainder
        let start = chunks * 4;
        for i in start..len {
            let diff = a[i] - b[i];
            sum_sq += diff * diff;
        }

        Ok(sum_sq.sqrt())
    }

    /// Calculate dot product using SIMD optimizations
    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: a.len().min(b.len()),
            });
        }

        // Try external SIMD library first
        #[cfg(feature = "simsimd")]
        {
            if let Some(product) = SpatialSimilarity::dot(a, b) {
                return Ok(product as f32);
            }
        }

        // Use optimized platform-specific implementations
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.has_avx2 && self.dimension >= 8 {
                return unsafe { self.dot_product_avx2(a, b) };
            } else if self.cpu_features.has_sse2 && self.dimension >= 4 {
                return unsafe { self.dot_product_sse2(a, b) };
            }
        }

        // Fallback to optimized scalar implementation
        self.dot_product_scalar_optimized(a, b)
    }

    /// AVX2-optimized dot product calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut sum = _mm256_setzero_ps();

        // Process 8 elements at a time
        let chunks = len / 8;
        for i in 0..chunks {
            let offset = i * 8;

            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let mut result = self.horizontal_sum_avx2(sum);

        // Handle remaining elements
        let remainder = len % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..len {
                result += a[i] * b[i];
            }
        }

        Ok(result)
    }

    /// SSE2-optimized dot product calculation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_product_sse2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut sum = _mm_setzero_ps();

        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm_loadu_ps(b.as_ptr().add(offset));

            sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        }

        let mut result = self.horizontal_sum_sse2(sum);

        // Handle remaining elements
        let remainder = len % 4;
        if remainder > 0 {
            let start = chunks * 4;
            for i in start..len {
                result += a[i] * b[i];
            }
        }

        Ok(result)
    }

    /// Optimized scalar dot product calculation
    fn dot_product_scalar_optimized(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let mut product = 0.0f32;

        // Unroll loop by 4
        let chunks = len / 4;
        for i in 0..chunks {
            let base = i * 4;

            product += a[base] * b[base] +
                      a[base + 1] * b[base + 1] +
                      a[base + 2] * b[base + 2] +
                      a[base + 3] * b[base + 3];
        }

        // Handle remainder
        let start = chunks * 4;
        for i in start..len {
            product += a[i] * b[i];
        }

        Ok(product)
    }

    /// Calculate Manhattan distance (L1 norm)
    fn manhattan_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != self.dimension || b.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: a.len().min(b.len()),
            });
        }

        let len = a.len();
        let mut sum = 0.0f32;

        // Unroll loop by 4
        let chunks = len / 4;
        for i in 0..chunks {
            let base = i * 4;

            sum += (a[base] - b[base]).abs() +
                   (a[base + 1] - b[base + 1]).abs() +
                   (a[base + 2] - b[base + 2]).abs() +
                   (a[base + 3] - b[base + 3]).abs();
        }

        // Handle remainder
        let start = chunks * 4;
        for i in start..len {
            sum += (a[i] - b[i]).abs();
        }

        Ok(sum)
    }

    /// Calculate distance using the specified metric
    fn calculate_distance(&self, metric: DistanceMetric, a: &[f32], b: &[f32]) -> Result<f32> {
        match metric {
            DistanceMetric::Cosine => {
                // Convert cosine similarity to distance (1 - similarity)
                let similarity = self.cosine_similarity(a, b)?;
                Ok(1.0 - similarity)
            }
            DistanceMetric::Euclidean => self.euclidean_distance(a, b),
            DistanceMetric::DotProduct => {
                // Convert dot product to distance (negative dot product for sorting)
                let product = self.dot_product(a, b)?;
                Ok(-product)
            }
            DistanceMetric::Manhattan => self.manhattan_distance(a, b),
        }
    }

    /// Batch distance calculation for multiple vectors
    fn calculate_distances_batch(
        &self,
        metric: DistanceMetric,
        query: &[f32],
        vectors: &[(String, &[f32])],
    ) -> Result<Vec<(String, f32)>> {
        // Use Rayon for parallel computation
        vectors.par_iter()
            .map(|(id, vector)| {
                let distance = self.calculate_distance(metric, query, vector)?;
                Ok((id.clone(), distance))
            })
            .collect()
    }

    /// Helper function for horizontal sum of AVX2 register
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn horizontal_sum_avx2(&self, v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;

        // Split into two 128-bit halves and add
        let v_low = _mm256_castps256_ps128(v);
        let v_high = _mm256_extractf128_ps(v, 1);
        let sum_128 = _mm_add_ps(v_low, v_high);

        // Horizontal sum of 128-bit register
        self.horizontal_sum_sse2(sum_128)
    }

    /// Helper function for horizontal sum of SSE2 register
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn horizontal_sum_sse2(&self, v: std::arch::x86_64::__m128) -> f32 {
        use std::arch::x86_64::*;

        let shuf = _mm_movehdup_ps(v);        // [1, 1, 3, 3]
        let sums = _mm_add_ps(v, shuf);       // [0+1, 1+1, 2+3, 3+3]
        let shuf2 = _mm_movehl_ps(shuf, sums); // [2+3, 3+3, ?, ?]
        let sum = _mm_add_ss(sums, shuf2);    // [0+1+2+3, ?, ?, ?]

        _mm_cvtss_f32(sum)
    }

    fn get_simd_acceleration_info(&self) -> String {
        let mut features = Vec::new();

        if self.cpu_features.has_avx512f {
            features.push("AVX512F");
        }
        if self.cpu_features.has_avx2 {
            features.push("AVX2");
        }
        if self.cpu_features.has_avx {
            features.push("AVX");
        }
        if self.cpu_features.has_fma {
            features.push("FMA");
        }
        if self.cpu_features.has_sse41 {
            features.push("SSE4.1");
        }
        if self.cpu_features.has_sse2 {
            features.push("SSE2");
        }
        if self.cpu_features.has_neon {
            features.push("NEON");
        }

        if features.is_empty() {
            "Scalar (no SIMD)".to_string()
        } else {
            format!("SIMD: {} (width: {})", features.join(", "), self.simd_width)
        }
    }
}

/// Optimized vector index with memory pooling and SIMD acceleration
pub struct OptimizedVectorIndex {
    /// Vector storage with metadata
    vectors: DashMap<String, (Vec<f32>, VectorMetadata)>,
    /// Configuration
    config: crate::config::VectorStoreConfig,
    /// Optimized distance calculator
    distance_calculator: Arc<OptimizedSimdCalculator>,
    /// Memory pool for efficient allocation
    memory_pool: Arc<VectorMemoryPool>,
    /// Statistics
    stats: RwLock<IndexStats>,
    /// Vector counter
    vector_counter: AtomicUsize,
}

impl OptimizedVectorIndex {
    /// Create a new optimized vector index
    pub fn new(config: crate::config::VectorStoreConfig) -> Result<Self> {
        let memory_pool = Arc::new(VectorMemoryPool::new(config.dimension * 2)); // Allow for some overhead
        let distance_calculator = Arc::new(OptimizedSimdCalculator::new(config.dimension, memory_pool.clone()));

        let stats = IndexStats {
            vector_count: 0,
            memory_usage: 0,
            dimension: config.dimension,
            distance_metric: config.distance_metric,
            avg_search_time_us: 0,
            total_searches: 0,
            build_time_ms: 0,
            memory_pool_stats: memory_pool.get_stats(),
            simd_acceleration: distance_calculator.get_simd_acceleration_info(),
        };

        Ok(Self {
            vectors: DashMap::new(),
            distance_calculator,
            memory_pool,
            config,
            stats: RwLock::new(stats),
            vector_counter: AtomicUsize::new(0),
        })
    }

    /// Add a single vector to the index with memory pooling
    pub fn add_vector(&self, id: &str, vector: &[f32], metadata: VectorMetadata) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        // Use memory pool for efficient allocation
        let mut pooled_vector = self.memory_pool.get_vector(self.config.dimension);
        pooled_vector.copy_from_slice(vector);

        self.vector_counter.fetch_add(1, Ordering::Relaxed);

        // Store vector and metadata
        self.vectors.insert(id.to_string(), (pooled_vector, metadata));

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_count = self.vectors.len();
            stats.memory_usage = self.estimate_memory_usage();
            stats.memory_pool_stats = self.memory_pool.get_stats();
        }

        Ok(())
    }

    /// Batch add multiple vectors with parallel processing
    pub fn add_vectors_batch(&self, vectors: Vec<(String, Vec<f32>, VectorMetadata)>) -> Result<()> {
        // Validate all vectors first
        for (_, vector, _) in &vectors {
            if vector.len() != self.config.dimension {
                return Err(VectorStoreError::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: vector.len(),
                });
            }
        }

        // Process vectors in parallel
        let processed_vectors: Result<Vec<_>> = vectors.into_par_iter()
            .map(|(id, vector, metadata)| {
                let mut pooled_vector = self.memory_pool.get_vector(self.config.dimension);
                pooled_vector.copy_from_slice(&vector);
                Ok((id, pooled_vector, metadata))
            })
            .collect();

        let processed = processed_vectors?;

        // Add to storage
        for (id, pooled_vector, metadata) in processed {
            self.vectors.insert(id, (pooled_vector, metadata));
            self.vector_counter.fetch_add(1, Ordering::Relaxed);
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_count = self.vectors.len();
            stats.memory_usage = self.estimate_memory_usage();
            stats.memory_pool_stats = self.memory_pool.get_stats();
        }

        Ok(())
    }

    /// Optimized search with parallel distance calculations
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&SearchFilter>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let start_time = Instant::now();

        // Collect vectors for parallel processing
        let vectors_to_search: Vec<_> = self.vectors.iter()
            .filter_map(|entry| {
                let (stored_vector, metadata) = entry.value();

                // Apply filter if provided
                if let Some(f) = filter {
                    if !f.matches(metadata) {
                        return None;
                    }
                }

                Some((entry.key().clone(), stored_vector.as_slice()))
            })
            .collect();

        // Parallel distance calculation using batch processing
        let distance_results = self.distance_calculator
            .calculate_distances_batch(self.config.distance_metric, query, &vectors_to_search)?;

        // Convert distances to similarity scores and collect results
        let mut search_results: Vec<SearchResult> = distance_results.into_par_iter()
            .filter_map(|(id, distance)| {
                if let Some(entry) = self.vectors.get(&id) {
                    let (_, metadata) = entry.value();
                    Some(SearchResult {
                        id,
                        score: self.distance_to_similarity(distance),
                        metadata: metadata.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score (higher is better) and limit results
        search_results.par_sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        search_results.truncate(k);

        // Update search stats
        {
            let mut stats = self.stats.write();
            stats.total_searches += 1;
            let elapsed_us = start_time.elapsed().as_micros() as u64;
            stats.avg_search_time_us =
                (stats.avg_search_time_us * (stats.total_searches - 1) + elapsed_us)
                / stats.total_searches;
        }

        Ok(search_results)
    }

    /// Remove a vector from the index with memory pool cleanup
    pub fn remove_vector(&self, id: &str) -> Result<()> {
        if let Some((_, (vector, _))) = self.vectors.remove(id) {
            // Return vector to memory pool
            self.memory_pool.return_vector(vector);

            // Update stats
            {
                let mut stats = self.stats.write();
                stats.vector_count = self.vectors.len();
                stats.memory_usage = self.estimate_memory_usage();
                stats.memory_pool_stats = self.memory_pool.get_stats();
            }

            Ok(())
        } else {
            Err(VectorStoreError::VectorNotFound { id: id.to_string() })
        }
    }

    /// Batch remove multiple vectors
    pub fn remove_vectors_batch(&self, ids: &[String]) -> Result<usize> {
        let mut removed_count = 0;
        let mut vectors_to_return = Vec::new();

        for id in ids {
            if let Some((_, (vector, _))) = self.vectors.remove(id) {
                vectors_to_return.push(vector);
                removed_count += 1;
            }
        }

        // Return vectors to memory pool
        for vector in vectors_to_return {
            self.memory_pool.return_vector(vector);
        }

        // Update stats if any vectors were removed
        if removed_count > 0 {
            let mut stats = self.stats.write();
            stats.vector_count = self.vectors.len();
            stats.memory_usage = self.estimate_memory_usage();
            stats.memory_pool_stats = self.memory_pool.get_stats();
        }

        Ok(removed_count)
    }

    /// Get vector by ID (zero-copy when possible)
    pub fn get_vector(&self, id: &str) -> Option<(Vec<f32>, VectorMetadata)> {
        self.vectors.get(id).map(|entry| entry.value().clone())
    }

    /// Get all vector IDs
    pub fn get_vector_ids(&self) -> Vec<String> {
        self.vectors.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get index statistics
    pub fn get_stats(&self) -> IndexStats {
        let mut stats = self.stats.read().clone();
        stats.memory_pool_stats = self.memory_pool.get_stats();
        stats
    }

    /// Clear all vectors from the index and reset memory pool
    pub fn clear(&self) -> Result<()> {
        // Collect all vectors to return to pool
        let vectors_to_return: Vec<Vec<f32>> = self.vectors.iter()
            .map(|entry| entry.value().0.clone())
            .collect();

        // Clear the storage
        self.vectors.clear();
        self.vector_counter.store(0, Ordering::Relaxed);

        // Return vectors to pool
        for vector in vectors_to_return {
            self.memory_pool.return_vector(vector);
        }

        // Reset stats
        {
            let mut stats = self.stats.write();
            stats.vector_count = 0;
            stats.memory_usage = 0;
            stats.total_searches = 0;
            stats.avg_search_time_us = 0;
            stats.memory_pool_stats = self.memory_pool.get_stats();
        }

        Ok(())
    }

    /// Convert distance to similarity score
    fn distance_to_similarity(&self, distance: f32) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::Cosine => 1.0 - distance, // Cosine distance to similarity
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance), // Euclidean to similarity
            DistanceMetric::DotProduct => -distance, // Negative distance (higher dot product = higher similarity)
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance), // Manhattan to similarity
        }
    }

    /// Estimate memory usage of the index
    fn estimate_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len() * (self.config.dimension * 4 + 256); // 4 bytes per f32 + metadata estimate
        let pool_memory = self.memory_pool.get_stats().current_memory_usage;
        vector_memory + pool_memory
    }
}

/// Main optimized vector store implementation
pub struct OptimizedVectorStore {
    /// Primary vector index
    index: Arc<OptimizedVectorIndex>,
    /// Configuration
    config: crate::config::VectorStoreConfig,
}

impl OptimizedVectorStore {
    /// Create a new optimized vector store
    pub fn new(config: crate::config::VectorStoreConfig) -> crate::error::Result<Self> {
        let index = Arc::new(OptimizedVectorIndex::new(config.clone())
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))?);

        Ok(Self {
            index,
            config,
        })
    }

    /// Add a vector to the store
    pub fn add_vector(&self, id: &str, vector: &[f32], metadata: serde_json::Value) -> crate::error::Result<()> {
        let vector_metadata = VectorMetadata {
            document_id: metadata.get("document_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            chunk_id: metadata.get("chunk_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            metadata: metadata.as_object()
                .map(|obj| obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect())
                .unwrap_or_default(),
            timestamp: chrono::Utc::now(),
            tags: metadata.get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default(),
        };

        self.index.add_vector(id, vector, vector_metadata)
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))
    }

    /// Batch add multiple vectors
    pub fn add_vectors_batch(&self, vectors: Vec<(String, Vec<f32>, serde_json::Value)>) -> crate::error::Result<()> {
        let processed_vectors: Vec<_> = vectors.into_iter()
            .map(|(id, vector, metadata)| {
                let vector_metadata = VectorMetadata {
                    document_id: metadata.get("document_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    chunk_id: metadata.get("chunk_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    metadata: metadata.as_object()
                        .map(|obj| obj.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect())
                        .unwrap_or_default(),
                    timestamp: chrono::Utc::now(),
                    tags: metadata.get("tags")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect())
                        .unwrap_or_default(),
                };
                (id, vector, vector_metadata)
            })
            .collect();

        self.index.add_vectors_batch(processed_vectors)
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&SearchFilter>,
    ) -> crate::error::Result<Vec<(String, f32, serde_json::Value)>> {
        let results = self.index.search(query, k, filter)
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))?;

        Ok(results.into_iter().map(|result| {
            let metadata = serde_json::json!({
                "document_id": result.metadata.document_id,
                "chunk_id": result.metadata.chunk_id,
                "timestamp": result.metadata.timestamp,
                "tags": result.metadata.tags,
                "metadata": result.metadata.metadata,
            });

            (result.id, result.score, metadata)
        }).collect())
    }

    /// Remove a vector from the store
    pub fn remove_vector(&self, id: &str) -> crate::error::Result<()> {
        self.index.remove_vector(id)
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))
    }

    /// Batch remove multiple vectors
    pub fn remove_vectors_batch(&self, ids: &[String]) -> crate::error::Result<usize> {
        self.index.remove_vectors_batch(ids)
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))
    }

    /// Get vector by ID
    pub fn get_vector(&self, id: &str) -> Option<(Vec<f32>, serde_json::Value)> {
        self.index.get_vector(id).map(|(vector, metadata)| {
            let json_metadata = serde_json::json!({
                "document_id": metadata.document_id,
                "chunk_id": metadata.chunk_id,
                "timestamp": metadata.timestamp,
                "tags": metadata.tags,
                "metadata": metadata.metadata,
            });
            (vector, json_metadata)
        })
    }

    /// Get all vector IDs
    pub fn get_vector_ids(&self) -> Vec<String> {
        self.index.get_vector_ids()
    }

    /// Get store statistics
    pub fn get_stats(&self) -> IndexStats {
        self.index.get_stats()
    }

    /// Clear all vectors
    pub fn clear(&self) -> crate::error::Result<()> {
        self.index.clear()
            .map_err(|e| crate::error::Error::VectorStore(e.to_string()))
    }

    /// Get the store configuration
    pub fn config(&self) -> &crate::config::VectorStoreConfig {
        &self.config
    }
}

// Thread safety implementations
unsafe impl Send for OptimizedVectorStore {}
unsafe impl Sync for OptimizedVectorStore {}
unsafe impl Send for OptimizedVectorIndex {}
unsafe impl Sync for OptimizedVectorIndex {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_config() -> crate::config::VectorStoreConfig {
        crate::config::VectorStoreConfig {
            dimension: 128, // Use larger dimension to test SIMD
            distance_metric: crate::config::DistanceMetric::Cosine,
            index_type: crate::config::IndexType::Flat,
            hnsw_m: 16,
            hnsw_ef_construction: 50,
            hnsw_ef_search: 20,
            max_vectors: Some(1000),
        }
    }

    #[test]
    fn test_optimized_vector_store_creation() {
        let config = create_test_config();
        let store = OptimizedVectorStore::new(config).unwrap();
        assert_eq!(store.config.dimension, 128);

        let stats = store.get_stats();
        assert!(stats.simd_acceleration.len() > 0);
    }

    #[test]
    fn test_memory_pool_functionality() {
        let pool = VectorMemoryPool::new(256);

        // Get vectors from pool
        let vec1 = pool.get_vector(128);
        let vec2 = pool.get_vector(128);

        assert_eq!(vec1.len(), 128);
        assert_eq!(vec2.len(), 128);

        // Return vectors to pool
        pool.return_vector(vec1);
        pool.return_vector(vec2);

        let stats = pool.get_stats();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_simd_distance_calculations() {
        let pool = Arc::new(VectorMemoryPool::new(128));
        let calculator = OptimizedSimdCalculator::new(4, pool);

        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let cosine_sim = calculator.cosine_similarity(&a, &b).unwrap();
        assert_eq!(cosine_sim, 0.0); // Orthogonal vectors

        let euclidean_dist = calculator.euclidean_distance(&a, &b).unwrap();
        assert!((euclidean_dist - std::f32::consts::SQRT_2).abs() < 1e-6);

        let dot_product = calculator.dot_product(&a, &b).unwrap();
        assert_eq!(dot_product, 0.0); // Orthogonal vectors
    }

    #[test]
    fn test_batch_operations() {
        let config = create_test_config();
        let store = OptimizedVectorStore::new(config).unwrap();

        // Create test vectors
        let mut vectors = Vec::new();
        for i in 0..100 {
            let mut vector = vec![0.0; 128];
            vector[i % 128] = 1.0; // One-hot encoding
            let metadata = serde_json::json!({
                "document_id": format!("doc_{}", i / 10),
                "chunk_id": format!("chunk_{}", i),
                "index": i
            });
            vectors.push((format!("vec_{}", i), vector, metadata));
        }

        // Batch add
        store.add_vectors_batch(vectors).unwrap();

        let stats = store.get_stats();
        assert_eq!(stats.vector_count, 100);

        // Test search
        let query = vec![1.0; 128];
        let results = store.search(&query, 10, None).unwrap();
        assert!(results.len() <= 10);

        // Batch remove
        let ids_to_remove: Vec<String> = (0..50).map(|i| format!("vec_{}", i)).collect();
        let removed_count = store.remove_vectors_batch(&ids_to_remove).unwrap();
        assert_eq!(removed_count, 50);

        let final_stats = store.get_stats();
        assert_eq!(final_stats.vector_count, 50);
    }

    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();
        let simd_width = features.best_simd_width();

        // Should detect at least some features on modern CPUs
        assert!(simd_width >= 1);

        // Print detected features for debugging
        println!("Detected CPU features: {:?}", features);
        println!("Best SIMD width: {}", simd_width);
    }

    #[test]
    fn test_parallel_search() {
        let config = create_test_config();
        let store = OptimizedVectorStore::new(config).unwrap();

        // Add many vectors for parallel processing test
        let mut vectors = Vec::new();
        for i in 0..1000 {
            let mut vector = vec![0.0; 128];
            // Create some variation in vectors
            for j in 0..128 {
                vector[j] = (i * j) as f32 / 1000.0;
            }
            let metadata = serde_json::json!({
                "document_id": format!("doc_{}", i),
                "chunk_id": format!("chunk_{}", i)
            });
            vectors.push((format!("vec_{}", i), vector, metadata));
        }

        store.add_vectors_batch(vectors).unwrap();

        // Perform search - should use parallel processing internally
        let query = vec![0.5; 128];
        let start = std::time::Instant::now();
        let results = store.search(&query, 50, None).unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 50);

        // Should be reasonably fast with optimizations
        println!("Search took: {:?}", duration);

        let stats = store.get_stats();
        println!("SIMD acceleration: {}", stats.simd_acceleration);
    }
}
