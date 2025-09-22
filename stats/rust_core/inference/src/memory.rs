//! Memory management for high-performance inference
//!
//! This module provides optimized memory allocation and management specifically
//! designed for neural network inference workloads.

use std::sync::Arc;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use bumpalo::Bump;
use crate::error::{InferenceError, InferenceResult};

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Total pool size in bytes
    pub pool_size: usize,
    /// Block size for allocations
    pub block_size: usize,
    /// Number of pre-allocated blocks
    pub num_blocks: usize,
    /// Enable memory mapping
    pub use_mmap: bool,
    /// Arena size for temporary allocations
    pub arena_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: 2usize << 30, // 2GB
            block_size: 64usize << 10, // 64KB blocks
            num_blocks: 1024,
            use_mmap: true,
            arena_size: 512usize << 20, // 512MB arena
        }
    }
}

/// High-performance memory pool for tensor operations
pub struct MemoryPool {
    config: MemoryConfig,
    free_blocks: Mutex<VecDeque<MemoryBlock>>,
    allocated_blocks: RwLock<Vec<MemoryBlock>>,
    arena: Arc<Mutex<TensorArena>>,
    stats: RwLock<PoolStats>,
}

/// Memory block for allocations
#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    id: usize,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Tensor arena for temporary allocations
pub struct TensorArena {
    bump: Bump,
    peak_usage: usize,
    current_usage: usize,
}

/// Memory pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub num_allocations: usize,
    pub num_deallocations: usize,
    pub fragmentation_ratio: f64,
}

/// Tensor shape representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Calculate total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Calculate size in bytes for given data type
    pub fn size_bytes<T>(&self) -> usize {
        self.size() * std::mem::size_of::<T>()
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryConfig) -> InferenceResult<Arc<Self>> {
        let arena = Arc::new(Mutex::new(TensorArena::new(config.arena_size)?));
        let pool = Arc::new(Self {
            config: config.clone(),
            free_blocks: Mutex::new(VecDeque::with_capacity(config.num_blocks)),
            allocated_blocks: RwLock::new(Vec::with_capacity(config.num_blocks)),
            arena,
            stats: RwLock::new(PoolStats::default()),
        });

        // Pre-allocate blocks
        pool.preallocate_blocks()?;
        Ok(pool)
    }

    /// Pre-allocate memory blocks
    fn preallocate_blocks(&self) -> InferenceResult<()> {
        let mut free_blocks = self.free_blocks.lock();

        for i in 0..self.config.num_blocks {
            let layout = Layout::from_size_align(self.config.block_size, 64)
                .map_err(|_| InferenceError::memory("Invalid block layout"))?;

            unsafe {
                let ptr = alloc(layout);
                if ptr.is_null() {
                    return Err(InferenceError::memory("Failed to allocate memory block"));
                }

                let block = MemoryBlock {
                    ptr: NonNull::new_unchecked(ptr),
                    size: self.config.block_size,
                    id: i,
                };
                free_blocks.push_back(block);
            }
        }

        Ok(())
    }

    /// Allocate a tensor with the given shape
    pub fn allocate_tensor<T>(&self, shape: &TensorShape) -> InferenceResult<TensorAllocation<T>>
    where
        T: Copy + Send + Sync + 'static,
    {
        let size_bytes = shape.size_bytes::<T>();
        let ptr = self.allocate_raw(size_bytes)?;

        Ok(TensorAllocation {
            ptr,
            shape: shape.clone(),
            pool: Arc::downgrade(&Arc::new(self.clone())),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Allocate raw memory
    pub fn allocate_raw(&self, size: usize) -> InferenceResult<NonNull<u8>> {
        if size <= self.config.block_size {
            self.allocate_from_pool(size)
        } else {
            self.allocate_large(size)
        }
    }

    /// Allocate from pre-allocated pool
    fn allocate_from_pool(&self, size: usize) -> InferenceResult<NonNull<u8>> {
        let mut free_blocks = self.free_blocks.lock();

        if let Some(block) = free_blocks.pop_front() {
            let mut allocated = self.allocated_blocks.write();
            allocated.push(block.clone());

            let mut stats = self.stats.write();
            stats.total_allocated += size;
            stats.num_allocations += 1;
            stats.peak_allocated = stats.peak_allocated.max(stats.total_allocated);

            Ok(block.ptr)
        } else {
            // Pool exhausted, allocate directly
            self.allocate_large(size)
        }
    }

    /// Allocate large memory block directly
    fn allocate_large(&self, size: usize) -> InferenceResult<NonNull<u8>> {
        let layout = Layout::from_size_align(size, 64)
            .map_err(|_| InferenceError::memory("Invalid allocation layout"))?;

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(InferenceError::memory("Failed to allocate large block"));
            }

            let mut stats = self.stats.write();
            stats.total_allocated += size;
            stats.num_allocations += 1;
            stats.peak_allocated = stats.peak_allocated.max(stats.total_allocated);

            Ok(NonNull::new_unchecked(ptr))
        }
    }

    /// Deallocate memory block
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        // Try to return to pool if it's a standard block
        if size <= self.config.block_size {
            self.return_to_pool(ptr, size);
        } else {
            // Deallocate large block directly
            unsafe {
                let layout = Layout::from_size_align(size, 64).unwrap();
                dealloc(ptr.as_ptr(), layout);
            }
        }

        let mut stats = self.stats.write();
        stats.total_allocated = stats.total_allocated.saturating_sub(size);
        stats.num_deallocations += 1;
    }

    /// Return block to pool
    fn return_to_pool(&self, ptr: NonNull<u8>, size: usize) {
        let mut allocated = self.allocated_blocks.write();

        // Find and remove the block from allocated list
        if let Some(pos) = allocated.iter().position(|block| block.ptr == ptr) {
            let block = allocated.remove(pos);
            let mut free_blocks = self.free_blocks.lock();
            free_blocks.push_back(block);
        } else {
            // Not from pool, deallocate directly
            unsafe {
                let layout = Layout::from_size_align(size, 64).unwrap();
                dealloc(ptr.as_ptr(), layout);
            }
        }
    }

    /// Get memory pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let stats = self.stats.read();
        let mut result = stats.clone();

        // Calculate fragmentation ratio
        let free_blocks = self.free_blocks.lock();
        let allocated_blocks = self.allocated_blocks.read();

        if !allocated_blocks.is_empty() {
            result.fragmentation_ratio = free_blocks.len() as f64 /
                (free_blocks.len() + allocated_blocks.len()) as f64;
        }

        result
    }

    /// Warm up the memory pool
    pub async fn warmup(&self) -> InferenceResult<()> {
        // Allocate and deallocate some tensors to warm up the pool
        let shape = TensorShape::new(vec![1024, 1024]);
        let tensor: TensorAllocation<f32> = self.allocate_tensor(&shape)?;
        drop(tensor);

        Ok(())
    }

    /// Access the tensor arena
    pub fn arena(&self) -> Arc<Mutex<TensorArena>> {
        self.arena.clone()
    }

    /// Reset the arena (clears all temporary allocations)
    pub fn reset_arena(&self) {
        let mut arena = self.arena.lock();
        arena.reset();
    }
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            free_blocks: Mutex::new(VecDeque::new()),
            allocated_blocks: RwLock::new(Vec::new()),
            arena: self.arena.clone(),
            stats: RwLock::new(PoolStats::default()),
        }
    }
}

impl TensorArena {
    /// Create a new tensor arena
    pub fn new(size: usize) -> InferenceResult<Self> {
        Ok(Self {
            bump: Bump::with_capacity(size),
            peak_usage: 0,
            current_usage: 0,
        })
    }

    /// Allocate temporary memory from the arena
    pub fn alloc<T>(&mut self, count: usize) -> &mut [T] {
        let allocation = self.bump.alloc_slice_fill_default(count);
        self.current_usage += count * std::mem::size_of::<T>();
        self.peak_usage = self.peak_usage.max(self.current_usage);
        allocation
    }

    /// Reset the arena (clears all allocations)
    pub fn reset(&mut self) {
        self.bump.reset();
        self.current_usage = 0;
    }

    /// Get current usage statistics
    pub fn usage_stats(&self) -> (usize, usize) {
        (self.current_usage, self.peak_usage)
    }
}

/// RAII wrapper for tensor allocations
pub struct TensorAllocation<T> {
    ptr: NonNull<u8>,
    shape: TensorShape,
    pool: std::sync::Weak<MemoryPool>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TensorAllocation<T>
where
    T: Copy + Send + Sync + 'static,
{
    /// Get the tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get raw pointer to data
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    /// Get mutable raw pointer to data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    /// Get data as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr(), self.shape.size())
        }
    }

    /// Get data as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.shape.size())
        }
    }
}

impl<T> Drop for TensorAllocation<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            pool.deallocate(self.ptr, self.shape.size_bytes::<T>());
        }
    }
}

unsafe impl<T: Send> Send for TensorAllocation<T> {}
unsafe impl<T: Sync> Sync for TensorAllocation<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        let stats = pool.get_stats();
        assert_eq!(stats.num_allocations, 0);
    }

    #[test]
    fn test_tensor_allocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();

        let shape = TensorShape::new(vec![10, 20]);
        let tensor: TensorAllocation<f32> = pool.allocate_tensor(&shape).unwrap();

        assert_eq!(tensor.shape().dims, vec![10, 20]);
        assert_eq!(tensor.as_slice().len(), 200);
    }

    #[tokio::test]
    async fn test_pool_warmup() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config).unwrap();

        pool.warmup().await.unwrap();
        let stats = pool.get_stats();
        assert!(stats.num_allocations > 0);
    }
}
