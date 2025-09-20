//! Memory pool implementation for efficient vector allocation
//!
//! This module provides object pooling for vectors and other frequently allocated objects
//! to reduce allocation overhead and memory fragmentation.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::mem;
use std::sync::Arc;

/// A pool for reusing vector allocations
pub struct VectorPool<T> {
    pools: Arc<Vec<Mutex<VecDeque<Vec<T>>>>>,
    max_pool_size: usize,
    dimension: usize,
}

impl<T: Default + Clone> VectorPool<T> {
    /// Create a new vector pool with specified dimension
    pub fn new(dimension: usize, max_pool_size: usize) -> Self {
        // Create multiple pools to reduce contention
        let num_pools = num_cpus::get();
        let mut pools = Vec::with_capacity(num_pools);

        for _ in 0..num_pools {
            pools.push(Mutex::new(VecDeque::with_capacity(
                max_pool_size / num_pools,
            )));
        }

        Self {
            pools: Arc::new(pools),
            max_pool_size,
            dimension,
        }
    }

    /// Get a vector from the pool or allocate a new one
    pub fn acquire(&self) -> PooledVector<T> {
        // Select pool based on thread ID to reduce contention
        let pool_idx = thread_id::get() % self.pools.len();
        let mut pool = self.pools[pool_idx].lock();

        let mut vec = if let Some(mut v) = pool.pop_front() {
            v.clear();
            v.resize(self.dimension, T::default());
            v
        } else {
            vec![T::default(); self.dimension]
        };

        // Ensure capacity is exactly what we need
        vec.shrink_to_fit();

        PooledVector {
            vec: Some(vec),
            pool: Arc::clone(&self.pools),
            pool_idx,
            max_pool_size: self.max_pool_size,
        }
    }

    /// Pre-warm the pool with vectors
    pub fn prewarm(&self, count: usize) {
        let per_pool = count / self.pools.len();

        for pool_mutex in self.pools.iter() {
            let mut pool = pool_mutex.lock();
            for _ in 0..per_pool {
                if pool.len() >= self.max_pool_size / self.pools.len() {
                    break;
                }
                pool.push_back(vec![T::default(); self.dimension]);
            }
        }
    }

    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        let mut total_pooled = 0;
        let mut total_capacity = 0;

        for pool in self.pools.iter() {
            let p = pool.lock();
            total_pooled += p.len();
            total_capacity += p.capacity();
        }

        PoolStats {
            pooled_count: total_pooled,
            total_capacity,
            pools_count: self.pools.len(),
        }
    }
}

/// RAII wrapper for pooled vectors
pub struct PooledVector<T> {
    vec: Option<Vec<T>>,
    pool: Arc<Vec<Mutex<VecDeque<Vec<T>>>>>,
    pool_idx: usize,
    max_pool_size: usize,
}

impl<T> PooledVector<T> {
    /// Get a reference to the inner vector
    pub fn get_ref(&self) -> &Vec<T> {
        self.vec.as_ref().unwrap()
    }

    /// Get a mutable reference to the inner vector
    pub fn get_mut(&mut self) -> &mut Vec<T> {
        self.vec.as_mut().unwrap()
    }

    /// Extract the inner vector, preventing it from being returned to the pool
    pub fn take(mut self) -> Vec<T> {
        self.vec.take().unwrap()
    }
}

impl<T> Drop for PooledVector<T> {
    fn drop(&mut self) {
        if let Some(vec) = self.vec.take() {
            let mut pool = self.pool[self.pool_idx].lock();

            // Only return to pool if under capacity limit
            if pool.len() < self.max_pool_size / self.pool.len() {
                pool.push_back(vec);
            }
        }
    }
}

impl<T> std::ops::Deref for PooledVector<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        self.get_ref()
    }
}

impl<T> std::ops::DerefMut for PooledVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Statistics for the vector pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub pooled_count: usize,
    pub total_capacity: usize,
    pub pools_count: usize,
}

/// Generic object pool for any cloneable type
pub struct ObjectPool<T: Clone> {
    pool: Mutex<Vec<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T: Clone> ObjectPool<T> {
    /// Create a new object pool with a factory function
    pub fn new(factory: impl Fn() -> T + Send + Sync + 'static, max_size: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::with_capacity(max_size)),
            factory: Box::new(factory),
            max_size,
        }
    }

    /// Acquire an object from the pool
    pub fn acquire(&self) -> T {
        let mut pool = self.pool.lock();
        pool.pop().unwrap_or_else(|| (self.factory)())
    }

    /// Return an object to the pool
    pub fn release(&self, obj: T) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_size {
            pool.push(obj);
        }
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.lock().clear();
    }
}

/// Slab allocator for fixed-size allocations
pub struct SlabAllocator<T> {
    slabs: Vec<Slab<T>>,
    free_list: VecDeque<SlabRef>,
    slab_size: usize,
}

struct Slab<T> {
    data: Vec<Option<T>>,
    free_count: usize,
}

#[derive(Clone, Copy)]
pub struct SlabRef {
    slab_idx: usize,
    slot_idx: usize,
}

impl<T> SlabAllocator<T> {
    /// Create a new slab allocator
    pub fn new(slab_size: usize) -> Self {
        Self {
            slabs: Vec::new(),
            free_list: VecDeque::new(),
            slab_size,
        }
    }

    /// Allocate a slot for an object
    pub fn allocate(&mut self, value: T) -> SlabRef {
        if let Some(slot_ref) = self.free_list.pop_front() {
            self.slabs[slot_ref.slab_idx].data[slot_ref.slot_idx] = Some(value);
            self.slabs[slot_ref.slab_idx].free_count -= 1;
            slot_ref
        } else {
            // Allocate new slab
            let slab_idx = self.slabs.len();
            let mut slab = Slab {
                data: Vec::with_capacity(self.slab_size),
                free_count: self.slab_size - 1,
            };

            // Initialize slab
            slab.data.push(Some(value));
            for i in 1..self.slab_size {
                slab.data.push(None);
                self.free_list.push_back(SlabRef {
                    slab_idx,
                    slot_idx: i,
                });
            }

            self.slabs.push(slab);

            SlabRef {
                slab_idx,
                slot_idx: 0,
            }
        }
    }

    /// Deallocate a slot
    pub fn deallocate(&mut self, slot_ref: SlabRef) {
        if let Some(slab) = self.slabs.get_mut(slot_ref.slab_idx) {
            slab.data[slot_ref.slot_idx] = None;
            slab.free_count += 1;
            self.free_list.push_back(slot_ref);
        }
    }

    /// Get a reference to an allocated object
    pub fn get(&self, slot_ref: SlabRef) -> Option<&T> {
        self.slabs
            .get(slot_ref.slab_idx)
            .and_then(|slab| slab.data.get(slot_ref.slot_idx))
            .and_then(|opt| opt.as_ref())
    }

    /// Get a mutable reference to an allocated object
    pub fn get_mut(&mut self, slot_ref: SlabRef) -> Option<&mut T> {
        self.slabs
            .get_mut(slot_ref.slab_idx)
            .and_then(|slab| slab.data.get_mut(slot_ref.slot_idx))
            .and_then(|opt| opt.as_mut())
    }
}

/// Arena allocator for batch allocations
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current: Vec<u8>,
    chunk_size: usize,
}

impl Arena {
    /// Create a new arena with specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current: Vec::with_capacity(chunk_size),
            chunk_size,
        }
    }

    /// Allocate bytes in the arena
    pub fn allocate(&mut self, size: usize) -> *mut u8 {
        if self.current.len() + size > self.chunk_size {
            // Move current chunk to storage and allocate new one
            let chunk = mem::replace(&mut self.current, Vec::with_capacity(self.chunk_size));
            if !chunk.is_empty() {
                self.chunks.push(chunk);
            }
        }

        let ptr = unsafe { self.current.as_mut_ptr().add(self.current.len()) };

        self.current.resize(self.current.len() + size, 0);
        ptr
    }

    /// Reset the arena, clearing all allocations
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.current.clear();
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity()).sum::<usize>() + self.current.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_pool() {
        let pool: VectorPool<f32> = VectorPool::new(128, 10);

        let mut vec1 = pool.acquire();
        vec1[0] = 1.0;

        let stats = pool.stats();
        assert_eq!(stats.pooled_count, 0);

        drop(vec1);

        let stats = pool.stats();
        assert!(stats.pooled_count > 0);
    }

    #[test]
    fn test_object_pool() {
        let pool = ObjectPool::new(|| vec![0u8; 1024], 5);

        let obj1 = pool.acquire();
        assert_eq!(obj1.len(), 1024);

        pool.release(obj1);
    }

    #[test]
    fn test_slab_allocator() {
        let mut allocator = SlabAllocator::new(10);

        let ref1 = allocator.allocate("hello");
        let ref2 = allocator.allocate("world");

        assert_eq!(allocator.get(ref1), Some(&"hello"));
        assert_eq!(allocator.get(ref2), Some(&"world"));

        allocator.deallocate(ref1);
        let ref3 = allocator.allocate("reused");
        assert_eq!(allocator.get(ref3), Some(&"reused"));
    }

    #[test]
    fn test_arena() {
        let mut arena = Arena::new(1024);

        let _ptr1 = arena.allocate(100);
        let _ptr2 = arena.allocate(200);

        assert!(arena.allocated_bytes() >= 300);

        arena.reset();
        assert_eq!(arena.allocated_bytes(), 0);
    }
}

// Helper module for thread ID
mod thread_id {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static ID: Cell<usize> = const { Cell::new(usize::MAX) };
    }

    pub fn get() -> usize {
        ID.with(|id| {
            if id.get() == usize::MAX {
                id.set(COUNTER.fetch_add(1, Ordering::Relaxed));
            }
            id.get()
        })
    }
}
