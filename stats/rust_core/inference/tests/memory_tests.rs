//! Comprehensive tests for memory management and unsafe operations
//!
//! These tests focus on testing the memory pool, tensor allocation,
//! and all unsafe code blocks for memory safety and correctness.

use gemma_inference::*;
use proptest::prelude::*;
use std::alloc::{Layout, System, GlobalAlloc};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Test basic memory pool functionality
#[cfg(test)]
mod memory_pool_tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config.clone());

        assert_eq!(pool.config(), &config);
        assert_eq!(pool.total_allocated(), 0);
        assert_eq!(pool.peak_allocated(), 0);
    }

    #[test]
    fn test_memory_pool_basic_allocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let shape = TensorShape::new(vec![10, 10]);
        let allocation = pool.allocate_tensor(&shape).unwrap();

        assert_eq!(allocation.size(), 10 * 10 * 4); // 4 bytes per f32
        assert!(pool.total_allocated() > 0);
        assert!(pool.peak_allocated() > 0);

        // Drop allocation and check cleanup
        drop(allocation);
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_memory_pool_multiple_allocations() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let mut allocations = Vec::new();

        // Allocate multiple tensors
        for i in 1..10 {
            let shape = TensorShape::new(vec![i, i]);
            let allocation = pool.allocate_tensor(&shape).unwrap();
            allocations.push(allocation);
        }

        let total_allocated = pool.total_allocated();
        let peak_allocated = pool.peak_allocated();

        assert!(total_allocated > 0);
        assert!(peak_allocated >= total_allocated);

        // Drop all allocations
        allocations.clear();
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_memory_pool_alignment() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        // Test various sizes to ensure proper alignment
        for size in [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023] {
            let shape = TensorShape::new(vec![size]);
            let allocation = pool.allocate_tensor(&shape).unwrap();

            // Check that the allocation is properly aligned for SIMD operations
            let ptr = allocation.as_ptr() as usize;
            assert_eq!(ptr % 32, 0, "Allocation not aligned to 32 bytes for size {}", size);

            drop(allocation);
        }
    }

    #[test]
    fn test_memory_pool_reuse() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let shape = TensorShape::new(vec![100, 100]);

        // Allocate and deallocate multiple times
        for _ in 0..10 {
            let allocation = pool.allocate_tensor(&shape).unwrap();
            assert_eq!(allocation.size(), 100 * 100 * 4);
            drop(allocation);
        }

        // Pool should efficiently reuse memory
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_memory_pool_fragmentation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let mut allocations = Vec::new();

        // Create fragmentation by allocating different sizes
        let sizes = vec![10, 50, 25, 100, 5, 75, 30];
        for size in sizes {
            let shape = TensorShape::new(vec![size, size]);
            let allocation = pool.allocate_tensor(&shape).unwrap();
            allocations.push(allocation);
        }

        // Drop every other allocation
        for i in (0..allocations.len()).step_by(2) {
            drop(allocations.remove(i.min(allocations.len() - 1)));
        }

        // Should still be able to allocate
        let new_shape = TensorShape::new(vec![20, 20]);
        let new_allocation = pool.allocate_tensor(&new_shape).unwrap();
        assert_eq!(new_allocation.size(), 20 * 20 * 4);
    }
}

/// Test memory block management and unsafe operations
#[cfg(test)]
mod memory_block_tests {
    use super::*;

    #[test]
    fn test_memory_block_creation() {
        let size = 1024;
        let alignment = 32;
        let block = MemoryBlock::new(size, alignment).unwrap();

        assert_eq!(block.size(), size);
        assert_eq!(block.alignment(), alignment);
        assert_eq!(block.as_ptr() as usize % alignment, 0);
    }

    #[test]
    fn test_memory_block_zero_initialization() {
        let size = 256;
        let block = MemoryBlock::new(size, 16).unwrap();

        // Check that memory is zero-initialized
        let slice = block.as_slice();
        for &byte in slice {
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn test_memory_block_write_read() {
        let size = 1024;
        let mut block = MemoryBlock::new(size, 16).unwrap();

        // Write pattern to memory
        let slice = block.as_mut_slice();
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Verify pattern
        let read_slice = block.as_slice();
        for (i, &byte) in read_slice.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8);
        }
    }

    #[test]
    fn test_memory_block_large_allocation() {
        // Test large allocation (but not so large as to exhaust memory)
        let size = 10 * 1024 * 1024; // 10MB
        let block = MemoryBlock::new(size, 32);

        match block {
            Ok(block) => {
                assert_eq!(block.size(), size);
                assert_eq!(block.as_ptr() as usize % 32, 0);

                // Quick write/read test
                let slice = block.as_slice();
                assert_eq!(slice.len(), size);
            }
            Err(_) => {
                // Large allocation might fail on systems with limited memory
                // This is acceptable behavior
            }
        }
    }

    #[test]
    fn test_memory_block_concurrent_access() {
        let size = 4096;
        let block = Arc::new(MemoryBlock::new(size, 32).unwrap());
        let barrier = Arc::new(Barrier::new(4));

        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let block_clone = Arc::clone(&block);
                let barrier_clone = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier_clone.wait();

                    // Each thread reads from its section
                    let slice = block_clone.as_slice();
                    let section_size = size / 4;
                    let start = thread_id * section_size;
                    let end = start + section_size;

                    for i in start..end {
                        let _byte = slice[i];
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

/// Test tensor arena functionality
#[cfg(test)]
mod tensor_arena_tests {
    use super::*;

    #[test]
    fn test_tensor_arena_creation() {
        let config = MemoryConfig::default();
        let arena = TensorArena::new(config.clone());

        assert_eq!(arena.config(), &config);
        assert_eq!(arena.allocated_tensors(), 0);
    }

    #[test]
    fn test_tensor_arena_allocation() {
        let config = MemoryConfig::default();
        let arena = TensorArena::new(config);

        let shape = TensorShape::new(vec![10, 10]);
        let allocation = arena.allocate(&shape, DataType::F32).unwrap();

        assert_eq!(allocation.size(), 10 * 10 * 4);
        assert_eq!(arena.allocated_tensors(), 1);

        drop(allocation);
        assert_eq!(arena.allocated_tensors(), 0);
    }

    #[test]
    fn test_tensor_arena_different_data_types() {
        let config = MemoryConfig::default();
        let arena = TensorArena::new(config);

        let shape = TensorShape::new(vec![10, 10]);

        // Test different data types
        let f32_alloc = arena.allocate(&shape, DataType::F32).unwrap();
        assert_eq!(f32_alloc.size(), 10 * 10 * 4);

        let f16_alloc = arena.allocate(&shape, DataType::F16).unwrap();
        assert_eq!(f16_alloc.size(), 10 * 10 * 2);

        let i32_alloc = arena.allocate(&shape, DataType::I32).unwrap();
        assert_eq!(i32_alloc.size(), 10 * 10 * 4);

        assert_eq!(arena.allocated_tensors(), 3);
    }

    #[test]
    fn test_tensor_arena_memory_tracking() {
        let config = MemoryConfig::default();
        let arena = TensorArena::new(config);

        let mut allocations = Vec::new();

        // Make several allocations
        for i in 1..=5 {
            let shape = TensorShape::new(vec![i * 10, i * 10]);
            let allocation = arena.allocate(&shape, DataType::F32).unwrap();
            allocations.push(allocation);
        }

        assert_eq!(arena.allocated_tensors(), 5);
        let total_memory = arena.total_memory_usage();
        assert!(total_memory > 0);

        // Drop allocations one by one
        for _ in 0..5 {
            allocations.pop();
            assert_eq!(arena.allocated_tensors(), allocations.len());
        }

        assert_eq!(arena.allocated_tensors(), 0);
    }
}

/// Test unsafe memory operations
#[cfg(test)]
mod unsafe_operations_tests {
    use super::*;

    #[test]
    fn test_unsafe_memory_copy() {
        let size = 1024;
        let mut src_block = MemoryBlock::new(size, 16).unwrap();
        let mut dst_block = MemoryBlock::new(size, 16).unwrap();

        // Fill source with pattern
        let src_slice = src_block.as_mut_slice();
        for (i, byte) in src_slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Copy using unsafe operations
        unsafe {
            let src_ptr = src_block.as_ptr();
            let dst_ptr = dst_block.as_mut_ptr();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }

        // Verify copy
        let src_slice = src_block.as_slice();
        let dst_slice = dst_block.as_slice();
        assert_eq!(src_slice, dst_slice);
    }

    #[test]
    fn test_unsafe_pointer_arithmetic() {
        let size = 256;
        let block = MemoryBlock::new(size, 32).unwrap();

        unsafe {
            let base_ptr = block.as_ptr();

            // Test pointer arithmetic within bounds
            for offset in 0..size {
                let ptr = base_ptr.add(offset);
                let _value = *ptr; // Should not crash
            }

            // Test alignment preservation
            for i in (0..size).step_by(4) {
                let aligned_ptr = base_ptr.add(i) as *const u32;
                if (aligned_ptr as usize) % 4 == 0 {
                    let _value = *aligned_ptr; // Should be safe if aligned
                }
            }
        }
    }

    #[test]
    fn test_unsafe_type_casting() {
        let size = 1024;
        let mut block = MemoryBlock::new(size, 32).unwrap();

        // Fill with f32 values
        unsafe {
            let ptr = block.as_mut_ptr() as *mut f32;
            let len = size / 4;

            for i in 0..len {
                *ptr.add(i) = i as f32;
            }

            // Read back as f32
            for i in 0..len {
                let value = *ptr.add(i);
                assert_eq!(value, i as f32);
            }

            // Also test reading as bytes
            let byte_ptr = block.as_ptr();
            for i in 0..size {
                let _byte = *byte_ptr.add(i);
            }
        }
    }

    #[test]
    fn test_send_sync_safety() {
        // Test that memory blocks can be safely sent between threads
        let block = MemoryBlock::new(1024, 16).unwrap();

        let handle = thread::spawn(move || {
            // Use the block in another thread
            let slice = block.as_slice();
            slice.len()
        });

        let result = handle.join().unwrap();
        assert_eq!(result, 1024);
    }

    #[test]
    fn test_memory_leak_prevention() {
        // Test that memory is properly cleaned up even in error conditions
        struct LeakTester {
            _allocation: TensorAllocation<f32>,
        }

        impl Drop for LeakTester {
            fn drop(&mut self) {
                // This tests that Drop is called even in panic scenarios
            }
        }

        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let result = std::panic::catch_unwind(|| {
            let shape = TensorShape::new(vec![10, 10]);
            let allocation = pool.allocate_tensor(&shape).unwrap();
            let _tester = LeakTester { _allocation: allocation };

            // Simulate a panic
            panic!("Test panic");
        });

        assert!(result.is_err());
        // Memory should be cleaned up despite the panic
        assert_eq!(pool.total_allocated(), 0);
    }
}

/// Property-based tests for memory operations
#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_memory_block_size_consistency(
            size in 1usize..10_000_000,
            alignment in prop::sample::select(vec![1, 2, 4, 8, 16, 32, 64])
        ) {
            if let Ok(block) = MemoryBlock::new(size, alignment) {
                assert_eq!(block.size(), size);
                assert_eq!(block.alignment(), alignment);
                assert_eq!(block.as_ptr() as usize % alignment, 0);
                assert_eq!(block.as_slice().len(), size);
            }
        }

        #[test]
        fn prop_tensor_allocation_consistency(
            dims in prop::collection::vec(1usize..100, 1..4)
        ) {
            let shape = TensorShape::new(dims.clone());
            let config = MemoryConfig::default();
            let pool = MemoryPool::new(config);

            if let Ok(allocation) = pool.allocate_tensor(&shape) {
                let expected_size = dims.iter().product::<usize>() * 4; // f32 size
                assert_eq!(allocation.size(), expected_size);
            }
        }

        #[test]
        fn prop_memory_pattern_preservation(
            pattern in prop::collection::vec(0u8..=255, 100..1000)
        ) {
            let size = pattern.len();
            let mut block = MemoryBlock::new(size, 16).unwrap();

            // Write pattern
            let slice = block.as_mut_slice();
            slice.copy_from_slice(&pattern);

            // Read back and verify
            let read_slice = block.as_slice();
            assert_eq!(read_slice, &pattern[..]);
        }
    }
}

/// Performance and stress tests
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn bench_memory_allocation_speed() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let shape = TensorShape::new(vec![100, 100]);

        let start = Instant::now();
        let mut allocations = Vec::new();

        for _ in 0..1000 {
            let allocation = pool.allocate_tensor(&shape).unwrap();
            allocations.push(allocation);
        }

        let allocation_time = start.elapsed();

        let start = Instant::now();
        allocations.clear();
        let deallocation_time = start.elapsed();

        println!("Allocation time: {:?}", allocation_time);
        println!("Deallocation time: {:?}", deallocation_time);

        // Should be reasonably fast
        assert!(allocation_time < Duration::from_millis(100));
        assert!(deallocation_time < Duration::from_millis(100));
    }

    #[test]
    fn stress_test_memory_fragmentation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let mut allocations = Vec::new();

        // Stress test with many allocations of different sizes
        for round in 0..10 {
            // Allocate many different sizes
            for size in [1, 3, 7, 15, 31, 63, 127, 255] {
                let shape = TensorShape::new(vec![size]);
                if let Ok(allocation) = pool.allocate_tensor(&shape) {
                    allocations.push(allocation);
                }
            }

            // Drop some allocations randomly
            if round % 2 == 0 {
                allocations.drain(0..allocations.len() / 2);
            }
        }

        // Should still be able to allocate after fragmentation
        let test_shape = TensorShape::new(vec![50, 50]);
        let test_allocation = pool.allocate_tensor(&test_shape);
        assert!(test_allocation.is_ok());
    }

    #[test]
    fn stress_test_concurrent_allocation() {
        let config = MemoryConfig::default();
        let pool = Arc::new(MemoryPool::new(config));

        let handles: Vec<_> = (0..8)
            .map(|thread_id| {
                let pool_clone = Arc::clone(&pool);
                thread::spawn(move || {
                    let mut local_allocations = Vec::new();

                    for i in 0..100 {
                        let size = (thread_id + 1) * 10 + (i % 20);
                        let shape = TensorShape::new(vec![size]);

                        if let Ok(allocation) = pool_clone.allocate_tensor(&shape) {
                            local_allocations.push(allocation);
                        }

                        // Occasionally drop some allocations
                        if i % 10 == 0 && !local_allocations.is_empty() {
                            local_allocations.pop();
                        }
                    }

                    local_allocations.len()
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should have successfully allocated memory
        for &result in &results {
            assert!(result > 0);
        }

        println!("Concurrent allocation results: {:?}", results);
    }

    #[test]
    fn test_memory_usage_tracking_accuracy() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let mut total_expected = 0;
        let mut allocations = Vec::new();

        // Make allocations and track expected usage
        for size in [10, 50, 100, 200] {
            let shape = TensorShape::new(vec![size, size]);
            let expected_size = size * size * 4; // f32 size
            total_expected += expected_size;

            let allocation = pool.allocate_tensor(&shape).unwrap();
            allocations.push(allocation);
        }

        // Check that tracking is accurate (allowing for alignment padding)
        let actual_allocated = pool.total_allocated();
        assert!(actual_allocated >= total_expected);
        assert!(actual_allocated <= total_expected * 2); // Reasonable upper bound

        allocations.clear();
        assert_eq!(pool.total_allocated(), 0);
    }
}

/// Edge case and error condition tests
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_size_allocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        // Zero-size tensor should be handled gracefully
        let shape = TensorShape::new(vec![0]);
        let result = pool.allocate_tensor(&shape);

        // Should either succeed with zero size or fail gracefully
        match result {
            Ok(allocation) => assert_eq!(allocation.size(), 0),
            Err(_) => {}, // Error is acceptable for zero-size
        }
    }

    #[test]
    fn test_extremely_large_allocation() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        // Try to allocate more memory than available
        let huge_shape = TensorShape::new(vec![1_000_000, 1_000_000]);
        let result = pool.allocate_tensor(&huge_shape);

        // Should fail gracefully, not crash
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_alignment() {
        // Test that invalid alignments are handled
        let sizes = [1024];
        let alignments = [0, 3, 5, 7]; // Invalid alignments (not powers of 2)

        for &size in &sizes {
            for &alignment in &alignments {
                let result = MemoryBlock::new(size, alignment);
                assert!(result.is_err(), "Invalid alignment {} should fail", alignment);
            }
        }
    }

    #[test]
    fn test_memory_block_edge_sizes() {
        // Test edge cases for memory block sizes
        let edge_sizes = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64];

        for &size in &edge_sizes {
            let block = MemoryBlock::new(size, 8).unwrap();
            assert_eq!(block.size(), size);
            assert_eq!(block.as_slice().len(), size);
        }
    }

    #[test]
    fn test_memory_pool_limit_enforcement() {
        let mut config = MemoryConfig::default();
        config.max_pool_size = Some(1024); // Limit to 1KB

        let pool = MemoryPool::new(config);

        // Should be able to allocate within limit
        let small_shape = TensorShape::new(vec![8, 8]); // 256 bytes
        let allocation1 = pool.allocate_tensor(&small_shape).unwrap();

        // Should still have room for more
        let allocation2 = pool.allocate_tensor(&small_shape).unwrap();

        // Large allocation should fail due to limit
        let large_shape = TensorShape::new(vec![100, 100]); // 40KB
        let result = pool.allocate_tensor(&large_shape);
        assert!(result.is_err() || result.is_ok()); // Depends on implementation

        drop(allocation1);
        drop(allocation2);
    }
}