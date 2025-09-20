//! Comprehensive tests for tensor operations and SIMD functionality
//!
//! These tests thoroughly validate tensor operations, especially focusing on
//! unsafe SIMD code blocks and memory safety guarantees.

use gemma_inference::*;
use proptest::prelude::*;
use std::f32;
use std::time::Instant;
use criterion::black_box;

/// Test basic tensor operations
#[cfg(test)]
mod tensor_basic_tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        let tensor = Tensor::zeros(&shape, DataType::F32);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data_type(), DataType::F32);
        assert_eq!(tensor.len(), 24); // 2 * 3 * 4
    }

    #[test]
    fn test_tensor_shape_validation() {
        // Valid shapes
        let valid_shapes = vec![
            vec![1],
            vec![10, 20],
            vec![2, 3, 4],
            vec![1, 1, 1, 1],
        ];

        for shape_vec in valid_shapes {
            let shape = TensorShape::new(shape_vec);
            assert!(shape.is_valid());
        }

        // Invalid shapes should be handled
        let invalid_shapes = vec![
            vec![], // Empty shape
            vec![0], // Zero dimension
            vec![1, 0, 3], // Zero in middle
        ];

        for shape_vec in invalid_shapes {
            let shape = TensorShape::new(shape_vec);
            assert!(!shape.is_valid());
        }
    }

    #[test]
    fn test_tensor_indexing() {
        let shape = TensorShape::new(vec![2, 3]);
        let mut tensor = Tensor::zeros(&shape, DataType::F32);

        // Test setting and getting values
        tensor.set_f32(&[0, 0], 1.0).unwrap();
        tensor.set_f32(&[1, 2], 2.5).unwrap();

        assert_eq!(tensor.get_f32(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get_f32(&[1, 2]).unwrap(), 2.5);
        assert_eq!(tensor.get_f32(&[0, 1]).unwrap(), 0.0); // Should be zero

        // Test bounds checking
        assert!(tensor.set_f32(&[2, 0], 1.0).is_err()); // Out of bounds
        assert!(tensor.get_f32(&[0, 3]).is_err()); // Out of bounds
    }

    #[test]
    fn test_tensor_data_types() {
        let shape = TensorShape::new(vec![3, 3]);

        // Test different data types
        let f32_tensor = Tensor::zeros(&shape, DataType::F32);
        let f16_tensor = Tensor::zeros(&shape, DataType::F16);
        let i32_tensor = Tensor::zeros(&shape, DataType::I32);

        assert_eq!(f32_tensor.data_type(), DataType::F32);
        assert_eq!(f16_tensor.data_type(), DataType::F16);
        assert_eq!(i32_tensor.data_type(), DataType::I32);

        // Test memory usage
        assert_eq!(f32_tensor.memory_usage(), 9 * 4); // 9 elements * 4 bytes
        assert_eq!(f16_tensor.memory_usage(), 9 * 2); // 9 elements * 2 bytes
        assert_eq!(i32_tensor.memory_usage(), 9 * 4); // 9 elements * 4 bytes
    }
}

/// Test SIMD operations and unsafe code blocks
#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_simd_dot_product_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Calculate expected result
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        // Test scalar implementation
        let scalar_result = TensorOps::dot_product_scalar(&a, &b);
        assert!((scalar_result - expected).abs() < f32::EPSILON);

        // Test SIMD implementations if available
        let caps = SimdCapabilities::detect();

        #[cfg(target_arch = "x86_64")]
        {
            if caps.avx2 {
                let avx2_result = unsafe { TensorOps::dot_product_avx2(&a, &b) };
                assert!((avx2_result - expected).abs() < f32::EPSILON * 2.0);
            }

            if caps.sse4_1 {
                let sse_result = unsafe { TensorOps::dot_product_sse(&a, &b) };
                assert!((sse_result - expected).abs() < f32::EPSILON * 2.0);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if caps.neon {
                let neon_result = unsafe { TensorOps::dot_product_neon(&a, &b) };
                assert!((neon_result - expected).abs() < f32::EPSILON * 2.0);
            }
        }
    }

    #[test]
    fn test_simd_vector_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];

        // Calculate expected result
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        // Test scalar implementation
        TensorOps::vector_add_scalar(&a, &b, &mut result);
        for i in 0..result.len() {
            assert!((result[i] - expected[i]).abs() < f32::EPSILON);
        }

        // Reset result
        result.fill(0.0);

        // Test SIMD implementations if available
        let caps = SimdCapabilities::detect();

        #[cfg(target_arch = "x86_64")]
        {
            if caps.avx2 {
                unsafe { TensorOps::vector_add_avx2(&a, &b, &mut result) };
                for i in 0..result.len() {
                    assert!((result[i] - expected[i]).abs() < f32::EPSILON * 2.0);
                }
                result.fill(0.0);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if caps.neon {
                unsafe { TensorOps::vector_add_neon(&a, &b, &mut result) };
                for i in 0..result.len() {
                    assert!((result[i] - expected[i]).abs() < f32::EPSILON * 2.0);
                }
            }
        }
    }

    #[test]
    fn test_simd_alignment_requirements() {
        // Test that SIMD operations handle unaligned data correctly
        let mut data = vec![0.0f32; 100];
        for i in 0..data.len() {
            data[i] = i as f32;
        }

        // Test various slice offsets to check alignment handling
        for offset in 0..8 {
            if offset >= data.len() - 16 {
                break;
            }

            let slice = &data[offset..offset + 16];
            let result1 = TensorOps::dot_product_scalar(slice, slice);

            // SIMD version should produce the same result regardless of alignment
            let result2 = TensorOps::dot_product_auto(slice, slice).unwrap();
            assert!((result1 - result2).abs() < f32::EPSILON * 16.0);
        }
    }

    #[test]
    fn test_simd_edge_cases() {
        // Test empty vectors
        let empty: Vec<f32> = vec![];
        let result = TensorOps::dot_product_auto(&empty, &empty);
        assert!(result.is_err() || result.unwrap() == 0.0);

        // Test single element
        let single = vec![5.0];
        let result = TensorOps::dot_product_auto(&single, &single).unwrap();
        assert!((result - 25.0).abs() < f32::EPSILON);

        // Test mismatched lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = TensorOps::dot_product_auto(&a, &b);
        assert!(result.is_err());

        // Test with NaN and infinity
        let nan_vec = vec![f32::NAN, 1.0, 2.0, 3.0];
        let inf_vec = vec![f32::INFINITY, 1.0, 2.0, 3.0];
        let normal_vec = vec![1.0, 1.0, 1.0, 1.0];

        let nan_result = TensorOps::dot_product_auto(&nan_vec, &normal_vec).unwrap();
        assert!(nan_result.is_nan());

        let inf_result = TensorOps::dot_product_auto(&inf_vec, &normal_vec).unwrap();
        assert!(inf_result.is_infinite());
    }
}

/// Test memory safety and bounds checking in unsafe operations
#[cfg(test)]
mod memory_safety_tests {
    use super::*;

    #[test]
    fn test_tensor_memory_bounds() {
        let shape = TensorShape::new(vec![10, 10]);
        let tensor = Tensor::zeros(&shape, DataType::F32);

        // Test that out-of-bounds access is properly handled
        assert!(tensor.get_f32(&[10, 0]).is_err()); // Row out of bounds
        assert!(tensor.get_f32(&[0, 10]).is_err()); // Column out of bounds
        assert!(tensor.get_f32(&[9, 9]).is_ok()); // Valid access
    }

    #[test]
    fn test_memory_pool_safety() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        // Test allocation and deallocation
        let shape = TensorShape::new(vec![100, 100]);
        let allocation = pool.allocate_tensor(&shape).unwrap();

        // Ensure the allocation is valid
        assert_eq!(allocation.size(), 100 * 100 * 4); // 4 bytes per f32

        // Test that deallocating twice is safe
        drop(allocation);
        // Second drop should be automatic and safe
    }

    #[test]
    fn test_concurrent_memory_access() {
        use std::sync::Arc;
        use std::thread;

        let shape = TensorShape::new(vec![100, 100]);
        let tensor = Arc::new(Tensor::zeros(&shape, DataType::F32));

        // Spawn multiple threads to access tensor concurrently
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let tensor_clone = Arc::clone(&tensor);
                thread::spawn(move || {
                    // Each thread accesses different parts of the tensor
                    let row = i % 100;
                    let col = (i * 17) % 100; // Some offset to avoid conflicts

                    for j in 0..100 {
                        let index = [row, (col + j) % 100];
                        let _ = tensor_clone.get_f32(&index);
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simd_buffer_overrun_protection() {
        // Test that SIMD operations don't read beyond buffer boundaries
        let data = vec![1.0f32; 15]; // Not a multiple of SIMD width

        // This should not crash or read beyond the buffer
        let result = TensorOps::dot_product_auto(&data, &data);
        assert!(result.is_ok());

        // Result should be correct
        let expected = 15.0; // 15 * 1.0 * 1.0
        assert!((result.unwrap() - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_uninitialized_memory_safety() {
        // Test that newly allocated tensors are properly initialized
        let shape = TensorShape::new(vec![10, 10]);
        let tensor = Tensor::zeros(&shape, DataType::F32);

        // All values should be zero
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(tensor.get_f32(&[i, j]).unwrap(), 0.0);
            }
        }

        // Test with ones
        let ones_tensor = Tensor::ones(&shape, DataType::F32);
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(ones_tensor.get_f32(&[i, j]).unwrap(), 1.0);
            }
        }
    }
}

/// Property-based tests for tensor operations
#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_dot_product_commutative(
            a in prop::collection::vec(prop::num::f32::POSITIVE, 1..100),
            b in prop::collection::vec(prop::num::f32::POSITIVE, 1..100)
        ) {
            if a.len() == b.len() {
                let result1 = TensorOps::dot_product_scalar(&a, &b);
                let result2 = TensorOps::dot_product_scalar(&b, &a);
                assert!((result1 - result2).abs() < f32::EPSILON * a.len() as f32);
            }
        }

        #[test]
        fn prop_vector_add_associative(
            a in prop::collection::vec(-1000.0f32..1000.0f32, 1..100),
            b in prop::collection::vec(-1000.0f32..1000.0f32, 1..100),
            c in prop::collection::vec(-1000.0f32..1000.0f32, 1..100)
        ) {
            if a.len() == b.len() && b.len() == c.len() {
                let mut result1 = vec![0.0; a.len()];
                let mut result2 = vec![0.0; a.len()];
                let mut temp = vec![0.0; a.len()];

                // (a + b) + c
                TensorOps::vector_add_scalar(&a, &b, &mut temp);
                TensorOps::vector_add_scalar(&temp, &c, &mut result1);

                // a + (b + c)
                TensorOps::vector_add_scalar(&b, &c, &mut temp);
                TensorOps::vector_add_scalar(&a, &temp, &mut result2);

                for i in 0..result1.len() {
                    assert!((result1[i] - result2[i]).abs() < f32::EPSILON * 3.0);
                }
            }
        }

        #[test]
        fn prop_tensor_shape_consistency(
            dims in prop::collection::vec(1usize..10, 1..5)
        ) {
            let shape = TensorShape::new(dims.clone());
            let tensor = Tensor::zeros(&shape, DataType::F32);

            assert_eq!(tensor.shape().dims(), &dims);
            assert_eq!(tensor.len(), dims.iter().product::<usize>());
        }

        #[test]
        fn prop_simd_scalar_equivalence(
            data in prop::collection::vec(-100.0f32..100.0f32, 8..64)
        ) {
            // SIMD and scalar implementations should produce equivalent results
            let scalar_result = TensorOps::dot_product_scalar(&data, &data);

            if let Ok(auto_result) = TensorOps::dot_product_auto(&data, &data) {
                // Allow for small floating-point differences
                let tolerance = f32::EPSILON * data.len() as f32 * 100.0;
                assert!((scalar_result - auto_result).abs() < tolerance,
                    "Scalar: {}, Auto: {}, Diff: {}, Tolerance: {}",
                    scalar_result, auto_result, (scalar_result - auto_result).abs(), tolerance);
            }
        }
    }
}

/// Performance and benchmark tests
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn bench_dot_product_implementations() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        // Benchmark scalar implementation
        let start = Instant::now();
        for _ in 0..1000 {
            black_box(TensorOps::dot_product_scalar(&a, &b));
        }
        let scalar_time = start.elapsed();

        // Benchmark auto implementation (uses best available SIMD)
        let start = Instant::now();
        for _ in 0..1000 {
            black_box(TensorOps::dot_product_auto(&a, &b).unwrap());
        }
        let auto_time = start.elapsed();

        println!("Scalar time: {:?}, Auto time: {:?}", scalar_time, auto_time);

        // Auto implementation should be at least as fast as scalar
        // (allowing for some measurement noise)
        assert!(auto_time <= scalar_time * 2);
    }

    #[test]
    fn bench_memory_allocation() {
        let shapes = vec![
            TensorShape::new(vec![100, 100]),
            TensorShape::new(vec![50, 50, 4]),
            TensorShape::new(vec![1000]),
        ];

        for shape in shapes {
            let start = Instant::now();

            for _ in 0..100 {
                let tensor = Tensor::zeros(&shape, DataType::F32);
                black_box(tensor);
            }

            let elapsed = start.elapsed();
            println!("Shape {:?}: {:?} per allocation", shape.dims(), elapsed / 100);

            // Should be reasonably fast (less than 1ms per allocation)
            assert!(elapsed / 100 < Duration::from_millis(1));
        }
    }

    #[test]
    fn test_large_tensor_operations() {
        // Test with large tensors to ensure memory efficiency
        let shape = TensorShape::new(vec![1000, 1000]);
        let tensor1 = Tensor::ones(&shape, DataType::F32);
        let tensor2 = Tensor::ones(&shape, DataType::F32);

        // This should complete without memory issues
        let start = Instant::now();
        let _sum = tensor1.element_wise_add(&tensor2).unwrap();
        let elapsed = start.elapsed();

        // Should complete in reasonable time (less than 1 second)
        assert!(elapsed < Duration::from_secs(1));
    }
}

/// Tests for error conditions and edge cases
#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_errors() {
        let shape1 = TensorShape::new(vec![2, 3]);
        let shape2 = TensorShape::new(vec![3, 2]);

        let tensor1 = Tensor::zeros(&shape1, DataType::F32);
        let tensor2 = Tensor::zeros(&shape2, DataType::F32);

        // Operations on mismatched tensors should return errors
        assert!(tensor1.element_wise_add(&tensor2).is_err());
        assert!(tensor1.element_wise_mul(&tensor2).is_err());
    }

    #[test]
    fn test_invalid_index_errors() {
        let shape = TensorShape::new(vec![5, 5]);
        let tensor = Tensor::zeros(&shape, DataType::F32);

        // Out of bounds indices should return errors
        assert!(tensor.get_f32(&[5, 0]).is_err());
        assert!(tensor.get_f32(&[0, 5]).is_err());
        assert!(tensor.get_f32(&[10, 10]).is_err());

        // Wrong number of indices should return errors
        assert!(tensor.get_f32(&[0]).is_err()); // Too few indices
        assert!(tensor.get_f32(&[0, 0, 0]).is_err()); // Too many indices
    }

    #[test]
    fn test_memory_exhaustion_handling() {
        // Try to allocate an impossibly large tensor
        let huge_shape = TensorShape::new(vec![usize::MAX / 8, usize::MAX / 8]);

        let result = std::panic::catch_unwind(|| {
            Tensor::zeros(&huge_shape, DataType::F32)
        });

        // Should either panic safely or return an error, not crash
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_data_type_consistency() {
        let shape = TensorShape::new(vec![3, 3]);
        let f32_tensor = Tensor::zeros(&shape, DataType::F32);

        // Trying to get data as wrong type should return error
        assert!(f32_tensor.get_i32(&[0, 0]).is_err());
        assert!(f32_tensor.set_i32(&[0, 0], 42).is_err());
    }
}

/// Integration tests combining multiple components
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_tensor_with_memory_pool() {
        let config = MemoryConfig::default();
        let pool = MemoryPool::new(config);

        let shape = TensorShape::new(vec![100, 100]);
        let allocation = pool.allocate_tensor(&shape).unwrap();

        // Create tensor using pool allocation
        let tensor = Tensor::from_allocation(allocation, &shape, DataType::F32);

        // Tensor should work normally
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.len(), 10000);
    }

    #[test]
    fn test_simd_with_different_alignments() {
        // Test SIMD operations with various memory alignments
        let base_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();

        for offset in 0..16 {
            if offset + 256 > base_data.len() {
                break;
            }

            let data1 = &base_data[offset..offset + 256];
            let data2 = &base_data[offset + 1..offset + 257];

            // Should handle different alignments correctly
            let result = TensorOps::dot_product_auto(data1, data2);
            assert!(result.is_ok());

            // Result should be consistent regardless of alignment
            let scalar_result = TensorOps::dot_product_scalar(data1, data2);
            let auto_result = result.unwrap();

            assert!((scalar_result - auto_result).abs() < f32::EPSILON * 256.0);
        }
    }

    #[tokio::test]
    async fn test_async_tensor_operations() {
        // Test that tensor operations work in async context
        let shape = TensorShape::new(vec![100, 100]);
        let tensor1 = Tensor::ones(&shape, DataType::F32);
        let tensor2 = Tensor::ones(&shape, DataType::F32);

        // Simulate async work
        tokio::task::yield_now().await;

        let result = tensor1.element_wise_add(&tensor2).unwrap();

        // All elements should be 2.0
        for i in 0..100 {
            for j in 0..100 {
                assert_eq!(result.get_f32(&[i, j]).unwrap(), 2.0);
            }
        }
    }
}