//! Comprehensive tests for the inference engine library
//!
//! These tests cover the core functionality, SIMD operations, memory management,
//! and error handling of the inference engine.

use gemma_inference::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task;
use proptest::prelude::*;

/// Test SIMD capability detection and functionality
#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        let level = caps.best_simd_level();

        // Should at least support scalar operations
        assert!(level >= SimdLevel::Scalar);

        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 should support at least SSE2
            assert!(caps.sse2);
        }

        // Test SIMD level ordering
        assert!(SimdLevel::Avx2 > SimdLevel::Avx);
        assert!(SimdLevel::Avx > SimdLevel::Sse41);
        assert!(SimdLevel::Sse41 > SimdLevel::Sse2);
        assert!(SimdLevel::Sse2 > SimdLevel::Scalar);
    }

    #[test]
    fn test_simd_consistency() {
        // Detection should be consistent across multiple calls
        let caps1 = SimdCapabilities::detect();
        let caps2 = SimdCapabilities::detect();

        assert_eq!(caps1.sse2, caps2.sse2);
        assert_eq!(caps1.avx, caps2.avx);
        assert_eq!(caps1.avx2, caps2.avx2);
        assert_eq!(caps1.neon, caps2.neon);
    }

    #[test]
    fn test_simd_level_properties() {
        let caps = SimdCapabilities::detect();

        // If AVX2 is supported, AVX should also be supported
        if caps.avx2 {
            assert!(caps.avx);
        }

        // If AVX is supported, SSE should also be supported
        if caps.avx {
            assert!(caps.sse4_1);
            assert!(caps.sse2);
        }

        // ARM and x86 SIMD are mutually exclusive
        #[cfg(target_arch = "x86_64")]
        {
            assert!(!caps.neon);
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(!caps.sse2);
            assert!(!caps.avx);
            assert!(!caps.avx2);
        }
    }

    #[test]
    fn bench_simd_detection() {
        let start = Instant::now();

        // SIMD detection should be very fast
        for _ in 0..1000 {
            let _caps = SimdCapabilities::detect();
        }

        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(100)); // Should complete in < 100ms
    }
}

/// Test memory detection and validation
#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_detection() {
        let memory = MemoryInfo::detect();

        // Should have some reasonable memory values
        assert!(memory.total_memory > 0);
        assert!(memory.available_memory > 0);
        assert!(memory.page_size > 0);

        // Available shouldn't exceed total
        assert!(memory.available_memory <= memory.total_memory);

        // Page size should be a power of 2
        assert!(memory.page_size.is_power_of_two());

        // Memory values should be reasonable (at least 128MB total)
        assert!(memory.total_memory >= 128 * 1024 * 1024);
    }

    #[test]
    fn test_memory_info_platform_specific() {
        let memory = MemoryInfo::detect();

        #[cfg(target_arch = "wasm32")]
        {
            assert_eq!(memory.page_size, 65536); // WASM page size
            assert!(memory.total_memory <= 2u64 << 30); // Should be <= 2GB for WASM
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // For non-WASM, page size is typically 4KB
            assert!(memory.page_size >= 4096);
            // Should have more than 1GB total memory on real systems
            assert!(memory.total_memory >= 1u64 << 30);
        }
    }

    #[test]
    fn test_memory_detection_consistency() {
        // Multiple calls should return consistent results
        let mem1 = MemoryInfo::detect();
        let mem2 = MemoryInfo::detect();

        // Total memory should be the same
        assert_eq!(mem1.total_memory, mem2.total_memory);
        assert_eq!(mem1.page_size, mem2.page_size);

        // Available memory might change slightly, but should be close
        let diff = if mem1.available_memory > mem2.available_memory {
            mem1.available_memory - mem2.available_memory
        } else {
            mem2.available_memory - mem1.available_memory
        };

        // Difference should be less than 10% of total memory
        assert!(diff < mem1.total_memory / 10);
    }
}

/// Test runtime capabilities integration
#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn test_runtime_capabilities() {
        let caps = RuntimeCapabilities::detect();

        // Should have valid SIMD capabilities
        let simd_level = caps.simd_support.best_simd_level();
        assert!(simd_level >= SimdLevel::Scalar);

        // Memory info should be valid
        assert!(caps.memory_info.total_memory > 0);
        assert!(caps.memory_info.page_size > 0);

        // GPU capabilities should be initialized (even if no GPU)
        assert!(caps.gpu_support.gpu_memory >= 0);
    }

    #[test]
    fn test_get_runtime_capabilities() {
        let caps1 = get_runtime_capabilities();
        let caps2 = get_runtime_capabilities();

        // Should be consistent
        assert_eq!(caps1.simd_support.sse2, caps2.simd_support.sse2);
        assert_eq!(caps1.memory_info.total_memory, caps2.memory_info.total_memory);
    }
}

/// Test engine registry and lifecycle
#[cfg(test)]
mod engine_tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_registry() {
        let config = EngineConfig::default();
        let engine1 = initialize_engine("test1", config.clone()).unwrap();
        let engine2 = get_engine("test1").unwrap();

        assert!(Arc::ptr_eq(&engine1, &engine2));

        // Test that different names create different engines
        let engine3 = initialize_engine("test2", config).unwrap();
        assert!(!Arc::ptr_eq(&engine1, &engine3));

        // Cleanup
        shutdown_engines();

        // After shutdown, engine should not be available
        assert!(get_engine("test1").is_none());
    }

    #[tokio::test]
    async fn test_engine_warmup() {
        let config = EngineConfig::default();
        let engine = initialize_engine("warmup_test", config).unwrap();

        // Warmup should complete without errors
        let result = warmup_engine(&engine).await;
        assert!(result.is_ok());

        shutdown_engines();
    }

    #[test]
    fn test_concurrent_engine_access() {
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let config = EngineConfig::default();
                    let name = format!("concurrent_test_{}", i);
                    initialize_engine(&name, config)
                })
            })
            .collect();

        let engines: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();

        // All engines should be successfully created
        assert_eq!(engines.len(), 10);
        for engine in engines {
            assert!(engine.is_ok());
        }

        shutdown_engines();
    }

    #[test]
    fn test_engine_stats() {
        let config1 = EngineConfig::default();
        let config2 = EngineConfig::default();

        let _engine1 = initialize_engine("stats_test_1", config1).unwrap();
        let _engine2 = initialize_engine("stats_test_2", config2).unwrap();

        let stats = get_engine_stats();
        assert_eq!(stats.len(), 2);

        let names: Vec<String> = stats.iter().map(|(name, _)| name.clone()).collect();
        assert!(names.contains(&"stats_test_1".to_string()));
        assert!(names.contains(&"stats_test_2".to_string()));

        shutdown_engines();
    }

    #[test]
    fn test_engine_duplicate_names() {
        let config1 = EngineConfig::default();
        let config2 = EngineConfig::default();

        // Create engine with name "duplicate"
        let engine1 = initialize_engine("duplicate", config1).unwrap();

        // Try to create another engine with same name
        let engine2 = initialize_engine("duplicate", config2).unwrap();

        // Should return the same engine instance
        assert!(Arc::ptr_eq(&engine1, &engine2));

        shutdown_engines();
    }
}

/// Stress tests and performance validation
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn stress_test_engine_creation_destruction() {
        const NUM_ITERATIONS: usize = 100;

        for i in 0..NUM_ITERATIONS {
            let config = EngineConfig::default();
            let name = format!("stress_test_{}", i);

            let engine = initialize_engine(&name, config).unwrap();
            assert!(get_engine(&name).is_some());

            // Simulate some work
            tokio::time::sleep(Duration::from_millis(1)).await;

            shutdown_engines();
            assert!(get_engine(&name).is_none());
        }
    }

    #[test]
    fn stress_test_concurrent_detection() {
        use std::thread;

        // Test that capability detection is thread-safe
        let handles: Vec<_> = (0..50)
            .map(|_| {
                thread::spawn(|| {
                    let caps = RuntimeCapabilities::detect();
                    caps.simd_support.best_simd_level()
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();

        // All results should be the same
        let first_result = results[0];
        for result in results {
            assert_eq!(result, first_result);
        }
    }

    #[tokio::test]
    async fn stress_test_memory_allocation() {
        let caps = RuntimeCapabilities::detect();
        let available_memory = caps.memory_info.available_memory;

        // Try to allocate and deallocate memory rapidly
        for _ in 0..100 {
            let size = (available_memory / 1000).min(1024 * 1024); // Max 1MB per allocation
            let vec: Vec<u8> = vec![0; size as usize];

            // Use the memory to prevent optimization
            let sum: u64 = vec.iter().map(|&x| x as u64).sum();
            assert_eq!(sum, 0); // All zeros

            tokio::task::yield_now().await;
        }
    }
}

/// Property-based tests using proptest
#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_simd_level_ordering(level1: SimdLevel, level2: SimdLevel) {
            // Test that ordering is consistent
            if level1 < level2 {
                assert!(level2 > level1);
            }
            if level1 == level2 {
                assert!(!(level1 < level2));
                assert!(!(level1 > level2));
            }
        }

        #[test]
        fn prop_memory_info_consistency(
            total in 1u64..u64::MAX,
            available in 0u64..u64::MAX,
            page_size in 1u64..65536u64
        ) {
            // Create a synthetic MemoryInfo for testing
            let memory = MemoryInfo {
                total_memory: total,
                available_memory: available.min(total),
                page_size: page_size.next_power_of_two(),
            };

            // Properties that should always hold
            assert!(memory.available_memory <= memory.total_memory);
            assert!(memory.page_size.is_power_of_two());
            assert!(memory.total_memory > 0);
            assert!(memory.page_size > 0);
        }

        #[test]
        fn prop_engine_name_handling(name in "\\PC*") {
            // Engine names should be handled consistently
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                let config = EngineConfig::default();
                let result = initialize_engine(&name, config);

                match result {
                    Ok(_) => {
                        // If successful, should be retrievable
                        assert!(get_engine(&name).is_some());
                        shutdown_engines();
                    }
                    Err(_) => {
                        // Error is acceptable for edge cases
                    }
                }
            }
        }
    }
}

/// Error handling and edge case tests
#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_error_handling_in_engine_init() {
        // Test with various configurations
        let config = EngineConfig::default();

        // Empty name should be handled gracefully
        let result = initialize_engine("", config.clone());
        // Should either succeed or return a proper error
        match result {
            Ok(_) => {
                shutdown_engines();
            }
            Err(e) => {
                let error_str = format!("{}", e);
                assert!(!error_str.is_empty());
            }
        }
    }

    #[test]
    fn test_get_nonexistent_engine() {
        // Getting a non-existent engine should return None
        assert!(get_engine("nonexistent").is_none());
        assert!(get_engine("").is_none());
    }

    #[test]
    fn test_double_shutdown() {
        let config = EngineConfig::default();
        let _engine = initialize_engine("double_shutdown_test", config).unwrap();

        // First shutdown should work
        shutdown_engines();
        assert!(get_engine("double_shutdown_test").is_none());

        // Second shutdown should be safe
        shutdown_engines();
        assert!(get_engine("double_shutdown_test").is_none());
    }

    #[test]
    fn test_engine_operations_after_shutdown() {
        let config = EngineConfig::default();
        let _engine = initialize_engine("shutdown_test", config).unwrap();

        shutdown_engines();

        // Operations after shutdown should be safe
        assert!(get_engine("shutdown_test").is_none());
        let stats = get_engine_stats();
        assert!(stats.is_empty());
    }
}

/// Documentation tests embedded in the code
#[cfg(test)]
mod doc_tests {
    use super::*;

    /// Test that the example in the module documentation works
    #[tokio::test]
    async fn test_documentation_example() {
        // This follows the pattern shown in the lib.rs documentation
        let config = EngineConfig::default();
        let engine = initialize_engine("doc_test", config).unwrap();

        // Get runtime capabilities
        let caps = get_runtime_capabilities();
        assert!(caps.simd_support.best_simd_level() >= SimdLevel::Scalar);

        // Warmup
        let result = warmup_engine(&engine).await;
        assert!(result.is_ok());

        // Get stats
        let stats = get_engine_stats();
        assert_eq!(stats.len(), 1);

        shutdown_engines();
    }

    #[test]
    fn test_simd_level_display() {
        // Test that SIMD levels can be displayed
        let scalar = SimdLevel::Scalar;
        let display_str = format!("{:?}", scalar);
        assert!(display_str.contains("Scalar"));
    }

    #[test]
    fn test_capabilities_clone() {
        // Test that capabilities can be cloned
        let caps1 = RuntimeCapabilities::detect();
        let caps2 = caps1.clone();

        assert_eq!(caps1.simd_support.sse2, caps2.simd_support.sse2);
        assert_eq!(caps1.memory_info.total_memory, caps2.memory_info.total_memory);
    }
}