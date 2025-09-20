//! Test module configuration for gemma-inference
//!
//! This module organizes and configures all tests for the inference engine,
//! ensuring proper test discovery and execution.

// Import all test modules
pub mod lib_tests;
pub mod tensor_tests;
pub mod memory_tests;

// Re-export common test utilities
pub use gemma_inference::*;

/// Common test utilities and helpers
pub mod test_utils {
    use super::*;
    use std::sync::Once;
    use tracing_subscriber;

    static INIT: Once = Once::new();

    /// Initialize test environment (logging, etc.)
    pub fn init_test_env() {
        INIT.call_once(|| {
            // Initialize tracing for tests
            let _ = tracing_subscriber::fmt()
                .with_env_filter("debug")
                .with_test_writer()
                .try_init();
        });
    }

    /// Create a test memory configuration
    pub fn test_memory_config() -> MemoryConfig {
        MemoryConfig {
            max_pool_size: Some(100 * 1024 * 1024), // 100MB limit for tests
            enable_reuse: true,
            alignment: 32,
            ..Default::default()
        }
    }

    /// Create a test engine configuration
    pub fn test_engine_config() -> EngineConfig {
        EngineConfig {
            memory_config: test_memory_config(),
            optimization_level: OptimizationLevel::Debug,
            enable_logging: true,
            ..Default::default()
        }
    }

    /// Generate test data for benchmarks and stress tests
    pub fn generate_test_data(size: usize) -> Vec<f32> {
        (0..size).map(|i| (i as f32) * 0.1).collect()
    }

    /// Generate random test data
    pub fn generate_random_data(size: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Assert that two float vectors are approximately equal
    pub fn assert_vec_approx_eq(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "Vector lengths don't match");

        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tolerance,
                "Values at index {} differ: {} vs {} (tolerance: {})",
                i, x, y, tolerance
            );
        }
    }

    /// Measure execution time of a closure
    pub fn measure_time<F, R>(f: F) -> (R, std::time::Duration)
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Check if we're running in CI environment
    pub fn is_ci() -> bool {
        std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
    }

    /// Skip test if running in CI (for resource-intensive tests)
    pub fn skip_if_ci() {
        if is_ci() {
            println!("Skipping resource-intensive test in CI environment");
            return;
        }
    }

    /// Create a temporary directory for test files
    pub fn create_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("Failed to create temporary directory")
    }

    /// Assert tensor shapes are equal
    pub fn assert_tensor_shape_eq(a: &TensorShape, b: &TensorShape) {
        assert_eq!(a.dims(), b.dims(), "Tensor shapes don't match");
        assert_eq!(a.total_elements(), b.total_elements(), "Tensor element counts don't match");
    }

    /// Create test tensor with specific pattern
    pub fn create_test_tensor(shape: &TensorShape, pattern: TestPattern) -> Tensor {
        let mut tensor = Tensor::zeros(shape, DataType::F32);

        match pattern {
            TestPattern::Zeros => {
                // Already zeros, nothing to do
            }
            TestPattern::Ones => {
                for i in 0..shape.total_elements() {
                    let indices = shape.linear_to_indices(i);
                    tensor.set_f32(&indices, 1.0).unwrap();
                }
            }
            TestPattern::Sequential => {
                for i in 0..shape.total_elements() {
                    let indices = shape.linear_to_indices(i);
                    tensor.set_f32(&indices, i as f32).unwrap();
                }
            }
            TestPattern::Random(seed) => {
                use rand::{Rng, SeedableRng};
                use rand_chacha::ChaCha8Rng;

                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                for i in 0..shape.total_elements() {
                    let indices = shape.linear_to_indices(i);
                    let value: f32 = rng.gen_range(-1.0..1.0);
                    tensor.set_f32(&indices, value).unwrap();
                }
            }
        }

        tensor
    }

    /// Test data patterns
    pub enum TestPattern {
        Zeros,
        Ones,
        Sequential,
        Random(u64), // seed
    }

    /// Verify SIMD correctness by comparing with scalar implementation
    pub fn verify_simd_correctness<F1, F2>(
        scalar_fn: F1,
        simd_fn: F2,
        test_data: &[Vec<f32>],
        tolerance: f32,
    ) where
        F1: Fn(&[f32], &[f32]) -> f32,
        F2: Fn(&[f32], &[f32]) -> f32,
    {
        for (i, data) in test_data.iter().enumerate() {
            if data.len() >= 2 {
                let mid = data.len() / 2;
                let a = &data[..mid];
                let b = &data[mid..];

                if a.len() == b.len() {
                    let scalar_result = scalar_fn(a, b);
                    let simd_result = simd_fn(a, b);

                    assert!(
                        (scalar_result - simd_result).abs() < tolerance,
                        "SIMD result differs from scalar for test case {}: scalar={}, simd={}, diff={}",
                        i, scalar_result, simd_result, (scalar_result - simd_result).abs()
                    );
                }
            }
        }
    }
}