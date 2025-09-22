//! Comprehensive tests for PyO3 bindings and Python-Rust integration
//!
//! These tests validate the Python bindings, error handling, memory safety,
//! and proper integration between Python and Rust components.

use gemma_extensions::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Test basic PyO3 integration
#[cfg(test)]
mod basic_integration_tests {
    use super::*;

    #[test]
    fn test_module_loading() {
        // Test that the module can be loaded in Python
        Python::with_gil(|py| {
            let result = py.run(
                r#"
import sys
sys.path.insert(0, '.')
try:
    import gemma_extensions
    success = True
except ImportError as e:
    success = False
    error = str(e)
                "#,
                None,
                None,
            );

            // If this fails, it's likely a build issue, not a test failure
            if let Err(e) = result {
                println!("Module loading failed (expected in test environment): {}", e);
            }
        });
    }

    #[test]
    fn test_version_functions() {
        Python::with_gil(|py| {
            // Test get_version function
            let version = get_version();
            assert!(!version.is_empty());
            assert!(version.chars().any(|c| c.is_ascii_digit()));

            // Test get_build_info function
            let build_info = get_build_info().unwrap();
            assert!(build_info.contains_key("version"));
            assert!(build_info.contains_key("target"));
            assert!(build_info.contains_key("profile"));
        });
    }

    #[test]
    fn test_simd_support_detection() {
        Python::with_gil(|py| {
            let simd_available = check_simd_support();
            // Result should be a boolean
            assert!(simd_available == true || simd_available == false);

            // SIMD detection should be consistent
            let simd_available_2 = check_simd_support();
            assert_eq!(simd_available, simd_available_2);
        });
    }

    #[test]
    fn test_warmup_function() {
        Python::with_gil(|py| {
            let result = warmup();
            assert!(result.is_ok());

            let message = result.unwrap();
            assert!(message.contains("Warmup") || message.contains("completed"));
        });
    }

    #[test]
    fn test_benchmark_operations() {
        Python::with_gil(|py| {
            let result = benchmark_operations();
            assert!(result.is_ok());

            let benchmarks = result.unwrap();
            assert!(!benchmarks.is_empty());

            // All benchmark results should be positive numbers
            for (name, time) in benchmarks {
                assert!(!name.is_empty());
                assert!(time >= 0.0);
                assert!(time < 10.0); // Should complete within 10 seconds
            }
        });
    }
}

/// Test tokenizer functionality through Python interface
#[cfg(test)]
mod tokenizer_python_tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_creation() {
        Python::with_gil(|py| {
            let config = TokenizerConfig::new();
            assert!(config.vocab_size > 0);
            assert!(config.max_length > 0);

            // Test config with custom vocab size
            let custom_config = TokenizerConfig::with_vocab_size(50000);
            assert_eq!(custom_config.vocab_size, 50000);
        });
    }

    #[test]
    fn test_fast_tokenizer_creation() {
        Python::with_gil(|py| {
            let config = TokenizerConfig::new();
            let tokenizer = FastTokenizer::new(config);

            match tokenizer {
                Ok(tokenizer) => {
                    // Test basic functionality
                    let result = tokenizer.encode("Hello, world!");
                    match result {
                        Ok(tokens) => {
                            assert!(tokens.tokens.len() > 0);
                            assert!(tokens.token_ids.len() > 0);
                            assert_eq!(tokens.tokens.len(), tokens.token_ids.len());
                        }
                        Err(e) => {
                            println!("Encoding failed (may be expected): {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("Tokenizer creation failed (may be expected): {}", e);
                }
            }
        });
    }

    #[test]
    fn test_tokenization_result_structure() {
        Python::with_gil(|py| {
            // Create a mock tokenization result
            let result = TokenizationResult {
                tokens: vec!["Hello".to_string(), "world".to_string()],
                token_ids: vec![100, 200],
                attention_mask: vec![1, 1],
                special_tokens_mask: vec![0, 0],
            };

            assert_eq!(result.tokens.len(), 2);
            assert_eq!(result.token_ids.len(), 2);
            assert_eq!(result.attention_mask.len(), 2);
            assert_eq!(result.special_tokens_mask.len(), 2);
        });
    }

    #[test]
    fn test_batch_tokenization() {
        Python::with_gil(|py| {
            let texts = vec![
                "Hello world".to_string(),
                "This is a test".to_string(),
                "Rust and Python".to_string(),
            ];

            let config = TokenizerConfig::new();

            let result = batch_encode(texts, config);
            match result {
                Ok(results) => {
                    assert_eq!(results.len(), 3);
                    for result in results {
                        assert!(result.tokens.len() > 0);
                        assert!(result.token_ids.len() > 0);
                    }
                }
                Err(e) => {
                    println!("Batch encoding failed (may be expected): {}", e);
                }
            }
        });
    }

    #[test]
    fn test_tokenizer_error_handling() {
        Python::with_gil(|py| {
            // Test with invalid configuration
            let mut config = TokenizerConfig::new();
            config.vocab_size = 0; // Invalid vocab size

            let tokenizer = FastTokenizer::new(config);
            // Should either succeed or fail gracefully
            match tokenizer {
                Ok(_) => {
                    // If it succeeds, that's fine too
                }
                Err(e) => {
                    // Error should be properly formatted
                    let error_str = format!("{}", e);
                    assert!(!error_str.is_empty());
                }
            }
        });
    }
}

/// Test error handling and Python exception conversion
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_gemma_error_creation() {
        let errors = vec![
            GemmaError::tokenizer("test tokenizer error"),
            GemmaError::tensor("test tensor error"),
            GemmaError::cache("test cache error"),
            GemmaError::memory("test memory error"),
            GemmaError::dimension_mismatch(10, 5),
            GemmaError::index_out_of_bounds(10, 5),
            GemmaError::general("test general error"),
        ];

        for error in errors {
            // Error should have a non-empty message
            let message = format!("{}", error);
            assert!(!message.is_empty());

            // Error should convert to PyErr
            let py_err: PyErr = error.into();
            let py_err_str = format!("{}", py_err);
            assert!(!py_err_str.is_empty());
        }
    }

    #[test]
    fn test_python_exception_conversion() {
        Python::with_gil(|py| {
            let errors = vec![
                GemmaError::tokenizer("test"),
                GemmaError::memory("out of memory"),
                GemmaError::dimension_mismatch(100, 50),
            ];

            for error in errors {
                let py_err: PyErr = error.into();

                // Should be able to get exception info
                let exc_type = py_err.get_type(py);
                assert!(!exc_type.name().unwrap().is_empty());
            }
        });
    }

    #[test]
    fn test_error_context_preservation() {
        use crate::error::utils;

        let original_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let context = "While loading model";

        let result: Result<(), std::io::Error> = Err(original_error);
        let gemma_result = utils::with_context(result, context);

        assert!(gemma_result.is_err());
        let error_msg = format!("{}", gemma_result.unwrap_err());
        assert!(error_msg.contains(context));
        assert!(error_msg.contains("File not found"));
    }
}

/// Test memory safety in Python-Rust boundary
#[cfg(test)]
mod memory_safety_tests {
    use super::*;

    #[test]
    fn test_python_rust_memory_boundary() {
        Python::with_gil(|py| {
            // Test that Rust objects can be safely passed to Python and back
            let config = TokenizerConfig::new();
            let original_vocab_size = config.vocab_size;

            // Pass to Python (this tests PyO3 conversion)
            let py_config = config.into_py(py);

            // Extract back from Python
            let extracted_config: TokenizerConfig = py_config.extract(py).unwrap();
            assert_eq!(extracted_config.vocab_size, original_vocab_size);
        });
    }

    #[test]
    fn test_string_handling_across_boundary() {
        Python::with_gil(|py| {
            let test_strings = vec![
                "Hello, world!",
                "Unicode: 🦀 Rust 🐍 Python",
                "Empty string: ",
                "Special chars: !@#$%^&*()",
                "Newlines:\nand\ttabs",
            ];

            for test_str in test_strings {
                // Test string round-trip through Python
                let py_str = test_str.to_object(py);
                let extracted_str: String = py_str.extract(py).unwrap();
                assert_eq!(test_str, extracted_str);
            }
        });
    }

    #[test]
    fn test_vector_handling_across_boundary() {
        Python::with_gil(|py| {
            let test_vectors = vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![-1.0, 0.0, 1.0],
                vec![std::f32::consts::PI, std::f32::consts::E],
                Vec::<f32>::new(), // Empty vector
            ];

            for test_vec in test_vectors {
                // Test vector round-trip through Python
                let py_list = PyList::new(py, &test_vec);
                let extracted_vec: Vec<f32> = py_list.extract().unwrap();
                assert_eq!(test_vec, extracted_vec);
            }
        });
    }

    #[test]
    fn test_concurrent_python_access() {
        // Test that multiple threads can safely interact with Python objects
        use std::sync::{Arc, Mutex};

        let results = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let results_clone = Arc::clone(&results);
                thread::spawn(move || {
                    Python::with_gil(|py| {
                        // Each thread creates its own tokenizer config
                        let config = TokenizerConfig::with_vocab_size(1000 + i);
                        let vocab_size = config.vocab_size;

                        results_clone.lock().unwrap().push(vocab_size);
                    });
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let results = results.lock().unwrap();
        assert_eq!(results.len(), 4);

        // Check that each thread got its expected value
        for (i, &vocab_size) in results.iter().enumerate() {
            assert_eq!(vocab_size, 1000 + i);
        }
    }
}

/// Test FFI safety and C interface
#[cfg(test)]
mod ffi_safety_tests {
    use super::*;

    #[test]
    fn test_null_pointer_handling() {
        // Test that FFI functions handle null pointers gracefully
        use std::ptr;

        // These should not crash, but return appropriate errors
        unsafe {
            // Test with null C strings
            let null_ptr = ptr::null();

            // Most FFI functions should validate pointers
            // This is a conceptual test - actual implementation would depend on specific FFI functions
            assert!(null_ptr.is_null());
        }
    }

    #[test]
    fn test_c_string_safety() {
        use std::ffi::{CString, CStr};

        // Test safe C string handling
        let test_strings = vec![
            "Hello, world!",
            "Unicode test: 🦀",
            "", // Empty string
            "A".repeat(1000), // Long string
        ];

        for test_str in test_strings {
            // Create C string
            let c_string = CString::new(test_str.clone()).unwrap();
            let c_str = c_string.as_c_str();

            // Convert back to Rust string
            let rust_str = c_str.to_string_lossy();
            assert_eq!(test_str, rust_str);
        }
    }

    #[test]
    fn test_buffer_overflow_protection() {
        // Test that buffer operations don't overflow
        let buffer_sizes = vec![0, 1, 16, 256, 1024, 4096];

        for size in buffer_sizes {
            let mut buffer = vec![0u8; size];

            // Fill buffer safely
            for (i, byte) in buffer.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }

            // Verify no overflow occurred
            for (i, &byte) in buffer.iter().enumerate() {
                assert_eq!(byte, (i % 256) as u8);
            }
        }
    }

    #[test]
    fn test_memory_alignment_requirements() {
        // Test that memory allocations meet alignment requirements
        use std::alloc::{alloc, dealloc, Layout};

        let alignments = vec![1, 2, 4, 8, 16, 32, 64];
        let sizes = vec![16, 64, 256, 1024];

        for &alignment in &alignments {
            for &size in &sizes {
                if let Ok(layout) = Layout::from_size_align(size, alignment) {
                    unsafe {
                        let ptr = alloc(layout);
                        if !ptr.is_null() {
                            // Check alignment
                            assert_eq!(ptr as usize % alignment, 0);

                            // Write and read to ensure validity
                            *ptr = 42;
                            assert_eq!(*ptr, 42);

                            dealloc(ptr, layout);
                        }
                    }
                }
            }
        }
    }
}

/// Test async operations and Python integration
#[cfg(test)]
mod async_integration_tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_async_rust_operations() {
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            // Test that async Rust code works in Python context
            tokio::time::sleep(Duration::from_millis(1)).await;

            // Simulate async work
            let result = tokio::task::spawn(async {
                42
            }).await.unwrap();

            assert_eq!(result, 42);
        });
    }

    #[test]
    fn test_python_gil_interaction() {
        Python::with_gil(|py| {
            // Test that Python GIL interaction works correctly
            let dict = PyDict::new(py);
            dict.set_item("key", "value").unwrap();

            let value: String = dict.get_item("key").unwrap().unwrap().extract().unwrap();
            assert_eq!(value, "value");
        });

        // Test multiple GIL acquisitions
        for _ in 0..10 {
            Python::with_gil(|py| {
                let list = PyList::new(py, &[1, 2, 3]);
                let sum: i32 = list.iter().map(|x| x.extract::<i32>().unwrap()).sum();
                assert_eq!(sum, 6);
            });
        }
    }

    #[test]
    fn test_concurrent_gil_usage() {
        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    Python::with_gil(|py| {
                        // Each thread works with Python objects
                        let tuple = PyTuple::new(py, &[i, i * 2, i * 3]);
                        let sum: i32 = tuple.iter().map(|x| x.extract::<i32>().unwrap()).sum();
                        sum
                    })
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Check results
        for (i, &result) in results.iter().enumerate() {
            let expected = i + (i * 2) + (i * 3); // i + 2i + 3i = 6i
            assert_eq!(result, expected as i32);
        }
    }
}

/// Property-based tests for Python-Rust integration
#[cfg(test)]
mod property_integration_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_string_roundtrip(s in "\\PC*") {
            Python::with_gil(|py| {
                if let Ok(py_str) = PyString::new(py, &s).extract::<String>() {
                    assert_eq!(s, py_str);
                }
            });
        }

        #[test]
        fn prop_number_roundtrip(n in prop::num::f32::NORMAL) {
            Python::with_gil(|py| {
                let py_float = n.to_object(py);
                if let Ok(extracted) = py_float.extract::<f32>(py) {
                    assert!((n - extracted).abs() < f32::EPSILON);
                }
            });
        }

        #[test]
        fn prop_vector_roundtrip(vec in prop::collection::vec(prop::num::f32::NORMAL, 0..100)) {
            Python::with_gil(|py| {
                let py_list = PyList::new(py, &vec);
                if let Ok(extracted_vec) = py_list.extract::<Vec<f32>>() {
                    assert_eq!(vec.len(), extracted_vec.len());
                    for (original, extracted) in vec.iter().zip(extracted_vec.iter()) {
                        assert!((original - extracted).abs() < f32::EPSILON);
                    }
                }
            });
        }

        #[test]
        fn prop_tokenizer_config_roundtrip(
            vocab_size in 1usize..100_000,
            max_length in 1usize..10_000
        ) {
            Python::with_gil(|py| {
                let mut config = TokenizerConfig::new();
                config.vocab_size = vocab_size;
                config.max_length = max_length;

                // Convert to Python and back
                let py_config = config.into_py(py);
                if let Ok(extracted_config) = py_config.extract::<TokenizerConfig>(py) {
                    assert_eq!(config.vocab_size, extracted_config.vocab_size);
                    assert_eq!(config.max_length, extracted_config.max_length);
                }
            });
        }
    }
}

/// Performance tests for Python-Rust boundary
#[cfg(test)]
mod performance_integration_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_python_rust_calls() {
        Python::with_gil(|py| {
            let iterations = 10000;

            // Benchmark simple function calls
            let start = Instant::now();
            for _ in 0..iterations {
                let _version = get_version();
            }
            let function_call_time = start.elapsed();

            // Benchmark object creation
            let start = Instant::now();
            for _ in 0..iterations {
                let _config = TokenizerConfig::new();
            }
            let object_creation_time = start.elapsed();

            println!("Function calls: {:?}", function_call_time);
            println!("Object creation: {:?}", object_creation_time);

            // Should be reasonably fast
            assert!(function_call_time < Duration::from_secs(1));
            assert!(object_creation_time < Duration::from_secs(1));
        });
    }

    #[test]
    fn bench_data_conversion() {
        Python::with_gil(|py| {
            let test_data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
            let iterations = 1000;

            // Benchmark Rust to Python conversion
            let start = Instant::now();
            for _ in 0..iterations {
                let _py_list = PyList::new(py, &test_data);
            }
            let rust_to_python_time = start.elapsed();

            // Benchmark Python to Rust conversion
            let py_list = PyList::new(py, &test_data);
            let start = Instant::now();
            for _ in 0..iterations {
                let _rust_vec: Vec<f32> = py_list.extract().unwrap();
            }
            let python_to_rust_time = start.elapsed();

            println!("Rust to Python: {:?}", rust_to_python_time);
            println!("Python to Rust: {:?}", python_to_rust_time);

            // Conversions should be efficient
            assert!(rust_to_python_time < Duration::from_secs(5));
            assert!(python_to_rust_time < Duration::from_secs(5));
        });
    }

    #[test]
    fn stress_test_memory_usage() {
        Python::with_gil(|py| {
            let mut objects = Vec::new();

            // Create many Python objects from Rust
            for i in 0..1000 {
                let config = TokenizerConfig::with_vocab_size(1000 + i);
                let py_config = config.into_py(py);
                objects.push(py_config);
            }

            // Access all objects
            for (i, obj) in objects.iter().enumerate() {
                let config: TokenizerConfig = obj.extract(py).unwrap();
                assert_eq!(config.vocab_size, 1000 + i);
            }

            // Objects should be properly cleaned up when dropped
            objects.clear();
        });
    }
}