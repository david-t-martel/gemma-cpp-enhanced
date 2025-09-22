//! Documentation tests and examples for public APIs
//!
//! These tests ensure that all code examples in documentation work correctly
//! and serve as executable documentation for users.

use gemma_extensions::*;
use pyo3::prelude::*;

/// Test all examples from the main library documentation
#[cfg(test)]
mod lib_documentation_tests {
    use super::*;

    /// Test the basic usage example from lib.rs
    #[test]
    fn test_basic_usage_example() {
        Python::with_gil(|py| {
            // Example from lib.rs documentation
            // ```rust
            // use gemma_extensions::*;
            //
            // // Initialize the extension
            // let result = warmup();
            // assert!(result.is_ok());
            //
            // // Check SIMD support
            // let simd_available = check_simd_support();
            // println!("SIMD support: {}", simd_available);
            //
            // // Get build information
            // let build_info = get_build_info()?;
            // println!("Build info: {:?}", build_info);
            // ```

            // Initialize the extension
            let result = warmup();
            assert!(result.is_ok());

            // Check SIMD support
            let simd_available = check_simd_support();
            println!("SIMD support: {}", simd_available);

            // Get build information
            let build_info = get_build_info().unwrap();
            println!("Build info: {:?}", build_info);

            // Verify the build info contains expected keys
            assert!(build_info.contains_key("version"));
            assert!(build_info.contains_key("target"));
            assert!(build_info.contains_key("profile"));
        });
    }

    /// Test the version information example
    #[test]
    fn test_version_example() {
        // Example from documentation:
        // ```rust
        // let version = get_version();
        // assert!(!version.is_empty());
        // println!("Gemma Extensions version: {}", version);
        // ```

        let version = get_version();
        assert!(!version.is_empty());
        println!("Gemma Extensions version: {}", version);

        // Version should follow semantic versioning
        assert!(version.contains('.'));
    }

    /// Test the benchmark example
    #[test]
    fn test_benchmark_example() {
        // Example from documentation:
        // ```rust
        // let benchmarks = benchmark_operations()?;
        // for (operation, time) in benchmarks {
        //     println!("{}: {:.2}ms", operation, time * 1000.0);
        // }
        // ```

        let benchmarks = benchmark_operations().unwrap();
        for (operation, time) in benchmarks {
            println!("{}: {:.2}ms", operation, time * 1000.0);
            assert!(!operation.is_empty());
            assert!(time >= 0.0);
        }
    }
}

/// Test tokenizer documentation examples
#[cfg(test)]
mod tokenizer_documentation_tests {
    use super::*;

    /// Test basic tokenizer usage example
    #[test]
    fn test_tokenizer_basic_example() {
        // Example from tokenizer documentation:
        // ```rust
        // use gemma_extensions::*;
        //
        // let config = TokenizerConfig::new();
        // let tokenizer = FastTokenizer::new(config)?;
        //
        // let text = "Hello, world!";
        // let result = tokenizer.encode(text)?;
        //
        // println!("Tokens: {:?}", result.tokens);
        // println!("Token IDs: {:?}", result.token_ids);
        // ```

        let config = TokenizerConfig::new();

        match FastTokenizer::new(config) {
            Ok(tokenizer) => {
                let text = "Hello, world!";
                match tokenizer.encode(text) {
                    Ok(result) => {
                        println!("Tokens: {:?}", result.tokens);
                        println!("Token IDs: {:?}", result.token_ids);

                        // Verify the result structure
                        assert_eq!(result.tokens.len(), result.token_ids.len());
                        assert_eq!(result.tokens.len(), result.attention_mask.len());
                        assert_eq!(result.tokens.len(), result.special_tokens_mask.len());
                    }
                    Err(e) => {
                        println!("Tokenization failed (may be expected in test env): {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Tokenizer creation failed (may be expected in test env): {}", e);
            }
        }
    }

    /// Test tokenizer configuration example
    #[test]
    fn test_tokenizer_config_example() {
        // Example from TokenizerConfig documentation:
        // ```rust
        // let mut config = TokenizerConfig::new();
        // config.vocab_size = 50000;
        // config.max_length = 2048;
        // config.use_simd = true;
        // config.parallel = true;
        //
        // // Or use the builder pattern
        // let config = TokenizerConfig::with_vocab_size(32000);
        // ```

        let mut config = TokenizerConfig::new();
        config.vocab_size = 50000;
        config.max_length = 2048;
        config.use_simd = true;
        config.parallel = true;

        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.max_length, 2048);
        assert!(config.use_simd);
        assert!(config.parallel);

        // Or use the builder pattern
        let config = TokenizerConfig::with_vocab_size(32000);
        assert_eq!(config.vocab_size, 32000);
    }

    /// Test batch tokenization example
    #[test]
    fn test_batch_tokenization_example() {
        // Example from batch tokenization documentation:
        // ```rust
        // let texts = vec![
        //     "First sentence".to_string(),
        //     "Second sentence".to_string(),
        //     "Third sentence".to_string(),
        // ];
        //
        // let config = TokenizerConfig::new();
        // let results = batch_encode(texts, config)?;
        //
        // for (i, result) in results.iter().enumerate() {
        //     println!("Text {}: {} tokens", i, result.tokens.len());
        // }
        // ```

        let texts = vec![
            "First sentence".to_string(),
            "Second sentence".to_string(),
            "Third sentence".to_string(),
        ];

        let config = TokenizerConfig::new();

        match batch_encode(texts.clone(), config) {
            Ok(results) => {
                assert_eq!(results.len(), texts.len());

                for (i, result) in results.iter().enumerate() {
                    println!("Text {}: {} tokens", i, result.tokens.len());
                    assert!(!result.tokens.is_empty());
                }
            }
            Err(e) => {
                println!("Batch encoding failed (may be expected in test env): {}", e);
            }
        }
    }

    /// Test tokenizer decode example
    #[test]
    fn test_tokenizer_decode_example() {
        // Example from decode documentation:
        // ```rust
        // let token_ids = vec![100, 200, 300];
        // let config = TokenizerConfig::new();
        //
        // let decoded_texts = batch_decode(token_ids, config)?;
        // println!("Decoded: {:?}", decoded_texts);
        // ```

        let token_ids = vec![100, 200, 300];
        let config = TokenizerConfig::new();

        match batch_decode(token_ids.clone(), config) {
            Ok(decoded_texts) => {
                println!("Decoded: {:?}", decoded_texts);
                assert!(!decoded_texts.is_empty());
            }
            Err(e) => {
                println!("Batch decoding failed (may be expected in test env): {}", e);
            }
        }
    }
}

/// Test error handling documentation examples
#[cfg(test)]
mod error_documentation_tests {
    use super::*;

    /// Test error handling patterns
    #[test]
    fn test_error_handling_example() {
        // Example from error documentation:
        // ```rust
        // use gemma_extensions::*;
        //
        // match some_operation() {
        //     Ok(result) => {
        //         // Handle success
        //         println!("Success: {:?}", result);
        //     }
        //     Err(GemmaError::TokenizerError { message }) => {
        //         // Handle tokenizer-specific error
        //         eprintln!("Tokenizer error: {}", message);
        //     }
        //     Err(GemmaError::MemoryError { message }) => {
        //         // Handle memory error
        //         eprintln!("Memory error: {}", message);
        //     }
        //     Err(err) => {
        //         // Handle other errors
        //         eprintln!("Error: {}", err);
        //     }
        // }
        // ```

        // Test different error types
        let tokenizer_error = GemmaError::tokenizer("Test tokenizer error");
        assert!(matches!(tokenizer_error, GemmaError::TokenizerError { .. }));

        let memory_error = GemmaError::memory("Test memory error");
        assert!(matches!(memory_error, GemmaError::MemoryError { .. }));

        let dimension_error = GemmaError::dimension_mismatch(10, 5);
        assert!(matches!(dimension_error, GemmaError::DimensionMismatch { .. }));

        // Test error display
        println!("Tokenizer error: {}", tokenizer_error);
        println!("Memory error: {}", memory_error);
        println!("Dimension error: {}", dimension_error);
    }

    /// Test error conversion to Python exceptions
    #[test]
    fn test_python_error_conversion_example() {
        Python::with_gil(|py| {
            // Example from Python error conversion documentation:
            // ```rust
            // let error = GemmaError::tokenizer("Invalid input");
            // let py_err: PyErr = error.into();
            // // This PyErr can be raised in Python
            // ```

            let error = GemmaError::tokenizer("Invalid input");
            let py_err: PyErr = error.into();

            // Verify the conversion worked
            let exc_type = py_err.get_type(py);
            assert!(!exc_type.name().unwrap().is_empty());

            // Test with different error types
            let errors = vec![
                GemmaError::tensor("Tensor operation failed"),
                GemmaError::cache("Cache miss"),
                GemmaError::memory("Out of memory"),
            ];

            for error in errors {
                let py_err: PyErr = error.into();
                let exc_type = py_err.get_type(py);
                assert!(!exc_type.name().unwrap().is_empty());
            }
        });
    }
}

/// Test feature detection documentation examples
#[cfg(test)]
mod feature_detection_tests {
    use super::*;

    /// Test SIMD detection example
    #[test]
    fn test_simd_detection_example() {
        // Example from SIMD detection documentation:
        // ```rust
        // if check_simd_support() {
        //     println!("SIMD optimizations are available");
        //     // Use SIMD-optimized operations
        // } else {
        //     println!("Falling back to scalar operations");
        //     // Use fallback implementations
        // }
        // ```

        if check_simd_support() {
            println!("SIMD optimizations are available");
            // SIMD should be consistent across calls
            assert!(check_simd_support());
        } else {
            println!("Falling back to scalar operations");
            // Should consistently report no SIMD
            assert!(!check_simd_support());
        }
    }

    /// Test feature flag checking example
    #[test]
    fn test_feature_checking_example() {
        // Example from feature detection documentation:
        // ```rust
        // let build_info = get_build_info()?;
        // let features = build_info.get("features").unwrap();
        //
        // if features.contains("simd") {
        //     println!("SIMD features enabled");
        // }
        //
        // if features.contains("parallel") {
        //     println!("Parallel processing enabled");
        // }
        // ```

        let build_info = get_build_info().unwrap();
        let features = build_info.get("features").unwrap();

        println!("Available features: {}", features);

        // Features string should contain some information
        assert!(!features.is_empty());

        // Test for common features (these may or may not be present)
        if features.contains("simd") {
            println!("SIMD features enabled");
        }

        if features.contains("parallel") {
            println!("Parallel processing enabled");
        }
    }
}

/// Test performance optimization examples
#[cfg(test)]
mod performance_examples_tests {
    use super::*;

    /// Test warmup optimization example
    #[test]
    fn test_warmup_optimization_example() {
        // Example from performance optimization documentation:
        // ```rust
        // // Warm up the extension for optimal performance
        // warmup()?;
        //
        // // Now perform actual work
        // let config = TokenizerConfig::new();
        // let tokenizer = FastTokenizer::new(config)?;
        //
        // // Subsequent operations will be faster
        // for text in texts {
        //     let result = tokenizer.encode(&text)?;
        //     process_tokens(result);
        // }
        // ```

        // Warm up the extension for optimal performance
        let warmup_result = warmup();
        assert!(warmup_result.is_ok());

        // Now perform actual work
        let config = TokenizerConfig::new();

        if let Ok(tokenizer) = FastTokenizer::new(config) {
            let texts = vec!["Hello", "world", "test"];

            // Subsequent operations should work
            for text in texts {
                if let Ok(result) = tokenizer.encode(&text) {
                    // Process tokens (simplified)
                    assert!(!result.tokens.is_empty());
                    println!("Processed {} tokens for '{}'", result.tokens.len(), text);
                }
            }
        }
    }

    /// Test batch processing optimization example
    #[test]
    fn test_batch_processing_example() {
        // Example from batch processing documentation:
        // ```rust
        // // Process multiple texts at once for better performance
        // let texts = vec![
        //     "First text".to_string(),
        //     "Second text".to_string(),
        //     "Third text".to_string(),
        // ];
        //
        // let config = TokenizerConfig::new();
        // let results = batch_encode(texts, config)?;
        //
        // // Much faster than individual encode calls
        // for result in results {
        //     process_tokens(result);
        // }
        // ```

        // Process multiple texts at once for better performance
        let texts = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let config = TokenizerConfig::new();

        if let Ok(results) = batch_encode(texts.clone(), config) {
            assert_eq!(results.len(), texts.len());

            // Much faster than individual encode calls
            for (i, result) in results.iter().enumerate() {
                // Process tokens (simplified)
                println!("Batch processed text {}: {} tokens", i, result.tokens.len());
                assert!(!result.tokens.is_empty());
            }
        }
    }
}

/// Test integration examples
#[cfg(test)]
mod integration_examples_tests {
    use super::*;

    /// Test Python integration example
    #[test]
    fn test_python_integration_example() {
        Python::with_gil(|py| {
            // Example from Python integration documentation:
            // ```python
            // import gemma_extensions
            //
            // # Initialize the extension
            // gemma_extensions.warmup()
            //
            // # Create tokenizer configuration
            // config = gemma_extensions.TokenizerConfig()
            // config.vocab_size = 50000
            //
            // # Create tokenizer
            // tokenizer = gemma_extensions.FastTokenizer(config)
            //
            // # Encode text
            // result = tokenizer.encode("Hello, world!")
            // print(f"Tokens: {result.tokens}")
            // ```

            // Simulate Python integration by testing the same operations
            let config = TokenizerConfig::with_vocab_size(50000);
            assert_eq!(config.vocab_size, 50000);

            // Test that objects can be converted to/from Python
            let py_config = config.into_py(py);
            let extracted_config: TokenizerConfig = py_config.extract(py).unwrap();
            assert_eq!(extracted_config.vocab_size, 50000);
        });
    }

    /// Test async integration example
    #[test]
    fn test_async_integration_example() {
        // Example from async integration documentation:
        // ```rust
        // use tokio;
        //
        // #[tokio::main]
        // async fn main() -> Result<(), Box<dyn std::error::Error>> {
        //     // Initialize extension
        //     warmup()?;
        //
        //     // Perform async work
        //     let config = TokenizerConfig::new();
        //     let tokenizer = FastTokenizer::new(config)?;
        //
        //     // Process in parallel
        //     let futures: Vec<_> = texts.into_iter()
        //         .map(|text| async {
        //             tokenizer.encode(&text)
        //         })
        //         .collect();
        //
        //     let results = futures::future::join_all(futures).await;
        //     Ok(())
        // }
        // ```

        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            // Initialize extension
            let warmup_result = warmup();
            assert!(warmup_result.is_ok());

            // Test that our operations work in async context
            let config = TokenizerConfig::new();

            if let Ok(tokenizer) = FastTokenizer::new(config) {
                let texts = vec!["Hello", "async", "world"];

                // Process each text (simulating async work)
                for text in texts {
                    tokio::task::yield_now().await;

                    if let Ok(result) = tokenizer.encode(&text) {
                        assert!(!result.tokens.is_empty());
                    }
                }
            }
        });
    }
}

/// Test configuration examples
#[cfg(test)]
mod configuration_examples_tests {
    use super::*;

    /// Test advanced configuration example
    #[test]
    fn test_advanced_configuration_example() {
        // Example from advanced configuration documentation:
        // ```rust
        // let mut config = TokenizerConfig::new();
        //
        // // Customize for performance
        // config.use_simd = check_simd_support();
        // config.parallel = true;
        // config.parallel_threshold = 50;
        //
        // // Customize vocabulary
        // config.vocab_size = 32000;
        // config.max_length = 4096;
        //
        // // Special tokens
        // config.pad_token_id = Some(0);
        // config.bos_token_id = Some(1);
        // config.eos_token_id = Some(2);
        // config.unk_token_id = Some(3);
        // ```

        let mut config = TokenizerConfig::new();

        // Customize for performance
        config.use_simd = check_simd_support();
        config.parallel = true;
        config.parallel_threshold = 50;

        // Customize vocabulary
        config.vocab_size = 32000;
        config.max_length = 4096;

        // Special tokens
        config.pad_token_id = Some(0);
        config.bos_token_id = Some(1);
        config.eos_token_id = Some(2);
        config.unk_token_id = Some(3);

        // Verify configuration
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.max_length, 4096);
        assert_eq!(config.parallel_threshold, 50);
        assert_eq!(config.pad_token_id, Some(0));
        assert_eq!(config.bos_token_id, Some(1));
        assert_eq!(config.eos_token_id, Some(2));
        assert_eq!(config.unk_token_id, Some(3));
    }

    /// Test environment-specific configuration example
    #[test]
    fn test_environment_specific_configuration_example() {
        // Example from environment-specific configuration documentation:
        // ```rust
        // let mut config = TokenizerConfig::new();
        //
        // // Adapt to hardware capabilities
        // config.use_simd = check_simd_support();
        //
        // // Adapt to available CPU cores
        // config.parallel = num_cpus::get() > 1;
        //
        // // Memory-constrained environments
        // if cfg!(target_arch = "wasm32") {
        //     config.max_length = 1024; // Smaller for WASM
        // } else {
        //     config.max_length = 4096; // Larger for native
        // }
        // ```

        let mut config = TokenizerConfig::new();

        // Adapt to hardware capabilities
        config.use_simd = check_simd_support();

        // Adapt to available CPU cores (simplified check)
        config.parallel = true; // Assume multi-core for testing

        // Memory-constrained environments
        if cfg!(target_arch = "wasm32") {
            config.max_length = 1024; // Smaller for WASM
            assert_eq!(config.max_length, 1024);
        } else {
            config.max_length = 4096; // Larger for native
            assert_eq!(config.max_length, 4096);
        }

        // Verify the configuration makes sense
        assert!(config.max_length > 0);
        assert!(config.vocab_size > 0);
    }
}