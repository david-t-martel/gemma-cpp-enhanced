//! Basic tests for the RAG-Redis system

use rag_redis_system::{Config, Error};

#[test]
fn test_config_creation() {
    let config = Config::default();
    assert!(config.vector_store.dimension > 0);
    assert!(!config.redis.url.is_empty());
    println!("✓ Config created successfully");
}

#[test]
fn test_config_validation() {
    let valid_config = Config::default();
    assert!(valid_config.validate().is_ok());

    let mut invalid_config = Config::default();
    invalid_config.vector_store.dimension = 0;
    assert!(invalid_config.validate().is_err());

    println!("✓ Config validation working correctly");
}

#[test]
fn test_error_types() {
    let redis_error = Error::Redis("Test Redis error".to_string());
    assert!(matches!(redis_error, Error::Redis(_)));

    let vector_error = Error::VectorStore("Test vector error".to_string());
    assert!(matches!(vector_error, Error::VectorStore(_)));

    println!("✓ Error types working correctly");
}

#[test]
fn test_error_display() {
    let errors = vec![
        Error::Redis("Redis connection failed".to_string()),
        Error::VectorStore("Vector index not found".to_string()),
        Error::Config("Invalid configuration".to_string()),
        Error::InvalidInput("Input validation failed".to_string()),
        Error::NotFound("Resource not found".to_string()),
    ];

    for error in errors {
        let error_string = error.to_string();
        assert!(
            !error_string.is_empty(),
            "Error should have a non-empty display message"
        );
    }

    println!("✓ Error display working correctly");
}
