# RAG-Redis System Test Suite

This document describes the comprehensive test suite created for the RAG-Redis system.

## Overview

The test suite provides thorough coverage of all system components including:
- Integration tests for complete system workflows
- Unit tests for individual components
- Performance benchmarks
- Error handling and edge cases
- Mock implementations for testing without external dependencies

## Test Structure

```
tests/
├── integration_test.rs     # End-to-end system tests
├── redis_test.rs          # Redis backend tests with mocking
├── vector_test.rs         # Vector store and search tests
├── mcp_test.rs           # MCP server protocol tests
├── basic_test.rs         # Basic functionality tests
└── lib.rs               # Test runner

benches/
├── vector_search.rs      # Vector search performance benchmarks
└── document_processing.rs # Document processing benchmarks
```

## Test Categories

### 1. Integration Tests (`tests/integration_test.rs`)

**Purpose:** Validate complete system workflows from document ingestion to search

**Key Test Areas:**
- System initialization and configuration
- Document ingestion pipeline (text → chunks → vectors → storage)
- Search functionality across different content types
- Research capabilities (local + web search combination)
- Error handling and recovery scenarios
- Concurrent operations and thread safety
- Performance under load

**Test Documents:**
- Technical documentation
- Research papers
- Code-heavy content
- Structured data
- Large documents (10K+ words)

**Example Tests:**
```rust
#[tokio::test]
async fn test_document_ingestion_workflow() {
    // Create system, ingest document, verify storage and search
}

#[tokio::test]
async fn test_concurrent_operations() {
    // Test multiple simultaneous ingestion and search operations
}
```

### 2. Redis Backend Tests (`tests/redis_test.rs`)

**Purpose:** Verify Redis integration with comprehensive mocking

**Key Test Areas:**
- Connection management and pooling
- Document and chunk storage/retrieval
- Embedding caching
- Search result caching
- Health checks and monitoring
- Error handling and retries
- Statistics tracking

**Mock Implementation:**
- `MockRedis` struct simulates Redis behavior
- Configurable failure modes and latency
- Memory-based storage for testing
- Connection counting and monitoring

**Example Tests:**
```rust
#[tokio::test]
async fn test_concurrent_operations() {
    // Test Redis under concurrent load
}

#[tokio::test]
async fn test_error_handling() {
    // Test Redis connection failures and recovery
}
```

### 3. Vector Store Tests (`tests/vector_test.rs`)

**Purpose:** Validate vector operations and similarity search

**Key Test Areas:**
- Vector storage with various dimensions (128, 384, 768, 1536)
- Distance metrics (Cosine, Euclidean, Dot Product, Manhattan)
- SIMD optimizations vs fallback implementations
- Search with filtering and metadata
- Memory efficiency and statistics
- Edge cases (zero vectors, dimension mismatches)

**Property-Based Testing:**
- Uses `proptest` for automated test case generation
- Validates vector operations across parameter ranges
- Ensures mathematical correctness of distance calculations

**Example Tests:**
```rust
#[test]
fn test_distance_metrics() {
    // Test all distance metrics with known vector relationships
}

proptest! {
    #[test]
    fn prop_search_consistency(vectors: Vec<Vec<f32>>) {
        // Verify search results are consistently ordered
    }
}
```

### 4. MCP Server Tests (`tests/mcp_test.rs`)

**Purpose:** Validate Model Context Protocol compliance and functionality

**Key Test Areas:**
- Protocol initialization and handshake
- Tool registration and execution
- Resource management and access
- Error handling and edge cases
- Concurrent message processing
- Performance benchmarks

**Mock MCP Server:**
- Complete MCP protocol implementation
- Tool execution simulation
- Resource management
- Performance monitoring

**Example Tests:**
```rust
#[tokio::test]
async fn test_mcp_protocol_compliance() {
    // Verify MCP messages follow protocol specification
}

#[tokio::test]
async fn test_concurrent_handling() {
    // Test multiple simultaneous MCP requests
}
```

## Benchmark Suite

### 1. Vector Search Benchmarks (`benches/vector_search.rs`)

**Performance Areas:**
- Vector addition throughput
- Search latency across dataset sizes
- Distance metric performance comparison
- SIMD vs fallback implementation speed
- Memory usage efficiency
- Concurrent search operations

**Benchmark Configurations:**
- Dimensions: 128, 384, 768, 1536
- Dataset sizes: 100 to 10,000 vectors
- Search limits: 1 to 50 results
- Distance metrics: All supported types

**Sample Results Expected:**
- Vector addition: ~10,000 vectors/second
- Search latency: <10ms for 1,000 vectors
- SIMD speedup: 2-4x over fallback
- Memory efficiency: <1GB for 10,000 768D vectors

### 2. Document Processing Benchmarks (`benches/document_processing.rs`)

**Performance Areas:**
- Document parsing and preprocessing
- Text chunking strategies
- Concurrent processing throughput
- Memory efficiency during processing
- Format detection speed

**Test Document Types:**
- Research papers (1,000 words)
- Technical documentation (5,000 words)
- Code-heavy content (2,000 words)
- Structured data (3,000 words)
- Large articles (10,000+ words)

**Chunking Methods Tested:**
- Token-based chunking (256, 512, 1024 tokens)
- Character-based chunking (1000, 2000 chars)
- Semantic chunking (experimental)

## Running Tests

### Prerequisites
```bash
# Ensure Redis is available (optional - tests include mocks)
redis-server

# Install Rust dependencies
cargo build
```

### Basic Test Execution
```bash
# Run all tests
cargo test

# Run specific test categories
cargo test integration_test
cargo test redis_test
cargo test vector_test
cargo test mcp_test

# Run with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads 4
```

### Benchmark Execution
```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench vector_search
cargo bench document_processing

# Generate HTML reports
cargo bench -- --output-format html
```

### Advanced Testing Options
```bash
# Test with specific features
cargo test --features "full"
cargo test --features "gpu,metrics"

# Test without default features
cargo test --no-default-features

# Integration tests only
cargo test --test integration_test

# Unit tests only
cargo test --lib
```

## Test Configuration

### Environment Variables
```bash
# Redis connection for integration tests
export REDIS_URL="redis://localhost:6379"

# Enable debug logging
export RUST_LOG="debug"

# Set test timeouts
export TEST_TIMEOUT_SECONDS="30"
```

### Test Data Generation
Tests automatically generate deterministic test data:
- Research papers with known word counts
- Technical documentation with code samples
- Structured data with tables and metrics
- Vector datasets with specific mathematical properties

## Performance Expectations

### Test Execution Times
- Unit tests: <1 second each
- Integration tests: 5-30 seconds each
- Full test suite: <5 minutes
- Benchmark suite: 10-20 minutes

### System Requirements
- Memory: 4GB available RAM minimum
- CPU: Multi-core recommended for parallel tests
- Storage: 1GB free space for test artifacts
- Network: Optional Redis connection

## Continuous Integration

### GitHub Actions Configuration
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:6
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
      - run: cargo test --all-features
      - run: cargo bench --no-run  # Compile benchmarks
```

### Test Coverage
- Target coverage: >85%
- Critical path coverage: >95%
- Generated reports: `target/coverage/`

## Debugging Failed Tests

### Common Issues
1. **Redis Connection Failures**
   - Tests include Redis mocks - should not fail without Redis
   - Check `REDIS_URL` environment variable

2. **Memory Issues**
   - Large dataset tests may require >4GB RAM
   - Use `cargo test -- --test-threads 1` to reduce memory usage

3. **Timeout Issues**
   - Increase timeout with `TEST_TIMEOUT_SECONDS`
   - Run tests sequentially if needed

4. **SIMD Compatibility**
   - SIMD tests automatically fall back on unsupported platforms
   - No action required for compatibility issues

### Debug Output
```bash
# Enable detailed test output
RUST_LOG=debug cargo test -- --nocapture

# Run specific failing test
cargo test test_failing_case -- --exact --nocapture

# Show test statistics
cargo test -- --report-time
```

## Contributing to Tests

### Adding New Tests
1. Follow existing test patterns
2. Include both success and failure cases
3. Add property-based tests for mathematical operations
4. Include performance considerations
5. Document expected behavior

### Test Guidelines
- Tests should be deterministic and repeatable
- Use mocks for external dependencies
- Include edge cases and error conditions
- Validate both behavior and performance
- Maintain compatibility across platforms

## Test Artifacts

### Generated Files
- `target/criterion/`: Benchmark reports and graphs
- `target/coverage/`: Code coverage reports
- `test-results.xml`: JUnit format results
- `benchmark-results.json`: Machine-readable benchmark data

### Log Files
- Test execution logs in `target/test-logs/`
- Performance metrics in `target/perf-logs/`
- Error details in `target/debug.log`

## Conclusion

This comprehensive test suite ensures the RAG-Redis system is thoroughly validated across all components and use cases. The combination of unit tests, integration tests, property-based testing, and performance benchmarks provides confidence in system reliability and performance.

The test suite is designed to:
- ✅ Validate correctness of all operations
- ✅ Ensure performance meets requirements
- ✅ Verify error handling and recovery
- ✅ Test edge cases and boundary conditions
- ✅ Provide regression testing capabilities
- ✅ Enable continuous integration workflows

Regular execution of this test suite is essential for maintaining system quality and preventing regressions during development.
