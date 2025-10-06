# RAG-Redis MCP Server Functional Tests

This directory contains comprehensive functional tests for the RAG-Redis MCP server that perform real operations against a Redis instance.

## Test Structure

### functional_test.rs
A comprehensive standalone test binary that:
- Automatically starts Redis if not running
- Tests all MCP server functionality end-to-end
- Includes performance benchmarks
- Provides detailed test reporting

### integration_test.rs
Unit integration tests that can be run with `cargo test`:
- Redis connection tests
- Document operations
- MCP handler creation
- Performance benchmarks
- System cleanup

## Prerequisites

1. **Redis Server**: Must be installed and accessible
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server

   # macOS
   brew install redis

   # Windows
   # Download from https://redis.io/download
   ```

2. **Rust Environment**: Ensure you have Rust and Cargo installed

## Running Tests

### Option 1: Run the comprehensive functional test
```bash
# From the mcp-server directory
cargo run --bin functional-test
```

### Option 2: Run individual integration tests
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_redis_connection
cargo test test_document_operations
cargo test test_mcp_handler_integration
cargo test test_performance_benchmarks
```

### Option 3: Use the test runner script
```bash
# From the mcp-server directory
./run_functional_tests.sh
```

## Test Coverage

The functional tests cover:

### Core Functionality
- ✅ Redis connection and health checks
- ✅ MCP protocol compliance (initialize, tools/list, ping)
- ✅ Document ingestion with metadata
- ✅ Vector search functionality
- ✅ Hybrid search capabilities
- ✅ Memory operations and statistics
- ✅ Research functionality
- ✅ Document retrieval and listing
- ✅ System health checks

### Performance Testing
- ✅ Document ingestion speed
- ✅ Search latency benchmarks
- ✅ Memory usage monitoring
- ✅ Concurrent operation handling

### Error Handling
- ✅ Invalid document IDs
- ✅ Network failures
- ✅ Invalid search queries
- ✅ Configuration errors

### System Integration
- ✅ Real Redis operations
- ✅ Actual vector embeddings
- ✅ Full MCP protocol stack
- ✅ Resource cleanup

## Test Configuration

Tests use Redis database 9 (`/9`) to avoid conflicts with production data:
```rust
redis_url: "redis://localhost:6379/9"
```

The test suite automatically:
- Starts Redis if not running
- Cleans the test database before tests
- Uses isolated test data
- Provides detailed error reporting

## Expected Output

Successful test run shows:
```
🚀 Starting RAG-Redis MCP Server Functional Tests
============================================================
✅ Redis started successfully
✅ MCP handler initialized successfully
🔌 Testing MCP protocol compliance...
✅ MCP tools list contains 14 tools
✅ MCP resources list completed
✅ MCP ping successful
🏥 Testing health check...
✅ Health check passed
📄 Testing document ingestion...
✅ Document ingested with ID: [uuid]
🔍 Testing vector search...
✅ Vector search found 1 results
✅ Specific search completed
🔀 Testing hybrid search...
✅ Hybrid search completed successfully
🧠 Testing memory operations...
✅ Memory stats retrieved successfully
✅ System metrics retrieved successfully
🔬 Testing research functionality...
✅ Research functionality completed
📚 Testing document operations...
✅ Document retrieval completed
✅ Document listing found 1 documents
============================================================
🎉 ALL FUNCTIONAL TESTS PASSED!
⏱️  Total test duration: [time]
============================================================
```

## Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
redis-cli ping

# Start Redis manually
redis-server

# Check Redis logs
redis-cli monitor
```

### Build Issues
```bash
# Clean and rebuild
cargo clean
cargo build

# Update dependencies
cargo update
```

### Test Failures
- Ensure Redis is accessible on localhost:6379
- Check that no other tests are using the same Redis instance
- Verify sufficient disk space for Redis operations
- Check system resources (memory, CPU)

## Performance Expectations

Typical performance benchmarks:
- Document ingestion: < 100ms per document
- Vector search: < 50ms
- MCP protocol operations: < 10ms
- System initialization: < 2 seconds

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Use database 9 for isolation
3. Include cleanup in test teardown
4. Add performance assertions where appropriate
5. Document expected behavior and error conditions