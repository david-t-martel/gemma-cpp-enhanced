# RAG Redis System - FFI Interface

This document describes the Foreign Function Interface (FFI) for the RAG Redis System, enabling integration with C and C++ applications.

## Overview

The RAG Redis System provides comprehensive FFI bindings that allow C and C++ applications to:

- Ingest and process documents with automatic chunking and embedding
- Perform high-performance vector similarity search
- Conduct research with web search capabilities
- Manage system configuration and monitor performance
- Handle errors safely with detailed error information

## Architecture

The FFI layer consists of three main components:

1. **Rust FFI Implementation** (`src/ffi.rs`) - Core C-compatible interface
2. **C Header** (`include/rag_redis.h`) - Complete C API definitions
3. **C++ Wrapper** (`include/rag_redis.hpp`) - Modern C++ RAII interface

## Features

### Memory Safety
- Automatic resource management through RAII patterns (C++)
- Explicit memory management functions for C
- No memory leaks with proper usage
- Thread-safe operations

### Error Handling
- Structured error codes matching Rust error types
- Detailed error messages
- C++ exceptions with error code information
- Safe error cleanup functions

### Thread Safety
- Multiple RAG system instances can run concurrently
- Internal thread synchronization using Tokio runtime
- Thread-safe string pools for C interop
- Atomic reference counting for shared resources

### Performance
- Minimal overhead FFI layer
- Efficient string conversion and memory management
- Connection pooling for Redis operations
- Async operations with synchronous FFI interface

## Building

### Prerequisites

- Rust 1.70+ with `cargo`
- C compiler (GCC, Clang, or MSVC)
- C++17-compatible compiler for C++ examples
- Redis server (for runtime)
- `cbindgen` for automatic header generation (optional)

### Build Steps

1. **Build the Rust library with FFI support:**
   ```bash
   cargo build --release --features ffi
   ```

2. **Generate C bindings (optional, requires cbindgen):**
   ```bash
   cargo install cbindgen
   cbindgen --config cbindgen.toml --crate rag-redis-system --output include/rag_redis_generated.h
   ```

3. **Build examples:**
   ```bash
   cd examples
   make all
   ```

## Usage

### C API Usage

```c
#include "rag_redis.h"
#include <stdio.h>

int main() {
    // Initialize library
    if (!rag_init()) {
        return -1;
    }

    // Create configuration
    RagConfig config;
    rag_config_default(&config);
    config.redis_url = "redis://127.0.0.1:6379";

    // Create RAG system
    RagErrorInfo error = {0};
    RagHandle handle = rag_create(&config, &error);
    if (handle == 0) {
        printf("Error: %s\n", error.message);
        rag_free_error_message(error.message);
        return -1;
    }

    // Ingest document
    char* doc_id = NULL;
    int result = rag_ingest_document(handle, "Sample content",
                                     "{\"title\": \"Sample\"}", &doc_id, &error);
    if (result) {
        printf("Document ID: %s\n", doc_id);
        rag_free_string(doc_id);
    }

    // Search
    RagSearchResults* results = rag_search(handle, "sample query", 10, &error);
    if (results) {
        printf("Found %u results\n", results->count);
        rag_free_search_results(results);
    }

    // Cleanup
    rag_destroy(handle);
    rag_cleanup();
    return 0;
}
```

### C++ API Usage

```cpp
#include "rag_redis.hpp"
#include <iostream>

int main() {
    try {
        // RAII library initialization
        rag::Library library;

        // Create configuration with fluent interface
        auto config = rag::Config()
            .redis_url("redis://127.0.0.1:6379")
            .vector_dimension(768)
            .document_chunk_size(512);

        // Create RAG system
        rag::System system(config);

        // Ingest document with metadata
        std::string doc_id = system.ingest_document(
            "Sample document content",
            {{"title", "Sample"}, {"author", "User"}}
        );
        std::cout << "Document ID: " << doc_id << std::endl;

        // Search with automatic memory management
        auto results = system.search("sample query", 10);
        std::cout << "Found " << results.size() << " results" << std::endl;

        for (const auto& result : results) {
            std::cout << "Score: " << result.score()
                      << " - " << result.text() << std::endl;
        }

        return 0;
    } catch (const rag::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## API Reference

### Core Functions (C API)

#### Library Management
- `int rag_init(void)` - Initialize library (call once)
- `int rag_cleanup(void)` - Clean up library (call once at exit)
- `const char* rag_version(void)` - Get version string

#### System Management
- `RagHandle rag_create(const RagConfig* config, RagErrorInfo* error_info)` - Create system instance
- `int rag_destroy(RagHandle handle)` - Destroy system instance
- `int rag_health_check(RagHandle handle)` - Check system health

#### Document Operations
- `int rag_ingest_document(RagHandle handle, const char* content, const char* metadata_json, char** document_id, RagErrorInfo* error_info)` - Ingest document

#### Search Operations
- `RagSearchResults* rag_search(RagHandle handle, const char* query, unsigned int limit, RagErrorInfo* error_info)` - Search documents
- `RagSearchResults* rag_research(RagHandle handle, const char* query, const char* const* sources, unsigned int sources_count, RagErrorInfo* error_info)` - Research with web search

#### Information
- `char* rag_get_stats(RagHandle handle, RagErrorInfo* error_info)` - Get system statistics

#### Memory Management
- `void rag_free_search_results(RagSearchResults* results)` - Free search results
- `void rag_free_error_message(char* message)` - Free error message
- `void rag_free_string(char* string)` - Free string

### Core Classes (C++ API)

#### Library Management
- `rag::Library` - RAII library initialization

#### Configuration
- `rag::Config` - Fluent configuration builder
- `rag::create_dev_config()` - Create development configuration
- `rag::create_production_config()` - Create production configuration

#### System
- `rag::System` - Main system class with RAII
- `rag::System::ingest_document()` - Ingest document
- `rag::System::search()` - Search documents
- `rag::System::research()` - Research with web search
- `rag::System::search_async()` - Async search
- `rag::System::get_stats()` - Get statistics

#### Results
- `rag::SearchResult` - Single search result
- `rag::SearchResults` - Result collection with STL compatibility

#### Error Handling
- `rag::Exception` - Exception with error codes
- Error codes: `Success`, `Redis`, `Serialization`, `IO`, `Config`, `Vector`, `Document`, `Embedding`, `Network`, `NotFound`, `InvalidInput`, `Memory`, `Timeout`, `Unknown`

## Configuration Options

### Redis Configuration
- `redis_url` - Redis connection URL
- `redis_pool_size` - Connection pool size (default: 10)
- `redis_connection_timeout_secs` - Connection timeout (default: 5)
- `redis_command_timeout_secs` - Command timeout (default: 10)
- `redis_max_retries` - Max retry attempts (default: 3)
- `redis_retry_delay_ms` - Retry delay in ms (default: 100)
- `redis_enable_cluster` - Enable cluster mode (default: false)

### Vector Store Configuration
- `vector_dimension` - Embedding dimension (default: 768)
- `vector_max_elements` - Max vectors (default: 100,000)
- `vector_m` - HNSW M parameter (default: 16)
- `vector_ef_construction` - HNSW construction parameter (default: 200)
- `vector_ef_search` - HNSW search parameter (default: 50)
- `vector_similarity_threshold` - Min similarity threshold (default: 0.7)

### Document Processing Configuration
- `doc_chunk_size` - Chunk size in tokens (default: 512)
- `doc_chunk_overlap` - Chunk overlap (default: 50)
- `doc_max_chunk_size` - Max chunk size (default: 1024)
- `doc_min_chunk_size` - Min chunk size (default: 100)
- `doc_enable_metadata_extraction` - Enable metadata extraction (default: true)

### Memory Management Configuration
- `memory_max_mb` - Max memory usage in MB (default: 1024)
- `memory_ttl_seconds` - Memory item TTL (default: 3600)
- `memory_cleanup_interval_secs` - Cleanup interval (default: 300)
- `memory_max_items` - Max memory items (default: 10,000)

### Research Configuration
- `research_enable_web_search` - Enable web search (default: true)
- `research_max_results` - Max research results (default: 10)
- `research_timeout_secs` - Research timeout (default: 30)
- `research_rate_limit_per_sec` - Rate limit per second (default: 5)

## Error Handling

### C Error Handling
All C functions that can fail accept an optional `RagErrorInfo*` parameter:

```c
RagErrorInfo error = {0};
RagHandle handle = rag_create(&config, &error);
if (handle == 0) {
    printf("Error [%d]: %s\n", error.code, error.message);
    rag_free_error_message(error.message);
}
```

### C++ Exception Handling
C++ functions throw `rag::Exception` on errors:

```cpp
try {
    rag::System system(config);
    // ... use system
} catch (const rag::Exception& e) {
    std::cerr << "RAG Error [" << e.code_string() << "]: " << e.what() << std::endl;
}
```

## Thread Safety

- Multiple `RagHandle` instances can be used concurrently
- Individual handles are not thread-safe - use separate instances per thread
- C++ `rag::System` instances are not thread-safe - create separate instances
- Global library state (`rag_init`/`rag_cleanup`) is thread-safe
- Memory management functions are thread-safe

## Performance Considerations

- Use connection pooling (increase `redis_pool_size` for high concurrency)
- Batch document ingestion when possible
- Configure vector store parameters based on dataset size
- Monitor memory usage and adjust `memory_max_mb` accordingly
- Use appropriate chunk sizes for your content type

## Examples

See the `examples/` directory for complete usage examples:

- `examples/c/basic_usage.c` - Basic C API usage
- `examples/cpp/basic_usage.cpp` - Basic C++ API usage
- `examples/cpp/advanced_usage.cpp` - Advanced C++ features

Build and run examples:

```bash
cd examples
make all
make test  # Requires Redis server running
```

## Troubleshooting

### Common Issues

1. **Library not found during linking**
   - Ensure the Rust library is built: `cargo build --release --features ffi`
   - Check library search path: `export LD_LIBRARY_PATH=$PWD/target/release:$LD_LIBRARY_PATH`

2. **Redis connection failed**
   - Ensure Redis server is running: `redis-server`
   - Check Redis URL in configuration
   - Verify network connectivity and firewall settings

3. **Memory leaks in C code**
   - Always call corresponding `rag_free_*` functions
   - Check error handling paths for proper cleanup
   - Use valgrind to detect leaks: `valgrind --leak-check=full ./your_program`

4. **Compilation errors**
   - Ensure C++17 compiler for C++ examples
   - Check include path: `-I../include`
   - Link against required system libraries (see Makefile)

### Debug Build

Build with debug symbols:

```bash
cargo build --features ffi  # Debug build
cd examples
make debug
```

### Logging

Enable debug logging by setting environment variable:

```bash
RUST_LOG=debug ./your_program
```

## Contributing

When contributing to the FFI interface:

1. Maintain C ABI compatibility
2. Add comprehensive error handling
3. Update both C header and C++ wrapper
4. Add examples for new functionality
5. Update documentation
6. Test with both C and C++ compilers
7. Ensure thread safety

## License

This FFI interface is part of the RAG Redis System and follows the same license terms as the main project.
