# RAG-Redis MCP Server

A high-performance Model Context Protocol (MCP) server for the RAG-Redis system, providing document ingestion, vector search, memory management, and research capabilities over JSON-RPC 2.0.

## Features

### Document Operations
- **Document Ingestion**: Process and store documents with automatic chunking and vector embedding
- **Vector Search**: Semantic similarity search using embeddings
- **Document Management**: List, retrieve, and delete documents
- **Batch Processing**: Efficient bulk document ingestion

### Research Capabilities
- **Local Search**: Query your ingested document collection
- **Web Research**: Combine local results with web search
- **Hybrid Search**: Blend semantic and keyword-based search
- **Contextual Search**: Enhanced search with additional context

### Memory Management
- **Memory Statistics**: Monitor storage usage and performance
- **Memory Clearing**: Clean up specific memory types or full reset
- **Memory Types**: Support for episodic, semantic, and procedural memory

### System Operations
- **Health Monitoring**: Comprehensive system health checks
- **Performance Metrics**: Real-time system performance data
- **Configuration**: Dynamic system configuration updates
- **Logging**: Configurable logging levels

## Installation

1. Build the MCP server:
```bash
cd mcp-server
cargo build --release
```

2. Run the server:
```bash
cargo run --release --bin rag-redis-mcp-server
```

## MCP Tools

### Document Operations

#### `ingest_document`
Process and store a document in the RAG system.

**Parameters:**
- `content` (required): Document text content
- `metadata` (optional): Document metadata object
- `document_id` (optional): Custom document ID

**Example:**
```json
{
  "name": "ingest_document",
  "arguments": {
    "content": "This is a sample document about machine learning.",
    "metadata": {
      "title": "ML Introduction",
      "author": "AI Researcher",
      "tags": ["machine-learning", "introduction"]
    }
  }
}
```

#### `search_documents`
Search documents using vector similarity.

**Parameters:**
- `query` (required): Search query text
- `limit` (optional): Maximum results (default: 10)
- `min_score` (optional): Minimum similarity score (default: 0.0)

#### `research_query`
Comprehensive research combining local and web search.

**Parameters:**
- `query` (required): Research query
- `sources` (optional): Web sources to search
- `local_only` (optional): Only search local documents

### Memory Management

#### `get_memory_stats`
Retrieve memory usage statistics.

#### `clear_memory`
Clear specified memory types (DESTRUCTIVE).

**Parameters:**
- `memory_type` (optional): Type to clear ("episodic", "semantic", "procedural", "all")
- `confirm` (required): Confirmation flag for safety

### System Monitoring

#### `health_check`
Perform comprehensive system health check.

#### `get_system_metrics`
Get real-time performance metrics.

## Protocol Implementation

The server implements MCP v2024-11-05 with the following capabilities:

- **Tools**: Full tool execution support
- **Resources**: System configuration and metrics access
- **Logging**: Configurable log levels
- **Progress**: Long-running operation progress tracking

## JSON-RPC 2.0 Communication

Communication uses JSON-RPC 2.0 over stdio:

### Request Format
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_documents",
    "arguments": {
      "query": "machine learning",
      "limit": 5
    }
  }
}
```

### Response Format
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Search results: 3 documents found..."
      }
    ]
  }
}
```

## Configuration

The server uses environment variables for configuration:

- `RUST_LOG`: Logging level (debug, info, warn, error)
- `REDIS_URL`: Redis connection URL
- `RAG_CONFIG`: Path to RAG system configuration file

## Error Handling

The server provides detailed error responses with appropriate JSON-RPC error codes:

- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid parameters
- `-32603`: Internal error
- `-32001`: Resource not found (MCP-specific)
- `-32002`: Tool not found (MCP-specific)
- `-32003`: Invalid tool input (MCP-specific)
- `-32004`: Operation failed (MCP-specific)

## Testing

Run the test suite:

```bash
cargo test
```

Key test scenarios:
- MCP initialization handshake
- Tool discovery and execution
- Error handling and validation
- Resource access
- Progress notifications

## Performance

The MCP server is optimized for:

- **Low Latency**: Sub-millisecond tool dispatch
- **High Throughput**: Concurrent request processing
- **Memory Efficiency**: Streaming large responses
- **Error Resilience**: Graceful failure handling

## Integration

The server integrates with:

- **Claude Code**: As an MCP server
- **Redis**: For vector storage and caching
- **RAG-Redis System**: Core functionality
- **Web Search APIs**: For research capabilities

## Development

### Adding New Tools

1. Define tool schema in `tools.rs`
2. Implement handler in `handlers.rs`
3. Add routing in `execute_tool` method
4. Add tests for new functionality

### Protocol Extensions

The server supports MCP protocol extensions for:

- Custom tool categories
- Streaming responses
- Batch operations
- Progress notifications

## License

MIT License - see LICENSE file for details.
