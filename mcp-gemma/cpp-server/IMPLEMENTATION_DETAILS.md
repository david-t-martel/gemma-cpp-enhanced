# MCP-Gemma WebSocket Server Implementation Details

This document provides comprehensive details about the WebSocket-based MCP (Model Context Protocol) server implementation for Gemma.cpp.

## Architecture Overview

### Core Components

1. **MCPServer Class** (`mcp_server.h/cpp`)
   - Main server interface with pImpl pattern
   - WebSocket server lifecycle management
   - Public API for client applications

2. **ConnectionMetadata Structure**
   - Tracks individual WebSocket connections
   - Maintains authentication state and statistics
   - Handles MCP handshake completion status

3. **WebSocket Integration**
   - Built on WebSocket++ library
   - Asynchronous message handling
   - Connection pooling and management

4. **MCP Protocol Implementation**
   - Full JSON-RPC 2.0 compliance
   - MCP-specific method handlers
   - Error handling and validation

## Key Features

### WebSocket Server Capabilities

- **Multi-client Support**: Handles up to 100 concurrent connections (configurable)
- **Protocol Validation**: Enforces MCP subprotocol requirements
- **Message Size Limits**: Configurable maximum message size (default: 1MB)
- **Connection Timeouts**: Automatic cleanup of inactive connections
- **Heartbeat System**: Periodic ping/pong to maintain connection health

### MCP Protocol Support

- **Handshake Management**: Proper initialize/initialized sequence
- **Tool Discovery**: Dynamic tool listing with JSON schemas
- **Tool Execution**: Secure tool invocation with parameter validation
- **Error Handling**: Comprehensive error codes and messages
- **Capability Negotiation**: Server and client capability exchange

### Production Features

- **Comprehensive Logging**: Timestamped, level-based logging system
- **Connection Statistics**: Real-time metrics and performance monitoring
- **Graceful Shutdown**: Clean connection termination on server stop
- **Thread Safety**: Mutex-protected shared resources
- **Memory Management**: RAII and smart pointer usage throughout

## Implementation Details

### WebSocket Server Configuration

```cpp
struct Config {
    std::string host = "localhost";
    int port = 8080;

    // WebSocket-specific settings
    int websocket_timeout_seconds = 30;
    int max_connections = 100;
    bool enable_compression = true;
    size_t max_message_size = 1024 * 1024; // 1MB
    int heartbeat_interval_seconds = 30;
    bool require_authentication = false;
    std::string auth_token;
};
```

### Connection Lifecycle

1. **Connection Validation** (`OnValidate`)
   - Checks connection limits
   - Validates subprotocol requirements
   - Rejects unauthorized connections

2. **Connection Establishment** (`OnOpen`)
   - Generates unique connection ID
   - Creates connection metadata
   - Initiates MCP handshake

3. **Message Processing** (`OnMessage`)
   - JSON parsing and validation
   - MCP method routing
   - Response generation

4. **Connection Cleanup** (`OnClose`)
   - Removes connection metadata
   - Updates statistics
   - Logs disconnection events

### MCP Method Handlers

#### Initialize Method
```cpp
void HandleInitialize(const std::string& connection_id, const nlohmann::json& request)
```
- Validates client protocol version
- Exchanges server and client capabilities
- Marks handshake as complete

#### Tools/List Method
```cpp
void HandleToolsList(const std::string& connection_id, const nlohmann::json& request)
```
- Returns available tools with descriptions
- Includes JSON schemas for input validation
- Supports dynamic tool registration

#### Tools/Call Method
```cpp
void HandleToolsCall(const std::string& connection_id, const nlohmann::json& request)
```
- Validates tool existence and parameters
- Executes tool with argument validation
- Returns formatted response or error

### Available Tools

1. **generate_text**
   - Generates text using Gemma model
   - Parameters: prompt, temperature, max_tokens, top_k, top_p, stop_sequence
   - Returns: generated text response

2. **get_model_info**
   - Retrieves loaded model information
   - Parameters: none
   - Returns: model metadata (name, version, parameters, etc.)

3. **tokenize_text**
   - Tokenizes input text using model tokenizer
   - Parameters: text
   - Returns: token IDs and metadata

4. **set_generation_params**
   - Updates default generation parameters
   - Parameters: temperature, max_tokens, top_k, top_p
   - Returns: confirmation of parameter updates

5. **get_server_status**
   - Returns comprehensive server status
   - Parameters: none
   - Returns: uptime, connections, statistics, configuration

### Error Handling

The implementation follows JSON-RPC 2.0 error codes:

- `-32700`: Parse error (invalid JSON)
- `-32600`: Invalid Request (malformed request)
- `-32601`: Method not found (unknown MCP method)
- `-32602`: Invalid params (missing or invalid parameters)
- `-32603`: Internal error (server-side exception)

### Thread Safety

- **Connection Management**: Mutex-protected connection map
- **Statistics**: Atomic counters for thread-safe updates
- **Message Handling**: Per-connection thread safety
- **Server Control**: Thread-safe start/stop operations

### Memory Management

- **RAII Pattern**: Automatic resource cleanup
- **Smart Pointers**: `std::unique_ptr` for implementation hiding
- **Connection Metadata**: Efficient storage and cleanup
- **JSON Processing**: Move semantics for large payloads

## Performance Considerations

### Scalability

- **Asynchronous I/O**: Non-blocking WebSocket operations
- **Connection Pooling**: Efficient connection management
- **Message Batching**: Optimized for high-throughput scenarios
- **Resource Limits**: Configurable limits prevent resource exhaustion

### Memory Usage

- **Minimal Overhead**: Lightweight connection tracking
- **JSON Optimization**: Efficient parsing and generation
- **Buffer Management**: Controlled message buffer sizes
- **Cleanup Policies**: Automatic cleanup of inactive resources

### CPU Optimization

- **Event-Driven**: Callback-based message processing
- **Thread Pool**: Background processing for long operations
- **Heartbeat Optimization**: Efficient keep-alive mechanisms
- **Compiler Optimizations**: Modern C++ with optimization flags

## Security Features

### Input Validation

- **JSON Schema Validation**: Tool parameter validation
- **Message Size Limits**: Prevention of DoS attacks
- **Connection Limits**: Resource exhaustion protection
- **Protocol Enforcement**: Strict MCP compliance

### Authentication Support

- **Token-Based Auth**: Optional authentication mechanism
- **Connection Tracking**: Per-client authentication state
- **Secure Headers**: Validation of security headers
- **Rate Limiting**: Configurable per-connection limits

### Error Information Disclosure

- **Sanitized Errors**: No internal details in client responses
- **Logging Separation**: Detailed server logs vs. client messages
- **Stack Trace Protection**: No sensitive information leakage

## Extensibility

### Custom Tool Registration

```cpp
server.RegisterTool("custom_tool", [](const nlohmann::json& params) {
    // Custom tool implementation
    return nlohmann::json{{"result", "custom response"}};
});
```

### Configuration Extensions

- **Plugin Architecture**: Support for loadable modules
- **Dynamic Configuration**: Runtime parameter updates
- **Monitoring Hooks**: Integration with monitoring systems
- **Custom Protocols**: Extension points for additional protocols

## Integration with Gemma.cpp

### Model Loading

- **Lazy Loading**: Model loaded on first use
- **Error Handling**: Graceful degradation when model unavailable
- **Resource Management**: Proper model lifecycle management
- **Configuration**: Flexible model and tokenizer paths

### Inference Integration

- **Parameter Passing**: Direct integration with Gemma inference
- **Stream Support**: Future support for streaming responses
- **Batch Processing**: Efficient batched inference operations
- **Error Propagation**: Proper error handling from inference layer

## Testing and Validation

### Unit Tests

- **Protocol Compliance**: MCP specification adherence
- **Error Scenarios**: Comprehensive error condition testing
- **Performance Tests**: Load and stress testing capabilities
- **Security Tests**: Vulnerability and edge case testing

### Integration Tests

- **Client Compatibility**: Testing with various MCP clients
- **Model Integration**: Testing with different Gemma models
- **Concurrency Tests**: Multi-client connection testing
- **Reliability Tests**: Long-running stability validation

## Deployment Considerations

### Build Requirements

- **C++17 Compiler**: Modern C++ standard compliance
- **CMake 3.16+**: Build system requirements
- **Dependencies**: WebSocket++, nlohmann/json, Boost
- **Platform Support**: Windows, Linux, macOS compatibility

### Runtime Requirements

- **Memory**: Depends on loaded models (2-8GB typical)
- **CPU**: Multi-core recommended for concurrent clients
- **Network**: Low latency for real-time interactions
- **Storage**: Model files and logging space

### Configuration Management

- **Environment Variables**: Runtime configuration override
- **Configuration Files**: Structured server configuration
- **Command Line**: Direct parameter specification
- **Docker Support**: Containerized deployment options

This implementation provides a robust, production-ready MCP server with comprehensive WebSocket support, making Gemma.cpp accessible through the standardized MCP protocol.