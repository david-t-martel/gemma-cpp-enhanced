# MCP Client Manager for Gemma CLI

Production-ready Model Context Protocol (MCP) client implementation with comprehensive features for robust server integration.

## Features

- **Multiple Transport Protocols**: stdio, HTTP, SSE, WebSocket
- **Connection Pooling**: Efficient connection management
- **Automatic Reconnection**: Exponential backoff with configurable retries
- **Tool Discovery**: Intelligent caching with configurable TTL
- **Health Monitoring**: Background health checks with automatic recovery
- **Error Handling**: Comprehensive exception handling with retries
- **Type Safety**: Full type hints compatible with `mypy --strict`
- **Statistics**: Detailed metrics for monitoring and debugging
- **Async-First**: Built on asyncio for optimal performance

## Installation

The MCP client is included with Gemma CLI. Ensure you have the required dependencies:

```bash
uv pip install -e .
```

Key dependencies:
- `mcp>=0.9.0` - Official MCP Python SDK
- `aiofiles>=23.2.1` - Async file operations
- `toml>=0.10.2` - Configuration file parsing

## Quick Start

### 1. Configuration

Create `config/mcp_servers.toml`:

```toml
[rag-redis]
enabled = true
transport = "stdio"
command = "rag-redis-server"
args = ["--config", "config/rag_redis.toml"]
auto_reconnect = true
max_reconnect_attempts = 5
connection_timeout = 10.0
request_timeout = 30.0
health_check_interval = 60.0

[filesystem]
enabled = true
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
connection_timeout = 10.0
```

### 2. Basic Usage

```python
from gemma_cli.mcp import MCPClientManager
from gemma_cli.mcp.config_loader import load_mcp_servers

# Create manager
manager = MCPClientManager(tool_cache_ttl=3600.0)

# Load server configurations
servers = load_mcp_servers()

# Connect to servers
for name, config in servers.items():
    await manager.connect_server(name, config)

# List available tools
tools = await manager.list_tools("rag-redis")
for tool in tools:
    print(f"Tool: {tool.name} - {tool.description}")

# Execute a tool
result = await manager.call_tool(
    server="rag-redis",
    tool="store_memory",
    args={
        "content": "Important information",
        "memory_type": "working",
        "importance": 0.9,
    },
    max_retries=3,
)

# Get statistics
stats = manager.get_stats()
print(f"Total requests: {stats['servers']['rag-redis']['total_requests']}")

# Cleanup
await manager.shutdown()
```

### 3. Using Context Manager (Recommended)

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def mcp_manager_context():
    """Context manager for MCP client lifecycle."""
    manager = MCPClientManager()
    try:
        servers = load_mcp_servers()
        for name, config in servers.items():
            await manager.connect_server(name, config)
        yield manager
    finally:
        await manager.shutdown()

# Usage
async with mcp_manager_context() as manager:
    tools = await manager.list_tools("rag-redis")
    # ... use manager ...
```

## Configuration Reference

### Server Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | required | Unique server identifier |
| `transport` | str | required | Transport protocol: "stdio", "http", "sse", "websocket" |
| `command` | str | required* | Command for stdio transport |
| `args` | list[str] | `[]` | Command arguments |
| `url` | str | required* | URL for HTTP/SSE/WebSocket |
| `env` | dict | `{}` | Environment variables |
| `enabled` | bool | `true` | Enable server |
| `auto_reconnect` | bool | `true` | Auto-reconnect on failure |
| `max_reconnect_attempts` | int | `5` | Max reconnection attempts |
| `reconnect_delay` | float | `1.0` | Initial reconnect delay (seconds) |
| `connection_timeout` | float | `10.0` | Connection timeout (seconds) |
| `request_timeout` | float | `30.0` | Request timeout (seconds) |
| `health_check_interval` | float | `60.0` | Health check interval (seconds) |

\* Required based on transport type

## API Reference

### MCPClientManager

Main client manager class for MCP operations.

#### Methods

##### `connect_server(name: str, config: MCPServerConfig) -> bool`

Connect to an MCP server.

```python
config = MCPServerConfig(
    name="my-server",
    transport=MCPTransportType.STDIO,
    command="my-mcp-server",
)
await manager.connect_server("my-server", config)
```

##### `disconnect_server(name: str) -> bool`

Disconnect from a server.

```python
await manager.disconnect_server("my-server")
```

##### `list_tools(server: str, force_refresh: bool = False) -> list[Tool]`

Get available tools from a server.

```python
tools = await manager.list_tools("my-server")
tools_fresh = await manager.list_tools("my-server", force_refresh=True)
```

##### `call_tool(server: str, tool: str, args: dict, max_retries: int = 3, retry_delay: float = 1.0) -> Any`

Execute a tool with retry logic.

```python
result = await manager.call_tool(
    server="my-server",
    tool="process_data",
    args={"data": "input"},
    max_retries=5,
    retry_delay=2.0,
)
```

##### `list_resources(server: str) -> list[Any]`

Get available resources from a server.

```python
resources = await manager.list_resources("filesystem")
```

##### `read_resource(server: str, uri: str) -> Any`

Read a resource from a server.

```python
content = await manager.read_resource("filesystem", "file:///path/to/file")
```

##### `health_check(server: str) -> bool`

Check server health.

```python
is_healthy = await manager.health_check("my-server")
```

##### `get_stats() -> dict[str, Any]`

Get connection statistics.

```python
stats = manager.get_stats()
print(f"Success rate: {stats['servers']['my-server']['success_rate']:.2%}")
```

##### `shutdown() -> None`

Shutdown all connections.

```python
await manager.shutdown()
```

### MCPServerConfig

Server configuration model.

```python
from gemma_cli.mcp import MCPServerConfig, MCPTransportType

config = MCPServerConfig(
    name="my-server",
    transport=MCPTransportType.STDIO,
    command="my-command",
    args=["--flag", "value"],
    env={"VAR": "value"},
    auto_reconnect=True,
    max_reconnect_attempts=3,
    connection_timeout=10.0,
)
```

### MCPToolRegistry

Tool registry with caching.

```python
from gemma_cli.mcp import MCPToolRegistry

registry = MCPToolRegistry(default_ttl=3600.0)

# Get tools (with caching)
tools = await registry.get_tools("server", fetch_function)

# Invalidate cache
await registry.invalidate("server")  # Specific server
await registry.invalidate()          # All servers

# Get cache stats
stats = registry.get_cache_stats()
```

## Error Handling

The MCP client provides specific exception types:

```python
from gemma_cli.mcp import (
    MCPError,                  # Base exception
    MCPConnectionError,        # Connection failures
    MCPToolExecutionError,     # Tool execution failures
    MCPResourceError,          # Resource operation failures
)

try:
    result = await manager.call_tool("server", "tool", {})
except MCPConnectionError as e:
    print(f"Connection error: {e}")
except MCPToolExecutionError as e:
    print(f"Tool execution failed: {e}")
except MCPError as e:
    print(f"General MCP error: {e}")
```

## Advanced Features

### Automatic Reconnection

The client automatically reconnects on failure with exponential backoff:

```python
config = MCPServerConfig(
    name="my-server",
    transport=MCPTransportType.STDIO,
    command="unstable-server",
    auto_reconnect=True,            # Enable auto-reconnect
    max_reconnect_attempts=5,       # Max 5 attempts
    reconnect_delay=2.0,            # Start with 2s delay
)

# Reconnection schedule:
# Attempt 1: 2s delay
# Attempt 2: 4s delay
# Attempt 3: 8s delay
# Attempt 4: 16s delay
# Attempt 5: 32s delay
```

### Health Monitoring

Background health checks run automatically:

```python
config = MCPServerConfig(
    name="my-server",
    transport=MCPTransportType.STDIO,
    command="my-server",
    health_check_interval=60.0,  # Check every 60 seconds
)

# Health check failures trigger auto-reconnection if enabled
```

### Tool Caching

Tools are cached to reduce overhead:

```python
# First call - fetches from server and caches
tools = await manager.list_tools("server")

# Second call - uses cache (fast)
tools = await manager.list_tools("server")

# Force refresh - bypasses cache
tools = await manager.list_tools("server", force_refresh=True)

# Cache TTL is configurable
manager = MCPClientManager(tool_cache_ttl=1800.0)  # 30 minutes
```

### Statistics and Monitoring

Comprehensive metrics for each server:

```python
stats = manager.get_stats()

for server_name, server_stats in stats["servers"].items():
    print(f"\nServer: {server_name}")
    print(f"  Status: {server_stats['status']}")
    print(f"  Uptime: {server_stats['uptime']:.1f}s")
    print(f"  Total Requests: {server_stats['total_requests']}")
    print(f"  Success Rate: {server_stats['success_rate']:.2%}")
    print(f"  Avg Latency: {server_stats['avg_latency']:.3f}s")
    print(f"  Min Latency: {server_stats['min_latency']:.3f}s")
    print(f"  Max Latency: {server_stats['max_latency']:.3f}s")

# Tool cache statistics
cache_stats = stats["tool_cache"]
print(f"\nTool Cache:")
print(f"  Servers Cached: {cache_stats['servers_cached']}")
print(f"  Total Tools: {cache_stats['total_tools']}")
print(f"  Valid Tools: {cache_stats['valid_tools']}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all MCP tests
uv run pytest tests/test_mcp_client.py -v

# Run specific test
uv run pytest tests/test_mcp_client.py::TestMCPClientManager::test_connect_server_success -v

# Run with coverage
uv run pytest tests/test_mcp_client.py --cov=src/gemma_cli/mcp --cov-report=term-missing
```

## Examples

See `example_usage.py` for comprehensive examples:

```bash
uv run python -m gemma_cli.mcp.example_usage
```

Examples include:
- Basic connection and tool discovery
- Tool execution with error handling
- Resource operations
- Health monitoring
- Concurrent operations
- Advanced features (caching, reconnection)

## Configuration Validation

Validate your configuration before use:

```python
from gemma_cli.mcp.config_loader import validate_mcp_config

is_valid, errors = validate_mcp_config()

if is_valid:
    print("✓ Configuration is valid")
else:
    print("✗ Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

Or via command line:

```bash
uv run python -c "from gemma_cli.mcp.config_loader import validate_mcp_config; print(validate_mcp_config())"
```

## Integration with Gemma CLI

The MCP client integrates seamlessly with Gemma CLI:

```python
from gemma_cli.core.gemma import GemmaEngine
from gemma_cli.mcp import MCPClientManager
from gemma_cli.mcp.config_loader import load_mcp_servers

async def enhanced_gemma_with_mcp():
    """Gemma inference with MCP tool integration."""
    # Initialize Gemma
    gemma = GemmaEngine(...)

    # Initialize MCP
    mcp_manager = MCPClientManager()
    servers = load_mcp_servers()

    for name, config in servers.items():
        await mcp_manager.connect_server(name, config)

    # Use MCP tools during inference
    tools = await mcp_manager.list_tools("rag-redis")

    # Execute tools as needed
    context = await mcp_manager.call_tool(
        "rag-redis",
        "retrieve_context",
        {"query": user_query, "top_k": 5}
    )

    # Generate response with RAG context
    response = await gemma.generate(prompt, context=context)

    await mcp_manager.shutdown()
```

## Performance Considerations

- **Connection Pooling**: Reuse connections across requests
- **Tool Caching**: Reduce tool discovery overhead
- **Async Operations**: Non-blocking I/O for all operations
- **Background Health Checks**: Proactive connection management
- **Retry Logic**: Automatic retry with exponential backoff

## Troubleshooting

### Connection Timeout

Increase timeout values:

```toml
connection_timeout = 30.0  # Increase from default 10s
```

### Tool Execution Failures

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Increase retries:

```python
result = await manager.call_tool(
    "server",
    "tool",
    {},
    max_retries=5,      # More attempts
    retry_delay=2.0,    # Longer initial delay
)
```

### Health Check Issues

Disable health checks for debugging:

```toml
health_check_interval = 0  # Disable health checks
```

### Cache Issues

Force cache refresh:

```python
tools = await manager.list_tools("server", force_refresh=True)
```

Or invalidate cache:

```python
await manager._tool_registry.invalidate("server")
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure:
- All tests pass: `pytest tests/test_mcp_client.py`
- Type checking passes: `mypy src/gemma_cli/mcp/`
- Code is formatted: `ruff format src/gemma_cli/mcp/`
- Linting passes: `ruff check src/gemma_cli/mcp/`

## Support

For issues or questions:
- GitHub Issues: [Project Issues](https://github.com/your-repo/issues)
- Documentation: [Full Docs](https://docs.example.com)
- MCP Protocol: [MCP Specification](https://spec.modelcontextprotocol.io/)
