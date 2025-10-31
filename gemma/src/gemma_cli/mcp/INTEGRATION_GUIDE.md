# MCP Client Integration Guide

This guide shows how to integrate the MCP client manager into Gemma CLI.

## Quick Integration

### Step 1: Import in Main CLI

```python
# In src/gemma_cli/cli.py or main entry point
from gemma_cli.mcp import MCPClientManager
from gemma_cli.mcp.config_loader import load_mcp_servers
from gemma_cli.config.settings import Settings

class GemmaCLI:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mcp_manager: Optional[MCPClientManager] = None

    async def initialize(self):
        """Initialize Gemma CLI with MCP support."""
        # Initialize MCP if enabled
        if self.settings.mcp.enabled:
            await self._initialize_mcp()

    async def _initialize_mcp(self):
        """Initialize MCP client manager."""
        try:
            self.mcp_manager = MCPClientManager(
                tool_cache_ttl=self.settings.mcp.tool_cache_ttl
            )

            # Load server configurations
            servers = load_mcp_servers(
                Path(self.settings.mcp.servers_config)
            )

            # Connect to all enabled servers
            for name, config in servers.items():
                try:
                    await self.mcp_manager.connect_server(name, config)
                    logger.info(f"Connected to MCP server: {name}")
                except Exception as e:
                    logger.warning(f"Failed to connect to {name}: {e}")

        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            self.mcp_manager = None

    async def shutdown(self):
        """Shutdown Gemma CLI and cleanup resources."""
        if self.mcp_manager:
            await self.mcp_manager.shutdown()
```

### Step 2: Use MCP Tools in Inference

```python
# In src/gemma_cli/core/gemma.py or inference module
async def generate_with_tools(
    self,
    prompt: str,
    available_tools: Optional[list[str]] = None,
) -> str:
    """Generate response with MCP tool support."""

    # Get available MCP tools
    if self.cli.mcp_manager:
        mcp_tools = []
        for server in self.cli.mcp_manager._connections.keys():
            tools = await self.cli.mcp_manager.list_tools(server)
            mcp_tools.extend(tools)

        # Filter by requested tools
        if available_tools:
            mcp_tools = [
                t for t in mcp_tools
                if t.name in available_tools
            ]

    # Generate initial response
    response = await self.generate(prompt)

    # Check if tool use is needed
    tool_calls = self._extract_tool_calls(response)

    for tool_call in tool_calls:
        # Execute tool via MCP
        result = await self.cli.mcp_manager.call_tool(
            server=tool_call.server,
            tool=tool_call.name,
            args=tool_call.args,
            max_retries=3,
        )

        # Incorporate result into context
        prompt += f"\n\nTool {tool_call.name} result: {result}"

    # Generate final response with tool results
    final_response = await self.generate(prompt)

    return final_response
```

### Step 3: Add RAG Integration

```python
# In src/gemma_cli/rag/enhanced_backend.py
async def retrieve_context(
    self,
    query: str,
    top_k: int = 5,
) -> list[str]:
    """Retrieve context using MCP RAG server."""

    if not self.cli.mcp_manager:
        # Fallback to local RAG
        return await self.local_retrieve(query, top_k)

    try:
        # Use MCP RAG-Redis server
        result = await self.cli.mcp_manager.call_tool(
            server="rag-redis",
            tool="retrieve_memory",
            args={
                "query": query,
                "top_k": top_k,
                "memory_types": ["working", "short_term", "long_term"],
            },
        )

        return result.get("contexts", [])

    except Exception as e:
        logger.warning(f"MCP RAG failed, using fallback: {e}")
        return await self.local_retrieve(query, top_k)
```

### Step 4: Add Configuration to Settings

```python
# Already exists in src/gemma_cli/config/settings.py
class MCPConfig(BaseModel):
    enabled: bool = True
    servers_config: str = "config/mcp_servers.toml"
    tool_cache_ttl: int = 3600
    connection_timeout: int = 10
    retry_count: int = 3
```

### Step 5: Add CLI Commands

```python
# In src/gemma_cli/commands/mcp_commands.py
import click
from gemma_cli.mcp import MCPClientManager

@click.group()
def mcp():
    """MCP server management commands."""
    pass

@mcp.command()
async def status():
    """Show MCP server status."""
    manager = get_mcp_manager()  # Get from CLI context
    stats = manager.get_stats()

    for name, server_stats in stats["servers"].items():
        click.echo(f"\n{name}:")
        click.echo(f"  Status: {server_stats['status']}")
        click.echo(f"  Uptime: {server_stats['uptime']:.1f}s")
        click.echo(f"  Success Rate: {server_stats['success_rate']:.2%}")

@mcp.command()
@click.argument("server")
async def tools(server: str):
    """List tools from a server."""
    manager = get_mcp_manager()
    tools = await manager.list_tools(server)

    for tool in tools:
        click.echo(f"\n{tool.name}:")
        click.echo(f"  {tool.description}")

@mcp.command()
@click.argument("server")
async def health(server: str):
    """Check server health."""
    manager = get_mcp_manager()
    is_healthy = await manager.health_check(server)

    if is_healthy:
        click.echo(f"✓ {server} is healthy")
    else:
        click.echo(f"✗ {server} is unhealthy")
```

## Complete Example

```python
# src/gemma_cli/enhanced_cli.py
import asyncio
from pathlib import Path
from typing import Optional

from gemma_cli.config.settings import load_config
from gemma_cli.core.gemma import GemmaEngine
from gemma_cli.mcp import MCPClientManager
from gemma_cli.mcp.config_loader import load_mcp_servers


class EnhancedGemmaCLI:
    """Gemma CLI with MCP integration."""

    def __init__(self):
        self.settings = load_config()
        self.gemma: Optional[GemmaEngine] = None
        self.mcp: Optional[MCPClientManager] = None

    async def initialize(self):
        """Initialize all components."""
        # Initialize Gemma engine
        self.gemma = GemmaEngine(self.settings.gemma)
        await self.gemma.initialize()

        # Initialize MCP if enabled
        if self.settings.mcp.enabled:
            await self._init_mcp()

    async def _init_mcp(self):
        """Initialize MCP client."""
        self.mcp = MCPClientManager(
            tool_cache_ttl=self.settings.mcp.tool_cache_ttl
        )

        servers = load_mcp_servers(Path(self.settings.mcp.servers_config))

        for name, config in servers.items():
            try:
                await self.mcp.connect_server(name, config)
                print(f"✓ Connected to {name}")
            except Exception as e:
                print(f"✗ Failed to connect to {name}: {e}")

    async def chat(self, prompt: str) -> str:
        """Chat with Gemma using MCP tools."""
        # Get RAG context if available
        context = ""
        if self.mcp:
            try:
                result = await self.mcp.call_tool(
                    "rag-redis",
                    "retrieve_memory",
                    {
                        "query": prompt,
                        "top_k": 3,
                        "memory_types": ["working", "short_term"],
                    },
                )
                context = "\n\n".join(result.get("contexts", []))
            except Exception as e:
                print(f"RAG retrieval failed: {e}")

        # Generate response
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = await self.gemma.generate(full_prompt)

        # Store conversation in memory
        if self.mcp:
            try:
                await self.mcp.call_tool(
                    "rag-redis",
                    "store_memory",
                    {
                        "content": f"User: {prompt}\nAssistant: {response}",
                        "memory_type": "working",
                        "importance": 0.7,
                    },
                )
            except Exception as e:
                print(f"Memory storage failed: {e}")

        return response

    async def shutdown(self):
        """Cleanup resources."""
        if self.mcp:
            await self.mcp.shutdown()

        if self.gemma:
            await self.gemma.shutdown()


async def main():
    """Main entry point."""
    cli = EnhancedGemmaCLI()

    try:
        await cli.initialize()

        # Interactive chat loop
        print("Gemma CLI with MCP (type 'exit' to quit)")
        print("=" * 50)

        while True:
            prompt = input("\nYou: ").strip()

            if prompt.lower() in ("exit", "quit"):
                break

            if not prompt:
                continue

            response = await cli.chat(prompt)
            print(f"\nGemma: {response}")

    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Integration

```python
# tests/test_integration.py
import pytest

from gemma_cli.enhanced_cli import EnhancedGemmaCLI


@pytest.mark.asyncio
async def test_mcp_integration():
    """Test MCP integration with Gemma CLI."""
    cli = EnhancedGemmaCLI()

    try:
        await cli.initialize()

        # Test MCP is initialized
        assert cli.mcp is not None

        # Test server connections
        stats = cli.mcp.get_stats()
        assert "servers" in stats
        assert len(stats["servers"]) > 0

        # Test chat with MCP
        response = await cli.chat("What is Python?")
        assert response
        assert len(response) > 0

    finally:
        await cli.shutdown()


@pytest.mark.asyncio
async def test_mcp_rag_retrieval():
    """Test RAG retrieval via MCP."""
    cli = EnhancedGemmaCLI()

    try:
        await cli.initialize()

        if cli.mcp:
            # Store test data
            await cli.mcp.call_tool(
                "rag-redis",
                "store_memory",
                {
                    "content": "Python is a programming language",
                    "memory_type": "working",
                    "importance": 0.9,
                },
            )

            # Retrieve
            result = await cli.mcp.call_tool(
                "rag-redis",
                "retrieve_memory",
                {"query": "What is Python?", "top_k": 1},
            )

            assert result
            assert len(result.get("contexts", [])) > 0

    finally:
        await cli.shutdown()
```

## Configuration Example

```toml
# config/config.toml
[mcp]
enabled = true
servers_config = "config/mcp_servers.toml"
tool_cache_ttl = 3600
connection_timeout = 10
retry_count = 3

# config/mcp_servers.toml
[rag-redis]
enabled = true
transport = "stdio"
command = "rag-redis-server"
args = ["--config", "config/rag_redis.toml"]
auto_reconnect = true
health_check_interval = 60.0
```

## Troubleshooting

### MCP Not Connecting

1. Check server configuration:
   ```python
   from gemma_cli.mcp.config_loader import validate_mcp_config
   is_valid, errors = validate_mcp_config()
   print(errors)
   ```

2. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. Test server manually:
   ```bash
   rag-redis-server --config config/rag_redis.toml
   ```

### Tool Execution Fails

1. Increase timeout:
   ```toml
   request_timeout = 60.0  # Increase from 30s
   ```

2. Check server logs:
   ```bash
   tail -f logs/rag-redis-server.log
   ```

3. Validate tool arguments:
   ```python
   tools = await manager.list_tools("rag-redis")
   for tool in tools:
       print(tool.name, tool.inputSchema)
   ```

### Health Checks Failing

1. Disable for debugging:
   ```toml
   health_check_interval = 0  # Disable
   ```

2. Manual health check:
   ```python
   is_healthy = await manager.health_check("rag-redis")
   print(f"Healthy: {is_healthy}")
   ```

3. Check server status:
   ```python
   stats = manager.get_stats()
   print(stats["servers"]["rag-redis"])
   ```

## Best Practices

1. **Always use context managers** for resource cleanup
2. **Enable health checks** in production
3. **Configure appropriate timeouts** for your use case
4. **Monitor statistics** for performance issues
5. **Handle connection failures** gracefully with fallbacks
6. **Cache tool lists** to reduce overhead
7. **Log errors** for debugging
8. **Test integration** thoroughly before deployment
9. **Use retry logic** for transient failures
10. **Validate configurations** on startup

## Performance Tips

1. **Connection Pooling**: Reuse connections across requests
2. **Tool Caching**: Set appropriate TTL for your use case
3. **Async Operations**: Use asyncio.gather for parallel operations
4. **Health Check Interval**: Balance between responsiveness and overhead
5. **Retry Delays**: Use exponential backoff to avoid overwhelming servers

## Next Steps

1. Integrate MCP client into main CLI
2. Add MCP commands to CLI interface
3. Test with real MCP servers
4. Monitor performance in production
5. Add metrics and monitoring
6. Document server-specific integrations

## Support

For issues or questions:
- Check README: `src/gemma_cli/mcp/README.md`
- Review examples: `src/gemma_cli/mcp/example_usage.py`
- Run tests: `pytest tests/test_mcp_client.py -v`
- Check logs for detailed error messages
