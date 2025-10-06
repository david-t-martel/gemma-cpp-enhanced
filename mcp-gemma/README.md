# MCP Gemma Server

A comprehensive Model Context Protocol (MCP) server integration for gemma.cpp, enabling seamless AI model integration across multiple platforms and transport protocols.

## Features

- **Multiple Transport Protocols**: stdio, HTTP REST API, and WebSocket support
- **Advanced Memory Management**: RAG-Redis integration with vector similarity search
- **Multi-tier Memory**: Working, short-term, long-term, and episodic memory tiers
- **Performance Optimized**: Built for Windows deployment with WSL integration
- **Framework Integration**: Direct integration with existing Python agent frameworks
- **Real-time Streaming**: WebSocket-based streaming text generation
- **Production Ready**: Service deployment, monitoring, and health checks

## Quick Start

### Prerequisites

- Python 3.8+
- Redis (optional, for memory features)
- Gemma model files (download separately)
- Windows 10/11 with WSL (for gemma.cpp integration)

### Installation

1. **Clone and setup environment:**
```powershell
git clone <repository-url>
cd mcp-gemma
.\scripts\setup-environment.ps1
```

2. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

3. **Configure your model path:**
```powershell
# Edit config/server_config.yaml or provide via command line
```

### Start the Server

**All transports (recommended):**
```powershell
.\scripts\start-server.ps1 -ModelPath "C:\path\to\your\model.sbs" -Mode all
```

**Specific transport only:**
```powershell
# Stdio only (for direct integration)
python server\main.py --mode stdio --model "C:\path\to\model.sbs"

# HTTP only (for REST API)
python server\main.py --mode http --port 8080 --model "C:\path\to\model.sbs"

# WebSocket only (for streaming)
python server\main.py --mode websocket --port 8081 --model "C:\path\to\model.sbs"
```

### Test the Installation

```powershell
.\scripts\test-server.ps1
```

## Usage Examples

### Python Client (Stdio)

```python
import asyncio
from client import GemmaStdioClient, GenerationRequest

async def main():
    async with GemmaStdioClient(model_path="C:/path/to/model.sbs") as client:
        # Simple text generation
        response = await client.simple_generate("Hello, how are you?")
        print(response)

        # Chat with context
        context = [{"role": "user", "content": "My name is Alice"}]
        response = await client.chat("What's my name?", context)
        print(response)

        # Memory operations
        await client.store_memory("user_pref", "Likes short responses", {"type": "preference"})
        memory = await client.retrieve_memory("user_pref")
        print(memory)

asyncio.run(main())
```

### Python Client (HTTP)

```python
import asyncio
from client import GemmaHTTPClient, GenerationRequest

async def main():
    async with GemmaHTTPClient("http://localhost:8080") as client:
        response = await client.simple_generate("Tell me a joke")
        print(response)

        # Get server metrics
        metrics = await client.get_metrics()
        print(f"Server uptime: {metrics['performance']['uptime_seconds']}s")

asyncio.run(main())
```

### REST API

```bash
# Health check
curl http://localhost:8080/health

# List available tools
curl http://localhost:8080/tools

# Generate text
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 100}'

# Call specific tool
curl -X POST http://localhost:8080/tools/generate_text/call \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"prompt": "Hello", "temperature": 0.8}}'
```

### WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8081');

ws.onopen = function() {
    // Start streaming generation
    ws.send(JSON.stringify({
        type: 'generate_text',
        request_id: 'req_001',
        prompt: 'Tell me a story',
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'generation_chunk') {
        process.stdout.write(data.chunk);
    }
};
```

## Integration with Existing Frameworks

### Stats Framework Integration

```python
from integration.stats_integration import GemmaMCPAgent, register_gemma_tools

# Create integrated agent
async with GemmaMCPAgent(client_type="http", base_url="http://localhost:8080") as agent:
    # Use with existing stats framework
    response = await agent.generate_response("Analyze this data...")

    # Store analysis results
    await agent.store_memory("analysis_001", response, {"type": "analysis"})
```

### RAG-Redis Integration

The server automatically integrates with RAG-Redis for enhanced memory management:

- **Vector Similarity Search**: Automatic embedding generation and similarity matching
- **Multi-tier Memory**: Automatic promotion/demotion based on access patterns
- **Search Indexing**: Keyword and semantic search capabilities
- **Memory Consolidation**: Background optimization of memory storage

## Configuration

### Server Configuration (config/server_config.yaml)

```yaml
# Model settings
model:
  model_path: "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs"
  tokenizer_path: "/c/codedev/llm/.models/tokenizer.spm"
  max_tokens: 2048
  temperature: 0.7

# Transport settings
transports:
  http:
    enabled: true
    port: 8080
  websocket:
    enabled: true
    port: 8081

# Redis memory settings
redis:
  enabled: true
  host: "localhost"
  port: 6379
```

### MCP Integration (config/mcp_integration.json)

```json
{
  "mcpServers": {
    "gemma-mcp": {
      "command": "python",
      "args": ["server/main.py", "--mode", "stdio", "--model", "C:/path/to/model.sbs"]
    }
  }
}
```

## Architecture

```
MCP Gemma Server
├── Server Layer
│   ├── Base Server (MCP protocol handling)
│   ├── Transport Layer (stdio/HTTP/WebSocket)
│   ├── Handler Layer (Generation/Memory/Metrics)
│   └── Integration Layer (Stats/RAG-Redis)
├── Client Layer
│   ├── Stdio Client (direct process communication)
│   ├── HTTP Client (REST API)
│   └── WebSocket Client (real-time streaming)
└── Integration Layer
    ├── Stats Framework Integration
    ├── RAG-Redis Memory System
    └── External Tool Registration
```

## Performance

- **Concurrency**: Supports multiple concurrent requests
- **Memory Efficiency**: Multi-tier memory management with automatic optimization
- **Streaming**: Real-time token streaming via WebSocket
- **Caching**: Redis-based response and embedding caching
- **Monitoring**: Built-in metrics and health monitoring

## Deployment

### Development

```powershell
.\scripts\setup-environment.ps1 -Environment development
.\scripts\start-server.ps1 -Debug
```

### Production

```powershell
# Install as Windows service
.\scripts\deploy-windows.ps1 -Action install -Environment production

# Start service
.\scripts\deploy-windows.ps1 -Action start

# Monitor service
.\scripts\deploy-windows.ps1 -Action status
```

## API Reference

### Tools

- **generate_text**: Text generation with configurable parameters
- **switch_model**: Dynamic model switching
- **store_memory**: Store information with metadata
- **retrieve_memory**: Retrieve stored information
- **search_memory**: Semantic and keyword search
- **get_metrics**: Server performance metrics

### Endpoints (HTTP)

- `GET /health` - Health check
- `GET /tools` - List available tools
- `POST /generate` - Direct text generation
- `POST /tools/{tool_name}/call` - Call specific tool
- `GET /metrics` - Server metrics

### WebSocket Messages

- `generate_text` - Start text generation
- `call_tool` - Call server tool
- `ping/pong` - Connection keepalive

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: [Full documentation](docs/)
- Issues: [GitHub Issues](https://github.com/llm-dev/mcp-gemma/issues)
- Discussions: [GitHub Discussions](https://github.com/llm-dev/mcp-gemma/discussions)