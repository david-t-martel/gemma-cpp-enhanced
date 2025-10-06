# Consolidated MCP Gemma Server

A unified Model Context Protocol (MCP) server for Gemma.cpp that combines the best features from all previous implementations.

## 🚀 Features

### Core Capabilities
- **High-Performance C++ Server**: Direct gemma.cpp integration for maximum speed
- **Feature-Rich Python Server**: Advanced conversation management and extensibility
- **Hybrid Architecture**: Can use either C++ or Python backends seamlessly

### Advanced Features
- ✅ **Conversation State Management**: Persistent chat sessions with history
- ✅ **Multiple Transport Protocols**: stdio, HTTP, WebSocket
- ✅ **Memory Backends**: Redis, in-memory storage
- ✅ **Legacy Compatibility**: All original tool names supported
- ✅ **Metrics & Monitoring**: Comprehensive performance tracking
- ✅ **Streaming Responses**: Real-time text generation
- ✅ **Model Management**: Dynamic model loading and information

## 📁 Project Structure

```
mcp-gemma/
├── cpp-server/              # High-performance C++ implementation
│   ├── mcp_server.{h,cpp}   # Core MCP protocol handling
│   ├── inference_handler.*  # Gemma inference integration
│   ├── model_manager.*      # Model loading and management
│   ├── main.cpp            # WebSocket server entry point
│   ├── mcp_stdio_server.cpp # Stdio server entry point
│   └── CMakeLists.txt      # Build configuration
│
├── server/                  # Python implementation with advanced features
│   ├── consolidated_server.py # Main unified server
│   ├── chat_handler.py     # Conversation management
│   ├── base.py            # Base server classes
│   ├── handlers.py        # Core request handlers
│   ├── transports.py      # Transport protocols
│   └── core/              # Core architecture components
│
├── client/                 # MCP client implementations
├── config/                # Configuration templates
├── tests/                 # Test suites
├── docs/                  # Documentation
├── examples/              # Usage examples
└── .archive/              # Archived original implementations
```

## 🔧 Installation

### Quick Start

```bash
# Clone or navigate to the project
cd /c/codedev/llm/mcp-gemma

# Build everything (Python + C++)
python build.py --all

# Test installation
python build.py --test
```

### Manual Setup

#### Python Environment
```bash
# Using uv (recommended)
uv pip install mcp asyncio redis aiohttp websockets pydantic

# Or using pip
pip install mcp asyncio redis aiohttp websockets pydantic
```

#### C++ Build
```bash
cd cpp-server
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 🚀 Usage

### Python Server (Recommended)

```bash
# Basic usage with stdio
python server/consolidated_server.py --model /path/to/model.sbs

# With additional options
python server/main.py \
  --mode stdio \
  --model /path/to/model.sbs \
  --tokenizer /path/to/tokenizer.spm \
  --max-tokens 2048 \
  --temperature 0.7

# Multi-transport mode
python server/main.py \
  --mode all \
  --model /path/to/model.sbs \
  --port 8080
```

### C++ Server (High Performance)

```bash
# Stdio mode
./cpp-server/build/gemma_mcp_stdio_server \
  --model /path/to/model.sbs \
  --tokenizer /path/to/tokenizer.spm

# WebSocket mode
./cpp-server/build/gemma_mcp_server \
  --host localhost \
  --port 8080 \
  --model /path/to/model.sbs
```

## 🛠️ Available Tools

### Core Generation
- `generate`: Advanced text generation with streaming support
- `chat`: Conversation management with persistent state
- `list_conversations`: View active chat sessions
- `get_conversation`: Retrieve conversation details

### Memory & Storage
- `store_memory`: Long-term memory storage
- `search_memory`: Semantic memory search

### Model Management
- `list_models`: Available model information
- `model_info`: Detailed model specifications
- `server_status`: Comprehensive metrics and health

### Legacy Compatibility
- `gemma_generate`: Original simple generation
- `gemma_chat`: Original chat functionality
- `gemma_models_list`: Original model listing
- `gemma_model_info`: Original model information

## 🔧 Configuration

### Environment Variables
```bash
# Model paths
export GEMMA_MODEL_PATH="/path/to/model.sbs"
export GEMMA_TOKENIZER_PATH="/path/to/tokenizer.spm"

# Redis configuration (optional)
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"

# Server configuration
export MCP_HOST="localhost"
export MCP_PORT="8080"
```

### Configuration File
```python
# config/server_config.py
from dataclasses import dataclass

@dataclass
class ServerConfig:
    model_path: str = "/path/to/model.sbs"
    tokenizer_path: str = "/path/to/tokenizer.spm"
    max_tokens: int = 2048
    temperature: float = 0.7
    max_context: int = 8192

    # Memory settings
    memory_backend: str = "redis"  # "redis" or "inmemory"
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Transport settings
    host: str = "localhost"
    port: int = 8080
    enable_websocket: bool = True
    enable_http: bool = True
```

## 🔄 Migration from Previous Implementations

### From Simple MCP Server
All existing tools work unchanged:
```python
# These continue to work exactly as before
await mcp_client.call_tool("gemma_generate", {"prompt": "Hello"})
await mcp_client.call_tool("gemma_chat", {"message": "Hi"})
```

### From Advanced MCP Server
Enhanced features available:
```python
# Enhanced generation with streaming
await mcp_client.call_tool("generate", {
    "prompt": "Hello",
    "stream": True,
    "max_tokens": 1000
})

# Persistent conversations
await mcp_client.call_tool("chat", {
    "message": "Hello",
    "conversation_id": "session_123",
    "system_prompt": "You are a helpful assistant"
})
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_consolidated_server.py
python -m pytest tests/test_chat_handler.py

# Integration tests (requires Redis)
python -m pytest tests/integration/ --redis

# Performance benchmarks
python tests/benchmark_server.py
```

## 📊 Performance Characteristics

### C++ Server
- **Latency**: <5ms per request (excluding model inference)
- **Throughput**: 100+ requests/second
- **Memory**: Minimal overhead beyond model weights
- **Startup**: Fast cold start

### Python Server
- **Latency**: 10-50ms per request (depending on features)
- **Throughput**: 50+ requests/second
- **Memory**: Additional overhead for conversation state
- **Features**: Full conversation management, memory, metrics

## 🔗 Integration Examples

### Claude Code MCP Configuration
```json
{
  "mcpServers": {
    "gemma-consolidated": {
      "command": "python",
      "args": ["/c/codedev/llm/mcp-gemma/server/consolidated_server.py",
               "--model", "/c/codedev/llm/.models/gemma-2b-it-sfp.sbs"],
      "env": {
        "PYTHONPATH": "/c/codedev/llm/mcp-gemma"
      }
    }
  }
}
```

### Direct API Usage
```python
from mcp_gemma.server.consolidated_server import ConsolidatedMCPServer
from mcp_gemma.server.base import ServerConfig

# Create configuration
config = ServerConfig(
    model_path="/path/to/model.sbs",
    temperature=0.8,
    max_tokens=1024
)

# Initialize server
server = ConsolidatedMCPServer(config)
await server.initialize()

# Generate text
response = await server.generation_handler.generate("Hello, world!")
print(response)
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Error: Model file not found: /path/to/model.sbs
   ```
   Solution: Verify model path and ensure file exists

2. **Redis connection failed**
   ```
   Error: Redis connection failed
   ```
   Solution: Start Redis server or use `--memory-backend inmemory`

3. **C++ build failed**
   ```
   Error: libgemma not found
   ```
   Solution: Build gemma.cpp first, then build MCP server

### Debug Mode
```bash
# Enable debug logging
python server/consolidated_server.py --model /path/to/model.sbs --debug

# Check server health
curl http://localhost:8080/health  # If running HTTP mode
```

## 📈 Roadmap

- [ ] **GPU Acceleration**: CUDA/ROCm support for C++ server
- [ ] **Distributed Inference**: Multi-node model serving
- [ ] **Advanced Memory**: Vector similarity search
- [ ] **Plugin System**: Custom tool development
- [ ] **Web UI**: Browser-based management interface
- [ ] **Docker Images**: Containerized deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project inherits the license from the parent Gemma project.

## 🙏 Acknowledgments

- Google's Gemma team for the base model
- MCP Protocol specification contributors
- Original MCP server implementations that informed this consolidation