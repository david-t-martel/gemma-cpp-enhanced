# MCP Server Integration Plan for Gemma.cpp

## Overview
Design and implementation plan for integrating Gemma.cpp with Claude Code through the Model Context Protocol (MCP).

## Architecture

### Server Structure
```
gemma/mcp-server/
├── server.py              # Main MCP server
├── gemma_handler.py       # Gemma subprocess management
├── tools.py              # Tool definitions
├── config.py             # Configuration management
├── requirements.txt      # Dependencies
└── README.md            # Setup instructions
```

### Technology Stack
- **Protocol**: MCP JSON-RPC
- **Framework**: Python MCP SDK
- **Async**: asyncio for concurrent handling
- **Process**: subprocess for gemma.exe
- **Streaming**: Server-sent events

## Tool Definitions

### 1. generate_text
```python
{
    "name": "generate_text",
    "description": "Generate text using Gemma models",
    "parameters": {
        "prompt": "string",
        "max_tokens": "integer (optional, default: 512)",
        "temperature": "float (optional, default: 0.7)",
        "model": "string (optional, default: gemma2-2b)"
    }
}
```

### 2. complete_code
```python
{
    "name": "complete_code",
    "description": "Complete code snippets using Gemma",
    "parameters": {
        "code": "string",
        "language": "string (optional)",
        "max_tokens": "integer (optional, default: 256)"
    }
}
```

### 3. analyze_image
```python
{
    "name": "analyze_image",
    "description": "Analyze images using PaliGemma",
    "parameters": {
        "image_path": "string",
        "question": "string (optional)",
        "model": "string (default: paligemma2-3b)"
    }
}
```

### 4. chat
```python
{
    "name": "chat",
    "description": "Interactive chat with context",
    "parameters": {
        "message": "string",
        "conversation_id": "string (optional)",
        "system_prompt": "string (optional)"
    }
}
```

## Implementation Details

### Server Core (server.py)
```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio
from gemma_handler import GemmaHandler

class GemmaServer:
    def __init__(self):
        self.server = Server("gemma")
        self.handler = GemmaHandler()
        self.setup_tools()

    def setup_tools(self):
        @self.server.tool()
        async def generate_text(prompt: str, **kwargs):
            return await self.handler.generate(prompt, **kwargs)

        @self.server.tool()
        async def complete_code(code: str, **kwargs):
            return await self.handler.complete(code, **kwargs)

    async def run(self):
        await self.server.run()
```

### Subprocess Handler (gemma_handler.py)
```python
import asyncio
import subprocess
import json
from pathlib import Path

class GemmaHandler:
    def __init__(self):
        self.executable = Path("../gemma.cpp/build/gemma.exe")
        self.model_path = Path("C:/codedev/llm/.models/")
        self.processes = {}

    async def generate(self, prompt, max_tokens=512, temperature=0.7):
        cmd = [
            str(self.executable),
            "--weights", str(self.model_path / "gemma2-2b-it-sfp.sbs"),
            "--tokenizer", str(self.model_path / "tokenizer.spm"),
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature)
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Stream output handling
        output = []
        async for line in proc.stdout:
            text = line.decode('utf-8')
            output.append(text)
            yield text  # Stream to client

        return ''.join(output)
```

### Configuration (config.py)
```python
import os
from pathlib import Path
from pydantic import BaseModel

class GemmaConfig(BaseModel):
    executable_path: Path = Path("../gemma.cpp/build/gemma.exe")
    model_dir: Path = Path("C:/codedev/llm/.models/")
    default_model: str = "gemma2-2b-it-sfp"
    max_concurrent: int = 5
    timeout: int = 300
    cache_responses: bool = True

    class Config:
        env_prefix = "GEMMA_"

config = GemmaConfig()
```

## MCP Configuration

### Addition to mcp.json
```json
{
    "gemma": {
        "command": "python",
        "args": ["-m", "gemma.mcp-server.server"],
        "cwd": "C:/codedev/llm/gemma",
        "env": {
            "GEMMA_MODEL_DIR": "${env:GEMMA_MODELS}",
            "GEMMA_DEFAULT_MODEL": "gemma2-2b-it-sfp",
            "PYTHONUNBUFFERED": "1"
        }
    }
}
```

## Features

### Streaming Support
- Real-time token streaming
- Progress indicators
- Cancellation support
- Partial response handling

### Conversation Management
- Session persistence
- Context window management
- History truncation
- Memory optimization

### Model Management
- Dynamic model switching
- Model preloading
- Resource cleanup
- Memory monitoring

### Error Handling
- Graceful degradation
- Retry logic
- Timeout management
- Error reporting

## Performance Optimizations

### Process Pool
- Pre-warmed processes
- Process recycling
- Resource limits
- Queue management

### Caching Strategy
- Response caching
- Prompt embedding cache
- Model state caching
- LRU eviction

### Concurrency
- Async request handling
- Request batching
- Priority queuing
- Load balancing

## Testing Strategy

### Unit Tests
```python
# test_server.py
import pytest
from mcp_server import GemmaServer

@pytest.mark.asyncio
async def test_generate_text():
    server = GemmaServer()
    result = await server.generate_text("Hello")
    assert len(result) > 0

@pytest.mark.asyncio
async def test_concurrent_requests():
    server = GemmaServer()
    tasks = [server.generate_text(f"Test {i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
```

### Integration Tests
- End-to-end MCP flow
- Claude Code integration
- Error scenarios
- Performance benchmarks

### Validation
```bash
# Validate with MCP inspector
npx @modelcontextprotocol/inspector --cli --server gemma

# Test specific tools
mcp-client test gemma generate_text --prompt "Test"
```

## Deployment

### Installation Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables
3. Add to mcp.json
4. Restart Claude Code
5. Verify connection

### Environment Variables
```bash
export GEMMA_MODEL_DIR="/c/codedev/llm/.models"
export GEMMA_DEFAULT_MODEL="gemma2-2b-it-sfp"
export GEMMA_MAX_CONCURRENT=5
export GEMMA_CACHE_SIZE=100
```

### Monitoring
- Request metrics
- Response times
- Error rates
- Resource usage
- Model performance

## Security Considerations

### Input Validation
- Prompt sanitization
- Size limits
- Content filtering
- Injection prevention

### Resource Protection
- Rate limiting
- Memory limits
- CPU throttling
- Disk usage caps

### Access Control
- Authentication (if needed)
- Authorization checks
- Audit logging
- Session management

## Future Enhancements

### Phase 1 (Immediate)
- Basic text generation
- Simple chat interface
- Error handling

### Phase 2 (Week 1)
- Streaming responses
- Conversation management
- Model switching

### Phase 3 (Week 2)
- Vision-language support
- Code completion
- Performance optimization

### Phase 4 (Future)
- Multi-model ensemble
- Fine-tuning interface
- Custom tools
- Analytics dashboard

## Success Metrics
- [ ] MCP server starts without errors
- [ ] Tools visible in Claude Code
- [ ] Text generation works
- [ ] Streaming responses functional
- [ ] Error handling robust
- [ ] Performance acceptable (<100ms overhead)
- [ ] Concurrent requests handled
- [ ] Memory usage stable
- [ ] Documentation complete