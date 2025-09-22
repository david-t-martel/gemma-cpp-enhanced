# Gemma.cpp MCP Server

This directory contains the consolidated MCP (Model Context Protocol) server implementation for Gemma.cpp.

## Overview

The MCP server provides a standardized interface for Gemma.cpp inference capabilities through the Model Context Protocol, enabling integration with MCP-compatible clients and tools.

## Features

- **Text Generation**: Generate text using Gemma models with customizable parameters
- **Token Counting**: Count tokens in text using the Gemma tokenizer
- **Model Information**: Get detailed information about loaded models
- **Protocol Support**: JSON-RPC 2.0 communication via multiple transports
- **Streaming Support**: Real-time token streaming during generation
- **Multiple Model Support**: Compatible with all Gemma model variants

## Directory Structure

```
mcp/
├── server/           # Core MCP server implementation
│   ├── MCPServer.cpp/h      # Main server class
│   ├── MCPProtocol.cpp/h    # MCP protocol implementation
│   ├── MCPTransport.cpp/h   # Transport layer (stdio/websocket)
│   └── MCPTools.cpp/h       # Tool implementations
└── README.md         # This file
```

## Tools Available

1. **generate_text** - Generate text from prompts with configurable parameters
2. **count_tokens** - Count tokens in input text using Gemma tokenizer
3. **get_model_info** - Retrieve model metadata and configuration

## Usage

The MCP server is integrated into the main Gemma.cpp build system and can be invoked through the main CLI interface or used as a standalone MCP server.

## Integration

This implementation consolidates the best features from multiple MCP implementations that were previously scattered across:
- `mcp/` (original C++ implementation)
- `mcp-server/` (Python implementation)
- `src/interfaces/mcp/` (enhanced C++ implementation)

The final implementation uses the most complete C++ codebase with full MCP protocol support.