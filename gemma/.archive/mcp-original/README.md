# Gemma.cpp MCP Server

This directory contains a complete MCP (Model Context Protocol) server implementation for Gemma.cpp, providing a standardized interface for text generation and model interaction.

## Overview

The MCP server exposes Gemma.cpp's inference capabilities through the Model Context Protocol, enabling integration with MCP-compatible clients and tools.

### Features

- **Text Generation**: Generate text using Gemma models with customizable parameters
- **Token Counting**: Count tokens in text using the Gemma tokenizer
- **Model Information**: Get detailed information about loaded models
- **Stdio Transport**: JSON-RPC 2.0 communication via stdin/stdout
- **Streaming Support**: Real-time token streaming during generation
- **Multiple Model Support**: Compatible with all Gemma model variants

### Supported Tools

1. **generate_text** - Generate text from prompts
   - Parameters: prompt, temperature, max_tokens, top_k, top_p
   - Returns: generated text, timing information, token counts

2. **count_tokens** - Tokenize and count tokens in text
   - Parameters: text, include_details (optional)
   - Returns: token count, token IDs, individual token details

3. **get_model_info** - Get information about the loaded model
   - Parameters: none
   - Returns: model name, architecture, configuration details

## Architecture

```
mcp/
├── server/
│   ├── mcp_server.h/.cpp           # Main MCP server implementation
│   ├── mcp_stdio_server.cpp        # Stdio transport server (for testing)
│   ├── inference_handler.h/.cpp    # Text generation and inference logic
│   ├── model_manager.h/.cpp        # Model loading and tokenization
│   └── main.cpp                    # WebSocket server entry point
├── CMakeLists.txt                  # Build configuration
└── README.md                       # This file
```

### Core Components

- **MCPServer**: Main server class handling JSON-RPC 2.0 protocol
- **InferenceHandler**: Manages text generation using Gemma C++ API
- **ModelManager**: Handles model loading, tokenization, and model info
- **MCPStdioServer**: Stdio transport implementation for testing

## Building

### Prerequisites

- CMake 3.16+
- C++20 compatible compiler
- Gemma.cpp dependencies (Highway, SentencePiece, etc.)

### Build Instructions

```bash
# From the main project directory
cmake -S . -B build -DGEMMA_BUILD_MCP_SERVER=ON
cmake --build build --target gemma_mcp_stdio_server

# Or build with existing Gemma.cpp build
cd gemma.cpp
cmake -S . -B build
cmake --build build --target libgemma
cd ../mcp
cmake -S . -B build -Dgemma_ROOT=../gemma.cpp/build
cmake --build build
```

## Usage

### Basic Usage

```bash
# Run the MCP server with a model
./gemma_mcp_stdio_server --model /path/to/model.sbs --tokenizer /path/to/tokenizer.spm

# For single-file models (with embedded tokenizer)
./gemma_mcp_stdio_server --model /path/to/model-single.sbs
```

### Command Line Options

```
Required:
  --model PATH          Path to model weights file

Optional:
  --tokenizer PATH      Path to tokenizer (for separate tokenizer files)
  --temperature F       Default temperature (default: 0.7)
  --max-tokens N        Default max tokens (default: 1024)
  --help               Show help message
```

### MCP Client Integration

The server communicates using JSON-RPC 2.0 over stdio. Example interaction:

```json
// Initialize
{"jsonrpc": "2.0", "id": "1", "method": "initialize", "params": {}}

// List available tools
{"jsonrpc": "2.0", "id": "2", "method": "tools/list", "params": {}}

// Generate text
{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "tools/call",
  "params": {
    "name": "generate_text",
    "arguments": {
      "prompt": "The future of AI is",
      "temperature": 0.8,
      "max_tokens": 100
    }
  }
}

// Count tokens
{
  "jsonrpc": "2.0",
  "id": "4",
  "method": "tools/call",
  "params": {
    "name": "count_tokens",
    "arguments": {
      "text": "Hello, world!",
      "include_details": true
    }
  }
}
```

## Integration with Gemma.cpp

The MCP server integrates directly with the Gemma.cpp inference engine:

### Model Loading
- Supports both separate tokenizer files and single-file models
- Auto-detects model architecture and configuration
- Handles various weight formats (SFP, BF16, etc.)

### Text Generation
- Uses Gemma's `Generate()` method with proper configuration
- Supports streaming with token callbacks
- Configurable sampling parameters (temperature, top-k, top-p)
- Proper timing and performance metrics

### Tokenization
- Direct integration with `GemmaTokenizer`
- Token counting and detailed token information
- Encoding/decoding with proper error handling

## Error Handling

The server implements comprehensive error handling:

- **Model Loading Errors**: Invalid paths, corrupted files, memory issues
- **Generation Errors**: Invalid parameters, context length exceeded
- **Protocol Errors**: Malformed JSON-RPC, missing parameters
- **Resource Errors**: Out of memory, threading issues

All errors are returned as proper JSON-RPC 2.0 error responses with descriptive messages.

## Performance Considerations

- **Memory Efficiency**: Uses Gemma's memory-mapped file loading
- **Threading**: Leverages Gemma's threading context for optimal performance
- **Batch Processing**: Supports batch inference for multiple requests
- **Streaming**: Minimal latency for real-time token generation

## Example Model Configurations

### Gemma-2B (Recommended for Testing)
```bash
./gemma_mcp_stdio_server \
  --model /models/gemma2-2b-it-sfp.sbs \
  --tokenizer /models/tokenizer.spm \
  --temperature 0.7 \
  --max-tokens 1024
```

### Gemma-9B (Higher Quality)
```bash
./gemma_mcp_stdio_server \
  --model /models/gemma2-9b-it-sfp.sbs \
  --tokenizer /models/tokenizer.spm \
  --temperature 0.8 \
  --max-tokens 2048
```

### Single-File Model (Easiest)
```bash
./gemma_mcp_stdio_server \
  --model /models/gemma2-2b-it-sfp-single.sbs \
  --temperature 0.7
```

## Development

### Adding New Tools

1. Define the tool in `MCPStdioServer::HandleToolsList()`
2. Add handler in `MCPStdioServer::HandleToolsCall()`
3. Implement logic in `InferenceHandler` or `ModelManager`
4. Update tool schemas with proper JSON schema validation

### Testing

```bash
# Test with echo client
echo '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}' | ./gemma_mcp_stdio_server --model /path/to/model.sbs

# Test text generation
echo '{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"generate_text","arguments":{"prompt":"Hello"}}}' | ./gemma_mcp_stdio_server --model /path/to/model.sbs
```

## Troubleshooting

### Common Issues

1. **Model not found**: Verify model path exists and is readable
2. **Tokenizer errors**: Ensure tokenizer path is correct for non-single-file models
3. **Memory errors**: Reduce model size or increase available memory
4. **Performance issues**: Use SFP models, check CPU performance mode

### Debug Options

- Enable verbose logging in `MCPServer::Config`
- Check stderr for detailed error messages
- Verify JSON-RPC message format
- Test with minimal prompts first

## Contributing

When contributing to the MCP server:

1. Follow C++20 best practices
2. Maintain compatibility with Gemma.cpp API
3. Add comprehensive error handling
4. Include proper JSON schema definitions
5. Test with various model sizes and formats

## License

This MCP server implementation follows the same Apache 2.0 license as Gemma.cpp.