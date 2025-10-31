# Tool Orchestration for Autonomous LLM Tool Calling

## Overview

The Gemma CLI now supports **autonomous tool calling**, enabling the LLM to automatically detect when external tools are needed and execute them to provide better answers. This feature integrates with the MCP (Model Context Protocol) framework to provide access to filesystem operations, memory storage, web search, and more.

## Architecture

### Components

1. **ToolOrchestrator** (`core/tool_orchestrator.py`)
   - Main orchestration engine
   - Manages tool discovery and execution
   - Enforces depth limits and safety checks
   - Handles multi-turn tool chains

2. **ToolSchemaFormatter**
   - Converts MCP tool schemas to LLM-friendly format
   - Generates system prompts with tool instructions
   - Supports multiple formats (JSON blocks, XML tags, etc.)

3. **ToolCallParser**
   - Detects tool calls in LLM responses
   - Parses JSON-formatted tool requests
   - Extracts server, tool name, and arguments

4. **Integration with Chat Loop** (`cli.py`)
   - Automatic tool discovery at startup
   - System prompt injection with tool descriptions
   - Response processing with tool execution
   - Result integration back into conversation

## How It Works

### 1. Startup Phase
When you start the chat with `--enable-mcp`:
- MCP servers are connected
- Available tools are discovered
- Tool schemas are formatted for the LLM
- System prompt is generated with tool instructions

### 2. Query Processing
When the user asks a question:
1. Query is sent to the LLM with tool instructions
2. LLM generates response, potentially including tool calls
3. Tool calls are detected and parsed from the response
4. Tools are executed via the MCP framework
5. Results are fed back to the LLM
6. Final response is generated using tool outputs

### 3. Tool Call Format
The LLM formats tool calls as JSON code blocks:

```json
{
  "tool": "filesystem.read_file",
  "args": {
    "path": "/path/to/file.txt"
  }
}
```

## Available Tools

### Filesystem Tools
- `filesystem.read_file` - Read file contents
- `filesystem.list_directory` - List directory contents
- `filesystem.write_file` - Write to files (with confirmation)

### Memory Tools
- `memory.store_memory` - Store information for later
- `memory.recall_memory` - Search stored memories

### Search Tools
- `brave_search.web_search` - Search the web
- `fetch.fetch_url` - Fetch content from URLs

### GitHub Tools
- `github.create_issue` - Create GitHub issues
- `github.list_repos` - List repositories

## Usage Examples

### Basic Usage
```bash
# Start chat with MCP tools enabled
gemma-cli chat --enable-mcp

# Ask questions that require tools
You: What's in my README.md file?
Assistant: Let me read that file for you...
[Tool executes automatically]
Assistant: Your README contains...
```

### Example Queries

1. **File Operations**
   - "What's in my config.json file?"
   - "List all Python files in the src directory"
   - "Show me the contents of main.py"

2. **Web Search**
   - "What's the latest Python release?"
   - "Search for information about LLM tool calling"
   - "Find documentation for the MCP protocol"

3. **Memory Operations**
   - "Remember that my API key is in the .env file"
   - "What did I tell you about the project structure?"
   - "Recall our discussion about authentication"

4. **Multi-Tool Chains**
   - "Search for Python docs and summarize the async/await section"
   - "Read my TODO.md and create GitHub issues for each item"
   - "Check what version is in package.json and search for updates"

## Configuration

### Enabling/Disabling Features

```python
# In cli.py or configuration
tool_orchestrator = ToolOrchestrator(
    mcp_manager=mcp_manager,
    available_tools=tools,
    format_type=ToolCallFormat.JSON_BLOCK,
    max_tool_depth=5,  # Maximum chain depth
    require_confirmation=False  # Set True for safety
)
```

### Tool Depth Limits
- Default: 5 tool calls per response
- Prevents infinite loops
- Configurable per session

### Confirmation Mode
- Can require user confirmation before tool execution
- Useful for write operations or sensitive actions
- Enable with `require_confirmation=True`

## Safety Features

1. **Path Validation**
   - All file paths are validated
   - Prevents directory traversal attacks
   - Restricts access to allowed directories

2. **Depth Limiting**
   - Maximum tool calls per response
   - Prevents runaway tool chains
   - Configurable limits

3. **Error Handling**
   - Graceful failure on tool errors
   - Clear error messages to user
   - Fallback to non-tool responses

4. **Resource Limits**
   - Timeout on tool execution
   - Memory limits for file operations
   - Rate limiting for API calls

## Testing

Run the test suite:
```bash
cd src/gemma_cli
uv run python test_tool_orchestration.py
```

Tests cover:
- Schema formatting
- Tool call parsing
- Multi-tool chains
- Depth limiting
- Error handling

## Troubleshooting

### Tools Not Working
1. Check MCP servers are running: `gemma-cli mcp status`
2. Verify tool availability: Use `/tools` command in chat
3. Check server logs: `~/.gemma_cli/logs/mcp.log`

### Tool Calls Not Detected
1. Ensure `--enable-mcp` flag is used
2. Check LLM is using correct format (JSON blocks)
3. Verify system prompt includes tool instructions

### Performance Issues
1. Reduce `max_tool_depth` if chains are too long
2. Enable tool caching in MCP client
3. Use local tools (filesystem, memory) when possible

## Future Enhancements

1. **More Tool Formats**
   - OpenAI function calling format
   - Claude tool use format
   - Custom formats for specific models

2. **Advanced Orchestration**
   - Parallel tool execution
   - Conditional tool chains
   - Tool result caching

3. **Additional Integrations**
   - Database operations
   - API integrations
   - Custom tool development

4. **Enhanced Safety**
   - Sandboxed execution
   - Permission systems
   - Audit logging

## API Reference

### ToolOrchestrator

```python
orchestrator = ToolOrchestrator(
    mcp_manager: MCPClientManager,
    available_tools: dict[str, list[Tool]],
    format_type: ToolCallFormat = ToolCallFormat.JSON_BLOCK,
    max_tool_depth: int = 5,
    require_confirmation: bool = False
)

# Get system prompt with tool instructions
system_prompt = orchestrator.get_system_prompt()

# Process response with tool execution
final_response, results = await orchestrator.process_response_with_tools(
    response: str,
    console: Optional[Console] = None,
    confirm_callback: Optional[Callable] = None
)
```

### ToolCallParser

```python
parser = ToolCallParser(format_type: ToolCallFormat)

# Parse tool calls from response
tool_calls: list[ToolCall] = parser.parse_response(response: str)
```

### ToolSchemaFormatter

```python
formatter = ToolSchemaFormatter(format_type: ToolCallFormat)

# Format tools for prompt
prompt = formatter.format_tools_for_prompt(
    tools_by_server: dict[str, list[Tool]]
)
```

## Contributing

To add new tool integrations:

1. Implement MCP server with tool definitions
2. Add server configuration to `mcp.json`
3. Test with `test_tool_orchestration.py`
4. Document tool usage and examples

## License

This feature is part of the Gemma CLI project and follows the same licensing terms.