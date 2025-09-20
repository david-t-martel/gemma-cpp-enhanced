# LLM Agent with Tool Calling

A complete, functional LLM agent system with tool calling capabilities, built using the Gemma model and transformers library.

## Features

- **Tool Calling**: Agent can use various tools to enhance its capabilities
- **Conversation History**: Maintains context across multiple interactions
- **Error Handling**: Robust error handling for tool execution
- **Multiple Tools**: 8+ built-in tools including calculator, file operations, web search, and more
- **Extensible**: Easy to add new tools and customize behavior

## Quick Start

### Installation

```bash
# Install dependencies
uv pip install -r requirements.txt
```

### Running the Agent

```bash
# Run interactive chat with lightweight pipeline (recommended for quick start)
uv run python main.py --lightweight

# Run with full model
uv run python main.py

# Run with specific model
uv run python main.py --model google/gemma-7b-it

# Run with 8-bit quantization (requires CUDA)
uv run python main.py --8bit
```

### Running Tests

```bash
# Run test suite to verify everything works
uv run python test_agent.py

# Run demonstration script
uv run python examples/agent_demo.py
```

## Available Tools

1. **Calculator** - Perform mathematical calculations
2. **Read File** - Read contents from files
3. **Write File** - Write content to files
4. **Web Search** - Search the web (simulated/basic)
5. **Fetch URL** - Extract text from web pages
6. **Get DateTime** - Get current date and time
7. **System Info** - Get system and hardware information
8. **List Directory** - List files and directories

## Project Structure

```
C:\codedev\llm\stats\
├── src/
│   └── agent/
│       ├── __init__.py         # Package initialization
│       ├── core.py             # Base agent class with tool calling
│       ├── tools.py            # Tool definitions and registry
│       └── gemma_agent.py      # Gemma-specific implementation
├── examples/
│   └── agent_demo.py           # Demonstration script
├── main.py                     # Entry point for interactive chat
├── test_agent.py               # Test suite
└── requirements.txt            # Dependencies
```

## Usage Examples

### Interactive Chat
```python
# In interactive mode, you can:
You: Calculate 100 * 50
Agent: [Executes calculator tool] The result is 5000

You: What time is it?
Agent: [Executes datetime tool] Current time is...

You: Write "Hello World" to test.txt
Agent: [Executes write_file tool] Successfully wrote to test.txt
```

### Programmatic Usage
```python
from src.agent.gemma_agent import create_gemma_agent
from src.agent.tools import tool_registry

# Create agent
agent = create_gemma_agent(
    lightweight=True,
    model_name="google/gemma-2b-it",
    verbose=True
)

# Chat with agent
response = agent.chat("Calculate the square root of 144")
print(response)

# Save conversation history
agent.save_history("conversation.json")
```

### Adding Custom Tools
```python
from src.agent.tools import ToolDefinition, ToolParameter, tool_registry

# Define a custom tool
custom_tool = ToolDefinition(
    name="my_tool",
    description="My custom tool",
    parameters=[
        ToolParameter(name="input", type="string", description="Input text")
    ],
    function=lambda input: f"Processed: {input}"
)

# Register the tool
tool_registry.register(custom_tool)
```

## Command-Line Options

```
--model MODEL           Model name to use (default: google/gemma-2b-it)
--lightweight          Use lightweight pipeline version (recommended)
--max-tokens N         Maximum tokens to generate (default: 512)
--temperature T        Sampling temperature (default: 0.7)
--top-p P             Top-p sampling parameter (default: 0.9)
--8bit                Use 8-bit quantization (requires CUDA)
--no-tools            Disable tool calling
--quiet               Reduce verbose output
```

## Chat Commands

When in interactive mode:
- `/help` - Show help message
- `/tools` - List available tools
- `/history` - Show conversation history
- `/save [path]` - Save conversation history
- `/load [path]` - Load conversation history
- `/reset` - Clear conversation history
- `/exit` - Exit the chat

## Architecture

### BaseAgent (core.py)
- Handles conversation management
- Parses tool calls from model output
- Executes tools and manages results
- Maintains conversation history

### ToolRegistry (tools.py)
- Manages available tools
- Executes tool functions
- Provides tool schemas for the model

### GemmaAgent (gemma_agent.py)
- Implements model-specific generation
- Two variants: Full model and Lightweight pipeline
- Handles tokenization and generation

## Troubleshooting

### Model Loading Issues
```bash
# If you encounter model loading issues:
1. Make sure you have enough memory (4GB+ for 2B model)
2. Try using lightweight mode: --lightweight
3. For Gemma models, authenticate with Hugging Face:
   huggingface-cli login
```

### CUDA/GPU Issues
```bash
# If CUDA is not available or causes issues:
1. The system will automatically fall back to CPU
2. You can explicitly use CPU by not using --8bit flag
3. Lightweight mode works well on CPU
```

### Tool Execution Errors
```bash
# Tools have built-in error handling
# Check verbose output for detailed error messages
# Most tools will return error messages instead of crashing
```

## Performance Notes

- **Lightweight mode**: Faster startup, lower memory usage
- **Full model mode**: More control over generation parameters
- **8-bit quantization**: Reduces memory usage by ~50% (GPU only)
- **Tool execution**: Typically < 100ms per tool call
- **Generation speed**: Varies by model size and hardware

## Limitations

- Web search is simulated/basic (can be enhanced with real API)
- Timezone support in datetime tool is simplified
- Model responses depend on model quality and size
- Tool calling patterns must be learned by the model

## Future Enhancements

- Add more sophisticated web search with real APIs
- Implement conversation branching
- Add tool result caching
- Support for streaming responses
- Multi-agent collaboration
- Tool result validation
- Custom tool discovery

## License

This project is provided as-is for educational and development purposes.
