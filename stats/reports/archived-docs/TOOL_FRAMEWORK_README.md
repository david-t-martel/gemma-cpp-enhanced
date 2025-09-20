# LLM Agent Tool Calling and Sandbox Framework

A comprehensive, production-ready framework for secure tool execution, parallel agent orchestration, and distributed tool hosting using the Model Context Protocol (MCP).

## 🚀 Features

### Core Components

- **🔧 Tool Framework**: Type-safe tool definitions with JSON schema validation
- **🛡️ Security Sandboxes**: Docker and process-based isolation for safe code execution
- **⚡ Parallel Orchestration**: Multi-agent task execution with workflows and dependencies
- **🌐 MCP Protocol**: Distributed tool hosting and client-server communication
- **📊 Built-in Tools**: File operations, web search, data analysis, and more
- **🔒 Security Levels**: Configurable security restrictions from minimal to maximum
- **📈 Performance Monitoring**: Resource usage tracking and execution metrics

### Architecture

```
src/
├── domain/tools/           # Core tool interfaces and schemas
├── infrastructure/
│   ├── sandbox/           # Docker and process sandboxes
│   ├── tools/            # Built-in tool implementations
│   └── mcp/              # MCP client and server
├── application/agents/    # Agent orchestration system
└── examples/             # Comprehensive usage examples
```

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Optional: Set up Docker** (for Docker sandbox):
   ```bash
   docker --version  # Ensure Docker is installed and running
   ```

## 📖 Quick Start

### Basic Tool Usage

```python
import asyncio
from src.domain.tools import get_global_registry, ToolExecutionContext
from src.infrastructure.tools import register_builtin_tools

async def main():
    # Get the global tool registry
    registry = get_global_registry()

    # Register built-in tools
    await register_builtin_tools()

    # Create execution context
    context = ToolExecutionContext(
        timeout=30,
        security_level="standard"
    )

    # Execute a tool
    result = await registry.execute_tool(
        "read_file",
        context,
        path="example.txt"
    )

    print(f"Success: {result.success}")
    if result.success:
        print(f"Content: {result.data}")

asyncio.run(main())
```

### Custom Tool Definition

```python
from src.domain.tools import BaseTool, tool, EnhancedToolSchema, ParameterSchema

@tool(name="calculator", description="Basic calculator")
class CalculatorTool(BaseTool):
    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="calculator",
            description="Perform basic arithmetic operations",
            parameters=[
                ParameterSchema(
                    name="operation",
                    type="string",
                    description="Arithmetic operation",
                    enum=["add", "subtract", "multiply", "divide"],
                    required=True
                ),
                ParameterSchema(
                    name="a",
                    type="number",
                    description="First number",
                    required=True
                ),
                ParameterSchema(
                    name="b",
                    type="number",
                    description="Second number",
                    required=True
                )
            ]
        )

    async def execute(self, context, **kwargs):
        op = kwargs["operation"]
        a, b = kwargs["a"], kwargs["b"]

        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }

        result = operations[op](a, b)

        return ToolResult(
            success=result is not None,
            data={"result": result},
            error="Division by zero" if result is None else None,
            execution_time=0,
            context=context
        )
```

### Secure Code Execution

```python
from src.infrastructure.sandbox import ProcessSandbox, DockerSandbox

# Process-based sandbox
process_sandbox = ProcessSandbox()
await process_sandbox.initialize()

result = await process_sandbox.execute_code(
    code="print('Hello from sandbox!')",
    language="python",
    context=context
)

print(f"Output: {result.stdout}")
print(f"Execution time: {result.execution_time}s")

# Docker-based sandbox (more secure)
docker_sandbox = DockerSandbox()
await docker_sandbox.initialize()

result = await docker_sandbox.execute_code(
    code="import numpy as np; print(np.array([1,2,3]))",
    language="python",
    requirements=["numpy"]
)
```

### Agent Orchestration

```python
from src.application.agents import (
    AgentOrchestrator, AgentTask, Workflow, WorkflowStep,
    ExecutionMode, TaskPriority
)

# Create orchestrator
orchestrator = AgentOrchestrator(tool_registry=registry)
await orchestrator.initialize()

# Create tasks
tasks = [
    AgentTask(
        name="analyze_data",
        tool_name="data_analyzer",
        parameters={"data": [1, 2, 3, 4, 5]},
        priority=TaskPriority.HIGH
    ),
    AgentTask(
        name="process_file",
        tool_name="read_file",
        parameters={"path": "data.txt"},
        dependencies=["analyze_data"]  # Runs after analyze_data
    )
]

# Execute tasks in parallel
results = await orchestrator.execute_tasks(tasks)

# Or create a workflow
workflow = Workflow(
    name="data_pipeline",
    steps=[
        WorkflowStep(
            name="parallel_analysis",
            tasks=tasks[:3],
            mode=ExecutionMode.PARALLEL
        ),
        WorkflowStep(
            name="final_processing",
            tasks=tasks[3:],
            depends_on=["parallel_analysis"],
            mode=ExecutionMode.SEQUENTIAL
        )
    ]
)

results = await orchestrator.execute_workflow(workflow)
```

### MCP Server/Client

```python
from src.infrastructure.mcp import create_mcp_server, McpClient, McpServerConfig

# Create and start MCP server
server = await create_mcp_server(
    name="my-tool-server",
    tool_registry=registry,
    host="localhost",
    port=8000,
    protocol="websocket"
)

await server.start()

# Connect client to server
client_config = McpServerConfig(
    name="remote-server",
    url="ws://localhost:8000",
    protocol="websocket"
)

client = McpClient(client_config)
await client.connect()

# Use remote tools
response = await client.call_tool(
    "remote_calculator",
    {"operation": "add", "a": 5, "b": 3}
)
```

## 🛡️ Security Features

### Security Levels

- **Minimal**: Basic validation only
- **Standard**: Process isolation, resource limits
- **Strict**: No network access, read-only filesystem
- **Maximum**: Minimal resources, strict containerization

### Sandbox Comparison

| Feature | Process Sandbox | Docker Sandbox |
|---------|----------------|-----------------|
| Startup Speed | ⚡ Fast | 🐌 Slower |
| Isolation | 🔒 Good | 🔒🔒 Excellent |
| Resource Control | ✅ Yes | ✅ Yes |
| Network Isolation | ⚠️ Limited | ✅ Complete |
| Language Support | 🌐 Wide | 🌐 Wide |
| Setup Complexity | 💚 Simple | 🟡 Moderate |

### Resource Limits

```python
from src.domain.tools.schemas import ResourceLimits

limits = ResourceLimits(
    max_memory_mb=256,
    max_cpu_percent=50.0,
    max_execution_time_seconds=30,
    max_file_size_mb=10,
    max_network_requests=5
)
```

## 📊 Built-in Tools

### File Operations
- `read_file`: Read file contents with encoding detection
- `write_file`: Write files with backup and atomic operations
- `list_directory`: List directory contents with metadata

### Web & Network
- `web_search`: Multi-engine web search with safe search
- `http_request`: HTTP requests with security controls

### Data Processing
- `hash_text`: Cryptographic hashing (MD5, SHA256, etc.)
- `data_analyzer`: Statistical analysis of numerical data

### Code Execution
- `python_code_executor`: Secure Python code execution
- Support for multiple languages (JS, Java, Rust, Go, etc.)

## 🔧 Configuration

### Tool Schema Validation

```python
from src.domain.tools.schemas import ParameterSchema

param = ParameterSchema(
    name="email",
    type="email",           # Built-in email validation
    description="User email address",
    required=True,
    pattern=r'^[^@]+@[^@]+\.[^@]+$',  # Additional regex validation
    examples=["user@example.com"]
)
```

### Execution Context

```python
context = ToolExecutionContext(
    execution_id="custom-id",
    agent_id="data-agent",
    timeout=60,
    retry_count=2,
    cache_enabled=True,
    sandbox_mode=True,
    security_level="strict",
    metadata={"source": "api", "user_id": "123"}
)
```

## 🚀 Performance

### Benchmarks (approximate)

- **Tool Registration**: ~1ms per tool
- **Parameter Validation**: ~0.1ms per parameter
- **Process Sandbox**: ~50-100ms startup
- **Docker Sandbox**: ~500-1000ms startup
- **Agent Task Routing**: ~1-5ms per task
- **Memory Usage**: ~50-100MB base + sandboxes

### Optimization Tips

1. **Reuse sandboxes** for multiple executions
2. **Enable caching** for idempotent operations
3. **Use process sandbox** for faster startup
4. **Batch similar tasks** in workflows
5. **Set appropriate timeouts** to prevent hanging

## 🔍 Monitoring & Debugging

### Execution Metrics

```python
# Get system statistics
stats = await orchestrator.get_system_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average execution time: {stats['average_execution_time']:.2f}s")

# Agent status
agent_status = await orchestrator.get_agent_status()
for agent in agent_status:
    print(f"Agent {agent['name']}: {agent['state']}")
```

### Logging

```python
import logging

# Configure framework logging
logging.getLogger('src.domain.tools').setLevel(logging.DEBUG)
logging.getLogger('src.infrastructure.sandbox').setLevel(logging.INFO)
logging.getLogger('src.application.agents').setLevel(logging.INFO)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
uv run python test_comprehensive_framework.py
```

Or run the example demonstrations:

```bash
uv run python src/examples/comprehensive_example.py
```

## 🤝 Contributing

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Implement `schema` property with `EnhancedToolSchema`
3. Implement `execute` method
4. Use `@tool` decorator for auto-registration
5. Add tests and documentation

### Adding New Sandboxes

1. Create class with `execute_code` and `execute_command` methods
2. Implement security restrictions and resource limits
3. Add to orchestrator initialization
4. Document security characteristics

## 🐛 Troubleshooting

### Common Issues

**Docker sandbox fails to start:**
- Ensure Docker daemon is running
- Check Docker permissions for current user
- Verify image availability

**Permission errors in process sandbox:**
- Check file/directory permissions
- Ensure user has required system access
- Consider running with elevated privileges (carefully)

**MCP connection failures:**
- Verify server is running and accessible
- Check firewall settings
- Validate authentication tokens

**Tool execution timeouts:**
- Increase timeout values in context
- Check resource limits
- Monitor system resource usage

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug execution context
context = ToolExecutionContext(
    timeout=300,  # Longer timeout
    security_level="minimal",  # Reduced restrictions
    metadata={"debug": True}
)
```

## 📄 License

This framework is part of the LLM Stats project. See the main project license for details.

## 🙏 Acknowledgments

- Built on modern Python async/await patterns
- Inspired by OpenAI function calling and Anthropic tool use
- Uses industry-standard security practices
- Follows MCP specification for distributed tools
