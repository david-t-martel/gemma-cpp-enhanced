"""Comprehensive example demonstrating the tool calling and sandbox framework."""

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any
from typing import Dict

from src.application.agents import Agent
from src.application.agents import AgentCapability
from src.application.agents import AgentOrchestrator
from src.application.agents import AgentTask
from src.application.agents import ExecutionMode
from src.application.agents import ExecutionPlan
from src.application.agents import TaskPriority
from src.application.agents import Workflow
from src.application.agents import WorkflowStep
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Framework imports
from src.domain.tools import BaseTool
from src.domain.tools import EnhancedToolSchema
from src.domain.tools import ParameterSchema
from src.domain.tools import ResourceLimits
from src.domain.tools import SecurityLevel
from src.domain.tools import ToolCategory
from src.domain.tools import ToolExecutionContext
from src.domain.tools import ToolResult
from src.domain.tools import ToolType
from src.domain.tools import get_global_registry
from src.domain.tools import tool
from src.infrastructure.mcp import McpClient
from src.infrastructure.mcp import McpServer
from src.infrastructure.mcp import McpServerConfig
from src.infrastructure.mcp import create_mcp_server
from src.infrastructure.sandbox import DockerSandbox
from src.infrastructure.sandbox import ProcessSandbox
from src.infrastructure.tools import register_builtin_tools

# Setup logging


@tool(
    name="python_code_executor",
    description="Execute Python code in a secure sandbox",
    category=ToolCategory.CODE_EXECUTION,
)
class PythonCodeExecutorTool(BaseTool):
    """Custom tool for executing Python code."""

    def __init__(self, docker_sandbox: DockerSandbox, process_sandbox: ProcessSandbox):
        super().__init__()
        self.docker_sandbox = docker_sandbox
        self.process_sandbox = process_sandbox

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="python_code_executor",
            description="Execute Python code in a secure sandboxed environment",
            category=ToolCategory.CODE_EXECUTION,
            type=ToolType.CUSTOM,
            parameters=[
                ParameterSchema(
                    name="code",
                    type="code",
                    description="Python code to execute",
                    required=True,
                    examples=["print('Hello, World!')", "import math; print(math.pi)"],
                ),
                ParameterSchema(
                    name="sandbox_type",
                    type="string",
                    description="Type of sandbox to use",
                    required=False,
                    default="process",
                    enum=["docker", "process"],
                ),
                ParameterSchema(
                    name="timeout",
                    type="integer",
                    description="Execution timeout in seconds",
                    required=False,
                    default=30,
                    minimum=1,
                    maximum=300,
                ),
                ParameterSchema(
                    name="requirements",
                    type="array",
                    description="Python packages to install",
                    required=False,
                    examples=[["numpy", "pandas"], ["requests"]],
                ),
            ],
            security=SecurityLevel(
                level="strict",
                sandbox_required=True,
                network_access=False,
                file_system_access=False,
                process_isolation=True,
            ),
            resource_limits=ResourceLimits(
                max_memory_mb=256, max_cpu_percent=50.0, max_execution_time_seconds=300
            ),
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute Python code in sandbox."""
        try:
            code = kwargs["code"]
            sandbox_type = kwargs.get("sandbox_type", "process")
            timeout = kwargs.get("timeout", 30)
            requirements = kwargs.get("requirements", [])

            logger.info(f"Executing Python code using {sandbox_type} sandbox")

            if sandbox_type == "docker":
                result = await self.docker_sandbox.execute_code(
                    code=code, language="python", context=context, requirements=requirements
                )

                return ToolResult(
                    success=result.success,
                    data={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code,
                        "execution_time": result.execution_time,
                        "resource_usage": result.resource_usage,
                        "sandbox_type": "docker",
                    },
                    error=result.stderr if not result.success else None,
                    execution_time=result.execution_time,
                    context=context,
                )
            else:
                result = await self.process_sandbox.execute_code(
                    code=code, language="python", context=context, requirements=requirements
                )

                return ToolResult(
                    success=result.success,
                    data={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code,
                        "execution_time": result.execution_time,
                        "resource_usage": result.resource_usage,
                        "sandbox_type": "process",
                    },
                    error=result.stderr if not result.success else None,
                    execution_time=result.execution_time,
                    context=context,
                )

        except Exception as e:
            logger.error(f"Python code execution failed: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)


@tool(
    name="data_analyzer",
    description="Analyze data using various statistical methods",
    category=ToolCategory.DATA_PROCESSING,
)
class DataAnalyzerTool(BaseTool):
    """Tool for analyzing data."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="data_analyzer",
            description="Analyze numerical data using statistical methods",
            category=ToolCategory.DATA_PROCESSING,
            type=ToolType.CUSTOM,
            parameters=[
                ParameterSchema(
                    name="data",
                    type="array",
                    description="Array of numerical data",
                    required=True,
                    examples=[[1, 2, 3, 4, 5], [10.5, 20.1, 30.7]],
                ),
                ParameterSchema(
                    name="analysis_type",
                    type="string",
                    description="Type of analysis to perform",
                    required=False,
                    default="basic",
                    enum=["basic", "descriptive", "correlation"],
                ),
                ParameterSchema(
                    name="round_digits",
                    type="integer",
                    description="Number of decimal places to round results",
                    required=False,
                    default=2,
                    minimum=0,
                    maximum=10,
                ),
            ],
            idempotent=True,
            cacheable=True,
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute data analysis."""
        try:
            import statistics

            data = kwargs["data"]
            analysis_type = kwargs.get("analysis_type", "basic")
            round_digits = kwargs.get("round_digits", 2)

            if not data or not all(isinstance(x, (int, float)) for x in data):
                return ToolResult(
                    success=False,
                    error="Invalid data: must be array of numbers",
                    execution_time=0,
                    context=context,
                )

            result = {}

            if analysis_type in ["basic", "descriptive"]:
                result.update(
                    {
                        "count": len(data),
                        "mean": round(statistics.mean(data), round_digits),
                        "median": round(statistics.median(data), round_digits),
                        "mode": statistics.multimode(data),
                        "min": min(data),
                        "max": max(data),
                        "range": round(max(data) - min(data), round_digits),
                    }
                )

            if analysis_type == "descriptive" and len(data) > 1:
                result.update(
                    {
                        "variance": round(statistics.variance(data), round_digits),
                        "std_deviation": round(statistics.stdev(data), round_digits),
                        "q1": round(statistics.quantiles(data, n=4)[0], round_digits)
                        if len(data) >= 4
                        else None,
                        "q3": round(statistics.quantiles(data, n=4)[2], round_digits)
                        if len(data) >= 4
                        else None,
                    }
                )

            return ToolResult(
                success=True,
                data=result,
                execution_time=0,  # Immediate for statistical calculations
                context=context,
                cached=False,
            )

        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)


async def demonstrate_basic_tools():
    """Demonstrate basic tool functionality."""
    print("\n=== Basic Tool Functionality ===")

    # Get the global registry
    registry = get_global_registry()

    # Register built-in tools
    await register_builtin_tools()

    # Create sandboxes
    docker_sandbox = DockerSandbox()
    process_sandbox = ProcessSandbox()
    await process_sandbox.initialize()

    # Register custom tools
    python_tool = PythonCodeExecutorTool(docker_sandbox, process_sandbox)
    data_tool = DataAnalyzerTool()

    await registry.register(python_tool)
    await registry.register(data_tool)

    print(f"Registered {len(registry.list_tools())} tools:")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")

    # Test Python code execution
    print("\n--- Testing Python Code Execution ---")
    python_task = ToolExecutionContext(timeout=30, security_level="strict")

    python_result = await registry.execute_tool(
        "python_code_executor",
        python_task,
        code="print('Hello from sandbox!')\nprint(2 + 2)\nimport sys\nprint(f'Python version: {sys.version}')",
        sandbox_type="process",
    )

    print(f"Success: {python_result.success}")
    if python_result.success:
        print(f"Output: {python_result.data['stdout']}")
    else:
        print(f"Error: {python_result.error}")

    # Test data analysis
    print("\n--- Testing Data Analysis ---")
    analysis_task = ToolExecutionContext()

    analysis_result = await registry.execute_tool(
        "data_analyzer",
        analysis_task,
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        analysis_type="descriptive",
    )

    print(f"Success: {analysis_result.success}")
    if analysis_result.success:
        print("Analysis results:")
        for key, value in analysis_result.data.items():
            print(f"  {key}: {value}")

    # Test file operations
    print("\n--- Testing File Operations ---")
    file_task = ToolExecutionContext()

    # Write a test file
    write_result = await registry.execute_tool(
        "write_file",
        file_task,
        path="test_output.txt",
        content="Hello from the LLM Agent Framework!\nThis is a test file.",
    )

    print(f"Write success: {write_result.success}")

    # Read the file back
    read_result = await registry.execute_tool("read_file", file_task, path="test_output.txt")

    print(f"Read success: {read_result.success}")
    if read_result.success:
        print(f"File content: {read_result.data['content']}")


async def demonstrate_agent_orchestration():
    """Demonstrate agent orchestration."""
    print("\n=== Agent Orchestration ===")

    registry = get_global_registry()

    # Create orchestrator
    orchestrator = AgentOrchestrator(tool_registry=registry, max_agents=5)
    await orchestrator.initialize()

    # Create some tasks
    tasks = [
        AgentTask(
            name="analyze_sales_data",
            description="Analyze sales data for trends",
            tool_name="data_analyzer",
            parameters={
                "data": [100, 150, 200, 180, 220, 250, 300],
                "analysis_type": "descriptive",
            },
            priority=TaskPriority.HIGH,
        ),
        AgentTask(
            name="analyze_customer_data",
            description="Analyze customer satisfaction scores",
            tool_name="data_analyzer",
            parameters={"data": [4.2, 3.8, 4.5, 4.1, 3.9, 4.3, 4.0], "analysis_type": "basic"},
            priority=TaskPriority.NORMAL,
        ),
        AgentTask(
            name="execute_calculation",
            description="Execute mathematical calculation",
            tool_name="python_code_executor",
            parameters={
                "code": "import math\nresult = math.factorial(10)\nprint(f'10! = {result}')",
                "sandbox_type": "process",
            },
        ),
    ]

    # Create a workflow
    workflow = Workflow(
        name="data_analysis_workflow",
        description="Comprehensive data analysis workflow",
        steps=[
            WorkflowStep(
                name="data_analysis",
                tasks=tasks[:2],  # Run data analysis in parallel
                mode=ExecutionMode.PARALLEL,
            ),
            WorkflowStep(
                name="calculation",
                tasks=[tasks[2]],  # Run calculation after analysis
                depends_on=[],  # No dependencies for demo
                mode=ExecutionMode.SEQUENTIAL,
            ),
        ],
    )

    print(f"Created workflow with {len(workflow.steps)} steps")

    # Execute workflow
    print("\n--- Executing Workflow ---")
    results = await orchestrator.execute_workflow(workflow)

    print(f"Workflow completed with {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  Task {i + 1}: Success={result.success}, Agent={result.agent_id}")
        if result.success and result.result:
            print(f"    Data: {json.dumps(result.result.data, indent=2)}")

    # Get system stats
    stats = await orchestrator.get_system_stats()
    print("\nSystem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def demonstrate_mcp_server():
    """Demonstrate MCP server functionality."""
    print("\n=== MCP Server ===")

    registry = get_global_registry()

    # Create and start MCP server
    server = await create_mcp_server(
        name="demo-mcp-server", tool_registry=registry, host="localhost", port=9000, protocol="both"
    )

    print(f"Created MCP server: {server.config.name}")
    print(f"Available tools: {len(registry.list_tools())}")

    # Start server in background
    server_task = asyncio.create_task(server.start())

    # Give server time to start
    await asyncio.sleep(1)

    try:
        # Test server health
        if server.is_running():
            print("MCP server is running successfully")
            print(f"Connections: {server.get_connection_count()}")

        # The server would normally run indefinitely
        # For demo purposes, we'll stop it quickly

    except Exception as e:
        print(f"MCP server error: {e}")
    finally:
        await server.stop()
        if not server_task.done():
            server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


async def demonstrate_security_features():
    """Demonstrate security features."""
    print("\n=== Security Features ===")

    registry = get_global_registry()

    # Test with different security levels
    security_levels = ["minimal", "standard", "strict", "maximum"]

    for level in security_levels:
        print(f"\n--- Testing Security Level: {level} ---")

        context = ToolExecutionContext(security_level=level, timeout=10, sandbox_mode=True)

        # Try to execute potentially dangerous code
        dangerous_code = """
import os
import sys

# Try to access environment variables
print("Environment variables:")
for key in ['HOME', 'PATH', 'USER']:
    print(f"{key}: {os.environ.get(key, 'Not found')}")

# Try to list directory
try:
    files = os.listdir('.')
    print(f"Current directory has {len(files)} items")
except Exception as e:
    print(f"Cannot list directory: {e}")

# Try to write a file
try:
    with open('test_security.txt', 'w') as f:
        f.write('Security test')
    print("File write successful")
except Exception as e:
    print(f"Cannot write file: {e}")
"""

        result = await registry.execute_tool(
            "python_code_executor", context, code=dangerous_code, sandbox_type="process"
        )

        print(f"Execution success: {result.success}")
        if result.success:
            print("Output:", result.data.get("stdout", "")[:200])
        else:
            print(f"Security blocked: {result.error}")


async def main():
    """Main demonstration function."""
    print("LLM Agent Framework - Comprehensive Demonstration")
    print("=" * 50)

    try:
        # Demonstrate different aspects of the framework
        await demonstrate_basic_tools()
        await demonstrate_agent_orchestration()
        await demonstrate_mcp_server()
        await demonstrate_security_features()

        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
