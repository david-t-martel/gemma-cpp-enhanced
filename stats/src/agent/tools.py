"""Tool definitions and implementations for the LLM agent."""

import ast
import json
import math
import operator
import platform
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from pydantic import Field


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str
    type: str
    description: str
    required: bool = True


class ToolDefinition(BaseModel):
    """Tool definition schema."""

    name: str
    description: str
    parameters: list[ToolParameter]
    function: Callable[..., Any] | None = Field(default=None, exclude=True)


class ToolResult(BaseModel):
    """Tool execution result."""

    success: bool
    output: Any
    error: str | None = None


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all available tools."""
        return list(self.tools.values())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools."""
        schemas = []
        for tool in self.tools.values():
            schema: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

            for param in tool.parameters:
                schema["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    schema["parameters"]["required"].append(param.name)

            schemas.append(schema)

        return schemas

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool with given arguments."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(success=False, output=None, error=f"Tool '{name}' not found")

        if not tool.function:
            return ToolResult(
                success=False, output=None, error=f"Tool '{name}' has no implementation"
            )

        try:
            result = tool.function(**arguments)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _register_default_tools(self) -> None:
        """Register default built-in tools."""

        # Calculator tool
        self.register(
            ToolDefinition(
                name="calculator",
                description="Perform mathematical calculations. Supports basic arithmetic, trigonometry, logarithms, and more.",
                parameters=[
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="Mathematical expression to evaluate (e.g., '2 + 2', 'sin(3.14)', 'log(10)')",
                    ),
                ],
                function=self._calculator_tool,
            )
        )

        # File reader tool
        self.register(
            ToolDefinition(
                name="read_file",
                description="Read the contents of a file from the filesystem.",
                parameters=[
                    ToolParameter(
                        name="path", type="string", description="Path to the file to read"
                    ),
                    ToolParameter(
                        name="encoding",
                        type="string",
                        description="File encoding (default: utf-8)",
                        required=False,
                    ),
                ],
                function=self._read_file_tool,
            )
        )

        # File writer tool
        self.register(
            ToolDefinition(
                name="write_file",
                description="Write content to a file on the filesystem.",
                parameters=[
                    ToolParameter(
                        name="path", type="string", description="Path to the file to write"
                    ),
                    ToolParameter(
                        name="content", type="string", description="Content to write to the file"
                    ),
                    ToolParameter(
                        name="encoding",
                        type="string",
                        description="File encoding (default: utf-8)",
                        required=False,
                    ),
                ],
                function=self._write_file_tool,
            )
        )

        # Web search tool (simulated)
        self.register(
            ToolDefinition(
                name="web_search",
                description="Search the web for information. Returns search results with titles and snippets.",
                parameters=[
                    ToolParameter(name="query", type="string", description="Search query"),
                    ToolParameter(
                        name="max_results",
                        type="integer",
                        description="Maximum number of results (default: 5)",
                        required=False,
                    ),
                ],
                function=self._web_search_tool,
            )
        )

        # Web scraper tool
        self.register(
            ToolDefinition(
                name="fetch_url",
                description="Fetch and extract text content from a URL.",
                parameters=[
                    ToolParameter(name="url", type="string", description="URL to fetch"),
                ],
                function=self._fetch_url_tool,
            )
        )

        # Datetime tool
        self.register(
            ToolDefinition(
                name="get_datetime",
                description="Get current date and time information.",
                parameters=[
                    ToolParameter(
                        name="tz",
                        type="string",
                        description="Timezone (e.g., 'UTC', 'US/Eastern', default: local timezone)",
                        required=False,
                    ),
                    ToolParameter(
                        name="fmt",
                        type="string",
                        description="Format string for the date and time (e.g., '%Y-%m-%d %H:%M:%S')",
                        required=False,
                    ),
                ],
                function=self._datetime_tool,
            )
        )

        # System info tool
        self.register(
            ToolDefinition(
                name="system_info",
                description="Get system and hardware information.",
                parameters=[
                    ToolParameter(
                        name="category",
                        type="string",
                        description="Category of system information to retrieve ('all', 'cpu', 'memory', 'disk', 'network', 'platform')",
                        required=False,
                    ),
                ],
                function=self._system_info_tool,
            )
        )

        # List directory tool
        self.register(
            ToolDefinition(
                name="list_directory",
                description="List files and directories in a given path.",
                parameters=[
                    ToolParameter(name="path", type="string", description="Directory path to list"),
                    ToolParameter(
                        name="recursive",
                        type="boolean",
                        description="Whether to list directory contents recursively (default: False)",
                        required=False,
                    ),
                ],
                function=self._list_directory_tool,
            )
        )

    def _calculator_tool(self, expression: str) -> str:
        """Execute mathematical calculations safely using AST parsing."""
        try:
            # Define allowed operations
            allowed_ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
                ast.Mod: operator.mod,
                ast.FloorDiv: operator.floordiv,
            }

            # Define allowed functions
            allowed_funcs = {
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "sinh": math.sinh,
                "cosh": math.cosh,
                "tanh": math.tanh,
                "exp": math.exp,
                "log": math.log,
                "log10": math.log10,
                "sqrt": math.sqrt,
                "ceil": math.ceil,
                "floor": math.floor,
                "abs": abs,
                "round": round,
            }

            # Define constants
            constants = {
                "pi": math.pi,
                "e": math.e,
            }

            def safe_eval(node: Any) -> Any:
                """Safely evaluate an AST node."""
                if isinstance(node, ast.Constant):  # Python 3.8+
                    return node.value
                elif isinstance(node, ast.Num):  # Backward compatibility
                    return node.n
                elif isinstance(node, ast.Name):
                    if node.id in constants:
                        return constants[node.id]
                    raise ValueError(f"Unknown variable: {node.id}")
                elif isinstance(node, ast.BinOp):
                    op_type = type(node.op)
                    if op_type not in allowed_ops:
                        raise ValueError(f"Unsupported operation: {op_type.__name__}")
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    op_func = allowed_ops[op_type]
                    return op_func(left, right)  # type: ignore[operator]
                elif isinstance(node, ast.UnaryOp):
                    unary_op_type: Any = type(node.op)
                    if unary_op_type not in allowed_ops:
                        raise ValueError(f"Unsupported operation: {unary_op_type.__name__}")
                    operand = safe_eval(node.operand)
                    op_func = allowed_ops[unary_op_type]
                    return op_func(operand)  # type: ignore[operator]
                elif isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name):
                        raise ValueError("Complex function calls not allowed")
                    func_name = node.func.id
                    if func_name not in allowed_funcs:
                        raise ValueError(f"Function not allowed: {func_name}")
                    if func_name == "pow" and len(node.args) == 2:
                        # Special handling for pow(x, y)
                        base = safe_eval(node.args[0])
                        exp = safe_eval(node.args[1])
                        return pow(base, exp)
                    args = [safe_eval(arg) for arg in node.args]
                    func = allowed_funcs[func_name]
                    return func(*args)  # type: ignore[operator]
                else:
                    raise ValueError(f"Unsupported expression type: {type(node).__name__}")

            # Parse and evaluate the expression safely
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)
            return f"Result: {result}"
        except SyntaxError as e:
            return f"Syntax error in expression '{expression}': {e!s}"
        except ZeroDivisionError:
            return f"Error: Division by zero in '{expression}'"
        except ValueError as e:
            return f"Invalid expression '{expression}': {e!s}"
        except Exception as e:
            return f"Error calculating '{expression}': {e!s}"

    def _read_file_tool(self, path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                return f"Error: File '{path}' does not exist"

            if not file_path.is_file():
                return f"Error: '{path}' is not a file"

            with open(file_path, encoding=encoding) as f:
                content = f.read()

            # Limit output size
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated)"

            return f"File contents of {path}:\n{content}"
        except Exception as e:
            return f"Error reading file '{path}': {e!s}"

    def _write_file_tool(self, path: str, content: str, encoding: str = "utf-8") -> str:
        """Write content to file."""
        try:
            file_path = Path(path).resolve()

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)

            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file '{path}': {e!s}"

    def _web_search_tool(self, query: str, max_results: int = 5) -> str:
        """Simulated web search (returns mock results for demo)."""
        # In a real implementation, this would use a search API
        # For demo purposes, we'll return mock results
        mock_results = [
            {
                "title": f"Result for: {query}",
                "snippet": f"This is a simulated search result for '{query}'. In a real implementation, this would fetch actual web results.",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            }
        ]

        # Try to fetch real results from DuckDuckGo HTML (basic scraping)
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(
                f"https://html.duckduckgo.com/html/?q={query}", headers=headers, timeout=5
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                for result in soup.find_all("div", class_="result", limit=max_results):
                    if not hasattr(result, "find"):
                        continue
                    title_elem = result.find("a", class_="result__a")
                    snippet_elem = result.find("a", class_="result__snippet")
                    if title_elem and hasattr(title_elem, "get_text"):
                        results.append(
                            {
                                "title": title_elem.get_text(strip=True),
                                "snippet": (
                                    snippet_elem.get_text(strip=True)
                                    if snippet_elem and hasattr(snippet_elem, "get_text")
                                    else "No snippet available"
                                ),
                                "url": title_elem.get("href", "#")
                                if hasattr(title_elem, "get")
                                else "#",
                            }
                        )
                if results:
                    mock_results = results
        except (requests.RequestException, ValueError, KeyError):
            # Fall back to mock results if web search fails
            return f"Search fallback: mock results for '{query}'"

        output = f"Search results for '{query}':\n\n"
        for i, search_result in enumerate(mock_results[:max_results], 1):
            output += f"{i}. {search_result['title']}\n"
            output += f"   {search_result['snippet']}\n"
            output += f"   URL: {search_result['url']}\n\n"

        return output

    def _fetch_url_tool(self, url: str) -> str:
        """Fetch and extract text from URL."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Limit output size
            if len(text) > 5000:
                text = text[:5000] + "\n... (truncated)"

            return f"Content from {url}:\n\n{text}"
        except Exception as e:
            return f"Error fetching URL '{url}': {e!s}"

    def _datetime_tool(self, tz: str = "UTC", fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Get current datetime."""
        try:
            now = datetime.now(UTC) if tz == "UTC" else datetime.now()
            formatted = now.strftime(fmt)

            output = f"Current datetime: {formatted}\n"
            output += f"Timezone: {tz}\n"
            output += f"Unix timestamp: {int(now.timestamp())}\n"
            output += f"ISO format: {now.isoformat()}"

            return output
        except Exception as e:
            return f"Error getting datetime: {e!s}"

    def _system_info_tool(self, category: str = "all") -> str:
        """Get system information."""
        try:
            info = {}

            if category in ["all", "platform"]:
                info["Platform"] = {
                    "System": platform.system(),
                    "Node": platform.node(),
                    "Release": platform.release(),
                    "Version": platform.version(),
                    "Machine": platform.machine(),
                    "Processor": platform.processor(),
                    "Python": platform.python_version(),
                }

            if category in ["all", "cpu"]:
                info["CPU"] = {
                    "Physical cores": psutil.cpu_count(logical=False),
                    "Total cores": psutil.cpu_count(logical=True),
                    "Max frequency": (
                        f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "N/A"
                    ),
                    "Current frequency": (
                        f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "N/A"
                    ),
                    "CPU usage": f"{psutil.cpu_percent(interval=1)}%",
                }

            if category in ["all", "memory"]:
                vm = psutil.virtual_memory()
                info["Memory"] = {
                    "Total": f"{vm.total / (1024**3):.2f} GB",
                    "Available": f"{vm.available / (1024**3):.2f} GB",
                    "Used": f"{vm.used / (1024**3):.2f} GB",
                    "Percentage": f"{vm.percent}%",
                }

            if category in ["all", "disk"]:
                disk = psutil.disk_usage("/")
                info["Disk"] = {
                    "Total": f"{disk.total / (1024**3):.2f} GB",
                    "Used": f"{disk.used / (1024**3):.2f} GB",
                    "Free": f"{disk.free / (1024**3):.2f} GB",
                    "Percentage": f"{disk.percent}%",
                }

            if category in ["all", "network"]:
                net = psutil.net_io_counters()
                info["Network"] = {
                    "Bytes sent": f"{net.bytes_sent / (1024**2):.2f} MB",
                    "Bytes received": f"{net.bytes_recv / (1024**2):.2f} MB",
                    "Packets sent": net.packets_sent,
                    "Packets received": net.packets_recv,
                }

            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error getting system info: {e!s}"

    def _list_directory_tool(self, path: str, recursive: bool = False) -> str:
        """List directory contents."""
        try:
            dir_path = Path(path).resolve()
            if not dir_path.exists():
                return f"Error: Path '{path}' does not exist"

            if not dir_path.is_dir():
                return f"Error: '{path}' is not a directory"

            output = f"Contents of {path}:\n\n"

            if recursive:
                for item in dir_path.rglob("*"):
                    rel_path = item.relative_to(dir_path)
                    if item.is_dir():
                        output += f"[DIR]  {rel_path}/\n"
                    else:
                        size = item.stat().st_size
                        output += f"[FILE] {rel_path} ({size} bytes)\n"
            else:
                for item in dir_path.iterdir():
                    if item.is_dir():
                        output += f"[DIR]  {item.name}/\n"
                    else:
                        size = item.stat().st_size
                        output += f"[FILE] {item.name} ({size} bytes)\n"

            return output
        except Exception as e:
            return f"Error listing directory '{path}': {e!s}"


# Create a global tool registry instance
tool_registry = ToolRegistry()
