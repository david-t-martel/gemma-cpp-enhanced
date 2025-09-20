"""Base tool interface and registry for the LLM agent framework."""

import asyncio
import inspect
import json
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

T = TypeVar("T", bound="BaseTool")


class ToolType(str, Enum):
    """Tool type enumeration."""

    BUILTIN = "builtin"
    CUSTOM = "custom"
    MCP = "mcp"
    COMPOSITE = "composite"


class ToolCategory(str, Enum):
    """Tool category enumeration."""

    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class ToolExecutionContext(BaseModel):
    """Context for tool execution."""

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timeout: int | None = None
    retry_count: int = 0
    max_retries: int = 3
    cache_enabled: bool = True
    sandbox_mode: bool = True
    security_level: str = "standard"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    execution_time: float
    context: ToolExecutionContext
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Check if the result represents an error."""
        return not self.success or self.error is not None


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    pattern: str | None = None
    format: str | None = None


class ToolSchema(BaseModel):
    """Tool schema definition."""

    name: str
    description: str
    parameters: list[ToolParameter]
    returns: str | None = None
    category: ToolCategory = ToolCategory.CUSTOM
    type: ToolType = ToolType.CUSTOM
    version: str = "1.0.0"
    author: str | None = None
    tags: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)
    security_requirements: dict[str, Any] = Field(default_factory=dict)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for function calling."""
        properties = {}
        required = []

        for param in self.parameters:
            prop_def = {"type": param.type, "description": param.description}

            if param.enum:
                prop_def["enum"] = param.enum
            if param.minimum is not None:
                prop_def["minimum"] = param.minimum
            if param.maximum is not None:
                prop_def["maximum"] = param.maximum
            if param.pattern:
                prop_def["pattern"] = param.pattern
            if param.format:
                prop_def["format"] = param.format
            if param.default is not None:
                prop_def["default"] = param.default

            properties[param.name] = prop_def

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self):
        self._schema: ToolSchema | None = None
        self._is_initialized = False

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Tool schema definition."""

    @abstractmethod
    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""

    async def initialize(self) -> None:
        """Initialize the tool (override if needed)."""
        self._is_initialized = True

    async def cleanup(self) -> None:
        """Cleanup resources (override if needed)."""

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        """Validate parameters against schema."""
        schema = self.schema
        provided_params = set(parameters.keys())
        required_params = {p.name for p in schema.parameters if p.required}

        # Check required parameters
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Check unknown parameters
        valid_params = {p.name for p in schema.parameters}
        unknown_params = provided_params - valid_params
        if unknown_params:
            raise ValueError(f"Unknown parameters: {unknown_params}")

        # Type and value validation
        for param in schema.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                self._validate_parameter_value(param, value)

    def _validate_parameter_value(self, param: ToolParameter, value: Any) -> None:
        """Validate individual parameter value."""
        # Type validation (simplified)
        if param.type == "string" and not isinstance(value, str):
            raise ValueError(f"Parameter {param.name} must be a string")
        elif param.type == "integer" and not isinstance(value, int):
            raise ValueError(f"Parameter {param.name} must be an integer")
        elif param.type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {param.name} must be a number")
        elif param.type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Parameter {param.name} must be a boolean")
        elif param.type == "array" and not isinstance(value, list):
            raise ValueError(f"Parameter {param.name} must be an array")
        elif param.type == "object" and not isinstance(value, dict):
            raise ValueError(f"Parameter {param.name} must be an object")

        # Enum validation
        if param.enum and value not in param.enum:
            raise ValueError(f"Parameter {param.name} must be one of {param.enum}")

        # Range validation
        if param.minimum is not None and isinstance(value, (int, float)) and value < param.minimum:
            raise ValueError(f"Parameter {param.name} must be >= {param.minimum}")
        if param.maximum is not None and isinstance(value, (int, float)) and value > param.maximum:
            raise ValueError(f"Parameter {param.name} must be <= {param.maximum}")

    @property
    def is_initialized(self) -> bool:
        """Check if tool is initialized."""
        return self._is_initialized


class ToolExecutionError(Exception):
    """Tool execution error."""

    def __init__(self, message: str, tool_name: str, context: ToolExecutionContext | None = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.context = context


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        async with self._lock:
            if not isinstance(tool, BaseTool):
                raise TypeError("Tool must inherit from BaseTool")

            schema = tool.schema
            if schema.name in self._tools:
                logger.warning(f"Tool {schema.name} is already registered, overwriting")

            await tool.initialize()
            self._tools[schema.name] = tool
            logger.info(f"Registered tool: {schema.name}")

    async def unregister(self, name: str) -> None:
        """Unregister a tool."""
        async with self._lock:
            if name in self._tools:
                tool = self._tools[name]
                await tool.cleanup()
                del self._tools[name]
                logger.info(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[ToolSchema]:
        """Get all tool schemas."""
        return [tool.schema for tool in self._tools.values()]

    def get_json_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas in JSON Schema format."""
        return [tool.schema.to_json_schema() for tool in self._tools.values()]

    async def execute_tool(self, name: str, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ToolExecutionError(f"Tool '{name}' not found", name, context)

        if not tool.is_initialized:
            await tool.initialize()

        try:
            # Validate parameters
            tool.validate_parameters(kwargs)

            # Execute tool
            start_time = asyncio.get_event_loop().time()
            result = await tool.execute(context, **kwargs)
            execution_time = asyncio.get_event_loop().time() - start_time

            # Update result with execution time
            result.execution_time = execution_time

            return result

        except Exception as e:
            execution_time = (
                asyncio.get_event_loop().time() - start_time if "start_time" in locals() else 0
            )
            logger.error(f"Tool execution failed: {name}: {e!s}")
            return ToolResult(
                success=False, error=str(e), execution_time=execution_time, context=context
            )

    async def cleanup_all(self) -> None:
        """Cleanup all registered tools."""
        async with self._lock:
            for tool in self._tools.values():
                try:
                    await tool.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up tool: {e}")
            self._tools.clear()


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def tool(
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    tool_type: ToolType = ToolType.CUSTOM,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a tool class."""

    def decorator(cls: type[T]) -> type[T]:
        # Auto-register the tool when imported
        async def auto_register():
            instance = cls()
            await _global_registry.register(instance)

        # Schedule auto-registration
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(auto_register())
        except RuntimeError:
            # No event loop running, registration will happen later
            pass

        return cls

    return decorator
