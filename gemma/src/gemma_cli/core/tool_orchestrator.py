"""Tool orchestration module for autonomous MCP tool calling in the chat loop.

This module enables the LLM to autonomously decide when to use MCP tools to answer
user questions by:
1. Converting MCP tool schemas to LLM-compatible format
2. Injecting tool instructions into prompts
3. Detecting and parsing tool call requests from LLM responses
4. Executing tools and feeding results back to the LLM
5. Supporting multi-turn tool chains with depth limits
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Use absolute import to avoid conflict with local mcp module
try:
    from mcp.types import Tool
except ImportError:
    # Fallback: define a simple Tool type for compatibility
    from dataclasses import dataclass
    from typing import Any as _Any, Optional as _Optional

    @dataclass
    class Tool:
        """Fallback Tool class if MCP package not available."""
        name: str
        description: _Optional[str] = None
        inputSchema: _Optional[dict[str, _Any]] = None

logger = logging.getLogger(__name__)


class ToolCallFormat(str, Enum):
    """Supported tool call formats for different LLMs."""
    JSON_BLOCK = "json_block"  # JSON code blocks: ```json{"tool": "name", "args": {}}```
    XML_TAGS = "xml_tags"  # XML-style tags: <tool>name</tool><args>{}</args>
    FUNCTION_CALL = "function_call"  # OpenAI-style function calling


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output."""
    server: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    format_type: ToolCallFormat = ToolCallFormat.JSON_BLOCK
    raw_match: str = ""


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ToolSchemaFormatter:
    """Formats MCP tool schemas for LLM consumption."""

    def __init__(self, format_type: ToolCallFormat = ToolCallFormat.JSON_BLOCK):
        """
        Initialize the formatter.

        Args:
            format_type: The tool call format to use
        """
        self.format_type = format_type

    def format_tools_for_prompt(
        self,
        tools_by_server: dict[str, list[Tool]]
    ) -> str:
        """
        Format MCP tools into a prompt-friendly description.

        Args:
            tools_by_server: Dictionary mapping server names to tool lists

        Returns:
            Formatted tool description for injection into prompts
        """
        if not tools_by_server:
            return ""

        lines = ["You have access to the following tools:"]
        lines.append("")

        for server_name, tools in tools_by_server.items():
            if not tools:
                continue

            lines.append(f"## {server_name} server:")
            for tool in tools:
                # Extract tool info
                tool_name = f"{server_name}.{tool.name}"
                description = tool.description or "No description available"

                # Format parameters if present
                params_info = ""
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    schema = tool.inputSchema
                    if isinstance(schema, dict):
                        properties = schema.get('properties', {})
                        required = schema.get('required', [])

                        if properties:
                            param_parts = []
                            for param_name, param_schema in properties.items():
                                param_type = param_schema.get('type', 'any')
                                param_desc = param_schema.get('description', '')
                                is_required = param_name in required

                                param_str = f"  - {param_name} ({param_type})"
                                if is_required:
                                    param_str += " [required]"
                                if param_desc:
                                    param_str += f": {param_desc}"
                                param_parts.append(param_str)

                            if param_parts:
                                params_info = "\n" + "\n".join(param_parts)

                lines.append(f"- **{tool_name}**: {description}{params_info}")
            lines.append("")

        # Add format instructions based on format type
        lines.extend(self._get_format_instructions())

        return "\n".join(lines)

    def _get_format_instructions(self) -> list[str]:
        """Get format-specific instructions for tool calling."""
        instructions = [
            "",
            "When you need to use a tool to answer a question or perform an action:",
        ]

        if self.format_type == ToolCallFormat.JSON_BLOCK:
            instructions.extend([
                "1. Think about which tool would be most appropriate",
                "2. Format your tool call as a JSON code block:",
                "```json",
                "{",
                '  "tool": "server.tool_name",',
                '  "args": {',
                '    "param1": "value1",',
                '    "param2": "value2"',
                "  }",
                "}",
                "```",
                "3. Wait for the tool result before continuing your response",
                "",
                "You can chain multiple tool calls if needed, but use them judiciously.",
                "Always explain what you're doing when calling tools."
            ])
        elif self.format_type == ToolCallFormat.XML_TAGS:
            instructions.extend([
                "Format tool calls using XML tags:",
                "<tool_call>",
                "  <server>server_name</server>",
                "  <tool>tool_name</tool>",
                "  <args>",
                '    {"param1": "value1", "param2": "value2"}',
                "  </args>",
                "</tool_call>"
            ])

        return instructions


class ToolCallParser:
    """Parses tool calls from LLM responses."""

    # Regex patterns for different formats
    JSON_BLOCK_PATTERN = re.compile(
        r'```json\s*\n?({[^`]*?"tool"[^`]*?})\s*\n?```',
        re.DOTALL | re.IGNORECASE
    )

    XML_PATTERN = re.compile(
        r'<tool_call>\s*<server>([^<]+)</server>\s*<tool>([^<]+)</tool>\s*<args>([^<]+)</args>\s*</tool_call>',
        re.DOTALL | re.IGNORECASE
    )

    def __init__(self, format_type: ToolCallFormat = ToolCallFormat.JSON_BLOCK):
        """
        Initialize the parser.

        Args:
            format_type: The tool call format to parse
        """
        self.format_type = format_type

    def parse_response(self, response: str) -> list[ToolCall]:
        """
        Parse tool calls from an LLM response.

        Args:
            response: The LLM response text

        Returns:
            List of parsed tool calls
        """
        tool_calls = []

        if self.format_type == ToolCallFormat.JSON_BLOCK:
            tool_calls = self._parse_json_blocks(response)
        elif self.format_type == ToolCallFormat.XML_TAGS:
            tool_calls = self._parse_xml_tags(response)

        return tool_calls

    def _parse_json_blocks(self, response: str) -> list[ToolCall]:
        """Parse JSON code block format tool calls."""
        tool_calls = []

        for match in self.JSON_BLOCK_PATTERN.finditer(response):
            try:
                json_str = match.group(1)
                data = json.loads(json_str)

                # Extract tool info
                tool_full = data.get("tool", "")
                args = data.get("args", {})

                # Split server.tool_name
                if "." in tool_full:
                    server, tool_name = tool_full.split(".", 1)
                else:
                    # Default to first available server if not specified
                    server = "default"
                    tool_name = tool_full

                tool_call = ToolCall(
                    server=server,
                    tool_name=tool_name,
                    arguments=args if isinstance(args, dict) else {},
                    format_type=ToolCallFormat.JSON_BLOCK,
                    raw_match=match.group(0)
                )
                tool_calls.append(tool_call)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

        return tool_calls

    def _parse_xml_tags(self, response: str) -> list[ToolCall]:
        """Parse XML tag format tool calls."""
        tool_calls = []

        for match in self.XML_PATTERN.finditer(response):
            try:
                server = match.group(1).strip()
                tool_name = match.group(2).strip()
                args_str = match.group(3).strip()

                # Parse arguments as JSON
                args = {}
                if args_str:
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse XML args as JSON: {args_str}")

                tool_call = ToolCall(
                    server=server,
                    tool_name=tool_name,
                    arguments=args if isinstance(args, dict) else {},
                    format_type=ToolCallFormat.XML_TAGS,
                    raw_match=match.group(0)
                )
                tool_calls.append(tool_call)

            except Exception as e:
                logger.warning(f"Failed to parse XML tool call: {e}")
                continue

        return tool_calls


class ToolOrchestrator:
    """Orchestrates autonomous tool calling in the chat loop."""

    def __init__(
        self,
        mcp_manager,
        available_tools: dict[str, list[Tool]],
        format_type: ToolCallFormat = ToolCallFormat.JSON_BLOCK,
        max_tool_depth: int = 5,
        require_confirmation: bool = False
    ):
        """
        Initialize the tool orchestrator.

        Args:
            mcp_manager: The MCP client manager instance
            available_tools: Dictionary of available tools by server
            format_type: The tool call format to use
            max_tool_depth: Maximum depth for tool call chains
            require_confirmation: Whether to require user confirmation for tool calls
        """
        self.mcp_manager = mcp_manager
        self.available_tools = available_tools
        self.format_type = format_type
        self.max_tool_depth = max_tool_depth
        self.require_confirmation = require_confirmation

        self.schema_formatter = ToolSchemaFormatter(format_type)
        self.parser = ToolCallParser(format_type)

        # Track tool call depth to prevent infinite loops
        self.current_depth = 0

    def get_system_prompt(self) -> str:
        """
        Generate the system prompt with tool instructions.

        Returns:
            System prompt including tool descriptions and usage instructions
        """
        base_prompt = (
            "You are a helpful AI assistant with access to external tools. "
            "Use these tools when needed to provide accurate, up-to-date information "
            "or to perform actions requested by the user.\n\n"
        )

        tools_prompt = self.schema_formatter.format_tools_for_prompt(self.available_tools)

        guidelines = (
            "\n\n## Tool Usage Guidelines:\n"
            "- Only use tools when necessary to answer the user's question\n"
            "- Explain what you're doing when calling tools\n"
            "- If a tool fails, explain the error and try an alternative approach\n"
            "- Don't call tools repeatedly if they keep failing\n"
            "- Respect the user's privacy - don't access files without permission\n"
        )

        return base_prompt + tools_prompt + guidelines

    async def process_response_with_tools(
        self,
        response: str,
        console=None,
        confirm_callback=None
    ) -> tuple[str, list[ToolResult]]:
        """
        Process an LLM response, execute any tool calls, and return enriched response.

        Args:
            response: The LLM response to process
            console: Optional console for displaying tool execution status
            confirm_callback: Optional callback for confirming tool execution

        Returns:
            Tuple of (final response with tool results integrated, list of tool results)
        """
        # Reset depth counter for new response
        self.current_depth = 0

        # Parse tool calls from response
        tool_calls = self.parser.parse_response(response)

        if not tool_calls:
            return response, []

        # Execute tool calls and collect results
        all_results = []
        response_with_results = response

        for tool_call in tool_calls:
            # Check depth limit
            if self.current_depth >= self.max_tool_depth:
                logger.warning(f"Tool depth limit ({self.max_tool_depth}) reached")
                break

            # Confirm if required
            if self.require_confirmation and confirm_callback:
                confirmed = await confirm_callback(tool_call)
                if not confirmed:
                    logger.info(f"Tool call cancelled by user: {tool_call.tool_name}")
                    continue

            # Execute tool
            result = await self.execute_tool(tool_call, console)
            all_results.append(result)
            self.current_depth += 1

            # Replace tool call in response with result
            if result.success:
                result_text = f"\n[Tool Result: {json.dumps(result.output, indent=2)}]\n"
            else:
                result_text = f"\n[Tool Error: {result.error}]\n"

            response_with_results = response_with_results.replace(
                tool_call.raw_match,
                result_text
            )

        return response_with_results, all_results

    async def execute_tool(
        self,
        tool_call: ToolCall,
        console=None
    ) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: The tool call to execute
            console: Optional console for status display

        Returns:
            Tool execution result
        """
        import time

        start_time = time.time()

        try:
            if console:
                console.print(
                    f"[dim]Calling tool: {tool_call.server}.{tool_call.tool_name}...[/dim]"
                )

            # Execute via MCP manager
            result = await self.mcp_manager.call_tool(
                server=tool_call.server,
                tool=tool_call.tool_name,
                args=tool_call.arguments
            )

            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                output=result,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=execution_time
            )

    def check_needs_tools(self, query: str) -> bool:
        """
        Heuristic to check if a query might need tool usage.

        Args:
            query: The user query

        Returns:
            True if tools might be helpful
        """
        # Keywords that suggest tool usage
        tool_keywords = [
            "file", "read", "write", "save", "load",
            "search", "web", "internet", "online",
            "fetch", "download", "url", "website",
            "memory", "remember", "recall", "store",
            "current", "latest", "today", "now",
            "check", "verify", "look up", "find"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in tool_keywords)