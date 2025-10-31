"""End-to-end integration tests for autonomous tool calling system.

Tests the complete chain:
1. User query → LLM with tool instructions
2. LLM returns tool call (JSON block format)
3. ToolOrchestrator parses and executes via MCPClientManager
4. Tool results returned to LLM
5. Final response generated with context

Covers:
- Single tool calling
- Multi-turn tool chains
- Depth limit enforcement
- Error handling
- Tool result integration
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemma_cli.core.tool_orchestrator import (
    ToolCall,
    ToolCallFormat,
    ToolCallParser,
    ToolOrchestrator,
    ToolResult,
    ToolSchemaFormatter,
)


@pytest.fixture
def mock_console():
    """Create a mock console for tool output."""
    console = MagicMock()
    console.print = MagicMock()
    return console


@pytest.fixture
def mock_mcp_manager():
    """Create a mock MCP manager with basic tool calling."""
    manager = AsyncMock()
    manager.call_tool = AsyncMock()
    return manager


@pytest.fixture
def sample_tools():
    """Create sample tool definitions."""
    from dataclasses import dataclass

    @dataclass
    class Tool:
        name: str
        description: str
        inputSchema: dict

    return {
        "filesystem": [
            Tool(
                name="read_file",
                description="Read a file from the filesystem",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="write_file",
                description="Write content to a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            ),
        ],
        "memory": [
            Tool(
                name="recall",
                description="Recall information from memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
        ],
        "web": [
            Tool(
                name="search",
                description="Search the web",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
        ],
    }


@pytest.fixture
def tool_orchestrator(mock_mcp_manager, sample_tools):
    """Create a tool orchestrator with sample tools."""
    return ToolOrchestrator(
        mcp_manager=mock_mcp_manager,
        available_tools=sample_tools,
        format_type=ToolCallFormat.JSON_BLOCK,
        max_tool_depth=5,
        require_confirmation=False,
    )


class TestToolSchemaFormatter:
    """Test tool schema formatting for LLM prompts."""

    def test_format_tools_for_prompt(self, sample_tools):
        """Test that tools are formatted correctly for LLM consumption."""
        formatter = ToolSchemaFormatter(format_type=ToolCallFormat.JSON_BLOCK)
        prompt = formatter.format_tools_for_prompt(sample_tools)

        # Should contain server names
        assert "filesystem server:" in prompt
        assert "memory server:" in prompt
        assert "web server:" in prompt

        # Should contain tool names
        assert "filesystem.read_file" in prompt
        assert "memory.recall" in prompt
        assert "web.search" in prompt

        # Should contain descriptions
        assert "Read a file from the filesystem" in prompt
        assert "Recall information from memory" in prompt

        # Should contain format instructions
        assert "```json" in prompt
        assert '"tool":' in prompt

    def test_empty_tools(self):
        """Test formatting with no tools."""
        formatter = ToolSchemaFormatter()
        prompt = formatter.format_tools_for_prompt({})
        assert prompt == ""

    def test_xml_format_instructions(self, sample_tools):
        """Test XML format instructions."""
        formatter = ToolSchemaFormatter(format_type=ToolCallFormat.XML_TAGS)
        prompt = formatter.format_tools_for_prompt(sample_tools)

        assert "<tool_call>" in prompt
        assert "<server>" in prompt
        assert "<tool>" in prompt


class TestToolCallParser:
    """Test parsing tool calls from LLM responses."""

    def test_parse_single_json_tool_call(self):
        """Test parsing a single JSON block tool call."""
        parser = ToolCallParser(format_type=ToolCallFormat.JSON_BLOCK)
        response = """
I'll read the file for you.

```json
{
  "tool": "filesystem.read_file",
  "args": {
    "path": "/home/user/document.txt"
  }
}
```

Let me check that file.
"""
        tool_calls = parser.parse_response(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].server == "filesystem"
        assert tool_calls[0].tool_name == "read_file"
        assert tool_calls[0].arguments == {"path": "/home/user/document.txt"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one response."""
        parser = ToolCallParser(format_type=ToolCallFormat.JSON_BLOCK)
        response = """
First, let me search the web:

```json
{
  "tool": "web.search",
  "args": {"query": "Python asyncio"}
}
```

Then I'll recall what I know:

```json
{
  "tool": "memory.recall",
  "args": {"query": "asyncio patterns"}
}
```
"""
        tool_calls = parser.parse_response(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].server == "web"
        assert tool_calls[0].tool_name == "search"
        assert tool_calls[1].server == "memory"
        assert tool_calls[1].tool_name == "recall"

    def test_parse_malformed_json(self):
        """Test handling of malformed JSON."""
        parser = ToolCallParser(format_type=ToolCallFormat.JSON_BLOCK)
        response = """
```json
{
  "tool": "filesystem.read_file",
  "args": {
    "path": "/invalid
  }
}
```
"""
        tool_calls = parser.parse_response(response)
        assert len(tool_calls) == 0  # Should skip malformed JSON

    def test_parse_no_tool_calls(self):
        """Test response with no tool calls."""
        parser = ToolCallParser(format_type=ToolCallFormat.JSON_BLOCK)
        response = "This is a normal response without any tool calls."

        tool_calls = parser.parse_response(response)
        assert len(tool_calls) == 0


class TestToolOrchestrator:
    """Test tool orchestration and execution."""

    @pytest.mark.asyncio
    async def test_get_system_prompt(self, tool_orchestrator):
        """Test system prompt generation with tool instructions."""
        prompt = tool_orchestrator.get_system_prompt()

        assert "helpful AI assistant" in prompt
        assert "filesystem server:" in prompt
        assert "Tool Usage Guidelines" in prompt

    @pytest.mark.asyncio
    async def test_execute_single_tool(self, tool_orchestrator, mock_mcp_manager, mock_console):
        """Test executing a single tool call."""
        # Setup mock response
        mock_mcp_manager.call_tool.return_value = {"content": "File contents here"}

        tool_call = ToolCall(
            server="filesystem",
            tool_name="read_file",
            arguments={"path": "/test.txt"},
            format_type=ToolCallFormat.JSON_BLOCK,
            raw_match='```json\n{"tool": "filesystem.read_file"}\n```',
        )

        result = await tool_orchestrator.execute_tool(tool_call, console=mock_console)

        assert result.success is True
        assert result.output == {"content": "File contents here"}
        assert result.execution_time_ms > 0

        # Verify MCP manager was called correctly
        mock_mcp_manager.call_tool.assert_called_once_with(
            server="filesystem",
            tool="read_file",
            args={"path": "/test.txt"},
        )

    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self, tool_orchestrator, mock_mcp_manager, mock_console):
        """Test handling tool execution errors."""
        # Setup mock to raise error
        mock_mcp_manager.call_tool.side_effect = Exception("Tool execution failed")

        tool_call = ToolCall(
            server="filesystem",
            tool_name="read_file",
            arguments={"path": "/nonexistent.txt"},
        )

        result = await tool_orchestrator.execute_tool(tool_call, console=mock_console)

        assert result.success is False
        assert result.error == "Tool execution failed"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_process_response_with_tool_call(
        self, tool_orchestrator, mock_mcp_manager, mock_console
    ):
        """Test processing a response that contains a tool call."""
        # Setup mock response
        mock_mcp_manager.call_tool.return_value = "File content: Hello World"

        response = """
I'll read that file for you.

```json
{
  "tool": "filesystem.read_file",
  "args": {"path": "/test.txt"}
}
```
"""

        processed_response, tool_results = await tool_orchestrator.process_response_with_tools(
            response=response,
            console=mock_console,
        )

        assert len(tool_results) == 1
        assert tool_results[0].success is True
        assert "[Tool Result:" in processed_response
        assert "File content: Hello World" in processed_response

    @pytest.mark.asyncio
    async def test_depth_limit_enforcement(self, tool_orchestrator, mock_mcp_manager, mock_console):
        """Test that tool depth limit is enforced."""
        # Create response with many tool calls
        tool_calls = []
        for i in range(10):  # More than max_tool_depth (5)
            tool_calls.append(
                ToolCall(
                    server="memory",
                    tool_name="recall",
                    arguments={"query": f"query{i}"},
                    raw_match=f"```json\n{{\"tool\": \"memory.recall\"}}\n```{i}",
                )
            )

        # Mock parser to return many tool calls
        with patch.object(tool_orchestrator.parser, "parse_response", return_value=tool_calls):
            mock_mcp_manager.call_tool.return_value = {"result": "data"}

            response = "Multiple tool calls here"
            processed_response, tool_results = await tool_orchestrator.process_response_with_tools(
                response=response,
                console=mock_console,
            )

            # Should only execute up to max_tool_depth
            assert len(tool_results) <= tool_orchestrator.max_tool_depth


class TestMultiTurnToolCalling:
    """Test multi-turn tool calling scenarios."""

    @pytest.mark.asyncio
    async def test_tool_calls_another_tool(self, mock_mcp_manager, sample_tools, mock_console):
        """Test scenario where one tool's result triggers another tool call."""
        orchestrator = ToolOrchestrator(
            mcp_manager=mock_mcp_manager,
            available_tools=sample_tools,
            format_type=ToolCallFormat.JSON_BLOCK,
            max_tool_depth=5,
        )

        # First tool call returns a path that needs to be read
        mock_mcp_manager.call_tool.side_effect = [
            {"file_path": "/discovered/file.txt"},  # First call result
            "File contents",  # Second call result
        ]

        # Response with first tool call
        response1 = """
Let me search for that file:

```json
{
  "tool": "web.search",
  "args": {"query": "config file location"}
}
```
"""

        processed_response1, results1 = await orchestrator.process_response_with_tools(
            response=response1,
            console=mock_console,
        )

        assert len(results1) == 1
        assert results1[0].output == {"file_path": "/discovered/file.txt"}

        # Second response using result from first tool
        response2 = """
Now let me read that file:

```json
{
  "tool": "filesystem.read_file",
  "args": {"path": "/discovered/file.txt"}
}
```
"""

        processed_response2, results2 = await orchestrator.process_response_with_tools(
            response=response2,
            console=mock_console,
        )

        assert len(results2) == 1
        assert results2[0].output == "File contents"

        # Verify both tools were called
        assert mock_mcp_manager.call_tool.call_count == 2


class TestToolCallingIntegration:
    """Full integration tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_read_file_modify_save(self, mock_mcp_manager, sample_tools, mock_console):
        """Test realistic scenario: read → modify → save."""
        orchestrator = ToolOrchestrator(
            mcp_manager=mock_mcp_manager,
            available_tools=sample_tools,
            max_tool_depth=3,
        )

        # Mock responses for read, then write
        mock_mcp_manager.call_tool.side_effect = [
            "Original file content",  # read
            {"success": True},  # write
        ]

        # Step 1: Read file
        response1 = """
```json
{
  "tool": "filesystem.read_file",
  "args": {"path": "/config.txt"}
}
```
"""
        _, results1 = await orchestrator.process_response_with_tools(response1, mock_console)
        assert results1[0].output == "Original file content"

        # Step 2: Write modified content
        response2 = """
```json
{
  "tool": "filesystem.write_file",
  "args": {
    "path": "/config.txt",
    "content": "Modified content"
  }
}
```
"""
        _, results2 = await orchestrator.process_response_with_tools(response2, mock_console)
        assert results2[0].output == {"success": True}

    @pytest.mark.asyncio
    async def test_web_search_then_memory_store(self, mock_mcp_manager, sample_tools, mock_console):
        """Test web search followed by memory storage."""
        orchestrator = ToolOrchestrator(
            mcp_manager=mock_mcp_manager,
            available_tools=sample_tools,
            max_tool_depth=3,
        )

        mock_mcp_manager.call_tool.side_effect = [
            {"results": ["Result 1", "Result 2"]},  # web search
            {"memory_id": "mem_123"},  # memory recall
        ]

        # Search web
        response1 = """
```json
{
  "tool": "web.search",
  "args": {"query": "Python best practices"}
}
```
"""
        _, results1 = await orchestrator.process_response_with_tools(response1, mock_console)
        assert "results" in results1[0].output

        # Recall from memory
        response2 = """
```json
{
  "tool": "memory.recall",
  "args": {"query": "Python best practices"}
}
```
"""
        _, results2 = await orchestrator.process_response_with_tools(response2, mock_console)
        assert "memory_id" in results2[0].output


class TestErrorHandling:
    """Test error handling in tool calling."""

    @pytest.mark.asyncio
    async def test_tool_not_found(self, tool_orchestrator, mock_mcp_manager, mock_console):
        """Test handling of non-existent tool."""
        mock_mcp_manager.call_tool.side_effect = Exception("Tool not found")

        response = """
```json
{
  "tool": "filesystem.nonexistent_tool",
  "args": {}
}
```
"""
        processed, results = await tool_orchestrator.process_response_with_tools(
            response, mock_console
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "Tool not found" in results[0].error

    @pytest.mark.asyncio
    async def test_invalid_arguments(self, tool_orchestrator, mock_mcp_manager, mock_console):
        """Test handling of invalid tool arguments."""
        mock_mcp_manager.call_tool.side_effect = Exception("Invalid argument: path required")

        response = """
```json
{
  "tool": "filesystem.read_file",
  "args": {}
}
```
"""
        processed, results = await tool_orchestrator.process_response_with_tools(
            response, mock_console
        )

        assert len(results) == 1
        assert results[0].success is False
        assert "Invalid argument" in results[0].error


@pytest.mark.asyncio
async def test_check_needs_tools_heuristic(tool_orchestrator):
    """Test heuristic for detecting if query needs tools."""
    # Should detect tool-needing queries
    assert tool_orchestrator.check_needs_tools("Can you read the file at /path/to/file?")
    assert tool_orchestrator.check_needs_tools("Search the web for Python tutorials")
    assert tool_orchestrator.check_needs_tools("What's the latest news today?")
    assert tool_orchestrator.check_needs_tools("Remember this information")

    # Should not trigger for simple queries
    assert not tool_orchestrator.check_needs_tools("Hello, how are you?")
    assert not tool_orchestrator.check_needs_tools("What is 2 + 2?")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
