"""Test script for verifying tool orchestration functionality.

This script tests the autonomous tool calling capabilities without
running the full chat loop.
"""

import asyncio
import logging
from pathlib import Path

from gemma_cli.core.tool_orchestrator import (
    ToolCall,
    ToolCallFormat,
    ToolCallParser,
    ToolOrchestrator,
    ToolSchemaFormatter,
)

# Mock Tool class to avoid circular import
from dataclasses import dataclass as dc
from typing import Any, Optional, Dict

@dc
class Tool:
    """Mock MCP Tool for testing."""
    name: str
    description: Optional[str] = None
    inputSchema: Optional[Dict[str, Any]] = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_mock_tools() -> dict[str, list[Tool]]:
    """Create mock MCP tools for testing."""

    # Mock filesystem tools
    filesystem_tools = [
        Tool(
            name="read_file",
            description="Read contents of a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_directory",
            description="List contents of a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory"
                    }
                },
                "required": ["path"]
            }
        ),
    ]

    # Mock memory tools
    memory_tools = [
        Tool(
            name="store_memory",
            description="Store information in memory for later retrieval",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to store"
                    },
                    "tags": {
                        "type": "array",
                        "description": "Tags for categorizing the memory"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="recall_memory",
            description="Search and retrieve stored memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        ),
    ]

    # Mock web search tool
    search_tools = [
        Tool(
            name="web_search",
            description="Search the web for information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
    ]

    return {
        "filesystem": filesystem_tools,
        "memory": memory_tools,
        "brave_search": search_tools,
    }


def test_schema_formatter():
    """Test the tool schema formatter."""
    print("\n=== Testing Schema Formatter ===")

    tools = create_mock_tools()
    formatter = ToolSchemaFormatter(ToolCallFormat.JSON_BLOCK)

    prompt = formatter.format_tools_for_prompt(tools)
    print(prompt)

    assert "filesystem.read_file" in prompt
    assert "memory.store_memory" in prompt
    assert "brave_search.web_search" in prompt
    assert "```json" in prompt  # Check for JSON format instructions

    print("[PASS] Schema formatter test passed")


def test_tool_call_parser():
    """Test the tool call parser."""
    print("\n=== Testing Tool Call Parser ===")

    parser = ToolCallParser(ToolCallFormat.JSON_BLOCK)

    # Test response with tool call
    response_with_tool = """
    I'll help you read that file. Let me check what's in it.

    ```json
    {
      "tool": "filesystem.read_file",
      "args": {
        "path": "/home/user/README.md"
      }
    }
    ```

    Waiting for the file contents...
    """

    tool_calls = parser.parse_response(response_with_tool)
    assert len(tool_calls) == 1
    assert tool_calls[0].server == "filesystem"
    assert tool_calls[0].tool_name == "read_file"
    assert tool_calls[0].arguments["path"] == "/home/user/README.md"

    print(f"[PASS] Found {len(tool_calls)} tool call(s)")

    # Test response without tool call
    response_without_tool = "This is just a regular response without any tool calls."
    tool_calls = parser.parse_response(response_without_tool)
    assert len(tool_calls) == 0

    print("[PASS] Tool call parser test passed")


def test_multiple_tool_calls():
    """Test parsing multiple tool calls in one response."""
    print("\n=== Testing Multiple Tool Calls ===")

    parser = ToolCallParser(ToolCallFormat.JSON_BLOCK)

    response = """
    I'll search for that information and store it for later.

    First, let me search the web:
    ```json
    {
      "tool": "brave_search.web_search",
      "args": {
        "query": "latest Python release 2024"
      }
    }
    ```

    Now I'll store this in memory:
    ```json
    {
      "tool": "memory.store_memory",
      "args": {
        "content": "Python 3.12 is the latest stable release",
        "tags": ["python", "programming", "version"]
      }
    }
    ```
    """

    tool_calls = parser.parse_response(response)
    assert len(tool_calls) == 2
    assert tool_calls[0].tool_name == "web_search"
    assert tool_calls[1].tool_name == "store_memory"

    print(f"[PASS] Found {len(tool_calls)} tool calls")
    print("[PASS] Multiple tool calls test passed")


class MockMCPManager:
    """Mock MCP manager for testing."""

    async def call_tool(self, server: str, tool: str, args: dict):
        """Mock tool execution."""
        if tool == "read_file":
            return "# README\n\nThis is a test file."
        elif tool == "web_search":
            return {"results": ["Python 3.12 released", "New features in Python 3.12"]}
        elif tool == "store_memory":
            return {"stored": True, "id": "mem_123"}
        else:
            raise ValueError(f"Unknown tool: {tool}")


async def test_orchestrator():
    """Test the full orchestrator."""
    print("\n=== Testing Tool Orchestrator ===")

    tools = create_mock_tools()
    mock_manager = MockMCPManager()

    orchestrator = ToolOrchestrator(
        mcp_manager=mock_manager,
        available_tools=tools,
        format_type=ToolCallFormat.JSON_BLOCK,
        max_tool_depth=3,
        require_confirmation=False
    )

    # Get system prompt
    system_prompt = orchestrator.get_system_prompt()
    assert "filesystem.read_file" in system_prompt
    print("[PASS] System prompt generated")

    # Test processing a response with tool call
    response = """
    Let me read that README file for you.

    ```json
    {
      "tool": "filesystem.read_file",
      "args": {
        "path": "README.md"
      }
    }
    ```
    """

    processed_response, results = await orchestrator.process_response_with_tools(response)

    assert len(results) == 1
    assert results[0].success
    assert "This is a test file" in results[0].output
    assert "[Tool Result:" in processed_response

    print(f"[PASS] Processed {len(results)} tool call(s)")
    print("[PASS] Orchestrator test passed")


async def test_tool_depth_limit():
    """Test that tool depth limits are enforced."""
    print("\n=== Testing Tool Depth Limit ===")

    tools = create_mock_tools()
    mock_manager = MockMCPManager()

    orchestrator = ToolOrchestrator(
        mcp_manager=mock_manager,
        available_tools=tools,
        format_type=ToolCallFormat.JSON_BLOCK,
        max_tool_depth=2,  # Limit to 2 calls
        require_confirmation=False
    )

    # Response with 3 tool calls (should stop at 2)
    response = """
    ```json
    {"tool": "filesystem.read_file", "args": {"path": "file1.txt"}}
    ```

    ```json
    {"tool": "filesystem.read_file", "args": {"path": "file2.txt"}}
    ```

    ```json
    {"tool": "filesystem.read_file", "args": {"path": "file3.txt"}}
    ```
    """

    processed_response, results = await orchestrator.process_response_with_tools(response)

    # Should only execute 2 tools due to depth limit
    assert len(results) == 2
    print(f"[PASS] Depth limit enforced: {len(results)} of 3 tools executed")
    print("[PASS] Depth limit test passed")


async def main():
    """Run all tests."""
    print("Starting Tool Orchestration Tests")
    print("=" * 40)

    try:
        # Synchronous tests
        test_schema_formatter()
        test_tool_call_parser()
        test_multiple_tool_calls()

        # Async tests
        await test_orchestrator()
        await test_tool_depth_limit()

        print("\n" + "=" * 40)
        print("All tests passed successfully!")
        print("\nThe tool orchestration system is ready for integration.")
        print("\nTo use in chat:")
        print("  1. Start chat with: gemma-cli chat --enable-mcp")
        print("  2. Ask questions that require tools:")
        print("     - 'What's in my README.md file?'")
        print("     - 'Search for the latest Python release'")
        print("     - 'Remember that Python 3.12 is the latest version'")

    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)