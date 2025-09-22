"""Comprehensive unit tests for the core agent implementation."""

import json
import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
from typing import Any

from src.agent.core import BaseAgent, Message, ConversationHistory
from src.agent.tools import ToolRegistry, ToolResult, ToolDefinition, ToolParameter


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.generated_responses: list[str] = []
        self.response_index = 0

    def set_responses(self, responses: list[str]) -> None:
        """Set responses to return from generate_response."""
        self.generated_responses = responses
        self.response_index = 0

    def generate_response(self, prompt: str) -> str:
        """Mock response generation."""
        if self.response_index < len(self.generated_responses):
            response = self.generated_responses[self.response_index]
            self.response_index += 1
            return response
        return "Mock response"


class TestMessage:
    """Test the Message dataclass."""

    def test_message_creation(self):
        """Test creating a message with all fields."""
        msg = Message(
            role="user",
            content="test content",
            tool_calls=[{"name": "test", "args": {}}],
            tool_call_id="test_id"
        )

        assert msg.role == "user"
        assert msg.content == "test content"
        assert msg.tool_calls == [{"name": "test", "args": {}}]
        assert msg.tool_call_id == "test_id"

    def test_message_minimal(self):
        """Test creating a minimal message."""
        msg = Message(role="assistant", content="hello")

        assert msg.role == "assistant"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None


class TestConversationHistory:
    """Test the ConversationHistory class."""

    def test_initialization(self):
        """Test history initialization."""
        history = ConversationHistory()

        assert history.messages == []
        assert history.max_length == 20

    def test_initialization_with_params(self):
        """Test history initialization with custom parameters."""
        history = ConversationHistory(max_length=5)

        assert history.messages == []
        assert history.max_length == 5

    def test_add_message(self):
        """Test adding messages to history."""
        history = ConversationHistory()
        msg = Message(role="user", content="test")

        history.add_message(msg)

        assert len(history.messages) == 1
        assert history.messages[0] == msg

    def test_max_length_enforcement(self):
        """Test that history respects max_length."""
        history = ConversationHistory(max_length=3)

        # Add more messages than max_length
        for i in range(5):
            msg = Message(role="user", content=f"message {i}")
            history.add_message(msg)

        # Should only keep the last 3 messages
        assert len(history.messages) == 3
        assert history.messages[0].content == "message 2"
        assert history.messages[1].content == "message 3"
        assert history.messages[2].content == "message 4"

    def test_get_messages(self):
        """Test getting all messages."""
        history = ConversationHistory()
        msg1 = Message(role="user", content="first")
        msg2 = Message(role="assistant", content="second")

        history.add_message(msg1)
        history.add_message(msg2)

        messages = history.get_messages()
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2

    def test_clear(self):
        """Test clearing history."""
        history = ConversationHistory()
        history.add_message(Message(role="user", content="test"))

        history.clear()

        assert len(history.messages) == 0

    def test_to_dict(self):
        """Test converting history to dictionary format."""
        history = ConversationHistory()
        msg = Message(
            role="user",
            content="test",
            tool_calls=[{"name": "test"}],
            tool_call_id="id"
        )
        history.add_message(msg)

        result = history.to_dict()

        expected = [{
            "role": "user",
            "content": "test",
            "tool_calls": [{"name": "test"}],
            "tool_call_id": "id"
        }]
        assert result == expected

    def test_to_dict_none_values(self):
        """Test to_dict with None values."""
        history = ConversationHistory()
        msg = Message(role="assistant", content="test")
        history.add_message(msg)

        result = history.to_dict()

        expected = [{
            "role": "assistant",
            "content": "test",
            "tool_calls": None,
            "tool_call_id": None
        }]
        assert result == expected


class TestBaseAgent:
    """Test the BaseAgent class."""

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        registry.get_tool_schemas.return_value = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        ]
        return registry

    @pytest.fixture
    def mock_agent(self, mock_tool_registry):
        """Create a mock agent instance."""
        return MockAgent(tool_registry=mock_tool_registry, verbose=False)

    def test_initialization_defaults(self):
        """Test agent initialization with defaults."""
        agent = MockAgent()

        assert agent.tool_registry is not None
        assert isinstance(agent.history, ConversationHistory)
        assert agent.max_iterations == 5
        assert agent.verbose is True
        assert "helpful AI assistant" in agent.system_prompt

    def test_initialization_custom_params(self, mock_tool_registry):
        """Test agent initialization with custom parameters."""
        custom_prompt = "Custom system prompt"
        agent = MockAgent(
            tool_registry=mock_tool_registry,
            system_prompt=custom_prompt,
            max_iterations=10,
            verbose=False
        )

        assert agent.tool_registry == mock_tool_registry
        assert agent.system_prompt == custom_prompt
        assert agent.max_iterations == 10
        assert agent.verbose is False

    def test_get_default_system_prompt(self, mock_tool_registry):
        """Test default system prompt generation."""
        agent = MockAgent(tool_registry=mock_tool_registry)

        prompt = agent._get_default_system_prompt()

        assert "helpful AI assistant" in prompt
        assert "tool_call" in prompt
        assert "test_tool" in prompt
        # Check that get_tool_schemas was called (may be multiple times due to other tests)
        assert mock_tool_registry.get_tool_schemas.called

    def test_parse_tool_calls_simple(self, mock_agent):
        """Test parsing simple tool calls."""
        text = """Here's a response.

```tool_call
{
    "name": "calculator",
    "arguments": {
        "expression": "2 + 2"
    }
}
```

That should work."""

        cleaned_text, tool_calls = mock_agent.parse_tool_calls(text)

        assert "Here's a response." in cleaned_text
        assert "That should work." in cleaned_text
        assert "tool_call" not in cleaned_text
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "calculator"
        assert tool_calls[0]["arguments"]["expression"] == "2 + 2"

    def test_parse_tool_calls_multiple(self, mock_agent):
        """Test parsing multiple tool calls."""
        text = """
```tool_call
{"name": "tool1", "arguments": {"arg1": "value1"}}
```

Some text between.

```tool_call
{"name": "tool2", "arguments": {"arg2": "value2"}}
```
"""

        cleaned_text, tool_calls = mock_agent.parse_tool_calls(text)

        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool1"
        assert tool_calls[1]["name"] == "tool2"
        assert "Some text between." in cleaned_text

    def test_parse_tool_calls_invalid_json(self, mock_agent):
        """Test parsing with invalid JSON."""
        text = """
```tool_call
{invalid json}
```
"""

        with patch('src.agent.core.logger') as mock_logger:
            cleaned_text, tool_calls = mock_agent.parse_tool_calls(text)

            assert len(tool_calls) == 0
            mock_logger.warning.assert_called_once()

    def test_parse_tool_calls_missing_fields(self, mock_agent):
        """Test parsing with missing required fields."""
        text = """
```tool_call
{"name": "tool1"}
```

```tool_call
{"arguments": {"arg": "value"}}
```
"""

        cleaned_text, tool_calls = mock_agent.parse_tool_calls(text)

        # Should ignore tool calls without required fields
        assert len(tool_calls) == 0

    def test_execute_tool_calls_success(self, mock_agent):
        """Test successful tool execution."""
        mock_agent.tool_registry.execute_tool.return_value = ToolResult(
            success=True, output="Success result"
        )

        tool_calls = [{"name": "test_tool", "arguments": {"arg": "value"}}]
        results = mock_agent.execute_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "Success result"
        mock_agent.tool_registry.execute_tool.assert_called_once_with(
            "test_tool", {"arg": "value"}
        )

    def test_execute_tool_calls_failure(self, mock_agent):
        """Test failed tool execution."""
        mock_agent.tool_registry.execute_tool.return_value = ToolResult(
            success=False, output=None, error="Tool failed"
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]
        results = mock_agent.execute_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error == "Tool failed"

    def test_execute_tool_calls_missing_name(self, mock_agent):
        """Test tool execution with missing tool name."""
        tool_calls = [{"arguments": {"arg": "value"}}]
        results = mock_agent.execute_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert "Tool name is required" in results[0].error

    def test_execute_tool_calls_verbose_output(self, mock_tool_registry):
        """Test verbose output during tool execution."""
        agent = MockAgent(tool_registry=mock_tool_registry, verbose=True)
        mock_tool_registry.execute_tool.return_value = ToolResult(
            success=True, output="Short result"
        )

        tool_calls = [{"name": "test_tool", "arguments": {"arg": "value"}}]

        with patch('builtins.print') as mock_print:
            agent.execute_tool_calls(tool_calls)

            # Should print execution details
            assert mock_print.call_count >= 2
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Executing tool" in call for call in print_calls)
            assert any("Result:" in call for call in print_calls)

    def test_execute_tool_calls_long_output_truncation(self, mock_tool_registry):
        """Test output truncation for long results."""
        agent = MockAgent(tool_registry=mock_tool_registry, verbose=True)
        long_output = "x" * 300
        mock_tool_registry.execute_tool.return_value = ToolResult(
            success=True, output=long_output
        )

        tool_calls = [{"name": "test_tool", "arguments": {}}]

        with patch('builtins.print') as mock_print:
            agent.execute_tool_calls(tool_calls)

            # Should truncate long output
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            result_call = next(call for call in print_calls if "Result:" in call)
            assert "..." in result_call

    def test_format_tool_results(self, mock_agent):
        """Test formatting tool results."""
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}}
        ]
        results = [
            ToolResult(success=True, output="Result 1"),
            ToolResult(success=False, output=None, error="Error 2")
        ]

        formatted = mock_agent.format_tool_results(tool_calls, results)

        assert "Tool Results:" in formatted
        assert "[tool1]:" in formatted
        assert "Result 1" in formatted
        assert "[tool2]:" in formatted
        assert "Error: Error 2" in formatted

    def test_chat_without_tools(self, mock_agent):
        """Test chat with tools disabled."""
        mock_agent.set_responses(["Simple response"])

        response = mock_agent.chat("Hello", use_tools=False)

        assert response == "Simple response"
        assert len(mock_agent.history.messages) == 2  # user + assistant
        assert mock_agent.history.messages[0].role == "user"
        assert mock_agent.history.messages[1].role == "assistant"

    def test_chat_with_tools_no_tool_calls(self, mock_agent):
        """Test chat with tools enabled but no tool calls."""
        mock_agent.set_responses(["Response without tools"])

        response = mock_agent.chat("Hello", use_tools=True)

        assert response == "Response without tools"
        assert len(mock_agent.history.messages) == 2

    def test_chat_with_tool_calls(self, mock_agent):
        """Test chat with tool calls."""
        mock_agent.set_responses([
            """First response.
```tool_call
{"name": "test_tool", "arguments": {"arg": "value"}}
```
""",
            "Final response after tool execution"
        ])

        mock_agent.tool_registry.execute_tool.return_value = ToolResult(
            success=True, output="Tool result"
        )

        response = mock_agent.chat("Hello with tools")

        assert response == "Final response after tool execution"
        # Should have user, assistant with tool call, tool result, and final assistant
        assert len(mock_agent.history.messages) == 4

    def test_chat_max_iterations(self, mock_agent):
        """Test chat respecting max iterations."""
        # Always return tool calls to force iteration limit
        tool_call_response = """
```tool_call
{"name": "test_tool", "arguments": {}}
```
"""
        mock_agent.set_responses([tool_call_response] * 10)  # More than max_iterations
        mock_agent.tool_registry.execute_tool.return_value = ToolResult(
            success=True, output="Tool result"
        )

        response = mock_agent.chat("Test max iterations")

        # Should stop at max_iterations
        assert response == ""  # No final response since we keep calling tools

    def test_build_context(self, mock_agent):
        """Test building conversation context."""
        # Add some history
        mock_agent.history.add_message(Message(role="user", content="User message"))
        mock_agent.history.add_message(Message(
            role="assistant",
            content="Assistant message",
            tool_calls=[{"name": "tool", "arguments": {}}]
        ))
        mock_agent.history.add_message(Message(role="tool", content="Tool result"))

        context = mock_agent._build_context()

        assert "System:" in context
        assert "User: User message" in context
        assert "Assistant: Assistant message" in context
        assert "tool_call" in context
        assert "Tool result" in context
        assert context.endswith("Assistant: ")

    def test_reset(self, mock_agent):
        """Test resetting conversation history."""
        mock_agent.history.add_message(Message(role="user", content="test"))

        # Test reset without verbose mode
        mock_agent.reset()
        assert len(mock_agent.history.messages) == 0

        # Test reset with verbose mode
        mock_agent.verbose = True
        mock_agent.history.add_message(Message(role="user", content="test2"))
        with patch('builtins.print') as mock_print:
            mock_agent.reset()
            assert len(mock_agent.history.messages) == 0
            assert mock_print.called

    def test_save_history(self, mock_agent):
        """Test saving conversation history."""
        mock_agent.history.add_message(Message(role="user", content="test"))

        mock_file = mock_open()

        # Test save without verbose mode
        with patch('builtins.open', mock_file):
            mock_agent.save_history("/test/path.json")
            mock_file.assert_called_with("/test/path.json", "w", encoding="utf-8")
            # Check that JSON was written
            handle = mock_file()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            assert "test" in written_data

        # Test save with verbose mode
        mock_agent.verbose = True
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
             patch('builtins.print') as mock_print:
            mock_agent.save_history("/test/path.json")
            assert mock_print.called

    def test_load_history(self, mock_agent):
        """Test loading conversation history."""
        history_data = [
            {
                "role": "user",
                "content": "test message",
                "tool_calls": None,
                "tool_call_id": None
            }
        ]

        mock_file = mock_open(read_data=json.dumps(history_data))

        # Test load without verbose mode
        with patch('builtins.open', mock_file):
            mock_agent.load_history("/test/path.json")
            mock_file.assert_called_with("/test/path.json", encoding="utf-8")
            assert len(mock_agent.history.messages) == 1
            assert mock_agent.history.messages[0].content == "test message"

        # Test load with verbose mode
        mock_agent.verbose = True
        mock_file = mock_open(read_data=json.dumps(history_data))
        with patch('builtins.open', mock_file), \
             patch('builtins.print') as mock_print:
            mock_agent.load_history("/test/path.json")
            assert mock_print.called

    def test_load_history_with_tool_calls(self, mock_agent):
        """Test loading history with tool calls."""
        history_data = [
            {
                "role": "assistant",
                "content": "response",
                "tool_calls": [{"name": "test", "args": {}}],
                "tool_call_id": "test_id"
            }
        ]

        mock_file = mock_open(read_data=json.dumps(history_data))
        with patch('builtins.open', mock_file):
            mock_agent.load_history("/test/path.json")

            message = mock_agent.history.messages[0]
            assert message.tool_calls == [{"name": "test", "args": {}}]
            assert message.tool_call_id == "test_id"

    def test_save_load_history_integration(self, mock_agent):
        """Test save and load history working together."""
        # Add some complex history
        mock_agent.history.add_message(Message(
            role="user",
            content="user message"
        ))
        mock_agent.history.add_message(Message(
            role="assistant",
            content="assistant response",
            tool_calls=[{"name": "test_tool", "arguments": {"arg": "value"}}]
        ))

        # Save history
        mock_file_write = mock_open()
        saved_data = ""

        def capture_write(data):
            nonlocal saved_data
            saved_data += data

        mock_file_write().write.side_effect = capture_write

        with patch('builtins.open', mock_file_write), \
             patch('builtins.print'):
            mock_agent.save_history("/test/save.json")

        # Clear history and load it back
        mock_agent.history.clear()
        assert len(mock_agent.history.messages) == 0

        mock_file_read = mock_open(read_data=saved_data)
        with patch('builtins.open', mock_file_read), \
             patch('builtins.print'):
            mock_agent.load_history("/test/save.json")

        # Verify history was restored
        assert len(mock_agent.history.messages) == 2
        assert mock_agent.history.messages[0].content == "user message"
        assert mock_agent.history.messages[1].content == "assistant response"
        assert mock_agent.history.messages[1].tool_calls == [{"name": "test_tool", "arguments": {"arg": "value"}}]