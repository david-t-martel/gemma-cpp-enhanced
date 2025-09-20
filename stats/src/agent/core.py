"""Core agent implementation with tool calling capability."""

import json
import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from ..shared.logging import get_logger
from .tools import ToolRegistry
from .tools import ToolResult

logger = get_logger(__name__)


@dataclass
class Message:
    """Represents a conversation message."""

    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ConversationHistory:
    """Manages conversation history."""

    messages: list[Message] = field(default_factory=list)
    max_length: int = 20

    def add_message(self, message: Message) -> None:
        """Add a message to history."""
        self.messages.append(message)
        # Keep only the last max_length messages
        if len(self.messages) > self.max_length:
            self.messages = self.messages[-self.max_length :]

    def get_messages(self) -> list[Message]:
        """Get all messages."""
        return self.messages

    def clear(self) -> None:
        """Clear history."""
        self.messages = []

    def to_dict(self) -> list[dict[str, Any]]:
        """Convert to dictionary format."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": msg.tool_calls,
                "tool_call_id": msg.tool_call_id,
            }
            for msg in self.messages
        ]


class BaseAgent(ABC):
    """Base agent class with tool calling capability."""

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        """Initialize the agent.

        Args:
            tool_registry: Registry of available tools
            system_prompt: System prompt for the agent
            max_iterations: Maximum number of tool calling iterations
            verbose: Whether to print verbose output
        """
        self.tool_registry = tool_registry or ToolRegistry()
        self.history = ConversationHistory()
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt with tool instructions."""
        tool_schemas = self.tool_registry.get_tool_schemas()
        tools_json = json.dumps(tool_schemas, indent=2)

        return f"""You are a helpful AI assistant with access to various tools.

You can call tools by outputting a special JSON block in your response.
When you need to use a tool, output it in this exact format:

```tool_call
{{
    "name": "tool_name",
    "arguments": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
```

Available tools:
{tools_json}

Instructions:
1. You can call multiple tools in a single response by including multiple tool_call blocks
2. After calling a tool, you will receive the result and can continue the conversation
3. Use tools when they would be helpful to answer the user's question
4. Always validate tool arguments before calling
5. If a tool returns an error, try to handle it gracefully

Be conversational and helpful. Explain what you're doing when using tools."""

    def parse_tool_calls(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse tool calls from model output.

        Args:
            text: Model output text

        Returns:
            Tuple of (cleaned_text, tool_calls)
        """
        tool_calls = []

        # Find all tool_call blocks
        pattern = r"```tool_call\s*\n(.*?)\n```"
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_json = match.group(1)
                tool_call = json.loads(tool_json)
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
                continue

        # Remove tool_call blocks from text
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

        return cleaned_text, tool_calls

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolResult]:
        """Execute a list of tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of ToolResult objects
        """
        results = []

        for tool_call in tool_calls:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if name is None:
                result = ToolResult(success=False, output=None, error="Tool name is required")
                results.append(result)
                continue

            if self.verbose:
                print(f"\n🔧 Executing tool: {name}")
                print(f"   Arguments: {json.dumps(arguments, indent=2)}")

            result = self.tool_registry.execute_tool(name, arguments)
            results.append(result)

            if self.verbose:
                if result.success:
                    output = str(result.output)
                    if len(output) > 200:
                        print(f"   ✅ Result: {output[:200]}...")
                    else:
                        print(f"   ✅ Result: {result.output}")
                else:
                    print(f"   ❌ Error: {result.error}")

        return results

    def format_tool_results(
        self, tool_calls: list[dict[str, Any]], results: list[ToolResult]
    ) -> str:
        """Format tool results for inclusion in conversation.

        Args:
            tool_calls: Original tool calls
            results: Execution results

        Returns:
            Formatted string of results
        """
        formatted = "Tool Results:\n"
        for tool_call, result in zip(tool_calls, results, strict=False):
            formatted += f"\n[{tool_call['name']}]:\n"
            if result.success:
                formatted += f"{result.output}\n"
            else:
                formatted += f"Error: {result.error}\n"

        return formatted

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response to a prompt.

        This method must be implemented by specific model implementations.

        Args:
            prompt: Input prompt

        Returns:
            Model response
        """

    def chat(self, user_input: str, use_tools: bool = True) -> str:
        """Main chat interface with tool calling support.

        Args:
            user_input: User input message
            use_tools: Whether to enable tool calling

        Returns:
            Final assistant response
        """
        # Add user message to history
        self.history.add_message(Message(role="user", content=user_input))

        # Build the conversation context
        context = self._build_context()

        # Main interaction loop with tool calling
        iterations = 0
        final_response = ""

        while iterations < self.max_iterations:
            iterations += 1

            if self.verbose:
                print(f"\n🤖 Generating response (iteration {iterations})...")

            # Generate response from model
            response = self.generate_response(context)

            if not use_tools:
                # If tools are disabled, return the response directly
                self.history.add_message(Message(role="assistant", content=response))
                return response

            # Parse tool calls from response
            cleaned_response, tool_calls = self.parse_tool_calls(response)

            if not tool_calls:
                # No tool calls, return the response
                final_response = cleaned_response
                self.history.add_message(
                    Message(role="assistant", content=final_response, tool_calls=None)
                )
                break

            # Execute tool calls
            results = self.execute_tool_calls(tool_calls)

            # Add assistant message with tool calls to history
            self.history.add_message(
                Message(role="assistant", content=cleaned_response, tool_calls=tool_calls)
            )

            # Format and add tool results to context
            tool_results = self.format_tool_results(tool_calls, results)
            self.history.add_message(Message(role="tool", content=tool_results))

            # Update context for next iteration
            context = self._build_context()

        return final_response

    def _build_context(self) -> str:
        """Build the conversation context for the model.

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add system prompt
        context_parts.append(f"System: {self.system_prompt}\n")

        # Add conversation history
        for msg in self.history.get_messages():
            if msg.role == "user":
                context_parts.append(f"\nUser: {msg.content}")
            elif msg.role == "assistant":
                content = msg.content
                if msg.tool_calls:
                    # Include tool calls in context
                    for tool_call in msg.tool_calls:
                        content += f"\n```tool_call\n{json.dumps(tool_call, indent=2)}\n```"
                context_parts.append(f"\nAssistant: {content}")
            elif msg.role == "tool":
                context_parts.append(f"\n{msg.content}")

        context_parts.append("\n\nAssistant: ")

        return "\n".join(context_parts)

    def reset(self) -> None:
        """Reset the conversation history."""
        self.history.clear()
        if self.verbose:
            print("Conversation history cleared.")

    def save_history(self, path: str) -> None:
        """Save conversation history to file.

        Args:
            path: Path to save file
        """
        history_data = self.history.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)
        if self.verbose:
            print(f"History saved to {path}")

    def load_history(self, path: str) -> None:
        """Load conversation history from file.

        Args:
            path: Path to load file
        """
        with open(path, encoding="utf-8") as f:
            history_data = json.load(f)

        self.history.clear()
        for msg_data in history_data:
            message = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
            )
            self.history.add_message(message)

        if self.verbose:
            print(f"History loaded from {path}")
