"""LLM Agent package with tool calling capabilities."""

from .core import BaseAgent
from .core import ConversationHistory
from .core import Message
from .gemma_agent import AgentMode
from .gemma_agent import GemmaAgent
from .gemma_agent import LightweightGemmaAgent
from .gemma_agent import UnifiedGemmaAgent
from .gemma_agent import create_gemma_agent
from .tools import ToolDefinition
from .tools import ToolParameter
from .tools import ToolRegistry
from .tools import ToolResult
from .tools import tool_registry

__all__ = [
    # Agent modes
    "AgentMode",
    # Core classes
    "BaseAgent",
    "ConversationHistory",
    # Legacy agents (for backwards compatibility)
    "GemmaAgent",
    "LightweightGemmaAgent",
    "Message",
    # Tool classes
    "ToolDefinition",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    # Unified agents
    "UnifiedGemmaAgent",
    # Factory functions
    "create_gemma_agent",
    "tool_registry",
]

__version__ = "1.0.0"
