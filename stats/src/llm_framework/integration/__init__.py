"""Integration components for the LLM Framework."""

from .agent_adapter import (
    FrameworkAgent,
    AgentFrameworkAdapter,
    LegacyAgentWrapper,
    create_framework_from_agent_config,
    create_gemma_agent,
    create_gemma_agent_async,
    upgrade_to_framework,
)

__all__ = [
    "FrameworkAgent",
    "AgentFrameworkAdapter", 
    "LegacyAgentWrapper",
    "create_framework_from_agent_config",
    "create_gemma_agent",
    "create_gemma_agent_async", 
    "upgrade_to_framework",
]