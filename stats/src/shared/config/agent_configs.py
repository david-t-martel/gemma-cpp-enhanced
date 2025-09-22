"""Agent-specific configurations for the Gemma LLM Stats project.

This module provides centralized configuration for different types of agents,
their behaviors, and their integration with various components.
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from .model_configs import DEFAULT_MODELS
from .model_configs import InferenceConfig
from .model_configs import ModelSpec
from .model_configs import get_model_spec


class AgentType(str, Enum):
    """Types of agents available in the system."""

    BASE = "base"
    REACT = "react"
    GEMMA = "gemma"
    LIGHTWEIGHT = "lightweight"
    RAG_ENHANCED = "rag_enhanced"


class AgentCapability(str, Enum):
    """Agent capabilities."""

    TOOL_CALLING = "tool_calling"
    PLANNING = "planning"
    REFLECTION = "reflection"
    REASONING = "reasoning"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    FILE_OPERATIONS = "file_operations"
    MEMORY = "memory"
    RAG = "rag"
    MULTIMODAL = "multimodal"


class ToolCategory(str, Enum):
    """Categories of tools available to agents."""

    CORE = "core"  # Essential tools (calculator, datetime)
    FILE = "file"  # File operations
    WEB = "web"  # Web search, URL fetching
    SYSTEM = "system"  # System information
    CODE = "code"  # Code execution
    MEMORY = "memory"  # Memory management
    RAG = "rag"  # RAG operations
    CUSTOM = "custom"  # Custom tools


@dataclass(frozen=True)
class ToolConfig:
    """Configuration for a tool."""

    name: str
    category: ToolCategory
    enabled: bool = True
    timeout_seconds: int = 30
    retry_count: int = 3
    parameters: dict[str, Any] = field(default_factory=dict)


class SystemPromptTemplate(str, Enum):
    """Pre-defined system prompt templates."""

    DEFAULT = "default"
    REACT = "react"
    CODING = "coding"
    RESEARCH = "research"
    ASSISTANT = "assistant"
    CREATIVE = "creative"


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS: dict[SystemPromptTemplate, str] = {
    SystemPromptTemplate.DEFAULT: """You are a helpful AI assistant powered by the Gemma language model.
You can help with a wide range of tasks including answering questions, providing explanations,
helping with analysis, and more. Be helpful, accurate, and concise in your responses.""",
    SystemPromptTemplate.REACT: """You are a reasoning AI agent that follows a structured thought process.

You have access to various tools that can help you gather information and perform actions.
For each user request, follow this pattern:

1. **Thought**: Think about what you need to do and what tools might help
2. **Action**: Choose and use appropriate tools to gather information
3. **Observation**: Analyze the results from your actions
4. **Reflection**: Consider if you have enough information or need to continue

Continue this process until you have enough information to provide a complete answer.

Available capabilities:
- Use tools to gather information
- Perform calculations and analysis
- Access files and system information
- Search the web for current information

Be systematic in your approach and explain your reasoning clearly.""",
    SystemPromptTemplate.CODING: """You are an expert programming assistant powered by Gemma.
You excel at writing, debugging, and explaining code in multiple programming languages.

Key capabilities:
- Write clean, efficient, and well-documented code
- Debug and fix programming issues
- Explain complex programming concepts clearly
- Provide code reviews and optimization suggestions
- Help with architecture and design decisions

When writing code:
- Use appropriate error handling
- Follow best practices and conventions
- Include helpful comments
- Consider performance and security
- Provide example usage when helpful""",
    SystemPromptTemplate.RESEARCH: """You are a research-focused AI assistant with access to various tools and information sources.

Your role is to:
- Conduct thorough research on topics
- Synthesize information from multiple sources
- Provide well-reasoned analysis and conclusions
- Cite sources when possible
- Identify knowledge gaps and limitations

Research process:
1. Break down complex questions into searchable components
2. Gather information from multiple sources
3. Cross-reference and verify facts
4. Synthesize findings into coherent insights
5. Acknowledge uncertainties and limitations

Always strive for accuracy and intellectual honesty in your research.""",
    SystemPromptTemplate.ASSISTANT: """You are a capable personal assistant that can help with a variety of tasks.

Your strengths include:
- Organizing and planning
- Information gathering and synthesis
- Task breakdown and project management
- Communication and writing assistance
- Problem-solving and analysis

Be proactive in:
- Asking clarifying questions
- Suggesting alternative approaches
- Breaking complex tasks into manageable steps
- Providing relevant resources and information

Maintain a professional yet friendly tone and always aim to be genuinely helpful.""",
    SystemPromptTemplate.CREATIVE: """You are a creative AI assistant specializing in generating original ideas and content.

Your creative capabilities include:
- Writing stories, poems, and creative content
- Brainstorming and ideation
- Creative problem-solving
- Content adaptation and variation
- Artistic and design suggestions

Creative principles:
- Embrace originality and innovation
- Consider multiple perspectives
- Balance creativity with practicality
- Encourage exploration of ideas
- Provide constructive creative feedback

Feel free to think outside the box while remaining helpful and appropriate.""",
}

# =============================================================================
# TOOL CONFIGURATIONS
# =============================================================================

DEFAULT_TOOLS: dict[str, ToolConfig] = {
    # Core tools
    "calculator": ToolConfig(name="calculator", category=ToolCategory.CORE, timeout_seconds=10),
    "datetime": ToolConfig(name="datetime", category=ToolCategory.CORE, timeout_seconds=5),
    # File operations
    "read_file": ToolConfig(name="read_file", category=ToolCategory.FILE, timeout_seconds=30),
    "write_file": ToolConfig(name="write_file", category=ToolCategory.FILE, timeout_seconds=30),
    "list_directory": ToolConfig(
        name="list_directory", category=ToolCategory.FILE, timeout_seconds=15
    ),
    # Web operations
    "web_search": ToolConfig(
        name="web_search", category=ToolCategory.WEB, timeout_seconds=30, retry_count=2
    ),
    "fetch_url": ToolConfig(
        name="fetch_url", category=ToolCategory.WEB, timeout_seconds=30, retry_count=2
    ),
    # System operations
    "system_info": ToolConfig(name="system_info", category=ToolCategory.SYSTEM, timeout_seconds=10),
}

# Tool sets for different agent types
AGENT_TOOL_SETS: dict[AgentType, list[str]] = {
    AgentType.BASE: ["calculator", "datetime"],
    AgentType.REACT: ["calculator", "datetime", "system_info"],
    AgentType.GEMMA: [
        "calculator",
        "datetime",
        "read_file",
        "write_file",
        "list_directory",
        "web_search",
        "fetch_url",
        "system_info",
    ],
    AgentType.LIGHTWEIGHT: ["calculator", "datetime", "system_info"],
    AgentType.RAG_ENHANCED: [
        "calculator",
        "datetime",
        "read_file",
        "write_file",
        "web_search",
        "fetch_url",
        "system_info",
    ],
}

# =============================================================================
# AGENT CONFIGURATIONS
# =============================================================================


class AgentConfig(BaseModel):
    """Base configuration for all agent types."""

    # Basic properties
    agent_type: AgentType = AgentType.BASE
    name: str = Field("agent", description="Agent instance name")
    description: str = Field("", description="Agent description")

    # Model configuration
    model_name: str = Field(DEFAULT_MODELS["default"], description="Model to use")
    lightweight: bool = Field(True, description="Use lightweight model loading")

    # Behavior configuration
    max_iterations: int = Field(5, ge=1, le=50, description="Maximum reasoning iterations")
    verbose: bool = Field(True, description="Enable verbose output")
    system_prompt_template: SystemPromptTemplate = SystemPromptTemplate.DEFAULT
    custom_system_prompt: str | None = Field(None, description="Custom system prompt")

    # Tool configuration
    enabled_tools: list[str] = Field(default_factory=list, description="List of enabled tool names")
    tool_timeout: int = Field(30, ge=1, le=300, description="Default tool timeout in seconds")

    # Performance configuration
    device: str = Field("auto", description="Device for model execution")
    use_quantization: bool = Field(False, description="Enable model quantization")

    # Memory and context
    max_context_length: int = Field(2048, ge=512, le=32768, description="Maximum context length")
    memory_enabled: bool = Field(False, description="Enable conversation memory")

    # Inference parameters (will use model defaults if not specified)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)

    @field_validator("enabled_tools")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        """Validate that enabled tools exist."""
        invalid_tools = [tool for tool in v if tool not in DEFAULT_TOOLS]
        if invalid_tools:
            raise ValueError(f"Unknown tools: {invalid_tools}")
        return v

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        if self.custom_system_prompt:
            return self.custom_system_prompt
        return SYSTEM_PROMPTS[self.system_prompt_template]

    def get_model_spec(self) -> ModelSpec:
        """Get the model specification for this agent."""
        return get_model_spec(self.model_name)

    def get_tool_configs(self) -> dict[str, ToolConfig]:
        """Get tool configurations for enabled tools."""
        return {
            tool_name: DEFAULT_TOOLS[tool_name]
            for tool_name in self.enabled_tools
            if tool_name in DEFAULT_TOOLS
        }


class ReActAgentConfig(AgentConfig):
    """Configuration specific to ReAct agents."""

    agent_type: AgentType = AgentType.REACT
    system_prompt_template: SystemPromptTemplate = SystemPromptTemplate.REACT
    max_iterations: int = Field(10, ge=1, le=50)

    # ReAct-specific settings
    enable_planning: bool = Field(True, description="Enable planning phase")
    enable_reflection: bool = Field(True, description="Enable reflection phase")
    planning_depth: int = Field(3, ge=1, le=10, description="Planning depth")
    reflection_frequency: int = Field(3, ge=1, le=10, description="Reflection frequency")

    # Enhanced reasoning
    require_reasoning: bool = Field(True, description="Require explicit reasoning")
    min_reasoning_steps: int = Field(1, ge=1, le=20, description="Minimum reasoning steps")


class GemmaAgentConfig(AgentConfig):
    """Configuration specific to Gemma agents."""

    agent_type: AgentType = AgentType.GEMMA
    model_name: str = Field("google/gemma-2b-it")

    # Gemma-specific performance settings
    use_flash_attention: bool = Field(False, description="Enable Flash Attention")
    use_bettertransformer: bool = Field(False, description="Enable BetterTransformer")
    use_torch_compile: bool = Field(False, description="Enable torch.compile")

    # Advanced model options
    load_in_8bit: bool = Field(False, description="Use 8-bit quantization")
    device_map: str = Field("auto", description="Device mapping strategy")


class RAGAgentConfig(AgentConfig):
    """Configuration for RAG-enhanced agents."""

    agent_type: AgentType = AgentType.RAG_ENHANCED
    memory_enabled: bool = Field(True)

    # RAG-specific settings
    rag_enabled: bool = Field(True, description="Enable RAG functionality")
    vector_store_path: Path | None = Field(None, description="Path to vector store")
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", description="Embedding model"
    )
    chunk_size: int = Field(512, ge=128, le=2048, description="Document chunk size")
    chunk_overlap: int = Field(50, ge=0, le=512, description="Chunk overlap")
    retrieval_top_k: int = Field(5, ge=1, le=50, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")

    # Memory tiers
    enable_short_term_memory: bool = Field(True, description="Enable short-term memory")
    enable_long_term_memory: bool = Field(True, description="Enable long-term memory")
    enable_episodic_memory: bool = Field(False, description="Enable episodic memory")
    enable_semantic_memory: bool = Field(True, description="Enable semantic memory")


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Quick development agent
LIGHTWEIGHT_DEV_CONFIG = AgentConfig(
    agent_type=AgentType.LIGHTWEIGHT,
    name="dev_assistant",
    description="Lightweight development assistant",
    model_name="google/gemma-2b-it",
    lightweight=True,
    max_iterations=5,
    enabled_tools=AGENT_TOOL_SETS[AgentType.LIGHTWEIGHT],
    system_prompt_template=SystemPromptTemplate.ASSISTANT,
)

# Full-featured research agent
RESEARCH_AGENT_CONFIG = ReActAgentConfig(
    name="research_agent",
    description="Research-focused ReAct agent",
    model_name="google/gemma-7b-it",
    lightweight=False,
    max_iterations=15,
    enabled_tools=AGENT_TOOL_SETS[AgentType.GEMMA],
    system_prompt_template=SystemPromptTemplate.RESEARCH,
    enable_planning=True,
    enable_reflection=True,
    use_quantization=True,
)

# High-performance coding assistant
CODING_AGENT_CONFIG = GemmaAgentConfig(
    name="coding_assistant",
    description="High-performance coding assistant",
    model_name="google/gemma-7b-it",
    lightweight=False,
    enabled_tools=AGENT_TOOL_SETS[AgentType.GEMMA],
    system_prompt_template=SystemPromptTemplate.CODING,
    use_flash_attention=True,
    use_quantization=True,
    max_tokens=2048,
)

# RAG-enhanced knowledge agent
RAG_KNOWLEDGE_CONFIG = RAGAgentConfig(
    name="knowledge_agent",
    description="RAG-enhanced knowledge agent",
    model_name="google/gemma-7b-it",
    lightweight=False,
    enabled_tools=AGENT_TOOL_SETS[AgentType.RAG_ENHANCED],
    system_prompt_template=SystemPromptTemplate.RESEARCH,
    rag_enabled=True,
    retrieval_top_k=10,
    enable_long_term_memory=True,
    enable_semantic_memory=True,
)

PRESET_CONFIGS: dict[str, AgentConfig] = {
    "lightweight": LIGHTWEIGHT_DEV_CONFIG,
    "research": RESEARCH_AGENT_CONFIG,
    "coding": CODING_AGENT_CONFIG,
    "knowledge": RAG_KNOWLEDGE_CONFIG,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_agent_config(
    preset: str | None = None,
    agent_type: AgentType = AgentType.BASE,
    model_name: str | None = None,
    **overrides,
) -> AgentConfig:
    """Create an agent configuration.

    Args:
        preset: Use a preset configuration
        agent_type: Type of agent to create
        model_name: Model to use (overrides preset)
        **overrides: Additional configuration overrides

    Returns:
        Configured AgentConfig object
    """
    if preset and preset in PRESET_CONFIGS:
        config = PRESET_CONFIGS[preset].copy()
    # Create base config based on agent type
    elif agent_type == AgentType.REACT:
        config = ReActAgentConfig()
    elif agent_type == AgentType.GEMMA:
        config = GemmaAgentConfig()
    elif agent_type == AgentType.RAG_ENHANCED:
        config = RAGAgentConfig()
    else:
        config = AgentConfig(agent_type=agent_type)

    # Apply overrides
    if model_name:
        overrides["model_name"] = model_name

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Set default tools if not specified
    if not config.enabled_tools:
        config.enabled_tools = AGENT_TOOL_SETS.get(config.agent_type, [])

    return config


def get_recommended_config(use_case: str, resource_level: str = "medium") -> AgentConfig:
    """Get recommended configuration for a use case.

    Args:
        use_case: Use case (development, research, coding, etc.)
        resource_level: Available resources (low, medium, high)

    Returns:
        Recommended AgentConfig
    """
    # Map use cases to presets
    use_case_presets = {
        "development": "lightweight",
        "research": "research",
        "coding": "coding",
        "knowledge": "knowledge",
        "general": "lightweight",
    }

    preset = use_case_presets.get(use_case, "lightweight")
    config = create_agent_config(preset=preset)

    # Adjust for resource level
    if resource_level == "low":
        config.model_name = "google/gemma-2b-it"
        config.lightweight = True
        config.use_quantization = True
        config.max_iterations = min(config.max_iterations, 5)
    elif resource_level == "high":
        config.model_name = "google/gemma-7b-it"
        config.lightweight = False
        config.max_iterations = min(config.max_iterations * 2, 20)

    return config


def validate_agent_config(config: AgentConfig) -> tuple[bool, list[str]]:
    """Validate an agent configuration.

    Args:
        config: AgentConfig to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Validate model exists
    try:
        model_spec = config.get_model_spec()
    except ValueError as e:
        issues.append(f"Invalid model: {e}")
        return False, issues

    # Validate tool availability
    invalid_tools = [tool for tool in config.enabled_tools if tool not in DEFAULT_TOOLS]
    if invalid_tools:
        issues.append(f"Unknown tools: {invalid_tools}")

    # Validate resource requirements
    if hasattr(config, "rag_enabled") and config.rag_enabled and not config.memory_enabled:
        issues.append("RAG requires memory to be enabled")

    # Check for conflicting settings
    if config.lightweight and hasattr(config, "use_flash_attention"):
        if config.use_flash_attention:
            issues.append("Flash Attention not available in lightweight mode")

    return len(issues) == 0, issues
