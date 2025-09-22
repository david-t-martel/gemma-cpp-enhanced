"""Professional LLM Framework for unified model interface."""

from .core import LLMFramework, LLMConfig, LLMProvider, quick_generate, create_framework
from .backends import ModelBackend, BackendType, GenerationConfig
from .models import ModelInfo, ModelRegistry, ModelType, ModelCapabilities
from .inference import InferenceEngine, InferenceRequest, InferenceResponse
from .plugins import PluginManager, BaseModelPlugin
from .config import ConfigLoader, load_framework_config
from .integration import (
    FrameworkAgent,
    AgentFrameworkAdapter,
    LegacyAgentWrapper,
    create_gemma_agent,
    upgrade_to_framework,
)
from .exceptions import (
    LLMFrameworkError,
    ModelNotFoundError,
    InferenceError,
    ConfigurationError,
    PluginError,
    ModelLoadError,
    BackendError,
)

__version__ = "1.0.0"

__all__ = [
    # Core framework
    "LLMFramework",
    "LLMConfig", 
    "LLMProvider",
    "quick_generate",
    "create_framework",
    # Backends
    "ModelBackend",
    "BackendType",
    "GenerationConfig",
    # Models
    "ModelInfo",
    "ModelRegistry",
    "ModelType",
    "ModelCapabilities",
    # Inference
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    # Plugins
    "PluginManager",
    "BaseModelPlugin",
    # Configuration
    "ConfigLoader",
    "load_framework_config",
    # Integration
    "FrameworkAgent",
    "AgentFrameworkAdapter",
    "LegacyAgentWrapper",
    "create_gemma_agent",
    "upgrade_to_framework",
    # Exceptions
    "LLMFrameworkError",
    "ModelNotFoundError", 
    "InferenceError",
    "ConfigurationError",
    "PluginError",
    "ModelLoadError",
    "BackendError",
]