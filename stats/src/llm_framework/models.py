"""Model registry and metadata management."""

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from .exceptions import ModelNotFoundError, ConfigurationError

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"      # < 1B parameters
    SMALL = "small"    # 1B - 3B parameters
    MEDIUM = "medium"  # 3B - 10B parameters
    LARGE = "large"    # 10B - 50B parameters
    XLARGE = "xlarge"  # > 50B parameters


@dataclass
class ModelCapabilities:
    """Model capabilities metadata."""
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_code: bool = False
    max_context_length: int = 2048
    supports_system_prompt: bool = True


@dataclass
class ModelInfo:
    """Complete model information."""
    name: str
    display_name: str
    model_type: ModelType
    backend_type: str
    size: ModelSize
    capabilities: ModelCapabilities
    
    # File/API information
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    model_id: Optional[str] = None  # For API models
    
    # Configuration
    default_temperature: float = 0.7
    default_max_tokens: int = 512
    default_top_p: float = 0.9
    
    # Metadata
    description: Optional[str] = None
    license: Optional[str] = None
    paper_url: Optional[str] = None
    huggingface_id: Optional[str] = None
    
    # Performance hints
    memory_requirement_gb: Optional[float] = None
    requires_gpu: bool = False
    recommended_batch_size: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        # Convert enums to strings
        result["model_type"] = self.model_type.value
        result["size"] = self.size.value
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from dictionary."""
        # Convert string enums back to enum types
        if isinstance(data["model_type"], str):
            data["model_type"] = ModelType(data["model_type"])
        if isinstance(data["size"], str):
            data["size"] = ModelSize(data["size"])
        
        # Handle capabilities
        if isinstance(data.get("capabilities"), dict):
            data["capabilities"] = ModelCapabilities(**data["capabilities"])
        
        return cls(**data)


class ModelRegistry:
    """Registry for managing available models."""
    
    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """Initialize model registry.
        
        Args:
            models_dir: Directory containing model files (defaults to /.models)
        """
        self.models_dir = models_dir or Path("/.models")
        self._models: Dict[str, ModelInfo] = {}
        self._load_default_models()
        
    def _load_default_models(self) -> None:
        """Load default model configurations."""
        
        # Gemma models (local)
        self.register_model(ModelInfo(
            name="gemma-2b-it",
            display_name="Gemma 2B Instruct",
            model_type=ModelType.CHAT,
            backend_type="gemma_native",
            size=ModelSize.SMALL,
            capabilities=ModelCapabilities(
                supports_streaming=True,
                max_context_length=8192,
                supports_system_prompt=True,
            ),
            model_path=str(self.models_dir / "gemma2-2b-it-sfp.sbs"),
            default_temperature=0.7,
            default_max_tokens=512,
            description="Fast local Gemma model for chat and instruction following",
            memory_requirement_gb=4.0,
            requires_gpu=False,
        ))
        
        self.register_model(ModelInfo(
            name="gemma-7b-it", 
            display_name="Gemma 7B Instruct",
            model_type=ModelType.CHAT,
            backend_type="gemma_native",
            size=ModelSize.MEDIUM,
            capabilities=ModelCapabilities(
                supports_streaming=True,
                max_context_length=8192,
                supports_system_prompt=True,
            ),
            model_path=str(self.models_dir / "gemma2-7b-it-sfp.sbs"),
            default_temperature=0.7,
            default_max_tokens=512,
            description="High-quality local Gemma model for complex tasks",
            memory_requirement_gb=14.0,
            requires_gpu=True,
        ))
        
        # OpenAI models (API)
        self.register_model(ModelInfo(
            name="gpt-4o",
            display_name="GPT-4o",
            model_type=ModelType.CHAT,
            backend_type="openai",
            size=ModelSize.XLARGE,
            capabilities=ModelCapabilities(
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                max_context_length=128000,
                supports_system_prompt=True,
            ),
            model_id="gpt-4o",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            default_temperature=0.7,
            default_max_tokens=4096,
            description="Most capable OpenAI model with vision and function calling",
        ))
        
        self.register_model(ModelInfo(
            name="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo", 
            model_type=ModelType.CHAT,
            backend_type="openai",
            size=ModelSize.LARGE,
            capabilities=ModelCapabilities(
                supports_streaming=True,
                supports_function_calling=True,
                max_context_length=16385,
                supports_system_prompt=True,
            ),
            model_id="gpt-3.5-turbo",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            default_temperature=0.7,
            default_max_tokens=4096,
            description="Fast and capable OpenAI model for most tasks",
        ))
        
        # Claude models (API)
        self.register_model(ModelInfo(
            name="claude-3-sonnet",
            display_name="Claude 3 Sonnet",
            model_type=ModelType.CHAT,
            backend_type="anthropic",
            size=ModelSize.XLARGE,
            capabilities=ModelCapabilities(
                supports_streaming=True,
                supports_vision=True,
                max_context_length=200000,
                supports_system_prompt=True,
            ),
            model_id="claude-3-sonnet-20240229",
            api_endpoint="https://api.anthropic.com/v1/messages",
            default_temperature=0.7,
            default_max_tokens=4096,
            description="Balanced Claude model for analysis and reasoning",
        ))
        
        # HuggingFace models
        self.register_model(ModelInfo(
            name="phi-2",
            display_name="Microsoft Phi-2",
            model_type=ModelType.TEXT_GENERATION,
            backend_type="huggingface",
            size=ModelSize.SMALL,
            capabilities=ModelCapabilities(
                supports_streaming=False,
                max_context_length=2048,
                supports_code=True,
            ),
            huggingface_id="microsoft/phi-2",
            default_temperature=0.7,
            default_max_tokens=512,
            description="Small but capable model for code and reasoning tasks",
            memory_requirement_gb=5.0,
        ))
    
    def register_model(self, model_info: ModelInfo) -> None:
        """Register a new model.
        
        Args:
            model_info: Model information to register
        """
        self._models[model_info.name] = model_info
        logger.debug(f"Registered model: {model_info.name}")
    
    def get_model(self, name: str) -> ModelInfo:
        """Get model information by name.
        
        Args:
            name: Model name
            
        Returns:
            Model information
            
        Raises:
            ModelNotFoundError: If model is not found
        """
        if name not in self._models:
            available = list(self._models.keys())
            raise ModelNotFoundError(name, available)
        
        return self._models[name]
    
    def list_models(self, 
                   model_type: Optional[ModelType] = None,
                   backend_type: Optional[str] = None,
                   size: Optional[ModelSize] = None,
                   local_only: bool = False) -> List[ModelInfo]:
        """List available models with optional filtering.
        
        Args:
            model_type: Filter by model type
            backend_type: Filter by backend type
            size: Filter by model size
            local_only: Only return locally available models
            
        Returns:
            List of matching models
        """
        models = list(self._models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if backend_type:
            models = [m for m in models if m.backend_type == backend_type]
        
        if size:
            models = [m for m in models if m.size == size]
        
        if local_only:
            models = [m for m in models if m.model_path is not None]
        
        return models
    
    def discover_models(self) -> List[ModelInfo]:
        """Discover models in the models directory.
        
        Returns:
            List of discovered models
        """
        discovered = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return discovered
        
        # Look for .sbs files (Gemma single-file format)
        for sbs_file in self.models_dir.glob("*.sbs"):
            if sbs_file.name not in [m.model_path for m in self._models.values()]:
                # Create basic model info for discovered model
                model_name = sbs_file.stem
                model_info = ModelInfo(
                    name=model_name,
                    display_name=model_name.replace("-", " ").title(),
                    model_type=ModelType.TEXT_GENERATION,
                    backend_type="gemma_native",
                    size=ModelSize.MEDIUM,  # Default assumption
                    capabilities=ModelCapabilities(),
                    model_path=str(sbs_file),
                    description=f"Discovered Gemma model: {model_name}",
                )
                discovered.append(model_info)
                logger.info(f"Discovered model: {model_name} at {sbs_file}")
        
        return discovered
    
    def save_registry(self, file_path: Path) -> None:
        """Save registry to JSON file.
        
        Args:
            file_path: Path to save registry
        """
        data = {
            "models": {name: info.to_dict() for name, info in self._models.items()}
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved model registry to {file_path}")
    
    def load_registry(self, file_path: Path) -> None:
        """Load registry from JSON file.
        
        Args:
            file_path: Path to load registry from
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            models_data = data.get("models", {})
            for name, model_data in models_data.items():
                model_info = ModelInfo.from_dict(model_data)
                self._models[name] = model_info
            
            logger.info(f"Loaded model registry from {file_path}")
            
        except FileNotFoundError:
            logger.warning(f"Registry file not found: {file_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load registry: {e}")
    
    def get_compatible_models(self, requirements: dict[str, Any]) -> List[ModelInfo]:
        """Get models compatible with given requirements.
        
        Args:
            requirements: Dictionary of requirements (e.g., {"supports_streaming": True})
            
        Returns:
            List of compatible models
        """
        compatible = []
        
        for model in self._models.values():
            if self._matches_requirements(model, requirements):
                compatible.append(model)
        
        return compatible
    
    def _matches_requirements(self, model: ModelInfo, requirements: dict[str, Any]) -> bool:
        """Check if model matches requirements."""
        for key, value in requirements.items():
            if hasattr(model.capabilities, key):
                if getattr(model.capabilities, key) != value:
                    return False
            elif hasattr(model, key):
                if getattr(model, key) != value:
                    return False
            else:
                # Requirement not supported
                return False
        
        return True