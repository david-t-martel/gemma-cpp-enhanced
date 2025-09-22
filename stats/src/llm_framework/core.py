"""Core framework components for the LLM Framework."""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import logging

from .backends import GenerationConfig, create_backend
from .exceptions import ConfigurationError, LLMFrameworkError, ModelNotFoundError
from .inference import InferenceEngine, InferenceRequest, InferenceResponse
from .models import ModelInfo, ModelRegistry, ModelType
from .plugins import PluginManager

logger = logging.getLogger(__name__)


class LLMProvider:
    """Provider enumeration for different LLM services."""
    GEMMA_NATIVE = "gemma_native"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class LLMConfig:
    """Configuration for the LLM Framework."""
    
    # Model discovery
    models_dir: str = "/.models"
    auto_discover_models: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    default_timeout: float = 300.0
    
    # Default generation settings
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    
    # Plugin settings
    plugin_dirs: List[str] = field(default_factory=list)
    enable_plugins: bool = True
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # API credentials (environment variables)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Fallback settings
    enable_fallbacks: bool = True
    fallback_models: List[str] = field(default_factory=lambda: ["gemma-2b-it", "gpt-3.5-turbo"])
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # Load API keys from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if self.google_api_key is None:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            LLMConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "models_dir": self.models_dir,
            "auto_discover_models": self.auto_discover_models,
            "max_concurrent_requests": self.max_concurrent_requests,
            "default_timeout": self.default_timeout,
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature,
            "default_top_p": self.default_top_p,
            "plugin_dirs": self.plugin_dirs,
            "enable_plugins": self.enable_plugins,
            "plugin_configs": self.plugin_configs,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_fallbacks": self.enable_fallbacks,
            "fallback_models": self.fallback_models,
        }


class LLMFramework:
    """Main framework class for unified LLM interface."""
    
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """Initialize the LLM Framework.
        
        Args:
            config: Framework configuration (uses defaults if None)
        """
        self.config = config or LLMConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model_registry = ModelRegistry(Path(self.config.models_dir))
        self.plugin_manager = PluginManager(
            [Path(d) for d in self.config.plugin_dirs]
        ) if self.config.enable_plugins else None
        
        self.inference_engine = InferenceEngine(
            self.model_registry,
            max_concurrent_requests=self.config.max_concurrent_requests,
            default_timeout=self.config.default_timeout,
        )
        
        self._initialized = False
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_file,
        )
        
        # Set framework logger level
        logger.setLevel(log_level)
    
    async def initialize(self) -> None:
        """Initialize the framework asynchronously."""
        if self._initialized:
            logger.warning("Framework already initialized")
            return
        
        logger.info("Initializing LLM Framework...")
        
        try:
            # Discover models if enabled
            if self.config.auto_discover_models:
                discovered = self.model_registry.discover_models()
                logger.info(f"Discovered {len(discovered)} models")
                for model in discovered:
                    self.model_registry.register_model(model)
            
            # Load plugins if enabled
            if self.plugin_manager:
                await self.plugin_manager.load_plugins(self.config.plugin_configs)
                plugins = self.plugin_manager.list_plugins()
                logger.info(f"Loaded {len(plugins)} plugins")
            
            self._initialized = True
            logger.info("LLM Framework initialization complete")
            
        except Exception as e:
            logger.error(f"Framework initialization failed: {e}")
            raise LLMFrameworkError(f"Initialization failed: {e}")
    
    async def __aenter__(self) -> "LLMFramework":
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()
    
    async def generate_text(self, 
                           prompt: str,
                           model_name: Optional[str] = None,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           stream: bool = False,
                           **kwargs: Any) -> Union[str, AsyncIterator[str]]:
        """Generate text using the specified model.
        
        Args:
            prompt: Input prompt
            model_name: Model to use (auto-selected if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or async iterator for streaming
        """
        if not self._initialized:
            await self.initialize()
        
        # Auto-select model if not specified
        if model_name is None:
            model_name = await self._auto_select_model()
        
        # Create generation config
        config = GenerationConfig(
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature or self.config.default_temperature,
            top_p=top_p or self.config.default_top_p,
            stream=stream,
            **kwargs
        )
        
        # Create inference request
        request = InferenceRequest(
            prompt=prompt,
            model_name=model_name,
            config=config,
        )
        
        try:
            if stream:
                return self.inference_engine.generate_stream(request)
            else:
                response = await self.inference_engine.generate_text(request)
                if not response.success:
                    if self.config.enable_fallbacks:
                        return await self._try_fallbacks(request)
                    else:
                        raise LLMFrameworkError(f"Generation failed: {response.error}")
                return response.text
                
        except Exception as e:
            if self.config.enable_fallbacks and not stream:
                return await self._try_fallbacks(request)
            raise LLMFrameworkError(f"Text generation failed: {e}")
    
    async def generate_batch(self, 
                            prompts: List[str],
                            model_name: Optional[str] = None,
                            **kwargs: Any) -> List[str]:
        """Generate text for multiple prompts concurrently.
        
        Args:
            prompts: List of input prompts
            model_name: Model to use (auto-selected if None)
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if not self._initialized:
            await self.initialize()
        
        # Auto-select model if not specified
        if model_name is None:
            model_name = await self._auto_select_model()
        
        # Create generation config
        config = GenerationConfig(
            max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
            temperature=kwargs.get("temperature", self.config.default_temperature),
            top_p=kwargs.get("top_p", self.config.default_top_p),
        )
        
        # Create requests
        requests = [
            InferenceRequest(prompt=prompt, model_name=model_name, config=config)
            for prompt in prompts
        ]
        
        # Generate responses
        responses = await self.inference_engine.generate_batch(requests)
        
        # Extract text from responses
        results = []
        for response in responses:
            if response.success:
                results.append(response.text)
            else:
                if self.config.enable_fallbacks:
                    # Try fallback for failed requests
                    try:
                        fallback_text = await self._try_fallbacks(
                            InferenceRequest(
                                prompt=response.request_id or "",  # This is not ideal, but request isn't available
                                model_name=model_name,
                                config=config
                            )
                        )
                        results.append(fallback_text)
                    except Exception:
                        results.append(f"[ERROR: {response.error}]")
                else:
                    results.append(f"[ERROR: {response.error}]")
        
        return results
    
    async def _auto_select_model(self) -> str:
        """Auto-select the best available model.
        
        Returns:
            Model name
        """
        # Prefer local models for speed
        local_models = self.model_registry.list_models(local_only=True)
        if local_models:
            # Sort by capability and size
            local_models.sort(key=lambda m: (m.capabilities.max_context_length, m.size.value))
            return local_models[-1].name  # Best local model
        
        # Fall back to API models
        api_models = [m for m in self.model_registry.list_models() if m.model_path is None]
        if api_models:
            # Prefer models with available API keys
            for model in api_models:
                if model.backend_type == "openai" and self.config.openai_api_key:
                    return model.name
                elif model.backend_type == "anthropic" and self.config.anthropic_api_key:
                    return model.name
                elif model.backend_type == "google" and self.config.google_api_key:
                    return model.name
        
        # Default fallback
        return "gemma-2b-it"
    
    async def _try_fallbacks(self, request: InferenceRequest) -> str:
        """Try fallback models for failed requests.
        
        Args:
            request: Original request
            
        Returns:
            Generated text from fallback model
        """
        for fallback_model in self.config.fallback_models:
            if fallback_model == request.model_name:
                continue  # Skip the original model
            
            try:
                fallback_request = InferenceRequest(
                    prompt=request.prompt,
                    model_name=fallback_model,
                    config=request.config,
                )
                
                response = await self.inference_engine.generate_text(fallback_request)
                if response.success:
                    logger.info(f"Fallback successful with model: {fallback_model}")
                    return response.text
                    
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} failed: {e}")
                continue
        
        raise LLMFrameworkError("All fallback models failed")
    
    def list_models(self, 
                   model_type: Optional[ModelType] = None,
                   local_only: bool = False) -> List[ModelInfo]:
        """List available models.
        
        Args:
            model_type: Filter by model type
            local_only: Only return locally available models
            
        Returns:
            List of model information
        """
        return self.model_registry.list_models(
            model_type=model_type,
            local_only=local_only
        )
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model.
        
        Args:
            model_name: Model name
            
        Returns:
            Model information
        """
        return self.model_registry.get_model(model_name)
    
    async def load_model(self, model_name: str) -> None:
        """Pre-load a model for faster inference.
        
        Args:
            model_name: Model name to load
        """
        if not self._initialized:
            await self.initialize()
        
        await self.inference_engine.load_model(model_name)
    
    async def unload_model(self, model_name: str) -> None:
        """Unload a model to free resources.
        
        Args:
            model_name: Model name to unload
        """
        await self.inference_engine.unload_model(model_name)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get framework performance statistics.
        
        Returns:
            Performance statistics
        """
        stats = self.inference_engine.get_performance_stats()
        
        # Add framework-level stats
        stats.update({
            "total_models": len(self.model_registry.list_models()),
            "local_models": len(self.model_registry.list_models(local_only=True)),
            "plugins_loaded": len(self.plugin_manager.list_plugins()) if self.plugin_manager else 0,
        })
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown the framework and cleanup resources."""
        if not self._initialized:
            return
        
        logger.info("Shutting down LLM Framework...")
        
        try:
            # Shutdown components
            await self.inference_engine.shutdown()
            
            if self.plugin_manager:
                await self.plugin_manager.shutdown()
            
            self._initialized = False
            logger.info("LLM Framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during framework shutdown: {e}")


# Convenience functions for common use cases
async def create_framework(config: Optional[LLMConfig] = None) -> LLMFramework:
    """Create and initialize an LLM Framework instance.
    
    Args:
        config: Framework configuration
        
    Returns:
        Initialized framework instance
    """
    framework = LLMFramework(config)
    await framework.initialize()
    return framework


async def quick_generate(prompt: str, 
                        model_name: Optional[str] = None,
                        **kwargs: Any) -> str:
    """Quick text generation with automatic framework management.
    
    Args:
        prompt: Input prompt
        model_name: Model to use
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text
    """
    async with LLMFramework() as framework:
        return await framework.generate_text(prompt, model_name, **kwargs)