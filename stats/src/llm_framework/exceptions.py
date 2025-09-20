"""Exception classes for the LLM Framework."""

from typing import Any, Optional


class LLMFrameworkError(Exception):
    """Base exception for LLM Framework errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ModelNotFoundError(LLMFrameworkError):
    """Raised when a requested model cannot be found."""
    
    def __init__(self, model_name: str, available_models: Optional[list[str]] = None) -> None:
        message = f"Model '{model_name}' not found"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        
        super().__init__(message, {"model_name": model_name, "available_models": available_models})
        self.model_name = model_name
        self.available_models = available_models


class InferenceError(LLMFrameworkError):
    """Raised when inference fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, backend: Optional[str] = None) -> None:
        super().__init__(message, {"model_name": model_name, "backend": backend})
        self.model_name = model_name
        self.backend = backend


class ConfigurationError(LLMFrameworkError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        super().__init__(message, {"config_key": config_key})
        self.config_key = config_key


class PluginError(LLMFrameworkError):
    """Raised when plugin operations fail."""
    
    def __init__(self, message: str, plugin_name: Optional[str] = None) -> None:
        super().__init__(message, {"plugin_name": plugin_name})
        self.plugin_name = plugin_name


class ModelLoadError(LLMFrameworkError):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, model_path: Optional[str] = None, backend: Optional[str] = None) -> None:
        super().__init__(message, {"model_path": model_path, "backend": backend})
        self.model_path = model_path
        self.backend = backend


class BackendError(LLMFrameworkError):
    """Raised when backend operations fail."""
    
    def __init__(self, message: str, backend_type: Optional[str] = None) -> None:
        super().__init__(message, {"backend_type": backend_type})
        self.backend_type = backend_type