"""Plugin system for extending the LLM Framework."""

import asyncio
import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import logging

from .backends import ModelBackend, GenerationConfig
from .exceptions import PluginError
from .models import ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    supported_backends: List[str]
    dependencies: List[str] = None
    
    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []


class BaseModelPlugin(ABC):
    """Base class for model plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin.
        
        Args:
            config: Plugin configuration
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources."""
        pass
    
    @abstractmethod
    def supports_model(self, model_info: ModelInfo) -> bool:
        """Check if plugin supports a given model.
        
        Args:
            model_info: Model information
            
        Returns:
            True if model is supported
        """
        pass
    
    @abstractmethod
    async def create_backend(self, model_info: ModelInfo) -> ModelBackend:
        """Create a backend for the model.
        
        Args:
            model_info: Model information
            
        Returns:
            Backend instance
        """
        pass
    
    async def pre_generate_hook(self, 
                               prompt: str, 
                               config: GenerationConfig) -> tuple[str, GenerationConfig]:
        """Hook called before text generation.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Modified prompt and config
        """
        return prompt, config
    
    async def post_generate_hook(self, 
                                prompt: str, 
                                response: str, 
                                config: GenerationConfig) -> str:
        """Hook called after text generation.
        
        Args:
            prompt: Original prompt
            response: Generated response
            config: Generation configuration
            
        Returns:
            Modified response
        """
        return response


class PluginBackend(ModelBackend):
    """Backend wrapper for plugin-provided backends."""
    
    def __init__(self, plugin: BaseModelPlugin, backend: ModelBackend) -> None:
        """Initialize plugin backend wrapper.
        
        Args:
            plugin: Plugin instance
            backend: Wrapped backend
        """
        super().__init__(backend.model_info)
        self.plugin = plugin
        self.backend = backend
    
    async def load_model(self) -> None:
        """Load the model."""
        await self.backend.load_model()
        self._loaded = self.backend.is_loaded
    
    async def unload_model(self) -> None:
        """Unload the model."""
        await self.backend.unload_model()
        self._loaded = False
    
    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text with plugin hooks."""
        # Pre-generation hook
        modified_prompt, modified_config = await self.plugin.pre_generate_hook(prompt, config)
        
        # Generate text
        response = await self.backend.generate_text(modified_prompt, modified_config)
        
        # Post-generation hook
        final_response = await self.plugin.post_generate_hook(prompt, response, config)
        
        return final_response
    
    async def generate_stream(self, prompt: str, config: GenerationConfig):
        """Generate streaming text with plugin hooks."""
        # Pre-generation hook
        modified_prompt, modified_config = await self.plugin.pre_generate_hook(prompt, config)
        
        # Generate streaming text
        response_chunks = []
        async for chunk in self.backend.generate_stream(modified_prompt, modified_config):
            response_chunks.append(chunk)
            yield chunk
        
        # Post-generation hook for complete response
        complete_response = ''.join(response_chunks)
        final_response = await self.plugin.post_generate_hook(prompt, complete_response, config)
        
        # If response was modified, yield the difference
        if final_response != complete_response:
            additional_text = final_response[len(complete_response):]
            if additional_text:
                yield additional_text


class PluginManager:
    """Manager for loading and managing plugins."""
    
    def __init__(self, plugin_dirs: Optional[List[Union[str, Path]]] = None) -> None:
        """Initialize plugin manager.
        
        Args:
            plugin_dirs: Directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, BaseModelPlugin] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # Add default plugin directories
        if not self.plugin_dirs:
            current_dir = Path(__file__).parent
            self.plugin_dirs = [
                current_dir / "plugins",
                Path.cwd() / "plugins",
                Path.home() / ".llm_framework" / "plugins",
            ]
    
    async def load_plugins(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load all available plugins.
        
        Args:
            config: Plugin configurations
        """
        config = config or {}
        self._plugin_configs = config
        
        # Discover and load plugins
        discovered_plugins = self._discover_plugins()
        
        for plugin_class in discovered_plugins:
            try:
                await self._load_plugin(plugin_class)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_class.__name__}: {e}")
    
    async def _load_plugin(self, plugin_class: Type[BaseModelPlugin]) -> None:
        """Load a single plugin.
        
        Args:
            plugin_class: Plugin class to load
        """
        # Create plugin instance
        plugin = plugin_class()
        metadata = plugin.metadata
        
        # Check if plugin is already loaded
        if metadata.name in self._plugins:
            logger.warning(f"Plugin {metadata.name} already loaded, skipping")
            return
        
        # Get plugin configuration
        plugin_config = self._plugin_configs.get(metadata.name, {})
        
        # Check dependencies
        missing_deps = self._check_dependencies(metadata.dependencies)
        if missing_deps:
            raise PluginError(
                f"Missing dependencies for plugin {metadata.name}: {missing_deps}",
                plugin_name=metadata.name
            )
        
        # Initialize plugin
        await plugin.initialize(plugin_config)
        
        # Register plugin
        self._plugins[metadata.name] = plugin
        logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
    
    def _discover_plugins(self) -> List[Type[BaseModelPlugin]]:
        """Discover plugin classes in plugin directories.
        
        Returns:
            List of plugin classes
        """
        plugin_classes = []
        
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Look for Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    # Import module
                    spec = importlib.util.spec_from_file_location(
                        py_file.stem, py_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find plugin classes
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, BaseModelPlugin) and 
                                obj != BaseModelPlugin and
                                not inspect.isabstract(obj)):
                                plugin_classes.append(obj)
                                logger.debug(f"Discovered plugin class: {name} in {py_file}")
                
                except Exception as e:
                    logger.warning(f"Failed to load plugin from {py_file}: {e}")
        
        return plugin_classes
    
    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check if plugin dependencies are available.
        
        Args:
            dependencies: List of dependency names
            
        Returns:
            List of missing dependencies
        """
        missing = []
        
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        
        return missing
    
    def get_plugin_for_model(self, model_info: ModelInfo) -> Optional[BaseModelPlugin]:
        """Get plugin that supports a given model.
        
        Args:
            model_info: Model information
            
        Returns:
            Plugin instance or None if no plugin supports the model
        """
        for plugin in self._plugins.values():
            if plugin.supports_model(model_info):
                return plugin
        
        return None
    
    async def create_backend(self, model_info: ModelInfo) -> Optional[ModelBackend]:
        """Create backend using appropriate plugin.
        
        Args:
            model_info: Model information
            
        Returns:
            Backend instance or None if no plugin supports the model
        """
        plugin = self.get_plugin_for_model(model_info)
        if not plugin:
            return None
        
        try:
            backend = await plugin.create_backend(model_info)
            return PluginBackend(plugin, backend)
        except Exception as e:
            raise PluginError(
                f"Failed to create backend for {model_info.name} using plugin {plugin.metadata.name}: {e}",
                plugin_name=plugin.metadata.name
            )
    
    def list_plugins(self) -> List[PluginMetadata]:
        """Get list of loaded plugins.
        
        Returns:
            List of plugin metadata
        """
        return [plugin.metadata for plugin in self._plugins.values()]
    
    def get_plugin(self, name: str) -> Optional[BaseModelPlugin]:
        """Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            await plugin.shutdown()
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")
    
    async def shutdown(self) -> None:
        """Shutdown all plugins."""
        logger.info("Shutting down plugin manager...")
        
        for name in list(self._plugins.keys()):
            try:
                await self.unload_plugin(name)
            except Exception as e:
                logger.error(f"Error unloading plugin {name}: {e}")
        
        logger.info("Plugin manager shutdown complete")


# Example plugin implementation
class ExamplePlugin(BaseModelPlugin):
    """Example plugin implementation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for demonstration",
            author="LLM Framework",
            supported_backends=["example"],
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.config = config
        logger.info("Example plugin initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        logger.info("Example plugin shutdown")
    
    def supports_model(self, model_info: ModelInfo) -> bool:
        """Check if plugin supports a model."""
        return model_info.backend_type == "example"
    
    async def create_backend(self, model_info: ModelInfo) -> ModelBackend:
        """Create a backend for the model."""
        # This would create an actual backend implementation
        raise NotImplementedError("Example plugin doesn't implement backends")
    
    async def pre_generate_hook(self, prompt: str, config: GenerationConfig) -> tuple[str, GenerationConfig]:
        """Example pre-generation hook."""
        # Add a prefix to all prompts
        modified_prompt = f"[Plugin Enhanced] {prompt}"
        return modified_prompt, config
    
    async def post_generate_hook(self, prompt: str, response: str, config: GenerationConfig) -> str:
        """Example post-generation hook."""
        # Add a suffix to all responses
        return f"{response} [Enhanced by Example Plugin]"