"""Configuration loader for the LLM Framework."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from ..core import LLMConfig
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages framework configuration from various sources."""
    
    @staticmethod
    def load_config(config_path: Optional[Path] = None,
                   config_dict: Optional[Dict[str, Any]] = None,
                   env_prefix: str = "LLM_FRAMEWORK_") -> LLMConfig:
        """Load configuration from file, dictionary, and environment variables.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (overrides file)
            env_prefix: Prefix for environment variables
            
        Returns:
            LLMConfig instance
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        # Start with default configuration
        config_data = {}
        
        # Load from file if provided
        if config_path:
            config_data.update(ConfigLoader._load_from_file(config_path))
        else:
            # Try to find default config file
            default_paths = [
                Path.cwd() / "framework_config.yaml",
                Path.cwd() / "config" / "framework_config.yaml",
                Path(__file__).parent / "framework_config.yaml",
                Path.home() / ".llm_framework" / "config.yaml",
            ]
            
            for path in default_paths:
                if path.exists():
                    config_data.update(ConfigLoader._load_from_file(path))
                    logger.info(f"Loaded configuration from: {path}")
                    break
        
        # Override with dictionary if provided
        if config_dict:
            config_data.update(config_dict)
        
        # Override with environment variables
        env_config = ConfigLoader._load_from_env(env_prefix)
        if env_config:
            config_data.update(env_config)
        
        # Flatten nested configuration for LLMConfig
        flattened_config = ConfigLoader._flatten_config(config_data)
        
        try:
            return LLMConfig(**flattened_config)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
    
    @staticmethod
    def _load_from_file(config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If file loading fails
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not isinstance(config_data, dict):
                raise ConfigurationError(f"Configuration file must contain a dictionary: {config_path}")
            
            return config_data
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    @staticmethod
    def _load_from_env(prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Define environment variable mappings
        env_mappings = {
            f"{prefix}MODELS_DIR": "models_dir",
            f"{prefix}MAX_CONCURRENT_REQUESTS": ("max_concurrent_requests", int),
            f"{prefix}DEFAULT_TIMEOUT": ("default_timeout", float),
            f"{prefix}DEFAULT_MAX_TOKENS": ("default_max_tokens", int),
            f"{prefix}DEFAULT_TEMPERATURE": ("default_temperature", float),
            f"{prefix}DEFAULT_TOP_P": ("default_top_p", float),
            f"{prefix}ENABLE_PLUGINS": ("enable_plugins", bool),
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}LOG_FILE": "log_file",
            f"{prefix}ENABLE_FALLBACKS": ("enable_fallbacks", bool),
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_key, tuple):
                    key, type_converter = config_key
                    try:
                        if type_converter == bool:
                            config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            config[key] = type_converter(env_value)
                    except ValueError as e:
                        logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")
                else:
                    config[config_key] = env_value
        
        # Handle list-type environment variables
        fallback_models_env = os.getenv(f"{prefix}FALLBACK_MODELS")
        if fallback_models_env:
            config["fallback_models"] = [model.strip() for model in fallback_models_env.split(",")]
        
        plugin_dirs_env = os.getenv(f"{prefix}PLUGIN_DIRS")
        if plugin_dirs_env:
            config["plugin_dirs"] = [dir.strip() for dir in plugin_dirs_env.split(",")]
        
        return config
    
    @staticmethod
    def _flatten_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration dictionary for LLMConfig.
        
        Args:
            config_data: Nested configuration dictionary
            
        Returns:
            Flattened configuration dictionary
        """
        flattened = {}
        
        # Direct mappings from nested structure to LLMConfig fields
        mappings = {
            # Models section
            ("models", "models_dir"): "models_dir",
            ("models", "auto_discover_models"): "auto_discover_models",
            
            # Performance section
            ("performance", "max_concurrent_requests"): "max_concurrent_requests",
            ("performance", "default_timeout"): "default_timeout",
            
            # Generation section
            ("generation", "default_max_tokens"): "default_max_tokens",
            ("generation", "default_temperature"): "default_temperature",
            ("generation", "default_top_p"): "default_top_p",
            
            # Plugins section
            ("plugins", "enable_plugins"): "enable_plugins",
            ("plugins", "plugin_dirs"): "plugin_dirs",
            ("plugins", "plugin_configs"): "plugin_configs",
            
            # Logging section
            ("logging", "level"): "log_level",
            ("logging", "file"): "log_file",
            
            # Fallbacks section
            ("fallbacks", "enable_fallbacks"): "enable_fallbacks",
            ("fallbacks", "fallback_models"): "fallback_models",
        }
        
        # Apply mappings
        for (section, key), target_key in mappings.items():
            if section in config_data and key in config_data[section]:
                flattened[target_key] = config_data[section][key]
        
        # Handle API credentials (extract from environment)
        if "api_credentials" in config_data:
            creds = config_data["api_credentials"]
            
            # OpenAI
            if "openai" in creds and "api_key_env" in creds["openai"]:
                flattened["openai_api_key"] = os.getenv(creds["openai"]["api_key_env"])
            
            # Anthropic
            if "anthropic" in creds and "api_key_env" in creds["anthropic"]:
                flattened["anthropic_api_key"] = os.getenv(creds["anthropic"]["api_key_env"])
            
            # Google
            if "google" in creds and "api_key_env" in creds["google"]:
                flattened["google_api_key"] = os.getenv(creds["google"]["api_key_env"])
        
        # Add any top-level keys that don't need mapping
        for key, value in config_data.items():
            if not isinstance(value, dict) and key not in flattened:
                flattened[key] = value
        
        return flattened
    
    @staticmethod
    def save_config(config: LLMConfig, config_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: LLMConfig instance to save
            config_path: Path to save configuration file
            
        Raises:
            ConfigurationError: If saving fails
        """
        try:
            # Convert LLMConfig to nested structure
            config_dict = ConfigLoader._unflatten_config(config.to_dict())
            
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}")
    
    @staticmethod
    def _unflatten_config(flattened: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flattened config back to nested structure.
        
        Args:
            flattened: Flattened configuration dictionary
            
        Returns:
            Nested configuration dictionary
        """
        config = {
            "models": {},
            "performance": {},
            "generation": {},
            "plugins": {},
            "logging": {},
            "fallbacks": {},
            "api_credentials": {
                "openai": {"api_key_env": "OPENAI_API_KEY"},
                "anthropic": {"api_key_env": "ANTHROPIC_API_KEY"},
                "google": {"api_key_env": "GOOGLE_API_KEY"},
            }
        }
        
        # Reverse mappings
        reverse_mappings = {
            "models_dir": ("models", "models_dir"),
            "auto_discover_models": ("models", "auto_discover_models"),
            "max_concurrent_requests": ("performance", "max_concurrent_requests"),
            "default_timeout": ("performance", "default_timeout"),
            "default_max_tokens": ("generation", "default_max_tokens"),
            "default_temperature": ("generation", "default_temperature"),
            "default_top_p": ("generation", "default_top_p"),
            "enable_plugins": ("plugins", "enable_plugins"),
            "plugin_dirs": ("plugins", "plugin_dirs"),
            "plugin_configs": ("plugins", "plugin_configs"),
            "log_level": ("logging", "level"),
            "log_file": ("logging", "file"),
            "enable_fallbacks": ("fallbacks", "enable_fallbacks"),
            "fallback_models": ("fallbacks", "fallback_models"),
        }
        
        # Apply reverse mappings
        for flat_key, (section, nested_key) in reverse_mappings.items():
            if flat_key in flattened:
                config[section][nested_key] = flattened[flat_key]
        
        return config


def load_framework_config(config_path: Optional[str] = None, **kwargs: Any) -> LLMConfig:
    """Convenience function to load framework configuration.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        LLMConfig instance
    """
    path = Path(config_path) if config_path else None
    return ConfigLoader.load_config(config_path=path, config_dict=kwargs)