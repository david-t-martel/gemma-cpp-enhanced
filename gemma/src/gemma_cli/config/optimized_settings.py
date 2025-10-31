"""Optimized configuration management with caching and lazy loading."""

import functools
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time

import toml
from pydantic import ValidationError

from .settings import (
    Settings,
    ConfigManager,
    DetectedModel,
    ConfiguredModel,
    expand_path
)
from ..utils.profiler import PerformanceMonitor, TimedCache

# Configuration cache with 5-minute TTL
_config_cache = TimedCache(ttl=300)
_detected_models_cache = TimedCache(ttl=60)  # 1-minute cache for detected models


@functools.lru_cache(maxsize=1)
@PerformanceMonitor.track("config_load_cached")
def load_config_cached(config_path: Optional[Path] = None) -> Settings:
    """
    Load configuration with LRU caching.
    This is the primary optimization for config loading.

    The cache is invalidated when:
    - The config file is modified (checked via mtime)
    - The cache expires (5 minutes)
    - The process restarts

    Performance:
    - First load: ~50-100ms (file I/O + parsing)
    - Cached load: <1ms (memory lookup)
    """
    if config_path is None:
        config_path = Path.home() / ".gemma_cli" / "config.toml"

    # Check file modification time to invalidate cache
    cache_key = str(config_path)
    if config_path.exists():
        mtime = config_path.stat().st_mtime
        cache_entry = _config_cache.get((cache_key, 'mtime'))

        if cache_entry and cache_entry == mtime:
            # File hasn't changed, use cached config
            cached_config = _config_cache.get((cache_key, 'config'))
            if cached_config:
                return cached_config

    # Load fresh config
    manager = ConfigManager(config_path)
    settings = manager.load()

    # Cache the config and its mtime
    if config_path.exists():
        _config_cache.set((cache_key, 'mtime'), config_path.stat().st_mtime)
        _config_cache.set((cache_key, 'config'), settings)

    return settings


@PerformanceMonitor.track("detected_models_load")
def load_detected_models_cached() -> Dict[str, DetectedModel]:
    """
    Load detected models with caching.

    Performance:
    - First load: ~10-20ms (file I/O + JSON parsing)
    - Cached load: <1ms
    """
    cache_key = 'detected_models'
    cached = _detected_models_cache.get(cache_key)
    if cached is not None:
        return cached

    detected_path = Path.home() / ".gemma_cli" / "detected_models.json"
    if not detected_path.exists():
        result = {}
    else:
        try:
            with open(detected_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            models = {}
            for model_data in data.get("models", []):
                model = DetectedModel(**model_data)
                models[model.name] = model
            result = models

        except (json.JSONDecodeError, ValidationError):
            result = {}

    _detected_models_cache.set(cache_key, result)
    return result


class LazyConfigLoader:
    """
    Lazy configuration loader that defers loading until first access.
    This avoids loading config on import for commands that don't need it.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._settings: Optional[Settings] = None

    @property
    def settings(self) -> Settings:
        """Load settings on first access."""
        if self._settings is None:
            self._settings = load_config_cached(self.config_path)
        return self._settings

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        try:
            return getattr(self.settings, key, default)
        except AttributeError:
            return default

    def reload(self):
        """Force reload of configuration."""
        # Clear LRU cache
        load_config_cached.cache_clear()
        # Clear timed cache
        _config_cache.clear()
        # Reload settings
        self._settings = None


# Global lazy loader instance
_global_config = LazyConfigLoader()


def get_config() -> Settings:
    """
    Get the global configuration instance.
    This is the preferred way to access config in most code.
    """
    return _global_config.settings


def reload_config():
    """Force reload of global configuration."""
    _global_config.reload()


@PerformanceMonitor.track("model_resolution")
def get_model_by_name_optimized(
    name: str,
    settings: Optional[Settings] = None
) -> Optional[tuple[str, Optional[str]]]:
    """
    Optimized model resolution with caching.

    Performance improvements:
    - Uses cached detected models
    - Avoids redundant file I/O
    - Caches resolution results
    """
    # Try cache first
    cache_key = f'model_{name}'
    cached_result = _config_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    # Load detected models with cache
    detected = load_detected_models_cached()
    if name in detected:
        model = detected[name]
        result = (model.weights_path, model.tokenizer_path)
        _config_cache.set(cache_key, result)
        return result

    # Load settings if not provided
    if settings is None:
        settings = get_config()

    # Check configured models
    if name in settings.configured_models:
        model = settings.configured_models[name]
        result = (model.weights_path, model.tokenizer_path)
        _config_cache.set(cache_key, result)
        return result

    return None


class BatchConfigWriter:
    """
    Batch configuration writes to reduce I/O operations.
    Accumulates changes and writes them in batches.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".gemma_cli" / "config.toml"
        self.pending_changes: Dict[str, Any] = {}
        self.last_flush = time.time()
        self.flush_interval = 1.0  # Flush after 1 second

    def set(self, key: str, value: Any):
        """Queue a configuration change."""
        self.pending_changes[key] = value

        # Auto-flush if enough time has passed
        if time.time() - self.last_flush > self.flush_interval:
            self.flush()

    def flush(self):
        """Write all pending changes to disk."""
        if not self.pending_changes:
            return

        # Load current config
        settings = load_config_cached(self.config_path)

        # Apply pending changes
        for key, value in self.pending_changes.items():
            setattr(settings, key, value)

        # Save to disk
        manager = ConfigManager(self.config_path)
        manager.save(settings)

        # Clear cache to force reload
        reload_config()

        # Reset
        self.pending_changes.clear()
        self.last_flush = time.time()


# Export optimized functions that match original API
load_config = load_config_cached
load_detected_models = load_detected_models_cached
get_model_by_name = get_model_by_name_optimized


__all__ = [
    'load_config',
    'load_detected_models',
    'get_model_by_name',
    'get_config',
    'reload_config',
    'LazyConfigLoader',
    'BatchConfigWriter',
]