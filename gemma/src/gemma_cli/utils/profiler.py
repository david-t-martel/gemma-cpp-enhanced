"""Performance profiling utilities for Gemma CLI."""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Lightweight performance monitoring for production use."""

    metrics: Dict[str, List[float]] = {}
    enabled: bool = True

    @classmethod
    def track(cls, name: str):
        """
        Decorator to track function performance.

        Usage:
            @PerformanceMonitor.track("config_load")
            def load_config():
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not cls.enabled:
                    return await func(*args, **kwargs)

                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    cls.record(name, duration)
                    if duration > 1.0:  # Log slow operations
                        logger.warning(f"Slow operation: {name} took {duration:.2f}s")
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start
                    cls.record(f"{name}_error", duration)
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not cls.enabled:
                    return func(*args, **kwargs)

                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    cls.record(name, duration)
                    if duration > 1.0:  # Log slow operations
                        logger.warning(f"Slow operation: {name} took {duration:.2f}s")
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start
                    cls.record(f"{name}_error", duration)
                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    @classmethod
    def record(cls, name: str, duration: float):
        """Record a performance metric."""
        if not cls.enabled:
            return

        if name not in cls.metrics:
            cls.metrics[name] = []
        cls.metrics[name].append(duration)

    @classmethod
    def report(cls) -> Dict[str, Any]:
        """Generate performance report with statistics."""
        report = {}
        for name, durations in cls.metrics.items():
            if durations:
                report[name] = {
                    'count': len(durations),
                    'mean_ms': sum(durations) * 1000 / len(durations),
                    'min_ms': min(durations) * 1000,
                    'max_ms': max(durations) * 1000,
                    'total_ms': sum(durations) * 1000
                }
        return report

    @classmethod
    def reset(cls):
        """Reset all metrics."""
        cls.metrics.clear()

    @classmethod
    def enable(cls):
        """Enable performance monitoring."""
        cls.enabled = True

    @classmethod
    def disable(cls):
        """Disable performance monitoring."""
        cls.enabled = False


class LazyImport:
    """
    Lazy import wrapper to defer module loading until first access.

    Usage:
        # Instead of: from heavy_module import HeavyClass
        HeavyClass = LazyImport('heavy_module', 'HeavyClass')
    """

    def __init__(self, module_name: str, attr_name: Optional[str] = None):
        self.module_name = module_name
        self.attr_name = attr_name
        self._module = None
        self._attr = None

    def _load(self):
        """Load the module and attribute on first access."""
        if self._module is None:
            start = time.perf_counter()
            self._module = __import__(self.module_name, fromlist=[''])
            duration = time.perf_counter() - start

            if duration > 0.1:  # Log slow imports
                logger.debug(f"Lazy loaded {self.module_name} in {duration*1000:.1f}ms")

            if self.attr_name:
                self._attr = getattr(self._module, self.attr_name)

    def __getattr__(self, name):
        """Proxy attribute access to the loaded module/attribute."""
        self._load()
        if self._attr:
            return getattr(self._attr, name)
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        """Make the lazy import callable."""
        self._load()
        if self._attr:
            return self._attr(*args, **kwargs)
        return self._module(*args, **kwargs)


def lazy_property(func: Callable) -> property:
    """
    Decorator for lazy property evaluation.
    Property is computed once on first access and cached.

    Usage:
        class MyClass:
            @lazy_property
            def expensive_property(self):
                # This is only computed once
                return expensive_computation()
    """
    attr_name = f'_lazy_{func.__name__}'

    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return property(wrapper)


class TimedCache:
    """
    Time-based cache with TTL support.

    Usage:
        cache = TimedCache(ttl=300)  # 5-minute cache

        @cache.cached
        def expensive_function(x):
            return x ** 2
    """

    def __init__(self, ttl: float = 300):
        self.ttl = ttl
        self.cache: Dict[Any, tuple[Any, float]] = {}

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: Any, value: Any):
        """Set value in cache with current timestamp."""
        self.cache[key] = (value, time.time())

    def cached(self, func: Callable) -> Callable:
        """Decorator to cache function results."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key = (func.__name__, args, frozenset(kwargs.items()) if kwargs else ())

            # Check cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value

            # Compute and cache result
            result = func(*args, **kwargs)
            self.set(key, result)
            return result

        return wrapper

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()