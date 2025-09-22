"""Logging decorators for function call tracking and performance monitoring.

This module provides decorators to automatically log function calls,
performance metrics, and errors across the application.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from .logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def log_function_calls(
    logger: Any | None = None,
    log_args: bool = True,
    log_return: bool = True,
    log_level: str = "DEBUG",
    exclude_args: list[str] | None = None,
    mask_args: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to log function calls with arguments and return values.

    Args:
        logger: Logger instance to use (defaults to function's module logger)
        log_args: Whether to log function arguments
        log_return: Whether to log return value
        log_level: Log level to use (DEBUG, INFO, etc.)
        exclude_args: Arguments to exclude from logging
        mask_args: Arguments to mask in logs (show only type)

    Returns:
        Decorated function
    """
    exclude_args = exclude_args or []
    mask_args = mask_args or []

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger for the function's module
            func_logger = logger or get_logger(func.__module__)

            func_name = f"{func.__module__}.{func.__name__}"

            # Prepare arguments for logging
            log_data = {"function": func_name}

            if log_args:
                # Process positional arguments
                if args:
                    arg_names = func.__code__.co_varnames[: len(args)]
                    logged_args = {}
                    for _i, (arg_name, arg_value) in enumerate(zip(arg_names, args, strict=False)):
                        if arg_name in exclude_args:
                            continue
                        elif arg_name in mask_args:
                            logged_args[arg_name] = f"<{type(arg_value).__name__}>"
                        # Limit string length to avoid log spam
                        elif isinstance(arg_value, str) and len(arg_value) > 200:
                            logged_args[arg_name] = arg_value[:200] + "..."
                        else:
                            logged_args[arg_name] = arg_value
                    if logged_args:
                        log_data["function_args"] = logged_args

                # Process keyword arguments
                if kwargs:
                    logged_kwargs = {}
                    for key, value in kwargs.items():
                        if key in exclude_args:
                            continue
                        elif key in mask_args:
                            logged_kwargs[key] = f"<{type(value).__name__}>"
                        elif isinstance(value, str) and len(value) > 200:
                            logged_kwargs[key] = value[:200] + "..."
                        else:
                            logged_kwargs[key] = value
                    if logged_kwargs:
                        log_data["kwargs"] = logged_kwargs

            # Log function entry
            getattr(func_logger, log_level.lower())(f"Entering {func_name}", extra=log_data)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log successful exit
                exit_data = {"function": func_name, "status": "success"}
                if log_return and result is not None:
                    # Limit return value logging
                    if isinstance(result, str) and len(result) > 200:
                        exit_data["return_value"] = result[:200] + "..."
                    elif hasattr(result, "__dict__") and len(str(result)) > 500:
                        exit_data["return_value"] = f"<{type(result).__name__}>"
                    else:
                        exit_data["return_value"] = result

                getattr(func_logger, log_level.lower())(f"Exiting {func_name}", extra=exit_data)

                return result

            except Exception as e:
                # Log exception
                func_logger.error(
                    f"Exception in {func_name}: {e}",
                    extra={"function": func_name, "error": str(e)},
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_performance(
    logger: Any | None = None,
    log_level: str = "INFO",
    threshold_ms: float | None = None,
    include_memory: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function performance metrics.

    Args:
        logger: Logger instance to use
        log_level: Log level for performance metrics
        threshold_ms: Only log if execution time exceeds threshold (milliseconds)
        include_memory: Include memory usage information

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"

            # Memory tracking setup
            memory_before = None
            if include_memory:
                try:
                    import psutil

                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass

            # Start timing
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # Calculate metrics
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                # Only log if above threshold
                if threshold_ms is None or duration_ms >= threshold_ms:
                    perf_data = {
                        "function": func_name,
                        "duration_ms": round(duration_ms, 2),
                        "success": success,
                    }

                    if error:
                        perf_data["error"] = error

                    if include_memory and memory_before is not None:
                        try:
                            import psutil

                            process = psutil.Process()
                            memory_after = process.memory_info().rss / 1024 / 1024  # MB
                            perf_data["memory_before_mb"] = round(memory_before, 2)
                            perf_data["memory_after_mb"] = round(memory_after, 2)
                            perf_data["memory_delta_mb"] = round(memory_after - memory_before, 2)
                        except (ImportError, psutil.NoSuchProcess):
                            pass

                    getattr(func_logger, log_level.lower())(
                        f"Performance: {func_name} took {duration_ms:.2f}ms", extra=perf_data
                    )

            return result

        return wrapper

    return decorator


def log_errors(
    logger: Any | None = None,
    log_level: str = "ERROR",
    reraise: bool = True,
    include_args: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function errors with context.

    Args:
        logger: Logger instance to use
        log_level: Log level for errors
        reraise: Whether to reraise the exception
        include_args: Include function arguments in error logs

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_data = {
                    "function": func_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

                if include_args:
                    if args:
                        error_data["function_args"] = [
                            f"<{type(arg).__name__}>" if len(str(arg)) > 100 else arg
                            for arg in args
                        ]
                    if kwargs:
                        error_data["kwargs"] = {
                            k: f"<{type(v).__name__}>" if len(str(v)) > 100 else v
                            for k, v in kwargs.items()
                        }

                getattr(func_logger, log_level.lower())(
                    f"Error in {func_name}: {e}", extra=error_data, exc_info=True
                )

                if reraise:
                    raise
                return None

        return wrapper

    return decorator


def log_async_function_calls(
    logger: Any | None = None,
    log_args: bool = True,
    log_return: bool = True,
    log_level: str = "DEBUG",
    exclude_args: list[str] | None = None,
    mask_args: list[str] | None = None,
) -> Callable[[F], F]:
    """Async version of log_function_calls decorator.

    Args:
        logger: Logger instance to use
        log_args: Whether to log function arguments
        log_return: Whether to log return value
        log_level: Log level to use
        exclude_args: Arguments to exclude from logging
        mask_args: Arguments to mask in logs

    Returns:
        Decorated async function
    """
    exclude_args = exclude_args or []
    mask_args = mask_args or []

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            func_name = f"{func.__module__}.{func.__name__}"

            # Similar logging logic as sync version
            log_data = {"function": func_name, "async": True}

            if log_args:
                # Process arguments (same as sync version)
                if args:
                    arg_names = func.__code__.co_varnames[: len(args)]
                    logged_args = {}
                    for arg_name, arg_value in zip(arg_names, args, strict=False):
                        if arg_name in exclude_args:
                            continue
                        elif arg_name in mask_args:
                            logged_args[arg_name] = f"<{type(arg_value).__name__}>"
                        elif isinstance(arg_value, str) and len(arg_value) > 200:
                            logged_args[arg_name] = arg_value[:200] + "..."
                        else:
                            logged_args[arg_name] = arg_value
                    if logged_args:
                        log_data["function_args"] = logged_args

                if kwargs:
                    logged_kwargs = {}
                    for key, value in kwargs.items():
                        if key in exclude_args:
                            continue
                        elif key in mask_args:
                            logged_kwargs[key] = f"<{type(value).__name__}>"
                        elif isinstance(value, str) and len(value) > 200:
                            logged_kwargs[key] = value[:200] + "..."
                        else:
                            logged_kwargs[key] = value
                    if logged_kwargs:
                        log_data["kwargs"] = logged_kwargs

            getattr(func_logger, log_level.lower())(f"Entering async {func_name}", extra=log_data)

            try:
                result = await func(*args, **kwargs)

                exit_data = {"function": func_name, "async": True, "status": "success"}
                if log_return and result is not None:
                    if isinstance(result, str) and len(result) > 200:
                        exit_data["return_value"] = result[:200] + "..."
                    elif hasattr(result, "__dict__") and len(str(result)) > 500:
                        exit_data["return_value"] = f"<{type(result).__name__}>"
                    else:
                        exit_data["return_value"] = result

                getattr(func_logger, log_level.lower())(
                    f"Exiting async {func_name}", extra=exit_data
                )

                return result

            except Exception as e:
                func_logger.error(
                    f"Exception in async {func_name}: {e}",
                    extra={"function": func_name, "async": True, "error": str(e)},
                    exc_info=True,
                )
                raise

        return async_wrapper

    return decorator


# Convenience aliases
log_calls = log_function_calls
log_perf = log_performance
log_async_calls = log_async_function_calls
