"""Middleware components for the FastAPI server.

This module provides middleware for CORS, authentication, rate limiting,
logging, and metrics collection with production-ready features.
"""

import asyncio
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import prometheus_client
import psutil
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..shared.config.settings import Settings
from ..shared.logging import get_logger
from .auth import add_jwt_authentication
from .security_headers import add_security_headers

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "endpoint"]
)

ACTIVE_CONNECTIONS = Gauge("active_connections_total", "Number of active connections")

MODEL_INFERENCE_COUNT = Counter(
    "model_inference_total", "Total model inferences", ["model_name", "endpoint"]
)

MODEL_INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name", "endpoint"],
)

MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes", ["type"])

CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")


class RateLimiter:
    """Simple in-memory rate limiter with sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.blocked_ips: dict[str, float] = {}
        self.cleanup_task = None

    def _get_client_key(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get real IP from headers (for proxy setups)
        client_ip = request.headers.get("X-Forwarded-For")
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()
        else:
            client_ip = request.headers.get("X-Real-IP")

        if not client_ip and request.client:
            client_ip = request.client.host

        # Fallback to a default if no IP is found
        return client_ip or "unknown"

    def _cleanup_old_requests(self):
        """Clean up old request records."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        for client_key in list(self.requests.keys()):
            # Remove old requests
            self.requests[client_key] = [
                req_time for req_time in self.requests[client_key] if req_time > cutoff_time
            ]

            # Remove empty entries
            if not self.requests[client_key]:
                del self.requests[client_key]

        # Clean up expired blocks
        for client_key in list(self.blocked_ips.keys()):
            if self.blocked_ips[client_key] < current_time:
                del self.blocked_ips[client_key]

    async def is_allowed(self, request: Request) -> tuple[bool, str | None]:
        """Check if request is allowed based on rate limits.

        Returns:
            (allowed, error_message)
        """
        client_key = self._get_client_key(request)
        current_time = time.time()

        # Check if client is temporarily blocked
        if client_key in self.blocked_ips:
            if self.blocked_ips[client_key] > current_time:
                return False, "Too many requests - temporarily blocked"
            else:
                del self.blocked_ips[client_key]

        # Clean up old requests periodically
        if len(self.requests) % 100 == 0:  # Every 100 requests
            self._cleanup_old_requests()

        # Add current request
        self.requests[client_key].append(current_time)

        # Count requests in current window
        cutoff_time = current_time - self.window_seconds
        recent_requests = [
            req_time for req_time in self.requests[client_key] if req_time > cutoff_time
        ]

        if len(recent_requests) > self.max_requests:
            # Block client for double the window time
            self.blocked_ips[client_key] = current_time + (self.window_seconds * 2)
            return (
                False,
                f"Rate limit exceeded: {len(recent_requests)} requests in {self.window_seconds}s",
            )

        return True, None


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware with API key support."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.security = HTTPBearer(auto_error=False)
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/health/",
            "/metrics",
        }

    async def dispatch(self, request: Request, call_next):
        """Process authentication for incoming requests."""
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints or request.url.path.startswith("/health/"):
            return await call_next(request)

        # Skip if authentication is not required
        if not self.settings.security.api_key_required:
            return await call_next(request)

        # Extract authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "Missing authorization header",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        scheme, credentials = get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid authentication scheme",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate API key (in production, this should check against a secure store)
        if not self._validate_api_key(credentials):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": {"type": "authentication_error", "message": "Invalid API key"}},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Add user info to request state
        request.state.user = {"api_key": credentials}

        return await call_next(request)

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        # Import here to avoid circular imports
        import os

        from ..domain.validators import APIKeyValidator

        # Get valid API keys from environment or settings
        valid_keys = []

        # Try to load keys from environment variables
        env_keys = os.getenv("GEMMA_API_KEYS", "").split(",")
        valid_keys.extend([k.strip() for k in env_keys if k.strip()])

        # Add keys from settings if available
        if hasattr(self.settings.security, "api_keys"):
            valid_keys.extend(self.settings.security.api_keys)

        # If no keys configured, reject all requests in production
        if not valid_keys:
            if self.settings.is_production():
                logger.error("No API keys configured in production mode")
                return False
            else:
                # In development, generate and log a temporary key
                from ..domain.validators import APIKeyValidator

                validator = APIKeyValidator()
                temp_key = validator.generate_key()
                logger.warning(f"No API keys configured. Temporary dev key: {temp_key}")
                valid_keys = [temp_key]

        # Validate the provided key
        validator = APIKeyValidator(valid_keys, use_hash=True)
        is_valid, error_msg = validator.validate(api_key)

        if not is_valid:
            logger.warning(f"API key validation failed: {error_msg}")

        return is_valid


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with configurable limits."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.rate_limiter = RateLimiter(
            max_requests=settings.security.rate_limit_per_minute, window_seconds=60
        )

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to incoming requests."""
        # Skip rate limiting for health checks and metrics
        if request.url.path.startswith(("/health", "/metrics")):
            return await call_next(request)

        # Check rate limit
        allowed, error_msg = await self.rate_limiter.is_allowed(request)
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {request.client.host if request.client else 'unknown'}: {error_msg}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "type": "rate_limit_error",
                        "message": error_msg,
                    }
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.settings.security.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + 60),
                },
            )

        response = await call_next(request)

        # Add rate limiting headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(self.settings.security.rate_limit_per_minute)
        # In a real implementation, calculate remaining requests
        response.headers["X-RateLimit-Remaining"] = str(
            self.settings.security.rate_limit_per_minute - 1
        )
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = get_logger("server.requests")

    async def dispatch(self, request: Request, call_next):
        """Log requests and responses."""
        start_time = time.time()
        client_ip = self._get_client_ip(request)

        # Log request
        self.logger.info(
            f"{request.method} {request.url.path} - {client_ip} - "
            f"User-Agent: {request.headers.get('user-agent', 'Unknown')}"
        )

        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            self.logger.info(
                f"{request.method} {request.url.path} - {response.status_code} - "
                f"{process_time:.3f}s - {client_ip}"
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(
                f"{request.method} {request.url.path} - ERROR: {e!s} - "
                f"{process_time:.3f}s - {client_ip}"
            )
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Try to get real IP from headers (for proxy setups)
        client_ip = request.headers.get("X-Forwarded-For")
        if client_ip:
            return client_ip.split(",")[0].strip()

        client_ip = request.headers.get("X-Real-IP")
        if client_ip:
            return client_ip

        return request.client.host if request.client else "unknown"


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Input validation middleware for request sanitization."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        from ..domain.validators import PromptValidator
        from ..domain.validators import RequestValidator

        self.prompt_validator = PromptValidator(
            max_length=settings.security.max_prompt_length,
            block_sensitive=settings.security.block_sensitive_patterns,
        )
        self.request_validator = RequestValidator(max_size_mb=settings.security.max_request_size_mb)

    async def dispatch(self, request: Request, call_next):
        """Validate and sanitize incoming requests."""
        # Skip validation for non-API endpoints
        if request.url.path in ["/", "/docs", "/redoc", "/openapi.json", "/health", "/metrics"]:
            return await call_next(request)

        # Validate Content-Type
        content_type = request.headers.get("content-type", "")
        if request.method in ["POST", "PUT", "PATCH"]:
            is_valid, error_msg = self.request_validator.validate_content_type(content_type)
            if not is_valid:
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={"error": {"type": "validation_error", "message": error_msg}},
                )

        # Validate Content-Length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                is_valid, error_msg = self.request_validator.validate_size(int(content_length))
                if not is_valid:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"error": {"type": "validation_error", "message": error_msg}},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": {
                            "type": "validation_error",
                            "message": "Invalid Content-Length header",
                        }
                    },
                )

        # Validate headers
        is_valid, error_msg = self.request_validator.validate_headers(dict(request.headers))
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": {"type": "validation_error", "message": error_msg}},
            )

        # For chat endpoints, validate prompt content
        if request.url.path.startswith("/v1/chat") and request.method == "POST":
            try:
                # Store original body for later use
                body = await request.body()
                import json

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "error": {
                                "type": "validation_error",
                                "message": "Invalid JSON in request body",
                            }
                        },
                    )

                # Extract and validate prompt/messages
                prompt = None
                if "messages" in data:
                    # OpenAI format
                    messages = data.get("messages", [])
                    if messages:
                        # Combine all user messages for validation
                        user_messages = [
                            m.get("content", "") for m in messages if m.get("role") == "user"
                        ]
                        prompt = " ".join(user_messages)
                elif "prompt" in data:
                    # Direct prompt format
                    prompt = data.get("prompt", "")

                if prompt:
                    validation_result = self.prompt_validator.validate(prompt)

                    if not validation_result.is_valid:
                        logger.warning(
                            f"Prompt validation failed: threat_level={validation_result.threat_level}, "
                            f"patterns={validation_result.detected_patterns}"
                        )
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={
                                "error": {
                                    "type": "security_error",
                                    "message": "Input validation failed",
                                    "threat_level": validation_result.threat_level,
                                    "risk_score": validation_result.risk_score,
                                }
                            },
                        )

                    # Update request with sanitized input if modifications were made
                    if validation_result.modifications:
                        if "messages" in data:
                            # Update the last user message with sanitized content
                            for i in range(len(data["messages"]) - 1, -1, -1):
                                if data["messages"][i].get("role") == "user":
                                    data["messages"][i]["content"] = (
                                        validation_result.sanitized_input
                                    )
                                    break
                        elif "prompt" in data:
                            data["prompt"] = validation_result.sanitized_input

                        # Create new request with sanitized data
                        from starlette.datastructures import Headers
                        from starlette.requests import \
                            Request as StarletteRequest

                        # Store sanitized body for the next handler
                        request._body = json.dumps(data).encode()

                # Store validation result in request state
                request.state.validation_result = validation_result if prompt else None

            except Exception as e:
                logger.error(f"Error during input validation: {e}")
                # Don't block on validation errors, but log them

        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection middleware."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        # Start background task for system metrics
        asyncio.create_task(self._collect_system_metrics())

    async def dispatch(self, request: Request, call_next):
        """Collect metrics for requests."""
        start_time = time.time()

        # Increment active connections
        ACTIVE_CONNECTIONS.inc()

        try:
            response = await call_next(request)

            # Record request metrics
            duration = time.time() - start_time
            endpoint = self._normalize_endpoint(request.url.path)

            REQUEST_COUNT.labels(
                method=request.method, endpoint=endpoint, status_code=response.status_code
            ).inc()

            REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(duration)

            return response

        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            endpoint = self._normalize_endpoint(request.url.path)

            REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status_code=500).inc()

            raise

        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint paths for metrics."""
        # Replace dynamic segments with placeholders to avoid high cardinality
        if path.startswith("/v1/chat"):
            return "/v1/chat/*"
        elif path.startswith("/v1/completions"):
            return "/v1/completions"
        elif path.startswith("/health"):
            return "/health/*"
        elif path.startswith("/models"):
            return "/models/*"
        else:
            return path

    async def _collect_system_metrics(self):
        """Collect system metrics in the background."""
        while True:
            try:
                # Memory metrics
                memory_info = psutil.virtual_memory()
                MEMORY_USAGE.labels(type="total").set(memory_info.total)
                MEMORY_USAGE.labels(type="used").set(memory_info.used)
                MEMORY_USAGE.labels(type="available").set(memory_info.available)

                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                CPU_USAGE.set(cpu_percent)

                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error


def add_cors_middleware(app: FastAPI, settings: Settings) -> None:
    """Add CORS middleware to the FastAPI app.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    # TODO: SECURITY CRITICAL - Permissive CORS configuration. allow_headers=["*"] is too permissive. Specify exact headers needed. allow_credentials=True with origins is dangerous if not properly validated
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-RateLimit-*"],
    )


def add_trusted_host_middleware(app: FastAPI, settings: Settings) -> None:
    """Add trusted host middleware for security.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    if settings.is_production():
        # In production, restrict to specific hosts
        allowed_hosts = [settings.server.host, "localhost"]
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)


def add_auth_middleware(app: FastAPI, settings: Settings) -> None:
    """Add authentication middleware.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    app.add_middleware(AuthenticationMiddleware, settings=settings)


def add_rate_limiting_middleware(app: FastAPI, settings: Settings) -> None:
    """Add rate limiting middleware.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    app.add_middleware(RateLimitingMiddleware, settings=settings)


def add_logging_middleware(app: FastAPI, settings: Settings) -> None:
    """Add request logging middleware.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    app.add_middleware(LoggingMiddleware, settings=settings)


def add_metrics_middleware(app: FastAPI, settings: Settings) -> None:
    """Add metrics collection middleware.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    app.add_middleware(MetricsMiddleware, settings=settings)


def add_input_validation_middleware(app: FastAPI, settings: Settings) -> None:
    """Add input validation middleware.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    if settings.security.enable_request_validation:
        app.add_middleware(InputValidationMiddleware, settings=settings)


def add_all_middleware(app: FastAPI, settings: Settings) -> None:
    """Add all middleware to the FastAPI app in the correct order.

    Args:
        app: FastAPI application
        settings: Application settings
    """
    # Order is important - middleware is applied in reverse order
    add_metrics_middleware(app, settings)
    add_logging_middleware(app, settings)
    add_input_validation_middleware(app, settings)  # Add validation before rate limiting
    add_rate_limiting_middleware(app, settings)

    # Add authentication - use JWT if secret is configured, otherwise API key
    if settings.security.jwt_secret or settings.is_production():
        # JWT authentication (preferred for production)
        add_jwt_authentication(app, settings)
    else:
        # Simple API key authentication (development)
        add_auth_middleware(app, settings)

    add_trusted_host_middleware(app, settings)
    add_security_headers(app, production_mode=settings.is_production())  # Add security headers
    add_cors_middleware(app, settings)
