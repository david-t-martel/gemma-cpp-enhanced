"""JWT Authentication module for secure token-based authentication.

This module provides JWT token generation, validation, and middleware
for protecting API endpoints with stateless authentication.
"""

import hashlib
import secrets
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import Optional

import jwt
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from pydantic import Field

from ..shared.config.settings import Settings
from ..shared.logging import get_logger

logger = get_logger(__name__)


class TokenData(BaseModel):
    """JWT token payload data."""

    sub: str = Field(..., description="Subject (user identifier)")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at time")
    jti: str = Field(..., description="JWT ID for token revocation")
    scope: str = Field("api", description="Token scope")
    api_key_hash: str = Field(..., description="Hash of the API key used")


class TokenResponse(BaseModel):
    """Response model for token generation."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    issued_at: datetime


class JWTAuthenticator:
    """JWT authentication handler with secure token management."""

    def __init__(self, settings: Settings):
        """Initialize the JWT authenticator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.security = HTTPBearer(auto_error=False)

        # Initialize JWT secret
        self.jwt_secret = self._get_or_create_jwt_secret()
        self.algorithm = settings.security.jwt_algorithm or "HS256"
        self.expiry_hours = settings.security.jwt_expiry_hours or 24

        # Token revocation set (in production, use Redis or database)
        self.revoked_tokens: set[str] = set()

    def _get_or_create_jwt_secret(self) -> str:
        """Get JWT secret from settings or environment.

        Returns:
            JWT secret key

        Raises:
            ValueError: If no secret is configured in production
        """
        import os
        from pathlib import Path

        # Try to get from settings first
        if self.settings.security.jwt_secret:
            return self.settings.security.jwt_secret

        # Try to get from environment
        env_secret = os.getenv("GEMMA_JWT_SECRET")
        if env_secret and len(env_secret) >= 32:  # Ensure minimum length
            return env_secret

        # In development, try to load or create a persistent secret file
        if not self.settings.is_production():
            secret_file = Path.home() / ".gemma" / "jwt_secret.key"
            secret_file.parent.mkdir(parents=True, exist_ok=True)

            if secret_file.exists():
                # Load existing secret
                try:
                    with open(secret_file, "r", encoding="utf-8") as f:
                        saved_secret = f.read().strip()
                        if saved_secret and len(saved_secret) >= 64:
                            logger.info("Loaded JWT secret from persistent storage")
                            return saved_secret
                except Exception as e:
                    logger.warning(f"Failed to load JWT secret file: {e}")

            # Generate and save a new secret
            secret = secrets.token_urlsafe(64)
            try:
                with open(secret_file, "w", encoding="utf-8") as f:
                    f.write(secret)
                # Secure file permissions (Unix-like systems)
                if hasattr(os, "chmod"):
                    os.chmod(secret_file, 0o600)
                logger.warning(
                    f"Generated and saved JWT secret to {secret_file}. "
                    "For production, set GEMMA_JWT_SECRET environment variable."
                )
            except Exception as e:
                logger.error(f"Failed to save JWT secret: {e}")

            return secret

        # Require explicit configuration in production
        raise ValueError(
            "JWT secret must be configured in production. "
            "Set GEMMA_JWT_SECRET environment variable with at least 32 characters."
        )

    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage in tokens.

        Args:
            api_key: The API key to hash

        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def generate_token(self, api_key: str, scope: str = "api") -> TokenResponse:
        """Generate a JWT token for an API key.

        Args:
            api_key: The API key to generate token for
            scope: Token scope/permissions

        Returns:
            Token response with access token and metadata
        """
        # Validate API key first
        from ..domain.validators import APIKeyValidator

        validator = APIKeyValidator(self.settings.security.api_keys, use_hash=True)
        is_valid, error_msg = validator.validate(api_key)

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid API key: {error_msg}",
            )

        # Generate token ID for revocation tracking
        jti = secrets.token_urlsafe(32)

        # Calculate expiration
        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self.expiry_hours)

        # Create token payload
        payload = {
            "sub": self._hash_api_key(api_key)[:16],  # Shortened subject
            "exp": expires_at,
            "iat": now,
            "jti": jti,
            "scope": scope,
            "api_key_hash": self._hash_api_key(api_key),
        }

        # Generate token
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)

        logger.info(f"Generated JWT token with JTI: {jti}, expires at: {expires_at}")

        return TokenResponse(
            access_token=token,
            token_type="Bearer",
            expires_in=self.expiry_hours * 3600,
            issued_at=now,
        )

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode a JWT token.

        Args:
            token: The JWT token to verify

        Returns:
            Decoded token data

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.algorithm],
                options={"verify_exp": True},
            )

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self.revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            # Convert to TokenData
            token_data = TokenData(
                sub=payload["sub"],
                exp=datetime.fromtimestamp(payload["exp"], UTC),
                iat=datetime.fromtimestamp(payload["iat"], UTC),
                jti=payload["jti"],
                scope=payload.get("scope", "api"),
                api_key_hash=payload["api_key_hash"],
            )

            return token_data

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        except KeyError as e:
            logger.error(f"Missing token field: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Malformed token",
            )

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI.

        Args:
            jti: JWT ID to revoke
        """
        self.revoked_tokens.add(jti)
        logger.info(f"Revoked token with JTI: {jti}")

    async def authenticate_request(self, request: Request) -> TokenData:
        """Authenticate a request using JWT token.

        Args:
            request: The incoming request

        Returns:
            Token data if authenticated

        Raises:
            HTTPException: If authentication fails
        """
        # Extract token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse Bearer token
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization format",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = parts[1]

        # Verify token
        return self.verify_token(token)

    def create_api_key(self, length: int = 48) -> str:
        """Generate a new secure API key.

        Args:
            length: Length of the API key

        Returns:
            Generated API key
        """
        return f"sk-{secrets.token_urlsafe(length)}"


class JWTMiddleware:
    """JWT authentication middleware for FastAPI."""

    def __init__(self, authenticator: JWTAuthenticator, public_paths: set[str] | None = None):
        """Initialize JWT middleware.

        Args:
            authenticator: JWT authenticator instance
            public_paths: Set of paths that don't require authentication
        """
        self.authenticator = authenticator
        self.public_paths = public_paths or {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/health/",
            "/metrics",
            "/v1/auth/token",  # Token generation endpoint
        }

    async def __call__(self, request: Request, call_next):
        """Process authentication for incoming requests.

        Args:
            request: The incoming request
            call_next: Next middleware or handler

        Returns:
            Response from the next handler
        """
        # Skip authentication for public endpoints
        if request.url.path in self.public_paths or request.url.path.startswith("/health/"):
            return await call_next(request)

        try:
            # Authenticate request
            token_data = await self.authenticator.authenticate_request(request)

            # Add token data to request state
            request.state.token_data = token_data
            request.state.authenticated = True

            return await call_next(request)

        except HTTPException as e:
            # Return authentication error
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "type": "authentication_error",
                        "message": e.detail,
                    }
                },
                headers=e.headers or {},
            )


def add_jwt_authentication(app, settings: Settings) -> JWTAuthenticator:
    """Add JWT authentication to the FastAPI app.

    Args:
        app: FastAPI application
        settings: Application settings

    Returns:
        JWT authenticator instance
    """
    from starlette.middleware.base import BaseHTTPMiddleware

    # Create authenticator
    authenticator = JWTAuthenticator(settings)

    # Add middleware
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=JWTMiddleware(authenticator).__call__,
    )

    # Add token generation endpoint
    from fastapi import APIRouter

    auth_router = APIRouter(prefix="/v1/auth", tags=["authentication"])

    @auth_router.post("/token", response_model=TokenResponse)
    async def generate_token(
        api_key: str = Field(..., description="API key to exchange for token"),
    ):
        """Generate a JWT token from an API key."""
        return authenticator.generate_token(api_key)

    @auth_router.post("/revoke")
    async def revoke_token(jti: str = Field(..., description="JWT ID to revoke")):
        """Revoke a JWT token by its JTI."""
        authenticator.revoke_token(jti)
        return {"message": f"Token {jti} has been revoked"}

    app.include_router(auth_router)

    logger.info("JWT authentication configured successfully")
    return authenticator
