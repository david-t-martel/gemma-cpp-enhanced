"""JWT authentication handler with production-ready security features.

This module provides secure JWT token generation and validation following
RFC 8725 best practices for JSON Web Token security.
"""

import secrets
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import ClassVar

import jwt
from fastapi import HTTPException
from fastapi import status
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from ...shared.logging import get_logger

logger = get_logger(__name__)


class TokenType(str, Enum):
    """Types of JWT tokens supported."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class TokenClaims(BaseModel):
    """Standard JWT claims with validation."""

    sub: str = Field(..., description="Subject (user ID)")
    exp: int = Field(..., description="Expiration time")
    iat: int = Field(..., description="Issued at time")
    nbf: int | None = Field(None, description="Not before time")
    jti: str | None = Field(None, description="JWT ID for revocation")
    token_type: TokenType = Field(TokenType.ACCESS, description="Token type")
    scope: str | None = Field(None, description="Token scope/permissions")

    @field_validator("exp")
    @classmethod
    def validate_expiration(cls, v: int) -> int:
        """Ensure expiration is in the future."""
        if v <= datetime.now(UTC).timestamp():
            raise ValueError("Token expiration must be in the future")
        return v


class JWTHandler:
    """Secure JWT token handler with best practices."""

    # Security constants
    MAX_TOKEN_AGE_DAYS = 30
    MIN_SECRET_LENGTH = 32
    SUPPORTED_ALGORITHMS: ClassVar[list[str]] = [
        "HS256",
        "HS384",
        "HS512",
        "RS256",
        "RS384",
        "RS512",
    ]

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        issuer: str = "gemma-llm",
        audience: str | None = None,
        public_key: str | None = None,
        private_key: str | None = None,
    ):
        """Initialize JWT handler with security validation.

        Args:
            secret: Secret key for HMAC algorithms (min 32 chars)
            algorithm: JWT signing algorithm
            issuer: Token issuer identifier
            audience: Expected audience for tokens
            public_key: Public key for RSA algorithms (verification)
            private_key: Private key for RSA algorithms (signing)
        """
        self._validate_configuration(secret, algorithm)

        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience

        # Use appropriate keys based on algorithm
        if algorithm.startswith("RS"):
            if not private_key or not public_key:
                raise ValueError("RSA algorithms require both public and private keys")
            self.signing_key = private_key
            self.verification_key = public_key
        else:
            self.signing_key = secret
            self.verification_key = secret

        # Token revocation set (in production, use Redis)
        self.revoked_tokens: set = set()

    def _validate_configuration(self, secret: str, algorithm: str):
        """Validate security configuration."""
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if algorithm.startswith("HS") and len(secret) < self.MIN_SECRET_LENGTH:
            raise ValueError(f"Secret must be at least {self.MIN_SECRET_LENGTH} characters")

    def create_access_token(
        self,
        subject: str,
        scope: str | None = None,
        expiry_hours: int = 1,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a secure access token.

        Args:
            subject: User/entity identifier
            scope: Permission scope for the token
            expiry_hours: Token validity in hours
            additional_claims: Extra claims to include

        Returns:
            Signed JWT token string
        """
        if expiry_hours > self.MAX_TOKEN_AGE_DAYS * 24:
            raise ValueError(f"Token expiry cannot exceed {self.MAX_TOKEN_AGE_DAYS} days")

        now = datetime.now(UTC)
        expiry = now + timedelta(hours=expiry_hours)

        # Build claims
        claims = {
            "sub": subject,
            "exp": expiry,
            "iat": now,
            "nbf": now,  # Not valid before now
            "iss": self.issuer,
            "jti": secrets.token_urlsafe(16),  # Unique token ID
            "token_type": TokenType.ACCESS.value,
            "scope": scope or "read",
        }

        # Add audience if configured
        if self.audience:
            claims["aud"] = self.audience

        # Add additional claims
        if additional_claims:
            # Prevent overriding standard claims
            protected_claims = {"sub", "exp", "iat", "nbf", "iss", "aud", "jti"}
            claims.update(
                {
                    key: value
                    for key, value in additional_claims.items()
                    if key not in protected_claims
                }
            )

        # Sign and return token
        token = jwt.encode(claims, self.signing_key, algorithm=self.algorithm)

        logger.info(f"Created access token for subject: {subject}, jti: {claims['jti']}")
        return token

    def create_refresh_token(self, subject: str, expiry_days: int = 7) -> str:
        """Create a refresh token with longer validity.

        Args:
            subject: User/entity identifier
            expiry_days: Token validity in days

        Returns:
            Signed refresh token
        """
        if expiry_days > self.MAX_TOKEN_AGE_DAYS:
            raise ValueError(f"Refresh token expiry cannot exceed {self.MAX_TOKEN_AGE_DAYS} days")

        now = datetime.now(UTC)
        expiry = now + timedelta(days=expiry_days)

        claims = {
            "sub": subject,
            "exp": expiry,
            "iat": now,
            "iss": self.issuer,
            "jti": secrets.token_urlsafe(16),
            "token_type": TokenType.REFRESH.value,
        }

        if self.audience:
            claims["aud"] = self.audience

        token = jwt.encode(claims, self.signing_key, algorithm=self.algorithm)

        logger.info(f"Created refresh token for subject: {subject}, jti: {claims['jti']}")
        return token

    def verify_token(
        self, token: str, expected_type: TokenType | None = None, verify_audience: bool = True
    ) -> dict[str, Any]:
        """Verify and decode a JWT token with security checks.

        Args:
            token: JWT token to verify
            expected_type: Expected token type
            verify_audience: Whether to verify audience claim

        Returns:
            Decoded token claims

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Decode options
            options = {
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": verify_audience and self.audience is not None,
                "require": ["exp", "iat", "sub", "jti"],
            }

            # Decode token
            claims = jwt.decode(
                token,
                self.verification_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience if verify_audience else None,
                options=options,
            )

            # Check if token is revoked
            if claims.get("jti") in self.revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked"
                )

            # Verify token type if specified
            if expected_type and claims.get("token_type") != expected_type.value:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {expected_type.value}",
                )

            # Additional security checks
            self._perform_security_checks(claims)

            return claims

        except jwt.ExpiredSignatureError as e:
            logger.warning("Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            ) from e
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e!s}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from e
        except HTTPException:
            # Re-raise HTTPExceptions (like token revocation) without masking
            raise
        except Exception as e:
            logger.error(f"Token verification error: {e!s}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed"
            ) from e

    def _perform_security_checks(self, claims: dict[str, Any]):
        """Perform additional security validations on claims."""
        # Check token age (prevent very old tokens)
        iat = claims.get("iat")
        if iat:
            token_age = datetime.now(UTC).timestamp() - iat
            max_age = self.MAX_TOKEN_AGE_DAYS * 24 * 3600
            if token_age > max_age:
                raise jwt.InvalidTokenError("Token is too old")

        # Check for required claims based on token type
        token_type = claims.get("token_type")
        if token_type == TokenType.ACCESS.value and not claims.get("scope"):
            raise jwt.InvalidTokenError("Access token missing scope")

    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """Exchange refresh token for new access token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        # Verify refresh token
        claims = self.verify_token(refresh_token, expected_type=TokenType.REFRESH)

        # Revoke old refresh token
        self.revoke_token(claims.get("jti"))

        # Create new tokens
        subject = claims["sub"]
        new_access = self.create_access_token(subject)
        new_refresh = self.create_refresh_token(subject)

        logger.info(f"Refreshed tokens for subject: {subject}")

        return new_access, new_refresh

    def revoke_token(self, jti: str | None):
        """Revoke a token by its JTI.

        Args:
            jti: JWT ID to revoke
        """
        if jti:
            self.revoked_tokens.add(jti)
            logger.info(f"Revoked token with jti: {jti}")

    def extract_token_from_header(self, authorization: str) -> str:
        """Extract token from Authorization header.

        Args:
            authorization: Authorization header value

        Returns:
            Extracted token

        Raises:
            HTTPException: If header format is invalid
        """
        try:
            scheme, token = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid scheme")
            return token
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            ) from e

    @staticmethod
    def generate_secret_key(length: int = 64) -> str:
        """Generate a cryptographically secure secret key.

        Args:
            length: Key length in bytes

        Returns:
            URL-safe secret key
        """
        return secrets.token_urlsafe(length)


class JWTBearer:
    """JWT Bearer authentication dependency for FastAPI."""

    def __init__(self, handler: JWTHandler, auto_error: bool = True):
        """Initialize JWT Bearer authentication.

        Args:
            handler: JWT handler instance
            auto_error: Whether to raise exception on auth failure
        """
        self.handler = handler
        self.auto_error = auto_error

    async def __call__(self, authorization: str | None = None) -> dict[str, Any] | None:
        """Validate JWT token from request.

        Args:
            authorization: Authorization header value

        Returns:
            Decoded token claims or None
        """
        if not authorization:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing"
                )
            return None

        try:
            token = self.handler.extract_token_from_header(authorization)
            claims = self.handler.verify_token(token)
            return claims
        except HTTPException:
            if self.auto_error:
                raise
            return None


# Export main components
__all__ = ["JWTBearer", "JWTHandler", "TokenClaims", "TokenType"]
