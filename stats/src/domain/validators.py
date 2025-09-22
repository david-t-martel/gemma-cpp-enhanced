"""Security validators for input sanitization and prompt injection prevention.

This module provides comprehensive validation and sanitization for user inputs
to prevent prompt injection, data leakage, and other security vulnerabilities.
"""

import hashlib
import re
import secrets
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from ..shared.logging import get_logger

logger = get_logger(__name__)


class ThreatLevel(str, Enum):
    """Threat level classification for detected patterns."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


class ValidationResult(BaseModel):
    """Result of input validation."""

    is_valid: bool
    sanitized_input: str
    threat_level: ThreatLevel
    detected_patterns: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)


class PromptValidator:
    """Validator for preventing prompt injection and sensitive data leakage."""

    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r"(?i)ignore\s+(all\s+)?previous\s+(instructions?|commands?)",
        r"(?i)forget\s+(everything|all|previous)",
        r"(?i)disregard\s+(all\s+)?(previous|above)",
        r"(?i)new\s+instructions?:",
        r"(?i)system\s*:\s*",
        r"(?i)admin\s*:\s*",
        r"(?i)root\s*:\s*",
        # Role manipulation attempts
        r"(?i)you\s+are\s+now\s+",
        r"(?i)act\s+as\s+",
        r"(?i)pretend\s+(to\s+be|you('re|are))",
        r"(?i)roleplay\s+as",
        r"(?i)simulate\s+being",
        # Data extraction attempts
        r"(?i)show\s+me\s+(all\s+)?(your\s+)?(instructions?|rules?|constraints?)",
        r"(?i)what\s+are\s+your\s+(instructions?|rules?|guidelines?)",
        r"(?i)reveal\s+(your\s+)?(system\s+)?prompt",
        r"(?i)display\s+(the\s+)?configuration",
        r"(?i)print\s+(your\s+)?source\s+code",
        # Boundary testing
        r"(?i)(\[|\]){3,}",  # Multiple brackets
        r"(?i)<\|.*?\|>",  # Special delimiters
        r"(?i)```system",  # Code block abuse
        r"(?i)###+\s*(system|admin|override)",
        # Command injection attempts
        r"(?i)execute\s*:\s*",
        r"(?i)run\s*:\s*",
        r"(?i)eval\s*\(",
        r"(?i)import\s+os",
        r"(?i)subprocess\.",
        r"(?i)\_\_.*?\_\_",  # Dunder methods
    ]

    # Sensitive data patterns to block
    SENSITIVE_PATTERNS = [
        # API keys and tokens
        r"(?i)(api[_\s\-]?key|token|secret|password)[\s:=]+[\w\-]{20,}",
        r"sk-[a-zA-Z0-9]{20,}",  # OpenAI/similar style keys (20+ chars)
        r"(?i)bearer\s+[\w\-\.]+",
        # Credit card patterns
        r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
        # Social security numbers
        r"\b\d{3}-\d{2}-\d{4}\b",
        # Email patterns (for spam prevention)
        r"(?i)[\w\.\-]+@[\w\.\-]+\.\w{2,4}",
        # Phone numbers
        r"\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b",
        # File paths
        r"(?i)[c-z]:\\[\w\\]+",  # Windows paths
        r"(?i)/(?:etc|usr|var|home)/[\w/]+",  # Unix paths
    ]

    # Harmful content patterns
    HARMFUL_PATTERNS = [
        r"(?i)\b(kill|harm|hurt|damage|destroy)\s+(yourself|someone|people)\b",
        r"(?i)\b(suicide|self[\s\-]?harm)\b",
        r"(?i)\b(illegal|illicit)\s+(drugs?|substances?)\b",
        r"(?i)\b(make|create|build)\s+a?\s?(bomb|explosive|weapon)\b",
        r"(?i)\b(hack|breach|exploit)\s+",
    ]

    def __init__(self, max_length: int = 4096, block_sensitive: bool = True):
        """Initialize the validator.

        Args:
            max_length: Maximum allowed prompt length
            block_sensitive: Whether to block sensitive patterns
        """
        self.max_length = max_length
        self.block_sensitive = block_sensitive
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.injection_regex = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self.sensitive_regex = [re.compile(p) for p in self.SENSITIVE_PATTERNS]
        self.harmful_regex = [re.compile(p) for p in self.HARMFUL_PATTERNS]

    def validate(self, prompt: str) -> ValidationResult:
        """Validate and sanitize a prompt.

        Args:
            prompt: The user prompt to validate

        Returns:
            ValidationResult with sanitization details
        """
        if not prompt:
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                threat_level=ThreatLevel.LOW,
                detected_patterns=["empty_prompt"],
                risk_score=0.1,
            )

        # Track modifications and detections
        detected_patterns = []
        modifications = []
        risk_score = 0.0
        sanitized = prompt

        # Check length
        if len(prompt) > self.max_length:
            sanitized = prompt[: self.max_length]
            modifications.append(f"truncated_to_{self.max_length}_chars")
            risk_score += 0.1

        # Check for injection patterns
        injection_count = 0
        for pattern in self.injection_regex:
            if pattern.search(sanitized):
                injection_count += 1
                detected_patterns.append(f"injection_pattern_{injection_count}")
                risk_score += 0.3

        # Check for sensitive data
        if self.block_sensitive:
            sensitive_count = 0
            for pattern in self.sensitive_regex:
                match = pattern.search(sanitized)
                if match:
                    sensitive_count += 1
                    detected_patterns.append(f"sensitive_data_{sensitive_count}")
                    # Redact sensitive data
                    sanitized = pattern.sub("[REDACTED]", sanitized)
                    modifications.append(f"redacted_sensitive_{sensitive_count}")
                    risk_score += 0.2

        # Check for harmful content
        harmful_count = 0
        for pattern in self.harmful_regex:
            if pattern.search(sanitized):
                harmful_count += 1
                detected_patterns.append(f"harmful_content_{harmful_count}")
                risk_score += 0.4

        # Sanitize special characters that could break parsing
        if "\\x00" in sanitized or "\x00" in sanitized:
            sanitized = sanitized.replace("\\x00", "").replace("\x00", "")
            modifications.append("removed_null_bytes")
            risk_score += 0.1

        # Remove excessive whitespace
        if re.search(r"\s{10,}", sanitized):
            sanitized = re.sub(r"\s{10,}", " ", sanitized)
            modifications.append("normalized_whitespace")

        # Escape potentially dangerous Unicode characters
        dangerous_unicode = [
            "\u202e",  # Right-to-left override
            "\u200b",  # Zero-width space
            "\ufeff",  # Zero-width no-break space
        ]
        for char in dangerous_unicode:
            if char in sanitized:
                sanitized = sanitized.replace(char, "")
                modifications.append("removed_dangerous_unicode")
                risk_score += 0.1

        # Determine threat level
        risk_score = min(risk_score, 1.0)
        if risk_score >= 0.7:
            threat_level = ThreatLevel.CRITICAL
        elif risk_score >= 0.5:
            threat_level = ThreatLevel.HIGH
        elif risk_score >= 0.3:
            threat_level = ThreatLevel.MEDIUM
        elif risk_score >= 0.1:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.SAFE

        # Determine if input is valid
        is_valid = threat_level not in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]

        # Log security events
        if detected_patterns:
            logger.warning(
                f"Security validation detected patterns: {detected_patterns}, "
                f"threat_level: {threat_level}, risk_score: {risk_score:.2f}"
            )

        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            threat_level=threat_level,
            detected_patterns=detected_patterns,
            modifications=modifications,
            risk_score=risk_score,
        )


class APIKeyValidator:
    """Validator for API key authentication."""

    def __init__(self, valid_keys: list[str] | None = None, use_hash: bool = True):
        """Initialize the API key validator.

        Args:
            valid_keys: List of valid API keys
            use_hash: Whether to store keys as hashes for security
        """
        self.use_hash = use_hash
        self.valid_keys = set()

        if valid_keys:
            for key in valid_keys:
                if use_hash:
                    self.valid_keys.add(self._hash_key(key))
                else:
                    self.valid_keys.add(key)

    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage.

        Args:
            key: The API key to hash

        Returns:
            Hashed key
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def validate(self, api_key: str) -> tuple[bool, str | None]:
        """Validate an API key.

        Args:
            api_key: The API key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key:
            return False, "API key is required"

        # Check key format
        if len(api_key) < 32:
            return False, "Invalid API key format"

        # Check against valid keys
        if self.use_hash:
            key_hash = self._hash_key(api_key)
            if key_hash not in self.valid_keys:
                return False, "Invalid API key"
        elif api_key not in self.valid_keys:
            return False, "Invalid API key"

        return True, None

    def generate_key(self, length: int = 48) -> str:
        """Generate a secure API key.

        Args:
            length: Length of the API key

        Returns:
            Generated API key
        """
        return secrets.token_urlsafe(length)


class RequestValidator:
    """Validator for HTTP request data."""

    def __init__(self, max_size_mb: int = 10):
        """Initialize the request validator.

        Args:
            max_size_mb: Maximum request size in MB
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate_content_type(self, content_type: str) -> tuple[bool, str | None]:
        """Validate request content type.

        Args:
            content_type: The Content-Type header value

        Returns:
            Tuple of (is_valid, error_message)
        """
        allowed_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ]

        if not content_type:
            return False, "Content-Type header is required"

        # Extract main type (ignore charset and other parameters)
        main_type = content_type.split(";")[0].strip().lower()

        if main_type not in allowed_types:
            return False, f"Unsupported Content-Type: {main_type}"

        return True, None

    def validate_size(self, content_length: int) -> tuple[bool, str | None]:
        """Validate request size.

        Args:
            content_length: The Content-Length header value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if content_length > self.max_size_bytes:
            return False, f"Request size exceeds maximum of {self.max_size_bytes} bytes"

        return True, None

    def validate_headers(self, headers: dict[str, str]) -> tuple[bool, str | None]:
        """Validate request headers for security issues.

        Args:
            headers: Dictionary of request headers

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for header injection attempts
        dangerous_headers = [
            "x-forwarded-host",
            "x-original-url",
            "x-rewrite-url",
        ]

        for header in dangerous_headers:
            if header.lower() in [h.lower() for h in headers]:
                logger.warning(f"Potentially dangerous header detected: {header}")

        # Check for oversized headers
        for name, value in headers.items():
            if len(name) > 256 or len(value) > 8192:
                return False, "Header size exceeds limits"

        return True, None


# Export main validators
__all__ = [
    "APIKeyValidator",
    "PromptValidator",
    "RequestValidator",
    "ThreatLevel",
    "ValidationResult",
]
