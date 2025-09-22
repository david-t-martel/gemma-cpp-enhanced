"""Custom exceptions for the LLM chatbot framework.

This module defines domain-specific exceptions that provide clear error handling
and debugging information throughout the application.
"""

from typing import Any


class ChatbotException(Exception):
    """Base exception for all chatbot-related errors.

    All custom exceptions in the framework should inherit from this base class
    to provide consistent error handling patterns.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        return f"{self.message} (code: {self.error_code})" if self.error_code else self.message


class LLMException(ChatbotException):
    """Base exception for LLM-related operations."""


class ModelLoadException(LLMException):
    """Raised when a model fails to load."""


class InferenceException(LLMException):
    """Raised during inference operations."""


class TokenizationException(LLMException):
    """Raised during text tokenization operations."""


class ModelNotFoundException(LLMException):
    """Raised when a requested model is not found."""


class ConfigurationException(ChatbotException):
    """Raised for configuration-related errors."""


class ValidationException(ChatbotException):
    """Raised for data validation errors."""


class ResourceException(ChatbotException):
    """Raised for resource management errors (memory, GPU, etc.)."""


class TimeoutException(ChatbotException):
    """Raised when operations exceed their timeout limits."""


class StreamingException(ChatbotException):
    """Raised during streaming operations."""
