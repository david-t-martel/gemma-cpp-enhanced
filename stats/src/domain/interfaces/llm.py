"""Abstract LLM interface using Protocol for type-safe LLM implementations.

This module defines the contract that all LLM implementations must follow,
ensuring consistency and enabling easy testing with mock implementations.
"""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from ..models.chat import ChatMessage
from ..models.chat import StreamingResponse
from ..models.chat import TokenUsage


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol defining the interface for LLM implementations.

    This protocol ensures all LLM implementations provide the same methods
    and can be used interchangeably throughout the application.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the loaded model."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device the model is running on."""
        ...

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported by the model."""
        ...

    @abstractmethod
    async def load_model(self) -> None:
        """Load the model into memory.

        Raises:
            ModelLoadException: If the model fails to load
            ResourceException: If insufficient resources are available
        """
        ...

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the model from memory to free resources.

        Raises:
            LLMException: If the model fails to unload cleanly
        """
        ...

    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict[str, str]],
        max_length: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Generate a single response for the given messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_length: Maximum length of the generated response
            temperature: Sampling temperature (0.0-2.0)
            top_p: Top-p (nucleus) sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty (>1.0 reduces repetition)
            stop_sequences: List of sequences that stop generation
            **kwargs: Additional model-specific parameters

        Returns:
            ChatMessage with the generated response

        Raises:
            InferenceException: If generation fails
            ValidationException: If parameters are invalid
            TimeoutException: If generation exceeds timeout
        """
        ...

    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: list[dict[str, str]],
        max_length: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Generate a streaming response for the given messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_length: Maximum length of the generated response
            temperature: Sampling temperature (0.0-2.0)
            top_p: Top-p (nucleus) sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty (>1.0 reduces repetition)
            stop_sequences: List of sequences that stop generation
            **kwargs: Additional model-specific parameters

        Yields:
            StreamingResponse chunks as they are generated

        Raises:
            InferenceException: If generation fails
            ValidationException: If parameters are invalid
            TimeoutException: If generation exceeds timeout
            StreamingException: If streaming fails
        """
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: Text to tokenize and count

        Returns:
            Number of tokens in the text

        Raises:
            TokenizationException: If tokenization fails
        """
        ...

    @abstractmethod
    async def validate_messages(self, messages: list[dict[str, str]]) -> bool:
        """Validate that messages are properly formatted and within limits.

        Args:
            messages: List of message dictionaries to validate

        Returns:
            True if messages are valid

        Raises:
            ValidationException: If messages are invalid
        """
        ...

    @abstractmethod
    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata and capabilities

        Raises:
            LLMException: If model info cannot be retrieved
        """
        ...

    @abstractmethod
    async def estimate_tokens_for_messages(self, messages: list[dict[str, str]]) -> TokenUsage:
        """Estimate token usage for a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            TokenUsage estimation for the messages

        Raises:
            TokenizationException: If token estimation fails
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check on the model.

        Returns:
            True if the model is healthy and ready for inference

        Raises:
            LLMException: If health check fails
        """
        ...


@runtime_checkable
class LLMManagerProtocol(Protocol):
    """Protocol for managing LLM lifecycle and resources.

    This protocol defines methods for managing model loading, unloading,
    and resource cleanup in a production environment.
    """

    @abstractmethod
    async def __aenter__(self) -> LLMProtocol:
        """Async context manager entry."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        ...

    @abstractmethod
    async def get_llm(self) -> LLMProtocol:
        """Get the LLM instance.

        Returns:
            LLM instance ready for inference
        """
        ...

    @abstractmethod
    async def warm_up_model(self) -> None:
        """Warm up the model with a test inference.

        This can help reduce latency for the first real inference request.
        """
        ...

    @abstractmethod
    async def cleanup_resources(self) -> None:
        """Clean up all resources used by the LLM."""
        ...

    @abstractmethod
    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        ...


class LLMContextManager(AbstractAsyncContextManager[LLMProtocol]):
    """Abstract base class for LLM context managers.

    Provides a template for implementing async context managers that handle
    LLM lifecycle management with proper resource cleanup.
    """

    def __init__(self, llm: LLMProtocol) -> None:
        self.llm = llm
        self._entered = False

    async def __aenter__(self) -> LLMProtocol:
        """Load the model and prepare for inference."""
        if not self.llm.is_loaded:
            await self.llm.load_model()
        self._entered = True
        return self.llm

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources."""
        if self._entered:
            await self.llm.unload_model()
            self._entered = False


# Type aliases for convenience
LLMInstance = LLMProtocol
LLMManager = LLMManagerProtocol
