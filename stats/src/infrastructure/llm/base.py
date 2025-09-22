"""Base LLM implementation providing common functionality.

This module provides a base class that implements common LLM functionality
and can be extended by specific model implementations.
"""

import asyncio
import time
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

from src.domain.interfaces.llm import LLMProtocol
from src.domain.models.chat import ChatMessage
from src.domain.models.chat import MessageRole
from src.domain.models.chat import StreamingResponse
from src.domain.models.chat import TokenUsage
from src.shared.config.settings import Settings
from src.shared.config.settings import get_settings
from src.shared.exceptions import InferenceException
from src.shared.exceptions import ModelLoadException
from src.shared.exceptions import ResourceException
from src.shared.exceptions import TimeoutException
from src.shared.exceptions import TokenizationException
from src.shared.exceptions import ValidationException
from src.shared.logging import get_logger

logger = get_logger(__name__)


class BaseLLM(LLMProtocol, ABC):
    """Base implementation of the LLM protocol.

    This class provides common functionality that can be shared across
    different model implementations while enforcing the LLM protocol contract.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the base LLM.

        Args:
            settings: Configuration settings (uses global settings if None)
        """
        self.settings = settings or get_settings()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Model-related attributes
        self._model: Any | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._device: str | None = None
        self._is_loaded = False

        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._total_tokens_generated = 0

    @property
    def model_name(self) -> str:
        """Get the name of the model."""
        return self.settings.model.name

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded and self._model is not None

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        if self._device is None:
            self._device = self.settings.get_device()
        return self._device

    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length."""
        return self.settings.model.max_length

    @abstractmethod
    async def _load_model_implementation(self) -> None:
        """Load the specific model implementation.

        This method must be implemented by subclasses to handle
        model-specific loading logic.
        """

    @abstractmethod
    async def _generate_implementation(
        self,
        prompt: str,
        **generation_kwargs: Any,
    ) -> str:
        """Generate text using the specific model implementation.

        This method must be implemented by subclasses to handle
        model-specific generation logic.

        Args:
            prompt: The input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Generated text
        """

    @abstractmethod
    async def _generate_streaming_implementation(
        self,
        prompt: str,
        **generation_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using the specific model implementation.

        This method must be implemented by subclasses to handle
        model-specific streaming generation logic.

        Args:
            prompt: The input prompt
            **generation_kwargs: Generation parameters

        Yields:
            Generated text chunks
        """

    async def load_model(self) -> None:
        """Load the model into memory."""
        if self.is_loaded:
            self.logger.info(f"Model {self.model_name} is already loaded")
            return

        self.logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            # Load tokenizer first
            await self._load_tokenizer()

            # Load the model implementation
            await asyncio.wait_for(
                self._load_model_implementation(), timeout=self.settings.model_load_timeout
            )

            # Verify the model loaded correctly
            if not await self.health_check():
                raise ModelLoadException("Model health check failed after loading")

            self._is_loaded = True
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

        except TimeoutError:
            raise TimeoutException(
                f"Model loading timed out after {self.settings.model_load_timeout} seconds"
            )
        except Exception as e:
            self._is_loaded = False
            self.logger.error(f"Failed to load model: {e}")
            raise ModelLoadException(f"Failed to load model {self.model_name}: {e}")

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if not self.is_loaded:
            return

        self.logger.info(f"Unloading model: {self.model_name}")

        try:
            # Clear model and tokenizer
            self._model = None
            self._tokenizer = None
            self._is_loaded = False

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("Model unloaded successfully")

        except Exception as e:
            self.logger.error(f"Error during model unloading: {e}")
            raise ResourceException(f"Failed to unload model: {e}")

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
        """Generate a complete response for the given messages."""
        if not self.is_loaded:
            raise InferenceException("Model is not loaded")

        # Validate messages
        await self.validate_messages(messages)

        # Prepare generation parameters
        generation_kwargs = self._prepare_generation_kwargs(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            **kwargs,
        )

        # Convert messages to prompt
        prompt = await self._messages_to_prompt(messages)

        # Generate response
        start_time = time.time()
        try:
            response_text = await asyncio.wait_for(
                self._generate_implementation(prompt, **generation_kwargs),
                timeout=self.settings.inference_timeout,
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Count tokens
            input_tokens = await self.count_tokens(prompt)
            output_tokens = await self.count_tokens(response_text)

            # Update statistics
            self._inference_count += 1
            self._total_inference_time += time.time() - start_time
            self._total_tokens_generated += output_tokens

            # Create response message
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_text,
                token_usage=TokenUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
                processing_time_ms=processing_time_ms,
            )

        except TimeoutError:
            raise TimeoutException(
                f"Inference timed out after {self.settings.inference_timeout} seconds"
            )
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise InferenceException(f"Failed to generate response: {e}")

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
        """Generate a streaming response for the given messages."""
        if not self.is_loaded:
            raise InferenceException("Model is not loaded")

        # Validate messages
        await self.validate_messages(messages)

        # Prepare generation parameters
        generation_kwargs = self._prepare_generation_kwargs(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            **kwargs,
        )

        # Convert messages to prompt
        prompt = await self._messages_to_prompt(messages)

        # Generate streaming response
        session_id = kwargs.get("session_id", "unknown")
        message_id = kwargs.get("message_id", "unknown")
        accumulated_content = ""

        try:
            async for chunk in self._generate_streaming_implementation(prompt, **generation_kwargs):
                accumulated_content += chunk
                token_count = len(chunk.split())  # Simple token approximation

                yield StreamingResponse(
                    session_id=session_id,
                    message_id=message_id,
                    content=chunk,
                    is_complete=False,
                    token_count=token_count,
                )

            # Send final chunk
            yield StreamingResponse(
                session_id=session_id,
                message_id=message_id,
                content="",
                is_complete=True,
                token_count=0,
            )

        except Exception as e:
            self.logger.error(f"Streaming inference failed: {e}")
            raise InferenceException(f"Failed to generate streaming response: {e}")

    async def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self._tokenizer is None:
            raise TokenizationException("Tokenizer is not loaded")

        try:
            # Run tokenization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(
                None, lambda: self._tokenizer.encode(text, add_special_tokens=True)
            )
            return len(tokens)

        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise TokenizationException(f"Failed to count tokens: {e}")

    async def validate_messages(self, messages: list[dict[str, str]]) -> bool:
        """Validate message format and content."""
        if not messages:
            raise ValidationException("Messages list cannot be empty")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationException(f"Message {i} must be a dictionary")

            if "role" not in message or "content" not in message:
                raise ValidationException(f"Message {i} must have 'role' and 'content' keys")

            if message["role"] not in {"system", "user", "assistant"}:
                raise ValidationException(f"Message {i} has invalid role: {message['role']}")

            if not isinstance(message["content"], str):
                raise ValidationException(f"Message {i} content must be a string")

        # Check total length
        total_length = sum(len(msg["content"]) for msg in messages)
        if total_length > self.max_sequence_length * 4:  # Rough character-to-token ratio
            raise ValidationException("Messages exceed maximum sequence length")

        return True

    async def estimate_tokens_for_messages(self, messages: list[dict[str, str]]) -> TokenUsage:
        """Estimate token usage for messages."""
        prompt = await self._messages_to_prompt(messages)
        prompt_tokens = await self.count_tokens(prompt)

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,  # Can't estimate completion tokens
            total_tokens=prompt_tokens,
        )

    async def health_check(self) -> bool:
        """Perform a health check."""
        try:
            if not self.is_loaded:
                return False

            # Simple test generation
            test_messages = [{"role": "user", "content": "Hello"}]
            prompt = await self._messages_to_prompt(test_messages)

            # Just check if we can tokenize
            await self.count_tokens(prompt)
            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "device": self.device,
            "max_length": self.max_sequence_length,
            "is_loaded": self.is_loaded,
            "inference_count": self._inference_count,
            "avg_inference_time": (
                self._total_inference_time / self._inference_count
                if self._inference_count > 0
                else 0
            ),
            "total_tokens_generated": self._total_tokens_generated,
        }

    async def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        try:
            self.logger.info(f"Loading tokenizer for {self.model_name}")
            loop = asyncio.get_event_loop()
            self._tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(self.model_name)
            )
            self.logger.info("Tokenizer loaded successfully")

        except Exception as e:
            raise ModelLoadException(f"Failed to load tokenizer: {e}")

    async def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to a prompt string.

        This is a basic implementation that can be overridden by subclasses
        for model-specific prompt formatting.
        """
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n".join(prompt_parts) + "\nAssistant: "

    def _prepare_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare generation parameters with defaults."""
        generation_kwargs = {}

        # Use provided values or fall back to settings
        generation_kwargs["max_length"] = kwargs.get("max_length") or self.settings.model.max_length
        generation_kwargs["temperature"] = (
            kwargs.get("temperature") or self.settings.model.temperature
        )
        generation_kwargs["top_p"] = kwargs.get("top_p") or self.settings.model.top_p
        generation_kwargs["top_k"] = kwargs.get("top_k") or self.settings.model.top_k
        generation_kwargs["repetition_penalty"] = (
            kwargs.get("repetition_penalty") or self.settings.model.repetition_penalty
        )
        generation_kwargs["do_sample"] = kwargs.get("do_sample", self.settings.model.do_sample)

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in generation_kwargs:
                generation_kwargs[key] = value

        return generation_kwargs
