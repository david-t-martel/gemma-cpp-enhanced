"""Inference service providing high-level LLM operations with async support.

This module implements the application layer service that orchestrates LLM
operations, handles caching, and provides a clean interface for the API layer.
"""

import hashlib
import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

from ...domain.interfaces.llm import LLMProtocol
from ...domain.models.chat import ChatMessage
from ...domain.models.chat import ChatSession
from ...domain.models.chat import StreamingResponse
from ...domain.models.chat import TokenUsage
from ...infrastructure.llm.gemma import GemmaLLM
from ...shared.config.settings import Settings
from ...shared.config.settings import get_settings
from ...shared.exceptions import ConfigurationException
from ...shared.exceptions import InferenceException
from ...shared.exceptions import ValidationException
from ...shared.logging import get_logger

logger = get_logger(__name__)


class InferenceService:
    """High-level service for LLM inference operations.

    This service provides a clean, async interface for generating responses,
    managing chat sessions, and handling streaming operations with proper
    resource management and error handling.
    """

    def __init__(
        self,
        llm: LLMProtocol | None = None,
        settings: Settings | None = None,
        enable_caching: bool | None = None,
    ) -> None:
        """Initialize the inference service.

        Args:
            llm: LLM implementation to use (creates default if None)
            settings: Configuration settings
            enable_caching: Enable response caching (uses settings default if None)
        """
        self.settings = settings or get_settings()
        self.logger = get_logger(f"{__name__}.InferenceService")

        # Initialize LLM
        if llm is not None:
            self._llm = llm
        else:
            self._llm = self._create_default_llm()

        # Caching configuration
        self.enable_caching = (
            enable_caching if enable_caching is not None else self.settings.cache.enabled
        )
        self._cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}

        # Resource management
        self._is_initialized = False
        self._exit_stack: AsyncExitStack | None = None

        # Performance tracking
        self._request_count = 0
        self._total_request_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

    async def __aenter__(self) -> "InferenceService":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the inference service and load the model."""
        if self._is_initialized:
            return

        self.logger.info("Initializing inference service")
        start_time = time.time()

        try:
            # Setup resource management
            self._exit_stack = AsyncExitStack()

            # Load the model
            if not self._llm.is_loaded:
                await self._llm.load_model()

            # Warm up the model
            await self._warm_up_model()

            self._is_initialized = True
            init_time = time.time() - start_time
            self.logger.info(f"Inference service initialized in {init_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Failed to initialize inference service: {e}")
            await self.cleanup()
            raise ConfigurationException(f"Service initialization failed: {e}")

    async def cleanup(self) -> None:
        """Clean up resources used by the service."""
        if not self._is_initialized:
            return

        self.logger.info("Cleaning up inference service")

        try:
            # Cleanup LLM
            if self._llm and self._llm.is_loaded:
                await self._llm.unload_model()

            # Cleanup resource stack
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None

            # Clear caches
            self._cache.clear()
            self._cache_timestamps.clear()

            self._is_initialized = False
            self.logger.info("Inference service cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def generate_response(
        self,
        session: ChatSession,
        user_message: str,
        **generation_params: Any,
    ) -> ChatMessage:
        """Generate a response for a user message in a chat session.

        Args:
            session: Chat session to add the message to
            user_message: User's input message
            **generation_params: Generation parameters to override defaults

        Returns:
            Generated assistant message

        Raises:
            InferenceException: If generation fails
            ValidationException: If inputs are invalid
        """
        await self._ensure_initialized()

        # Validate inputs
        if not user_message.strip():
            raise ValidationException("User message cannot be empty")

        # Add user message to session
        user_msg = session.add_user_message(user_message)
        self.logger.info(f"Processing message for session {session.id}")

        try:
            # Check cache first
            if self.enable_caching:
                cached_response = await self._get_cached_response(session, generation_params)
                if cached_response:
                    self._cache_hits += 1
                    self.logger.debug("Cache hit for response")
                    session.add_message(cached_response)
                    return cached_response

            self._cache_misses += 1

            # Prepare messages for inference
            messages = session.get_messages_for_inference()

            # Generate response
            start_time = time.time()
            response = await self._llm.generate_response(messages, **generation_params)

            # Update session
            session.add_message(response)

            # Cache the response
            if self.enable_caching:
                await self._cache_response(session, generation_params, response)

            # Update statistics
            self._request_count += 1
            self._total_request_time += time.time() - start_time

            self.logger.info(
                f"Generated response for session {session.id} in {response.processing_time_ms}ms"
            )
            return response

        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            # Remove the user message if generation failed
            if session.messages and session.messages[-1].id == user_msg.id:
                session.messages.pop()
            raise

    async def generate_streaming_response(
        self,
        session: ChatSession,
        user_message: str,
        **generation_params: Any,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Generate a streaming response for a user message.

        Args:
            session: Chat session to add the message to
            user_message: User's input message
            **generation_params: Generation parameters

        Yields:
            StreamingResponse chunks

        Raises:
            InferenceException: If generation fails
            ValidationException: If inputs are invalid
        """
        await self._ensure_initialized()

        # Validate inputs
        if not user_message.strip():
            raise ValidationException("User message cannot be empty")

        # Add user message to session
        user_msg = session.add_user_message(user_message)
        self.logger.info(f"Processing streaming message for session {session.id}")

        # Prepare for streaming
        accumulated_content = ""
        assistant_msg = None

        try:
            # Prepare messages for inference
            messages = session.get_messages_for_inference()

            # Add streaming parameters
            streaming_params = {
                **generation_params,
                "session_id": session.id,
                "message_id": user_msg.id,
            }

            # Generate streaming response
            async for chunk in self._llm.generate_streaming_response(messages, **streaming_params):
                accumulated_content += chunk.content

                # Create assistant message on first chunk
                if assistant_msg is None and chunk.content:
                    assistant_msg = session.add_assistant_message(chunk.content)

                # Update message content
                if assistant_msg and chunk.content:
                    assistant_msg.content = accumulated_content

                # Update chunk with correct message ID
                chunk.message_id = assistant_msg.id if assistant_msg else chunk.message_id

                yield chunk

            # Finalize the assistant message
            if assistant_msg:
                assistant_msg.content = accumulated_content.strip()
                # Update token usage if available
                try:
                    input_tokens = await self._llm.count_tokens(
                        " ".join(msg["content"] for msg in messages)
                    )
                    output_tokens = await self._llm.count_tokens(accumulated_content)
                    assistant_msg.token_usage = TokenUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to count tokens: {e}")

            self.logger.info(f"Completed streaming response for session {session.id}")

        except Exception as e:
            self.logger.error(f"Failed to generate streaming response: {e}")
            # Clean up partial message
            if assistant_msg and assistant_msg in session.messages:
                session.messages.remove(assistant_msg)
            if session.messages and session.messages[-1].id == user_msg.id:
                session.messages.pop()
            raise

    async def continue_conversation(
        self,
        session: ChatSession,
        **generation_params: Any,
    ) -> ChatMessage:
        """Continue a conversation without adding a new user message.

        This is useful for having the assistant continue its previous response
        or generate follow-up content.

        Args:
            session: Chat session to continue
            **generation_params: Generation parameters

        Returns:
            Generated assistant message
        """
        await self._ensure_initialized()

        if not session.has_messages():
            raise ValidationException("Cannot continue conversation with empty session")

        try:
            messages = session.get_messages_for_inference()
            response = await self._llm.generate_response(messages, **generation_params)
            session.add_message(response)

            self.logger.info(f"Continued conversation for session {session.id}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to continue conversation: {e}")
            raise

    async def estimate_tokens(
        self,
        session: ChatSession,
        additional_message: str | None = None,
    ) -> TokenUsage:
        """Estimate token usage for a session and optional additional message.

        Args:
            session: Chat session to estimate
            additional_message: Optional additional message to include

        Returns:
            Token usage estimation
        """
        await self._ensure_initialized()

        messages = session.get_messages_for_inference()
        if additional_message:
            messages.append({"role": "user", "content": additional_message})

        return await self._llm.estimate_tokens_for_messages(messages)

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        await self._ensure_initialized()
        info = await self._llm.get_model_info()
        return dict(info) if info else {}

    async def health_check(self) -> dict[str, Any]:
        """Perform a comprehensive health check of the service."""
        try:
            # Check if service is initialized
            if not self._is_initialized:
                return {"status": "unhealthy", "reason": "Service not initialized"}

            # Check LLM health
            llm_healthy = await self._llm.health_check()
            if not llm_healthy:
                return {"status": "unhealthy", "reason": "LLM health check failed"}

            # Get service statistics
            stats = await self.get_statistics()

            return {
                "status": "healthy",
                "model_info": await self._llm.get_model_info(),
                "statistics": stats,
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "reason": str(e)}

    async def get_statistics(self) -> dict[str, Any]:
        """Get service performance statistics."""
        avg_request_time = (
            self._total_request_time / self._request_count if self._request_count > 0 else 0
        )

        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0
        )

        return {
            "request_count": self._request_count,
            "avg_request_time": avg_request_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "is_initialized": self._is_initialized,
        }

    def _create_default_llm(self) -> LLMProtocol:
        """Create the default LLM implementation."""
        model_name = self.settings.model.name.lower()

        if "gemma" in model_name:
            return GemmaLLM(self.settings)
        else:
            # Default to Gemma for now
            self.logger.warning(f"Unknown model {model_name}, defaulting to Gemma")
            return GemmaLLM(self.settings)

    async def _ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self._is_initialized:
            raise InferenceException("Service is not initialized. Call initialize() first.")

    async def _warm_up_model(self) -> None:
        """Warm up the model with a test inference."""
        try:
            self.logger.info("Warming up model")
            test_messages = [{"role": "user", "content": "Hello"}]
            await self._llm.generate_response(test_messages, max_length=10)
            self.logger.info("Model warm-up completed")
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")

    async def _get_cached_response(
        self,
        session: ChatSession,
        generation_params: dict[str, Any],
    ) -> ChatMessage | None:
        """Get a cached response if available."""
        try:
            cache_key = self._generate_cache_key(session, generation_params)

            if cache_key in self._cache:
                # Check if cache entry is still valid
                timestamp = self._cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < self.settings.cache.ttl_seconds:
                    return self._cache[cache_key]
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
                    del self._cache_timestamps[cache_key]

            return None

        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {e}")
            return None

    async def _cache_response(
        self,
        session: ChatSession,
        generation_params: dict[str, Any],
        response: ChatMessage,
    ) -> None:
        """Cache a response."""
        try:
            cache_key = self._generate_cache_key(session, generation_params)

            # Check cache size limits
            if len(self._cache) >= 1000:  # Simple size limit
                self._evict_oldest_entries()

            self._cache[cache_key] = response
            self._cache_timestamps[cache_key] = time.time()

        except Exception as e:
            self.logger.warning(f"Response caching failed: {e}")

    def _generate_cache_key(
        self,
        session: ChatSession,
        generation_params: dict[str, Any],
    ) -> str:
        """Generate a cache key for the session and parameters."""
        # Create a deterministic hash of messages and parameters
        messages_str = str(session.get_messages_for_inference())
        params_str = str(sorted(generation_params.items()))
        key_string = f"{messages_str}:{params_str}"

        return hashlib.md5(key_string.encode()).hexdigest()

    def _evict_oldest_entries(self, keep_count: int = 800) -> None:
        """Evict oldest cache entries to free space."""
        if len(self._cache) <= keep_count:
            return

        # Sort by timestamp and keep only the newest entries
        sorted_keys = sorted(self._cache_timestamps.items(), key=lambda x: x[1], reverse=True)

        keys_to_keep = {key for key, _ in sorted_keys[:keep_count]}

        # Remove old entries
        keys_to_remove = set(self._cache.keys()) - keys_to_keep
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
