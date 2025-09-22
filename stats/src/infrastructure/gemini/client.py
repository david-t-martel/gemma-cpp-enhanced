"""Gemini API client wrapper for interacting with Google's Gemini models.

This module provides a robust, async-first client for interacting with
Google's Gemini API, including error handling, retries, and rate limiting.
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import httpx
from pydantic import BaseModel
from pydantic import Field

from ...shared.logging import get_logger

logger = get_logger(__name__)


class GeminiModel(str, Enum):
    """Available Gemini model variants."""

    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_PRO_LATEST = "gemini-2.5-pro-latest"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LATEST = "gemini-2.5-flash-latest"


class HarmCategory(str, Enum):
    """Safety harm categories."""

    HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"


class HarmBlockThreshold(str, Enum):
    """Safety block thresholds."""

    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_HIGH_AND_ABOVE = "BLOCK_HIGH_AND_ABOVE"


@dataclass
class GeminiConfig:
    """Configuration for Gemini client."""

    api_key: str | None = None
    model: GeminiModel = GeminiModel.GEMINI_2_5_FLASH
    max_retries: int = 3
    timeout: float = 30.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    candidate_count: int = 1
    stop_sequences: list[str] = None
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = None
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def __post_init__(self):
        """Post-initialization processing."""
        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            logger.warning(
                "No Gemini API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable or pass api_key to GeminiConfig."
            )

        # Initialize default safety settings if not provided
        if self.safety_settings is None:
            self.safety_settings = {
                HarmCategory.HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

        if self.stop_sequences is None:
            self.stop_sequences = []


class GeminiResponse(BaseModel):
    """Structured response from Gemini API."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    safety_ratings: dict[str, Any] | None = None


class GeminiException(Exception):
    """Base exception for Gemini client errors."""


class GeminiRateLimitError(GeminiException):
    """Rate limit exceeded error."""


class GeminiAuthenticationError(GeminiException):
    """Authentication error."""


class GeminiClient:
    """Async client for interacting with Google's Gemini API."""

    def __init__(self, config: GeminiConfig | None = None):
        """Initialize the Gemini client.

        Args:
            config: Configuration for the client. Uses defaults if not provided.
        """
        self.config = config or GeminiConfig()
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get or create an HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                headers={"Content-Type": "application/json"},
            )
        try:
            yield self._client
        except Exception as e:
            logger.error(f"HTTP client error: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _build_request_body(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build the request body for Gemini API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            **kwargs: Additional generation parameters

        Returns:
            Request body dictionary
        """
        # Prepare contents
        contents = []

        # Add system prompt if provided
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append(
                {
                    "role": "model",
                    "parts": [{"text": "Understood. I'll follow these instructions."}],
                }
            )

        # Add user prompt
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        # Build generation config
        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "topP": kwargs.get("top_p", self.config.top_p),
            "topK": kwargs.get("top_k", self.config.top_k),
            "maxOutputTokens": kwargs.get("max_output_tokens", self.config.max_output_tokens),
            "candidateCount": kwargs.get("candidate_count", self.config.candidate_count),
        }

        if self.config.stop_sequences:
            generation_config["stopSequences"] = self.config.stop_sequences

        # Build safety settings
        safety_settings = []
        for category, threshold in self.config.safety_settings.items():
            safety_settings.append({"category": category.value, "threshold": threshold.value})

        return {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings,
        }

    async def _make_request(
        self,
        endpoint: str,
        data: dict[str, Any],
        stream: bool = False,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Make a request to the Gemini API with retry logic.

        Args:
            endpoint: API endpoint
            data: Request data
            stream: Whether to stream the response

        Returns:
            Response data or async generator for streaming
        """
        if not self.config.api_key:
            raise GeminiAuthenticationError("API key is required but not configured")

        url = f"{self.config.base_url}/models/{self.config.model.value}:{endpoint}"
        params = {"key": self.config.api_key}

        async with self._rate_limiter:
            for attempt in range(self.config.max_retries):
                try:
                    async with self._get_client() as client:
                        if stream:
                            # Return async generator for streaming
                            return self._stream_response(client, url, params, data)
                        else:
                            response = await client.post(url, params=params, json=data)
                            response.raise_for_status()
                            return response.json()

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        # Rate limit - exponential backoff
                        wait_time = 2**attempt
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    elif e.response.status_code == 401:
                        raise GeminiAuthenticationError("Invalid API key")
                    else:
                        logger.error(f"HTTP error: {e}")
                        raise GeminiException(f"API request failed: {e}")

                except Exception as e:
                    logger.error(f"Request error on attempt {attempt + 1}: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise GeminiException(
                            f"Failed after {self.config.max_retries} attempts: {e}"
                        )
                    await asyncio.sleep(1)

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str],
        data: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream response from Gemini API.

        Args:
            client: HTTP client
            url: API URL
            params: Query parameters
            data: Request data

        Yields:
            Response chunks
        """
        # Modify endpoint for streaming
        stream_url = url.replace(":generateContent", ":streamGenerateContent")

        async with client.stream("POST", stream_url, params=params, json=data) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        # Parse JSON response
                        chunk = json.loads(line)
                        yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming chunk: {line}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GeminiResponse:
        """Generate a response from the Gemini model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            **kwargs: Additional generation parameters

        Returns:
            GeminiResponse object
        """
        request_body = self._build_request_body(prompt, system_prompt, **kwargs)

        try:
            response = await self._make_request("generateContent", request_body)

            # Extract the generated text
            if response.get("candidates"):
                candidate = response["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                text = "".join(part.get("text", "") for part in parts)

                # Extract safety ratings
                safety_ratings = candidate.get("safetyRatings", [])

                # Extract token counts if available
                usage_metadata = response.get("usageMetadata", {})

                return GeminiResponse(
                    content=text,
                    model=self.config.model.value,
                    prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                    total_tokens=usage_metadata.get("totalTokenCount", 0),
                    finish_reason=candidate.get("finishReason"),
                    safety_ratings=safety_ratings,
                )
            else:
                raise GeminiException("No candidates in response")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream generated text from the Gemini model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        request_body = self._build_request_body(prompt, system_prompt, **kwargs)

        try:
            async for chunk in await self._make_request(
                "generateContent", request_body, stream=True
            ):
                if chunk.get("candidates"):
                    candidate = chunk["candidates"][0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            yield part["text"]

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        request_body = {"contents": [{"parts": [{"text": text}]}]}

        try:
            response = await self._make_request("countTokens", request_body)
            return response.get("totalTokens", 0)

        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            # Return estimate based on word count as fallback
            return len(text.split()) * 2  # Rough estimate

    async def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Generate embeddings for the given text.

        Args:
            text: Text to generate embeddings for
            task_type: Task type for embeddings (e.g., RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY)

        Returns:
            Embedding vector
        """
        # Note: This requires a specific embedding model like text-embedding-004
        embedding_model = "text-embedding-004"
        url = f"{self.config.base_url}/models/{embedding_model}:embedContent"
        params = {"key": self.config.api_key}

        request_body = {
            "model": f"models/{embedding_model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }

        try:
            async with self._get_client() as client:
                response = await client.post(url, params=params, json=request_body)
                response.raise_for_status()
                data = response.json()
                return data.get("embedding", {}).get("values", [])

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise GeminiException(f"Failed to generate embeddings: {e}")
