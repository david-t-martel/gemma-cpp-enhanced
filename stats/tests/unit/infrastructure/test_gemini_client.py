"""Comprehensive unit tests for Gemini API client infrastructure.

Tests all Gemini client functionality including API interactions, streaming,
error handling, retries, and rate limiting with proper mocking.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from src.infrastructure.gemini.client import (
    GeminiClient,
    GeminiConfig,
    GeminiException,
    GeminiModel,
    GeminiRateLimitError,
    GeminiResponse,
    HarmBlockThreshold,
    HarmCategory,
)


class TestGeminiConfig:
    """Test suite for GeminiConfig configuration class."""

    def test_config_default_initialization(self):
        """Test default configuration values."""
        config = GeminiConfig()

        assert config.model == GeminiModel.GEMINI_2_5_FLASH
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.max_output_tokens == 8192
        assert config.candidate_count == 1
        assert config.stop_sequences == []
        assert config.base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_config_custom_initialization(self):
        """Test configuration with custom values."""
        custom_safety = {
            HarmCategory.HARASSMENT: HarmBlockThreshold.BLOCK_HIGH_AND_ABOVE,
            HarmCategory.HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        }
        config = GeminiConfig(
            api_key="test-key",
            model=GeminiModel.GEMINI_PRO,
            temperature=0.9,
            stop_sequences=["END", "STOP"],
            safety_settings=custom_safety
        )

        assert config.api_key == "test-key"
        assert config.model == GeminiModel.GEMINI_PRO
        assert config.temperature == 0.9
        assert config.stop_sequences == ["END", "STOP"]
        assert config.safety_settings == custom_safety

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"})
    def test_config_api_key_from_environment(self):
        """Test API key retrieval from environment variable."""
        config = GeminiConfig()
        assert config.api_key == "env-key"

    @patch.dict("os.environ", {"HF_TOKEN": "hf-token"})
    def test_config_api_key_environment_fallback(self):
        """Test API key fallback from HF_TOKEN environment variable."""
        # Clear GOOGLE_API_KEY if it exists
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiConfig()
            # Should not find API key since GOOGLE_API_KEY is not set
            assert config.api_key is None

    def test_config_default_safety_settings(self):
        """Test default safety settings initialization."""
        config = GeminiConfig()

        expected_settings = {
            HarmCategory.HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        assert config.safety_settings == expected_settings


class TestGeminiResponse:
    """Test suite for GeminiResponse model."""

    def test_response_basic_creation(self):
        """Test basic response creation."""
        response = GeminiResponse(
            content="Hello, world!",
            model="gemini-pro",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )

        assert response.content == "Hello, world!"
        assert response.model == "gemini-pro"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15
        assert response.finish_reason is None
        assert response.safety_ratings is None

    def test_response_with_optional_fields(self):
        """Test response creation with optional fields."""
        safety_ratings = [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "LOW"}]
        response = GeminiResponse(
            content="Test content",
            model="gemini-2.5-pro",
            finish_reason="STOP",
            safety_ratings=safety_ratings
        )

        assert response.finish_reason == "STOP"
        assert response.safety_ratings == safety_ratings


class TestGeminiClient:
    """Test suite for GeminiClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = GeminiConfig(api_key="test-api-key")
        self.client = GeminiClient(self.config)

    def test_client_initialization_default_config(self):
        """Test client initialization with default configuration."""
        client = GeminiClient()
        assert client.config is not None
        assert isinstance(client.config, GeminiConfig)

    def test_client_initialization_custom_config(self):
        """Test client initialization with custom configuration."""
        assert self.client.config == self.config
        assert self.client._client is None
        assert self.client._rate_limiter._value == 10  # Default semaphore value

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as async context manager."""
        async with GeminiClient(self.config) as client:
            assert isinstance(client, GeminiClient)

        # Client should be closed after context exit
        assert client._client is None

    def test_build_request_body_basic(self):
        """Test basic request body building."""
        request_body = self.client._build_request_body("Hello, how are you?")

        assert "contents" in request_body
        assert len(request_body["contents"]) == 1
        assert request_body["contents"][0]["role"] == "user"
        assert request_body["contents"][0]["parts"][0]["text"] == "Hello, how are you?"

        assert "generationConfig" in request_body
        assert request_body["generationConfig"]["temperature"] == 0.7
        assert request_body["generationConfig"]["maxOutputTokens"] == 8192

        assert "safetySettings" in request_body
        assert len(request_body["safetySettings"]) == 4  # Four default safety categories

    def test_build_request_body_with_system_prompt(self):
        """Test request body building with system prompt."""
        request_body = self.client._build_request_body(
            "User question",
            system_prompt="You are a helpful assistant."
        )

        assert len(request_body["contents"]) == 3
        # System prompt
        assert request_body["contents"][0]["role"] == "user"
        assert request_body["contents"][0]["parts"][0]["text"] == "You are a helpful assistant."
        # Model acknowledgment
        assert request_body["contents"][1]["role"] == "model"
        assert "Understood" in request_body["contents"][1]["parts"][0]["text"]
        # User prompt
        assert request_body["contents"][2]["role"] == "user"
        assert request_body["contents"][2]["parts"][0]["text"] == "User question"

    def test_build_request_body_with_kwargs(self):
        """Test request body building with custom parameters."""
        request_body = self.client._build_request_body(
            "Test prompt",
            temperature=0.9,
            top_p=0.8,
            max_output_tokens=1000
        )

        config = request_body["generationConfig"]
        assert config["temperature"] == 0.9
        assert config["topP"] == 0.8
        assert config["maxOutputTokens"] == 1000

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        mock_response = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello! How can I help you?"}]},
                "finishReason": "STOP",
                "safetyRatings": []
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 8,
                "totalTokenCount": 13
            }
        }

        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_http_response = AsyncMock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            mock_client.post.return_value = mock_http_response
            mock_get_client.return_value.__aenter__.return_value = mock_client

            response = await self.client._make_request("generateContent", {"test": "data"})

            assert response == mock_response
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_authentication_error(self):
        """Test API request with authentication error."""
        self.client.config.api_key = None

        with pytest.raises(Exception) as exc_info:
            await self.client._make_request("generateContent", {"test": "data"})

        assert "API key is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_retry(self):
        """Test API request with rate limiting and retry."""
        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()

            # First call returns 429, second call succeeds
            rate_limit_response = AsyncMock()
            rate_limit_response.status_code = 429
            rate_limit_error = httpx.HTTPStatusError(
                "Rate limited", request=Mock(), response=rate_limit_response
            )

            success_response = AsyncMock()
            success_response.raise_for_status.return_value = None
            success_response.json.return_value = {"success": True}

            mock_client.post.side_effect = [rate_limit_error, success_response]
            mock_get_client.return_value.__aenter__.return_value = mock_client

            with patch('asyncio.sleep') as mock_sleep:
                response = await self.client._make_request("generateContent", {"test": "data"})

                assert response == {"success": True}
                assert mock_client.post.call_count == 2
                mock_sleep.assert_called_once_with(2)  # Exponential backoff starts at 2^0 = 1, but first retry is 2^1 = 2

    @pytest.mark.asyncio
    async def test_make_request_http_error(self):
        """Test API request with HTTP error."""
        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            http_response = AsyncMock()
            http_response.status_code = 500
            http_error = httpx.HTTPStatusError(
                "Server error", request=Mock(), response=http_response
            )
            mock_client.post.side_effect = http_error
            mock_get_client.return_value.__aenter__.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await self.client._make_request("generateContent", {"test": "data"})

            assert "API request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful text generation."""
        mock_api_response = {
            "candidates": [{
                "content": {"parts": [{"text": "Generated response"}]},
                "finishReason": "STOP",
                "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "LOW"}]
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 15,
                "totalTokenCount": 25
            }
        }

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_api_response

            response = await self.client.generate("Test prompt")

            assert isinstance(response, GeminiResponse)
            assert response.content == "Generated response"
            assert response.model == self.config.model.value
            assert response.prompt_tokens == 10
            assert response.completion_tokens == 15
            assert response.total_tokens == 25
            assert response.finish_reason == "STOP"
            assert response.safety_ratings == [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "LOW"}]

    @pytest.mark.asyncio
    async def test_generate_no_candidates(self):
        """Test generation when API returns no candidates."""
        mock_api_response = {"candidates": []}

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_api_response

            with pytest.raises(Exception) as exc_info:
                await self.client.generate("Test prompt")

            assert "No candidates in response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        mock_api_response = {
            "candidates": [{
                "content": {"parts": [{"text": "System-guided response"}]},
                "finishReason": "STOP"
            }]
        }

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_api_response

            response = await self.client.generate(
                "User question",
                system_prompt="You are a helpful coding assistant."
            )

            assert response.content == "System-guided response"
            # Verify the request body was built with system prompt
            call_args = mock_request.call_args[0]
            assert call_args[0] == "generateContent"

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming text generation."""
        async def mock_stream_response():
            chunks = [
                {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": " world"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "!"}]}}]}
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_stream_response()

            chunks = []
            async for chunk in self.client.generate_stream("Test prompt"):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_empty_chunks(self):
        """Test streaming with empty or malformed chunks."""
        async def mock_stream_response():
            chunks = [
                {"candidates": [{"content": {"parts": [{"text": "Valid"}]}}]},
                {"candidates": []},  # Empty candidates
                {"candidates": [{"content": {"parts": []}}]},  # Empty parts
                {"candidates": [{"content": {"parts": [{"text": "Another valid"}]}}]}
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_stream_response()

            chunks = []
            async for chunk in self.client.generate_stream("Test prompt"):
                chunks.append(chunk)

            assert chunks == ["Valid", "Another valid"]

    @pytest.mark.asyncio
    async def test_count_tokens_success(self):
        """Test successful token counting."""
        mock_response = {"totalTokens": 42}

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_response

            token_count = await self.client.count_tokens("Test text for counting")

            assert token_count == 42
            mock_request.assert_called_once_with(
                "countTokens",
                {"contents": [{"parts": [{"text": "Test text for counting"}]}]}
            )

    @pytest.mark.asyncio
    async def test_count_tokens_error_fallback(self):
        """Test token counting with error fallback."""
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = Exception("API error")

            token_count = await self.client.count_tokens("Test text for counting")

            # Should fall back to word-based estimation
            assert token_count == 8  # 4 words * 2

    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        """Test successful text embedding generation."""
        mock_response = {
            "embedding": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }

        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_http_response = AsyncMock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            mock_client.post.return_value = mock_http_response
            mock_get_client.return_value.__aenter__.return_value = mock_client

            embeddings = await self.client.embed_text("Test text")

            assert embeddings == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text_with_custom_task_type(self):
        """Test text embedding with custom task type."""
        mock_response = {"embedding": {"values": [0.1, 0.2]}}

        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_http_response = AsyncMock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            mock_client.post.return_value = mock_http_response
            mock_get_client.return_value.__aenter__.return_value = mock_client

            embeddings = await self.client.embed_text("Query text", task_type="RETRIEVAL_QUERY")

            assert embeddings == [0.1, 0.2]

            # Verify the request was made with correct task type
            call_args = mock_client.post.call_args
            request_body = call_args[1]["json"]
            assert request_body["taskType"] == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_embed_text_error(self):
        """Test text embedding with API error."""
        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Embedding API error")
            mock_get_client.return_value.__aenter__.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await self.client.embed_text("Test text")

            assert "Failed to generate embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_response_json_parsing(self):
        """Test streaming response JSON parsing."""
        mock_lines = [
            '{"candidates": [{"content": {"parts": [{"text": "chunk1"}]}}]}',
            'invalid json line',  # Should be skipped
            '{"candidates": [{"content": {"parts": [{"text": "chunk2"}]}}]}'
        ]

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines.return_value = mock_aiter_lines()

        with patch('src.infrastructure.gemini.client.logger') as mock_logger:
            chunks = []
            async for chunk in self.client._stream_response(
                mock_client, "test-url", {}, {}
            ):
                chunks.append(chunk)

            # Should have received 2 valid chunks and logged 1 warning
            assert len(chunks) == 2
            assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "chunk1"
            assert chunks[1]["candidates"][0]["content"]["parts"][0]["text"] == "chunk2"
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_requests(self):
        """Test rate limiting with concurrent requests."""
        # Reduce semaphore for testing
        self.client._rate_limiter = asyncio.Semaphore(2)

        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"success": True}
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__aenter__.return_value = mock_client

            # Start 5 concurrent requests (more than semaphore limit)
            tasks = [
                self.client._make_request("generateContent", {"test": f"data{i}"})
                for i in range(5)
            ]

            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert len(responses) == 5
            assert all(r["success"] for r in responses)

    def test_exception_classes(self):
        """Test custom exception classes."""
        # Test base exception
        base_exc = GeminiException("Base error")
        assert str(base_exc) == "Base error"
        assert isinstance(base_exc, Exception)

        # Test rate limit exception
        rate_exc = GeminiRateLimitError("Rate limited")
        assert str(rate_exc) == "Rate limited"
        assert isinstance(rate_exc, GeminiException)


class TestGeminiEnums:
    """Test enum classes for Gemini configuration."""

    def test_gemini_model_enum(self):
        """Test GeminiModel enum values."""
        assert GeminiModel.GEMINI_PRO == "gemini-pro"
        assert GeminiModel.GEMINI_PRO_VISION == "gemini-pro-vision"
        assert GeminiModel.GEMINI_2_5_PRO == "gemini-2.5-pro"
        assert GeminiModel.GEMINI_2_5_FLASH == "gemini-2.5-flash"

    def test_harm_category_enum(self):
        """Test HarmCategory enum values."""
        assert HarmCategory.HARASSMENT == "HARM_CATEGORY_HARASSMENT"
        assert HarmCategory.HATE_SPEECH == "HARM_CATEGORY_HATE_SPEECH"
        assert HarmCategory.SEXUALLY_EXPLICIT == "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        assert HarmCategory.DANGEROUS_CONTENT == "HARM_CATEGORY_DANGEROUS_CONTENT"

    def test_harm_block_threshold_enum(self):
        """Test HarmBlockThreshold enum values."""
        assert HarmBlockThreshold.BLOCK_NONE == "BLOCK_NONE"
        assert HarmBlockThreshold.BLOCK_LOW_AND_ABOVE == "BLOCK_LOW_AND_ABOVE"
        assert HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE == "BLOCK_MEDIUM_AND_ABOVE"
        assert HarmBlockThreshold.BLOCK_HIGH_AND_ABOVE == "BLOCK_HIGH_AND_ABOVE"


class TestErrorConditions:
    """Test various error conditions and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = GeminiConfig(api_key="test-api-key")
        self.client = GeminiClient(self.config)

    @pytest.mark.asyncio
    async def test_client_close_before_use(self):
        """Test closing client before using it."""
        await self.client.close()
        # Should not raise an error
        assert self.client._client is None

    @pytest.mark.asyncio
    async def test_multiple_close_calls(self):
        """Test multiple close calls on client."""
        await self.client.close()
        await self.client.close()  # Should not raise error
        assert self.client._client is None

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        with patch.object(self.client, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            mock_get_client.return_value.__aenter__.return_value = mock_client

            with pytest.raises(Exception) as exc_info:
                await self.client._make_request("generateContent", {"test": "data"})

            assert "Failed after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_malformed_api_response(self):
        """Test handling of malformed API responses."""
        malformed_response = {
            "candidates": [{
                "content": {},  # Missing parts
                "finishReason": "STOP"
            }]
        }

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = malformed_response

            response = await self.client.generate("Test prompt")
            # Should handle missing parts gracefully
            assert response.content == ""

    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        # Test with extreme values
        config = GeminiConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=1,
            timeout=0.1
        )

        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.max_output_tokens == 1
        assert config.timeout == 0.1

    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        mock_response = {
            "candidates": [{
                "content": {"parts": [{"text": "Please provide a prompt."}]},
                "finishReason": "STOP"
            }]
        }

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_response

            response = await self.client.generate("")
            assert response.content == "Please provide a prompt."

    @pytest.mark.asyncio
    async def test_very_long_prompt_handling(self):
        """Test handling of very long prompts."""
        long_prompt = "test " * 10000  # Very long prompt

        mock_response = {
            "candidates": [{
                "content": {"parts": [{"text": "Response to long prompt"}]},
                "finishReason": "STOP"
            }]
        }

        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.return_value = mock_response

            response = await self.client.generate(long_prompt)
            assert response.content == "Response to long prompt"