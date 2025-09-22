"""API data models for requests and responses with OpenAI compatibility.

This module defines Pydantic models that ensure type safety, validation,
and OpenAI API compatibility for all server endpoints.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class ChatRole(str, Enum):
    """Chat message roles compatible with OpenAI API."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class FinishReason(str, Enum):
    """Reasons for completion finish."""

    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    NULL = "null"


# Request Models


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: ChatRole = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")
    name: str | None = Field(None, description="Optional name of the message author")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate message content is not empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions endpoint."""

    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    model: str | None = Field("gemma-2b-it", description="ID of the model to use")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum number of tokens to generate")
    stream: bool = Field(False, description="Whether to stream responses")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float | None = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: dict[str, float] | None = Field(None, description="Token logit bias")
    user: str | None = Field(None, description="Unique identifier for the end-user")
    seed: int | None = Field(None, description="Random seed for deterministic outputs")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Validate messages list is not empty."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name."""
        if v and not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class CompletionRequest(BaseModel):
    """Request model for text completions endpoint."""

    prompt: str | list[str] = Field(..., description="The prompt to complete")
    model: str | None = Field("gemma-2b-it", description="ID of the model to use")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int | None = Field(16, ge=1, description="Maximum number of tokens to generate")
    stream: bool = Field(False, description="Whether to stream responses")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float | None = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: dict[str, float] | None = Field(None, description="Token logit bias")
    user: str | None = Field(None, description="Unique identifier for the end-user")
    seed: int | None = Field(None, description="Random seed for deterministic outputs")
    suffix: str | None = Field(None, description="Suffix for completion")
    echo: bool = Field(False, description="Whether to echo the prompt")
    best_of: int | None = Field(1, ge=1, description="Number of completions to generate")
    n: int = Field(1, ge=1, le=128, description="Number of completions to return")
    logprobs: int | None = Field(None, ge=0, le=5, description="Include log probabilities")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str | list[str]) -> str | list[str]:
        """Validate prompt is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Prompt cannot be empty")
        elif isinstance(v, list) and not v:
            raise ValueError("Prompt list cannot be empty")
        return v


class EmbeddingsRequest(BaseModel):
    """Request model for embeddings endpoint."""

    input: str | list[str] = Field(..., description="Input text to embed")
    model: str | None = Field("text-embedding-ada-002", description="ID of the model to use")
    encoding_format: str = Field("float", description="Format of the embeddings")
    user: str | None = Field(None, description="Unique identifier for the end-user")

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str | list[str]) -> str | list[str]:
        """Validate input is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Input cannot be empty")
        elif isinstance(v, list) and not v:
            raise ValueError("Input list cannot be empty")
        return v


# Response Models


class TokenUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int | None = Field(None, description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class ChatChoice(BaseModel):
    """A choice in a chat completion response."""

    index: int = Field(..., description="The index of the choice")
    message: ChatMessage = Field(..., description="The message content")
    finish_reason: FinishReason | None = Field(None, description="Reason for completion finish")


class CompletionChoice(BaseModel):
    """A choice in a text completion response."""

    text: str = Field(..., description="The completion text")
    index: int = Field(..., description="The index of the choice")
    logprobs: dict[str, Any] | None = Field(None, description="Log probabilities")
    finish_reason: FinishReason | None = Field(None, description="Reason for completion finish")


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:29]}",
        description="Unique completion ID",
    )
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(
        default_factory=lambda: int(datetime.utcnow().timestamp()), description="Unix timestamp"
    )
    model: str = Field(..., description="The model used for completion")
    choices: list[ChatChoice] = Field(..., description="List of completion choices")
    usage: TokenUsage = Field(..., description="Token usage information")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


class CompletionResponse(BaseModel):
    """Response model for text completions."""

    id: str = Field(
        default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:29]}", description="Unique completion ID"
    )
    object: str = Field("text_completion", description="Object type")
    created: int = Field(
        default_factory=lambda: int(datetime.utcnow().timestamp()), description="Unix timestamp"
    )
    model: str = Field(..., description="The model used for completion")
    choices: list[CompletionChoice] = Field(..., description="List of completion choices")
    usage: TokenUsage = Field(..., description="Token usage information")


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completions."""

    id: str = Field(..., description="Unique completion ID")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="The model used for completion")
    choices: list[dict[str, Any]] = Field(..., description="List of completion choices")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


class CompletionChunk(BaseModel):
    """Streaming chunk for text completions."""

    id: str = Field(..., description="Unique completion ID")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="The model used for completion")
    choices: list[dict[str, Any]] = Field(..., description="List of completion choices")


class Embedding(BaseModel):
    """An embedding vector."""

    object: str = Field("embedding", description="Object type")
    index: int = Field(..., description="The index of the embedding")
    embedding: list[float] = Field(..., description="The embedding vector")


class EmbeddingsResponse(BaseModel):
    """Response model for embeddings."""

    object: str = Field("list", description="Object type")
    data: list[Embedding] = Field(..., description="List of embeddings")
    model: str = Field(..., description="The model used for embeddings")
    usage: TokenUsage = Field(..., description="Token usage information")


class ModelInfo(BaseModel):
    """Information about a model."""

    id: str = Field(..., description="Model ID")
    object: str = Field("model", description="Object type")
    created: int | None = Field(None, description="Unix timestamp of creation")
    owned_by: str = Field("gemma-chatbot", description="Organization that owns the model")
    permission: list[dict[str, Any]] = Field(default_factory=list, description="Model permissions")
    root: str | None = Field(None, description="Root model ID")
    parent: str | None = Field(None, description="Parent model ID")


class ModelsResponse(BaseModel):
    """Response model for listing models."""

    object: str = Field("list", description="Object type")
    data: list[ModelInfo] = Field(..., description="List of available models")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: dict[str, Any] = Field(..., description="Error details")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: dict[str, float] = Field(..., description="Memory usage statistics")
    performance_stats: dict[str, Any] = Field(..., description="Performance statistics")


class MetricsResponse(BaseModel):
    """Metrics response model."""

    requests_total: int = Field(..., description="Total number of requests")
    requests_per_second: float = Field(..., description="Current requests per second")
    average_response_time: float = Field(..., description="Average response time in seconds")
    active_connections: int = Field(..., description="Number of active connections")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    model_info: dict[str, Any] = Field(..., description="Model information")
    system_info: dict[str, Any] = Field(..., description="System information")


# WebSocket Models


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str = Field(..., description="Message type")
    session_id: str | None = Field(None, description="Session identifier")
    data: dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class WebSocketChatMessage(BaseModel):
    """WebSocket chat message model."""

    type: str = Field("chat", description="Message type")
    session_id: str = Field(..., description="Chat session identifier")
    message: str = Field(..., description="Chat message content")
    stream: bool = Field(True, description="Whether to stream the response")
    user_id: str | None = Field(None, description="User identifier")


class WebSocketResponse(BaseModel):
    """WebSocket response model."""

    type: str = Field(..., description="Response type")
    session_id: str = Field(..., description="Session identifier")
    data: dict[str, Any] = Field(..., description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    error: str | None = Field(None, description="Error message if any")


# Server-Sent Events Models


class SSEEvent(BaseModel):
    """Server-Sent Event model."""

    event: str | None = Field(None, description="Event type")
    data: str = Field(..., description="Event data")
    id: str | None = Field(None, description="Event ID")
    retry: int | None = Field(None, description="Retry timeout in milliseconds")

    def format(self) -> str:
        """Format as SSE event string."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        if self.event:
            lines.append(f"event: {self.event}")
        if self.retry:
            lines.append(f"retry: {self.retry}")

        # Split data into multiple lines if necessary
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")

        return "\n".join(lines) + "\n\n"
