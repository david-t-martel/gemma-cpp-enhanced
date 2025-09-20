"""Chat API endpoints with REST and WebSocket support.

This module implements OpenAI-compatible chat completion endpoints
with streaming support via Server-Sent Events and WebSocket connections.
"""

import asyncio
import json
import time
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.shared.logging import get_logger

from ...application.inference.service import InferenceService
from ...domain.models.chat import ChatSession
from ...domain.models.chat import MessageRole as DomainMessageRole
from ...shared.exceptions import InferenceException
from ...shared.exceptions import ValidationException
from ..state import get_inference_service
from .schemas import ChatChoice
from .schemas import ChatCompletionChunk
from .schemas import ChatCompletionRequest
from .schemas import ChatCompletionResponse
from .schemas import ChatMessage
from .schemas import CompletionChoice
from .schemas import CompletionChunk
from .schemas import CompletionRequest
from .schemas import CompletionResponse
from .schemas import FinishReason
from .schemas import SSEEvent
from .schemas import TokenUsage

logger = get_logger(__name__)
chat_router = APIRouter()


def get_inference_service_dependency() -> InferenceService:
    """Dependency to get the inference service."""
    service = get_inference_service()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not available",
        )
    return service


def convert_to_domain_role(role: str) -> DomainMessageRole:
    """Convert API role to domain role."""
    role_mapping = {
        "system": DomainMessageRole.SYSTEM,
        "user": DomainMessageRole.USER,
        "assistant": DomainMessageRole.ASSISTANT,
        "function": DomainMessageRole.FUNCTION,
    }
    return role_mapping.get(role, DomainMessageRole.USER)


def create_session_from_messages(messages: list[ChatMessage]) -> ChatSession:
    """Create a chat session from API messages."""
    session = ChatSession()

    # Set system prompt if first message is system
    if messages and messages[0].role == "system":
        session.system_prompt = messages[0].content
        messages = messages[1:]  # Remove system message from conversation

    # Add remaining messages
    for msg in messages:
        domain_role = convert_to_domain_role(msg.role)
        session.add_message(
            {
                "role": domain_role,
                "content": msg.content,
            }
        )

    return session


@chat_router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        200: {"description": "Successful completion"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
):
    """Create a chat completion, optionally streaming the response.

    This endpoint is compatible with OpenAI's chat completions API.
    """
    logger.info(f"Chat completion request: model={request.model}, stream={request.stream}")

    try:
        # Create session from messages
        session = create_session_from_messages(request.messages)

        # Extract generation parameters
        generation_params = {}
        if request.temperature is not None:
            generation_params["temperature"] = request.temperature
        if request.top_p is not None:
            generation_params["top_p"] = request.top_p
        if request.top_k is not None:
            generation_params["top_k"] = request.top_k
        if request.max_tokens is not None:
            generation_params["max_length"] = request.max_tokens

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(session, inference_service, request, generation_params),
                media_type="text/plain",
            )
        else:
            return await generate_chat_completion(
                session, inference_service, request, generation_params
            )

    except ValidationException as e:
        logger.warning(f"Validation error in chat completion: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except InferenceException as e:
        logger.error(f"Inference error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@chat_router.post(
    "/completions",
    response_model=CompletionResponse,
    responses={
        200: {"description": "Successful completion"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    },
)
async def create_completion(
    request: CompletionRequest,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
):
    """Create a text completion.

    This endpoint is compatible with OpenAI's completions API.
    """
    logger.info(f"Completion request: model={request.model}, stream={request.stream}")

    try:
        # Convert prompt to chat format
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_completion(prompts[0], inference_service, request),
                media_type="text/plain",
            )
        else:
            return await generate_completion(prompts[0], inference_service, request)

    except ValidationException as e:
        logger.warning(f"Validation error in completion: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except InferenceException as e:
        logger.error(f"Inference error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@chat_router.get("/chat/completions/stream")
async def stream_chat_completions_sse(
    request: Request,
    inference_service: InferenceService = Depends(get_inference_service_dependency),
):
    """Server-Sent Events endpoint for streaming chat completions."""
    return EventSourceResponse(sse_chat_completion_generator(request, inference_service))


async def generate_chat_completion(
    session: ChatSession,
    inference_service: InferenceService,
    request: ChatCompletionRequest,
    generation_params: dict[str, Any],
) -> ChatCompletionResponse:
    """Generate a non-streaming chat completion response."""
    start_time = time.time()

    # Get the last user message
    last_message = session.get_last_message()
    if not last_message or last_message.role != DomainMessageRole.USER:
        raise ValidationException("Last message must be from user")

    # Generate response
    response_message = await inference_service.generate_response(
        session, last_message.content, **generation_params
    )

    # Create API response
    completion_response = ChatCompletionResponse(
        model=request.model or "gemma-2b-it",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response_message.content,
                ),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage=TokenUsage(
            prompt_tokens=(
                response_message.token_usage.prompt_tokens if response_message.token_usage else 0
            ),
            completion_tokens=(
                response_message.token_usage.completion_tokens
                if response_message.token_usage
                else 0
            ),
            total_tokens=(
                response_message.token_usage.total_tokens if response_message.token_usage else 0
            ),
        ),
    )

    processing_time = time.time() - start_time
    logger.info(f"Generated chat completion in {processing_time:.2f}s")

    return completion_response


async def stream_chat_completion(
    session: ChatSession,
    inference_service: InferenceService,
    request: ChatCompletionRequest,
    generation_params: dict[str, Any],
):
    """Generate a streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    # Get the last user message
    last_message = session.get_last_message()
    if not last_message or last_message.role != DomainMessageRole.USER:
        raise ValidationException("Last message must be from user")

    try:
        # Generate streaming response
        async for chunk in inference_service.generate_streaming_response(
            session, last_message.content, **generation_params
        ):
            # Create streaming chunk
            chunk_data = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model or "gemma-2b-it",
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": chunk.content} if chunk.content else {},
                        "finish_reason": "stop" if chunk.is_complete else None,
                    }
                ],
            )

            yield f"data: {chunk_data.json()}\n\n"

        # Send final chunk
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model or "gemma-2b-it",
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        )
        yield f"data: {final_chunk.json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming chat completion: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def generate_completion(
    prompt: str,
    inference_service: InferenceService,
    request: CompletionRequest,
) -> CompletionResponse:
    """Generate a non-streaming text completion response."""
    start_time = time.time()

    # Create session with user message
    session = ChatSession()
    session.add_user_message(prompt)

    # Extract generation parameters
    generation_params = {}
    if request.temperature is not None:
        generation_params["temperature"] = request.temperature
    if request.top_p is not None:
        generation_params["top_p"] = request.top_p
    if request.max_tokens is not None:
        generation_params["max_length"] = request.max_tokens

    # Generate response
    response_message = await inference_service.generate_response(
        session, prompt, **generation_params
    )

    # Create API response
    completion_response = CompletionResponse(
        model=request.model or "gemma-2b-it",
        choices=[
            CompletionChoice(
                text=response_message.content,
                index=0,
                finish_reason=FinishReason.STOP,
            )
        ],
        usage=TokenUsage(
            prompt_tokens=(
                response_message.token_usage.prompt_tokens if response_message.token_usage else 0
            ),
            completion_tokens=(
                response_message.token_usage.completion_tokens
                if response_message.token_usage
                else 0
            ),
            total_tokens=(
                response_message.token_usage.total_tokens if response_message.token_usage else 0
            ),
        ),
    )

    processing_time = time.time() - start_time
    logger.info(f"Generated completion in {processing_time:.2f}s")

    return completion_response


async def stream_completion(
    prompt: str,
    inference_service: InferenceService,
    request: CompletionRequest,
):
    """Generate a streaming text completion response."""
    completion_id = f"cmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    # Create session with user message
    session = ChatSession()
    session.add_user_message(prompt)

    # Extract generation parameters
    generation_params = {}
    if request.temperature is not None:
        generation_params["temperature"] = request.temperature
    if request.top_p is not None:
        generation_params["top_p"] = request.top_p
    if request.max_tokens is not None:
        generation_params["max_length"] = request.max_tokens

    try:
        # Generate streaming response
        async for chunk in inference_service.generate_streaming_response(
            session, prompt, **generation_params
        ):
            # Create streaming chunk
            chunk_data = CompletionChunk(
                id=completion_id,
                created=created,
                model=request.model or "gemma-2b-it",
                choices=[
                    {
                        "text": chunk.content,
                        "index": 0,
                        "finish_reason": "stop" if chunk.is_complete else None,
                    }
                ],
            )

            yield f"data: {chunk_data.json()}\n\n"

        # Send final chunk
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming completion: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def sse_chat_completion_generator(
    request: Request,
    inference_service: InferenceService,
):
    """Server-Sent Events generator for chat completions."""
    try:
        # Parse request body
        body = await request.body()
        request_data = json.loads(body)
        chat_request = ChatCompletionRequest(**request_data)

        # Create session from messages
        session = create_session_from_messages(chat_request.messages)

        # Extract generation parameters
        generation_params = {}
        if chat_request.temperature is not None:
            generation_params["temperature"] = chat_request.temperature
        if chat_request.top_p is not None:
            generation_params["top_p"] = chat_request.top_p
        if chat_request.max_tokens is not None:
            generation_params["max_length"] = chat_request.max_tokens

        # Get the last user message
        last_message = session.get_last_message()
        if not last_message or last_message.role != DomainMessageRole.USER:
            yield SSEEvent(
                event="error",
                data=json.dumps({"error": "Last message must be from user"}),
            ).format()
            return

        # Generate streaming response
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        async for chunk in inference_service.generate_streaming_response(
            session, last_message.content, **generation_params
        ):
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_request.model or "gemma-2b-it",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.content} if chunk.content else {},
                        "finish_reason": "stop" if chunk.is_complete else None,
                    }
                ],
            }

            yield SSEEvent(
                event="chunk",
                data=json.dumps(chunk_data),
                id=completion_id,
            ).format()

            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected from SSE stream")
                break

        # Send completion event
        yield SSEEvent(
            event="done",
            data="[DONE]",
            id=completion_id,
        ).format()

    except Exception as e:
        logger.error(f"Error in SSE chat completion: {e}")
        yield SSEEvent(
            event="error",
            data=json.dumps({"error": str(e)}),
        ).format()
