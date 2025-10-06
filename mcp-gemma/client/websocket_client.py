"""
WebSocket MCP client for real-time streaming communication with Gemma server.
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .base_client import (
    BaseGemmaClient,
    GemmaClientError,
    GemmaConnectionError,
    GemmaTimeoutError,
    GenerationRequest,
    GenerationResponse,
    MemoryEntry,
)


class GemmaWebSocketClient(BaseGemmaClient):
    """WebSocket client for real-time MCP Gemma server communication."""

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        debug: bool = False,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
    ):
        super().__init__(timeout=timeout, debug=debug)

        self.url = url
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # Message handling
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.streaming_handlers: Dict[str, Callable] = {}

        # Background tasks
        self.listen_task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None

        self._setup_message_handlers()

    def _setup_message_handlers(self):
        """Setup message handlers for different message types."""
        self.message_handlers = {
            "welcome": self._handle_welcome,
            "tool_result": self._handle_tool_result,
            "tool_error": self._handle_tool_error,
            "generation_start": self._handle_generation_start,
            "generation_chunk": self._handle_generation_chunk,
            "generation_complete": self._handle_generation_complete,
            "generation_error": self._handle_generation_error,
            "pong": self._handle_pong,
            "error": self._handle_error,
        }

    async def connect(self) -> None:
        """Connect to the WebSocket server."""
        if self.websocket is not None:
            return

        try:
            self.websocket = await websockets.connect(
                self.url, ping_interval=self.ping_interval, ping_timeout=self.ping_timeout
            )

            # Start background tasks
            self.listen_task = asyncio.create_task(self._listen_loop())
            self.ping_task = asyncio.create_task(self._ping_loop())

            self.logger.info(f"Connected to WebSocket server at {self.url}")

        except Exception as e:
            await self.disconnect()
            raise GemmaConnectionError(f"Failed to connect to WebSocket server: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        # Cancel background tasks
        if self.listen_task and not self.listen_task.done():
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass

        if self.ping_task and not self.ping_task.done():
            self.ping_task.cancel()
            try:
                await self.ping_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

        # Cancel pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()

    async def _listen_loop(self):
        """Background task to listen for messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    self.logger.error(f"Received invalid JSON: {message}")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
        except ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Listen loop error: {e}")

    async def _ping_loop(self):
        """Background task to send periodic pings."""
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                if self.websocket and not self.websocket.closed:
                    await self._send_message({"type": "ping"})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Ping loop error: {e}")

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if self.websocket is None or self.websocket.closed:
            raise GemmaConnectionError("WebSocket not connected")

        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            raise GemmaConnectionError(f"Failed to send message: {e}")

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming message."""
        message_type = data.get("type")
        handler = self.message_handlers.get(message_type)

        if handler:
            await handler(data)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")

    async def _handle_welcome(self, data: Dict[str, Any]):
        """Handle welcome message."""
        self.logger.info(f"Received welcome from {data.get('server')} v{data.get('version')}")

    async def _handle_tool_result(self, data: Dict[str, Any]):
        """Handle tool result message."""
        request_id = data.get("request_id")
        if request_id and request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            if not future.done():
                future.set_result(data.get("result"))

    async def _handle_tool_error(self, data: Dict[str, Any]):
        """Handle tool error message."""
        request_id = data.get("request_id")
        if request_id and request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            if not future.done():
                error_msg = data.get("error", "Unknown error")
                future.set_exception(GemmaClientError(f"Tool error: {error_msg}"))

    async def _handle_generation_start(self, data: Dict[str, Any]):
        """Handle generation start message."""
        request_id = data.get("request_id")
        if request_id in self.streaming_handlers:
            handler = self.streaming_handlers[request_id]
            await handler("start", data)

    async def _handle_generation_chunk(self, data: Dict[str, Any]):
        """Handle generation chunk message."""
        request_id = data.get("request_id")
        if request_id in self.streaming_handlers:
            handler = self.streaming_handlers[request_id]
            chunk = data.get("chunk", "")
            await handler("chunk", chunk)

    async def _handle_generation_complete(self, data: Dict[str, Any]):
        """Handle generation complete message."""
        request_id = data.get("request_id")
        if request_id in self.streaming_handlers:
            handler = self.streaming_handlers[request_id]
            await handler("complete", data.get("result"))
            del self.streaming_handlers[request_id]

        # Also handle as regular result for non-streaming
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            if not future.done():
                future.set_result(data.get("result"))

    async def _handle_generation_error(self, data: Dict[str, Any]):
        """Handle generation error message."""
        request_id = data.get("request_id")
        error_msg = data.get("error", "Generation error")

        if request_id in self.streaming_handlers:
            handler = self.streaming_handlers[request_id]
            await handler("error", error_msg)
            del self.streaming_handlers[request_id]

        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            if not future.done():
                future.set_exception(GemmaClientError(error_msg))

    async def _handle_pong(self, data: Dict[str, Any]):
        """Handle pong message."""
        self.logger.debug("Received pong")

    async def _handle_error(self, data: Dict[str, Any]):
        """Handle error message."""
        error_msg = data.get("message", "Unknown error")
        self.logger.error(f"Server error: {error_msg}")

    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the server."""
        self._validate_request(request)
        self._log_request(request)

        start_time = time.time()
        request_id = str(uuid.uuid4())

        message = {"type": "generate_text", "request_id": request_id, "prompt": request.prompt}

        if request.max_tokens is not None:
            message["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            message["temperature"] = request.temperature

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            await self._send_message(message)

            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=self.timeout)

            response_time = time.time() - start_time

            response = GenerationResponse(
                text=str(result),
                response_time=response_time,
                tokens_generated=len(str(result).split()),
            )

            self._log_response(response)
            return response

        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise GemmaTimeoutError(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise

    async def generate_text_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with real-time streaming."""
        self._validate_request(request)
        self._log_request(request)

        request_id = str(uuid.uuid4())

        message = {
            "type": "generate_text",
            "request_id": request_id,
            "prompt": request.prompt,
            "stream": True,
        }

        if request.max_tokens is not None:
            message["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            message["temperature"] = request.temperature

        # Setup streaming handler
        chunks = []
        stream_complete = asyncio.Event()
        stream_error = None

        async def stream_handler(event_type: str, data: Any):
            nonlocal stream_error

            if event_type == "start":
                pass  # Generation started
            elif event_type == "chunk":
                chunks.append(data)
            elif event_type == "complete":
                stream_complete.set()
            elif event_type == "error":
                stream_error = data
                stream_complete.set()

        self.streaming_handlers[request_id] = stream_handler

        try:
            await self._send_message(message)

            # Yield chunks as they arrive
            last_yielded = 0
            while not stream_complete.is_set():
                # Yield any new chunks
                while last_yielded < len(chunks):
                    yield chunks[last_yielded]
                    last_yielded += 1

                # Wait a bit before checking again
                try:
                    await asyncio.wait_for(stream_complete.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

            # Yield any remaining chunks
            while last_yielded < len(chunks):
                yield chunks[last_yielded]
                last_yielded += 1

            # Check for errors
            if stream_error:
                raise GemmaClientError(f"Streaming error: {stream_error}")

        except Exception as e:
            self.streaming_handlers.pop(request_id, None)
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server."""
        request_id = str(uuid.uuid4())

        message = {
            "type": "call_tool",
            "request_id": request_id,
            "tool_name": tool_name,
            "arguments": arguments,
        }

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            await self._send_message(message)
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise GemmaTimeoutError(f"Tool call timed out after {self.timeout} seconds")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise

    async def switch_model(self, model_path: str, tokenizer_path: Optional[str] = None) -> bool:
        """Switch to a different model."""
        arguments = {"model_path": model_path}
        if tokenizer_path:
            arguments["tokenizer_path"] = tokenizer_path

        try:
            result = await self.call_tool("switch_model", arguments)
            self.logger.info(f"Model switched: {result}")
            return True
        except Exception as e:
            self.logger.error(f"Model switch failed: {e}")
            return False

    async def store_memory(self, entry: MemoryEntry) -> str:
        """Store content in memory."""
        arguments = {"key": entry.key, "content": entry.content}
        if entry.metadata:
            arguments["metadata"] = entry.metadata

        try:
            result = await self.call_tool("store_memory", arguments)
            return str(result)
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise

    async def retrieve_memory(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve content from memory."""
        try:
            result = await self.call_tool("retrieve_memory", {"key": key})

            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            if "error" in data:
                return None

            return MemoryEntry(
                key=key,
                content=data.get("content", ""),
                metadata=data.get("metadata", {}),
                timestamp=data.get("timestamp"),
                id=data.get("id"),
            )
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory: {e}")
            return None

    async def search_memory(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory by content."""
        try:
            result = await self.call_tool("search_memory", {"query": query, "limit": limit})

            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            entries = []
            for item in data.get("results", []):
                entry = MemoryEntry(
                    key=item.get("key", ""),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                    timestamp=item.get("timestamp"),
                )
                entries.append(entry)

            return entries
        except Exception as e:
            self.logger.error(f"Failed to search memory: {e}")
            return []

    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        try:
            result = await self.call_tool("get_metrics", {})
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Send ping and wait for pong
            await self._send_message({"type": "ping"})
            await asyncio.sleep(0.1)  # Wait for pong

            return {
                "status": "healthy",
                "websocket_connected": not self.websocket.closed if self.websocket else False,
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}
