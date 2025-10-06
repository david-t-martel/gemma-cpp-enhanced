"""
Transport adapters implementing the Adapter pattern.
Each adapter converts transport-specific logic to a common interface.
"""

import asyncio
import json
import logging
import sys
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
import websockets

from .core.contracts import (
    ConnectionError,
    ITransportAdapter,
    RequestError,
    ServerError,
    TimeoutError,
)


class BaseTransportAdapter(ITransportAdapter):
    """Base class for transport adapters."""

    def __init__(self, timeout: float = 30.0, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._connected

    def _log_request(self, method: str, params: Dict[str, Any]):
        """Log request if debug is enabled."""
        if self.debug:
            self.logger.debug(f"Request: {method} with params: {params}")

    def _log_response(self, response: Any):
        """Log response if debug is enabled."""
        if self.debug:
            self.logger.debug(f"Response: {response}")


class HTTPAdapter(BaseTransportAdapter):
    """Adapter for HTTP transport."""

    def __init__(
        self, url: str, timeout: float = 30.0, debug: bool = False, headers: Optional[dict] = None
    ):
        super().__init__(timeout, debug)
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        """Connect to the server."""
        if self._connected:
            return

        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), headers=self.headers
            )

            # Test connection with health check
            async with self.session.get(f"{self.url}/health") as response:
                if response.status == 200:
                    self._connected = True
                    self.logger.info(f"Connected to HTTP server at {self.url}")
                else:
                    raise ConnectionError(f"Server returned status {response.status}")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection timeout to {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.url}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
        self.logger.info("Disconnected from HTTP server")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request to the server."""
        if not self._connected or not self.session:
            raise ConnectionError("Not connected to server")

        self._log_request(method, params)

        # Build JSON-RPC request
        request_data = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

        try:
            async with self.session.post(f"{self.url}/rpc", json=request_data) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ServerError(f"Server error {response.status}: {text}")

                data = await response.json()
                self._log_response(data)

                if "error" in data:
                    raise ServerError(f"RPC error: {data['error']}")

                return data.get("result")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timeout for {method}")
        except aiohttp.ClientError as e:
            raise RequestError(f"HTTP request failed: {e}")

    async def send_stream_request(
        self, method: str, params: Dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        """Send a streaming request to the server."""
        # For HTTP, we'll use Server-Sent Events or fallback to regular request
        params["stream"] = True
        result = await self.send_request(method, params)

        # Simulate streaming by yielding the result in chunks
        if isinstance(result, str):
            # Split into words for demo purposes
            words = result.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.01)  # Small delay to simulate streaming
        else:
            yield result


class WebSocketAdapter(BaseTransportAdapter):
    """Adapter for WebSocket transport."""

    def __init__(self, url: str, timeout: float = 30.0, debug: bool = False):
        super().__init__(timeout, debug)
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._request_id = 0

    async def connect(self) -> None:
        """Connect to the server."""
        if self._connected:
            return

        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.url), timeout=self.timeout
            )
            self._connected = True
            self.logger.info(f"Connected to WebSocket server at {self.url}")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection timeout to {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.url}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self._connected = False
        self.logger.info("Disconnected from WebSocket server")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request to the server."""
        if not self._connected or not self.websocket:
            raise ConnectionError("Not connected to server")

        self._log_request(method, params)

        # Build JSON-RPC request
        self._request_id += 1
        request_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            # Send request
            await self.websocket.send(json.dumps(request_data))

            # Wait for response with timeout
            response_text = await asyncio.wait_for(self.websocket.recv(), timeout=self.timeout)

            response = json.loads(response_text)
            self._log_response(response)

            if "error" in response:
                raise ServerError(f"RPC error: {response['error']}")

            return response.get("result")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timeout for {method}")
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            raise ConnectionError("WebSocket connection closed")
        except Exception as e:
            raise RequestError(f"WebSocket request failed: {e}")

    async def send_stream_request(
        self, method: str, params: Dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        """Send a streaming request to the server."""
        if not self._connected or not self.websocket:
            raise ConnectionError("Not connected to server")

        params["stream"] = True
        self._request_id += 1

        request_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            await self.websocket.send(json.dumps(request_data))

            # Receive streaming responses
            while True:
                try:
                    response_text = await asyncio.wait_for(
                        self.websocket.recv(), timeout=self.timeout
                    )

                    response = json.loads(response_text)

                    if "error" in response:
                        raise ServerError(f"Stream error: {response['error']}")

                    if "result" in response:
                        yield response["result"]

                        # Check if this is the final chunk
                        if response.get("final", False):
                            break

                except asyncio.TimeoutError:
                    # Timeout might mean stream ended
                    break

        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            raise ConnectionError("WebSocket connection closed")
        except Exception as e:
            raise RequestError(f"WebSocket stream failed: {e}")


class StdioAdapter(BaseTransportAdapter):
    """Adapter for stdio transport (for testing and CLI usage)."""

    def __init__(self, timeout: float = 30.0, debug: bool = False):
        super().__init__(timeout, debug)
        self._request_id = 0

    async def connect(self) -> None:
        """Connect to the server (no-op for stdio)."""
        self._connected = True
        self.logger.info("Stdio transport ready")

    async def disconnect(self) -> None:
        """Disconnect from the server (no-op for stdio)."""
        self._connected = False
        self.logger.info("Stdio transport closed")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request to the server via stdio."""
        if not self._connected:
            raise ConnectionError("Not connected")

        self._log_request(method, params)

        # Build JSON-RPC request
        self._request_id += 1
        request_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            # Write to stdout
            request_text = json.dumps(request_data)
            print(request_text, flush=True)

            # Read from stdin with timeout
            loop = asyncio.get_event_loop()
            response_text = await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline), timeout=self.timeout
            )

            if not response_text:
                raise ConnectionError("No response from server")

            response = json.loads(response_text.strip())
            self._log_response(response)

            if "error" in response:
                raise ServerError(f"RPC error: {response['error']}")

            return response.get("result")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timeout for {method}")
        except json.JSONDecodeError as e:
            raise RequestError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise RequestError(f"Stdio request failed: {e}")

    async def send_stream_request(
        self, method: str, params: Dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        """Send a streaming request to the server."""
        # For stdio, we'll simulate streaming
        params["stream"] = True
        result = await self.send_request(method, params)

        # Yield the result as a single chunk
        yield result
