"""
HTTP MCP client for REST API communication with Gemma server.
"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from .base_client import (
    BaseGemmaClient,
    GemmaClientError,
    GemmaConnectionError,
    GemmaServerError,
    GemmaTimeoutError,
    GenerationRequest,
    GenerationResponse,
    MemoryEntry,
)


class GemmaHTTPClient(BaseGemmaClient):
    """HTTP client for MCP Gemma server."""

    def __init__(
        self, base_url: str, timeout: float = 30.0, debug: bool = False, max_connections: int = 10
    ):
        super().__init__(timeout=timeout, debug=debug)

        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_connections = max_connections

    async def connect(self) -> None:
        """Create HTTP session."""
        if self.session is not None:
            return

        # Configure connection limits
        connector = aiohttp.TCPConnector(
            limit=self.max_connections, limit_per_host=self.max_connections
        )

        # Configure timeout
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Test connection with health check
        try:
            await self._health_check_request()
            self.logger.info(f"Connected to MCP server at {self.base_url}")
        except Exception as e:
            await self.disconnect()
            raise GemmaConnectionError(f"Failed to connect to server: {e}")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make HTTP request to the server."""
        if self.session is None:
            raise GemmaConnectionError("Not connected to server")

        url = f"{self.base_url}{path}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status >= 500:
                    error_text = await response.text()
                    raise GemmaServerError(f"Server error ({response.status}): {error_text}")
                else:
                    error_text = await response.text()
                    raise GemmaClientError(f"Request failed ({response.status}): {error_text}")

        except aiohttp.ClientError as e:
            raise GemmaConnectionError(f"Connection error: {e}")
        except asyncio.TimeoutError:
            raise GemmaTimeoutError(f"Request timed out after {self.timeout} seconds")

    async def _health_check_request(self) -> Dict[str, Any]:
        """Internal health check request."""
        return await self._request("GET", "/health")

    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the server."""
        self._validate_request(request)
        self._log_request(request)

        start_time = time.time()

        data = {"prompt": request.prompt, "stream": False}

        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            data["temperature"] = request.temperature

        try:
            result = await self._request("POST", "/generate", json=data)

            text = result.get("text", "")
            response_time = time.time() - start_time

            response = GenerationResponse(
                text=text,
                response_time=response_time,
                tokens_generated=len(text.split()),  # Rough estimate
                metadata=result.get("metadata"),
            )

            self._log_response(response)
            return response

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

    async def generate_text_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        self._validate_request(request)
        self._log_request(request)

        data = {"prompt": request.prompt, "stream": True}

        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            data["temperature"] = request.temperature

        # For HTTP, we simulate streaming by chunking the response
        # In a real implementation, this would use Server-Sent Events or chunked transfer
        response = await self.generate_text(
            GenerationRequest(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )
        )

        # Simulate streaming by yielding chunks
        chunk_size = 5
        words = response.text.split()

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i : i + chunk_size]
            chunk = " ".join(chunk_words)
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.05)  # Small delay to simulate streaming

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on the server."""
        try:
            result = await self._request(
                "POST", f"/tools/{tool_name}/call", json={"arguments": arguments}
            )
            return result.get("result")
        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
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

    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        try:
            return await self._request("GET", "/metrics")
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        try:
            result = await self._request("GET", "/tools")
            return result.get("tools", [])
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            raise

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

    async def list_memory_keys(self) -> List[str]:
        """List all memory keys."""
        try:
            result = await self.call_tool("list_memory_keys", {})
            if isinstance(result, str):
                data = json.loads(result)
                return data.get("keys", [])
            return result.get("keys", [])
        except Exception as e:
            self.logger.error(f"Failed to list memory keys: {e}")
            return []

    async def delete_memory(self, key: str) -> bool:
        """Delete a memory entry."""
        try:
            result = await self.call_tool("delete_memory", {"key": key})
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            return await self._health_check_request()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            metrics = await self.get_metrics()
            return metrics.get("model_info", {})
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            raise


class GemmaHTTPBatchClient(GemmaHTTPClient):
    """HTTP client optimized for batch operations."""

    async def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate responses for multiple requests concurrently."""
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def _generate_with_semaphore(request):
            async with semaphore:
                return await self.generate_text(request)

        tasks = [_generate_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def store_memory_batch(self, entries: List[MemoryEntry]) -> List[str]:
        """Store multiple memory entries concurrently."""
        semaphore = asyncio.Semaphore(10)  # Higher limit for memory operations

        async def _store_with_semaphore(entry):
            async with semaphore:
                return await self.store_memory(entry)

        tasks = [_store_with_semaphore(entry) for entry in entries]
        return await asyncio.gather(*tasks)
