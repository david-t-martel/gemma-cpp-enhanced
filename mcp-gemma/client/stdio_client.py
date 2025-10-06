"""
Stdio MCP client for direct communication with Gemma server.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from .base_client import (
    BaseGemmaClient,
    GemmaClientError,
    GemmaConnectionError,
    GemmaTimeoutError,
    GenerationRequest,
    GenerationResponse,
    MemoryEntry,
)


class GemmaStdioClient(BaseGemmaClient):
    """Stdio client for MCP Gemma server."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        gemma_executable: Optional[str] = None,
        server_script: Optional[str] = None,
        timeout: float = 30.0,
        debug: bool = False,
    ):
        super().__init__(timeout=timeout, debug=debug)

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.gemma_executable = (
            gemma_executable or "/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma"
        )

        # Default to the server script in the same directory structure
        if server_script is None:
            server_script = str(Path(__file__).parent.parent / "server" / "main.py")
        self.server_script = server_script

        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0

    async def connect(self) -> None:
        """Start the MCP server process and connect via stdio."""
        if self.process is not None:
            return

        # Build command to start the server
        cmd = [sys.executable, self.server_script, "--mode", "stdio", "--model", self.model_path]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        cmd.extend(["--gemma-executable", self.gemma_executable])

        if self.debug:
            cmd.append("--debug")

        self.logger.info(f"Starting MCP server: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )

            # Wait a moment for the server to start
            await asyncio.sleep(1)

            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else "No stderr"
                raise GemmaConnectionError(f"Server process failed to start: {stderr}")

            self.logger.info("MCP server started successfully")

        except Exception as e:
            if self.process:
                self.process.terminate()
                self.process = None
            raise GemmaConnectionError(f"Failed to start MCP server: {e}")

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self.process is None:
            return

        try:
            self.process.terminate()

            # Wait for process to terminate gracefully
            try:
                await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(self.process.wait)), timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Server process did not terminate gracefully, killing...")
                self.process.kill()
                await asyncio.to_thread(self.process.wait)

        except Exception as e:
            self.logger.error(f"Error stopping server process: {e}")
        finally:
            self.process = None

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server."""
        if self.process is None:
            raise GemmaConnectionError("Not connected to server")

        self.request_id += 1
        request = {"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params}

        request_json = json.dumps(request) + "\n"

        try:
            # Send request
            self.process.stdin.write(request_json)
            self.process.stdin.flush()

            # Read response with timeout
            response_json = await asyncio.wait_for(
                asyncio.to_thread(self.process.stdout.readline), timeout=self.timeout
            )

            if not response_json:
                raise GemmaConnectionError("Server process terminated unexpectedly")

            response = json.loads(response_json.strip())

            # Check for errors
            if "error" in response:
                error = response["error"]
                raise GemmaClientError(f"Server error: {error.get('message', 'Unknown error')}")

            return response.get("result", {})

        except asyncio.TimeoutError:
            raise GemmaTimeoutError(f"Request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise GemmaClientError(f"Invalid JSON response: {e}")

    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the server."""
        self._validate_request(request)
        self._log_request(request)

        start_time = time.time()

        params = {"name": "generate_text", "arguments": {"prompt": request.prompt, "stream": False}}

        if request.max_tokens is not None:
            params["arguments"]["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["arguments"]["temperature"] = request.temperature

        try:
            result = await self._send_request("tools/call", params)

            # Extract text from MCP response structure
            if isinstance(result, dict) and "content" in result:
                text = result["content"][0]["text"] if result["content"] else ""
            else:
                text = str(result)

            response_time = time.time() - start_time

            response = GenerationResponse(
                text=text,
                response_time=response_time,
                tokens_generated=len(text.split()),  # Rough estimate
            )

            self._log_response(response)
            return response

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

    async def generate_text_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate text with streaming (Note: stdio doesn't support true streaming)."""
        # For stdio, we simulate streaming by yielding the complete response
        response = await self.generate_text(request)

        # Simulate streaming by yielding chunks
        chunk_size = 10
        words = response.text.split()

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i : i + chunk_size]
            chunk = " ".join(chunk_words)
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Small delay to simulate streaming

    async def switch_model(self, model_path: str, tokenizer_path: Optional[str] = None) -> bool:
        """Switch to a different model."""
        params = {"name": "switch_model", "arguments": {"model_path": model_path}}

        if tokenizer_path:
            params["arguments"]["tokenizer_path"] = tokenizer_path

        try:
            result = await self._send_request("tools/call", params)
            self.logger.info(f"Model switched: {result}")
            return True
        except Exception as e:
            self.logger.error(f"Model switch failed: {e}")
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        params = {"name": "get_metrics", "arguments": {}}

        try:
            result = await self._send_request("tools/call", params)
            if isinstance(result, dict) and "content" in result:
                metrics_text = result["content"][0]["text"]
                return json.loads(metrics_text)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise

    async def store_memory(self, entry: MemoryEntry) -> str:
        """Store content in memory."""
        params = {"name": "store_memory", "arguments": {"key": entry.key, "content": entry.content}}

        if entry.metadata:
            params["arguments"]["metadata"] = entry.metadata

        try:
            result = await self._send_request("tools/call", params)
            if isinstance(result, dict) and "content" in result:
                return result["content"][0]["text"]
            return str(result)
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise

    async def retrieve_memory(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve content from memory."""
        params = {"name": "retrieve_memory", "arguments": {"key": key}}

        try:
            result = await self._send_request("tools/call", params)
            if isinstance(result, dict) and "content" in result:
                data_text = result["content"][0]["text"]
                data = json.loads(data_text)
            else:
                data = json.loads(str(result))

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
        params = {"name": "search_memory", "arguments": {"query": query, "limit": limit}}

        try:
            result = await self._send_request("tools/call", params)
            if isinstance(result, dict) and "content" in result:
                data_text = result["content"][0]["text"]
                data = json.loads(data_text)
            else:
                data = json.loads(str(result))

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

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Simple health check by getting metrics
            metrics = await self.get_metrics()
            return {
                "status": "healthy",
                "server_info": metrics.get("server_info", {}),
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}
