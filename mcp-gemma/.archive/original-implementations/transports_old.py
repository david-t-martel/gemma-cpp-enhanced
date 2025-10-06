"""
Transport layer implementations for MCP Gemma server.

Supports multiple transport protocols:
- stdio: Direct command-line integration
- HTTP: REST API interface
- WebSocket: Real-time streaming interface
"""

import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import aiohttp_cors
import websockets
from aiohttp import WSMsgType, web


class Transport(ABC):
    """Abstract base class for transport implementations."""

    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def start(self, *args, **kwargs):
        """Start the transport."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the transport."""
        pass


class StdioTransport(Transport):
    """Standard input/output transport for MCP protocol."""

    def __init__(self, server):
        super().__init__(server)
        self.running = False

    async def start(self):
        """Start stdio transport."""
        self.running = True
        self.logger.info("Starting stdio transport")

        try:
            # Use MCP's built-in stdio server if available
            try:
                from mcp.server.stdio import stdio_server

                async with stdio_server() as streams:
                    await self.server.get_server().run(
                        streams[0],
                        streams[1],
                        self.server.get_server().create_initialization_options(),
                    )
            except ImportError:
                # Fallback implementation
                await self._run_stdio_fallback()

        except Exception as e:
            self.logger.error(f"Stdio transport error: {e}")
            raise

    async def _run_stdio_fallback(self):
        """Fallback stdio implementation."""
        self.logger.info("Using fallback stdio implementation")

        while self.running:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON-RPC request
                try:
                    request = json.loads(line)
                    response = await self._handle_request(request)
                    if response:
                        print(json.dumps(response))
                        sys.stdout.flush()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received: {line}")

            except Exception as e:
                self.logger.error(f"Error processing stdin: {e}")
                break

    async def _handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "tools/list":
                tools_result = await self.server._setup_handlers()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": [tool.dict() for tool in tools_result.tools]},
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self.server._handle_call_tool(tool_name, arguments)
                return {"jsonrpc": "2.0", "id": request_id, "result": result.dict()}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    async def stop(self):
        """Stop stdio transport."""
        self.running = False
        self.logger.info("Stdio transport stopped")


class HTTPTransport(Transport):
    """HTTP REST API transport."""

    def __init__(self, server, host: str = "localhost", port: int = 8080):
        super().__init__(server)
        self.host = host
        self.port = port
        self.app = None
        self.runner = None

    async def start(self):
        """Start HTTP server."""
        self.logger.info(f"Starting HTTP transport on {self.host}:{self.port}")

        self.app = web.Application()

        # Setup CORS
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*"
                )
            },
        )

        # Add routes
        self.app.router.add_get("/health", self._health_check)
        self.app.router.add_get("/tools", self._list_tools)
        self.app.router.add_post("/tools/{tool_name}/call", self._call_tool)
        self.app.router.add_get("/metrics", self._get_metrics)
        self.app.router.add_post("/generate", self._generate_text)

        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        self.logger.info(f"HTTP server started at http://{self.host}:{self.port}")

    async def _health_check(self, request):
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "server": self.server.server.name,
                "version": self.server.server.version,
            }
        )

    async def _list_tools(self, request):
        """List available tools endpoint."""
        try:
            # This needs to be fixed to properly call the list_tools handler
            tools = [
                {"name": "generate_text", "description": "Generate text using the Gemma model"},
                {"name": "switch_model", "description": "Switch to a different model"},
                {"name": "get_metrics", "description": "Get server performance metrics"},
            ]

            if self.server.redis_client:
                tools.extend(
                    [
                        {"name": "store_memory", "description": "Store information in memory"},
                        {"name": "retrieve_memory", "description": "Retrieve stored memory by key"},
                        {"name": "search_memory", "description": "Search memory by content"},
                    ]
                )

            return web.json_response({"tools": tools})

        except Exception as e:
            self.logger.error(f"Error listing tools: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _call_tool(self, request):
        """Call a tool endpoint."""
        tool_name = request.match_info["tool_name"]

        try:
            data = await request.json()
            arguments = data.get("arguments", {})

            # Call the appropriate handler method
            if tool_name == "generate_text":
                result = await self.server._handle_generate_text(arguments)
            elif tool_name == "switch_model":
                result = await self.server._handle_switch_model(arguments)
            elif tool_name == "get_metrics":
                result = await self.server._handle_get_metrics(arguments)
            elif tool_name == "store_memory":
                result = await self.server._handle_store_memory(arguments)
            elif tool_name == "retrieve_memory":
                result = await self.server._handle_retrieve_memory(arguments)
            elif tool_name == "search_memory":
                result = await self.server._handle_search_memory(arguments)
            else:
                return web.json_response({"error": f"Unknown tool: {tool_name}"}, status=400)

            return web.json_response({"result": result})

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _get_metrics(self, request):
        """Get metrics endpoint."""
        try:
            result = await self.server._handle_get_metrics({})
            return web.json_response(json.loads(result))
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _generate_text(self, request):
        """Direct text generation endpoint."""
        try:
            data = await request.json()
            result = await self.server._handle_generate_text(data)
            return web.json_response({"text": result})
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def stop(self):
        """Stop HTTP server."""
        if self.runner:
            await self.runner.cleanup()
        self.logger.info("HTTP transport stopped")


class WebSocketTransport(Transport):
    """WebSocket transport for real-time streaming."""

    def __init__(self, server, host: str = "localhost", port: int = 8081):
        super().__init__(server)
        self.host = host
        self.port = port
        self.websocket_server = None
        self.clients = set()

    async def start(self):
        """Start WebSocket server."""
        self.logger.info(f"Starting WebSocket transport on {self.host}:{self.port}")

        self.websocket_server = await websockets.serve(self._handle_client, self.host, self.port)

        self.logger.info(f"WebSocket server started at ws://{self.host}:{self.port}")

    async def _handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        self.clients.add(websocket)
        client_id = id(websocket)
        self.logger.info(f"Client {client_id} connected")

        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "welcome",
                        "server": self.server.server.name,
                        "version": self.server.server.version,
                    }
                )
            )

            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    await websocket.send(json.dumps({"type": "error", "message": str(e)}))

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        finally:
            self.clients.discard(websocket)

    async def _handle_websocket_message(self, websocket, data: Dict[str, Any]):
        """Handle WebSocket message."""
        message_type = data.get("type")

        if message_type == "generate_text":
            await self._handle_streaming_generation(websocket, data)
        elif message_type == "call_tool":
            await self._handle_tool_call(websocket, data)
        elif message_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
        else:
            await websocket.send(
                json.dumps({"type": "error", "message": f"Unknown message type: {message_type}"})
            )

    async def _handle_streaming_generation(self, websocket, data: Dict[str, Any]):
        """Handle streaming text generation."""
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", self.server.config.max_tokens)
        temperature = data.get("temperature", self.server.config.temperature)

        try:
            # Send start message
            await websocket.send(json.dumps({"type": "generation_start", "prompt": prompt}))

            # Stream callback function
            async def stream_callback(chunk):
                await websocket.send(json.dumps({"type": "generation_chunk", "chunk": chunk}))

            # Generate with streaming
            result = await self.server._handle_generate_text(
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
            )

            # Send completion
            await websocket.send(json.dumps({"type": "generation_complete", "result": result}))

        except Exception as e:
            await websocket.send(json.dumps({"type": "generation_error", "error": str(e)}))

    async def _handle_tool_call(self, websocket, data: Dict[str, Any]):
        """Handle tool call via WebSocket."""
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})

        try:
            if tool_name == "generate_text":
                result = await self.server._handle_generate_text(arguments)
            elif tool_name == "switch_model":
                result = await self.server._handle_switch_model(arguments)
            elif tool_name == "get_metrics":
                result = await self.server._handle_get_metrics(arguments)
            elif tool_name == "store_memory":
                result = await self.server._handle_store_memory(arguments)
            elif tool_name == "retrieve_memory":
                result = await self.server._handle_retrieve_memory(arguments)
            elif tool_name == "search_memory":
                result = await self.server._handle_search_memory(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            await websocket.send(
                json.dumps({"type": "tool_result", "tool_name": tool_name, "result": result})
            )

        except Exception as e:
            await websocket.send(
                json.dumps({"type": "tool_error", "tool_name": tool_name, "error": str(e)})
            )

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients],
                return_exceptions=True,
            )

    async def stop(self):
        """Stop WebSocket server."""
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        self.logger.info("WebSocket transport stopped")
