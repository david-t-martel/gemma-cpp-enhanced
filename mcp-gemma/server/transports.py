"""
Transport implementations using Strategy pattern.
Each transport strategy can be swapped without affecting the server.
"""

import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import aiohttp_cors
import websockets
from aiohttp import WSMsgType, web

from .core.contracts import ITransport
from .core.server import MCPServer


class TransportStrategy(ITransport):
    """Base class for transport strategies."""

    def __init__(self, server: MCPServer):
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the transport."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        self.running = False

    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming request."""
        pass


class StdioTransportStrategy(TransportStrategy):
    """Standard input/output transport strategy."""

    async def start(self) -> None:
        """Start stdio transport."""
        self.running = True
        self.logger.info("Starting stdio transport")

        try:
            # Try to use MCP's built-in stdio server
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
                    response = await self.handle_request(request)
                    if response:
                        print(json.dumps(response))
                        sys.stdout.flush()
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received: {line}")

            except Exception as e:
                self.logger.error(f"Error processing stdin: {e}")
                break

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request."""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            # Route to appropriate handler
            for handler in self.server.handlers:
                if handler.can_handle(method):
                    result = await handler.handle(method, params)
                    return {"jsonrpc": "2.0", "id": request_id, "result": result}

            # Method not found
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        except Exception as e:
            self.logger.error(f"Request handling error: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)},
            }

    async def stop(self) -> None:
        """Stop the transport."""
        self.running = False
        self.logger.info("Stdio transport stopped")


class HTTPTransportStrategy(TransportStrategy):
    """HTTP transport strategy using aiohttp."""

    def __init__(self, server: MCPServer, host: str = "localhost", port: int = 8080):
        super().__init__(server)
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None

    async def start(self) -> None:
        """Start HTTP server."""
        self.running = True
        self.logger.info(f"Starting HTTP transport on {self.host}:{self.port}")

        # Create aiohttp application
        self.app = web.Application()
        self._setup_routes()
        self._setup_cors()

        # Create and start runner
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        self.logger.info(f"HTTP server started on http://{self.host}:{self.port}")

    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post("/rpc", self._handle_rpc)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/metrics", self._handle_metrics)
        self.app.router.add_post("/generate", self._handle_generate)

    def _setup_cors(self):
        """Setup CORS for the application."""
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*"
                )
            },
        )

        for route in list(self.app.router.routes()):
            cors.add(route)

    async def _handle_rpc(self, request: web.Request) -> web.Response:
        """Handle JSON-RPC requests."""
        try:
            data = await request.json()
            response = await self.handle_request(data)
            return web.json_response(response)
        except Exception as e:
            self.logger.error(f"RPC error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        if self.server.metrics_service:
            health = await self.server.metrics_service.health_check()
            return web.json_response(health)
        else:
            return web.json_response({"status": "healthy"})

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle metrics requests."""
        if self.server.metrics_service:
            metrics = self.server.metrics_service.get_metrics()
            return web.json_response(
                {
                    "total_requests": metrics.total_requests,
                    "total_tokens": metrics.total_tokens,
                    "avg_response_time": metrics.avg_response_time,
                    "requests_per_minute": metrics.requests_per_minute,
                    "uptime_seconds": metrics.uptime_seconds,
                }
            )
        else:
            return web.json_response({"error": "Metrics not enabled"}, status=404)

    async def _handle_generate(self, request: web.Request) -> web.Response:
        """Handle direct generation requests."""
        try:
            data = await request.json()

            # Call generate_text through handlers
            for handler in self.server.handlers:
                if handler.can_handle("generate_text"):
                    result = await handler.handle("generate_text", data)
                    return web.json_response({"text": result})

            return web.json_response({"error": "Generation handler not found"}, status=404)

        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request."""
        # Similar to stdio implementation
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            for handler in self.server.handlers:
                if handler.can_handle(method):
                    result = await handler.handle(method, params)
                    return {"jsonrpc": "2.0", "id": request_id, "result": result}

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)},
            }

    async def stop(self) -> None:
        """Stop the HTTP server."""
        self.running = False
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        self.logger.info("HTTP transport stopped")


class WebSocketTransportStrategy(TransportStrategy):
    """WebSocket transport strategy."""

    def __init__(self, server: MCPServer, host: str = "localhost", port: int = 8081):
        super().__init__(server)
        self.host = host
        self.port = port
        self.websocket_server = None
        self.clients = set()

    async def start(self) -> None:
        """Start WebSocket server."""
        self.running = True
        self.logger.info(f"Starting WebSocket transport on ws://{self.host}:{self.port}")

        self.websocket_server = await websockets.serve(self._handle_client, self.host, self.port)

        self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

        # Keep the server running
        await asyncio.Future()  # This will never complete

    async def _handle_client(self, websocket, path):
        """Handle a WebSocket client connection."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected from {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    response = await self.handle_request(request)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    self.logger.error(f"WebSocket error: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected from {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket request."""
        # Same as other transports
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            for handler in self.server.handlers:
                if handler.can_handle(method):
                    result = await handler.handle(method, params)
                    return {"jsonrpc": "2.0", "id": request_id, "result": result}

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)},
            }

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients], return_exceptions=True
            )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self.running = False
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()

        # Close all client connections
        for client in self.clients:
            await client.close()

        self.clients.clear()
        self.logger.info("WebSocket transport stopped")


class CompositeTransport(TransportStrategy):
    """Composite transport that can run multiple transports simultaneously."""

    def __init__(self, server: MCPServer):
        super().__init__(server)
        self.transports = []

    def add_transport(self, transport: TransportStrategy):
        """Add a transport to the composite."""
        self.transports.append(transport)

    async def start(self) -> None:
        """Start all transports."""
        self.running = True
        tasks = []
        for transport in self.transports:
            tasks.append(asyncio.create_task(transport.start()))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop all transports."""
        self.running = False
        tasks = []
        for transport in self.transports:
            tasks.append(asyncio.create_task(transport.stop()))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Not used for composite transport."""
        raise NotImplementedError("Composite transport doesn't handle requests directly")
