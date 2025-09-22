"""MCP (Model Context Protocol) server implementation for tool hosting."""

import asyncio
import json
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import uuid4

import websockets
from aiohttp import web
from aiohttp import web_request
from pydantic import BaseModel
from pydantic import ValidationError

from src.domain.tools.base import BaseTool
from src.domain.tools.base import ToolExecutionContext
from src.domain.tools.base import ToolRegistry
from src.domain.tools.schemas import EnhancedToolSchema

logger = logging.getLogger(__name__)


class McpServerConfig(BaseModel):
    """MCP server configuration."""

    name: str
    host: str = "localhost"
    port: int = 8000
    protocol: str = "websocket"  # websocket, http, both
    max_connections: int = 100
    request_timeout: int = 60
    enable_cors: bool = True
    cors_origins: list[str] = ["*"]
    auth_required: bool = False
    auth_tokens: list[str] = []
    rate_limit_per_minute: int = 1000
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    log_requests: bool = True


class McpRequest(BaseModel):
    """MCP protocol request."""

    id: str
    method: str
    params: dict[str, Any] = {}


class McpResponse(BaseModel):
    """MCP protocol response."""

    id: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class McpError(BaseModel):
    """MCP protocol error."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self._connections: dict[str, websockets.WebSocketServerProtocol] = {}
        self._connection_metadata: dict[str, dict[str, Any]] = {}

    async def add_connection(
        self, connection_id: str, websocket: websockets.WebSocketServerProtocol
    ) -> None:
        """Add a new WebSocket connection."""
        self._connections[connection_id] = websocket
        self._connection_metadata[connection_id] = {
            "connected_at": time.time(),
            "remote_address": websocket.remote_address,
            "requests_handled": 0,
            "last_activity": time.time(),
        }
        logger.info(f"New WebSocket connection: {connection_id}")

    async def remove_connection(self, connection_id: str) -> None:
        """Remove a WebSocket connection."""
        if connection_id in self._connections:
            del self._connections[connection_id]
            del self._connection_metadata[connection_id]
            logger.info(f"WebSocket connection closed: {connection_id}")

    def get_connection(self, connection_id: str) -> websockets.WebSocketServerProtocol | None:
        """Get a WebSocket connection."""
        return self._connections.get(connection_id)

    def update_activity(self, connection_id: str) -> None:
        """Update last activity timestamp."""
        if connection_id in self._connection_metadata:
            self._connection_metadata[connection_id]["last_activity"] = time.time()
            self._connection_metadata[connection_id]["requests_handled"] += 1

    def list_connections(self) -> dict[str, dict[str, Any]]:
        """List all active connections."""
        return self._connection_metadata.copy()

    async def broadcast(self, message: dict[str, Any], exclude: str | None = None) -> None:
        """Broadcast message to all connections."""
        message_json = json.dumps(message)
        disconnected = []

        for conn_id, websocket in self._connections.items():
            if exclude and conn_id == exclude:
                continue

            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(conn_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {conn_id}: {e}")
                disconnected.append(conn_id)

        # Remove disconnected connections
        for conn_id in disconnected:
            await self.remove_connection(conn_id)


class McpServer:
    """MCP protocol server."""

    def __init__(self, config: McpServerConfig, tool_registry: ToolRegistry):
        self.config = config
        self.registry = tool_registry
        self._connection_manager = ConnectionManager()
        self._app: web.Application | None = None
        self._websocket_server: websockets.WebSocketServer | None = None
        self._running = False
        self._request_handlers: dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "server/capabilities": self._handle_server_capabilities,
            "server/status": self._handle_server_status,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }

    async def start(self) -> None:
        """Start the MCP server."""
        try:
            if self.config.protocol in ["websocket", "both"]:
                await self._start_websocket_server()

            if self.config.protocol in ["http", "both"]:
                await self._start_http_server()

            self._running = True
            logger.info(
                f"MCP server '{self.config.name}' started on {self.config.host}:{self.config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the MCP server."""
        try:
            self._running = False

            # Stop WebSocket server
            if self._websocket_server:
                self._websocket_server.close()
                await self._websocket_server.wait_closed()

            # Stop HTTP server
            if self._app:
                await self._app.shutdown()
                await self._app.cleanup()

            logger.info(f"MCP server '{self.config.name}' stopped")

        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")

    async def _start_websocket_server(self) -> None:
        """Start WebSocket server."""
        self._websocket_server = await websockets.serve(
            self._handle_websocket_connection,
            self.config.host,
            self.config.port,
            max_size=self.config.max_request_size,
            max_queue=self.config.max_connections,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10,
        )

    async def _start_http_server(self) -> None:
        """Start HTTP server."""
        self._app = web.Application(client_max_size=self.config.max_request_size)

        # Add CORS middleware if enabled
        if self.config.enable_cors:
            self._setup_cors()

        # Add routes
        self._app.router.add_post("/mcp", self._handle_http_request)
        self._app.router.add_get("/health", self._handle_health_check)
        self._app.router.add_get("/status", self._handle_status)

        # Start server
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port + 1)
        await site.start()

    def _setup_cors(self) -> None:
        """Setup CORS middleware."""

        async def cors_middleware(request: web_request.Request, handler):
            # Handle preflight requests
            if request.method == "OPTIONS":
                response = web.Response()
            else:
                response = await handler(request)

            # Add CORS headers
            origin = request.headers.get("Origin", "")
            if origin in self.config.cors_origins or "*" in self.config.cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                response.headers["Access-Control-Max-Age"] = "86400"

            return response

        self._app.middlewares.append(cors_middleware)

    async def _handle_websocket_connection(
        self, websocket: websockets.WebSocketServerProtocol, path: str
    ) -> None:
        """Handle WebSocket connection."""
        connection_id = str(uuid4())

        try:
            await self._connection_manager.add_connection(connection_id, websocket)

            async for message in websocket:
                try:
                    self._connection_manager.update_activity(connection_id)

                    if isinstance(message, str):
                        data = json.loads(message)
                        response = await self._process_request(data, connection_id)
                        await websocket.send(json.dumps(response.dict()))
                    else:
                        await websocket.send(
                            json.dumps(
                                {"error": {"code": -32600, "message": "Invalid message format"}}
                            )
                        )

                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps({"error": {"code": -32700, "message": "Parse error"}})
                    )
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket.send(
                        json.dumps({"error": {"code": -32603, "message": "Internal error"}})
                    )

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            await self._connection_manager.remove_connection(connection_id)

    async def _handle_http_request(self, request: web_request.Request) -> web.Response:
        """Handle HTTP request."""
        try:
            # Authentication check
            if self.config.auth_required:
                auth_header = request.headers.get("Authorization", "")
                if not self._validate_auth(auth_header):
                    return web.json_response(
                        {"error": {"code": 401, "message": "Unauthorized"}}, status=401
                    )

            # Parse request
            data = await request.json()
            response = await self._process_request(data)

            return web.json_response(response.dict())

        except json.JSONDecodeError:
            return web.json_response(
                {"error": {"code": -32700, "message": "Parse error"}}, status=400
            )
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return web.json_response(
                {"error": {"code": -32603, "message": "Internal error"}}, status=500
            )

    async def _handle_health_check(self, request: web_request.Request) -> web.Response:
        """Handle health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "server": self.config.name,
                "uptime": time.time(),
                "connections": len(self._connection_manager.list_connections()),
                "tools": len(self.registry.list_tools()),
            }
        )

    async def _handle_status(self, request: web_request.Request) -> web.Response:
        """Handle status endpoint."""
        return web.json_response(
            {
                "server": self.config.name,
                "protocol": self.config.protocol,
                "running": self._running,
                "connections": self._connection_manager.list_connections(),
                "tools": [schema.name for schema in self.registry.get_schemas()],
            }
        )

    def _validate_auth(self, auth_header: str) -> bool:
        """Validate authentication."""
        if not self.config.auth_tokens:
            return True

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return token in self.config.auth_tokens

        return False

    async def _process_request(
        self, data: dict[str, Any], connection_id: str | None = None
    ) -> McpResponse:
        """Process MCP request."""
        try:
            # Validate request format
            try:
                request = McpRequest(**data)
            except ValidationError as e:
                return McpResponse(
                    id=data.get("id", "unknown"),
                    error={"code": -32602, "message": f"Invalid params: {e}"},
                )

            # Log request if enabled
            if self.config.log_requests:
                logger.debug(f"MCP request: {request.method} from {connection_id}")

            # Handle request
            handler = self._request_handlers.get(request.method)
            if handler:
                result = await handler(request, connection_id)
                return McpResponse(id=request.id, result=result)
            else:
                return McpResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {request.method}"},
                )

        except Exception as e:
            logger.error(f"Request processing error: {e}\n{traceback.format_exc()}")
            return McpResponse(
                id=data.get("id", "unknown"), error={"code": -32603, "message": "Internal error"}
            )

    async def _handle_initialize(
        self, request: McpRequest, connection_id: str | None
    ) -> dict[str, Any]:
        """Handle initialization request."""
        client_info = request.params.get("client_info", {})
        protocol_version = request.params.get("protocol_version", "2024-11-05")

        logger.info(
            f"Client initialized: {client_info.get('name', 'unknown')} v{client_info.get('version', 'unknown')}"
        )

        return {
            "server_info": {"name": self.config.name, "version": "1.0.0"},
            "protocol_version": protocol_version,
            "capabilities": {"tools": {"list_changed": True}, "server": {"status": True}},
        }

    async def _handle_server_capabilities(
        self, request: McpRequest, connection_id: str | None
    ) -> dict[str, Any]:
        """Handle server capabilities request."""
        return {
            "capabilities": {
                "tools": {"list_changed": True, "call": True},
                "server": {"status": True, "progress": False},
            },
            "server_info": {
                "name": self.config.name,
                "version": "1.0.0",
                "description": "LLM Agent Framework MCP Server",
            },
        }

    async def _handle_server_status(
        self, request: McpRequest, connection_id: str | None
    ) -> dict[str, Any]:
        """Handle server status request."""
        return {
            "status": "running" if self._running else "stopped",
            "uptime": time.time(),
            "connections": len(self._connection_manager.list_connections()),
            "tools_count": len(self.registry.list_tools()),
            "memory_usage": self._get_memory_usage(),
            "request_stats": self._get_request_stats(),
        }

    async def _handle_tools_list(
        self, request: McpRequest, connection_id: str | None
    ) -> dict[str, Any]:
        """Handle tools list request."""
        try:
            schemas = self.registry.get_schemas()
            tools = []

            for schema in schemas:
                # Convert our schema to MCP format
                tool_def = {"name": schema.name, "description": schema.description}

                # Add input schema
                if schema.parameters:
                    properties = {}
                    required = []

                    for param in schema.parameters:
                        prop_def = {"type": param.type, "description": param.description}

                        if param.enum:
                            prop_def["enum"] = param.enum
                        if param.minimum is not None:
                            prop_def["minimum"] = param.minimum
                        if param.maximum is not None:
                            prop_def["maximum"] = param.maximum
                        if param.pattern:
                            prop_def["pattern"] = param.pattern
                        if param.default is not None:
                            prop_def["default"] = param.default

                        properties[param.name] = prop_def

                        if param.required:
                            required.append(param.name)

                    tool_def["inputSchema"] = {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }

                tools.append(tool_def)

            return {"tools": tools}

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise

    async def _handle_tools_call(
        self, request: McpRequest, connection_id: str | None
    ) -> dict[str, Any]:
        """Handle tool call request."""
        try:
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            context_data = request.params.get("context", {})

            if not tool_name:
                raise ValueError("Tool name is required")

            # Create execution context
            context = ToolExecutionContext(
                execution_id=context_data.get("execution_id", str(uuid4())),
                agent_id=context_data.get("agent_id"),
                session_id=context_data.get("session_id"),
                timeout=context_data.get("timeout"),
                security_level=context_data.get("security_level", "standard"),
                metadata={"mcp_connection": connection_id},
            )

            # Execute tool
            result = await self.registry.execute_tool(tool_name, context, **arguments)

            if result.success:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                json.dumps(result.data)
                                if result.data
                                else "Tool executed successfully"
                            ),
                        }
                    ],
                    "isError": False,
                    "metadata": {
                        "execution_time": result.execution_time,
                        "cached": result.cached,
                        **result.metadata,
                    },
                }
            else:
                return {
                    "content": [{"type": "text", "text": result.error or "Tool execution failed"}],
                    "isError": True,
                    "metadata": {"execution_time": result.execution_time, **result.metadata},
                }

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "content": [{"type": "text", "text": f"Tool execution failed: {e!s}"}],
                "isError": True,
            }

    async def _handle_ping(self, request: McpRequest, connection_id: str | None) -> dict[str, Any]:
        """Handle ping request."""
        return {"pong": True, "timestamp": time.time(), "server": self.config.name}

    async def _handle_shutdown(
        self, request: McpRequest, connection_id: str | None
    ) -> dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Shutdown requested via MCP")
        asyncio.create_task(self.stop())
        return {"shutting_down": True}

    def _get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    def _get_request_stats(self) -> dict[str, Any]:
        """Get request statistics."""
        connections = self._connection_manager.list_connections()
        total_requests = sum(conn["requests_handled"] for conn in connections.values())

        return {
            "total_connections": len(connections),
            "total_requests": total_requests,
            "average_requests_per_connection": (
                total_requests / len(connections) if connections else 0
            ),
        }

    async def notify_tools_changed(self) -> None:
        """Notify clients that tools list has changed."""
        notification = {"method": "tools/list_changed", "params": {}}
        await self._connection_manager.broadcast(notification)

    async def send_progress(self, progress_token: str, progress: float, message: str) -> None:
        """Send progress notification to clients."""
        notification = {
            "method": "server/progress",
            "params": {"token": progress_token, "progress": progress, "message": message},
        }
        await self._connection_manager.broadcast(notification)

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connection_manager.list_connections())


# Factory function for easy server creation
async def create_mcp_server(
    name: str, tool_registry: ToolRegistry, host: str = "localhost", port: int = 8000, **kwargs
) -> McpServer:
    """Create and configure an MCP server."""
    config = McpServerConfig(name=name, host=host, port=port, **kwargs)

    server = McpServer(config, tool_registry)
    return server
