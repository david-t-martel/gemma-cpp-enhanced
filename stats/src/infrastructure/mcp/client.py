"""MCP (Model Context Protocol) client implementation for distributed tools."""

import asyncio
import contextlib
import json
import time
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
from uuid import uuid4

import aiohttp
import websockets
from pydantic import BaseModel
from pydantic import Field

from src.domain.tools.base import BaseTool
from src.domain.tools.base import ToolExecutionContext
from src.domain.tools.base import ToolRegistry
from src.domain.tools.base import ToolResult
from src.domain.tools.schemas import EnhancedToolSchema
from src.domain.tools.schemas import ParameterSchema
from src.domain.tools.schemas import ToolCategory
from src.domain.tools.schemas import ToolType


class McpServerConfig(BaseModel):
    """MCP server configuration."""

    name: str
    url: str
    protocol: str = "websocket"  # websocket, http, stdio
    auth_token: str | None = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    heartbeat_interval: int = 30
    max_concurrent_requests: int = 10
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class McpMessage(BaseModel):
    """MCP protocol message."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class McpResponse(BaseModel):
    """MCP protocol response."""

    id: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    timestamp: float = Field(default_factory=time.time)


class McpTool(BaseTool):
    """Remote MCP tool wrapper."""

    def __init__(self, schema: EnhancedToolSchema, client: "McpClient"):
        super().__init__()
        self._schema = schema
        self._client = client

    @property
    def schema(self) -> EnhancedToolSchema:
        return self._schema

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute remote tool via MCP client."""
        try:
            start_time = time.time()

            # Call remote tool
            response = await self._client.call_tool(self._schema.name, kwargs, context)

            execution_time = time.time() - start_time

            if response.error:
                return ToolResult(
                    success=False,
                    error=response.error.get("message", "Unknown MCP error"),
                    execution_time=execution_time,
                    context=context,
                    metadata={"mcp_server": self._client.config.name},
                )

            return ToolResult(
                success=True,
                data=response.result,
                execution_time=execution_time,
                context=context,
                metadata={"mcp_server": self._client.config.name},
            )

        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=0,
                context=context,
                metadata={"mcp_server": self._client.config.name},
            )


class McpClient:
    """MCP protocol client."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self._connection: websockets.WebSocketServerProtocol | aiohttp.ClientSession | None = None
        self._connected = False
        self._tools: dict[str, EnhancedToolSchema] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._heartbeat_task: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._request_id_counter = 0

    async def connect(self) -> bool:
        """Connect to MCP server."""
        try:
            if self.config.protocol == "websocket":
                await self._connect_websocket()
            elif self.config.protocol == "http":
                await self._connect_http()
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

            # Initialize connection
            await self._initialize_connection()

            # Start heartbeat
            if self.config.heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self._connected = True
            logger.info(f"Connected to MCP server: {self.config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        try:
            self._connected = False

            # Cancel heartbeat
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._heartbeat_task

            # Close connection
            if self._connection and self.config.protocol in {"websocket", "http"}:
                await self._connection.close()

            # Cancel pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()

            logger.info(f"Disconnected from MCP server: {self.config.name}")

        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {self.config.name}: {e}")

    async def _connect_websocket(self) -> None:
        """Connect via WebSocket."""
        headers = {}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"

        self._connection = await websockets.connect(
            self.config.url,
            extra_headers=headers,
            ping_interval=self.config.heartbeat_interval,
            ping_timeout=self.config.timeout,
            close_timeout=self.config.timeout,
        )

        # Start message handling
        asyncio.create_task(self._handle_websocket_messages())

    async def _connect_http(self) -> None:
        """Connect via HTTP."""
        headers = {}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"

        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        self._connection = aiohttp.ClientSession(
            headers=headers, connector=connector, timeout=timeout
        )

    async def _initialize_connection(self) -> None:
        """Initialize MCP connection."""
        # Send initialization message
        init_message = McpMessage(
            method="initialize",
            params={
                "client_info": {"name": "llm-agent-framework", "version": "1.0.0"},
                "protocol_version": "2024-11-05",
            },
        )

        response = await self._send_message(init_message)
        if response.error:
            raise RuntimeError(f"MCP initialization failed: {response.error}")

        # Get server capabilities
        await self._get_server_capabilities()

        # List available tools
        await self._list_server_tools()

    async def _get_server_capabilities(self) -> None:
        """Get server capabilities."""
        message = McpMessage(method="server/capabilities")
        response = await self._send_message(message)

        if response.result:
            capabilities = response.result.get("capabilities", {})
            self.config.capabilities = list(capabilities.keys())
            logger.debug(f"Server {self.config.name} capabilities: {self.config.capabilities}")

    async def _list_server_tools(self) -> None:
        """List available tools on the server."""
        message = McpMessage(method="tools/list")
        response = await self._send_message(message)

        if response.result and "tools" in response.result:
            for tool_data in response.result["tools"]:
                try:
                    # Convert MCP tool format to our schema format
                    schema = self._convert_mcp_tool_to_schema(tool_data)
                    self._tools[schema.name] = schema
                    logger.debug(f"Registered MCP tool: {schema.name}")
                except Exception as e:
                    logger.warning(f"Failed to convert MCP tool: {e}")

        logger.info(f"Discovered {len(self._tools)} tools on server {self.config.name}")

    def _convert_mcp_tool_to_schema(self, tool_data: dict[str, Any]) -> EnhancedToolSchema:
        """Convert MCP tool format to our schema format."""
        name = tool_data["name"]
        description = tool_data.get("description", "")

        # Convert parameters
        parameters = []
        if "inputSchema" in tool_data:
            input_schema = tool_data["inputSchema"]
            if "properties" in input_schema:
                required_params = set(input_schema.get("required", []))
                for param_name, param_def in input_schema["properties"].items():
                    param_schema = ParameterSchema(
                        name=param_name,
                        type=param_def.get("type", "string"),
                        description=param_def.get("description", ""),
                        required=param_name in required_params,
                        default=param_def.get("default"),
                        enum=param_def.get("enum"),
                        minimum=param_def.get("minimum"),
                        maximum=param_def.get("maximum"),
                        pattern=param_def.get("pattern"),
                        format=param_def.get("format"),
                    )
                    parameters.append(param_schema)

        return EnhancedToolSchema(
            name=name,
            description=description,
            category=ToolCategory.CUSTOM,
            type=ToolType.MCP,
            parameters=parameters,
            metadata={"server": self.config.name, "original_schema": tool_data},
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: ToolExecutionContext | None = None
    ) -> McpResponse:
        """Call a remote tool."""
        if not self._connected:
            raise RuntimeError(f"Not connected to MCP server: {self.config.name}")

        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not available on server {self.config.name}")

        async with self._semaphore:
            message = McpMessage(method="tools/call", params={"name": name, "arguments": arguments})

            # Add context information if available
            if context:
                message.params["context"] = {
                    "execution_id": context.execution_id,
                    "agent_id": context.agent_id,
                    "session_id": context.session_id,
                    "timeout": context.timeout,
                    "security_level": context.security_level,
                }

            response = await self._send_message(message)
            return response

    async def _send_message(self, message: McpMessage) -> McpResponse:
        """Send message and wait for response."""
        if not self._connection:
            raise RuntimeError("No active connection")

        message_data = message.dict()

        try:
            if self.config.protocol == "websocket":
                return await self._send_websocket_message(message_data)
            elif self.config.protocol == "http":
                return await self._send_http_message(message_data)
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

        except Exception as e:
            logger.error(f"Failed to send MCP message: {e}")
            raise

    async def _send_websocket_message(self, message_data: dict[str, Any]) -> McpResponse:
        """Send message via WebSocket."""
        message_id = message_data["id"]

        # Create future for response
        future = asyncio.Future()
        self._pending_requests[message_id] = future

        try:
            # Send message
            await self._connection.send(json.dumps(message_data))

            # Wait for response
            response_data = await asyncio.wait_for(future, timeout=self.config.timeout)
            return McpResponse(**response_data)

        except TimeoutError:
            raise TimeoutError(f"MCP request timed out: {message_id}")
        finally:
            self._pending_requests.pop(message_id, None)

    async def _send_http_message(self, message_data: dict[str, Any]) -> McpResponse:
        """Send message via HTTP."""
        async with self._connection.post(self.config.url, json=message_data) as response:
            response.raise_for_status()
            response_data = await response.json()
            return McpResponse(**response_data)

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self._connection:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {self.config.name}")
            self._connected = False
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
            self._connected = False

    async def _process_message(self, data: dict[str, Any]) -> None:
        """Process incoming message."""
        message_id = data.get("id")

        if message_id and message_id in self._pending_requests:
            # Response to pending request
            future = self._pending_requests[message_id]
            if not future.done():
                future.set_result(data)
        else:
            # Notification or unsolicited message
            method = data.get("method")
            if method:
                await self._handle_notification(method, data.get("params", {}))

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle server notifications."""
        if method == "tools/list_changed":
            # Tool list changed, refresh
            await self._list_server_tools()
        elif method == "server/progress":
            # Progress notification
            logger.debug(f"Server progress: {params}")
        else:
            logger.debug(f"Unknown notification: {method}")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to keep connection alive."""
        try:
            while self._connected:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self._connected:
                    break

                try:
                    # Send ping/heartbeat
                    if self.config.protocol == "websocket":
                        await self._connection.ping()
                    else:
                        # HTTP keepalive
                        message = McpMessage(method="ping")
                        await self._send_message(message)

                except Exception as e:
                    logger.warning(f"Heartbeat failed for {self.config.name}: {e}")
                    self._connected = False
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")

    def get_tools(self) -> dict[str, EnhancedToolSchema]:
        """Get available tools."""
        return self._tools.copy()

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def health_check(self) -> bool:
        """Perform health check."""
        if not self._connected:
            return False

        try:
            message = McpMessage(method="server/status")
            response = await self._send_message(message)
            return response.error is None
        except Exception:
            return False


class McpClientManager:
    """Manager for multiple MCP clients."""

    def __init__(self):
        self._clients: dict[str, McpClient] = {}
        self._registry: ToolRegistry | None = None

    async def add_server(self, config: McpServerConfig) -> bool:
        """Add and connect to an MCP server."""
        try:
            client = McpClient(config)
            if await client.connect():
                self._clients[config.name] = client

                # Register tools with registry if available
                if self._registry:
                    await self._register_client_tools(client)

                return True
            return False

        except Exception as e:
            logger.error(f"Failed to add MCP server {config.name}: {e}")
            return False

    async def remove_server(self, name: str) -> bool:
        """Remove and disconnect from an MCP server."""
        try:
            if name in self._clients:
                client = self._clients[name]
                await client.disconnect()

                # Unregister tools
                if self._registry:
                    await self._unregister_client_tools(client)

                del self._clients[name]
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to remove MCP server {name}: {e}")
            return False

    async def _register_client_tools(self, client: McpClient) -> None:
        """Register client tools with the tool registry."""
        for tool_schema in client.get_tools().values():
            try:
                mcp_tool = McpTool(tool_schema, client)
                await self._registry.register(mcp_tool)
            except Exception as e:
                logger.warning(f"Failed to register MCP tool {tool_schema.name}: {e}")

    async def _unregister_client_tools(self, client: McpClient) -> None:
        """Unregister client tools from the tool registry."""
        for tool_schema in client.get_tools().values():
            try:
                await self._registry.unregister(tool_schema.name)
            except Exception as e:
                logger.warning(f"Failed to unregister MCP tool {tool_schema.name}: {e}")

    def set_tool_registry(self, registry: ToolRegistry) -> None:
        """Set the tool registry for automatic tool registration."""
        self._registry = registry

    def get_client(self, name: str) -> McpClient | None:
        """Get MCP client by name."""
        return self._clients.get(name)

    def list_clients(self) -> list[str]:
        """List connected MCP clients."""
        return list(self._clients.keys())

    async def health_check_all(self) -> dict[str, bool]:
        """Perform health check on all clients."""
        results = {}
        for name, client in self._clients.items():
            results[name] = await client.health_check()
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all MCP clients."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting client: {e}")

        self._clients.clear()

    def get_all_tools(self) -> dict[str, dict[str, EnhancedToolSchema]]:
        """Get all tools from all clients."""
        all_tools = {}
        for name, client in self._clients.items():
            all_tools[name] = client.get_tools()
        return all_tools
