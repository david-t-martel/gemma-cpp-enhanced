"""Comprehensive unit tests for MCP (Model Context Protocol) client infrastructure.

Tests all MCP client functionality including protocol handling, tool execution,
connection management, message handling, and error scenarios with proper mocking.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import aiohttp
import pytest
import websockets

from src.domain.tools.base import ToolExecutionContext, ToolResult
from src.domain.tools.schemas import EnhancedToolSchema, ParameterSchema, ToolCategory, ToolType
from src.infrastructure.mcp.client import (
    McpClient,
    McpClientManager,
    McpMessage,
    McpResponse,
    McpServerConfig,
    McpTool,
)


class TestMcpServerConfig:
    """Test suite for MCP server configuration."""

    def test_config_default_initialization(self):
        """Test default configuration values."""
        config = McpServerConfig(name="test-server", url="ws://localhost:8080")

        assert config.name == "test-server"
        assert config.url == "ws://localhost:8080"
        assert config.protocol == "websocket"
        assert config.auth_token is None
        assert config.timeout == 30
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.heartbeat_interval == 30
        assert config.max_concurrent_requests == 10
        assert config.capabilities == []
        assert config.metadata == {}

    def test_config_custom_initialization(self):
        """Test configuration with custom values."""
        config = McpServerConfig(
            name="custom-server",
            url="http://api.example.com",
            protocol="http",
            auth_token="secret-token",
            timeout=60,
            retry_attempts=5,
            capabilities=["tools", "resources"],
            metadata={"version": "1.0", "environment": "production"}
        )

        assert config.name == "custom-server"
        assert config.url == "http://api.example.com"
        assert config.protocol == "http"
        assert config.auth_token == "secret-token"
        assert config.timeout == 60
        assert config.retry_attempts == 5
        assert config.capabilities == ["tools", "resources"]
        assert config.metadata == {"version": "1.0", "environment": "production"}


class TestMcpMessage:
    """Test suite for MCP message structure."""

    def test_message_default_creation(self):
        """Test message creation with defaults."""
        message = McpMessage(method="test_method")

        assert message.method == "test_method"
        assert message.params == {}
        assert isinstance(message.id, str)
        assert len(message.id) > 0
        assert isinstance(message.timestamp, float)
        assert message.timestamp > 0

    def test_message_custom_creation(self):
        """Test message creation with custom values."""
        custom_id = "custom-id-123"
        custom_timestamp = time.time()
        params = {"param1": "value1", "param2": 42}

        message = McpMessage(
            id=custom_id,
            method="custom_method",
            params=params,
            timestamp=custom_timestamp
        )

        assert message.id == custom_id
        assert message.method == "custom_method"
        assert message.params == params
        assert message.timestamp == custom_timestamp


class TestMcpResponse:
    """Test suite for MCP response structure."""

    def test_response_success_creation(self):
        """Test successful response creation."""
        result_data = {"output": "success", "value": 42}
        response = McpResponse(id="test-id", result=result_data)

        assert response.id == "test-id"
        assert response.result == result_data
        assert response.error is None
        assert isinstance(response.timestamp, float)

    def test_response_error_creation(self):
        """Test error response creation."""
        error_data = {"code": 500, "message": "Internal error"}
        response = McpResponse(id="test-id", error=error_data)

        assert response.id == "test-id"
        assert response.result is None
        assert response.error == error_data

    def test_response_both_result_and_error(self):
        """Test response with both result and error."""
        # This might be invalid according to MCP spec, but we test the model handles it
        response = McpResponse(
            id="test-id",
            result={"data": "result"},
            error={"code": 400, "message": "error"}
        )

        assert response.result == {"data": "result"}
        assert response.error == {"code": 400, "message": "error"}


class TestMcpTool:
    """Test suite for MCP tool wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parameter_schema = ParameterSchema(
            name="input_text",
            type="string",
            description="Text to process",
            required=True
        )

        self.tool_schema = EnhancedToolSchema(
            name="test_tool",
            description="Test tool for MCP",
            category=ToolCategory.CUSTOM,
            type=ToolType.MCP,
            parameters=[self.parameter_schema],
            metadata={"server": "test-server"}
        )

        self.mock_client = Mock(spec=McpClient)
        self.mcp_tool = McpTool(self.tool_schema, self.mock_client)

    def test_tool_schema_property(self):
        """Test tool schema property."""
        assert self.mcp_tool.schema == self.tool_schema

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        context = ToolExecutionContext(
            execution_id="exec-123",
            agent_id="agent-456",
            session_id="session-789",
            timeout=30.0,
            security_level="standard"
        )

        mock_response = McpResponse(
            id="response-id",
            result={"output": "Tool executed successfully", "status": "completed"}
        )

        self.mock_client.call_tool = AsyncMock(return_value=mock_response)

        result = await self.mcp_tool.execute(context, input_text="test input")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == {"output": "Tool executed successfully", "status": "completed"}
        assert result.error is None
        assert result.context == context
        assert result.metadata["mcp_server"] == "test-server"
        assert result.execution_time > 0

        # Verify client was called correctly
        self.mock_client.call_tool.assert_called_once_with(
            "test_tool",
            {"input_text": "test input"},
            context
        )

    @pytest.mark.asyncio
    async def test_execute_error_response(self):
        """Test tool execution with error response."""
        context = ToolExecutionContext(
            execution_id="exec-123",
            agent_id="agent-456",
            session_id="session-789"
        )

        mock_response = McpResponse(
            id="response-id",
            error={"code": 400, "message": "Invalid input parameter"}
        )

        self.mock_client.call_tool = AsyncMock(return_value=mock_response)

        result = await self.mcp_tool.execute(context, input_text="invalid")

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error == "Invalid input parameter"
        assert result.data is None
        assert result.context == context
        assert result.metadata["mcp_server"] == "test-server"

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        """Test tool execution with client exception."""
        context = ToolExecutionContext(execution_id="exec-123")

        self.mock_client.call_tool = AsyncMock(side_effect=Exception("Connection error"))

        result = await self.mcp_tool.execute(context, input_text="test")

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Connection error" in result.error
        assert result.execution_time == 0
        assert result.metadata["mcp_server"] == "test-server"


class TestMcpClient:
    """Test suite for MCP client."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = McpServerConfig(
            name="test-server",
            url="ws://localhost:8080",
            protocol="websocket",
            auth_token="test-token"
        )
        self.client = McpClient(self.config)

    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.config == self.config
        assert self.client._connection is None
        assert self.client._connected is False
        assert self.client._tools == {}
        assert self.client._pending_requests == {}
        assert self.client._heartbeat_task is None
        assert self.client._semaphore._value == 10  # max_concurrent_requests
        assert self.client._request_id_counter == 0

    @pytest.mark.asyncio
    async def test_connect_websocket_success(self):
        """Test successful WebSocket connection."""
        mock_websocket = AsyncMock()

        with patch('websockets.connect', return_value=mock_websocket) as mock_connect, \
             patch.object(self.client, '_initialize_connection') as mock_init, \
             patch.object(self.client, '_handle_websocket_messages') as mock_handle:

            mock_init.return_value = None
            result = await self.client.connect()

            assert result is True
            assert self.client._connected is True
            assert self.client._connection == mock_websocket

            # Verify connection parameters
            mock_connect.assert_called_once_with(
                "ws://localhost:8080",
                extra_headers={"Authorization": "Bearer test-token"},
                ping_interval=30,
                ping_timeout=30,
                close_timeout=30
            )

    @pytest.mark.asyncio
    async def test_connect_http_success(self):
        """Test successful HTTP connection."""
        self.client.config.protocol = "http"

        with patch('aiohttp.ClientSession') as mock_session_class, \
             patch.object(self.client, '_initialize_connection') as mock_init:

            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_init.return_value = None

            result = await self.client.connect()

            assert result is True
            assert self.client._connected is True
            assert self.client._connection == mock_session

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            result = await self.client.connect()

            assert result is False
            assert self.client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self):
        """Test disconnection from WebSocket."""
        mock_websocket = AsyncMock()
        mock_heartbeat_task = AsyncMock()

        self.client._connection = mock_websocket
        self.client._connected = True
        self.client._heartbeat_task = mock_heartbeat_task
        self.client._pending_requests = {"req1": AsyncMock(), "req2": AsyncMock()}

        await self.client.disconnect()

        assert self.client._connected is False
        mock_heartbeat_task.cancel.assert_called_once()
        mock_websocket.close.assert_called_once()

        # Verify pending requests were cancelled
        for future in self.client._pending_requests.values():
            future.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_connection(self):
        """Test connection initialization."""
        mock_init_response = McpResponse(
            id="init-id",
            result={"protocol_version": "2024-11-05", "capabilities": {"tools": True}}
        )

        with patch.object(self.client, '_send_message', return_value=mock_init_response) as mock_send, \
             patch.object(self.client, '_get_server_capabilities') as mock_capabilities, \
             patch.object(self.client, '_list_server_tools') as mock_list_tools:

            await self.client._initialize_connection()

            # Verify initialization message was sent
            call_args = mock_send.call_args[0][0]
            assert call_args.method == "initialize"
            assert call_args.params["protocol_version"] == "2024-11-05"

            mock_capabilities.assert_called_once()
            mock_list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_server_tools(self):
        """Test listing server tools."""
        mock_tools_response = McpResponse(
            id="tools-id",
            result={
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Perform calculations",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string", "description": "Math expression"}
                            },
                            "required": ["expression"]
                        }
                    },
                    {
                        "name": "weather",
                        "description": "Get weather information",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "Location name"}
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        )

        with patch.object(self.client, '_send_message', return_value=mock_tools_response):
            await self.client._list_server_tools()

            assert len(self.client._tools) == 2
            assert "calculator" in self.client._tools
            assert "weather" in self.client._tools

            # Verify tool schema conversion
            calc_tool = self.client._tools["calculator"]
            assert calc_tool.name == "calculator"
            assert calc_tool.description == "Perform calculations"
            assert calc_tool.category == ToolCategory.CUSTOM
            assert calc_tool.type == ToolType.MCP
            assert len(calc_tool.parameters) == 1
            assert calc_tool.parameters[0].name == "expression"
            assert calc_tool.parameters[0].required is True

    def test_convert_mcp_tool_to_schema(self):
        """Test conversion of MCP tool format to schema format."""
        mcp_tool_data = {
            "name": "file_reader",
            "description": "Read file contents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file",
                        "pattern": "^/.*"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8",
                        "enum": ["utf-8", "ascii", "latin1"]
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to read",
                        "minimum": 1,
                        "maximum": 10000
                    }
                },
                "required": ["file_path"]
            }
        }

        schema = self.client._convert_mcp_tool_to_schema(mcp_tool_data)

        assert schema.name == "file_reader"
        assert schema.description == "Read file contents"
        assert schema.category == ToolCategory.CUSTOM
        assert schema.type == ToolType.MCP
        assert len(schema.parameters) == 3

        # Check file_path parameter
        file_path_param = next(p for p in schema.parameters if p.name == "file_path")
        assert file_path_param.type == "string"
        assert file_path_param.required is True
        assert file_path_param.pattern == "^/.*"

        # Check encoding parameter
        encoding_param = next(p for p in schema.parameters if p.name == "encoding")
        assert encoding_param.type == "string"
        assert encoding_param.required is False
        assert encoding_param.default == "utf-8"
        assert encoding_param.enum == ["utf-8", "ascii", "latin1"]

        # Check max_lines parameter
        max_lines_param = next(p for p in schema.parameters if p.name == "max_lines")
        assert max_lines_param.type == "integer"
        assert max_lines_param.minimum == 1
        assert max_lines_param.maximum == 10000

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        self.client._connected = True
        self.client._tools = {
            "test_tool": EnhancedToolSchema(
                name="test_tool",
                description="Test tool",
                category=ToolCategory.CUSTOM,
                type=ToolType.MCP,
                parameters=[]
            )
        }

        context = ToolExecutionContext(execution_id="exec-123")
        mock_response = McpResponse(
            id="tool-call-id",
            result={"output": "Tool result", "success": True}
        )

        with patch.object(self.client, '_send_message', return_value=mock_response):
            response = await self.client.call_tool("test_tool", {"param": "value"}, context)

            assert response == mock_response

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        """Test tool call when not connected."""
        with pytest.raises(RuntimeError) as exc_info:
            await self.client.call_tool("test_tool", {}, None)

        assert "Not connected to MCP server" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_unavailable(self):
        """Test tool call for unavailable tool."""
        self.client._connected = True

        with pytest.raises(ValueError) as exc_info:
            await self.client.call_tool("nonexistent_tool", {}, None)

        assert "Tool 'nonexistent_tool' not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_websocket_message(self):
        """Test sending message via WebSocket."""
        mock_websocket = AsyncMock()
        self.client._connection = mock_websocket
        self.client.config.protocol = "websocket"

        message_data = {"id": "test-id", "method": "test_method", "params": {}}

        # Mock future for response
        mock_future = asyncio.Future()
        mock_future.set_result({"id": "test-id", "result": {"success": True}})

        with patch('asyncio.Future', return_value=mock_future), \
             patch('asyncio.wait_for', return_value={"id": "test-id", "result": {"success": True}}):

            response = await self.client._send_websocket_message(message_data)

            assert isinstance(response, McpResponse)
            assert response.result == {"success": True}
            mock_websocket.send.assert_called_once_with(json.dumps(message_data))

    @pytest.mark.asyncio
    async def test_send_http_message(self):
        """Test sending message via HTTP."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"id": "test-id", "result": {"success": True}}
        mock_session.post.return_value.__aenter__.return_value = mock_response

        self.client._connection = mock_session
        self.client.config.protocol = "http"

        message_data = {"id": "test-id", "method": "test_method", "params": {}}

        response = await self.client._send_http_message(message_data)

        assert isinstance(response, McpResponse)
        assert response.result == {"success": True}
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_websocket_messages(self):
        """Test WebSocket message handling."""
        mock_websocket = AsyncMock()

        # Mock message iterator
        messages = [
            '{"id": "req1", "result": {"data": "response1"}}',
            '{"method": "tools/list_changed", "params": {}}',
            'invalid json',  # Should be handled gracefully
            '{"id": "req2", "result": {"data": "response2"}}'
        ]

        async def mock_messages():
            for msg in messages:
                yield msg

        mock_websocket.__aiter__.return_value = mock_messages()
        self.client._connection = mock_websocket

        # Set up pending requests
        future1 = asyncio.Future()
        future2 = asyncio.Future()
        self.client._pending_requests = {"req1": future1, "req2": future2}

        with patch.object(self.client, '_list_server_tools') as mock_refresh_tools:
            # Start message handling (it will process all messages)
            task = asyncio.create_task(self.client._handle_websocket_messages())

            # Give it time to process messages
            await asyncio.sleep(0.1)
            task.cancel()

            # Check that futures were resolved
            assert future1.done()
            assert future2.done()
            assert future1.result() == {"id": "req1", "result": {"data": "response1"}}
            assert future2.result() == {"id": "req2", "result": {"data": "response2"}}

            # Verify tools refresh was called due to tools/list_changed notification
            mock_refresh_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_loop_websocket(self):
        """Test heartbeat loop for WebSocket connection."""
        mock_websocket = AsyncMock()
        self.client._connection = mock_websocket
        self.client._connected = True
        self.client.config.protocol = "websocket"
        self.client.config.heartbeat_interval = 0.1  # Fast for testing

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.client._heartbeat_loop())

        # Let it run a couple of cycles
        await asyncio.sleep(0.25)

        # Stop the task
        heartbeat_task.cancel()

        # Verify ping was called multiple times
        assert mock_websocket.ping.call_count >= 2

    @pytest.mark.asyncio
    async def test_heartbeat_loop_http(self):
        """Test heartbeat loop for HTTP connection."""
        self.client.config.protocol = "http"
        self.client.config.heartbeat_interval = 0.1  # Fast for testing
        self.client._connected = True

        mock_response = McpResponse(id="ping-id", result={"pong": True})

        with patch.object(self.client, '_send_message', return_value=mock_response) as mock_send:
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self.client._heartbeat_loop())

            # Let it run a couple of cycles
            await asyncio.sleep(0.25)

            # Stop the task
            heartbeat_task.cancel()

            # Verify ping message was sent multiple times
            assert mock_send.call_count >= 2
            # Check that ping method was called
            ping_calls = [call for call in mock_send.call_args_list if call[0][0].method == "ping"]
            assert len(ping_calls) >= 2

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        self.client._connected = True
        mock_response = McpResponse(id="health-id", result={"status": "healthy"})

        with patch.object(self.client, '_send_message', return_value=mock_response):
            result = await self.client.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        self.client._connected = True

        with patch.object(self.client, '_send_message', side_effect=Exception("Health check failed")):
            result = await self.client.health_check()

            assert result is False

    def test_health_check_not_connected(self):
        """Test health check when not connected."""
        result = asyncio.run(self.client.health_check())
        assert result is False

    def test_get_tools(self):
        """Test getting available tools."""
        mock_tool = EnhancedToolSchema(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.CUSTOM,
            type=ToolType.MCP,
            parameters=[]
        )

        self.client._tools = {"test_tool": mock_tool}

        tools = self.client.get_tools()

        assert len(tools) == 1
        assert "test_tool" in tools
        assert tools["test_tool"] == mock_tool
        # Verify it returns a copy
        assert tools is not self.client._tools

    def test_is_connected(self):
        """Test connection status check."""
        assert self.client.is_connected() is False

        self.client._connected = True
        assert self.client.is_connected() is True


class TestMcpClientManager:
    """Test suite for MCP client manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = McpClientManager()

    @pytest.mark.asyncio
    async def test_add_server_success(self):
        """Test successful server addition."""
        config = McpServerConfig(name="test-server", url="ws://localhost:8080")

        with patch('src.infrastructure.mcp.client.McpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect.return_value = True
            mock_client_class.return_value = mock_client

            result = await self.manager.add_server(config)

            assert result is True
            assert "test-server" in self.manager._clients
            assert self.manager._clients["test-server"] == mock_client

    @pytest.mark.asyncio
    async def test_add_server_failure(self):
        """Test failed server addition."""
        config = McpServerConfig(name="test-server", url="ws://localhost:8080")

        with patch('src.infrastructure.mcp.client.McpClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect.return_value = False
            mock_client_class.return_value = mock_client

            result = await self.manager.add_server(config)

            assert result is False
            assert "test-server" not in self.manager._clients

    @pytest.mark.asyncio
    async def test_remove_server_success(self):
        """Test successful server removal."""
        mock_client = AsyncMock()
        self.manager._clients["test-server"] = mock_client

        result = await self.manager.remove_server("test-server")

        assert result is True
        assert "test-server" not in self.manager._clients
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_server_not_found(self):
        """Test removal of non-existent server."""
        result = await self.manager.remove_server("nonexistent-server")

        assert result is False

    @pytest.mark.asyncio
    async def test_register_client_tools(self):
        """Test tool registration with registry."""
        from src.domain.tools.base import ToolRegistry

        mock_registry = AsyncMock(spec=ToolRegistry)
        self.manager.set_tool_registry(mock_registry)

        mock_client = Mock()
        mock_tool_schema = EnhancedToolSchema(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.CUSTOM,
            type=ToolType.MCP,
            parameters=[]
        )
        mock_client.get_tools.return_value = {"test_tool": mock_tool_schema}

        await self.manager._register_client_tools(mock_client)

        # Verify tool was registered
        mock_registry.register.assert_called_once()
        registered_tool = mock_registry.register.call_args[0][0]
        assert isinstance(registered_tool, McpTool)
        assert registered_tool.schema == mock_tool_schema

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health check of all clients."""
        mock_client1 = AsyncMock()
        mock_client1.health_check.return_value = True
        mock_client2 = AsyncMock()
        mock_client2.health_check.return_value = False

        self.manager._clients = {
            "server1": mock_client1,
            "server2": mock_client2
        }

        results = await self.manager.health_check_all()

        assert results == {"server1": True, "server2": False}

    def test_get_client(self):
        """Test getting client by name."""
        mock_client = Mock()
        self.manager._clients["test-server"] = mock_client

        result = self.manager.get_client("test-server")
        assert result == mock_client

        result = self.manager.get_client("nonexistent")
        assert result is None

    def test_list_clients(self):
        """Test listing client names."""
        self.manager._clients = {
            "server1": Mock(),
            "server2": Mock(),
            "server3": Mock()
        }

        result = self.manager.list_clients()

        assert set(result) == {"server1", "server2", "server3"}

    def test_get_all_tools(self):
        """Test getting tools from all clients."""
        mock_tool1 = EnhancedToolSchema(
            name="tool1",
            description="Tool 1",
            category=ToolCategory.CUSTOM,
            type=ToolType.MCP,
            parameters=[]
        )
        mock_tool2 = EnhancedToolSchema(
            name="tool2",
            description="Tool 2",
            category=ToolCategory.CUSTOM,
            type=ToolType.MCP,
            parameters=[]
        )

        mock_client1 = Mock()
        mock_client1.get_tools.return_value = {"tool1": mock_tool1}
        mock_client2 = Mock()
        mock_client2.get_tools.return_value = {"tool2": mock_tool2}

        self.manager._clients = {
            "server1": mock_client1,
            "server2": mock_client2
        }

        result = self.manager.get_all_tools()

        assert "server1" in result
        assert "server2" in result
        assert result["server1"]["tool1"] == mock_tool1
        assert result["server2"]["tool2"] == mock_tool2

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """Test disconnecting all clients."""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        mock_client2.disconnect.side_effect = Exception("Disconnect error")

        self.manager._clients = {
            "server1": mock_client1,
            "server2": mock_client2
        }

        await self.manager.disconnect_all()

        # All clients should be removed even if one fails
        assert len(self.manager._clients) == 0
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = McpServerConfig(name="test-server", url="ws://localhost:8080")
        self.client = McpClient(self.config)

    @pytest.mark.asyncio
    async def test_connection_lost_during_operation(self):
        """Test handling of connection loss during operation."""
        self.client._connected = True

        with patch.object(self.client, '_send_message', side_effect=Exception("Connection lost")):
            with pytest.raises(Exception) as exc_info:
                await self.client.call_tool("test_tool", {}, None)

            assert "Connection lost" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_websocket_connection_closed(self):
        """Test handling of WebSocket connection closure."""
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.side_effect = websockets.exceptions.ConnectionClosed(1000, "Normal closure")

        self.client._connection = mock_websocket
        self.client._connected = True

        # This should handle the exception gracefully
        await self.client._handle_websocket_messages()

        assert self.client._connected is False

    @pytest.mark.asyncio
    async def test_invalid_server_response(self):
        """Test handling of invalid server responses."""
        self.client._connected = True

        # Mock a response without proper structure
        invalid_response = McpResponse(id="test-id", result=None, error=None)

        with patch.object(self.client, '_send_message', return_value=invalid_response):
            # This should not crash but handle gracefully
            response = await self.client.call_tool("test_tool", {}, None)
            assert response == invalid_response

    def test_tool_schema_conversion_errors(self):
        """Test handling of malformed tool schemas."""
        # Test with missing required fields
        malformed_tool = {
            "name": "malformed_tool",
            # Missing description
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }

        # Should handle gracefully with empty description
        schema = self.client._convert_mcp_tool_to_schema(malformed_tool)
        assert schema.name == "malformed_tool"
        assert schema.description == ""

    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self):
        """Test concurrent request limiting."""
        self.client._semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent requests
        self.client._connected = True

        mock_response = McpResponse(id="test-id", result={"success": True})

        call_count = 0

        async def mock_send_message(message):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate network delay
            return mock_response

        with patch.object(self.client, '_send_message', side_effect=mock_send_message):
            # Start 5 concurrent requests
            tasks = [
                self.client.call_tool("test_tool", {"param": i}, None)
                for i in range(5)
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete successfully despite the limit
            assert len(responses) == 5
            # The semaphore should have controlled concurrency
            assert call_count == 5

    @pytest.mark.asyncio
    async def test_heartbeat_failure_disconnection(self):
        """Test disconnection on heartbeat failure."""
        mock_websocket = AsyncMock()
        mock_websocket.ping.side_effect = Exception("Ping failed")

        self.client._connection = mock_websocket
        self.client._connected = True
        self.client.config.protocol = "websocket"
        self.client.config.heartbeat_interval = 0.1

        # Start heartbeat and let it fail
        heartbeat_task = asyncio.create_task(self.client._heartbeat_loop())

        # Wait for heartbeat to fail
        await asyncio.sleep(0.2)

        # Should have disconnected due to ping failure
        assert self.client._connected is False

        heartbeat_task.cancel()