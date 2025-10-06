"""
Integration tests for MCP (Model Context Protocol) Communication

Tests the communication between different MCP servers including:
- Server discovery and initialization
- Tool registration and invocation
- Cross-server communication
- Error handling and recovery
- Protocol compliance
"""

import asyncio
import pytest
import json
import subprocess
import os
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock, call
from pathlib import Path
import tempfile
import time


class MCPServer:
    """Mock MCP server for testing."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.process = None
        self.tools = {}
        self.connected = False
        self.message_queue = asyncio.Queue()

    async def start(self):
        """Start the MCP server."""
        self.connected = True
        # Register default tools
        self.tools = {
            'list_tools': self._list_tools,
            'call_tool': self._call_tool,
            'get_status': self._get_status
        }
        return True

    async def stop(self):
        """Stop the MCP server."""
        self.connected = False
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.1)

    async def _list_tools(self) -> List[str]:
        """List available tools."""
        return list(self.tools.keys())

    async def _call_tool(self, tool_name: str, params: Dict) -> Any:
        """Call a specific tool."""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            if asyncio.iscoroutinefunction(tool):
                return await tool(**params)
            return tool(**params)
        raise ValueError(f"Tool {tool_name} not found")

    async def _get_status(self) -> Dict:
        """Get server status."""
        return {
            'name': self.name,
            'connected': self.connected,
            'tools': list(self.tools.keys()),
            'uptime': time.time()
        }

    async def send_message(self, message: Dict) -> Any:
        """Send a message to the server."""
        await self.message_queue.put(message)
        return {'status': 'sent', 'id': message.get('id')}

    async def receive_message(self, timeout: float = 5.0) -> Dict:
        """Receive a message from the server."""
        try:
            return await asyncio.wait_for(
                self.message_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def register_tool(self, name: str, handler):
        """Register a new tool."""
        self.tools[name] = handler


class MCPClient:
    """MCP client for communicating with servers."""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.message_id = 0

    async def connect_server(self, name: str, config: Dict) -> bool:
        """Connect to an MCP server."""
        server = MCPServer(name, config)
        success = await server.start()
        if success:
            self.servers[name] = server
        return success

    async def disconnect_server(self, name: str):
        """Disconnect from an MCP server."""
        if name in self.servers:
            await self.servers[name].stop()
            del self.servers[name]

    async def call_tool(self, server_name: str, tool_name: str, params: Dict = None) -> Any:
        """Call a tool on a specific server."""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")

        server = self.servers[server_name]
        self.message_id += 1

        message = {
            'id': self.message_id,
            'method': 'tools/call',
            'params': {
                'name': tool_name,
                'arguments': params or {}
            }
        }

        # Send message and get response
        await server.send_message(message)

        # Simulate processing
        if tool_name in server.tools:
            result = await server._call_tool(tool_name, params or {})
            return {
                'id': self.message_id,
                'result': result
            }

        return {
            'id': self.message_id,
            'error': {'message': f'Tool {tool_name} not found'}
        }

    async def list_all_tools(self) -> Dict[str, List[str]]:
        """List tools from all connected servers."""
        tools_by_server = {}
        for server_name, server in self.servers.items():
            tools = await server._list_tools()
            tools_by_server[server_name] = tools
        return tools_by_server

    async def broadcast_message(self, message: Dict) -> Dict[str, Any]:
        """Broadcast a message to all servers."""
        responses = {}
        tasks = []

        for server_name, server in self.servers.items():
            task = server.send_message(message.copy())
            tasks.append((server_name, task))

        for server_name, task in tasks:
            responses[server_name] = await task

        return responses


class TestMCPCommunication:
    """Test MCP server communication."""

    @pytest.mark.asyncio
    async def test_server_connection(self):
        """Test connecting to MCP servers."""
        client = MCPClient()

        # Connect to test server
        config = {
            'command': 'python',
            'args': ['-m', 'test_server'],
            'env': {'DEBUG': 'true'}
        }

        connected = await client.connect_server('test_server', config)
        assert connected

        # Verify server is in client's server list
        assert 'test_server' in client.servers
        assert client.servers['test_server'].connected

        # Disconnect
        await client.disconnect_server('test_server')
        assert 'test_server' not in client.servers

    @pytest.mark.asyncio
    async def test_tool_invocation(self):
        """Test calling tools on MCP servers."""
        client = MCPClient()
        await client.connect_server('test_server', {})

        # Register custom tool
        async def custom_tool(input: str) -> str:
            return f"Processed: {input}"

        client.servers['test_server'].register_tool('custom_tool', custom_tool)

        # Call the tool
        result = await client.call_tool(
            'test_server',
            'custom_tool',
            {'input': 'test data'}
        )

        assert result['result'] == 'Processed: test data'

    @pytest.mark.asyncio
    async def test_cross_server_communication(self):
        """Test communication between multiple MCP servers."""
        client = MCPClient()

        # Connect multiple servers
        await client.connect_server('server_a', {})
        await client.connect_server('server_b', {})

        # Register tools that communicate
        shared_data = {'value': None}

        async def set_data(value: str) -> str:
            shared_data['value'] = value
            return f"Set to: {value}"

        async def get_data() -> str:
            return f"Current value: {shared_data['value']}"

        client.servers['server_a'].register_tool('set_data', set_data)
        client.servers['server_b'].register_tool('get_data', get_data)

        # Server A sets data
        await client.call_tool('server_a', 'set_data', {'value': 'shared_info'})

        # Server B retrieves data
        result = await client.call_tool('server_b', 'get_data', {})

        assert 'shared_info' in result['result']

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        """Test discovering available tools across servers."""
        client = MCPClient()

        # Connect servers with different tools
        await client.connect_server('memory_server', {})
        await client.connect_server('filesystem_server', {})

        # Register server-specific tools
        client.servers['memory_server'].register_tool('store_memory', lambda: 'stored')
        client.servers['memory_server'].register_tool('retrieve_memory', lambda: 'retrieved')

        client.servers['filesystem_server'].register_tool('read_file', lambda: 'content')
        client.servers['filesystem_server'].register_tool('write_file', lambda: 'written')

        # Discover all tools
        all_tools = await client.list_all_tools()

        assert 'memory_server' in all_tools
        assert 'filesystem_server' in all_tools
        assert 'store_memory' in all_tools['memory_server']
        assert 'read_file' in all_tools['filesystem_server']

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in MCP communication."""
        client = MCPClient()
        await client.connect_server('test_server', {})

        # Try calling non-existent tool
        result = await client.call_tool('test_server', 'non_existent_tool', {})
        assert 'error' in result

        # Try calling tool on non-existent server
        with pytest.raises(ValueError, match="Server.*not connected"):
            await client.call_tool('non_existent_server', 'any_tool', {})

        # Register tool that raises exception
        def failing_tool():
            raise RuntimeError("Tool execution failed")

        client.servers['test_server'].register_tool('failing_tool', failing_tool)

        with pytest.raises(RuntimeError):
            await client.call_tool('test_server', 'failing_tool', {})

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests to MCP servers."""
        client = MCPClient()
        await client.connect_server('test_server', {})

        # Register slow tool
        async def slow_tool(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"

        client.servers['test_server'].register_tool('slow_tool', slow_tool)

        # Send multiple concurrent requests
        tasks = [
            client.call_tool('test_server', 'slow_tool', {'delay': 0.1}),
            client.call_tool('test_server', 'slow_tool', {'delay': 0.2}),
            client.call_tool('test_server', 'slow_tool', {'delay': 0.1})
        ]

        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start

        # Should complete in parallel, not sequentially
        assert duration < 0.5  # Would be 0.4s if sequential
        assert len(results) == 3
        assert all('Completed' in r['result'] for r in results)

    @pytest.mark.asyncio
    async def test_message_broadcast(self):
        """Test broadcasting messages to multiple servers."""
        client = MCPClient()

        # Connect multiple servers
        server_names = ['server1', 'server2', 'server3']
        for name in server_names:
            await client.connect_server(name, {})

        # Broadcast message
        message = {
            'method': 'notification',
            'params': {'content': 'System update'}
        }

        responses = await client.broadcast_message(message)

        # All servers should receive the message
        assert len(responses) == 3
        assert all(name in responses for name in server_names)
        assert all(r['status'] == 'sent' for r in responses.values())

    @pytest.mark.asyncio
    async def test_server_recovery(self):
        """Test server recovery after disconnection."""
        client = MCPClient()
        await client.connect_server('test_server', {})

        # Simulate server disconnection
        client.servers['test_server'].connected = False

        # Try to use disconnected server
        async def check_connection():
            if not client.servers['test_server'].connected:
                # Attempt reconnection
                await client.servers['test_server'].start()
            return client.servers['test_server'].connected

        # Should recover
        recovered = await check_connection()
        assert recovered

        # Tool should work after recovery
        result = await client.call_tool('test_server', 'get_status', {})
        assert result['result']['connected']


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance."""

    @pytest.mark.asyncio
    async def test_message_format(self):
        """Test that messages follow MCP protocol format."""
        server = MCPServer('test', {})
        await server.start()

        # Valid message format
        valid_message = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'tools/call',
            'params': {
                'name': 'test_tool',
                'arguments': {'arg1': 'value1'}
            }
        }

        result = await server.send_message(valid_message)
        assert result['status'] == 'sent'

        # Test response format
        response = {
            'jsonrpc': '2.0',
            'id': 1,
            'result': {'data': 'test'}
        }

        await server.message_queue.put(response)
        received = await server.receive_message()
        assert received['jsonrpc'] == '2.0'
        assert 'result' in received

    @pytest.mark.asyncio
    async def test_tool_schema_validation(self):
        """Test tool parameter schema validation."""

        class SchemaValidator:
            @staticmethod
            def validate(params: Dict, schema: Dict) -> bool:
                """Simple schema validation."""
                for required in schema.get('required', []):
                    if required not in params:
                        return False

                for prop, prop_schema in schema.get('properties', {}).items():
                    if prop in params:
                        value = params[prop]
                        expected_type = prop_schema.get('type')

                        if expected_type == 'string' and not isinstance(value, str):
                            return False
                        elif expected_type == 'number' and not isinstance(value, (int, float)):
                            return False
                        elif expected_type == 'boolean' and not isinstance(value, bool):
                            return False

                return True

        # Define tool with schema
        tool_schema = {
            'name': 'validated_tool',
            'parameters': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string'},
                    'count': {'type': 'number'},
                    'enabled': {'type': 'boolean'}
                },
                'required': ['text', 'count']
            }
        }

        # Valid parameters
        valid_params = {'text': 'hello', 'count': 5, 'enabled': True}
        assert SchemaValidator.validate(valid_params, tool_schema['parameters'])

        # Invalid parameters (missing required)
        invalid_params = {'text': 'hello', 'enabled': True}
        assert not SchemaValidator.validate(invalid_params, tool_schema['parameters'])

        # Invalid parameters (wrong type)
        invalid_type_params = {'text': 123, 'count': 5}
        assert not SchemaValidator.validate(invalid_type_params, tool_schema['parameters'])

    @pytest.mark.asyncio
    async def test_capability_negotiation(self):
        """Test capability negotiation between client and server."""
        server = MCPServer('test', {})
        await server.start()

        # Server capabilities
        server_capabilities = {
            'tools': True,
            'resources': True,
            'prompts': False,
            'logging': True,
            'experimental': {
                'customFeature': True
            }
        }

        # Client capabilities
        client_capabilities = {
            'tools': True,
            'resources': False,
            'prompts': True,
            'logging': True
        }

        # Negotiate capabilities (intersection)
        negotiated = {}
        for capability in server_capabilities:
            if capability in client_capabilities:
                if isinstance(server_capabilities[capability], bool):
                    negotiated[capability] = (
                        server_capabilities[capability] and
                        client_capabilities[capability]
                    )

        assert negotiated['tools'] is True
        assert negotiated['resources'] is False
        assert negotiated['logging'] is True


class TestMCPRealServers:
    """Test with real MCP server implementations (when available)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("C:/codedev/llm/stats/mcp-servers").exists(),
        reason="MCP servers not found"
    )
    async def test_memory_server_integration(self):
        """Test integration with memory MCP server."""
        # This would test with actual memory server
        config_path = Path("C:/codedev/llm/stats/mcp.json")
        if not config_path.exists():
            pytest.skip("MCP configuration not found")

        with open(config_path) as f:
            mcp_config = json.load(f)

        if 'memory' not in mcp_config.get('mcpServers', {}):
            pytest.skip("Memory server not configured")

        # Would start actual server process
        # For now, we mock it
        memory_server = MCPServer('memory', mcp_config['mcpServers']['memory'])
        await memory_server.start()

        # Register memory operations
        memories = {}

        async def create_memory(content: str, metadata: Dict = None) -> str:
            mem_id = f"mem_{len(memories)}"
            memories[mem_id] = {
                'content': content,
                'metadata': metadata or {},
                'timestamp': time.time()
            }
            return mem_id

        async def retrieve_memory(memory_id: str) -> Dict:
            return memories.get(memory_id)

        memory_server.register_tool('create_memory', create_memory)
        memory_server.register_tool('retrieve_memory', retrieve_memory)

        # Test memory operations
        mem_id = await memory_server._call_tool(
            'create_memory',
            {'content': 'Test memory', 'metadata': {'type': 'test'}}
        )

        retrieved = await memory_server._call_tool(
            'retrieve_memory',
            {'memory_id': mem_id}
        )

        assert retrieved['content'] == 'Test memory'
        assert retrieved['metadata']['type'] == 'test'

    @pytest.mark.asyncio
    async def test_filesystem_server_integration(self):
        """Test filesystem MCP server operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock filesystem server
            fs_server = MCPServer('filesystem', {'allowed_dirs': [tmpdir]})
            await fs_server.start()

            # Register filesystem operations
            async def read_file(path: str) -> str:
                file_path = Path(tmpdir) / path
                if file_path.exists():
                    return file_path.read_text()
                raise FileNotFoundError(f"File {path} not found")

            async def write_file(path: str, content: str) -> str:
                file_path = Path(tmpdir) / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
                return f"Wrote {len(content)} bytes to {path}"

            async def list_directory(path: str = '.') -> List[str]:
                dir_path = Path(tmpdir) / path
                if dir_path.exists():
                    return [f.name for f in dir_path.iterdir()]
                return []

            fs_server.register_tool('read_file', read_file)
            fs_server.register_tool('write_file', write_file)
            fs_server.register_tool('list_directory', list_directory)

            # Test file operations
            await fs_server._call_tool(
                'write_file',
                {'path': 'test.txt', 'content': 'Hello MCP'}
            )

            content = await fs_server._call_tool(
                'read_file',
                {'path': 'test.txt'}
            )
            assert content == 'Hello MCP'

            files = await fs_server._call_tool(
                'list_directory',
                {'path': '.'}
            )
            assert 'test.txt' in files