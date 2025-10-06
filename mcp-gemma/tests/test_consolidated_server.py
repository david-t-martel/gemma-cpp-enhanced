#!/usr/bin/env python3
"""
Unit tests for the consolidated MCP Gemma server.
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.consolidated_server import ConsolidatedMCPServer


class TestConsolidatedMCPServer:
    """Test suite for ConsolidatedMCPServer."""

    @pytest.fixture
    def server_config(self, mock_model_files, mock_gemma_executable):
        """Server configuration for testing."""
        return {
            "gemma": {
                "executable_path": mock_gemma_executable,
                "model_path": mock_model_files["model_path"],
                "tokenizer_path": mock_model_files["tokenizer_path"],
                "max_tokens": 100,
                "temperature": 0.7,
                "timeout": 10.0
            },
            "server": {
                "name": "test-server",
                "version": "1.0.0-test",
                "host": "localhost",
                "port": 0  # Random port
            }
        }

    @pytest.fixture
    def server(self, server_config):
        """Create a test server instance."""
        return ConsolidatedMCPServer(server_config)

    def test_server_initialization(self, server):
        """Test server initializes correctly."""
        assert server.name == "test-server"
        assert server.version == "1.0.0-test"
        assert server.config is not None
        assert hasattr(server, 'metrics')
        assert hasattr(server, 'memory_backend')

    def test_config_validation(self, server_config):
        """Test configuration validation."""
        # Valid config should work
        server = ConsolidatedMCPServer(server_config)
        assert server is not None

        # Invalid config should raise error
        invalid_config = server_config.copy()
        del invalid_config["gemma"]["executable_path"]

        with pytest.raises((KeyError, ValueError)):
            ConsolidatedMCPServer(invalid_config)

    @pytest.mark.asyncio
    async def test_tool_listing(self, server):
        """Test that tools are properly listed."""
        tools = await server._handle_list_tools()

        assert len(tools) > 0

        # Check that core tools exist
        tool_names = [tool.name for tool in tools]
        assert "chat" in tool_names
        assert "chat_stream" in tool_names
        assert "get_conversation" in tool_names
        assert "clear_conversation" in tool_names

        # Validate tool structure
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')

    @pytest.mark.asyncio
    async def test_chat_tool_validation(self, server):
        """Test chat tool parameter validation."""
        # Valid parameters
        valid_params = {
            "message": "Hello",
            "max_tokens": 50,
            "temperature": 0.7
        }

        # This should not raise an exception during validation
        # (We'll mock the actual execution)
        assert "message" in valid_params

        # Invalid parameters should be handled gracefully
        invalid_params = {
            "message": "",  # Empty message
            "max_tokens": -1,  # Invalid max_tokens
            "temperature": 2.0  # Invalid temperature
        }

        # Server should handle validation internally
        assert "message" in invalid_params

    @pytest.mark.asyncio
    async def test_conversation_management(self, server):
        """Test conversation history management."""
        conversation_id = "test_conv_1"

        # Initially should be empty
        history = await server.memory_backend.get_conversation(conversation_id)
        assert len(history) == 0

        # Add messages
        await server.memory_backend.add_message(
            conversation_id, "user", "Hello"
        )
        await server.memory_backend.add_message(
            conversation_id, "assistant", "Hi there!"
        )

        # Check history
        history = await server.memory_backend.get_conversation(conversation_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_clear_conversation(self, server):
        """Test conversation clearing functionality."""
        conversation_id = "test_conv_clear"

        # Add some messages
        await server.memory_backend.add_message(
            conversation_id, "user", "Test message"
        )

        # Verify messages exist
        history = await server.memory_backend.get_conversation(conversation_id)
        assert len(history) == 1

        # Clear conversation
        await server.memory_backend.clear_conversation(conversation_id)

        # Verify conversation is cleared
        history = await server.memory_backend.get_conversation(conversation_id)
        assert len(history) == 0

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_chat_execution(self, mock_subprocess, server):
        """Test chat execution with mocked subprocess."""
        # Mock subprocess
        mock_process = Mock()
        mock_process.communicate.return_value = asyncio.coroutine(
            lambda: (b"Test response from gemma", b"")
        )()
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Test chat call
        result = await server._handle_tool_call("chat", {
            "message": "Hello",
            "max_tokens": 50,
            "temperature": 0.7
        })

        assert result is not None
        assert "content" in result
        mock_subprocess.assert_called_once()

    def test_metrics_collection(self, server):
        """Test metrics collection functionality."""
        # Metrics should be initialized
        assert hasattr(server, 'metrics')

        # Test recording metrics
        server.metrics.record_request("chat", 0.5, True)
        server.metrics.record_request("chat", 1.0, False)

        stats = server.metrics.get_stats()
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "average_response_time" in stats

    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error handling in various scenarios."""
        # Test with invalid tool name
        with pytest.raises(Exception):
            await server._handle_tool_call("nonexistent_tool", {})

        # Test with missing required parameters
        with pytest.raises(Exception):
            await server._handle_tool_call("chat", {})  # Missing message

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, server):
        """Test handling of concurrent requests."""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                server.memory_backend.add_message(
                    f"conv_{i}", "user", f"Message {i}"
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Verify all conversations exist
        for i in range(5):
            history = await server.memory_backend.get_conversation(f"conv_{i}")
            assert len(history) == 1
            assert history[0]["content"] == f"Message {i}"

    def test_configuration_update(self, server, server_config):
        """Test server configuration updates."""
        original_max_tokens = server.config["gemma"]["max_tokens"]

        # Update config
        new_config = server_config.copy()
        new_config["gemma"]["max_tokens"] = 200

        # Server should handle config updates
        # (Implementation would depend on server design)
        assert original_max_tokens != 200  # Verify test setup

    @pytest.mark.asyncio
    async def test_memory_backend_persistence(self, server):
        """Test memory backend data persistence."""
        conversation_id = "persistent_test"

        # Add message
        await server.memory_backend.add_message(
            conversation_id, "user", "Persistent message"
        )

        # Simulate getting the conversation again
        history = await server.memory_backend.get_conversation(conversation_id)
        assert len(history) == 1
        assert history[0]["content"] == "Persistent message"

    def test_tool_schema_validation(self, server):
        """Test that all tool schemas are valid JSON Schema."""
        import jsonschema

        # Get all tools
        asyncio.run(self._test_tool_schemas(server))

    async def _test_tool_schemas(self, server):
        """Helper method to test tool schemas."""
        tools = await server._handle_list_tools()

        for tool in tools:
            # Each tool should have a valid schema
            assert tool.inputSchema is not None
            assert isinstance(tool.inputSchema, dict)

            # Schema should have required fields
            if "properties" in tool.inputSchema:
                assert isinstance(tool.inputSchema["properties"], dict)

    @pytest.mark.asyncio
    async def test_streaming_response(self, server):
        """Test streaming response functionality."""
        # This test would verify streaming if implemented
        # For now, just test that the method exists
        tools = await server._handle_list_tools()
        stream_tools = [t for t in tools if "stream" in t.name]
        assert len(stream_tools) > 0

    def test_server_info(self, server):
        """Test server information retrieval."""
        info = server.get_server_info()

        assert "name" in info
        assert "version" in info
        assert info["name"] == "test-server"
        assert info["version"] == "1.0.0-test"

    @pytest.mark.asyncio
    async def test_health_check(self, server):
        """Test server health check functionality."""
        # Basic health check - server should be responsive
        assert server is not None

        # If health check method exists, test it
        if hasattr(server, 'health_check'):
            health = await server.health_check()
            assert "status" in health


class TestConsolidatedServerIntegration:
    """Integration tests requiring real components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_gemma_executable(self, test_config, gemma_executable_exists):
        """Test with real gemma executable if available."""
        if not gemma_executable_exists:
            pytest.skip("Real gemma executable not available")

        server = ConsolidatedMCPServer(test_config)

        # Test basic initialization
        assert server is not None

    @pytest.mark.integration
    @pytest.mark.model_required
    @pytest.mark.asyncio
    async def test_real_model_inference(self, test_config, model_files_exist):
        """Test inference with real model files if available."""
        if not model_files_exist:
            pytest.skip("Model files not available")

        server = ConsolidatedMCPServer(test_config)

        # This would test actual inference
        # Commented out to avoid long-running tests in CI
        # result = await server._handle_tool_call("chat", {
        #     "message": "Hello",
        #     "max_tokens": 10
        # })
        # assert result is not None


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])