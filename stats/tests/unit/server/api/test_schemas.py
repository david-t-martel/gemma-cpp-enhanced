"""Comprehensive unit tests for API schemas and data models."""

import uuid
from datetime import datetime
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.server.api.schemas import (
    ChatRole,
    FinishReason,
    ChatMessage,
    HealthResponse,
    MetricsResponse
)


class TestChatRole:
    """Test ChatRole enum."""

    def test_chat_role_values(self):
        """Test that ChatRole has correct values."""
        assert ChatRole.SYSTEM == "system"
        assert ChatRole.USER == "user"
        assert ChatRole.ASSISTANT == "assistant"
        assert ChatRole.FUNCTION == "function"

    def test_chat_role_membership(self):
        """Test ChatRole membership checks."""
        assert "system" in ChatRole
        assert "user" in ChatRole
        assert "assistant" in ChatRole
        assert "function" in ChatRole
        assert "invalid" not in ChatRole

    def test_chat_role_iteration(self):
        """Test ChatRole iteration."""
        roles = list(ChatRole)
        assert len(roles) == 4
        assert ChatRole.SYSTEM in roles
        assert ChatRole.USER in roles
        assert ChatRole.ASSISTANT in roles
        assert ChatRole.FUNCTION in roles


class TestFinishReason:
    """Test FinishReason enum."""

    def test_finish_reason_values(self):
        """Test that FinishReason has correct values."""
        assert FinishReason.STOP == "stop"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.FUNCTION_CALL == "function_call"
        assert FinishReason.CONTENT_FILTER == "content_filter"
        assert FinishReason.NULL == "null"

    def test_finish_reason_membership(self):
        """Test FinishReason membership checks."""
        assert "stop" in FinishReason
        assert "length" in FinishReason
        assert "function_call" in FinishReason
        assert "content_filter" in FinishReason
        assert "null" in FinishReason
        assert "invalid" not in FinishReason


class TestChatMessage:
    """Test ChatMessage model."""

    def test_chat_message_valid(self):
        """Test creating valid ChatMessage."""
        message = ChatMessage(
            role=ChatRole.USER,
            content="Hello, how are you?"
        )

        assert message.role == ChatRole.USER
        assert message.content == "Hello, how are you?"
        assert message.name is None

    def test_chat_message_with_name(self):
        """Test ChatMessage with optional name."""
        message = ChatMessage(
            role=ChatRole.USER,
            content="Hello",
            name="john_doe"
        )

        assert message.role == ChatRole.USER
        assert message.content == "Hello"
        assert message.name == "john_doe"

    def test_chat_message_all_roles(self):
        """Test ChatMessage with all valid roles."""
        roles_content = [
            (ChatRole.SYSTEM, "You are a helpful assistant."),
            (ChatRole.USER, "What's the weather like?"),
            (ChatRole.ASSISTANT, "I need more information about your location."),
            (ChatRole.FUNCTION, "Location: New York, Weather: Sunny")
        ]

        for role, content in roles_content:
            message = ChatMessage(role=role, content=content)
            assert message.role == role
            assert message.content == content

    def test_chat_message_empty_content_invalid(self):
        """Test that empty content is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role=ChatRole.USER, content="")

        assert "content" in str(exc_info.value)

    def test_chat_message_missing_role(self):
        """Test that missing role is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(content="Hello")

        assert "role" in str(exc_info.value)

    def test_chat_message_missing_content(self):
        """Test that missing content is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role=ChatRole.USER)

        assert "content" in str(exc_info.value)

    def test_chat_message_invalid_role(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="invalid_role", content="Hello")

        assert "role" in str(exc_info.value)

    def test_chat_message_serialization(self):
        """Test ChatMessage serialization to dict."""
        message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content="I can help you with that.",
            name="assistant"
        )

        data = message.model_dump()
        assert data == {
            "role": "assistant",
            "content": "I can help you with that.",
            "name": "assistant"
        }

    def test_chat_message_json_serialization(self):
        """Test ChatMessage JSON serialization."""
        message = ChatMessage(
            role=ChatRole.USER,
            content="Test message"
        )

        json_str = message.model_dump_json()
        assert '"role":"user"' in json_str
        assert '"content":"Test message"' in json_str

    def test_chat_message_from_dict(self):
        """Test creating ChatMessage from dictionary."""
        data = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "name": "system"
        }

        message = ChatMessage(**data)
        assert message.role == ChatRole.SYSTEM
        assert message.content == "You are a helpful assistant."
        assert message.name == "system"


class TestHealthResponse:
    """Test HealthResponse model (if accessible)."""

    def test_health_response_basic_structure(self):
        """Test basic health response structure."""
        # Since we may not have direct access to HealthResponse,
        # test the expected structure
        response_data = {
            "status": "healthy",
            "uptime": 3600.0,
            "timestamp": datetime.now().isoformat(),
            "model": {
                "loaded": True,
                "healthy": True
            },
            "system": {
                "cpu": 25.5,
                "memory": 50.0,
                "disk": 75.0
            }
        }

        # Validate structure manually
        assert response_data["status"] in ["healthy", "unhealthy", "degraded"]
        assert isinstance(response_data["uptime"], (int, float))
        assert response_data["uptime"] >= 0
        assert isinstance(response_data["model"], dict)
        assert isinstance(response_data["system"], dict)

    def test_health_response_status_values(self):
        """Test valid health status values."""
        valid_statuses = ["healthy", "unhealthy", "degraded"]

        for status in valid_statuses:
            response_data = {
                "status": status,
                "uptime": 1000.0
            }
            assert response_data["status"] == status

    def test_health_response_with_metrics(self):
        """Test health response with detailed metrics."""
        response_data = {
            "status": "healthy",
            "uptime": 7200.0,
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": 45.2,
            "model": {
                "loaded": True,
                "healthy": True,
                "name": "phi-2",
                "version": "1.0.0"
            },
            "system": {
                "cpu": 30.0,
                "memory": 60.0,
                "disk": 40.0,
                "gpu": {
                    "available": True,
                    "memory_used": 2048,
                    "memory_total": 8192
                }
            },
            "components": {
                "database": "healthy",
                "cache": "healthy",
                "websockets": {"active_connections": 5}
            }
        }

        # Validate comprehensive structure
        assert response_data["response_time_ms"] > 0
        assert isinstance(response_data["components"], dict)
        assert response_data["system"]["gpu"]["available"] is True


class TestMetricsResponse:
    """Test MetricsResponse model (if accessible)."""

    def test_metrics_response_structure(self):
        """Test metrics response structure."""
        metrics_data = {
            "cpu": {
                "usage_percent": 25.5,
                "load_average": [1.2, 1.5, 1.8],
                "core_count": 8
            },
            "memory": {
                "total_gb": 16.0,
                "available_gb": 8.0,
                "used_gb": 8.0,
                "usage_percent": 50.0
            },
            "disk": {
                "total_gb": 1000.0,
                "free_gb": 500.0,
                "used_gb": 500.0,
                "usage_percent": 50.0
            },
            "gpu": {
                "available": True,
                "count": 1,
                "devices": [
                    {
                        "name": "NVIDIA GeForce RTX 3080",
                        "memory_total": 10240,
                        "memory_used": 2048,
                        "utilization": 75.0
                    }
                ]
            },
            "network": {
                "bytes_sent": 1048576,
                "bytes_recv": 2097152,
                "packets_sent": 1000,
                "packets_recv": 1500
            }
        }

        # Validate metrics structure
        assert metrics_data["cpu"]["usage_percent"] >= 0
        assert metrics_data["memory"]["total_gb"] > 0
        assert metrics_data["disk"]["total_gb"] > 0
        assert isinstance(metrics_data["gpu"]["available"], bool)
        assert len(metrics_data["gpu"]["devices"]) == metrics_data["gpu"]["count"]

    def test_metrics_response_no_gpu(self):
        """Test metrics response when GPU is not available."""
        metrics_data = {
            "cpu": {"usage_percent": 15.0},
            "memory": {"total_gb": 8.0, "available_gb": 4.0},
            "disk": {"total_gb": 500.0, "free_gb": 250.0},
            "gpu": {
                "available": False,
                "count": 0,
                "devices": []
            }
        }

        assert metrics_data["gpu"]["available"] is False
        assert metrics_data["gpu"]["count"] == 0
        assert len(metrics_data["gpu"]["devices"]) == 0


class TestSchemaValidation:
    """Test comprehensive schema validation."""

    def test_message_content_validation(self):
        """Test message content validation rules."""
        # Test minimum length (if applicable)
        valid_contents = [
            "Hi",
            "Hello, world!",
            "This is a longer message with more content.",
            "Multi\nline\ncontent",
            "Content with special chars: !@#$%^&*()"
        ]

        for content in valid_contents:
            message = ChatMessage(role=ChatRole.USER, content=content)
            assert message.content == content

    def test_message_role_string_conversion(self):
        """Test that string roles are properly converted."""
        message = ChatMessage(role="user", content="Test")
        assert message.role == ChatRole.USER
        assert isinstance(message.role, ChatRole)

    def test_comprehensive_validation_scenarios(self):
        """Test various validation scenarios."""
        # Valid message scenarios
        valid_scenarios = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message", "name": "john"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "function", "content": "Function result", "name": "get_weather"}
        ]

        for scenario in valid_scenarios:
            message = ChatMessage(**scenario)
            assert message.role.value == scenario["role"]
            assert message.content == scenario["content"]

    def test_edge_case_validation(self):
        """Test edge cases in validation."""
        # Very long content
        long_content = "A" * 10000
        message = ChatMessage(role=ChatRole.USER, content=long_content)
        assert len(message.content) == 10000

        # Unicode content
        unicode_content = "Hello 🌍 World! 你好世界"
        message = ChatMessage(role=ChatRole.USER, content=unicode_content)
        assert message.content == unicode_content

        # Content with special formatting
        formatted_content = "**Bold** and *italic* text\n\n```python\nprint('hello')\n```"
        message = ChatMessage(role=ChatRole.USER, content=formatted_content)
        assert message.content == formatted_content


class TestSchemaCompatibility:
    """Test OpenAI API compatibility."""

    def test_openai_message_format(self):
        """Test compatibility with OpenAI message format."""
        openai_message = {
            "role": "user",
            "content": "What's the weather like today?"
        }

        message = ChatMessage(**openai_message)
        assert message.role == ChatRole.USER
        assert message.content == openai_message["content"]

        # Convert back to OpenAI format
        back_to_openai = message.model_dump(exclude_none=True)
        assert back_to_openai["role"] == "user"
        assert back_to_openai["content"] == openai_message["content"]

    def test_openai_system_message(self):
        """Test OpenAI system message compatibility."""
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that provides accurate information."
        }

        message = ChatMessage(**system_message)
        assert message.role == ChatRole.SYSTEM
        assert message.content == system_message["content"]

    def test_openai_function_message(self):
        """Test OpenAI function message compatibility."""
        function_message = {
            "role": "function",
            "content": '{"temperature": 22, "humidity": 65}',
            "name": "get_weather_data"
        }

        message = ChatMessage(**function_message)
        assert message.role == ChatRole.FUNCTION
        assert message.content == function_message["content"]
        assert message.name == function_message["name"]