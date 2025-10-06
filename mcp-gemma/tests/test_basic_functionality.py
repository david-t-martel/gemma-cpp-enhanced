#!/usr/bin/env python3
"""
Basic functionality tests for MCP Gemma server.

This test script validates core functionality without requiring a full model.
"""

import asyncio
import sys
import tempfile
import json
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all major components can be imported."""
    print("Testing imports...")

    try:
        from server.base import GemmaServer, GemmaConfig
        print("✓ Server base imports work")
    except ImportError as e:
        print(f"✗ Server base import failed: {e}")
        return False

    try:
        from server.transports import StdioTransport, HTTPTransport, WebSocketTransport
        print("✓ Transport imports work")
    except ImportError as e:
        print(f"✗ Transport import failed: {e}")
        return False

    try:
        from client.base_client import BaseGemmaClient, GenerationRequest, GenerationResponse
        print("✓ Client base imports work")
    except ImportError as e:
        print(f"✗ Client base import failed: {e}")
        return False

    try:
        from client.http_client import GemmaHTTPClient
        print("✓ HTTP client imports work")
    except ImportError as e:
        print(f"✗ HTTP client import failed: {e}")
        return False

    return True

def test_configuration():
    """Test configuration loading and validation."""
    print("Testing configuration...")

    try:
        from server.base import GemmaConfig

        # Test with minimal config
        config = GemmaConfig(
            model_path="/fake/path/model.sbs",
            enable_redis=False  # Disable Redis for testing
        )

        assert config.model_path == "/fake/path/model.sbs"
        assert config.max_tokens == 2048  # Default value
        assert config.temperature == 0.7  # Default value
        assert config.enable_redis == False

        print("✓ Configuration creation and validation work")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_client_request_validation():
    """Test client request validation."""
    print("Testing client request validation...")

    try:
        from client.base_client import GenerationRequest, BaseGemmaClient

        # Test valid request
        request = GenerationRequest(
            prompt="Hello world",
            max_tokens=100,
            temperature=0.8
        )
        assert request.prompt == "Hello world"
        assert request.max_tokens == 100
        assert request.temperature == 0.8

        # Test validation (requires a concrete client implementation)
        class TestClient(BaseGemmaClient):
            async def connect(self): pass
            async def disconnect(self): pass
            async def generate_text(self, request): return None
            async def generate_text_stream(self, request): yield ""

        client = TestClient()

        # Test valid request validation
        try:
            client._validate_request(request)
            print("✓ Valid request validation works")
        except Exception as e:
            print(f"✗ Valid request validation failed: {e}")
            return False

        # Test invalid request validation
        try:
            invalid_request = GenerationRequest(prompt="", max_tokens=-1)
            client._validate_request(invalid_request)
            print("✗ Invalid request validation should have failed")
            return False
        except ValueError:
            print("✓ Invalid request validation works")

        return True

    except Exception as e:
        print(f"✗ Client request validation test failed: {e}")
        return False

async def test_server_initialization():
    """Test server initialization without starting."""
    print("Testing server initialization...")

    try:
        from server.base import GemmaServer, GemmaConfig

        # Mock the GemmaInterface import for testing
        import unittest.mock
        with unittest.mock.patch('server.base.GemmaInterface') as mock_interface:
            mock_interface.return_value = unittest.mock.MagicMock()

            config = GemmaConfig(
                model_path="/fake/path/model.sbs",
                enable_redis=False  # Disable Redis for testing
            )

            try:
                server = GemmaServer(config)
                print("✓ Server initialization works")
                return True
            except Exception as e:
                print(f"✗ Server initialization failed: {e}")
                return False

    except Exception as e:
        print(f"✗ Server initialization test failed: {e}")
        return False

def test_integration_imports():
    """Test integration module imports."""
    print("Testing integration imports...")

    try:
        from integration.stats_integration import GemmaMCPAgent
        print("✓ Stats integration imports work")
    except ImportError as e:
        print(f"⚠ Stats integration import failed (expected if stats framework not available): {e}")

    try:
        from integration.rag_redis_integration import RAGRedisMemoryHandler
        print("✓ RAG-Redis integration imports work")
    except ImportError as e:
        print(f"⚠ RAG-Redis integration import failed (expected if dependencies not available): {e}")

    return True

def test_config_files():
    """Test that configuration files are valid."""
    print("Testing configuration files...")

    # Test YAML config
    try:
        import yaml
        config_path = PROJECT_ROOT / "config" / "server_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Server YAML config is valid")
        else:
            print("⚠ Server YAML config not found")
    except Exception as e:
        print(f"✗ Server YAML config test failed: {e}")
        return False

    # Test JSON config
    try:
        config_path = PROJECT_ROOT / "config" / "mcp_integration.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✓ MCP integration JSON config is valid")
        else:
            print("⚠ MCP integration JSON config not found")
    except Exception as e:
        print(f"✗ MCP integration JSON config test failed: {e}")
        return False

    return True

async def main():
    """Run all tests."""
    print("=" * 60)
    print("MCP Gemma Server - Basic Functionality Tests")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Client Request Validation", test_client_request_validation),
        ("Server Initialization", test_server_initialization),
        ("Integration Imports", test_integration_imports),
        ("Configuration Files", test_config_files),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! MCP Gemma basic functionality is working.")
        return 0
    else:
        print(f"⚠ {total - passed} tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)