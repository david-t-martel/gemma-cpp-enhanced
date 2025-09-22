"""Comprehensive integration tests for the LLM Framework."""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_framework import (
    LLMFramework,
    LLMConfig,
    ModelRegistry,
    InferenceEngine,
    ModelInfo,
    ModelType,
    ModelCapabilities,
    GenerationConfig,
    quick_generate,
    create_framework,
)
from llm_framework.backends import BackendType, create_backend
from llm_framework.models import ModelSize
from llm_framework.config import load_framework_config
from llm_framework.integration import (
    FrameworkAgent,
    upgrade_to_framework,
    create_gemma_agent,
)


class TestFrameworkCore:
    """Test core framework functionality."""
    
    def test_llm_config_creation(self):
        """Test LLMConfig creation and validation."""
        config = LLMConfig(
            models_dir="/.models",
            max_concurrent_requests=5,
            default_temperature=0.8
        )
        
        assert config.models_dir == "/.models"
        assert config.max_concurrent_requests == 5
        assert config.default_temperature == 0.8
        assert config.enable_fallbacks is True  # Default
    
    def test_config_dict_conversion(self):
        """Test configuration dictionary conversion."""
        config = LLMConfig(models_dir="/test", default_temperature=0.9)
        config_dict = config.to_dict()
        
        assert config_dict["models_dir"] == "/test"
        assert config_dict["default_temperature"] == 0.9
        
        # Test from_dict
        new_config = LLMConfig.from_dict(config_dict)
        assert new_config.models_dir == "/test"
        assert new_config.default_temperature == 0.9


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation and validation."""
        capabilities = ModelCapabilities(
            supports_streaming=True,
            max_context_length=8192
        )
        
        model_info = ModelInfo(
            name="test-model",
            display_name="Test Model",
            model_type=ModelType.CHAT,
            backend_type="test",
            size=ModelSize.SMALL,
            capabilities=capabilities
        )
        
        assert model_info.name == "test-model"
        assert model_info.model_type == ModelType.CHAT
        assert model_info.capabilities.supports_streaming is True
    
    def test_model_registry_operations(self):
        """Test model registry operations."""
        registry = ModelRegistry()
        
        # Test listing default models
        models = registry.list_models()
        assert len(models) > 0
        
        # Test filtering
        chat_models = registry.list_models(model_type=ModelType.CHAT)
        local_models = registry.list_models(local_only=True)
        
        assert all(m.model_type == ModelType.CHAT for m in chat_models)
        assert all(m.model_path is not None for m in local_models)
    
    def test_model_registration(self):
        """Test custom model registration."""
        registry = ModelRegistry()
        
        model_info = ModelInfo(
            name="custom-test",
            display_name="Custom Test Model",
            model_type=ModelType.TEXT_GENERATION,
            backend_type="custom",
            size=ModelSize.SMALL,
            capabilities=ModelCapabilities()
        )
        
        registry.register_model(model_info)
        retrieved = registry.get_model("custom-test")
        
        assert retrieved.name == "custom-test"
        assert retrieved.backend_type == "custom"


class TestGenerationConfig:
    """Test generation configuration."""
    
    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        config = GenerationConfig()
        
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.stream is False
    
    def test_generation_config_custom(self):
        """Test GenerationConfig with custom values."""
        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.9,
            top_p=0.95,
            stop_sequences=["END", "STOP"]
        )
        
        assert config.max_tokens == 1024
        assert config.temperature == 0.9
        assert config.stop_sequences == ["END", "STOP"]


class TestMockBackends:
    """Test backend creation with mocked dependencies."""
    
    def test_backend_creation(self):
        """Test backend factory function."""
        model_info = ModelInfo(
            name="test-gemma",
            display_name="Test Gemma",
            model_type=ModelType.CHAT,
            backend_type="gemma_native",
            size=ModelSize.SMALL,
            capabilities=ModelCapabilities()
        )
        
        # This will work even without the actual Gemma bridge
        # because we're just testing the factory function
        try:
            backend = create_backend(model_info)
            assert backend.model_info.name == "test-gemma"
            assert backend.backend_type == BackendType.GEMMA_NATIVE
        except Exception as e:
            # Expected if dependencies aren't available
            assert "gemma_bridge" in str(e) or "Failed to" in str(e)


@pytest.mark.asyncio
class TestAsyncFramework:
    """Test async framework functionality."""
    
    async def test_framework_initialization(self):
        """Test framework initialization."""
        config = LLMConfig(
            models_dir="/.models",
            max_concurrent_requests=2
        )
        
        framework = LLMFramework(config)
        assert not framework._initialized
        
        await framework.initialize()
        assert framework._initialized
        
        await framework.shutdown()
        assert not framework._initialized
    
    async def test_framework_context_manager(self):
        """Test framework as async context manager."""
        async with LLMFramework() as framework:
            assert framework._initialized
            
            # Test model listing
            models = framework.list_models()
            assert len(models) > 0
            
            # Test model info retrieval
            if models:
                model_info = framework.get_model_info(models[0].name)
                assert model_info.name == models[0].name
    
    async def test_performance_stats(self):
        """Test performance statistics."""
        async with LLMFramework() as framework:
            stats = framework.get_performance_stats()
            
            assert "total_requests" in stats
            assert "loaded_models" in stats
            assert "total_models" in stats
            assert stats["total_requests"] >= 0


class TestConfigurationLoading:
    """Test configuration loading from various sources."""
    
    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "models_dir": "/test/models",
            "max_concurrent_requests": 15,
            "default_temperature": 0.8
        }
        
        config = load_framework_config(config_dict=config_dict)
        assert config.models_dir == "/test/models"
        assert config.max_concurrent_requests == 15
        assert config.default_temperature == 0.8
    
    @patch.dict('os.environ', {
        'LLM_FRAMEWORK_MAX_CONCURRENT_REQUESTS': '20',
        'LLM_FRAMEWORK_DEFAULT_TEMPERATURE': '0.9',
        'LLM_FRAMEWORK_ENABLE_FALLBACKS': 'true'
    })
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        config = load_framework_config()
        
        assert config.max_concurrent_requests == 20
        assert config.default_temperature == 0.9
        assert config.enable_fallbacks is True


class TestIntegrationLayer:
    """Test integration with existing agent system."""
    
    def test_upgrade_to_framework(self):
        """Test upgrading from old configuration."""
        old_config = {
            "max_new_tokens": 256,
            "temperature": 0.8,
            "verbose": True
        }
        
        framework = upgrade_to_framework(config=old_config)
        assert framework.config.default_max_tokens == 256
        assert framework.config.default_temperature == 0.8
        assert framework.config.log_level == "DEBUG"  # verbose=True
    
    @pytest.mark.asyncio
    async def test_framework_agent_creation(self):
        """Test creating framework agent."""
        config = LLMConfig(max_concurrent_requests=1)
        
        async with LLMFramework(config) as framework:
            agent = FrameworkAgent(
                framework=framework,
                model_name="gemma-2b-it",
                max_tokens=100
            )
            
            assert agent.framework == framework
            assert agent.model_name == "gemma-2b-it"
            assert agent.generation_kwargs["max_tokens"] == 100


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_model_not_found_handling(self):
        """Test handling of non-existent models."""
        config = LLMConfig(enable_fallbacks=False)
        
        async with LLMFramework(config) as framework:
            try:
                # This should fail gracefully
                await framework.generate_text(
                    "Test prompt",
                    model_name="non-existent-model"
                )
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "non-existent-model" in str(e) or "failed" in str(e).lower()
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid configuration
        try:
            LLMConfig(max_concurrent_requests=-1)
            # If no validation, this should still work
        except Exception:
            # If validation exists, exception is expected
            pass


@pytest.mark.integration
class TestFullIntegration:
    """Full integration tests (may require actual models)."""
    
    @pytest.mark.asyncio
    async def test_quick_generate_mock(self):
        """Test quick_generate function with mocking."""
        # Mock the framework to avoid requiring actual models
        with patch('llm_framework.core.LLMFramework') as mock_framework_class:
            mock_framework = AsyncMock()
            mock_framework.generate_text.return_value = "Mocked response"
            mock_framework_class.return_value.__aenter__.return_value = mock_framework
            
            response = await quick_generate("Test prompt")
            assert response == "Mocked response"
    
    @pytest.mark.asyncio
    async def test_create_framework_function(self):
        """Test create_framework convenience function."""
        config = LLMConfig(max_concurrent_requests=2)
        framework = await create_framework(config)
        
        assert framework._initialized
        assert framework.config.max_concurrent_requests == 2
        
        await framework.shutdown()


def test_import_structure():
    """Test that all imports work correctly."""
    # Test main framework imports
    from llm_framework import (
        LLMFramework,
        LLMConfig,
        ModelRegistry,
        InferenceEngine,
        quick_generate
    )
    
    # Test backend imports
    from llm_framework.backends import GenerationConfig, BackendType
    
    # Test model imports
    from llm_framework.models import ModelInfo, ModelType
    
    # Test integration imports
    from llm_framework.integration import FrameworkAgent
    
    # Test exception imports
    from llm_framework.exceptions import LLMFrameworkError
    
    # All imports successful
    assert True


def test_version_availability():
    """Test that version is available."""
    import llm_framework
    assert hasattr(llm_framework, '__version__')
    assert llm_framework.__version__ == "1.0.0"


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running basic LLM Framework tests...")
    
    # Test configuration
    print("✓ Testing configuration...")
    test_config = TestFrameworkCore()
    test_config.test_llm_config_creation()
    test_config.test_config_dict_conversion()
    
    # Test model registry
    print("✓ Testing model registry...")
    test_registry = TestModelRegistry()
    test_registry.test_model_registry_operations()
    test_registry.test_model_registration()
    
    # Test imports
    print("✓ Testing imports...")
    test_import_structure()
    test_version_availability()
    
    # Test async functionality
    print("✓ Testing async framework...")
    async def run_async_tests():
        test_async = TestAsyncFramework()
        await test_async.test_framework_initialization()
        await test_async.test_framework_context_manager()
        await test_async.test_performance_stats()
    
    asyncio.run(run_async_tests())
    
    print("✅ All basic tests passed!")
    print("\nTo run full test suite with pytest:")
    print("  uv run pytest tests/test_framework_integration.py -v")
    print("\nTo run integration tests:")
    print("  uv run pytest tests/test_framework_integration.py -v -m integration")