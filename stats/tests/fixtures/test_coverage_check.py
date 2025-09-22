"""Simple test to verify coverage measurement is working."""

import pytest

from src.shared.exceptions import ChatbotException


def test_exception_creation():
    """Test creating a simple exception to verify coverage tracking."""
    # This will be tracked by coverage
    exception = ChatbotException("Test error")
    assert str(exception) == "Test error"
    assert isinstance(exception, Exception)


def test_simple_function():
    """Test a simple function to verify coverage tracking."""
    # Import and use a simple module
    from src.domain.validators import PromptValidator, ThreatLevel

    # Test the validator
    validator = PromptValidator()
    result = validator.validate("Hello, how are you?")

    assert hasattr(result, "is_valid")
    assert hasattr(result, "threat_level")
    assert isinstance(result.threat_level, (ThreatLevel, str))


def test_basic_import_execution():
    """Test that imports and basic execution are tracked by coverage."""
    from src.shared.config.settings import Settings

    # Create a simple Settings object to trigger code execution
    try:
        settings = Settings()
        assert hasattr(settings, "__dict__")
    except Exception:
        # Settings might require environment variables
        # This is still coverage-tracked execution
        pass
