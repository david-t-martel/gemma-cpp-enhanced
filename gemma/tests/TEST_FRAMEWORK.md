# Gemma CLI Test Framework

Comprehensive pytest-based testing framework for the Gemma CLI project.

## Overview

This test framework provides a complete testing infrastructure with:
- **Shared fixtures** for common test data and mocks
- **Utility functions** for test operations
- **Data fixtures** with realistic sample data
- **pytest configuration** with markers and options

## Quick Start

### Installation

```bash
# Install test dependencies
uv pip install -r requirements.txt

# Or install in development mode
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_mcp_client.py

# Run specific test
uv run pytest tests/test_mcp_client.py::TestMCPClientManager::test_connect_server_success

# Run tests with verbose output
uv run pytest -v

# Run tests in parallel (fast)
uv run pytest -n auto

# Run tests with coverage
uv run pytest --cov=src/gemma_cli --cov-report=html
```

### Test Markers

Use markers to run specific test categories:

```bash
# Run only unit tests
uv run pytest -m unit

# Run integration tests
uv run pytest -m integration

# Run tests excluding slow ones
uv run pytest -m "not slow"

# Run tests requiring Redis
uv run pytest -m redis

# Run async tests only
uv run pytest -m asyncio

# Combine markers
uv run pytest -m "unit and not slow"
```

Available markers:
- `unit` - Fast, isolated unit tests
- `integration` - Multi-component integration tests
- `slow` - Tests taking >1 second
- `asyncio` - Async tests
- `redis` - Requires Redis connection
- `model` - Requires actual model files
- `subprocess` - Spawns subprocesses
- `mcp` - MCP client tests
- `rag` - RAG/memory system tests
- `ui` - UI component tests
- `config` - Configuration tests
- `onboarding` - Onboarding system tests
- `performance` - Performance/benchmarks
- `smoke` - Quick smoke tests
- `regression` - Regression tests

## Framework Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── pytest.ini               # Pytest configuration
├── TEST_FRAMEWORK.md        # This file
├── test_framework_example.py # Usage examples
│
├── utils/
│   ├── test_helpers.py      # Test utility functions
│   └── fixtures.py          # Sample data fixtures
│
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── functional/              # Functional tests
└── performance/             # Performance tests
```

## Available Fixtures

### Interface Mocks

#### `mock_gemma_interface`
Mock GemmaInterface for testing without actual model inference.

```python
@pytest.mark.asyncio
async def test_generation(mock_gemma_interface):
    response = await mock_gemma_interface.generate_response("test")
    assert "sample response" in response
```

#### `mock_subprocess_call`
Mock subprocess calls to prevent actual process execution.

```python
def test_subprocess(mock_subprocess_call):
    # Your subprocess code here
    mock_subprocess_call.assert_called_once()
```

### Database Mocks

#### `mock_redis`
Mock Redis client with in-memory storage for testing.

```python
@pytest.mark.asyncio
async def test_redis(mock_redis):
    await mock_redis.set("key", "value")
    result = await mock_redis.get("key")
    assert result == "value"
```

#### `mock_redis_unavailable`
Mock Redis that simulates connection failure.

```python
async def test_fallback(mock_redis_unavailable):
    with pytest.raises(ConnectionError):
        await mock_redis_unavailable.ping()
```

### Configuration Fixtures

#### `temp_config_dir`
Temporary directory for configuration files (auto-cleanup).

```python
def test_config(temp_config_dir):
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("[section]\\nkey = 'value'")
```

#### `sample_model_preset`
Sample ModelPreset with typical configuration.

```python
def test_preset(sample_model_preset):
    assert sample_model_preset.name == "gemma-2b-fast"
    assert sample_model_preset.temperature == 0.7
```

#### `sample_performance_profile`
Sample PerformanceProfile with balanced settings.

```python
def test_profile(sample_performance_profile):
    assert sample_performance_profile.batch_size == 32
```

#### `sample_settings`
Complete Settings object with all subsystems configured.

```python
def test_settings(sample_settings):
    assert sample_settings.gemma.default_model.endswith(".sbs")
    assert sample_settings.redis.host == "localhost"
```

### File System Fixtures

#### `mock_model_file`
Creates a mock model file (.sbs).

```python
def test_model(mock_model_file):
    assert mock_model_file.exists()
    assert mock_model_file.suffix == ".sbs"
```

#### `mock_tokenizer_file`
Creates a mock tokenizer file (.spm).

```python
def test_tokenizer(mock_tokenizer_file):
    assert mock_tokenizer_file.exists()
```

#### `mock_model_directory`
Complete directory structure with model and tokenizer.

```python
def test_directory(mock_model_directory):
    models = list(mock_model_directory.glob("*.sbs"))
    assert len(models) == 1
```

### Sample Data Fixtures

#### `sample_prompts`
List of diverse test prompts.

```python
def test_prompts(sample_prompts):
    for prompt in sample_prompts:
        # Test with each prompt
        pass
```

#### `sample_responses`
List of realistic model responses.

```python
def test_responses(sample_responses):
    for response in sample_responses:
        assert_valid_response(response)
```

#### `sample_conversation_history`
Conversation turns with role and content.

```python
def test_conversation(sample_conversation_history):
    assert len(sample_conversation_history) > 0
```

## Utility Functions

### File Creation

```python
from tests.utils.test_helpers import (
    create_mock_model_file,
    create_mock_tokenizer,
    create_mock_config_file,
    create_complete_test_environment,
)

# Create mock model
model_path = create_mock_model_file("test.sbs", size_mb=50)

# Create mock tokenizer
tokenizer_path = create_mock_tokenizer("tokenizer.spm", vocab_size=1000)

# Create complete test environment
env = create_complete_test_environment("/tmp/test")
```

### Validation

```python
from tests.utils.test_helpers import (
    assert_valid_response,
    assert_valid_config,
    assert_valid_model_path,
    assert_valid_prompt,
)

# Validate response
assert_valid_response(
    response="The capital is Paris.",
    min_length=5,
    required_keywords=["Paris"]
)

# Validate config
assert_valid_config(config, required_sections=["gemma", "redis"])

# Validate prompt
assert_valid_prompt("Tell me about Python")
```

### Data Generation

```python
from tests.utils.test_helpers import (
    generate_random_prompt,
    generate_random_response,
    get_default_test_config,
)

# Generate random data
prompt = generate_random_prompt(min_words=10, topic="Python")
response = generate_random_response(min_words=50)
config = get_default_test_config()
```

### Subprocess Mocking

```python
from tests.utils.test_helpers import (
    mock_subprocess_call,
    mock_gemma_executable,
)

# Mock subprocess
with patch("subprocess.Popen", mock_subprocess_call(stdout="output")):
    # Your subprocess code
    pass

# Mock Gemma executable
mock = mock_gemma_executable(response="Hello!", delay_seconds=0.5)
```

## Sample Data

### Accessing Sample Data

```python
from tests.utils.fixtures import (
    SAMPLE_CONFIGS,
    SAMPLE_PROMPTS,
    SAMPLE_RESPONSES,
    SAMPLE_CONVERSATIONS,
    get_sample_config,
    get_sample_prompts,
)

# Get sample config
config = get_sample_config("development")

# Get sample prompts by category
simple_prompts = get_sample_prompts("simple")
coding_prompts = get_sample_prompts("coding")

# Get sample conversation
conversation = SAMPLE_CONVERSATIONS[0]
```

### Available Sample Data

- **SAMPLE_CONFIGS**: Configuration presets (minimal, development, production, testing)
- **SAMPLE_PROMPTS**: Prompts by category (simple, factual, technical, coding, creative)
- **SAMPLE_RESPONSES**: Realistic model responses matching prompt categories
- **SAMPLE_CONVERSATIONS**: Complete conversation examples
- **SAMPLE_MODEL_PRESETS**: Model configuration presets
- **SAMPLE_PERFORMANCE_PROFILES**: Performance profiles (fast, balanced, quality)
- **SAMPLE_ERROR_CASES**: Invalid inputs for negative testing

## Writing Tests

### Test Structure (Arrange-Act-Assert)

```python
import pytest

@pytest.mark.unit
def test_feature():
    # Arrange - Set up test data and mocks
    config = get_sample_config("minimal")

    # Act - Execute the code under test
    result = process_config(config)

    # Assert - Verify the results
    assert result is not None
    assert_valid_config(result)
```

### Async Tests

```python
@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_feature(mock_redis):
    # Arrange
    await mock_redis.set("key", "value")

    # Act
    result = await fetch_from_redis("key")

    # Assert
    assert result == "value"
```

### Parameterized Tests

```python
@pytest.mark.unit
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("test", "TEST"),
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.redis
async def test_full_pipeline(mock_gemma_interface, mock_redis):
    # Test multiple components together
    prompt = "What is Python?"

    # Store in Redis
    await mock_redis.set("last_prompt", prompt)

    # Generate response
    response = await mock_gemma_interface.generate_response(prompt)

    # Validate
    assert_valid_response(response)
    stored = await mock_redis.get("last_prompt")
    assert stored == prompt
```

### Performance Tests

```python
@pytest.mark.performance
def test_inference_speed(benchmark_timer):
    with benchmark_timer as timer:
        # Code to benchmark
        result = expensive_operation()

    assert timer.elapsed < 1.0  # Should complete in 1 second
```

## Best Practices

### 1. Use Appropriate Markers

Always mark your tests appropriately:

```python
@pytest.mark.unit          # Fast, isolated
@pytest.mark.integration   # Multiple components
@pytest.mark.slow          # Takes >1 second
@pytest.mark.redis         # Requires Redis
```

### 2. Keep Tests Independent

Each test should be able to run independently:

```python
# Good - independent
def test_feature_a():
    result = feature_a()
    assert result == expected

def test_feature_b():
    result = feature_b()
    assert result == expected

# Bad - dependent
def test_feature_a():
    global state
    state = feature_a()

def test_feature_b():
    # Relies on test_feature_a running first
    assert state == expected
```

### 3. Use Fixtures for Setup

Use fixtures instead of setup methods:

```python
# Good - using fixtures
@pytest.fixture
def test_data():
    return {"key": "value"}

def test_feature(test_data):
    assert test_data["key"] == "value"

# Avoid - setup methods
class TestFeature:
    def setup_method(self):
        self.data = {"key": "value"}
```

### 4. Write Clear Test Names

Test names should describe what they test:

```python
# Good - descriptive names
def test_user_registration_with_valid_email():
    pass

def test_user_registration_with_invalid_email_raises_error():
    pass

# Bad - unclear names
def test_user_1():
    pass

def test_user_2():
    pass
```

### 5. Use Validation Helpers

Use provided validation helpers for consistency:

```python
# Good - using helpers
assert_valid_response(response, min_length=10)
assert_valid_config(config, required_sections=["gemma"])

# Avoid - manual validation
assert len(response) >= 10
assert "gemma" in config
```

## Debugging Tests

### Run Single Test with Verbose Output

```bash
uv run pytest tests/test_file.py::test_function -vvs
```

### Run with pdb on Failure

```bash
uv run pytest --pdb
```

### Show Local Variables on Failure

```bash
uv run pytest -l
```

### Run Last Failed Tests

```bash
uv run pytest --lf
```

### Generate HTML Coverage Report

```bash
uv run pytest --cov=src/gemma_cli --cov-report=html
# Open htmlcov/index.html in browser
```

## Continuous Integration

The test framework is designed for CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest -m unit --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Troubleshooting

### Redis Connection Errors

If Redis tests fail:
1. Check Redis is running: `redis-cli ping`
2. Skip Redis tests: `pytest -m "not redis"`
3. Use mock Redis: Tests should auto-fallback to mocks

### Import Errors

If imports fail:
1. Install in development mode: `uv pip install -e .`
2. Check Python path: `echo $PYTHONPATH`
3. Run from project root: `cd /path/to/gemma && pytest`

### Async Test Issues

If async tests fail:
1. Check pytest-asyncio is installed
2. Verify `asyncio_mode = auto` in pytest.ini
3. Mark tests with `@pytest.mark.asyncio`

## Contributing

When adding new tests:
1. Use existing fixtures and utilities
2. Add appropriate markers
3. Follow naming conventions
4. Write clear docstrings
5. Ensure tests are independent
6. Update this documentation if needed

## Examples

See `tests/test_framework_example.py` for comprehensive usage examples.

## Support

For questions or issues:
- Check existing test files for patterns
- Review this documentation
- See `tests/test_framework_example.py`
- Check pytest documentation: https://docs.pytest.org/
