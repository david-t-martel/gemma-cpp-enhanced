# Python Development Quick Start - Gemma CLI

## Essential Commands Reference

### Setup (One-Time)

```bash
# Install core dependencies
uv pip install -e .

# Install with dev tools (recommended)
uv pip install -e ".[dev]"

# Install everything (ML + dev tools)
uv pip install -e ".[dev,ml]"
```

### Daily Development Workflow

```bash
# 1. Format your code
uv run ruff format src tests

# 2. Fix linting issues automatically
uv run ruff check src tests --fix

# 3. Check types
uv run mypy src

# 4. Run tests with coverage
uv run pytest tests/ --cov=src/gemma_cli --cov-fail-under=85 -v
```

### Quick Checks

```bash
# Lint only (no fix)
uv run ruff check src

# Type check a specific file
uv run mypy src/gemma_cli/core/gemma.py

# Run specific test
uv run pytest tests/test_cli.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "test_command" -v
```

### Test Options

```bash
# Run with coverage report
uv run pytest --cov=src/gemma_cli --cov-report=html

# Run only fast tests
uv run pytest -m "not slow"

# Run only unit tests
uv run pytest -m unit

# Verbose output with full tracebacks
uv run pytest -vv --tb=long

# Stop at first failure
uv run pytest -x

# Run last failed tests
uv run pytest --lf
```

### Before Committing

```bash
# Complete pre-commit check
uv run ruff format src tests && \
uv run ruff check src tests --fix && \
uv run mypy src && \
uv run pytest tests/ --cov=src/gemma_cli --cov-fail-under=85
```

### Building

```bash
# Build distribution packages
uv run python -m build

# Install built package locally
uv pip install dist/gemma_cli-0.4.0-py3-none-any.whl

# Test the installed CLI
gemma-cli --help
```

### Troubleshooting

```bash
# Show Ruff configuration
uv run ruff check --show-settings

# Show MyPy configuration
uv run mypy --config-file pyproject.toml --show-config-file

# List pytest markers
uv run pytest --markers

# Show coverage report
uv run pytest --cov=src/gemma_cli --cov-report=term-missing
```

## Configuration Files

- **`pyproject.toml`** - Main configuration (build, tools, dependencies)
- **`requirements.txt`** - Legacy format (for reference only)
- **`ruff.toml`** - Legacy Ruff config (consolidated into pyproject.toml)

## Key Standards

- **Python Version**: >=3.11
- **Line Length**: 100 characters
- **Coverage Target**: 85% minimum
- **Type Checking**: Strict mode enabled
- **Import Sorting**: isort via Ruff

## Common Patterns

### Adding a New Dependency

```bash
# 1. Add to pyproject.toml dependencies list
# 2. Install in development mode
uv pip install -e .

# For dev-only dependencies, add to [project.optional-dependencies.dev]
```

### Writing Type-Safe Code

```python
from typing import Any
from pydantic import BaseModel

def process_data(data: dict[str, Any]) -> list[str]:
    """Process data with type hints."""
    return [str(v) for v in data.values()]

class Config(BaseModel):
    """Config with Pydantic validation."""
    name: str
    port: int = 8080
```

### Async Testing

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async code."""
    result = await some_async_function()
    assert result is not None
```

## VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests",
    "-v",
    "--cov=src/gemma_cli"
  ]
}
```

## CI/CD Integration

For GitHub Actions:

```yaml
- name: Install dependencies
  run: uv pip install -e ".[dev]"

- name: Lint
  run: uv run ruff check src tests

- name: Type check
  run: uv run mypy src

- name: Test
  run: uv run pytest --cov=src/gemma_cli --cov-fail-under=85
```

## Performance Tips

1. **Ruff is fast** - Use it instead of multiple linters
2. **MyPy caching** - Subsequent runs are much faster
3. **Pytest parallel** - Install `pytest-xdist` for parallel tests
4. **Coverage caching** - Enable for faster reruns

---

**Last Updated**: 2025-01-22
**Configuration**: `pyproject.toml` v0.4.0
