# Python Project Management Framework - Gemma CLI

## Overview

Comprehensive Python project configuration for the Gemma CLI project with modern tooling, strict type checking, and quality assurance.

## Configuration Summary

### Project Metadata
- **Name**: gemma-cli
- **Version**: 0.4.0
- **Python**: >=3.11
- **Build System**: setuptools + wheel

### Core Dependencies (18 packages)

**CLI & UI:**
- click>=8.1.7 - Command-line interface framework
- rich>=13.7.0 - Rich text and beautiful formatting
- prompt-toolkit>=3.0.43 - Interactive command line
- colorama>=0.4.6 - Cross-platform colored terminal text

**Data & Validation:**
- pydantic>=2.5.0 - Data validation using Python type hints
- pydantic-settings>=2.1.0 - Settings management
- psutil>=5.9.0 - System and process utilities
- numpy>=1.24.0 - Numerical computing

**Configuration:**
- PyYAML>=6.0 - YAML parser and emitter
- tomli-w>=1.0.0 - TOML writer
- toml>=0.10.2 - TOML parser

**Async Support:**
- aioredis>=2.0.1 - Async Redis client
- aiofiles>=23.2.1 - Async file operations
- aioconsole>=0.7.0 - Async console I/O

**Infrastructure:**
- redis>=5.0.0 - Redis client
- mcp>=0.9.0 - Model Context Protocol
- sentence-transformers>=2.2.0 - Sentence embeddings
- tiktoken>=0.5.0 - OpenAI tokenizer

### Development Dependencies (9 packages)

**Testing:**
- pytest>=7.4.0
- pytest-cov>=4.1.0
- pytest-asyncio>=0.21.0

**Linting & Formatting:**
- ruff>=0.1.0 - Fast Python linter
- black>=23.0.0 - Code formatter
- mypy>=1.7.0 - Static type checker

**Type Stubs:**
- types-redis>=4.6.0
- types-toml>=0.10.8
- types-PyYAML>=6.0.12

### Optional Dependencies

**ML Support:**
- torch>=2.0.0
- transformers>=4.30.0

**Fast Embeddings:**
- onnxruntime>=1.16.0

**Rust FFI:**
- rag-redis-system>=0.1.0 (local package)

## Tool Configuration

### Ruff (Linter & Formatter)

**Configuration:**
- Target: Python 3.11
- Line length: 100 characters
- Auto-fix: Enabled

**Selected Rules (16 categories):**
- E/W - pycodestyle errors and warnings
- F - pyflakes
- I - isort (import sorting)
- N - pep8-naming
- UP - pyupgrade (modern Python idioms)
- ANN - flake8-annotations (type hints)
- B - flake8-bugbear (common bugs)
- C4 - flake8-comprehensions
- SIM - flake8-simplify
- PTH - flake8-use-pathlib
- ASYNC - async/await patterns
- S - security (bandit)
- ARG - unused arguments
- RET - return statements
- RUF - Ruff-specific rules

**Ignored Rules:**
- ANN101/102 - Type annotations for self/cls
- ANN401 - Dynamically typed expressions (Any)
- E501 - Line too long (handled by formatter)
- S101 - Assert statements (common in tests)

**Per-File Ignores:**
- `tests/**/*.py` - Relaxed type annotations and assert rules
- `scripts/**/*.py` - Allow print statements and subprocess
- `**/__init__.py` - Unused imports OK (re-exports)

### MyPy (Type Checker)

**Strict Configuration:**
- Python version: 3.11
- `disallow_untyped_defs`: true
- `disallow_incomplete_defs`: true
- `check_untyped_defs`: true
- `warn_return_any`: true
- `warn_unused_configs`: true
- `strict_equality`: true
- `ignore_missing_imports`: true (for third-party)

**Ignored Third-Party Modules:**
- redis, aioredis
- sentence_transformers
- mcp, torch, transformers
- tiktoken, prompt_toolkit

### Pytest (Test Framework)

**Configuration:**
- Test paths: `tests/`
- Test patterns: `test_*.py`, `*_test.py`
- Async mode: auto
- Coverage target: **85%**

**Addopts:**
- `--strict-markers` - Strict marker validation
- `--strict-config` - Strict configuration
- `--cov=src/gemma_cli` - Coverage reporting
- `--cov-fail-under=85` - Fail if coverage < 85%
- `-ra` - Show all test summary info

**Test Markers:**
- `slow` - Slow tests (can be skipped)
- `integration` - Integration tests
- `unit` - Unit tests
- `asyncio` - Async tests

### Coverage

**Configuration:**
- Source: `src/gemma_cli`
- Branch coverage: Enabled
- Fail under: **85%**
- Parallel mode: Enabled

**Excluded Lines:**
- `pragma: no cover`
- `def __repr__`, `def __str__`
- `raise AssertionError`, `raise NotImplementedError`
- `if __name__ == .__main__.:`
- `if TYPE_CHECKING:`
- `@abstractmethod`, `@abc.abstractmethod`
- Protocol classes
- `@overload`

**Reports:**
- Terminal: Term-missing with skip-covered
- HTML: `htmlcov/`
- XML: `coverage.xml`

## Usage Instructions

### Installation

```bash
# Install core dependencies only
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"

# Install with ML dependencies
uv pip install -e ".[ml]"

# Install all optional dependencies
uv pip install -e ".[all]"
```

### Running Tools

```bash
# Linting with Ruff
uv run ruff check src --fix
uv run ruff format src

# Type checking with MyPy
uv run mypy src --ignore-missing-imports

# Running tests
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=src/gemma_cli --cov-report=html

# Run only unit tests
uv run pytest tests/ -v -m unit

# Run only fast tests (skip slow)
uv run pytest tests/ -v -m "not slow"

# Code formatting with Black
uv run black src tests --line-length 100
```

### Build and Distribution

```bash
# Build source distribution and wheel
uv run python -m build

# Check package metadata
uv run python -m build --check

# Install locally
uv pip install dist/gemma_cli-0.4.0-py3-none-any.whl
```

### Pre-Commit Workflow

```bash
# 1. Format code
uv run ruff format src tests

# 2. Fix linting issues
uv run ruff check src tests --fix

# 3. Type check
uv run mypy src

# 4. Run tests with coverage
uv run pytest tests/ --cov=src/gemma_cli --cov-fail-under=85

# 5. Build package
uv run python -m build
```

## Project Structure

```
gemma/
├── src/
│   └── gemma_cli/
│       ├── __init__.py
│       ├── cli.py
│       ├── commands/
│       ├── config/
│       ├── core/
│       ├── mcp/
│       ├── onboarding/
│       ├── rag/
│       └── ui/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml        # Main configuration file
├── requirements.txt      # Legacy dependencies (for reference)
├── ruff.toml            # Legacy Ruff config (consolidated into pyproject.toml)
└── README_enhanced_cli.md
```

## Key Features

### 1. Modern Build System
- Uses setuptools with PEP 517/518 compliance
- Automatic package discovery in `src/` layout
- Proper namespace handling

### 2. Comprehensive Type Safety
- Strict MyPy configuration
- Type hints enforced via Ruff ANN rules
- Third-party library type stubs

### 3. High Test Coverage
- 85% minimum coverage requirement
- Branch coverage enabled
- Comprehensive exclusion patterns

### 4. Fast Development Workflow
- Ruff provides fast linting and formatting
- Auto-fix enabled for common issues
- Pre-configured for async/await patterns

### 5. Security Focused
- Bandit security checks enabled
- Hardcoded secret detection
- Subprocess validation

## Verification

The configuration has been validated:

```
SUCCESS: TOML syntax is valid
Project: gemma-cli v0.4.0
Python requirement: >=3.11
Build backend: setuptools.build_meta
Core dependencies: 18 packages
Dev dependencies: 9 packages
Ruff rules selected: 16 categories
Ruff line length: 100
MyPy check_untyped_defs: True
Coverage target: 85%
```

## Integration with Claude Code

This configuration follows the Prime Directive from CLAUDE.md:
- ✅ All Python commands use `uv run`
- ✅ Type hints enforced via MyPy and Ruff
- ✅ 85% test coverage minimum
- ✅ Security checks enabled
- ✅ Modern Python 3.11+ features

## Next Steps

1. **Install dependencies**: `uv pip install -e ".[dev]"`
2. **Run initial checks**: `uv run ruff check src --fix`
3. **Add type hints**: `uv run mypy src`
4. **Write tests**: Ensure 85% coverage
5. **Build package**: `uv run python -m build`

---

**Generated**: 2025-01-22
**Configuration File**: `C:\codedev\llm\gemma\pyproject.toml`
**Status**: ✅ Validated and ready for use
