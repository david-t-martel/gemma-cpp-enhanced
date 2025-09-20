# Pre-Commit Hooks Guide for Gemma LLM Stats

This document explains the comprehensive pre-commit hook setup for the Gemma LLM Stats project, which uses Python 3.13, Rust, and TypeScript technologies.

## Overview

The pre-commit configuration implements a 7-phase quality pipeline designed to catch issues early and maintain code quality:

1. **Basic File Validation** - File format, whitespace, and syntax checks
2. **Python Code Quality** - Formatting, linting, and type checking
3. **Security Scanning** - Vulnerability detection and secrets scanning
4. **Rust Code Quality** - Formatting, linting, and security audits
5. **Documentation & Config** - Markdown and YAML validation
6. **Project-Specific Validation** - Custom validation for this project
7. **Infrastructure Validation** - Docker, GitHub Actions, and environment checks

## Installation and Setup

### Prerequisites

```bash
# Install Python dependencies
uv pip install pre-commit isort docstr-coverage pydocstyle safety detect-secrets

# Install Rust security tools
cargo install cargo-audit cargo-deny

# Install Docker (optional, for Dockerfile validation)
# Download from: https://www.docker.com/products/docker-desktop
```

### Initial Setup

```bash
# Navigate to project root
cd /c/codedev/llm/stats

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# Run hooks on all files (initial setup)
uv run pre-commit run --all-files
```

## Hook Categories and Phases

### Phase 1: Basic File Validation

- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newlines
- **mixed-line-ending**: Normalizes line endings to LF
- **check-yaml/json/toml**: Validates configuration file syntax
- **check-added-large-files**: Prevents committing large files (>50MB for models)
- **check-case-conflict**: Prevents case-sensitivity issues
- **check-merge-conflict**: Detects unresolved merge conflicts
- **detect-private-key**: Scans for accidentally committed private keys
- **check-ast**: Validates Python syntax
- **debug-statements**: Removes debug prints and statements

### Phase 2: Python Code Quality

```bash
# Run Python-specific hooks
uv run pre-commit run ruff --all-files
uv run pre-commit run ruff-format --all-files
uv run pre-commit run mypy --all-files
```

- **Ruff**: Ultra-fast Python linter and formatter (replaces flake8, isort, black)
- **MyPy**: Static type checking with Python 3.13 support
- Configured for strict typing with project-specific exclusions

### Phase 3: Security Scanning

```bash
# Run security scans
uv run pre-commit run bandit --all-files
uv run pre-commit run detect-secrets --all-files
uv run pre-commit run python-safety-dependencies-check --all-files
```

- **Bandit**: Python security vulnerability scanner
- **detect-secrets**: Prevents secrets from being committed
- **Safety**: Checks Python dependencies for known vulnerabilities

### Phase 4: Rust Code Quality

```bash
# Run Rust-specific hooks
uv run pre-commit run rust-format --all-files
uv run pre-commit run rust-clippy --all-files
uv run pre-commit run rust-audit --all-files
```

- **cargo fmt**: Code formatting with strict standards
- **cargo clippy**: Comprehensive linting with pedantic rules
- **cargo audit**: Security vulnerability checking
- **cargo deny**: License and dependency policy enforcement

### Phase 5: Documentation & Configuration

- **markdownlint**: Markdown formatting and consistency
- **yamllint**: YAML syntax and style validation
- Configured to allow GitHub Actions extensions

### Phase 6: Project-Specific Validation

```bash
# Test suite hooks
uv run pre-commit run pytest-fast --all-files        # Fast unit tests
uv run pre-commit run coverage-check --all-files     # 85% coverage minimum

# Code organization hooks
uv run pre-commit run optimize-imports --all-files   # Import sorting
uv run pre-commit run validate-pyproject --all-files # Project config validation
uv run pre-commit run check-model-size --all-files   # Model file size limits
uv run pre-commit run validate-mcp-config --all-files # MCP server validation
```

### Phase 7: Infrastructure Validation

- **docker-validate**: Dockerfile linting with Hadolint
- **github-actions-validate**: Workflow YAML validation
- **env-file-check**: Environment file security scanning

## Hook Execution Stages

### Pre-Commit Stage (runs on `git commit`)

- File format validation
- Python formatting and linting
- Security scans
- Rust formatting and basic linting
- Fast unit tests
- Documentation validation

### Pre-Push Stage (runs on `git push`)

- Comprehensive test coverage check (85% minimum)
- Full Rust test suite
- Rust security audits
- Performance benchmarks
- Infrastructure validation

## Configuration Files

### `.pre-commit-config.yaml`
Main configuration with all hook definitions and execution rules.

### `.secrets.baseline`
Baseline file for detect-secrets to track known/acceptable secrets.

### `.markdownlint.json`
Markdown formatting rules allowing common HTML elements.

### `deny.toml`
Rust dependency and license policy configuration.

### `.isort.cfg`
Python import sorting configuration optimized for the project.

## Common Commands

### Manual Hook Execution

```bash
# Run all hooks
uv run pre-commit run --all-files

# Run specific hook category
uv run pre-commit run --all-files --hook-stage pre-commit
uv run pre-commit run --all-files --hook-stage pre-push

# Run individual hooks
uv run pre-commit run ruff --all-files
uv run pre-commit run rust-clippy --all-files
uv run pre-commit run pytest-fast --all-files

# Show what would run without executing
uv run pre-commit run --all-files --verbose --show-diff-on-failure
```

### Maintenance Commands

```bash
# Update hook versions
uv run pre-commit autoupdate

# Clean hook environments
uv run pre-commit clean

# Install hooks after config changes
uv run pre-commit install --overwrite

# Skip hooks temporarily (not recommended)
SKIP=rust-clippy git commit -m "WIP: quick fix"
```

### Troubleshooting

```bash
# Debug hook failures
uv run pre-commit run --verbose --all-files

# Check configuration
uv run pre-commit validate-config

# View hook logs
cat ~/.cache/pre-commit/pre-commit.log

# Run hooks in isolation
uv run pre-commit try-repo . ruff --all-files
```

## CI/CD Integration

The configuration includes CI-specific settings:

- **autofix_commit_msg**: Standardized commit messages for auto-fixes
- **autofix_prs**: Enable automatic PR fixes
- **autoupdate_schedule**: Monthly dependency updates
- **skip**: Skip heavy Rust checks in CI environments

For GitHub Actions integration:

```yaml
- name: Run pre-commit
  uses: pre-commit/action@v3.0.1
  with:
    extra_args: --all-files --show-diff-on-failure
```

## Performance Optimizations

### Hook Execution Speed

- **fail_fast: false**: Continue running all hooks for complete feedback
- **Language-specific caching**: Environments are cached between runs
- **File-type filtering**: Hooks only run on relevant file types
- **Smart exclusions**: Skip generated files, caches, and archives

### Resource Management

- **Memory usage**: Hooks are designed to run with <500MB baseline memory
- **Parallel execution**: Independent hooks run concurrently
- **Incremental mode**: Only changed files are checked by default

## Project-Specific Notes

### Python 3.13 Support
- Updated MyPy configuration for Python 3.13
- UV package manager integration throughout
- Pydantic v2 compatibility

### Rust Integration
- Multi-workspace support (rust_extensions, rag-redis-system)
- Performance-optimized builds (LTO, codegen-units=1)
- SIMD optimization verification

### Model File Handling
- Large model files (>50MB) are flagged for review
- Cached models are excluded from hooks
- Binary format validation for .bin, .safetensors, .pt files

### Security Focus
- Zero-tolerance for hardcoded secrets
- Dependency vulnerability scanning
- License policy enforcement
- Environment file protection

## Getting Help

### Common Issues

1. **Hook fails on first run**: Install missing dependencies
2. **Rust hooks timeout**: Ensure cargo tools are installed
3. **MyPy import errors**: Install missing type stubs
4. **Large file warnings**: Move models to proper cache directory

### Support Resources

- Pre-commit documentation: https://pre-commit.com/
- Project issue tracker: Check repository issues
- Development team: Tag @dev-team for hook-related questions

### Contributing New Hooks

1. Add hook definition to `.pre-commit-config.yaml`
2. Test with `pre-commit run <hook-name> --all-files`
3. Update this documentation
4. Submit PR with test results

## Appendix: Hook Reference

### File Patterns

- Python: `*.py`, `*.pyi`
- Rust: `*.rs`
- Config: `*.yaml`, `*.yml`, `*.json`, `*.toml`
- Docs: `*.md`
- Docker: `Dockerfile*`
- Models: `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.gguf`

### Exclusion Patterns

- Cache directories: `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- Build artifacts: `dist/`, `build/`, `target/`
- Virtual environments: `.venv/`, `venv/`
- Archive/deprecated: `archived/`, `deprecated/`, `legacy/`
- Generated files: `site/`, `docs/_build/`

### Exit Codes

- **0**: Success
- **1**: Hook failed/found issues
- **2**: Configuration error
- **3**: Missing dependencies
