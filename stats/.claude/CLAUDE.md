# CLAUDE.md - Project-Specific Directives

## 🚨 CRITICAL DIRECTIVES
1. **TREAT ALL WARNINGS AS ERRORS** - Fix immediately, no exceptions
2. **TOKEN EFFICIENCY FIRST** - Minimize context, use parallel agents
3. **NO STUB CODE** - All implementations must be functional
4. **COMMIT FREQUENTLY** - Git add/commit after each significant change
5. **BUILD CONTEXT FIRST** - Research thoroughly before writing code

## Project Architecture
- **Location**: C:\codedev\llm\stats
- **Stack**: Python 3.13 + Rust + C++ (gemma.cpp)
- **Model**: Gemma/Phi-2 for text, Gemma3 for vision
- **Memory**: RAG-Redis with 5-tier system

## Integration Status
### ✅ Working
- Redis on port 6380 (Windows-friendly)
- RAG-Redis MCP server
- FastAPI on port 8001
- Phi-2 model downloaded (5.3GB)
- Test coverage (8.28%)

### 🔧 In Progress
- gemma.cpp integration via FFI
- PyO3 bindings fixes
- Stub code removal

### ❌ Blocked
- PyTorch 2.8.0 incompatibility
- Gemma HF license (401 error)

## Development Patterns
```python
# ALWAYS use UV
uv run python  # Never bare python
uv pip install  # Never bare pip

# Testing commands
uv run pytest --cov=src
uv run cargo clippy -- -D warnings
uv run ruff check src --fix
```

## Git Configuration
```bash
# .gitignore additions
gemma.cpp/
*.pyc
__pycache__/
.coverage
htmlcov/
*.dll
*.so
```

## Debugging Tools Priority
1. `cargo clippy` - Rust linting
2. `ruff` - Python linting
3. `mypy` - Type checking
4. `rust-analyzer` - IDE support

## MCP Tools Usage
- **sequential-thinking**: Plan complex tasks
- **memory**: Store critical context
- **desktop-commander**: System operations
- **Task**: Launch parallel agents

## Code Quality Standards
- Zero warnings policy
- 85% test coverage target
- Type hints required
- Error handling mandatory
- No `print()` - use logging

## Performance Targets
- Memory: <500MB baseline
- Startup: <2s
- Inference: <100ms/token
- Test suite: <30s

## Security Requirements
- Path traversal protection
- Input validation
- No hardcoded credentials
- Sanitize all file operations

## Documentation Updates
- Update CLAUDE.md after major changes
- Keep TODO.md current
- Document all public APIs
- Include usage examples
