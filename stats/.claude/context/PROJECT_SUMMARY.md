# Gemma LLM Stats - Project Context Summary
*Last Updated: 2025-01-15*

## 🎯 Quick Reference

### Project Status
- **Location**: `C:\codedev\llm\stats`
- **Repository**: https://github.com/david-t-martel/gemma-llm-stats (private)
- **Test Coverage**: 5.11% → Target: 85%
- **Build Status**: ✅ Compiles | ⚠️ Pre-commit hooks failing
- **Services**: Redis (6379) ✅ | FastAPI (8001) ✅

### Recent Achievements
✅ Fixed critical PyO3 vulnerability (0.20.3 → 0.24.2)
✅ Consolidated duplicate code (-96 lines)
✅ Standardized logging (44+ files)
✅ Fixed all syntax errors
✅ 176 tests collecting successfully

### Current Focus
🔧 Removing stub code implementations
🔧 gemma.cpp FFI integration
🔧 Increasing test coverage
🔧 Fixing pre-commit hooks

## 🏗️ Architecture Overview

```
Local LLM Framework
├── ReAct Agent (Reasoning + Acting)
├── Rust Extensions (PyO3)
├── gemma.cpp (5x faster inference)
├── RAG-Redis System
└── Multi-tier Memory
```

### Technology Stack
- **Core**: Python 3.13 + Rust + C++
- **Models**: Gemma-2B/7B, Phi-2
- **Backend**: Redis, FastAPI
- **Tools**: UV, Cargo, CMake

## 📝 Critical Conventions

### Always Use UV
```bash
uv run python       # Never bare python
uv pip install      # Never bare pip
uv run pytest       # For testing
```

### Import Pattern
```python
from src.agent.core import Agent  # Absolute imports
from src.utils import get_logger  # Standardized logging
```

### Quality Standards
- Zero warnings policy
- Type hints required
- Error handling mandatory
- No print() - use logging

## 🚀 Quick Start Commands

### Development
```bash
# Run agent
uv run python main.py --lightweight

# Run tests
uv run pytest tests/ -v

# Check quality
uv run ruff check src --fix
cargo clippy -- -D warnings
```

### Building
```bash
# Python + Rust
uv run maturin develop --release

# RAG-Redis
cd rag-redis-system && cargo build --release

# gemma.cpp (Windows)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 🔍 Known Issues

1. **PyTorch 2.8.0**: Incompatibility issue
2. **Gemma HF License**: 401 authentication error
3. **Pre-commit Hooks**: Currently failing
4. **GitHub Vulnerability**: 1 reported (may be cached)

## 🎯 Immediate Priorities

1. Remove all stub code
2. Complete gemma.cpp integration
3. Fix pre-commit hooks
4. Increase test coverage to 85%
5. Document public APIs

## 🤖 Agent Deployment History

### Successful Patterns
- Security audit → Python fixes → Rust compilation
- Parallel agents for independent tasks
- Specialized agents for language-specific issues

### Key Agents Used
- **security-auditor**: Vulnerability fixes
- **python-pro**: Linting and imports
- **rust-pro**: Compilation and dependencies
- **legacy-modernizer**: Code consolidation

## 📊 Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Memory | ~500MB | <400MB |
| Startup | ~2s | <1s |
| Inference | ~100ms/token | <50ms/token |
| Test Coverage | 5.11% | 85% |
| Test Runtime | Unknown | <30s |

## 🔮 Roadmap

### Near Term (This Week)
- [ ] Complete stub removal
- [ ] Fix pre-commit hooks
- [ ] gemma.cpp integration
- [ ] Test coverage to 25%

### Medium Term (This Month)
- [ ] Streaming generation
- [ ] Image analysis (Gemma3)
- [ ] Test coverage to 85%
- [ ] API documentation

### Long Term (Q1 2025)
- [ ] GPU acceleration
- [ ] INT4/INT8 quantization
- [ ] Distributed inference
- [ ] Production deployment

## 📁 Key Files

- **Entry**: `main.py` - CLI interface
- **Core**: `src/agent/core.py` - Agent system
- **Config**: `.env`, `pyproject.toml`
- **Tests**: `test_*.py`, `tests/`
- **Rust**: `rust_extensions/`, `rag-redis-system/`

## 🔗 Integration Points

- **Redis**: localhost:6379 (RAG backend)
- **FastAPI**: localhost:8001 (HTTP API)
- **MCP Servers**: RAG-Redis, Gemini, GitHub
- **Models**: ./models/ (Gemma, Phi-2)

## 💡 Tips for Next Session

1. Check Redis is running: `redis-cli ping`
2. Activate environment: `source .venv/Scripts/activate`
3. Pull latest: `git pull origin main`
4. Check tests: `uv run pytest tests/ -v`
5. Review TODO.md for current tasks

---

*For full context, see `project-context-2025-01-15.json`*
