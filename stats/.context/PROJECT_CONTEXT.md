# Gemma Chatbot Project Context
**Generated**: 2025-09-13
**Location**: C:\codedev\llm\stats

## Quick Reference

### Project Identity
- **Type**: Local LLM CLI/HTTP chatbot
- **Model**: Google Gemma with PyTorch
- **Architecture**: Hybrid Python/Rust with clean architecture
- **Stack**: Python 3.11+, FastAPI, PyTorch, Rust (PyO3), MCP servers

### Current Sprint Focus
1. ✅ Pre-commit hooks and GitHub workflows implemented
2. 🔄 Gemini integration in progress
3. 🔄 Optimization scripts development
4. 🔄 Memory improvements

### Critical Fixes Required (Priority 1)
```python
# SECURITY: Fix CORS - Change from:
cors_origins = ["*"]  # ❌ VULNERABLE

# To:
cors_origins = ["http://localhost:3000", "https://app.example.com"]  # ✅ SECURE

# SECURITY: Enable API Authentication
api_key_required = True  # Currently False

# PERFORMANCE: Reduce timeout
inference_timeout = 30  # Currently 300 seconds
```

### Performance Targets
| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Memory | 500MB | 167MB | -67% |
| Startup | 2s | 0.5s | -75% |
| Throughput | 1x | 3x | +200% |
| Test Coverage | <50% | 85%+ | +35% |
| Security Grade | C | A | +3 levels |

### Architecture Patterns to Implement
- **Repository Pattern**: Data access abstraction
- **Dependency Injection**: Remove global singletons
- **Circuit Breaker**: Fault tolerance
- **CQRS**: Command/Query separation
- **Event Sourcing**: Audit trail

### Development Phases
**Phase 1 (Current - Weeks 1-4)**: Foundation
- Dependency injection container
- Repository pattern implementation
- Comprehensive test suite (85%+ coverage)

**Phase 2 (Weeks 5-8)**: Separation
- Microservices architecture
- API gateway implementation
- Circuit breakers for resilience

**Phase 3 (Weeks 9-12)**: Optimization
- CQRS implementation
- Event sourcing
- Rust-Python boundary optimization

**Phase 4 (Weeks 13-16)**: Scale
- Service mesh deployment
- Distributed tracing
- Full hexagonal architecture

### Key Files & Locations
```
C:\codedev\llm\stats\
├── .pre-commit-config.yaml     # Quality gates
├── .github/workflows/
│   ├── ci.yml                  # CI pipeline
│   ├── auto-fix.yml           # Auto corrections
│   └── code-quality.yml       # Quality analysis
├── IMPROVEMENT_REPORT.md       # Full analysis
└── .context/
    └── PROJECT_CONTEXT.md      # This file
```

### Agent Contributions Summary
- **code-reviewer**: 5 critical security issues identified
- **architect-reviewer**: Clean architecture violations found
- **python-pro**: Async and memory optimizations provided
- **rust-pro**: SIMD and allocator improvements suggested
- **docs-architect**: Comprehensive report generated

### Next Actions (TODO)
1. 🔴 Fix CORS configuration (security critical)
2. 🔴 Enable API authentication (security critical)
3. 🟡 Reduce inference timeout to 30s
4. 🟡 Add dependency injection container
5. 🟢 Complete Gemini-LangChain integration
6. 🟢 Create optimization scripts suite
7. 🟢 Implement memory profiling

### Recovery Commands
```bash
# Restore context in new session
mcp__memory__open_nodes --names "Gemma Chatbot Project"

# Get full project graph
mcp__memory__read_graph

# Search specific context
mcp__memory__search_nodes --query "Critical Issues"
```

---
*Context saved to memory graph with timestamp. Use recovery commands above to restore in future sessions.*
