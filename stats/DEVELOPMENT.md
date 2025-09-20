# Development Guide

## Architecture Overview

This LLM chatbot framework follows Domain-Driven Design (DDD) principles with clean separation of concerns:

```
src/
├── domain/              # Pure business logic (no external dependencies)
│   ├── interfaces/      # Abstract protocols and interfaces
│   └── models/          # Domain entities and value objects
├── infrastructure/      # External integrations and implementations
│   └── llm/            # LLM implementations (Gemma, etc.)
├── application/         # Use cases and services
│   └── inference/       # High-level inference orchestration
└── shared/             # Common utilities
    ├── config/         # Configuration management
    └── exceptions.py   # Custom exceptions
```

## Key Design Patterns

### 1. Protocol-Based Design
- `LLMProtocol` defines the contract for all LLM implementations
- Enables easy testing with mock implementations
- Supports different model backends (Gemma, future: GPT, Claude, etc.)

### 2. Async/Await Throughout
- All I/O operations are async for better performance
- Supports concurrent request handling
- Non-blocking model operations

### 3. Resource Management
- Context managers for proper cleanup
- Memory-efficient streaming responses
- GPU memory management

### 4. Type Safety
- Full type hints with mypy validation
- Pydantic models for runtime validation
- Protocol-based interfaces for compile-time safety

## Core Components

### InferenceService
High-level service providing:
- Response generation (single and streaming)
- Session management
- Caching
- Error handling
- Performance monitoring

```python
async with InferenceService() as service:
    response = await service.generate_response(
        session=session,
        user_message="Hello!",
        temperature=0.7
    )
```

### ChatSession
Domain model for conversation state:
- Message history
- Token usage tracking
- Metadata management
- Format conversion for inference

```python
session = ChatSession(title="My Chat")
session.system_prompt = "You are helpful."
session.add_user_message("Hello!")
```

### GemmaLLM
Production-ready Gemma implementation:
- Model loading with optimization
- Quantization support (int4, int8, fp16)
- Streaming generation
- Memory management

## Development Workflow

### 1. Setup Environment
```bash
# Install dependencies
uv pip install -r requirements.txt
uv pip install -e ".[dev]"

# Verify installation
python test_framework.py
```

### 2. Code Quality
```bash
# Type checking
uv run mypy src/

# Linting and auto-fixes
uv run ruff check src/ --fix

# Code formatting
uv run black src/ --line-length 100

# Run tests
uv run pytest tests/ -v
```

### 3. Configuration
Settings are managed through Pydantic with environment variable support:

```python
# Environment variables (optional)
export GEMMA_MODEL__NAME="google/gemma-7b-it"
export GEMMA_MODEL__TEMPERATURE="0.8"
export GEMMA_PERFORMANCE__DEVICE="cuda"
export GEMMA_CACHE__ENABLED="true"

# Or use defaults from settings.py
settings = get_settings()
```

## Adding New LLM Implementations

1. **Create implementation class**:
```python
class NewLLM(BaseLLM):
    async def _load_model_implementation(self) -> None:
        # Load your model here
        pass

    async def _generate_implementation(self, prompt: str, **kwargs) -> str:
        # Generate response
        pass
```

2. **Register in InferenceService**:
```python
def _create_default_llm(self) -> LLMProtocol:
    model_name = self.settings.model.name.lower()

    if "new-model" in model_name:
        return NewLLM(self.settings)
    elif "gemma" in model_name:
        return GemmaLLM(self.settings)
```

3. **Add tests**:
```python
def test_new_llm():
    llm = NewLLM(test_settings)
    assert llm.model_name == "new-model"
```

## Testing Strategy

### Unit Tests
- Mock external dependencies
- Test business logic in isolation
- Fast execution (<1s per test)

### Integration Tests
- Test with actual models (optional)
- Verify end-to-end workflows
- Performance benchmarks

### Example Test Structure
```python
@pytest.mark.asyncio
async def test_inference_service():
    # Arrange
    mock_llm = create_mock_llm()
    service = InferenceService(llm=mock_llm)

    # Act
    async with service:
        response = await service.generate_response(
            session=ChatSession(),
            user_message="test"
        )

    # Assert
    assert response.content == "mock response"
    assert response.token_usage is not None
```

## Performance Considerations

### Memory Management
- Use `torch.cuda.empty_cache()` after inference
- Implement model offloading for multi-model scenarios
- Monitor memory usage with `get_model_info()`

### Optimization Features
- **Flash Attention**: Faster attention computation
- **BetterTransformer**: Optimized transformer layers
- **torch.compile**: JIT compilation (PyTorch 2.0+)
- **Quantization**: Reduced memory usage (int4/int8)

### Caching Strategy
- Response caching based on input hash
- TTL-based cache expiration
- Memory-bounded cache with LRU eviction

## Monitoring and Observability

### Built-in Metrics
```python
# Service metrics
stats = await service.get_statistics()
# {
#   "request_count": 42,
#   "avg_request_time": 1.2,
#   "cache_hit_rate": 0.85,
#   "cache_size": 100
# }

# Model metrics
info = await service.get_model_info()
# {
#   "inference_count": 42,
#   "total_tokens_generated": 1337,
#   "memory_usage": {...}
# }
```

### Health Checks
```python
health = await service.health_check()
# {
#   "status": "healthy",
#   "model_info": {...},
#   "statistics": {...}
# }
```

## Deployment

### Production Checklist
- [ ] Set `environment="production"` in settings
- [ ] Configure proper logging levels
- [ ] Set up GPU monitoring
- [ ] Enable response caching
- [ ] Configure security settings
- [ ] Set resource limits (memory, timeouts)

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["python", "-m", "src.server.main"]
```

### Environment Variables
```bash
GEMMA_ENVIRONMENT=production
GEMMA_LOG_LEVEL=INFO
GEMMA_MODEL__NAME=google/gemma-2b-it
GEMMA_PERFORMANCE__DEVICE=cuda
GEMMA_CACHE__ENABLED=true
GEMMA_SERVER__HOST=0.0.0.0
GEMMA_SERVER__PORT=8000
```

## Troubleshooting

### Common Issues

**"Model not found"**
- Verify model name in settings
- Check internet connection for downloads
- Ensure sufficient disk space

**"CUDA out of memory"**
- Reduce batch size
- Enable CPU offloading
- Use quantization (int8/int4)
- Clear GPU cache between requests

**"Import errors"**
- Run `uv pip install -e .` to install in development mode
- Check Python path configuration
- Verify all dependencies are installed

**"Type errors"**
- Run `mypy src/` to check type consistency
- Update type annotations
- Use `typing_extensions` for newer features

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose model loading
settings.log_level = LogLevel.DEBUG
```

## Contributing

1. **Fork and clone**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Follow code style**: Run linting and formatting
4. **Add tests**: Maintain >85% coverage
5. **Update docs**: Include docstrings and examples
6. **Submit PR**: With clear description and tests
