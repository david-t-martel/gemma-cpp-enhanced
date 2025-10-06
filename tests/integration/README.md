# LLM Integration Test Suite

Comprehensive end-to-end testing for the LLM development ecosystem including Python stats framework, RAG-Redis memory system, MCP servers, and Gemma inference engine.

## 📁 Test Structure

```
tests/integration/
├── agent/                 # Agent framework tests
│   └── test_react_agent_integration.py
├── rag/                   # RAG and Redis memory tests
│   └── test_rag_redis_memory.py
├── mcp/                   # MCP communication tests
│   └── test_mcp_communication.py
├── performance/           # Performance benchmarks
│   └── test_performance_benchmarks.py
├── stress/                # Stress testing
│   └── test_stress_testing.py
├── reports/               # Test report generation
│   └── test_report_generator.py
├── fixtures/              # Shared test fixtures
├── conftest.py            # Pytest configuration and fixtures
└── run_integration_tests.py  # Main test runner
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio pytest-cov fakeredis psutil

# Start Redis (required for RAG tests)
redis-server

# Ensure models are downloaded
uv run python -m src.gcp.gemma_download --auto
```

### Running Tests

```bash
# Run all integration tests
uv run python tests/integration/run_integration_tests.py

# Run quick smoke tests
uv run python tests/integration/run_integration_tests.py --suite quick

# Run performance benchmarks
uv run python tests/integration/run_integration_tests.py --suite performance

# Run stress tests (slow, resource intensive)
uv run python tests/integration/run_integration_tests.py --suite stress

# Run specific test pattern
uv run python tests/integration/run_integration_tests.py --suite specific --pattern "test_agent"

# Run with coverage
uv run python tests/integration/run_integration_tests.py --coverage
```

## 📊 Test Coverage

### Agent Tests (`agent/`)
- **ReAct Agent Integration**: Tool execution chains, error recovery, parallel execution
- **RAG Integration**: Document retrieval and augmentation
- **Memory Management**: Context window handling, memory consolidation
- **Multi-model Support**: Model fallback and switching

### RAG-Redis Memory Tests (`rag/`)
- **Multi-tier Memory**: Working, short-term, long-term, episodic, semantic
- **Vector Similarity Search**: Embedding-based retrieval
- **Memory Consolidation**: Automatic tier transitions
- **Document Processing**: Chunking, embedding, indexing

### MCP Communication Tests (`mcp/`)
- **Server Management**: Connection, disconnection, recovery
- **Tool Discovery**: Cross-server tool registration
- **Protocol Compliance**: Message format validation
- **Concurrent Operations**: Parallel server communication

### Performance Benchmarks (`performance/`)
- **Inference Latency**: Single and batch inference timing
- **Concurrency**: Parallel request handling
- **Memory Usage**: Leak detection, efficiency monitoring
- **Throughput**: Operations per second metrics

### Stress Tests (`stress/`)
- **Multiple Agents**: 100+ concurrent agents
- **Large Documents**: Gigabyte-scale processing
- **Resource Exhaustion**: CPU, memory, file descriptor limits
- **Network Failures**: Intermittent failures, partition recovery

## 📈 Performance Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Inference P95 Latency | < 500ms | 95th percentile inference time |
| Inference P99 Latency | < 1000ms | 99th percentile inference time |
| Requests/Second | > 10 | Minimum concurrent request handling |
| Memory Growth | < 100MB | Maximum memory increase during tests |
| Redis Ops/Second | > 1000 | Minimum Redis operation throughput |
| RAG Search Latency | < 100ms | Vector similarity search time |
| Agent Response Time | < 2000ms | Full pipeline response time |

## 🔧 Test Configuration

### Using Configuration File

Create `test_config.json`:

```json
{
  "coverage": true,
  "markers": "not slow",
  "include_stress": false,
  "performance": {
    "iterations": 100,
    "concurrent_requests": 20
  },
  "redis": {
    "host": "localhost",
    "port": 6379
  }
}
```

Run with configuration:

```bash
uv run python run_integration_tests.py --config-file test_config.json
```

## 📝 Test Reports

Reports are generated in multiple formats:

- **JSON**: Machine-readable detailed results
- **HTML**: Interactive web report with visualizations
- **Markdown**: Documentation-friendly format
- **Charts**: Performance visualization (requires matplotlib)

### Report Location

Reports are saved to `test_reports/` directory:

```
test_reports/
├── test_report_[timestamp].json
├── test_report_[timestamp].html
├── test_report_[timestamp].md
├── test_charts_[timestamp].png
└── coverage/               # If coverage enabled
    └── index.html
```

## 🧪 Writing New Tests

### Test Structure

```python
import pytest
from pathlib import Path

class TestNewFeature:
    """Test suite for new feature."""

    @pytest.mark.asyncio
    async def test_async_operation(self, async_redis_client):
        """Test async operation with Redis."""
        result = await async_redis_client.set("key", "value")
        assert result

    def test_sync_operation(self, performance_monitor):
        """Test with performance monitoring."""
        performance_monitor.start()

        # Your test code here
        result = expensive_operation()

        metrics = performance_monitor.stop()
        performance_monitor.assert_performance(
            max_duration=1.0,  # 1 second max
            max_memory=50      # 50MB max
        )
```

### Available Fixtures

| Fixture | Description |
|---------|-------------|
| `redis_client` | Synchronous FakeRedis client |
| `async_redis_client` | Async FakeRedis client |
| `react_agent` | Configured ReAct agent |
| `tool_registry` | Tool registry with sample tools |
| `mcp_server_config` | MCP server configuration |
| `memory_config` | Memory system configuration |
| `performance_monitor` | Performance monitoring utility |
| `sample_documents` | Test documents for RAG |

## 🐛 Debugging Failed Tests

### Verbose Output

```bash
# Run with verbose output
pytest -vvs tests/integration/agent/

# Run specific test with debugging
pytest -vvs tests/integration/agent/test_react_agent_integration.py::TestReActAgentIntegration::test_agent_with_redis_memory --pdb
```

### Check Logs

Test logs are available in the test output and can be enhanced:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔍 Known Issues and Limitations

1. **FakeRedis TTL**: TTL expiration may not work exactly like real Redis
2. **Mock Model Latency**: Mock models have artificial delays for testing
3. **Memory Tests**: Some memory leak tests require actual model loading
4. **Stress Tests**: May require significant resources (CPU, memory)

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FakeRedis Documentation](https://github.com/cunla/fakeredis-py)
- [Model Context Protocol](https://github.com/anthropics/model-context-protocol)

## 🤝 Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Add appropriate markers (`@pytest.mark.slow`, `@pytest.mark.asyncio`)
3. Include docstrings explaining test purpose
4. Update this README with new test coverage
5. Ensure tests are idempotent and don't leave side effects