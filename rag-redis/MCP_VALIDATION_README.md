# RAG-Redis MCP Validation Suite

This comprehensive validation suite tests the RAG-Redis MCP server functionality, including all tools, Claude CLI integration, performance benchmarks, and error handling.

## Overview

The validation suite consists of three main components:

1. **`validate_mcp.py`** - Comprehensive Python validation script
2. **`test_claude_integration.sh`** - Claude CLI integration tests  
3. **`run_mcp_validation.py`** - Orchestration script that runs everything

## Prerequisites

### Required Software
- **Python 3.8+** with asyncio support
- **Rust/Cargo** for building the MCP server
- **Redis Server** running on port 6380
- **Claude CLI** (optional, for integration tests)
- **Bash** (WSL, Git Bash, or native Linux/macOS)

### Required Services
```bash
# Start Redis server (Windows)
redis-server --port 6380

# Start Redis server (Linux/macOS)
redis-server --port 6379  # Note: Scripts expect 6380, may need adjustment
```

### Build MCP Server
```bash
cd rag-redis-system/mcp-server
cargo build --release
```

## Quick Start

### Option 1: Run Everything (Recommended)
```bash
# Run the complete validation suite
python run_mcp_validation.py c:\codedev\llm\rag-redis
```

This will:
1. Check all prerequisites
2. Build the MCP server if needed
3. Run comprehensive Python validation
4. Run Claude CLI integration tests
5. Generate a consolidated report

### Option 2: Run Individual Components

#### Python Validation Only
```bash
python validate_mcp.py c:\codedev\llm\rag-redis
```

#### Claude Integration Only
```bash
bash test_claude_integration.sh
```

## Test Categories

### Python Validation (`validate_mcp.py`)

**Pre-flight Checks:**
- ✅ MCP configuration validation
- ✅ Redis connectivity test
- ✅ Server binary verification

**Core Functionality Tests:**
- ✅ Server startup and initialization
- ✅ Health check tool
- ✅ Document ingestion (`ingest_document`)
- ✅ Semantic search (`search`)
- ✅ Hybrid search (`hybrid_search`)

**Memory System Tests:**
- ✅ Basic memory operations (`memory_store`, `memory_recall`)
- ✅ Agent-specific memory (`agent_memory_store`, `agent_memory_retrieve`)
- ✅ Project context (`project_context_save`, `project_context_load`)

**Error Handling Tests:**
- ✅ Invalid tool calls
- ✅ Invalid parameters
- ✅ Graceful error responses

**Performance Benchmarks:**
- ⏱️ Document ingestion timing
- ⏱️ Search operation timing  
- ⏱️ Memory operation timing
- 📊 Performance metrics collection

### Claude CLI Integration (`test_claude_integration.sh`)

**Integration Tests:**
- 🔗 Basic Claude MCP connection
- 📄 Document ingestion through Claude
- 🔍 Search functionality through Claude
- 🧠 Memory operations through Claude
- 🤖 Agent-specific memory through Claude
- 📋 Project context through Claude
- ❌ Error handling through Claude
- 🔄 Comprehensive workflow testing
- ⚡ Performance stress testing

## Expected Performance Metrics

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Document Ingestion | < 500ms | Per document with embedding |
| Vector Search | < 100ms | 10K vectors with SIMD |
| Memory Store | < 50ms | Single memory operation |
| Memory Recall | < 100ms | Query with filtering |
| Health Check | < 50ms | Basic system status |

## Output and Reports

### Console Output
- ✅ **Green checkmarks** for passed tests
- ❌ **Red X marks** for failed tests  
- ⚠️ **Yellow warnings** for non-critical issues
- 📊 **Performance metrics** for benchmarks

### Generated Files
- `mcp_validation_results.json` - Detailed Python validation results
- `claude_integration_test.log` - Claude integration test log
- `mcp_validation_consolidated_report.json` - Combined results summary

### Sample Report Structure
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total_tests": 15,
    "passed_tests": 14,
    "failed_tests": 1,
    "success_rate": 93.3,
    "overall_status": "PARTIAL_SUCCESS"
  },
  "performance_benchmarks": {
    "avg_ingestion_ms": 245.6,
    "avg_search_ms": 45.2,
    "avg_memory_store_ms": 23.1
  }
}
```

## Troubleshooting

### Common Issues

**Redis Connection Failed**
```bash
# Check if Redis is running
redis-cli -p 6380 ping

# Start Redis on correct port
redis-server --port 6380
```

**MCP Server Binary Not Found**
```bash
# Build the server
cd rag-redis-system/mcp-server
cargo build --release
```

**Claude CLI Not Available**
```bash
# Install Claude CLI (if available)
# Or skip Claude integration tests
python validate_mcp.py c:\codedev\llm\rag-redis
```

**Permission Issues (Windows)**
```powershell
# Run as Administrator or adjust execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**WSL Path Issues**
- The scripts handle Windows/WSL path conversion automatically
- Ensure WSL is properly installed if using `wsl` commands

### Debug Mode

Enable debug output:
```bash
# Python validation with verbose output
python validate_mcp.py c:\codedev\llm\rag-redis --debug

# Claude integration with debug logging
RUST_LOG=debug bash test_claude_integration.sh
```

### Test Specific Components

**Test Single Tool:**
```python
# Modify validate_mcp.py to run specific test
await self._run_test("Health Check Only", self.test_health_check_tool)
```

**Test Specific Memory Tier:**
```python
# Test specific memory type
"memory_type": "long_term"  # or working, short_term, episodic, semantic
```

## Configuration Validation

The scripts validate the MCP configuration (`mcp.json`) including:

- ✅ Server binary path exists
- ✅ Required environment variables present  
- ✅ Tool definitions complete
- ✅ Memory tier configurations valid
- ✅ Agent configurations proper

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | Critical failures (server won't start) |
| 2 | Non-critical failures (some tools failed) |
| 130 | User interrupted (Ctrl+C) |

## Extension Points

### Adding New Tests

**Python Validation:**
```python
async def test_my_new_feature(self) -> TestResult:
    # Your test implementation
    return TestResult(test_name="My Test", passed=True, duration_ms=100)

# Add to run_all_tests()
await self._run_test("My New Feature", self.test_my_new_feature)
```

**Claude Integration:**
```bash
test_my_new_feature() {
    print_test "Testing my new feature..."
    # Your test implementation
}

# Add to test_functions array
test_functions=(
    # ... existing tests
    "test_my_new_feature"
)
```

### Custom Performance Thresholds

Modify performance expectations in `validate_mcp.py`:
```python
# Adjust performance warnings
if perf_result.performance_metrics.get("avg_search_ms", 0) > 200:  # Custom threshold
    warnings.append("Search performance slower than expected")
```

## Continuous Integration

For CI/CD integration:
```bash
#!/bin/bash
# ci_validation.sh

# Start Redis in background
redis-server --port 6380 --daemonize yes

# Run validation
python run_mcp_validation.py /path/to/rag-redis

# Capture exit code
EXIT_CODE=$?

# Cleanup
redis-cli -p 6380 shutdown

exit $EXIT_CODE
```

## Support

If validation fails:
1. Check the detailed logs in generated JSON files
2. Verify all prerequisites are met
3. Test individual components separately
4. Check Redis server status and connectivity
5. Ensure MCP server binary is properly built

For performance issues:
1. Review benchmark results in the report
2. Check Redis memory usage: `redis-cli -p 6380 info memory`
3. Monitor system resources during testing
4. Consider adjusting batch sizes or timeouts