# MCP Configuration Testing Framework

## Overview

This document describes the comprehensive MCP (Model Context Protocol) configuration and protocol testing framework that has been integrated into the `test_functional_rag.py` file. The framework provides production-ready validation for MCP servers, configurations, and protocol compliance.

## Features

### 1. MCP JSON Schema Validation
- **Purpose**: Validates MCP configuration files against proper JSON schema
- **Tests**:
  - Server configuration schema validation (`mcp-config.json`)
  - Claude MCP configuration schema validation (`rag-redis-mcp-corrected.json`)
  - Additional configuration files validation
- **Schema Coverage**:
  - MCP server configuration structure
  - Tool definitions with input schemas
  - Resource definitions with URI validation
  - Prompt template definitions
  - Configuration parameters

### 2. Tool Input/Output Schema Validation
- **Purpose**: Ensures all MCP tools have proper JSON schema definitions
- **Validations**:
  - Input schema JSON Schema Draft 7 compliance
  - Required field consistency with properties
  - Property type definitions
  - Schema completeness
- **Coverage**: All 7 RAG-Redis tools validated

### 3. Resource URI Validation
- **Purpose**: Validates resource URI format and structure
- **Validations**:
  - URI scheme validation
  - Custom scheme support (e.g., `rag://`)
  - Required field validation (name, description)
  - MIME type validation
- **Supported Schemes**: Standard HTTP/HTTPS, custom RAG scheme

### 4. Prompt Template Validation
- **Purpose**: Ensures prompt templates are properly structured
- **Validations**:
  - Required fields (name, description)
  - Argument structure validation
  - Boolean type validation for required flags
- **Coverage**: All prompt templates in configuration

### 5. MCP Server Communication Testing
- **Purpose**: Tests JSON-RPC 2.0 protocol compliance
- **Protocol Tests**:
  - Request message structure
  - Response message structure
  - Error message format
  - Notification format
- **Standards**: Full JSON-RPC 2.0 compliance

### 6. Configuration Parameter Validation
- **Purpose**: Validates specific configuration parameters
- **Parameter Categories**:
  - **Redis Configuration**:
    - Host string validation
    - Port range validation (1-65535)
    - Database number validation (non-negative)
  - **Embedding Configuration**:
    - Provider validation (local, openai, huggingface)
    - Model name validation
  - **Memory Configuration**:
    - Positive integer validation for limits
    - Consolidation interval validation
  - **Claude MCP Configuration**:
    - Command path existence
    - Environment variable naming
    - Additional properties support

### 7. Tool Execution Interface Testing
- **Purpose**: Validates tool execution through MCP interface
- **Testing**:
  - JSON-RPC tool call request generation
  - Parameter generation based on schema
  - Request structure validation
- **Coverage**: All defined tools with proper parameter generation

### 8. Protocol Compliance Comprehensive Testing
- **Purpose**: Comprehensive JSON-RPC 2.0 protocol compliance
- **Tests**:
  - Method naming compliance
  - Standard error code compliance (-32700 to -32603)
  - Notification format validation
  - Request/response structure validation
- **Compliance Threshold**: 80% minimum for passing

### 9. Network Connectivity Requirements Testing
- **Purpose**: Validates infrastructure requirements
- **Tests**:
  - Redis server connectivity (127.0.0.1:6379)
  - File system access validation
  - Executable path validation
  - Directory creation and write permissions
- **Scope**: All required infrastructure components

## Usage

### Command Line Options

```bash
# Run all tests (MCP + Functional)
uv run python test_functional_rag.py

# Run only MCP tests
uv run python test_functional_rag.py --mcp-only

# Run only functional tests
uv run python test_functional_rag.py --functional-only

# Run with verbose output
uv run python test_functional_rag.py --verbose
```

### Test Reports

The framework generates detailed reports:
- **MCP Test Report**: `mcp_test_report.md`
- **Functional Test Report**: `functional_test_report.md`
- **Console Output**: Real-time test results with emojis

## Configuration Files Tested

### 1. Server Configuration
- **File**: `C:/codedev/llm/stats/mcp-servers/rag-redis/mcp-config.json`
- **Type**: MCP server definition
- **Contents**: Tools, resources, prompts, configuration parameters

### 2. Claude Configuration
- **File**: `C:/codedev/llm/stats/rag-redis-mcp-corrected.json`
- **Type**: Claude MCP client configuration
- **Contents**: Server commands, arguments, environment variables

### 3. Additional Configurations
- Various MCP configuration files found in the project
- Automatic type detection and appropriate schema validation

## Schema Definitions

### MCP Server Configuration Schema
```json
{
  "type": "object",
  "required": ["name", "version", "description"],
  "properties": {
    "name": {"type": "string", "minLength": 1},
    "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+"},
    "description": {"type": "string"},
    "tools": {"type": "array"},
    "resources": {"type": "array"},
    "prompts": {"type": "array"},
    "configuration": {"type": "object"}
  }
}
```

### Claude MCP Configuration Schema
```json
{
  "type": "object",
  "required": ["mcpServers"],
  "properties": {
    "mcpServers": {
      "type": "object",
      "patternProperties": {
        "^[a-zA-Z0-9_-]+$": {
          "type": "object",
          "required": ["command"],
          "properties": {
            "command": {"type": "string"},
            "args": {"type": "array"},
            "env": {"type": "object"},
            "cwd": {"type": "string"},
            "autoStart": {"type": "boolean"}
          }
        }
      }
    }
  }
}
```

## Dependencies

Required Python packages:
- `jsonschema`: JSON Schema validation
- `pathlib`: Path handling
- `urllib.parse`: URI parsing
- `socket`: Network connectivity testing
- `re`: Regular expression validation
- `json`: JSON processing

## Error Handling

The framework provides comprehensive error handling:
- **Validation Errors**: Detailed schema validation messages
- **Network Errors**: Connection timeout and error reporting
- **File System Errors**: Permission and access error handling
- **Configuration Errors**: Missing or invalid configuration reporting

## Test Results Interpretation

### Success Indicators
- ✅ **SUCCESS**: Test passed completely
- ⚠️ **WARNING**: Test passed with warnings
- ❌ **FAILURE**: Test failed

### Success Criteria
- **100% Success Rate**: All tests pass without issues
- **≥80% Success Rate**: Acceptable with documented warnings
- **<80% Success Rate**: Configuration requires fixes

## Production Readiness Features

1. **Comprehensive Coverage**: Tests all aspects of MCP configuration
2. **Standards Compliance**: Full JSON-RPC 2.0 and MCP protocol compliance
3. **Flexible Schema**: Supports real-world configuration extensions
4. **Detailed Reporting**: Machine-readable and human-readable reports
5. **Infrastructure Validation**: Tests actual connectivity and permissions
6. **Error Resilience**: Continues testing even with individual failures
7. **Performance Metrics**: Network and file system performance testing

## Integration with CI/CD

The framework is designed for integration with CI/CD pipelines:
- Exit codes: 0 for success, 1 for failure
- Machine-readable reports in markdown format
- Configurable verbosity levels
- Isolated test execution options

## Future Enhancements

Potential areas for expansion:
1. **Live Server Testing**: Actual MCP server communication testing
2. **Performance Benchmarking**: Response time and throughput testing
3. **Security Testing**: Authentication and authorization validation
4. **Load Testing**: Concurrent client testing
5. **Integration Testing**: End-to-end workflow validation

## Conclusion

This MCP testing framework provides production-ready validation for MCP configurations and protocol compliance. It ensures that MCP servers and clients are properly configured and comply with protocol standards, providing confidence in deployment and operation.
