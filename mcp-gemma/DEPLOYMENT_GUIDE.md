# MCP Gemma Deployment Guide

This guide provides step-by-step instructions for deploying the MCP Gemma server on Windows systems.

## Prerequisites

### System Requirements

- **Windows 10/11** with PowerShell 5.1 or later
- **Python 3.8+** with pip
- **WSL** (Windows Subsystem for Linux) for gemma.cpp integration
- **Redis** (optional, for memory features)
- **Git** for cloning repositories

### Hardware Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 10GB+ free space for models and cache
- **CPU**: Multi-core processor (Intel/AMD x64)

## Installation Steps

### 1. Environment Setup

```powershell
# Clone the repository
git clone <repository-url>
cd mcp-gemma

# Setup Python environment
.\scripts\setup-environment.ps1
```

### 2. Install Dependencies

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies (enhanced features)
pip install -r requirements-optional.txt
```

### 3. Download Gemma Models

Download Gemma model files and place them in `/c/codedev/llm/.models/`:

- `gemma2-2b-it-sfp.sbs` (2B parameter model, fastest)
- `gemma2-7b-it-sfp.sbs` (7B parameter model, better quality)
- `tokenizer.spm` (tokenizer file)

### 4. Configure Redis (Optional)

If using memory features:

```powershell
# Install Redis for Windows
# Download from: https://github.com/microsoftarchive/redis/releases

# Start Redis service
redis-server
```

### 5. Build Gemma.cpp (Required)

```powershell
# In WSL, build gemma.cpp
wsl
cd /mnt/c/codedev/llm/gemma/gemma.cpp
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Configuration

### Server Configuration

Edit `config/server_config.yaml`:

```yaml
model:
  model_path: "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs"
  tokenizer_path: "/c/codedev/llm/.models/tokenizer.spm"
  gemma_executable: "/mnt/c/codedev/llm/gemma/gemma.cpp/build/gemma"

transports:
  http:
    enabled: true
    port: 8080
  websocket:
    enabled: true
    port: 8081

redis:
  enabled: true
  host: "localhost"
  port: 6379
```

## Deployment Options

### Option 1: Development Mode

For development and testing:

```powershell
# Start server directly
.\scripts\start-server.ps1 -ModelPath "C:\codedev\llm\.models\gemma2-2b-it-sfp.sbs" -Debug

# Or start specific transport
python server\main.py --mode http --port 8080 --model "C:\path\to\model.sbs" --debug
```

### Option 2: Production Mode

For production deployment:

```powershell
# Install as Windows service (requires Admin)
.\scripts\deploy-windows.ps1 -Action install -Environment production

# Start service
.\scripts\deploy-windows.ps1 -Action start

# Check service status
.\scripts\deploy-windows.ps1 -Action status
```

### Option 3: All Transports

Start all transport protocols simultaneously:

```powershell
python server\main.py --mode all --host 0.0.0.0 --port 8080 --ws-port 8081 --model "C:\path\to\model.sbs"
```

## Verification

### 1. Basic Functionality Test

```powershell
# Run basic tests
python tests\test_basic_functionality.py
```

### 2. Server Health Check

```powershell
# Test HTTP endpoint
.\scripts\test-server.ps1

# Or manually
curl http://localhost:8080/health
```

### 3. Client Demo

```powershell
# Run client demonstration
python examples\demo_client.py
```

## Integration with Other Projects

### MCP Integration

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "gemma-mcp": {
      "command": "python",
      "args": [
        "C:/codedev/llm/mcp-gemma/server/main.py",
        "--mode", "stdio",
        "--model", "C:/codedev/llm/.models/gemma2-2b-it-sfp.sbs"
      ]
    }
  }
}
```

### Stats Framework Integration

```python
from integration.stats_integration import GemmaMCPAgent

# Use as HTTP client
async with GemmaMCPAgent(client_type="http", base_url="http://localhost:8080") as agent:
    response = await agent.generate_response("Hello world")
    print(response)
```

### Claude Code Integration

Add to your Claude Code configuration:

```json
{
  "mcpServers": {
    "gemma": {
      "command": "python",
      "args": ["C:/codedev/llm/mcp-gemma/server/main.py", "--mode", "stdio", "--model", "C:/codedev/llm/.models/gemma2-2b-it-sfp.sbs"]
    }
  }
}
```

## Monitoring and Maintenance

### Log Files

- Server logs: `logs/server.log`
- Deployment logs: `logs/deploy.log`
- Error logs: `logs/error.log`

### Performance Monitoring

```powershell
# Get server metrics
curl http://localhost:8080/metrics

# Check Redis memory usage
redis-cli info memory
```

### Maintenance Tasks

```powershell
# Update dependencies
pip install -r requirements.txt --upgrade

# Optimize memory
# (Run Redis FLUSHDB if needed)

# Restart service
.\scripts\deploy-windows.ps1 -Action restart
```

## Troubleshooting

### Common Issues

**1. Model Not Found**
```
Error: Model file not found
Solution: Verify model path in configuration
```

**2. WSL Connection Failed**
```
Error: WSL not available
Solution: Install WSL and build gemma.cpp
```

**3. Redis Connection Failed**
```
Error: Redis connection refused
Solution: Start Redis service or disable Redis in config
```

**4. Port Already in Use**
```
Error: Address already in use
Solution: Change port in configuration or stop conflicting service
```

### Debug Mode

Enable debug logging:

```powershell
# Start with debug
python server\main.py --debug --model "C:\path\to\model.sbs"

# Set environment variable
$env:GEMMA_LOG_LEVEL="DEBUG"
```

### Performance Tuning

**Memory Optimization:**
- Adjust `max_tokens` in configuration
- Use smaller models for faster response
- Enable Redis for caching

**Concurrency:**
- Adjust `max_concurrent_requests`
- Use connection pooling
- Enable HTTP keep-alive

## Security Considerations

### Network Security

- Bind to localhost for local use only
- Use reverse proxy for public access
- Enable HTTPS in production

### Access Control

- Implement authentication if needed
- Use firewall rules to restrict access
- Monitor access logs

## Support

### Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory
- **Configuration**: `config/` directory

### Getting Help

1. Check the troubleshooting section
2. Review log files for errors
3. Run basic functionality tests
4. Check GitHub issues for known problems

### Reporting Issues

When reporting issues, include:

- System information (Windows version, Python version)
- Configuration files (remove sensitive data)
- Log files
- Steps to reproduce the issue
- Expected vs actual behavior