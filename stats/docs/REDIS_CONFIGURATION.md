# Redis Configuration Guide

This guide explains the centralized Redis configuration system implemented in the LLM Stats project.

## Overview

The project now uses a centralized Redis configuration system that:
- **Automatically detects the environment** (Windows, Docker, Linux)
- **Uses appropriate default ports** (6380 for Windows, 6379 for Docker/Linux)
- **Supports environment variables** for all configuration options
- **Provides fallback mechanisms** when Redis is unavailable
- **Centralizes all Redis settings** across Python, Rust, and configuration files

## Quick Start

### 1. Windows Development

For Windows development, Redis typically runs on port 6380 to avoid conflicts:

```bash
# Set environment variables
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0

# Or use the Redis URL format
REDIS_URL=redis://localhost:6380/0
```

### 2. Docker Development

When using Docker, Redis runs on the standard port 6379 within the container network:

```bash
# These are set automatically in Docker compose
DOCKER_CONTAINER=true
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://redis:6379/0
```

### 3. Linux/Unix Development

For Linux/Unix systems, use the standard Redis port:

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0
```

## Environment Variables

### Basic Connection Settings

| Variable | Description | Default (Windows) | Default (Docker) |
|----------|-------------|-------------------|------------------|
| `REDIS_HOST` | Redis server hostname | `localhost` | `redis` |
| `REDIS_PORT` | Redis server port | `6380` | `6379` |
| `REDIS_DB` | Redis database number | `0` | `0` |
| `REDIS_URL` | Complete Redis URL | `redis://localhost:6380/0` | `redis://redis:6379/0` |
| `REDIS_USERNAME` | Redis username (Redis 6.0+) | - | - |
| `REDIS_PASSWORD` | Redis password | - | - |

### Connection Pool Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_MAX_CONNECTIONS` | Maximum pool connections | `20` |
| `REDIS_MIN_CONNECTIONS` | Minimum pool connections | `1` |
| `REDIS_CONNECTION_TIMEOUT` | Connection timeout (seconds) | `5` |
| `REDIS_SOCKET_TIMEOUT` | Socket timeout (seconds) | `10` |

### Retry and Error Handling

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_MAX_RETRIES` | Maximum retry attempts | `3` |
| `REDIS_RETRY_DELAY_MS` | Initial retry delay (ms) | `100` |
| `REDIS_ENABLE_FALLBACK` | Enable in-memory fallback | `true` |

### Advanced Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_KEY_PREFIX` | Key prefix for all operations | `llm_stats:` |
| `REDIS_ENCODING` | String encoding | `utf-8` |
| `REDIS_DECODE_RESPONSES` | Auto-decode responses | `true` |
| `REDIS_SSL_ENABLED` | Enable SSL/TLS | `false` |

## Configuration Files

### Environment Files

The project provides several environment file templates:

1. **`.env.redis`** - Template with all Redis configuration options
2. **`.env.redis.windows`** - Windows-optimized configuration
3. **`.env.redis.docker`** - Docker-optimized configuration

To use an environment file:

```bash
# Copy template and customize
cp .env.redis .env.redis.local

# Load environment variables
source .env.redis.local
# or
export $(cat .env.redis.local | xargs)
```

### Docker Compose

Redis port mapping in Docker Compose is now configurable:

```yaml
services:
  redis:
    ports:
      - "${REDIS_HOST_PORT:-6380}:6379"  # Maps to host port 6380 by default
```

Override the host port:

```bash
REDIS_HOST_PORT=6379 docker-compose up
```

## Configuration Helper Script

Use the Redis configuration helper script for setup and validation:

```bash
# Validate current configuration
python scripts/redis-config-helper.py validate

# Setup configuration for your environment
python scripts/redis-config-helper.py setup windows
python scripts/redis-config-helper.py setup docker

# Test Redis connection
python scripts/redis-config-helper.py test

# Show configuration info
python scripts/redis-config-helper.py info
```

## Python Usage

### Using the Configuration

```python
from src.shared.config.redis_config import get_redis_config, get_redis_client

# Get configuration (auto-detects environment)
config = get_redis_config()

# Create Redis client
client = get_redis_client(config)

# Or use default configuration
client = get_redis_client()
```

### Custom Configuration

```python
from src.shared.config.redis_config import RedisConfig

# Create custom configuration
config = RedisConfig(
    host="my-redis-server",
    port=6379,
    db=1,
    password="my-password"  # pragma: allowlist secret
)

client = config.create_client()
```

### Test Configuration

```python
from src.shared.config.redis_test_utils import RedisTestFixture, requires_redis

# Use in tests
@requires_redis()
def test_my_feature():
    with RedisTestFixture() as redis:
        redis.set("test_key", "test_value")
        assert redis.get("test_key") == "test_value"
```

## Rust Usage

Rust components automatically load configuration from environment variables:

```rust
use crate::config::RedisConfig;

// Configuration is loaded from environment
let config = RedisConfig::default();
println!("Redis URL: {}", config.url);
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Windows: Check if Redis is running on port 6380
   - Docker: Ensure Redis service is healthy
   - Linux: Check if Redis is running on port 6379

2. **Port Conflicts**
   - Windows: Use port 6380 instead of 6379
   - Check for other services using the port: `netstat -an | grep 6379`

3. **Environment Detection Issues**
   - Manually set `DOCKER_CONTAINER=true` in Docker
   - Override auto-detection with explicit environment variables

### Debugging

Enable debug logging:

```bash
export RUST_LOG=debug
export REDIS_LOG_LEVEL=debug
```

Test connection manually:

```bash
# Test Redis connection
redis-cli -h localhost -p 6380 ping

# Check Redis info
redis-cli -h localhost -p 6380 info
```

### Fallback Behavior

When Redis is unavailable and fallback is enabled:
- Python: Uses in-memory cache (limited functionality)
- Rust: Returns errors but doesn't crash
- Tests: Skip Redis-dependent tests automatically

## Migration from Old Configuration

### Automatic Migration

Run the migration helper:

```bash
python scripts/redis-config-helper.py migrate
```

### Manual Migration

1. **Replace hardcoded values** with environment variables
2. **Update port references** from 6379 to 6380 (Windows)
3. **Use centralized configuration** instead of inline Redis settings
4. **Update test files** to use `redis_test_utils`

### Before (Old Configuration)

```python
import redis
client = redis.Redis(host='localhost', port=6379, db=0)
```

### After (New Configuration)

```python
from src.shared.config.redis_config import get_redis_client
client = get_redis_client()  # Auto-configured
```

## Production Considerations

### Security

- **Never hardcode credentials** in configuration files
- **Use environment variables** for sensitive information
- **Enable SSL/TLS** for production deployments
- **Configure authentication** (Redis 6.0+)

### Performance

- **Tune connection pool settings** based on load
- **Enable pipelining** for bulk operations
- **Monitor memory usage** and set appropriate limits
- **Configure persistence** (AOF/RDB) based on requirements

### Monitoring

The configuration system provides built-in health checks:

```python
# Check Redis health
config = get_redis_config()
is_healthy = config.test_connection()

# Get connection info
result = test_redis_connection(config)
print(f"Redis version: {result['info']['redis_version']}")
```

## Advanced Configuration

### SSL/TLS Configuration

```bash
REDIS_SSL_ENABLED=true
REDIS_SSL_CERT_FILE=/path/to/client.crt
REDIS_SSL_KEY_FILE=/path/to/client.key
REDIS_SSL_CA_CERTS=/path/to/ca.crt
```

### Cluster Configuration

```bash
REDIS_ENABLE_CLUSTER=true
REDIS_CLUSTER_NODES=node1:6379,node2:6379,node3:6379
```

### Sentinel Configuration

```bash
REDIS_SENTINEL_HOSTS=sentinel1:26379,sentinel2:26379
REDIS_SENTINEL_SERVICE=mymaster
```

## Integration with CI/CD

### GitHub Actions

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - 6379:6379
    options: >-
      --health-cmd "redis-cli ping"
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5

env:
  REDIS_HOST: localhost
  REDIS_PORT: 6379
  REDIS_DB: 1
```

### Local Testing

```bash
# Start Redis for testing
docker run -d --name test-redis -p 6380:6379 redis:7-alpine

# Run tests with custom Redis
REDIS_HOST=localhost REDIS_PORT=6380 pytest tests/

# Cleanup
docker stop test-redis && docker rm test-redis
```

This centralized configuration system ensures consistent Redis connectivity across all components while providing flexibility for different deployment environments.
