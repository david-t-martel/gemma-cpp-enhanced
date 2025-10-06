# RAG-Redis MCP Server Deployment Guide

This guide provides comprehensive instructions for deploying the RAG-Redis MCP Server across multiple platforms and environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Methods](#deployment-methods)
- [Docker Deployment](#docker-deployment)
- [Linux Deployment](#linux-deployment)
- [Windows Deployment](#windows-deployment)
- [Environment Configuration](#environment-configuration)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Maintenance](#maintenance)

## Overview

The RAG-Redis MCP Server is a high-performance Retrieval-Augmented Generation system built in Rust with Redis backend. It provides MCP (Model Context Protocol) interface for AI assistants with advanced vector search, multi-tier memory management, and document processing capabilities.

### Architecture Components

- **MCP Server**: Rust-based server implementing MCP protocol
- **Redis Backend**: Vector storage and memory management
- **Multi-tier Memory**: Working, Short-term, Long-term, Episodic, and Semantic memory tiers
- **Vector Store**: SIMD-optimized similarity search
- **Document Processing**: PDF, text, and markdown ingestion

### Supported Platforms

- **Docker**: Containerized deployment with Redis
- **Linux**: Native systemd service (Ubuntu, Debian, CentOS, RHEL, Fedora)
- **Windows**: Native Windows service
- **WSL**: Windows Subsystem for Linux

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 2GB
- Storage: 5GB free space
- OS: Linux (kernel 3.10+), Windows 10+, or macOS 10.15+

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- Network: 1Gbps

### Software Dependencies

**Required:**
- Rust 1.75.0+ (for building from source)
- Redis 6.0+ (if not using Docker)

**Optional:**
- Docker 20.10+ and Docker Compose 2.0+
- Prometheus and Grafana (for monitoring)
- NSSM (for Windows service management)

### Network Requirements

**Ports:**
- 8080: MCP Server (configurable)
- 6379: Redis (configurable)
- 9090: Prometheus metrics (optional)
- 3000: Grafana dashboard (optional)

## Deployment Methods

### Quick Start Matrix

| Method | Best For | Setup Time | Scalability |
|--------|----------|------------|-------------|
| Docker Compose | Development, Testing | 5 minutes | Medium |
| Linux Service | Production Linux | 15 minutes | High |
| Windows Service | Production Windows | 15 minutes | High |
| Kubernetes | Enterprise, Cloud | 30 minutes | Very High |

## Docker Deployment

### Docker Compose (Recommended)

1. **Clone and prepare:**
   ```bash
   git clone <repository-url>
   cd rag-redis
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Deploy with monitoring:**
   ```bash
   docker-compose --profile monitoring up -d
   ```

4. **Deploy basic setup:**
   ```bash
   docker-compose up -d
   ```

5. **Verify deployment:**
   ```bash
   docker-compose ps
   docker-compose logs rag-mcp-server
   ```

### Docker Compose Profiles

- **Default**: MCP Server + Redis
- **monitoring**: Adds Prometheus + Grafana
- **tools**: Adds Redis Insight
- **development**: Development build with hot reload

### Environment Variables

Create `.env` file in project root:

```bash
# Required
REDIS_PORT=6379
MCP_SERVER_PORT=8080

# Optional API Keys
ANTHROPIC_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here

# Performance
VECTOR_BATCH_SIZE=100
MEMORY_CONSOLIDATION_INTERVAL=300

# Monitoring (with --profile monitoring)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=secure_password
```

### Docker Commands

```bash
# View logs
docker-compose logs -f rag-mcp-server

# Scale services
docker-compose up -d --scale rag-mcp-server=3

# Update and restart
docker-compose pull && docker-compose up -d

# Health check
docker-compose exec rag-mcp-server /app/bin/health-check

# Enter container
docker-compose exec rag-mcp-server /bin/bash

# Backup data
docker-compose exec redis redis-cli --rdb /data/backup.rdb

# Remove deployment
docker-compose down -v
```

## Linux Deployment

### Automated Installation

**Ubuntu/Debian:**
```bash
# Download and run deployment script
curl -fsSL https://raw.githubusercontent.com/your-org/rag-redis/main/deploy/scripts/deploy-linux.sh | sudo bash -s -- --install-service --start-service --install-redis
```

**Manual Installation:**
```bash
# Make script executable
chmod +x deploy/scripts/deploy-linux.sh

# Run with options
sudo ./deploy/scripts/deploy-linux.sh \
  --install-service \
  --start-service \
  --install-redis \
  --environment production
```

### Installation Options

```bash
# Full production deployment
sudo ./deploy/scripts/deploy-linux.sh \
  --environment production \
  --install-service \
  --start-service \
  --install-redis

# Development deployment
sudo ./deploy/scripts/deploy-linux.sh \
  --environment development \
  --skip-build \
  --user developer

# Custom paths
sudo ./deploy/scripts/deploy-linux.sh \
  --install-dir /opt/custom/rag-redis \
  --data-dir /var/custom/rag-redis \
  --config-dir /etc/custom/rag-redis
```

### Manual Steps

1. **Install dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install -y curl git build-essential pkg-config libssl-dev redis-server

   # CentOS/RHEL/Fedora
   sudo yum install -y curl git gcc openssl-devel pkg-config redis
   ```

2. **Install Rust:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

3. **Build and install:**
   ```bash
   cargo build --release --bin mcp-server
   sudo cp target/release/mcp-server /opt/rag-redis/bin/
   ```

4. **Create user and directories:**
   ```bash
   sudo useradd --system --shell /bin/false rag-redis
   sudo mkdir -p /opt/rag-redis/{bin,config}
   sudo mkdir -p /var/lib/rag-redis/{data,cache}
   sudo mkdir -p /var/log/rag-redis
   ```

5. **Install systemd service:**
   ```bash
   sudo cp deploy/systemd/rag-redis-mcp-server.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable rag-redis-mcp-server
   sudo systemctl start rag-redis-mcp-server
   ```

### Systemd Service Management

```bash
# Service control
sudo systemctl start rag-redis-mcp-server
sudo systemctl stop rag-redis-mcp-server
sudo systemctl restart rag-redis-mcp-server
sudo systemctl reload rag-redis-mcp-server

# Status and logs
sudo systemctl status rag-redis-mcp-server
sudo journalctl -u rag-redis-mcp-server -f
sudo journalctl -u rag-redis-mcp-server --since "1 hour ago"

# Enable/disable auto-start
sudo systemctl enable rag-redis-mcp-server
sudo systemctl disable rag-redis-mcp-server
```

## Windows Deployment

### Automated PowerShell Installation

**Run as Administrator:**
```powershell
# Download and run deployment script
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/your-org/rag-redis/main/deploy/scripts/deploy-windows.ps1'))

# Or with parameters
.\deploy\scripts\deploy-windows.ps1 -InstallService -StartService -Environment production
```

### Manual Installation

1. **Prerequisites:**
   ```powershell
   # Install Rust
   Invoke-WebRequest -Uri "https://win.rustup.rs/" -OutFile "rustup-init.exe"
   .\rustup-init.exe -y

   # Install Redis (via Chocolatey)
   choco install redis-64
   ```

2. **Build application:**
   ```powershell
   cargo build --release --bin mcp-server
   ```

3. **Create directories:**
   ```powershell
   New-Item -ItemType Directory -Path "C:\Program Files\RAG-Redis\bin" -Force
   New-Item -ItemType Directory -Path "C:\ProgramData\RAG-Redis\{data,cache,logs}" -Force
   ```

4. **Install as Windows service:**
   ```powershell
   # Copy binary
   Copy-Item "target\release\mcp-server.exe" "C:\Program Files\RAG-Redis\bin\"

   # Install service
   .\deploy\windows\install-service.ps1 -InstallService -StartService
   ```

### Windows Service Management

```powershell
# Service control
Start-Service "RagRedisMcpServer"
Stop-Service "RagRedisMcpServer"
Restart-Service "RagRedisMcpServer"

# Service status
Get-Service "RagRedisMcpServer"
Get-EventLog -LogName Application -Source "RagRedisMcpServer" -Newest 10

# Using sc.exe
sc.exe query RagRedisMcpServer
sc.exe start RagRedisMcpServer
sc.exe stop RagRedisMcpServer
```

### NSSM (Advanced Service Management)

```powershell
# Install NSSM
choco install nssm

# Configure service with NSSM
nssm install RagRedisMcpServer "C:\Program Files\RAG-Redis\bin\mcp-server.exe"
nssm set RagRedisMcpServer AppDirectory "C:\ProgramData\RAG-Redis"
nssm set RagRedisMcpServer DisplayName "RAG-Redis MCP Server"
nssm set RagRedisMcpServer Description "High-performance RAG system with Redis backend"

# Advanced configuration
nssm set RagRedisMcpServer AppStdout "C:\ProgramData\RAG-Redis\logs\service.log"
nssm set RagRedisMcpServer AppStderr "C:\ProgramData\RAG-Redis\logs\error.log"
nssm set RagRedisMcpServer AppRotateFiles 1
nssm set RagRedisMcpServer AppRotateBytes 10485760

# Start service
nssm start RagRedisMcpServer
```

## Environment Configuration

### Environment Management

The deployment supports three environments with different configurations:

1. **Development**: Debug logging, relaxed security, smaller memory tiers
2. **Staging**: Production-like settings with enhanced logging
3. **Production**: Optimized performance, strict security, full scale

### Environment Setup

**Linux:**
```bash
# Set up production environment
sudo ./deploy/scripts/setup-environment.sh setup production

# Switch between environments
sudo ./deploy/scripts/setup-environment.sh switch development

# Validate configuration
sudo ./deploy/scripts/setup-environment.sh validate production

# Compare environments
sudo ./deploy/scripts/setup-environment.sh diff production staging
```

**Manual Configuration:**
```bash
# Copy environment template
cp deploy/config/environments/production.env /etc/rag-redis/.env

# Edit configuration
sudo nano /etc/rag-redis/.env
```

### Key Configuration Variables

```bash
# Server Configuration
MCP_SERVER_HOST=0.0.0.0          # Bind address
MCP_SERVER_PORT=8080             # Server port
RUST_LOG=info                    # Log level (debug, info, warn, error)

# Redis Configuration
REDIS_URL=redis://127.0.0.1:6379 # Redis connection
REDIS_MAX_CONNECTIONS=20         # Connection pool size

# Data Directories
RAG_DATA_DIR=/var/lib/rag-redis/data
EMBEDDING_CACHE_DIR=/var/lib/rag-redis/cache
LOG_DIR=/var/log/rag-redis

# Performance Tuning
VECTOR_BATCH_SIZE=100            # Vector processing batch size
MEMORY_CONSOLIDATION_INTERVAL=300 # Memory cleanup interval (seconds)
WORKER_THREADS=4                 # Tokio worker threads
BLOCKING_THREADS=8               # Blocking thread pool size

# Memory Tiers (Production Scale)
WORKING_MEMORY_SIZE=100          # Working memory items
SHORT_TERM_MEMORY_SIZE=1000      # Short-term memory items
LONG_TERM_MEMORY_SIZE=10000      # Long-term memory items
EPISODIC_MEMORY_SIZE=5000        # Episodic memory items
SEMANTIC_MEMORY_SIZE=50000       # Semantic memory items

# API Keys (Optional)
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Security
MAX_REQUEST_SIZE=10485760        # 10MB max request
CONNECTION_IDLE_TIMEOUT=300      # Connection timeout
RATE_LIMIT_REQUESTS=100          # Requests per window
RATE_LIMIT_WINDOW=60             # Rate limit window (seconds)
```

### Redis Configuration

**Production Redis Configuration:**
```bash
# Copy Redis config
sudo cp deploy/config/redis/redis.conf /etc/redis/redis.conf

# Key settings for RAG workloads
maxmemory 2gb                    # Memory limit
maxmemory-policy allkeys-lru     # Eviction policy
save 900 1 300 10 60 10000      # Persistence
appendonly yes                   # AOF enabled
```

## Monitoring and Health Checks

### Built-in Health Checks

**Manual Health Check:**
```bash
# Linux
./deploy/scripts/health-check.sh

# With custom parameters
./deploy/scripts/health-check.sh \
  --url http://localhost:8080/health \
  --redis-url redis://localhost:6379 \
  --verbose

# Docker
docker-compose exec rag-mcp-server /app/bin/health-check
```

**Health Check Features:**
- MCP server response validation
- Redis connectivity testing
- Vector store functionality verification
- Memory tier status checking
- System resource monitoring
- Log analysis for errors

### Prometheus Metrics

**Available Metrics:**
- `rag_redis_requests_total`: Total HTTP requests
- `rag_redis_request_duration_seconds`: Request latency
- `rag_redis_memory_tier_size`: Memory tier item counts
- `rag_redis_vector_operations_total`: Vector store operations
- `rag_redis_redis_operations_total`: Redis operations
- `rag_redis_errors_total`: Error counts by type

**Prometheus Configuration:**
```yaml
# Add to prometheus.yml
scrape_configs:
  - job_name: 'rag-redis-mcp-server'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard

**Pre-configured Dashboards:**
- System Overview: CPU, memory, disk usage
- Application Metrics: Request rates, latency, errors
- Redis Metrics: Memory usage, operations, connections
- Memory Tiers: Tier sizes, consolidation rates
- Vector Store: Search operations, indexing performance

**Import Dashboard:**
1. Access Grafana at http://localhost:3000
2. Login with admin/admin (change password)
3. Import dashboard from `deploy/monitoring/grafana/dashboards/`

### Alerting Rules

**Critical Alerts:**
- Service down for >1 minute
- Redis disconnection
- High error rate (>5%)
- Memory usage >90%
- Disk space <5%

**Warning Alerts:**
- High latency (>2s 95th percentile)
- Memory tier overflow
- Vector indexing failures
- Redis connection pool exhaustion

### Log Management

**Log Locations:**
- **Linux**: `/var/log/rag-redis/`
- **Windows**: `C:\ProgramData\RAG-Redis\logs\`
- **Docker**: Use `docker-compose logs`

**Log Rotation:**
```bash
# Linux logrotate configuration
sudo cat > /etc/logrotate.d/rag-redis << EOF
/var/log/rag-redis/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        systemctl reload rag-redis-mcp-server > /dev/null 2>&1 || true
    endscript
}
EOF
```

**Centralized Logging:**
```bash
# Ship logs to ELK stack
filebeat.inputs:
- type: log
  paths:
    - /var/log/rag-redis/*.log
  fields:
    service: rag-redis-mcp-server
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptoms:**
- Service fails to start
- Connection refused errors
- Port already in use

**Solutions:**
```bash
# Check port usage
sudo netstat -tulpn | grep :8080
sudo lsof -i :8080

# Check configuration
sudo ./deploy/scripts/setup-environment.sh validate production

# Check logs
sudo journalctl -u rag-redis-mcp-server --since "10 minutes ago"

# Verify binary
/opt/rag-redis/bin/mcp-server --help
```

#### 2. Redis Connection Issues

**Symptoms:**
- Redis connection timeouts
- Memory operations failing
- Inconsistent behavior

**Solutions:**
```bash
# Test Redis connectivity
redis-cli ping
redis-cli -u redis://localhost:6379 ping

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log

# Verify Redis configuration
redis-cli config get maxmemory
redis-cli info memory

# Restart Redis
sudo systemctl restart redis
```

#### 3. High Memory Usage

**Symptoms:**
- System memory exhaustion
- OOM killer activating
- Slow performance

**Solutions:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Adjust memory limits
# Edit /etc/rag-redis/.env
WORKING_MEMORY_SIZE=50
SHORT_TERM_MEMORY_SIZE=500

# Restart service
sudo systemctl restart rag-redis-mcp-server
```

#### 4. Vector Search Performance

**Symptoms:**
- Slow search responses
- High CPU usage during searches
- Search timeouts

**Solutions:**
```bash
# Check vector store metrics
curl http://localhost:8080/metrics | grep vector

# Optimize batch sizes
VECTOR_BATCH_SIZE=50

# Check SIMD support
cat /proc/cpuinfo | grep -E "(sse|avx)"

# Monitor search operations
curl http://localhost:8080/api/health
```

### Debug Mode

**Enable Debug Logging:**
```bash
# Temporary (until restart)
sudo systemctl edit rag-redis-mcp-server --runtime
# Add: Environment=RUST_LOG=debug

# Permanent
sudo nano /etc/rag-redis/.env
# Set: RUST_LOG=debug

sudo systemctl restart rag-redis-mcp-server
```

**Debug Tools:**
```bash
# Live log monitoring
sudo journalctl -u rag-redis-mcp-server -f

# Performance profiling
sudo perf record -g /opt/rag-redis/bin/mcp-server
sudo perf report

# Memory analysis
sudo valgrind --tool=memcheck /opt/rag-redis/bin/mcp-server
```

### Performance Tuning

**CPU Optimization:**
```bash
# Set CPU affinity
sudo systemctl edit rag-redis-mcp-server
# Add: ExecStart=/usr/bin/taskset -c 0,1,2,3 /opt/rag-redis/bin/mcp-server

# Adjust thread pools
WORKER_THREADS=8
BLOCKING_THREADS=16
```

**Memory Optimization:**
```bash
# Tune memory allocator
export MALLOC_CONF="dirty_decay_ms:1000,muzzy_decay_ms:1000"

# Optimize Redis memory
redis-cli config set maxmemory-policy allkeys-lru
redis-cli config set hash-max-ziplist-entries 512
```

**Network Optimization:**
```bash
# Increase connection limits
echo 'net.core.somaxconn = 4096' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 4096' >> /etc/sysctl.conf
sysctl -p
```

## Security Considerations

### Network Security

**Firewall Configuration:**
```bash
# UFW (Ubuntu/Debian)
sudo ufw allow 8080/tcp comment "RAG-Redis MCP Server"
sudo ufw deny 6379/tcp comment "Redis - internal only"

# firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --remove-port=6379/tcp
sudo firewall-cmd --reload

# iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 6379 -s 127.0.0.1 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 6379 -j DROP
```

**TLS Configuration:**
```bash
# Generate certificates
sudo openssl req -x509 -newkey rsa:4096 -keyout /etc/rag-redis/ssl/key.pem -out /etc/rag-redis/ssl/cert.pem -days 365 -nodes

# Configure TLS in environment
TLS_CERT_PATH=/etc/rag-redis/ssl/cert.pem
TLS_KEY_PATH=/etc/rag-redis/ssl/key.pem
ENABLE_TLS=true
```

### Application Security

**API Key Management:**
```bash
# Use environment-specific keys
export ANTHROPIC_API_KEY_PROD="sk-ant-api03-prod-key"
export ANTHROPIC_API_KEY_STAGING="sk-ant-api03-staging-key"

# Rotate keys regularly (90 days recommended)
```

**Access Control:**
```bash
# Create dedicated user
sudo useradd --system --shell /bin/false --home /var/lib/rag-redis rag-redis

# Set restrictive permissions
sudo chmod 700 /var/lib/rag-redis
sudo chmod 640 /etc/rag-redis/.env
sudo chown rag-redis:rag-redis /var/lib/rag-redis
```

**Redis Security:**
```bash
# Enable Redis AUTH
redis-cli config set requirepass "your-secure-password"

# Disable dangerous commands
redis-cli config set rename-command FLUSHDB ""
redis-cli config set rename-command FLUSHALL ""
redis-cli config set rename-command DEBUG ""

# Update Redis URL
REDIS_URL=redis://:your-secure-password@127.0.0.1:6379
```

### System Hardening

**Systemd Security Features:**
```ini
# In /etc/systemd/system/rag-redis-mcp-server.service
[Service]
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/rag-redis /var/log/rag-redis
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictRealtime=true
RestrictNamespaces=true
LockPersonality=true
MemoryDenyWriteExecute=true
SystemCallFilter=@system-service
SystemCallFilter=~@debug @mount @cpu-emulation @obsolete
```

## Maintenance

### Backup and Recovery

**Database Backup:**
```bash
# Redis snapshot backup
redis-cli --rdb /backup/redis/dump-$(date +%Y%m%d).rdb

# AOF backup
cp /var/lib/redis/appendonly.aof /backup/redis/aof-$(date +%Y%m%d).aof

# Automated backup script
sudo cat > /etc/cron.daily/rag-redis-backup << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/rag-redis/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Redis backup
redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# Configuration backup
cp -r /etc/rag-redis "$BACKUP_DIR/"

# Data directory backup
tar -czf "$BACKUP_DIR/data.tar.gz" /var/lib/rag-redis/

# Clean old backups (keep 30 days)
find /backup/rag-redis -type d -mtime +30 -exec rm -rf {} +
EOF

sudo chmod +x /etc/cron.daily/rag-redis-backup
```

**Configuration Backup:**
```bash
# Backup current configuration
sudo ./deploy/scripts/setup-environment.sh backup production

# Restore from backup
sudo ./deploy/scripts/setup-environment.sh restore production /etc/rag-redis/backup/production_20241122_143000.env
```

### Updates and Upgrades

**Application Updates:**
```bash
# 1. Stop service
sudo systemctl stop rag-redis-mcp-server

# 2. Backup current installation
sudo cp /opt/rag-redis/bin/mcp-server /opt/rag-redis/bin/mcp-server.backup

# 3. Build new version
git pull origin main
cargo build --release --bin mcp-server

# 4. Install new binary
sudo cp target/release/mcp-server /opt/rag-redis/bin/

# 5. Update configuration if needed
sudo ./deploy/scripts/setup-environment.sh validate production

# 6. Start service
sudo systemctl start rag-redis-mcp-server

# 7. Verify deployment
sudo ./deploy/scripts/health-check.sh
```

**System Updates:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y

# Restart if kernel updated
sudo reboot
```

### Monitoring Maintenance

**Log Cleanup:**
```bash
# Clean old logs (manual)
sudo find /var/log/rag-redis -name "*.log" -mtime +30 -delete

# Clear journal logs
sudo journalctl --vacuum-time=30d
sudo journalctl --vacuum-size=1G
```

**Database Maintenance:**
```bash
# Redis memory optimization
redis-cli memory purge

# Compact Redis database
redis-cli debug restart

# Check Redis consistency
redis-cli debug check
```

### Performance Monitoring

**Regular Health Checks:**
```bash
# Add to crontab (every 5 minutes)
*/5 * * * * /opt/rag-redis/scripts/health-check.sh >> /var/log/rag-redis/health-check.log 2>&1
```

**Performance Baselines:**
```bash
# Establish performance baselines
./deploy/scripts/health-check.sh --verbose > /var/log/rag-redis/baseline-$(date +%Y%m%d).log

# Compare current performance
./deploy/scripts/health-check.sh --verbose | diff /var/log/rag-redis/baseline-20241122.log -
```

---

## Support and Resources

### Documentation
- [API Documentation](./docs/api.md)
- [Configuration Reference](./docs/configuration.md)
- [Development Guide](./docs/development.md)

### Community
- [GitHub Issues](https://github.com/your-org/rag-redis/issues)
- [Discussions](https://github.com/your-org/rag-redis/discussions)
- [Discord Community](https://discord.gg/your-invite)

### Professional Support
- [Enterprise Support](mailto:enterprise@your-org.com)
- [Consulting Services](https://your-org.com/consulting)
- [Training Programs](https://your-org.com/training)

---

**Last Updated:** 2024-11-22
**Version:** 1.0.0
**Deployment Guide Version:** 1.0