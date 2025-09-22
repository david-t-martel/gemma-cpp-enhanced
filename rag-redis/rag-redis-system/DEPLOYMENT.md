# RAG-Redis System Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Strategies](#deployment-strategies)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Bare Metal Deployment](#bare-metal-deployment)
7. [Cloud Deployments](#cloud-deployments)
8. [Configuration Management](#configuration-management)
9. [Monitoring & Observability](#monitoring--observability)
10. [Security Hardening](#security-hardening)
11. [Performance Tuning](#performance-tuning)
12. [Backup & Recovery](#backup--recovery)
13. [Troubleshooting](#troubleshooting)
14. [Maintenance](#maintenance)

## Overview

This guide covers production deployment of the RAG-Redis System across various environments, from single-server deployments to distributed cloud architectures.

### Deployment Architecture Options

```
┌──────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│                  (Nginx/HAProxy/ALB)                     │
└─────────┬────────────────┬────────────────┬─────────────┘
          │                │                │
     ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
     │ RAG-1   │     │ RAG-2   │     │ RAG-3   │
     │ Server  │     │ Server  │     │ Server  │
     └────┬────┘     └────┬────┘     └────┬────┘
          │                │                │
          └────────────────┼────────────────┘
                          │
                ┌─────────▼─────────┐
                │   Redis Cluster    │
                │   (3+ nodes)       │
                └───────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU**: 4 cores (8+ recommended)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 100GB SSD (NVMe preferred)
- **Network**: 1Gbps (10Gbps for high throughput)
- **OS**: Ubuntu 22.04 LTS, RHEL 9, or similar

### Software Dependencies

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    redis-server \
    nginx \
    supervisor \
    prometheus-node-exporter \
    docker.io \
    docker-compose

# Install Rust (if building from source)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Network Requirements

```yaml
# Required ports
ports:
  - 8080  # RAG API
  - 6379  # Redis
  - 9090  # Metrics
  - 3000  # Grafana (optional)
  - 9093  # Alertmanager (optional)
```

## Deployment Strategies

### 1. Single Server Deployment

Best for: Development, testing, small-scale production (<1000 QPS)

```bash
# Simple deployment script
#!/bin/bash
./rag-redis-server \
    --config /etc/rag-redis/config.json \
    --port 8080 \
    --workers 4
```

### 2. High Availability Deployment

Best for: Production environments requiring 99.9%+ uptime

- Multiple RAG server instances
- Redis Sentinel or Cluster
- Load balancer with health checks
- Automatic failover

### 3. Microservices Deployment

Best for: Large-scale systems with specialized components

- Separate embedding service
- Dedicated vector index service
- Independent document processor
- API gateway

## Docker Deployment

### Dockerfile

```dockerfile
# Multi-stage build for optimal size
FROM rust:1.75 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release binary
RUN cargo build --release --features "full"

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/rag-redis-server /usr/local/bin/

# Copy configuration
COPY config/production.json /etc/rag-redis/config.json

# Create non-root user
RUN useradd -m -u 1000 raguser && \
    mkdir -p /var/lib/rag-redis && \
    chown -R raguser:raguser /var/lib/rag-redis

USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Run server
CMD ["rag-redis-server", "--config", "/etc/rag-redis/config.json"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  rag-redis:
    build: .
    image: rag-redis:latest
    container_name: rag-redis-server
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - RUST_LOG=info
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/etc/rag-redis
      - rag-data:/var/lib/rag-redis
    depends_on:
      - redis
    networks:
      - rag-network

  redis:
    image: redis:7-alpine
    container_name: rag-redis-db
    restart: unless-stopped
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - rag-network

  nginx:
    image: nginx:alpine
    container_name: rag-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/certs:/etc/nginx/certs
    depends_on:
      - rag-redis
    networks:
      - rag-network

volumes:
  rag-data:
  redis-data:

networks:
  rag-network:
    driver: bridge
```

### Building and Running

```bash
# Build the Docker image
docker build -t rag-redis:latest .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f rag-redis

# Scale horizontally
docker-compose up -d --scale rag-redis=3
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  config.json: |
    {
      "redis": {
        "url": "redis://redis-service:6379",
        "pool_size": 20
      },
      "vector_store": {
        "dimension": 768,
        "index_type": "hnsw"
      },
      "server": {
        "port": 8080,
        "workers": 4
      }
    }
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-redis-deployment
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-redis
  template:
    metadata:
      labels:
        app: rag-redis
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
      - name: rag-redis
        image: rag-redis:latest
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        volumeMounts:
        - name: config
          mountPath: /etc/rag-redis
        - name: data
          mountPath: /var/lib/rag-redis
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: rag-config
      - name: data
        persistentVolumeClaim:
          claimName: rag-data-pvc
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-redis-service
  namespace: rag-system
spec:
  selector:
    app: rag-redis
  ports:
  - name: api
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-redis-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-redis-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-redis-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.rag-system.com
    secretName: rag-tls-secret
  rules:
  - host: api.rag-system.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-redis-service
            port:
              number: 8080
```

### Helm Chart

```yaml
# values.yaml
replicaCount: 3

image:
  repository: rag-redis
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 8080

ingress:
  enabled: true
  hostname: api.rag-system.com
  tls: true

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 3

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

## Bare Metal Deployment

### System Setup

```bash
#!/bin/bash
# Production deployment script

# Create user and directories
sudo useradd -m -s /bin/bash raguser
sudo mkdir -p /opt/rag-redis/{bin,config,data,logs}
sudo chown -R raguser:raguser /opt/rag-redis

# Copy binary and configuration
sudo cp target/release/rag-redis-server /opt/rag-redis/bin/
sudo cp config/production.json /opt/rag-redis/config/

# Set up systemd service
sudo cat > /etc/systemd/system/rag-redis.service <<EOF
[Unit]
Description=RAG Redis System
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=raguser
Group=raguser
WorkingDirectory=/opt/rag-redis
ExecStart=/opt/rag-redis/bin/rag-redis-server --config /opt/rag-redis/config/production.json
Restart=always
RestartSec=10
StandardOutput=append:/opt/rag-redis/logs/stdout.log
StandardError=append:/opt/rag-redis/logs/stderr.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/rag-redis/data /opt/rag-redis/logs

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable rag-redis
sudo systemctl start rag-redis
```

### Nginx Configuration

```nginx
upstream rag_backend {
    least_conn;
    server 127.0.0.1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.rag-system.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.rag-system.com;

    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req zone=api_limit burst=200 nodelay;

    location / {
        proxy_pass http://rag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://rag_backend;
    }
}
```

## Cloud Deployments

### AWS Deployment

#### ECS Task Definition

```json
{
  "family": "rag-redis-task",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "rag-redis",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/rag-redis:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "REDIS_URL",
          "value": "redis://elasticache-cluster.abc123.cache.amazonaws.com:6379"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-redis",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### CloudFormation Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: RAG Redis System Infrastructure

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  # ElastiCache Redis Cluster
  RedisCluster:
    Type: AWS::ElastiCache::ReplicationGroup
    Properties:
      ReplicationGroupId: rag-redis-cluster
      ReplicationGroupDescription: Redis cluster for RAG system
      Engine: redis
      EngineVersion: 7.0
      CacheNodeType: cache.r6g.xlarge
      NumCacheClusters: 3
      AutomaticFailoverEnabled: true
      MultiAZEnabled: true
      SecurityGroupIds:
        - !Ref RedisSecurityGroup

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: rag-redis-alb
      Type: application
      Scheme: internet-facing
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: rag-redis-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  # Auto Scaling
  AutoScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      ServiceNamespace: ecs
      ScalableDimension: ecs:service:DesiredCount
      ResourceId: !Sub service/${ECSCluster}/rag-redis-service
      MinCapacity: 3
      MaxCapacity: 10

  AutoScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: rag-redis-scaling-policy
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref AutoScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ECSServiceAverageCPUUtilization
        TargetValue: 70.0
```

### GCP Deployment

#### Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rag-redis-service
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "3"
        autoscaling.knative.dev/maxScale: "100"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/rag-redis:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: redis://10.0.0.3:6379
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
```

### Azure Deployment

#### Azure Container Instances

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-09-01",
      "name": "rag-redis-container",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "rag-redis",
            "properties": {
              "image": "ragredis.azurecr.io/rag-redis:latest",
              "ports": [
                {
                  "port": 8080,
                  "protocol": "TCP"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              },
              "environmentVariables": [
                {
                  "name": "REDIS_URL",
                  "value": "redis://rag-redis.redis.cache.windows.net:6379"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "restartPolicy": "Always",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8080,
              "protocol": "TCP"
            }
          ]
        }
      }
    }
  ]
}
```

## Configuration Management

### Environment-Specific Configurations

```bash
# Directory structure
/opt/rag-redis/config/
├── base.json           # Shared configuration
├── development.json    # Development overrides
├── staging.json       # Staging overrides
└── production.json    # Production overrides
```

### Configuration Merging

```rust
// config_loader.rs
use serde_json::{Value, Map};

pub fn load_config(environment: &str) -> Result<Config> {
    let base = load_file("base.json")?;
    let env_specific = load_file(&format!("{}.json", environment))?;

    let merged = merge_configs(base, env_specific);

    // Override with environment variables
    let final_config = apply_env_overrides(merged)?;

    Ok(serde_json::from_value(final_config)?)
}
```

### Secret Management

```bash
# Using HashiCorp Vault
vault kv put secret/rag-redis \
    openai_api_key="sk-..." \
    redis_password="..." \
    jwt_secret="..."

# Using AWS Secrets Manager
aws secretsmanager create-secret \
    --name rag-redis-secrets \
    --secret-string '{"openai_api_key":"sk-..."}'

# Using Kubernetes Secrets
kubectl create secret generic rag-secrets \
    --from-literal=openai-api-key="sk-..." \
    --namespace=rag-system
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-redis'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RAG Redis System",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Search Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(search_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Vector Store Size",
        "targets": [
          {
            "expr": "vector_store_total_vectors"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```yaml
# Vector.toml for log aggregation
[sources.rag_logs]
type = "file"
include = ["/opt/rag-redis/logs/*.log"]

[transforms.parse_logs]
type = "remap"
inputs = ["rag_logs"]
source = '''
  . = parse_json!(.message)
  .timestamp = parse_timestamp!(.timestamp, "%Y-%m-%dT%H:%M:%S%.fZ")
'''

[sinks.elasticsearch]
type = "elasticsearch"
inputs = ["parse_logs"]
endpoint = "https://elasticsearch:9200"
index = "rag-redis-%Y.%m.%d"
```

## Security Hardening

### Network Security

```bash
# Firewall rules (iptables)
sudo iptables -A INPUT -p tcp --dport 8080 -m connlimit --connlimit-above 100 -j REJECT
sudo iptables -A INPUT -p tcp --dport 8080 -m state --state NEW -m recent --set
sudo iptables -A INPUT -p tcp --dport 8080 -m state --state NEW -m recent --update --seconds 60 --hitcount 100 -j DROP
```

### TLS Configuration

```rust
// TLS configuration
use rustls::{ServerConfig, Certificate, PrivateKey};

pub fn configure_tls() -> ServerConfig {
    let cert = load_certs("cert.pem");
    let key = load_private_key("key.pem");

    ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(cert, key)
        .expect("Invalid certificate/key")
}
```

### API Authentication

```rust
// JWT middleware
use jsonwebtoken::{decode, encode, Header, Validation};

pub async fn auth_middleware(
    req: Request<Body>,
    next: Next<Body>
) -> Result<Response<Body>> {
    let token = extract_token(&req)?;
    let claims = decode::<Claims>(
        &token,
        &DECODING_KEY,
        &Validation::default()
    )?;

    req.extensions_mut().insert(claims);
    Ok(next.run(req).await)
}
```

## Performance Tuning

### System Tuning

```bash
# /etc/sysctl.conf
# Network tuning
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30

# Memory tuning
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File descriptors
fs.file-max = 2097152
```

### Redis Tuning

```bash
# redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
tcp-backlog 511
tcp-keepalive 300
timeout 0

# Persistence
save 900 1
save 300 10
save 60 10000

# Replication
repl-diskless-sync yes
repl-diskless-sync-delay 5
```

### Application Tuning

```json
{
  "server": {
    "workers": 8,
    "max_connections": 10000,
    "keep_alive": 75,
    "request_timeout": 30
  },
  "redis": {
    "pool_size": 50,
    "connection_timeout": 5,
    "max_retries": 3
  },
  "vector_store": {
    "batch_size": 1000,
    "index_threads": 4,
    "search_threads": 8
  }
}
```

## Backup & Recovery

### Backup Strategy

```bash
#!/bin/bash
# Daily backup script

BACKUP_DIR="/backups/rag-redis/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup Redis data
redis-cli BGSAVE
sleep 10
cp /var/lib/redis/dump.rdb $BACKUP_DIR/

# Backup vector index
tar czf $BACKUP_DIR/vector-index.tar.gz /opt/rag-redis/data/

# Backup configuration
cp -r /opt/rag-redis/config $BACKUP_DIR/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://backups/rag-redis/$(date +%Y%m%d)/

# Clean old backups (keep 30 days)
find /backups/rag-redis -type d -mtime +30 -exec rm -rf {} \;
```

### Disaster Recovery

```bash
#!/bin/bash
# Recovery script

RESTORE_DATE=$1
BACKUP_DIR="/backups/rag-redis/$RESTORE_DATE"

# Stop services
systemctl stop rag-redis
systemctl stop redis

# Restore Redis
cp $BACKUP_DIR/dump.rdb /var/lib/redis/
chown redis:redis /var/lib/redis/dump.rdb

# Restore vector index
tar xzf $BACKUP_DIR/vector-index.tar.gz -C /

# Restore configuration
cp -r $BACKUP_DIR/config/* /opt/rag-redis/config/

# Start services
systemctl start redis
systemctl start rag-redis

# Verify
curl http://localhost:8080/health
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

```bash
# Check memory usage
free -h
ps aux | grep rag-redis

# Analyze heap
gdb -p $(pgrep rag-redis)
(gdb) heap
```

**Solution:**
- Adjust `max_vectors` in configuration
- Enable memory limits in systemd
- Implement cache eviction

#### 2. Slow Search Performance

```bash
# Profile the application
perf record -p $(pgrep rag-redis) -g
perf report

# Check index statistics
curl http://localhost:8080/stats | jq .vector_index
```

**Solution:**
- Optimize HNSW parameters
- Enable SIMD optimizations
- Add more search threads

#### 3. Redis Connection Issues

```bash
# Test Redis connectivity
redis-cli ping

# Check connection pool
netstat -an | grep 6379 | wc -l
```

**Solution:**
- Increase pool size
- Adjust timeout values
- Check network connectivity

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug,rag_redis_system=trace

# Run with debugging
RUST_BACKTRACE=full ./rag-redis-server

# Enable core dumps
ulimit -c unlimited
```

### Performance Profiling

```bash
# CPU profiling with flamegraph
cargo install flamegraph
flamegraph --bin rag-redis-server

# Memory profiling with Valgrind
valgrind --leak-check=full --show-leak-kinds=all \
    ./rag-redis-server

# Heap profiling with jemalloc
export MALLOC_CONF=prof:true,prof_prefix:jeprof.out
./rag-redis-server
jeprof --show_bytes ./rag-redis-server jeprof.out.*
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Compact vector index
curl -X POST http://localhost:8080/admin/compact

# Optimize Redis
redis-cli MEMORY PURGE

# Clean old logs
find /opt/rag-redis/logs -type f -mtime +30 -delete

# Update dependencies
cd /opt/rag-redis
cargo update
cargo audit

# Rebuild if needed
cargo build --release
```

### Health Checks

```bash
#!/bin/bash
# Health check script

# Check service status
systemctl is-active rag-redis || exit 1

# Check API health
curl -f http://localhost:8080/health || exit 1

# Check Redis
redis-cli ping || exit 1

# Check disk space
df -h | grep -E "/$|/opt" | awk '{if ($5+0 > 80) exit 1}'

# Check memory
free -m | awk 'NR==2{if ($3/$2 > 0.9) exit 1}'
```

### Version Upgrades

```bash
#!/bin/bash
# Zero-downtime upgrade

# Build new version
cd /tmp/rag-redis-new
cargo build --release

# Start new instance on different port
./target/release/rag-redis-server --port 8081 &
NEW_PID=$!

# Wait for health check
sleep 10
curl -f http://localhost:8081/health || exit 1

# Update load balancer
# ... update nginx/haproxy config ...

# Stop old instance
systemctl stop rag-redis

# Move new binary
mv /tmp/rag-redis-new/target/release/rag-redis-server \
   /opt/rag-redis/bin/

# Start with systemd
systemctl start rag-redis
```

## Conclusion

This deployment guide covers comprehensive production deployment strategies for the RAG-Redis System. Key considerations:

1. **High Availability**: Always deploy multiple instances with load balancing
2. **Monitoring**: Implement comprehensive monitoring and alerting
3. **Security**: Follow security best practices and regular updates
4. **Performance**: Tune system based on workload characteristics
5. **Backup**: Implement regular backup and tested recovery procedures
6. **Maintenance**: Schedule regular maintenance windows

For additional support and updates, refer to the project documentation and community resources.
