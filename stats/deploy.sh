#!/bin/bash

# Production Deployment Script for Gemma LLM Server
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Gemma LLM Production Deployment${NC}"
echo "======================================"

# Check prerequisites
check_requirements() {
    echo -e "\n${YELLOW}Checking requirements...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker found${NC}"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker Compose found${NC}"

    # Check .env file
    if [ ! -f .env ]; then
        echo -e "${YELLOW}⚠️ .env file not found, copying template...${NC}"
        cp .env.template .env
        echo -e "${YELLOW}Please edit .env with your configuration${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Environment configuration found${NC}"
}

# Create necessary directories
setup_directories() {
    echo -e "\n${YELLOW}Setting up directories...${NC}"

    mkdir -p nginx/ssl
    mkdir -p nginx/logs
    mkdir -p logs/{app1,app2,app3}
    mkdir -p prometheus
    mkdir -p grafana/{dashboards,datasources}
    mkdir -p loki
    mkdir -p promtail
    mkdir -p redis

    echo -e "${GREEN}✅ Directories created${NC}"
}

# Generate self-signed SSL certificates (for development)
generate_ssl() {
    echo -e "\n${YELLOW}Generating SSL certificates...${NC}"

    if [ ! -f nginx/ssl/cert.pem ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=gemma-api.local"
        echo -e "${GREEN}✅ SSL certificates generated${NC}"
    else
        echo -e "${GREEN}✅ SSL certificates already exist${NC}"
    fi
}

# Create monitoring configurations
setup_monitoring() {
    echo -e "\n${YELLOW}Setting up monitoring...${NC}"

    # Prometheus configuration
    cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gemma-apps'
    static_configs:
      - targets: ['app1:8000', 'app2:8000', 'app3:8000']
    metrics_path: '/metrics'

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:8080']
    metrics_path: '/nginx_status'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

    # Loki configuration
    cat > loki/loki-config.yml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
EOF

    # Promtail configuration
    cat > promtail/promtail-config.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: apps
    static_configs:
      - targets:
          - localhost
        labels:
          job: gemma-apps
          __path__: /var/log/apps/**/*.log
EOF

    echo -e "${GREEN}✅ Monitoring configured${NC}"
}

# Build and deploy
deploy() {
    echo -e "\n${YELLOW}Building and deploying...${NC}"

    # Pull latest changes
    if [ -d .git ]; then
        echo "Pulling latest changes..."
        git pull
    fi

    # Build images
    echo "Building Docker images..."
    docker-compose -f docker-compose.production.yml build

    # Start services
    echo "Starting services..."
    docker-compose -f docker-compose.production.yml up -d

    # Wait for health checks
    echo -e "\n${YELLOW}Waiting for services to be healthy...${NC}"
    sleep 10

    # Check service status
    docker-compose -f docker-compose.production.yml ps

    echo -e "\n${GREEN}✅ Deployment complete!${NC}"
}

# Show access information
show_info() {
    echo -e "\n${GREEN}📋 Access Information:${NC}"
    echo "======================================"
    echo "Main API: https://localhost (Nginx)"
    echo "Direct API: http://localhost:8000-8002 (Apps)"
    echo "WebSocket: wss://localhost/ws"
    echo "Redis: localhost:6379"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin)"
    echo "Loki: http://localhost:3100"
    echo ""
    echo "Health Check: https://localhost/health"
    echo "API Docs: https://localhost/docs"
    echo ""
    echo -e "${YELLOW}Add to /etc/hosts:${NC}"
    echo "127.0.0.1 gemma-api.local"
}

# Main execution
main() {
    case "${1:-deploy}" in
        check)
            check_requirements
            ;;
        setup)
            check_requirements
            setup_directories
            generate_ssl
            setup_monitoring
            ;;
        deploy)
            check_requirements
            setup_directories
            generate_ssl
            setup_monitoring
            deploy
            show_info
            ;;
        stop)
            echo -e "${YELLOW}Stopping services...${NC}"
            docker-compose -f docker-compose.production.yml down
            echo -e "${GREEN}✅ Services stopped${NC}"
            ;;
        restart)
            echo -e "${YELLOW}Restarting services...${NC}"
            docker-compose -f docker-compose.production.yml restart
            echo -e "${GREEN}✅ Services restarted${NC}"
            ;;
        logs)
            docker-compose -f docker-compose.production.yml logs -f ${2:-}
            ;;
        status)
            docker-compose -f docker-compose.production.yml ps
            ;;
        *)
            echo "Usage: $0 {check|setup|deploy|stop|restart|logs|status}"
            exit 1
            ;;
    esac
}

main "$@"
