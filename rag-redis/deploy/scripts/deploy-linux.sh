#!/bin/bash
# RAG-Redis MCP Server Linux Deployment Script
# Supports Ubuntu, Debian, CentOS, RHEL, and other systemd-based distributions

set -euo pipefail

# Configuration variables
ENVIRONMENT="${ENVIRONMENT:-production}"
SERVICE_NAME="${SERVICE_NAME:-rag-redis-mcp-server}"
INSTALL_DIR="${INSTALL_DIR:-/opt/rag-redis}"
DATA_DIR="${DATA_DIR:-/var/lib/rag-redis}"
LOG_DIR="${LOG_DIR:-/var/log/rag-redis}"
CONFIG_DIR="${CONFIG_DIR:-/etc/rag-redis}"
USER_NAME="${USER_NAME:-rag-redis}"
GROUP_NAME="${GROUP_NAME:-rag-redis}"

# Deployment options
SKIP_BUILD="${SKIP_BUILD:-false}"
INSTALL_SERVICE="${INSTALL_SERVICE:-false}"
START_SERVICE="${START_SERVICE:-false}"
INSTALL_REDIS="${INSTALL_REDIS:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Detect Linux distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        log_error "Cannot detect Linux distribution"
        exit 1
    fi

    log_info "Detected distribution: $DISTRO $VERSION"
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."

    case $DISTRO in
        ubuntu|debian)
            apt-get update
            apt-get install -y curl wget git build-essential pkg-config libssl-dev
            if [[ "$INSTALL_REDIS" == "true" ]]; then
                apt-get install -y redis-server
            fi
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                dnf install -y curl wget git gcc openssl-devel pkg-config
                if [[ "$INSTALL_REDIS" == "true" ]]; then
                    dnf install -y redis
                fi
            else
                yum install -y curl wget git gcc openssl-devel pkg-config
                if [[ "$INSTALL_REDIS" == "true" ]]; then
                    yum install -y redis
                fi
            fi
            ;;
        *)
            log_warn "Unsupported distribution: $DISTRO"
            log_warn "Attempting to continue anyway..."
            ;;
    esac
}

# Install Rust if not present
install_rust() {
    if ! command -v cargo &> /dev/null; then
        log_info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    else
        log_info "Rust already installed"
    fi
}

# Create system user and group
create_user() {
    if ! getent group "$GROUP_NAME" > /dev/null; then
        log_info "Creating group: $GROUP_NAME"
        groupadd --system "$GROUP_NAME"
    fi

    if ! getent passwd "$USER_NAME" > /dev/null; then
        log_info "Creating user: $USER_NAME"
        useradd --system --gid "$GROUP_NAME" --home-dir "$DATA_DIR" \
                --shell /bin/false --comment "RAG-Redis MCP Server" "$USER_NAME"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."

    local directories=(
        "$INSTALL_DIR"
        "$INSTALL_DIR/bin"
        "$DATA_DIR"
        "$DATA_DIR/data"
        "$DATA_DIR/cache"
        "$DATA_DIR/models"
        "$LOG_DIR"
        "$CONFIG_DIR"
    )

    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_debug "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done

    # Set ownership
    chown -R "$USER_NAME:$GROUP_NAME" "$DATA_DIR" "$LOG_DIR"
    chown -R root:root "$INSTALL_DIR" "$CONFIG_DIR"
    chmod 755 "$INSTALL_DIR" "$CONFIG_DIR"
    chmod 750 "$DATA_DIR" "$LOG_DIR"
}

# Build the project
build_project() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warn "Skipping build (SKIP_BUILD=true)"
        return
    fi

    log_info "Building RAG-Redis MCP Server..."

    # Find project root
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root="$(cd "$script_dir/../.." && pwd)"

    cd "$project_root"

    # Build the project
    cargo build --release --bin mcp-server

    if [[ ! -f "target/release/mcp-server" ]]; then
        log_error "Build failed - binary not found"
        exit 1
    fi

    log_info "Build completed successfully"
}

# Copy application files
copy_application_files() {
    log_info "Copying application files..."

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root="$(cd "$script_dir/../.." && pwd)"
    local binary_path="$project_root/target/release/mcp-server"

    if [[ ! -f "$binary_path" ]]; then
        log_error "Binary not found at: $binary_path"
        log_error "Please build the project first or set SKIP_BUILD=false"
        exit 1
    fi

    # Copy binary
    cp "$binary_path" "$INSTALL_DIR/bin/"
    chmod +x "$INSTALL_DIR/bin/mcp-server"

    # Copy configuration files if they exist
    local config_source="$project_root/deploy/config/linux"
    if [[ -d "$config_source" ]]; then
        cp -r "$config_source"/* "$CONFIG_DIR/"
        log_info "Copied configuration files"
    fi

    # Create default environment file
    cat > "$CONFIG_DIR/.env" << EOF
# RAG-Redis MCP Server Environment Configuration
RUST_LOG=info
REDIS_URL=redis://127.0.0.1:6379
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8080
RAG_DATA_DIR=$DATA_DIR/data
EMBEDDING_CACHE_DIR=$DATA_DIR/cache
LOG_DIR=$LOG_DIR
VECTOR_BATCH_SIZE=100
MEMORY_CONSOLIDATION_INTERVAL=300
EMBEDDING_MODEL=all-MiniLM-L6-v2
EOF

    chown root:root "$CONFIG_DIR/.env"
    chmod 640 "$CONFIG_DIR/.env"

    log_info "Created environment configuration"
}

# Create systemd service file
create_systemd_service() {
    if [[ "$INSTALL_SERVICE" != "true" ]]; then
        log_warn "Skipping service installation (INSTALL_SERVICE!=true)"
        return
    fi

    log_info "Creating systemd service..."

    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=RAG-Redis MCP Server
Documentation=https://github.com/david-t-martel/llm-stats
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$DATA_DIR
ExecStart=$INSTALL_DIR/bin/mcp-server
EnvironmentFile=$CONFIG_DIR/.env
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# Security settings
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=$DATA_DIR $LOG_DIR

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"

    log_info "Systemd service created and enabled"
}

# Configure Redis if installed
configure_redis() {
    if [[ "$INSTALL_REDIS" != "true" ]]; then
        return
    fi

    log_info "Configuring Redis..."

    # Create Redis configuration directory if it doesn't exist
    mkdir -p /etc/redis

    # Backup original config if it exists
    if [[ -f /etc/redis/redis.conf ]]; then
        cp /etc/redis/redis.conf /etc/redis/redis.conf.backup.$(date +%Y%m%d_%H%M%S)
    fi

    # Configure Redis for RAG-Redis
    cat > /etc/redis/redis.conf << EOF
# Redis configuration for RAG-Redis MCP Server
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300

# Memory configuration
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Working directory
dir /var/lib/redis
EOF

    # Enable and start Redis
    systemctl enable redis-server 2>/dev/null || systemctl enable redis 2>/dev/null || true
    systemctl restart redis-server 2>/dev/null || systemctl restart redis 2>/dev/null || true

    log_info "Redis configured and started"
}

# Configure firewall
configure_firewall() {
    log_info "Configuring firewall..."

    # UFW (Ubuntu/Debian)
    if command -v ufw &> /dev/null; then
        ufw allow 8080/tcp comment "RAG-Redis MCP Server"
        log_info "UFW firewall rule added"
    fi

    # firewalld (CentOS/RHEL/Fedora)
    if command -v firewall-cmd &> /dev/null; then
        firewall-cmd --permanent --add-port=8080/tcp
        firewall-cmd --reload
        log_info "firewalld rule added"
    fi

    # iptables fallback
    if command -v iptables &> /dev/null; then
        iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
        log_info "iptables rule added (temporary)"
    fi
}

# Start the service
start_service() {
    if [[ "$START_SERVICE" != "true" ]]; then
        log_warn "Skipping service start (START_SERVICE!=true)"
        return
    fi

    log_info "Starting service: $SERVICE_NAME"
    systemctl start "$SERVICE_NAME"

    # Wait for service to start
    local timeout=30
    local timer=0

    while [[ $timer -lt $timeout ]]; do
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log_info "Service started successfully"
            return
        fi
        sleep 1
        ((timer++))
    done

    log_warn "Service did not start within $timeout seconds"
    systemctl status "$SERVICE_NAME" --no-pager
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Check binary
    if [[ -x "$INSTALL_DIR/bin/mcp-server" ]]; then
        log_info "Binary installed correctly"
    else
        log_error "Binary not found or not executable"
        return 1
    fi

    # Check service if installed
    if [[ "$INSTALL_SERVICE" == "true" ]]; then
        if systemctl is-enabled --quiet "$SERVICE_NAME"; then
            log_info "Service enabled correctly"
        else
            log_warn "Service not enabled"
        fi

        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log_info "Service is running"
        else
            log_warn "Service is not running"
        fi
    fi

    # Check directories
    local directories=("$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR" "$CONFIG_DIR")
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            log_debug "Directory exists: $dir"
        else
            log_error "Directory missing: $dir"
            return 1
        fi
    done

    log_info "Installation verification completed"
}

# Print deployment summary
print_summary() {
    log_info ""
    log_info "=== Deployment Summary ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Service Name: $SERVICE_NAME"
    log_info "Install Directory: $INSTALL_DIR"
    log_info "Data Directory: $DATA_DIR"
    log_info "Log Directory: $LOG_DIR"
    log_info "Config Directory: $CONFIG_DIR"
    log_info "User: $USER_NAME"
    log_info "Group: $GROUP_NAME"
    log_info ""
    log_info "Next steps:"
    log_info "1. Review configuration in $CONFIG_DIR/.env"
    log_info "2. Start/restart the service: sudo systemctl restart $SERVICE_NAME"
    log_info "3. Check service status: sudo systemctl status $SERVICE_NAME"
    log_info "4. View logs: sudo journalctl -u $SERVICE_NAME -f"
    log_info "5. Check Redis: redis-cli ping"
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [options]

Options:
    -e, --environment ENV       Deployment environment (default: production)
    -s, --service-name NAME     Service name (default: rag-redis-mcp-server)
    -i, --install-dir DIR       Installation directory (default: /opt/rag-redis)
    -d, --data-dir DIR          Data directory (default: /var/lib/rag-redis)
    -l, --log-dir DIR           Log directory (default: /var/log/rag-redis)
    -c, --config-dir DIR        Config directory (default: /etc/rag-redis)
    -u, --user USER             Service user (default: rag-redis)
    -g, --group GROUP           Service group (default: rag-redis)
    --skip-build                Skip building the project
    --install-service           Install systemd service
    --start-service             Start service after installation
    --install-redis             Install and configure Redis
    -h, --help                  Show this help message

Environment variables:
    ENVIRONMENT, SERVICE_NAME, INSTALL_DIR, DATA_DIR, LOG_DIR, CONFIG_DIR
    USER_NAME, GROUP_NAME, SKIP_BUILD, INSTALL_SERVICE, START_SERVICE, INSTALL_REDIS

Example:
    sudo $0 --install-service --start-service --install-redis
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--service-name)
                SERVICE_NAME="$2"
                shift 2
                ;;
            -i|--install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            -d|--data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            -l|--log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            -c|--config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            -u|--user)
                USER_NAME="$2"
                shift 2
                ;;
            -g|--group)
                GROUP_NAME="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --install-service)
                INSTALL_SERVICE="true"
                shift
                ;;
            --start-service)
                START_SERVICE="true"
                shift
                ;;
            --install-redis)
                INSTALL_REDIS="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Main deployment function
main() {
    parse_args "$@"

    log_info "=== RAG-Redis MCP Server Linux Deployment ==="
    log_info "Starting deployment process..."

    check_root
    detect_distro
    install_dependencies
    install_rust
    create_user
    create_directories
    build_project
    copy_application_files
    create_systemd_service
    configure_redis
    configure_firewall
    start_service
    verify_installation
    print_summary

    log_info "Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"