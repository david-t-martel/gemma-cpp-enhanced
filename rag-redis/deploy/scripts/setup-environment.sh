#!/bin/bash
# Environment Configuration Management Script for RAG-Redis MCP Server
# Manages environment-specific configurations across development, staging, and production

set -euo pipefail

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-development}"
CONFIG_DIR="${CONFIG_DIR:-/etc/rag-redis}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Print usage information
usage() {
    cat << EOF
Usage: $0 [options] <command>

Commands:
    setup ENV          Set up environment configuration for ENV (development, staging, production)
    validate ENV       Validate environment configuration for ENV
    switch ENV         Switch active environment to ENV
    backup ENV         Backup current environment configuration
    restore ENV FILE   Restore environment configuration from backup
    list               List available environments
    diff ENV1 ENV2     Compare two environment configurations
    update KEY VALUE   Update a configuration value in current environment

Options:
    -c, --config-dir DIR    Configuration directory (default: /etc/rag-redis)
    -d, --dry-run          Show what would be done without making changes
    -v, --verbose          Verbose output
    -h, --help             Show this help message

Environment variables:
    ENVIRONMENT            Target environment (development, staging, production)
    CONFIG_DIR             Configuration directory path

Examples:
    $0 setup production
    $0 validate staging
    $0 switch development
    $0 update REDIS_URL redis://localhost:6380
    $0 diff production staging
EOF
}

# Validate environment name
validate_environment() {
    local env="$1"
    case "$env" in
        development|staging|production)
            return 0
            ;;
        *)
            log_error "Invalid environment: $env"
            log_error "Valid environments: development, staging, production"
            return 1
            ;;
    esac
}

# Create configuration directory structure
create_config_structure() {
    local config_dir="$1"

    log_info "Creating configuration directory structure..."

    local directories=(
        "$config_dir"
        "$config_dir/environments"
        "$config_dir/redis"
        "$config_dir/ssl"
        "$config_dir/backup"
    )

    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_debug "Created directory: $dir"
        fi
    done

    # Set appropriate permissions
    chmod 755 "$config_dir"
    chmod 700 "$config_dir/ssl" "$config_dir/backup"
    chmod 755 "$config_dir/environments" "$config_dir/redis"
}

# Copy environment configurations
copy_environment_configs() {
    local config_dir="$1"
    local source_dir="$PROJECT_ROOT/deploy/config/environments"

    log_info "Copying environment configurations..."

    if [[ -d "$source_dir" ]]; then
        cp "$source_dir"/*.env "$config_dir/environments/"
        chmod 640 "$config_dir/environments"/*.env
        log_info "Environment configurations copied"
    else
        log_warn "Source environment directory not found: $source_dir"
    fi
}

# Copy Redis configuration
copy_redis_config() {
    local config_dir="$1"
    local source_file="$PROJECT_ROOT/deploy/config/redis/redis.conf"

    log_info "Copying Redis configuration..."

    if [[ -f "$source_file" ]]; then
        cp "$source_file" "$config_dir/redis/"
        chmod 644 "$config_dir/redis/redis.conf"
        log_info "Redis configuration copied"
    else
        log_warn "Source Redis config not found: $source_file"
    fi
}

# Set up environment
setup_environment() {
    local env="$1"
    local config_dir="$2"

    validate_environment "$env"

    log_info "Setting up environment: $env"

    # Create directory structure
    create_config_structure "$config_dir"

    # Copy configurations
    copy_environment_configs "$config_dir"
    copy_redis_config "$config_dir"

    # Create active environment symlink
    local env_file="$config_dir/environments/$env.env"
    local active_link="$config_dir/.env"

    if [[ -f "$env_file" ]]; then
        ln -sf "$env_file" "$active_link"
        log_info "Active environment set to: $env"
    else
        log_error "Environment file not found: $env_file"
        return 1
    fi

    log_info "Environment setup completed"
}

# Validate environment configuration
validate_configuration() {
    local env="$1"
    local config_dir="$2"

    validate_environment "$env"

    log_info "Validating environment: $env"

    local env_file="$config_dir/environments/$env.env"

    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi

    # Check required variables
    local required_vars=(
        "RUST_LOG"
        "MCP_SERVER_HOST"
        "MCP_SERVER_PORT"
        "REDIS_URL"
        "RAG_DATA_DIR"
        "EMBEDDING_CACHE_DIR"
        "LOG_DIR"
    )

    local missing_vars=()

    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var=" "$env_file"; then
            missing_vars+=("$var")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required variables in $env environment:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        return 1
    fi

    # Validate Redis URL format
    local redis_url=$(grep "^REDIS_URL=" "$env_file" | cut -d'=' -f2)
    if [[ ! "$redis_url" =~ ^redis://[^:]+:[0-9]+$ ]]; then
        log_warn "Invalid Redis URL format: $redis_url"
    fi

    # Check directory paths
    local data_dir=$(grep "^RAG_DATA_DIR=" "$env_file" | cut -d'=' -f2)
    local cache_dir=$(grep "^EMBEDDING_CACHE_DIR=" "$env_file" | cut -d'=' -f2)
    local log_dir=$(grep "^LOG_DIR=" "$env_file" | cut -d'=' -f2)

    for dir in "$data_dir" "$cache_dir" "$log_dir"; do
        if [[ -n "$dir" && ! -d "$dir" ]]; then
            log_warn "Directory does not exist: $dir"
        fi
    done

    log_info "Environment validation completed"
}

# Switch active environment
switch_environment() {
    local env="$1"
    local config_dir="$2"

    validate_environment "$env"

    local env_file="$config_dir/environments/$env.env"
    local active_link="$config_dir/.env"

    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi

    # Backup current environment if it exists
    if [[ -L "$active_link" ]]; then
        local current_env=$(readlink "$active_link" | sed 's|.*/||' | sed 's|\.env$||')
        log_info "Switching from $current_env to $env"
    fi

    ln -sf "$env_file" "$active_link"
    log_info "Active environment switched to: $env"
}

# Backup environment configuration
backup_environment() {
    local env="$1"
    local config_dir="$2"

    validate_environment "$env"

    local env_file="$config_dir/environments/$env.env"
    local backup_dir="$config_dir/backup"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$backup_dir/${env}_${timestamp}.env"

    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi

    mkdir -p "$backup_dir"
    cp "$env_file" "$backup_file"

    log_info "Environment backed up to: $backup_file"
}

# Restore environment configuration
restore_environment() {
    local env="$1"
    local backup_file="$2"
    local config_dir="$3"

    validate_environment "$env"

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    local env_file="$config_dir/environments/$env.env"

    # Backup current configuration before restore
    if [[ -f "$env_file" ]]; then
        backup_environment "$env" "$config_dir"
    fi

    cp "$backup_file" "$env_file"
    log_info "Environment restored from: $backup_file"
}

# List available environments
list_environments() {
    local config_dir="$1"
    local env_dir="$config_dir/environments"

    log_info "Available environments:"

    if [[ -d "$env_dir" ]]; then
        for env_file in "$env_dir"/*.env; do
            if [[ -f "$env_file" ]]; then
                local env_name=$(basename "$env_file" .env)
                local is_active=""

                # Check if this is the active environment
                local active_link="$config_dir/.env"
                if [[ -L "$active_link" ]]; then
                    local active_env=$(readlink "$active_link" | sed 's|.*/||' | sed 's|\.env$||')
                    if [[ "$active_env" == "$env_name" ]]; then
                        is_active=" (active)"
                    fi
                fi

                echo "  - $env_name$is_active"
            fi
        done
    else
        log_warn "Environment directory not found: $env_dir"
    fi
}

# Compare two environments
diff_environments() {
    local env1="$1"
    local env2="$2"
    local config_dir="$3"

    validate_environment "$env1"
    validate_environment "$env2"

    local file1="$config_dir/environments/$env1.env"
    local file2="$config_dir/environments/$env2.env"

    if [[ ! -f "$file1" ]]; then
        log_error "Environment file not found: $file1"
        return 1
    fi

    if [[ ! -f "$file2" ]]; then
        log_error "Environment file not found: $file2"
        return 1
    fi

    log_info "Comparing $env1 vs $env2:"
    diff -u "$file1" "$file2" || true
}

# Update configuration value
update_config() {
    local key="$1"
    local value="$2"
    local config_dir="$3"

    local active_link="$config_dir/.env"

    if [[ ! -L "$active_link" ]]; then
        log_error "No active environment found"
        return 1
    fi

    local env_file=$(readlink "$active_link")

    # Backup before update
    local env_name=$(basename "$env_file" .env)
    backup_environment "$env_name" "$config_dir"

    # Update the value
    if grep -q "^$key=" "$env_file"; then
        sed -i "s|^$key=.*|$key=$value|" "$env_file"
        log_info "Updated $key in active environment"
    else
        echo "$key=$value" >> "$env_file"
        log_info "Added $key to active environment"
    fi
}

# Parse command line arguments
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        setup)
            if [[ $# -lt 2 ]]; then
                log_error "Environment name required for setup command"
                exit 1
            fi
            COMMAND="setup"
            TARGET_ENV="$2"
            shift 2
            ;;
        validate)
            if [[ $# -lt 2 ]]; then
                log_error "Environment name required for validate command"
                exit 1
            fi
            COMMAND="validate"
            TARGET_ENV="$2"
            shift 2
            ;;
        switch)
            if [[ $# -lt 2 ]]; then
                log_error "Environment name required for switch command"
                exit 1
            fi
            COMMAND="switch"
            TARGET_ENV="$2"
            shift 2
            ;;
        backup)
            if [[ $# -lt 2 ]]; then
                log_error "Environment name required for backup command"
                exit 1
            fi
            COMMAND="backup"
            TARGET_ENV="$2"
            shift 2
            ;;
        restore)
            if [[ $# -lt 3 ]]; then
                log_error "Environment name and backup file required for restore command"
                exit 1
            fi
            COMMAND="restore"
            TARGET_ENV="$2"
            BACKUP_FILE="$3"
            shift 3
            ;;
        list)
            COMMAND="list"
            shift
            ;;
        diff)
            if [[ $# -lt 3 ]]; then
                log_error "Two environment names required for diff command"
                exit 1
            fi
            COMMAND="diff"
            ENV1="$2"
            ENV2="$3"
            shift 3
            ;;
        update)
            if [[ $# -lt 3 ]]; then
                log_error "Key and value required for update command"
                exit 1
            fi
            COMMAND="update"
            UPDATE_KEY="$2"
            UPDATE_VALUE="$3"
            shift 3
            ;;
        *)
            log_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if command was provided
if [[ -z "${COMMAND:-}" ]]; then
    log_error "No command specified"
    usage
    exit 1
fi

# Main execution
case "$COMMAND" in
    setup)
        setup_environment "$TARGET_ENV" "$CONFIG_DIR"
        ;;
    validate)
        validate_configuration "$TARGET_ENV" "$CONFIG_DIR"
        ;;
    switch)
        switch_environment "$TARGET_ENV" "$CONFIG_DIR"
        ;;
    backup)
        backup_environment "$TARGET_ENV" "$CONFIG_DIR"
        ;;
    restore)
        restore_environment "$TARGET_ENV" "$BACKUP_FILE" "$CONFIG_DIR"
        ;;
    list)
        list_environments "$CONFIG_DIR"
        ;;
    diff)
        diff_environments "$ENV1" "$ENV2" "$CONFIG_DIR"
        ;;
    update)
        update_config "$UPDATE_KEY" "$UPDATE_VALUE" "$CONFIG_DIR"
        ;;
esac