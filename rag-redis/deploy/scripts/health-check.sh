#!/bin/bash
# RAG-Redis MCP Server Health Check Script
# Comprehensive health checking for production deployments

set -euo pipefail

# Configuration
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8080/health}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"
TIMEOUT="${TIMEOUT:-10}"
VERBOSE="${VERBOSE:-false}"
CHECK_REDIS="${CHECK_REDIS:-true}"
CHECK_VECTOR_STORE="${CHECK_VECTOR_STORE:-true}"
CHECK_MEMORY_TIERS="${CHECK_MEMORY_TIERS:-true}"

# Colors
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
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Health check results
HEALTH_STATUS=0
CHECKS_PASSED=0
CHECKS_TOTAL=0

# Function to run a health check
run_check() {
    local check_name="$1"
    local check_function="$2"

    ((CHECKS_TOTAL++))
    log_debug "Running check: $check_name"

    if $check_function; then
        log_info "✓ $check_name"
        ((CHECKS_PASSED++))
        return 0
    else
        log_error "✗ $check_name"
        HEALTH_STATUS=1
        return 1
    fi
}

# Check if MCP server is responding
check_mcp_server() {
    if command -v curl &> /dev/null; then
        curl -f -s --max-time "$TIMEOUT" "$HEALTH_CHECK_URL" > /dev/null 2>&1
    elif command -v wget &> /dev/null; then
        wget -q --timeout="$TIMEOUT" --spider "$HEALTH_CHECK_URL" > /dev/null 2>&1
    else
        # Fallback using bash TCP
        timeout "$TIMEOUT" bash -c "</dev/tcp/localhost/8080" 2>/dev/null
    fi
}

# Check Redis connectivity
check_redis() {
    if [[ "$CHECK_REDIS" != "true" ]]; then
        return 0
    fi

    if command -v redis-cli &> /dev/null; then
        redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1
    else
        # Extract host and port from Redis URL
        local redis_host=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\):.*|\1|p')
        local redis_port=$(echo "$REDIS_URL" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')

        redis_host="${redis_host:-127.0.0.1}"
        redis_port="${redis_port:-6379}"

        timeout "$TIMEOUT" bash -c "</dev/tcp/$redis_host/$redis_port" 2>/dev/null
    fi
}

# Check vector store functionality
check_vector_store() {
    if [[ "$CHECK_VECTOR_STORE" != "true" ]]; then
        return 0
    fi

    # Test vector store via MCP API
    local test_payload='{"method":"search","params":{"query":"test","limit":1}}'

    if command -v curl &> /dev/null; then
        local response=$(curl -s --max-time "$TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "$test_payload" \
            "$HEALTH_CHECK_URL/api/search" 2>/dev/null)

        # Check if response contains expected structure
        echo "$response" | grep -q '"status"' 2>/dev/null
    else
        # Simplified check - just verify the endpoint exists
        if command -v wget &> /dev/null; then
            wget -q --timeout="$TIMEOUT" --spider "$HEALTH_CHECK_URL/api/search" > /dev/null 2>&1
        else
            return 0  # Skip if no HTTP client available
        fi
    fi
}

# Check memory tier functionality
check_memory_tiers() {
    if [[ "$CHECK_MEMORY_TIERS" != "true" ]]; then
        return 0
    fi

    # Test memory tier status via MCP API
    if command -v curl &> /dev/null; then
        local response=$(curl -s --max-time "$TIMEOUT" \
            "$HEALTH_CHECK_URL/api/memory/status" 2>/dev/null)

        # Check if response contains expected memory tiers
        echo "$response" | grep -q '"working".*"short_term".*"long_term"' 2>/dev/null
    else
        return 0  # Skip if no HTTP client available
    fi
}

# Check system resources
check_system_resources() {
    # Check available memory (at least 512MB)
    if [[ -r /proc/meminfo ]]; then
        local available_mem=$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
        if [[ $available_mem -lt 524288 ]]; then  # 512MB in KB
            log_warn "Low available memory: ${available_mem}KB"
            return 1
        fi
    fi

    # Check disk space (at least 1GB free)
    local disk_free=$(df / | awk 'NR==2 {print $4}' 2>/dev/null || echo "0")
    if [[ $disk_free -lt 1048576 ]]; then  # 1GB in KB
        log_warn "Low disk space: ${disk_free}KB"
        return 1
    fi

    return 0
}

# Check process status
check_process() {
    # Check if MCP server process is running
    if pgrep -f "mcp-server" > /dev/null 2>&1; then
        return 0
    else
        log_error "MCP server process not found"
        return 1
    fi
}

# Check log files for recent errors
check_logs() {
    local log_file="${LOG_DIR:-/var/log/rag-redis}/mcp-server.log"

    if [[ -r "$log_file" ]]; then
        # Check for recent errors (last 5 minutes)
        local recent_errors=$(find "$log_file" -mmin -5 -exec grep -c "ERROR" {} \; 2>/dev/null || echo "0")
        if [[ $recent_errors -gt 10 ]]; then
            log_warn "High error rate in logs: $recent_errors errors in last 5 minutes"
            return 1
        fi
    fi

    return 0
}

# Generate health report
generate_report() {
    echo ""
    echo "=== Health Check Report ==="
    echo "Timestamp: $(date)"
    echo "Checks Passed: $CHECKS_PASSED/$CHECKS_TOTAL"
    echo "Overall Status: $(if [[ $HEALTH_STATUS -eq 0 ]]; then echo "HEALTHY"; else echo "UNHEALTHY"; fi)"
    echo ""

    if [[ $HEALTH_STATUS -eq 0 ]]; then
        log_info "All health checks passed"
    else
        log_error "One or more health checks failed"
        echo "Please check the service logs and configuration."
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            HEALTH_CHECK_URL="$2"
            shift 2
            ;;
        --redis-url)
            REDIS_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="true"
            shift
            ;;
        --skip-redis)
            CHECK_REDIS="false"
            shift
            ;;
        --skip-vector-store)
            CHECK_VECTOR_STORE="false"
            shift
            ;;
        --skip-memory-tiers)
            CHECK_MEMORY_TIERS="false"
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [options]

Options:
    --url URL                Health check URL (default: http://localhost:8080/health)
    --redis-url URL          Redis URL (default: redis://127.0.0.1:6379)
    --timeout SECONDS        Request timeout (default: 10)
    --verbose, -v            Verbose output
    --skip-redis             Skip Redis connectivity check
    --skip-vector-store      Skip vector store functionality check
    --skip-memory-tiers      Skip memory tier functionality check
    --help, -h               Show this help message

Environment variables:
    HEALTH_CHECK_URL, REDIS_URL, TIMEOUT, VERBOSE
    CHECK_REDIS, CHECK_VECTOR_STORE, CHECK_MEMORY_TIERS
EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main health check execution
main() {
    log_info "Starting health check for RAG-Redis MCP Server"
    log_debug "Health check URL: $HEALTH_CHECK_URL"
    log_debug "Redis URL: $REDIS_URL"
    log_debug "Timeout: ${TIMEOUT}s"

    # Run all health checks
    run_check "MCP Server Response" check_mcp_server
    run_check "Redis Connectivity" check_redis
    run_check "Vector Store Functionality" check_vector_store
    run_check "Memory Tier Functionality" check_memory_tiers
    run_check "System Resources" check_system_resources
    run_check "Process Status" check_process
    run_check "Log Analysis" check_logs

    # Generate report
    generate_report

    # Exit with appropriate code
    exit $HEALTH_STATUS
}

# Run main function
main