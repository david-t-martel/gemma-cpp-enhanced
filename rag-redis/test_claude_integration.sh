#!/bin/bash

# Comprehensive Claude CLI Integration Test for RAG-Redis MCP Server
# This script tests the MCP server integration with Claude CLI using various scenarios

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/c/codedev/llm/rag-redis"
MCP_CONFIG="$PROJECT_ROOT/mcp.json"
LOG_FILE="$PROJECT_ROOT/claude_integration_test.log"
REDIS_PORT=6380

# Utility functions
print_header() {
    echo -e "\n${BOLD}${BLUE}================================================================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}================================================================================${NC}\n"
}

print_test() {
    echo -e "${CYAN}[TEST] $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Claude CLI
    if ! command -v claude &> /dev/null; then
        print_error "Claude CLI not found. Please install Claude CLI first."
        exit 1
    fi
    print_success "Claude CLI found: $(claude --version)"
    
    # Check Redis
    if ! redis-cli -p $REDIS_PORT ping &> /dev/null; then
        print_error "Redis not running on port $REDIS_PORT. Please start Redis server."
        exit 1
    fi
    print_success "Redis server running on port $REDIS_PORT"
    
    # Check MCP config
    if [ ! -f "$MCP_CONFIG" ]; then
        print_error "MCP configuration not found at $MCP_CONFIG"
        exit 1
    fi
    print_success "MCP configuration found"
    
    # Check server binary
    SERVER_BINARY="$PROJECT_ROOT/rag-redis-system/mcp-server/target/release/mcp-server.exe"
    if [ ! -f "$SERVER_BINARY" ]; then
        print_error "MCP server binary not found at $SERVER_BINARY"
        print_info "Please build the server first: cd rag-redis-system/mcp-server && cargo build --release"
        exit 1
    fi
    print_success "MCP server binary found"
    
    # Validate MCP config with Claude
    print_test "Validating MCP configuration with Claude CLI..."
    if claude --validate-mcp-config --mcp-config "$MCP_CONFIG" &> /dev/null; then
        print_success "MCP configuration is valid"
    else
        print_warning "MCP configuration validation had issues (continuing anyway)"
    fi
}

# Test basic Claude integration
test_basic_integration() {
    print_header "Basic Claude Integration Test"
    
    print_test "Testing basic Claude MCP connection..."
    
    # Simple health check through Claude
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Test prompt for health check
    TEST_PROMPT="Use the rag-redis MCP server to run a health check. Please call the health_check tool with include_metrics set to true."
    
    echo "Running: $CLAUDE_CMD \"$TEST_PROMPT\"" >> "$LOG_FILE"
    
    if timeout 60s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Basic integration test passed"
    else
        print_error "Basic integration test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Test document ingestion through Claude
test_document_ingestion() {
    print_header "Document Ingestion Test"
    
    print_test "Testing document ingestion through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Create test document content
    TEST_CONTENT="This is a comprehensive test document about machine learning and artificial intelligence. It covers topics like neural networks, deep learning, natural language processing, and computer vision. The document is being ingested to test the RAG-Redis system's capability to handle document storage and retrieval."
    
    TEST_PROMPT="Use the rag-redis MCP server to ingest a document. Call the ingest_document tool with this content: '$TEST_CONTENT'. Include metadata with source='claude_integration_test', title='Test Document', and timestamp with current time."
    
    echo "Running document ingestion test..." >> "$LOG_FILE"
    
    if timeout 120s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Document ingestion test passed"
    else
        print_error "Document ingestion test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Test search functionality through Claude
test_search_functionality() {
    print_header "Search Functionality Test"
    
    print_test "Testing search functionality through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Test semantic search
    TEST_PROMPT="Use the rag-redis MCP server to search for content. Call the search tool with query 'machine learning neural networks', limit of 5, and threshold of 0.6. Then also test hybrid_search with the same query and keyword_weight of 0.3."
    
    echo "Running search functionality test..." >> "$LOG_FILE"
    
    if timeout 120s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Search functionality test passed"
    else
        print_error "Search functionality test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Test memory operations through Claude
test_memory_operations() {
    print_header "Memory Operations Test"
    
    print_test "Testing memory operations through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Test memory store and recall
    TEST_PROMPT="Use the rag-redis MCP server to test memory operations. First, call memory_store to store 'User prefers detailed technical explanations with code examples' with memory_type 'working', importance 0.8, and context_hints ['user_preference', 'explanation_style']. Then call memory_recall with query 'user preferences' and memory_type 'working'."
    
    echo "Running memory operations test..." >> "$LOG_FILE"
    
    if timeout 120s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Memory operations test passed"
    else
        print_error "Memory operations test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Test agent-specific memory operations
test_agent_memory() {
    print_header "Agent Memory Test"
    
    print_test "Testing agent-specific memory operations through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Test agent memory functions
    TEST_PROMPT="Use the rag-redis MCP server to test agent memory. First, call agent_memory_store with content 'Claude works best with structured responses and clear formatting', agent_type 'claude', context_hints ['response_format', 'claude_preferences'], and importance 0.9. Then call agent_memory_retrieve with query 'claude preferences' and agent_type 'claude'. Finally, call memory_digest with agent_type 'claude' and topic 'preferences'."
    
    echo "Running agent memory test..." >> "$LOG_FILE"
    
    if timeout 120s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Agent memory test passed"
    else
        print_error "Agent memory test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Test project context operations
test_project_context() {
    print_header "Project Context Test"
    
    print_test "Testing project context operations through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Generate unique project ID for this test
    PROJECT_ID="claude-integration-test-$(date +%s)"
    
    # Test project context save and load
    TEST_PROMPT="Use the rag-redis MCP server to test project context. First, call project_context_save with project_id '$PROJECT_ID', context containing current_task='Claude integration testing', progress='Running comprehensive tests', and next_steps=['Performance validation', 'Error handling'], and metadata with test_run=true. Then call project_context_load with the same project_id and include_memories=true."
    
    echo "Running project context test..." >> "$LOG_FILE"
    
    if timeout 120s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Project context test passed"
    else
        print_error "Project context test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Test error handling
test_error_handling() {
    print_header "Error Handling Test"
    
    print_test "Testing error handling through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Test with invalid tool call
    TEST_PROMPT="Use the rag-redis MCP server to test error handling. Try to call a non-existent tool called 'invalid_tool_name' with some parameters. This should fail gracefully."
    
    echo "Running error handling test..." >> "$LOG_FILE"
    
    # For error handling, we expect the command to complete but with errors
    if timeout 120s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Error handling test completed (expected to have controlled errors)"
    else
        print_warning "Error handling test had issues - check $LOG_FILE for details"
    fi
}

# Test comprehensive workflow
test_comprehensive_workflow() {
    print_header "Comprehensive Workflow Test"
    
    print_test "Testing comprehensive workflow through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Complex workflow test
    TEST_PROMPT="Use the rag-redis MCP server for a comprehensive workflow test:

1. First, run health_check with include_metrics=true
2. Ingest a document about 'Python programming best practices'
3. Store a memory about user preferences for Python coding style
4. Search for content related to 'programming practices'
5. Retrieve memories about coding preferences
6. Generate a memory digest for agent type 'claude'
7. Save the current workflow state as project context

Please execute these steps in order and provide a summary of the results."
    
    echo "Running comprehensive workflow test..." >> "$LOG_FILE"
    
    if timeout 300s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Comprehensive workflow test passed"
    else
        print_error "Comprehensive workflow test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Performance stress test
test_performance_stress() {
    print_header "Performance Stress Test"
    
    print_test "Testing performance under stress through Claude..."
    
    CLAUDE_CMD="claude --mcp-config \"$MCP_CONFIG\" --debug"
    
    # Performance test with multiple operations
    TEST_PROMPT="Use the rag-redis MCP server for performance testing. Perform the following operations rapidly:

1. Ingest 3 different documents about technology topics
2. Perform 5 different search queries
3. Store 5 different memories with varying importance levels
4. Retrieve memories with different query patterns
5. Run health_check with metrics to see performance impact

Focus on speed and efficiency. Provide timing information if available."
    
    echo "Running performance stress test..." >> "$LOG_FILE"
    
    if timeout 300s $CLAUDE_CMD "$TEST_PROMPT" >> "$LOG_FILE" 2>&1; then
        print_success "Performance stress test passed"
    else
        print_error "Performance stress test failed - check $LOG_FILE for details"
        return 1
    fi
}

# Run all tests
run_all_tests() {
    print_header "RAG-Redis Claude CLI Integration Test Suite"
    
    # Initialize log file
    echo "=== RAG-Redis Claude CLI Integration Test Log ===" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "Project Root: $PROJECT_ROOT" >> "$LOG_FILE"
    echo "MCP Config: $MCP_CONFIG" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    local tests_passed=0
    local tests_failed=0
    local test_results=()
    
    # Run all test functions
    local test_functions=(
        "test_basic_integration"
        "test_document_ingestion"
        "test_search_functionality"
        "test_memory_operations"
        "test_agent_memory"
        "test_project_context"
        "test_error_handling"
        "test_comprehensive_workflow"
        "test_performance_stress"
    )
    
    for test_func in "${test_functions[@]}"; do
        if $test_func; then
            ((tests_passed++))
            test_results+=("✓ $test_func")
        else
            ((tests_failed++))
            test_results+=("✗ $test_func")
        fi
        echo "" >> "$LOG_FILE"
    done
    
    # Print summary
    print_header "Test Results Summary"
    
    echo -e "${BOLD}Individual Test Results:${NC}"
    for result in "${test_results[@]}"; do
        if [[ $result == ✓* ]]; then
            echo -e "${GREEN}$result${NC}"
        else
            echo -e "${RED}$result${NC}"
        fi
    done
    
    echo ""
    echo -e "${BOLD}Overall Statistics:${NC}"
    echo -e "Total Tests: $((tests_passed + tests_failed))"
    echo -e "${GREEN}Passed: $tests_passed${NC}"
    echo -e "${RED}Failed: $tests_failed${NC}"
    echo -e "Success Rate: $(( tests_passed * 100 / (tests_passed + tests_failed) ))%"
    
    # Log summary
    echo "" >> "$LOG_FILE"
    echo "=== Test Summary ===" >> "$LOG_FILE"
    echo "Tests Passed: $tests_passed" >> "$LOG_FILE"
    echo "Tests Failed: $tests_failed" >> "$LOG_FILE"
    echo "Completed: $(date)" >> "$LOG_FILE"
    
    print_info "Detailed log available at: $LOG_FILE"
    
    # Exit with appropriate code
    if [ $tests_failed -eq 0 ]; then
        print_success "All tests passed! RAG-Redis MCP integration with Claude CLI is working correctly."
        exit 0
    else
        print_error "Some tests failed. Please check the log file for details."
        exit 1
    fi
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    # Add any cleanup operations here if needed
}

# Signal handling
trap cleanup EXIT

# Main execution
main() {
    echo -e "${BOLD}${BLUE}RAG-Redis Claude CLI Integration Test${NC}"
    echo -e "${BLUE}Testing MCP server integration with Claude CLI${NC}\n"
    
    check_prerequisites
    run_all_tests
}

# Run main function
main "$@"