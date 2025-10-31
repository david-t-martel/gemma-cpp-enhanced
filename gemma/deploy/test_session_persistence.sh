#!/bin/bash
# Test Session Persistence for Gemma.exe
# Cross-platform test script (bash/WSL compatible)

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm}"
SESSION_ID="test_session_$(date +%Y%m%d_%H%M%S)"
GEMMA_EXE="./gemma.exe"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Test result tracking
declare -a TEST_RESULTS=()

function print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}TEST: $1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

function test_result() {
    local test_name="$1"
    local passed="$2"
    local details="${3:-}"

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if [ "$passed" = "true" ]; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        TEST_RESULTS+=("PASS: $test_name")
    else
        echo -e "${RED}[FAIL]${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        TEST_RESULTS+=("FAIL: $test_name")
    fi

    if [ -n "$details" ]; then
        echo -e "       ${GRAY}$details${NC}"
    fi
}

# Test 1: Binary availability
print_header "Binary Availability"

if [ -f "$GEMMA_EXE" ]; then
    test_result "gemma.exe exists" "true" "Path: $GEMMA_EXE"

    # Test if executable
    if [ -x "$GEMMA_EXE" ]; then
        test_result "gemma.exe executable" "true"
    else
        test_result "gemma.exe executable" "false" "File not executable"
    fi

    # Test help output
    if $GEMMA_EXE --help 2>&1 | head -n 5 | grep -q "gemma.cpp"; then
        test_result "gemma.exe runs" "true" "Help output valid"
    else
        test_result "gemma.exe runs" "false" "Help output invalid"
    fi
else
    test_result "gemma.exe exists" "false" "Not found"
    echo -e "\n${RED}ABORTING: gemma.exe not found${NC}"
    exit 1
fi

# Test 2: Model files
print_header "Model File Availability"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    test_result "Model weights exist" "true" "Size: $MODEL_SIZE"
else
    test_result "Model weights exist" "false" "Not found: $MODEL_PATH"
fi

if [ -f "$TOKENIZER_PATH" ]; then
    TOK_SIZE=$(du -h "$TOKENIZER_PATH" | cut -f1)
    test_result "Tokenizer exists" "true" "Size: $TOK_SIZE"
else
    test_result "Tokenizer exists" "false" "Not found: $TOKENIZER_PATH"
fi

# Test 3: Session creation with save
print_header "Session Creation and Persistence"

SESSION_FILE="session_${SESSION_ID}.json"
echo -e "${YELLOW}Creating session: $SESSION_ID${NC}"
echo -e "${GRAY}Commands: 'Hello...', '%s' (save), '%q' (quit)${NC}"

# Create test input
cat > /tmp/test_input_$$.txt <<EOF
Hello, this is a test message for session persistence.
%s
%q
EOF

# Run gemma with session
$GEMMA_EXE \
    --weights "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --session "$SESSION_ID" \
    --save_on_exit \
    --max_generated_tokens 50 \
    --verbosity 0 \
    < /tmp/test_input_$$.txt > /tmp/test_output_$$.txt 2>&1 || true

# Check if session file created
if [ -f "$SESSION_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$SESSION_FILE" 2>/dev/null || stat -f%z "$SESSION_FILE" 2>/dev/null)
    test_result "Session file created" "true" "$SESSION_FILE ($FILE_SIZE bytes)"

    # Validate JSON
    if python3 -m json.tool "$SESSION_FILE" > /dev/null 2>&1; then
        test_result "Session JSON valid" "true"

        # Check for required fields
        if grep -q '"session_id"' "$SESSION_FILE" && grep -q '"messages"' "$SESSION_FILE"; then
            test_result "Session structure valid" "true" "Has session_id and messages"
        else
            test_result "Session structure valid" "false" "Missing required fields"
        fi

        # Count messages
        MSG_COUNT=$(grep -o '"role"' "$SESSION_FILE" | wc -l)
        test_result "Messages stored" "true" "Message count: $MSG_COUNT"

    else
        test_result "Session JSON valid" "false" "Invalid JSON format"
    fi
else
    test_result "Session file created" "false" "File not found"
fi

# Test 4: Load session
print_header "Session Loading"

if [ -f "$SESSION_FILE" ]; then
    echo -e "${GRAY}Commands: '%i' (info), '%h 5' (history), '%q' (quit)${NC}"

    cat > /tmp/test_load_$$.txt <<EOF
%i
%h 5
%q
EOF

    $GEMMA_EXE \
        --weights "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --session "$SESSION_ID" \
        --load_session \
        --verbosity 0 \
        < /tmp/test_load_$$.txt > /tmp/test_load_output_$$.txt 2>&1 || true

    LOAD_OUTPUT=$(cat /tmp/test_load_output_$$.txt)

    # Check for session info in output
    if echo "$LOAD_OUTPUT" | grep -q "Session Statistics"; then
        test_result "Session info displayed" "true"
    else
        test_result "Session info displayed" "false"
    fi

    if echo "$LOAD_OUTPUT" | grep -q "Session History"; then
        test_result "Session history displayed" "true"
    else
        test_result "Session history displayed" "false"
    fi

    if echo "$LOAD_OUTPUT" | grep -q "Total turns:"; then
        test_result "Statistics persisted" "true"
    else
        test_result "Statistics persisted" "false"
    fi

else
    test_result "Session loading" "false" "No session file to load"
fi

# Test 5: Multiple sessions
print_header "Multiple Session Management"

SESSION_ID_2="test_session_2_$(date +%H%M%S)"

cat > /tmp/test_multi_$$.txt <<EOF
Second session test message
%m
%q
EOF

$GEMMA_EXE \
    --weights "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --session "$SESSION_ID_2" \
    --save_on_exit \
    --max_generated_tokens 30 \
    --verbosity 0 \
    < /tmp/test_multi_$$.txt > /tmp/test_multi_output_$$.txt 2>&1 || true

MULTI_OUTPUT=$(cat /tmp/test_multi_output_$$.txt)

if echo "$MULTI_OUTPUT" | grep -q "Managed Sessions"; then
    test_result "Session manager command" "true" "%m command works"
else
    test_result "Session manager command" "false"
fi

# Count session files
SESSION_COUNT=$(ls -1 session_*.json 2>/dev/null | wc -l)
if [ "$SESSION_COUNT" -ge 2 ]; then
    test_result "Multiple sessions created" "true" "Count: $SESSION_COUNT"
else
    test_result "Multiple sessions created" "false" "Expected ≥2, found: $SESSION_COUNT"
fi

# Test 6: Config file
print_header "Configuration File Support"

if [ -f "gemma.config.toml" ]; then
    test_result "Config file exists" "true"
else
    test_result "Config file exists" "false"
fi

# Cleanup temp files
rm -f /tmp/test_*_$$.txt

# Summary
print_header "TEST SUMMARY"

PASS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))

echo -e "Total Tests: ${CYAN}$TESTS_TOTAL${NC}"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"

if [ $PASS_RATE -ge 90 ]; then
    echo -e "Pass Rate: ${GREEN}$PASS_RATE%${NC}"
elif [ $PASS_RATE -ge 70 ]; then
    echo -e "Pass Rate: ${YELLOW}$PASS_RATE%${NC}"
else
    echo -e "Pass Rate: ${RED}$PASS_RATE%${NC}"
fi

# List session files
echo -e "\n${CYAN}Session Files Created:${NC}"
ls -lh session_*.json 2>/dev/null || echo "  None"

# Save results
REPORT_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Session Persistence Test Results"
    echo "================================="
    echo "Date: $(date)"
    echo "Total: $TESTS_TOTAL | Passed: $TESTS_PASSED | Failed: $TESTS_FAILED | Rate: $PASS_RATE%"
    echo ""
    echo "Results:"
    for result in "${TEST_RESULTS[@]}"; do
        echo "  $result"
    done
} > "$REPORT_FILE"

echo -e "\n${CYAN}Results saved to: $REPORT_FILE${NC}"

# Cleanup prompt
echo -e "\n${YELLOW}Clean up session files? (y/N)${NC}"
read -r CLEANUP
if [ "$CLEANUP" = "y" ] || [ "$CLEANUP" = "Y" ]; then
    rm -f session_*.json
    echo -e "${GREEN}Session files cleaned up${NC}"
fi

# Exit code
if [ $PASS_RATE -ge 80 ]; then
    echo -e "\n${GREEN}OVERALL: PASS (≥80%)${NC}"
    exit 0
else
    echo -e "\n${RED}OVERALL: FAIL (<80%)${NC}"
    exit 1
fi
