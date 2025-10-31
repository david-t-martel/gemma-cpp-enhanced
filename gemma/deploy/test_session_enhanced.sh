#!/bin/bash
# Enhanced Session Persistence Tests for Gemma.exe
# Comprehensive testing with feature detection and detailed validation

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm}"
SESSION_ID="test_session_$(date +%Y%m%d_%H%M%S)"
GEMMA_EXE="./gemma.exe"
TEST_TIMEOUT=120

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
TESTS_TOTAL=0

# Feature flags
SESSION_SUPPORTED=false
LOAD_SUPPORTED=false
SAVE_SUPPORTED=false

# Test results
declare -a TEST_RESULTS=()

function print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
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
    elif [ "$passed" = "skip" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $test_name"
        TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
        TEST_RESULTS+=("SKIP: $test_name")
    else
        echo -e "${RED}[FAIL]${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        TEST_RESULTS+=("FAIL: $test_name")
    fi

    if [ -n "$details" ]; then
        echo -e "       ${GRAY}$details${NC}"
    fi
}

# ============================================
# PHASE 1: Binary and Environment Validation
# ============================================

print_header "PHASE 1: Binary and Environment"

# Test 1.1: Binary exists
if [ -f "$GEMMA_EXE" ]; then
    BIN_SIZE=$(du -h "$GEMMA_EXE" | cut -f1)
    BIN_DATE=$(stat -c%y "$GEMMA_EXE" 2>/dev/null || stat -f"%Sm" "$GEMMA_EXE" 2>/dev/null)
    test_result "Binary exists" "true" "Size: $BIN_SIZE, Date: $BIN_DATE"
else
    test_result "Binary exists" "false" "Not found: $GEMMA_EXE"
    echo -e "\n${RED}ABORTING: gemma.exe not found${NC}"
    exit 1
fi

# Test 1.2: Binary executable
if [ -x "$GEMMA_EXE" ]; then
    test_result "Binary executable" "true"
else
    chmod +x "$GEMMA_EXE" 2>/dev/null && test_result "Binary executable" "true" "Permissions fixed" || test_result "Binary executable" "false"
fi

# Test 1.3: Help runs successfully
if timeout 10s $GEMMA_EXE --help >/dev/null 2>&1; then
    test_result "Help command works" "true"
else
    test_result "Help command works" "false" "Exit code: $?"
fi

# Test 1.4: Model files available
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    test_result "Model weights exist" "true" "Size: $MODEL_SIZE"
else
    test_result "Model weights exist" "false" "Path: $MODEL_PATH"
    echo -e "\n${YELLOW}WARNING: Model files not found. Session tests will fail.${NC}"
fi

if [ -f "$TOKENIZER_PATH" ]; then
    TOK_SIZE=$(du -h "$TOKENIZER_PATH" | cut -f1)
    test_result "Tokenizer exists" "true" "Size: $TOK_SIZE"
else
    test_result "Tokenizer exists" "false" "Path: $TOKENIZER_PATH"
fi

# ============================================
# PHASE 2: Feature Detection
# ============================================

print_header "PHASE 2: Feature Detection"

HELP_OUTPUT=$($GEMMA_EXE --help 2>&1 || true)

# Test 2.1: Session flag support
if echo "$HELP_OUTPUT" | grep -q "\-\-session"; then
    test_result "Session flag (--session)" "true" "Feature available"
    SESSION_SUPPORTED=true
else
    test_result "Session flag (--session)" "false" "Feature NOT available - binary needs rebuild"
    SESSION_SUPPORTED=false
fi

# Test 2.2: Load session support
if echo "$HELP_OUTPUT" | grep -q "\-\-load_session"; then
    test_result "Load flag (--load_session)" "true"
    LOAD_SUPPORTED=true
else
    test_result "Load flag (--load_session)" "false"
    LOAD_SUPPORTED=false
fi

# Test 2.3: Save on exit support
if echo "$HELP_OUTPUT" | grep -q "\-\-save_on_exit"; then
    test_result "Save flag (--save_on_exit)" "true"
    SAVE_SUPPORTED=true
else
    test_result "Save flag (--save_on_exit)" "false"
    SAVE_SUPPORTED=false
fi

# Test 2.4: Check binary build date
EXPECTED_DATE="2025-10-23 07:30"
if [ -n "$BIN_DATE" ]; then
    if [[ "$BIN_DATE" > "$EXPECTED_DATE" ]]; then
        test_result "Binary build date" "true" "Binary includes session code"
    else
        test_result "Binary build date" "false" "Binary predates session code ($EXPECTED_DATE)"
    fi
fi

# Abort if session features not available
if [ "$SESSION_SUPPORTED" = "false" ]; then
    echo -e "\n${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  SESSION FEATURES NOT AVAILABLE - ABORTING SESSION TESTS  ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}Binary needs to be rebuilt with session management code.${NC}"
    echo -e "${YELLOW}Expected: Build date after $EXPECTED_DATE${NC}"
    echo -e "${YELLOW}Actual:   $BIN_DATE${NC}"
    echo -e "\n${CYAN}To rebuild:${NC}"
    echo -e "  1. Open Visual Studio 2022"
    echo -e "  2. File → Open → CMake → Select CMakeLists.txt"
    echo -e "  3. Build → Build All (Ctrl+Shift+B)"
    echo -e "  4. Copy binary from out/build/x64-Release/gemma.exe to deploy/\n"

    # Generate summary report
    print_header "TEST SUMMARY (PARTIAL - SESSION TESTS SKIPPED)"

    PASS_RATE=0
    if [ $TESTS_TOTAL -gt 0 ]; then
        PASS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    fi

    echo -e "Total Tests: ${CYAN}$TESTS_TOTAL${NC}"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo -e "Skipped: ${YELLOW}Session tests (binary lacks features)${NC}"
    echo -e "Pass Rate: ${YELLOW}$PASS_RATE%${NC} (of preliminary tests)\n"

    exit 1
fi

echo -e "\n${GREEN}✓ All session features detected - proceeding with tests${NC}"

# ============================================
# PHASE 3: Session Creation and Persistence
# ============================================

print_header "PHASE 3: Session Creation"

SESSION_FILE="session_${SESSION_ID}.json"
echo -e "${YELLOW}Creating session: $SESSION_ID${NC}"

# Create test input
cat > /tmp/test_input_$$.txt <<EOF
Hello, this is a test message for session persistence testing.
%s
%q
EOF

# Run gemma with session
echo -e "${GRAY}Running: gemma --session $SESSION_ID --save_on_exit${NC}"
timeout $TEST_TIMEOUT $GEMMA_EXE \
    --weights "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --session "$SESSION_ID" \
    --save_on_exit \
    --max_generated_tokens 50 \
    --verbosity 0 \
    < /tmp/test_input_$$.txt > /tmp/test_output_$$.txt 2>&1 || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        test_result "Session creation (timeout)" "false" "Process timed out after ${TEST_TIMEOUT}s"
    else
        test_result "Session creation (exit code)" "false" "Exit code: $EXIT_CODE"
    fi
}

# Test 3.1: Session file created
if [ -f "$SESSION_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$SESSION_FILE" 2>/dev/null || stat -f%z "$SESSION_FILE" 2>/dev/null)
    test_result "Session file created" "true" "$SESSION_FILE ($FILE_SIZE bytes)"
else
    test_result "Session file created" "false" "File not found"
    echo -e "${YELLOW}Skipping validation tests - no session file${NC}"
    SESSION_FILE=""
fi

# ============================================
# PHASE 4: Session File Validation
# ============================================

if [ -n "$SESSION_FILE" ] && [ -f "$SESSION_FILE" ]; then
    print_header "PHASE 4: Session File Validation"

    # Test 4.1: Valid JSON
    if python3 -m json.tool "$SESSION_FILE" > /dev/null 2>&1; then
        test_result "JSON structure valid" "true"

        # Test 4.2: Required fields present
        if grep -q '"session_id"' "$SESSION_FILE"; then
            test_result "Field: session_id" "true"
        else
            test_result "Field: session_id" "false"
        fi

        if grep -q '"messages"' "$SESSION_FILE"; then
            test_result "Field: messages" "true"
        else
            test_result "Field: messages" "false"
        fi

        if grep -q '"conversation_length"' "$SESSION_FILE"; then
            test_result "Field: conversation_length" "true"
        else
            test_result "Field: conversation_length" "false"
        fi

        # Test 4.3: Session ID matches
        SAVED_ID=$(python3 -c "import json; print(json.load(open('$SESSION_FILE'))['session_id'])" 2>/dev/null || echo "PARSE_ERROR")
        if [ "$SAVED_ID" = "$SESSION_ID" ]; then
            test_result "Session ID preserved" "true" "ID: $SAVED_ID"
        else
            test_result "Session ID preserved" "false" "Expected: $SESSION_ID, Got: $SAVED_ID"
        fi

        # Test 4.4: Messages stored
        MSG_COUNT=$(python3 -c "import json; print(len(json.load(open('$SESSION_FILE')).get('messages', [])))" 2>/dev/null || echo "0")
        if [ "$MSG_COUNT" -gt 0 ]; then
            test_result "Messages persisted" "true" "Count: $MSG_COUNT"
        else
            test_result "Messages persisted" "false" "No messages in file (SaveToFile may be incomplete)"
        fi

        # Test 4.5: Conversation length
        CONV_LEN=$(python3 -c "import json; print(json.load(open('$SESSION_FILE')).get('conversation_length', 0))" 2>/dev/null || echo "0")
        if [ "$CONV_LEN" -gt 0 ]; then
            test_result "Conversation length" "true" "Length: $CONV_LEN"
        else
            test_result "Conversation length" "false" "Zero length"
        fi

    else
        test_result "JSON structure valid" "false" "Invalid JSON"
    fi
fi

# ============================================
# PHASE 5: Session Loading
# ============================================

if [ -n "$SESSION_FILE" ] && [ -f "$SESSION_FILE" ]; then
    print_header "PHASE 5: Session Loading"

    cat > /tmp/test_load_$$.txt <<EOF
%i
%h 5
%q
EOF

    echo -e "${GRAY}Running: gemma --session $SESSION_ID --load_session${NC}"
    timeout $TEST_TIMEOUT $GEMMA_EXE \
        --weights "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --session "$SESSION_ID" \
        --load_session \
        --verbosity 0 \
        < /tmp/test_load_$$.txt > /tmp/test_load_output_$$.txt 2>&1 || true

    LOAD_OUTPUT=$(cat /tmp/test_load_output_$$.txt)

    # Test 5.1: Load message displayed
    if echo "$LOAD_OUTPUT" | grep -qi "loaded\|loading"; then
        test_result "Load confirmation message" "true"
    else
        test_result "Load confirmation message" "false" "No load message found"
    fi

    # Test 5.2: Session info command works
    if echo "$LOAD_OUTPUT" | grep -q "Session Statistics"; then
        test_result "Session info (%i) works" "true"
    else
        test_result "Session info (%i) works" "false"
    fi

    # Test 5.3: Session history command works
    if echo "$LOAD_OUTPUT" | grep -q "Session History"; then
        test_result "Session history (%h) works" "true"
    else
        test_result "Session history (%h) works" "false"
    fi

    # Test 5.4: Statistics persisted
    if echo "$LOAD_OUTPUT" | grep -q "Total turns:"; then
        TURNS=$(echo "$LOAD_OUTPUT" | grep "Total turns:" | head -n1 | grep -o '[0-9]\+')
        if [ -n "$TURNS" ] && [ "$TURNS" -gt 0 ]; then
            test_result "Turn count persisted" "true" "Turns: $TURNS"
        else
            test_result "Turn count persisted" "false" "Zero turns"
        fi
    else
        test_result "Turn count persisted" "false" "No statistics in output"
    fi

    # Test 5.5: File integrity after load
    if python3 -m json.tool "$SESSION_FILE" > /dev/null 2>&1; then
        test_result "File integrity after load" "true"
    else
        test_result "File integrity after load" "false" "File corrupted"
    fi
fi

# ============================================
# PHASE 6: Interactive Commands
# ============================================

print_header "PHASE 6: Interactive Command Testing"

COMMANDS=(
    "i:Session Statistics:info"
    "h:Session History:history"
    "s:saved:save"
    "m:Managed Sessions:manager"
    "c:cleared:clear"
)

for cmd_spec in "${COMMANDS[@]}"; do
    IFS=':' read -r cmd keyword desc <<< "$cmd_spec"

    cat > /tmp/cmd_test_$$.txt <<EOF
%$cmd
%q
EOF

    timeout 30s $GEMMA_EXE \
        --weights "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --session "cmd_test_$cmd" \
        --verbosity 0 \
        < /tmp/cmd_test_$$.txt > /tmp/cmd_output_$$.txt 2>&1 || true

    OUTPUT=$(cat /tmp/cmd_output_$$.txt)

    if echo "$OUTPUT" | grep -qi "$keyword"; then
        test_result "Command %$cmd ($desc)" "true"
    else
        test_result "Command %$cmd ($desc)" "false" "Expected '$keyword' in output"
    fi
done

# ============================================
# PHASE 7: Multi-Session Management
# ============================================

print_header "PHASE 7: Multi-Session Management"

SESSION_ID_2="test_session_2_$(date +%H%M%S)"

cat > /tmp/multi_test_$$.txt <<EOF
Second session test message
%m
%q
EOF

timeout $TEST_TIMEOUT $GEMMA_EXE \
    --weights "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --session "$SESSION_ID_2" \
    --save_on_exit \
    --max_generated_tokens 30 \
    --verbosity 0 \
    < /tmp/multi_test_$$.txt > /tmp/multi_output_$$.txt 2>&1 || true

MULTI_OUTPUT=$(cat /tmp/multi_output_$$.txt)

# Test 7.1: Manager command shows sessions
if echo "$MULTI_OUTPUT" | grep -q "Managed Sessions"; then
    test_result "Session manager (%m)" "true"
else
    test_result "Session manager (%m)" "false"
fi

# Test 7.2: Multiple sessions created
SESSION_COUNT=$(ls -1 session_*.json 2>/dev/null | wc -l)
if [ "$SESSION_COUNT" -ge 2 ]; then
    test_result "Multiple sessions" "true" "Count: $SESSION_COUNT"
else
    test_result "Multiple sessions" "false" "Expected ≥2, found: $SESSION_COUNT"
fi

# ============================================
# PHASE 8: Error Handling
# ============================================

print_header "PHASE 8: Error Handling"

# Test 8.1: Load non-existent session
cat > /tmp/error_test_$$.txt <<EOF
%l nonexistent_session_file.json
%q
EOF

timeout 30s $GEMMA_EXE \
    --weights "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --session "error_test" \
    --verbosity 0 \
    < /tmp/error_test_$$.txt > /tmp/error_output_$$.txt 2>&1 || true

ERROR_OUTPUT=$(cat /tmp/error_output_$$.txt)

if echo "$ERROR_OUTPUT" | grep -qi "failed\|not found\|error"; then
    test_result "Missing file error handling" "true" "Graceful error message"
else
    test_result "Missing file error handling" "false" "No error message"
fi

# ============================================
# CLEANUP AND SUMMARY
# ============================================

# Cleanup temp files
rm -f /tmp/test_*.txt /tmp/cmd_*.txt /tmp/multi_*.txt /tmp/error_*.txt

print_header "TEST SUMMARY"

PASS_RATE=0
if [ $TESTS_TOTAL -gt 0 ]; then
    PASS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))
fi

echo -e "Total Tests: ${CYAN}$TESTS_TOTAL${NC}"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo -e "Skipped: ${YELLOW}$TESTS_SKIPPED${NC}"

if [ $PASS_RATE -ge 90 ]; then
    echo -e "Pass Rate: ${GREEN}$PASS_RATE%${NC} ✓ EXCELLENT"
elif [ $PASS_RATE -ge 80 ]; then
    echo -e "Pass Rate: ${GREEN}$PASS_RATE%${NC} ✓ GOOD"
elif [ $PASS_RATE -ge 70 ]; then
    echo -e "Pass Rate: ${YELLOW}$PASS_RATE%${NC} ⚠ ACCEPTABLE"
else
    echo -e "Pass Rate: ${RED}$PASS_RATE%${NC} ✗ NEEDS IMPROVEMENT"
fi

# Failed tests
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed Tests:${NC}"
    for result in "${TEST_RESULTS[@]}"; do
        if [[ $result == FAIL:* ]]; then
            echo -e "  ${RED}✗${NC} ${result#FAIL: }"
        fi
    done
fi

# Session files
echo -e "\n${CYAN}Session Files Created:${NC}"
if ls session_*.json >/dev/null 2>&1; then
    ls -lh session_*.json | awk '{printf "  - %-40s %8s\n", $9, $5}'
else
    echo "  None"
fi

# Save report
REPORT_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Gemma.exe Enhanced Session Persistence Test Results"
    echo "===================================================="
    echo "Date: $(date)"
    echo "Binary: $GEMMA_EXE"
    echo "Binary Date: $BIN_DATE"
    echo "Total: $TESTS_TOTAL | Passed: $TESTS_PASSED | Failed: $TESTS_FAILED | Skipped: $TESTS_SKIPPED"
    echo "Pass Rate: $PASS_RATE%"
    echo ""
    echo "Results:"
    for result in "${TEST_RESULTS[@]}"; do
        echo "  $result"
    done
} > "$REPORT_FILE"

echo -e "\n${CYAN}Detailed results: $REPORT_FILE${NC}"

# Cleanup prompt
echo -e "\n${YELLOW}Clean up session files? (y/N)${NC}"
read -r CLEANUP
if [ "$CLEANUP" = "y" ] || [ "$CLEANUP" = "Y" ]; then
    rm -f session_*.json
    echo -e "${GREEN}Session files removed${NC}"
else
    echo -e "${GRAY}Session files preserved for inspection${NC}"
fi

# Exit code
if [ $PASS_RATE -ge 80 ]; then
    echo -e "\n${GREEN}╔════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  OVERALL RESULT: PASS (≥80%)       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "\n${RED}╔════════════════════════════════════╗${NC}"
    echo -e "${RED}║  OVERALL RESULT: FAIL (<80%)       ║${NC}"
    echo -e "${RED}╚════════════════════════════════════╝${NC}"
    exit 1
fi
