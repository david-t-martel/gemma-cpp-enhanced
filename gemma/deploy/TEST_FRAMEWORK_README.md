# Gemma.exe Session Persistence Test Framework

Comprehensive test suite for validating session management features in gemma.exe.

## Test Scripts

### 1. `test_session_enhanced.sh` (Recommended)
**Full-featured test suite with 8 test phases:**

#### Phase 1: Binary and Environment Validation
- Binary existence, permissions, execution
- Model file availability
- Help command functionality

#### Phase 2: Feature Detection
- Automatically detects if binary supports session features
- Checks for `--session`, `--load_session`, `--save_on_exit` flags
- Validates binary build date
- **Aborts gracefully** if session features not available (prevents false failures)

#### Phase 3: Session Creation
- Creates test session with automated input
- Saves session to JSON file
- Validates file creation

#### Phase 4: Session File Validation
- JSON structure validation
- Required fields check (session_id, messages, conversation_length)
- Session ID preservation
- Message persistence verification
- Conversation length tracking

#### Phase 5: Session Loading
- Loads existing session
- Tests `%i` (info) and `%h` (history) commands
- Validates statistics persistence
- Checks file integrity after load

#### Phase 6: Interactive Command Testing
- Individual tests for each session command:
  - `%i` - Session statistics
  - `%h` - Session history
  - `%s` - Save session
  - `%m` - Session manager
  - `%c` - Clear session

#### Phase 7: Multi-Session Management
- Creates multiple sessions
- Tests session manager listing
- Validates concurrent session handling

#### Phase 8: Error Handling
- Tests loading non-existent files
- Validates graceful error messages

### 2. `test_session_persistence.sh` (Basic)
**Original test suite with fundamental checks:**
- Binary availability
- Session creation and save
- Session loading
- Basic validation

### 3. `test_session_persistence.ps1` (PowerShell)
**Windows-native PowerShell version:**
- Similar coverage to basic suite
- Better Windows integration
- Detailed JSON reporting

## Running Tests

### Prerequisites
```bash
# Ensure model files exist
ls -lh /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/

# Make test script executable
chmod +x test_session_enhanced.sh
```

### Quick Start
```bash
cd deploy

# Run enhanced test suite
./test_session_enhanced.sh

# Run basic test suite
./test_session_persistence.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File test_session_persistence.ps1
```

### Custom Configuration
```bash
# Use different model
MODEL_PATH=/path/to/model.sbs \
TOKENIZER_PATH=/path/to/tokenizer.spm \
./test_session_enhanced.sh

# Adjust timeout (default: 120s)
TEST_TIMEOUT=180 ./test_session_enhanced.sh
```

## Expected Results

### With Session-Enabled Binary
✅ **Pass Rate: 90-100%** (All phases should pass)

```
PHASE 1: Binary and Environment     [5/5 PASS]
PHASE 2: Feature Detection          [4/4 PASS]
PHASE 3: Session Creation           [1/1 PASS]
PHASE 4: Session File Validation    [5/5 PASS]
PHASE 5: Session Loading            [5/5 PASS]
PHASE 6: Interactive Commands       [5/5 PASS]
PHASE 7: Multi-Session Management   [2/2 PASS]
PHASE 8: Error Handling             [1/1 PASS]
```

### With Outdated Binary (No Session Support)
⚠️ **Graceful Abort After Phase 2**

The enhanced test will:
1. Detect missing session features in Phase 2
2. Display clear error message
3. Provide rebuild instructions
4. Skip remaining tests (prevents false failures)
5. Exit with informative summary

## Understanding Test Results

### Pass Rate Thresholds
- **≥90%**: Excellent - Full session functionality working
- **80-89%**: Good - Core features working, minor issues
- **70-79%**: Acceptable - Session basics work, implementation incomplete
- **<70%**: Needs Improvement - Significant issues

### Common Failure Patterns

#### 1. Binary Lacks Session Features
**Symptoms:**
- Phase 2 tests fail
- No `--session` flags in --help

**Solution:**
```bash
# Rebuild with session code
# Option 1: Visual Studio 2022 IDE
# - Open CMakeLists.txt
# - Build → Build All
# - Copy gemma.exe to deploy/

# Option 2: Build script (if working)
./build_oneapi.ps1 -Config perfpack -Jobs 10
```

#### 2. Session File Not Created
**Symptoms:**
- Phase 3 fails
- No session_*.json files created

**Possible Causes:**
- Write permissions in directory
- `SaveToFile()` implementation incomplete
- Binary crash during save

**Debug:**
```bash
# Check permissions
ls -la deploy/

# Run with verbose output
./gemma.exe --session test --save_on_exit --verbosity 2
```

#### 3. Session Data Not Persisted
**Symptoms:**
- Phase 4 tests fail
- JSON file exists but `messages` array empty
- `conversation_length` is 0

**Cause:**
- `SaveToFile()` implementation only saves metadata, not full conversation
- Need to implement message serialization

**Location:** `gemma.cpp/gemma/session.cc` lines 288-306

#### 4. Session Load Doesn't Restore State
**Symptoms:**
- Phase 5 tests fail
- Load command runs but session state not restored

**Cause:**
- `LoadFromFile()` implementation is stub (lines 308-328)
- Only validates JSON, doesn't deserialize data

**Fix:** Implement full deserialization in `LoadFromFile()`

## Test Artifacts

### Created Files
```
deploy/
├── session_test_*.json      # Test session files
├── test_results_*.txt        # Detailed test results
└── test_session_enhanced.sh  # Test script
```

### Session File Format
```json
{
  "session_id": "test_session_20251023_123456",
  "created_at": "2025-10-23T12:34:56Z",
  "conversation_length": 4,
  "total_turns": 2,
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "tokens": [2, 1234, 5678, 3]
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help?",
      "tokens": [2, 4321, 8765, 9876, 3]
    }
  ]
}
```

## Cleanup

### Manual Cleanup
```bash
# Remove test session files
rm -f session_*.json

# Remove test result reports
rm -f test_results_*.txt
```

### Automated Cleanup
The test script prompts for cleanup at the end:
```
Clean up session files? (y/N)
```

## Troubleshooting

### Test Hangs or Times Out
```bash
# Reduce timeout
TEST_TIMEOUT=60 ./test_session_enhanced.sh

# Check model loading
./gemma.exe --weights /path/to/model.sbs --tokenizer /path/to/tokenizer.spm --verbosity 2
```

### Python JSON Validation Fails
```bash
# Ensure Python 3 available
python3 --version

# Manually validate JSON
python3 -m json.tool session_test_*.json
```

### Permission Errors
```bash
# Fix script permissions
chmod +x test_session_enhanced.sh

# Fix binary permissions
chmod +x gemma.exe

# Check directory permissions
ls -ld deploy/
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run Session Tests
  run: |
    cd deploy
    chmod +x test_session_enhanced.sh
    ./test_session_enhanced.sh
  env:
    MODEL_PATH: ${{ github.workspace }}/.models/gemma-2b-it.sbs
    TOKENIZER_PATH: ${{ github.workspace }}/.models/tokenizer.spm
```

### Exit Codes
- **0**: Pass rate ≥80% (CI should pass)
- **1**: Pass rate <80% or critical failure (CI should fail)

## Advanced Usage

### Parallel Testing
```bash
# Test multiple binaries
for binary in gemma_v1.exe gemma_v2.exe; do
    GEMMA_EXE=$binary ./test_session_enhanced.sh
done
```

### Performance Benchmarking
```bash
# Compare session vs non-session performance
time ./gemma.exe --session test --max_generated_tokens 100 < input.txt
time ./gemma.exe --max_generated_tokens 100 < input.txt
```

### Coverage Analysis
```bash
# Count lines executed in session.cc
gcov gemma.cpp/gemma/session.cc
```

## Contributing

When adding new session features:
1. Add corresponding test case to Phase 6 (Interactive Commands)
2. Update expected pass rate documentation
3. Add troubleshooting section for common failures
4. Update session file format documentation

## Support

- **Build Issues**: See `DEPLOYMENT_GUIDE.md`
- **Session Features**: See `deploy/README.txt`
- **Test Failures**: Check `test_results_*.txt` for details
