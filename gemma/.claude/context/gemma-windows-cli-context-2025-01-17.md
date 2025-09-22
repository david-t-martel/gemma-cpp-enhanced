# Gemma.cpp Windows CLI Project Context
**Timestamp**: 2025-01-17
**Project Path**: C:\codedev\llm\gemma
**Repository**: https://github.com/david-t-martel/dtm-gemma (private)

## 1. Project Overview

### Goals
- Build Windows-native Gemma.cpp inference engine with CLI interface similar to Claude/Gemini-CLI
- Provide local LLM inference without cloud dependencies
- Create user-friendly Python wrapper for interactive chat

### Key Achievements
- ✅ Successfully compiled Windows-native gemma.exe (3.1MB) without WSL dependency
- ✅ Python CLI wrapper with chat interface implemented
- ✅ Static build avoiding DLL dependencies
- ⚠️ Runtime error when loading models (needs resolution)

### Technology Stack
- **C++**: C++20 with template metaprogramming
- **Compiler**: Visual Studio 2022 (MSVC)
- **Build System**: CMake with presets
- **SIMD**: Highway library for vectorization
- **Python**: CLI wrapper using subprocess
- **Package Manager**: uv for Python dependencies

## 2. Current State

### Working Components
- **gemma.exe**: Compiled successfully, shows help, accepts parameters
- **Python CLI**: gemma-cli.py provides interactive chat interface
- **Build System**: CMake configuration with Visual Studio generator
- **Documentation**: Comprehensive docs (GEMMA_CLI_USAGE.md, WINDOWS_BUILD_SUCCESS.md)

### Known Issues
- **Error 3221226356**: Runtime error when loading model weights
- **Griffin Support**: Disabled due to MSVC template instantiation issues
- **Model Compatibility**: Potential mismatch between binary and model format

### Available Models
Location: `C:\codedev\llm\.models\`
- gemma2-2b-it-sfp.sbs (recommended starter)
- gemma3-1b-it-sfp.sbs (fastest)
- tokenizer.spm
- Other 4B variants

## 3. Design Decisions

### Build Architecture
- **Visual Studio 2022**: Chosen over MinGW for superior Windows integration
- **Static Linking**: 3.1MB executable avoids runtime DLL dependencies
- **Griffin Stub**: Created griffin_stub.cc to bypass MSVC compilation issues
- **CMake Presets**: Organized build configurations (windows, windows-dll)

### Python CLI Design
- **Subprocess Management**: Uses STARTUPINFO for hidden console windows
- **Streaming Output**: Parses stdout for real-time token streaming
- **Error Handling**: Comprehensive try-catch with graceful fallbacks
- **Package Management**: uv as per CLAUDE.md requirements

### Code Patterns
```python
# Windows subprocess configuration
startupinfo = subprocess.STARTUPINFO()
startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startupinfo.wShowWindow = subprocess.SW_HIDE

# Streaming token output
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    startupinfo=startupinfo
)
```

## 4. File Structure

```
gemma/
├── gemma.cpp/              # Core C++ inference engine
│   ├── build/              # Build output directory
│   │   └── gemma.exe       # Windows executable (3.1MB)
│   ├── gemma/              # Source code
│   │   ├── griffin_stub.cc # MSVC compatibility stub
│   │   └── ...
│   ├── CMakeLists.txt      # Build configuration
│   └── CMakePresets.json   # Build presets
├── gemma-cli.py            # Python CLI wrapper
├── demo_cli.py             # Demo script
├── example_usage.py        # Usage examples
├── test_results.md         # Test documentation
└── .claude/
    └── context/            # Project context storage
```

## 5. Agent Coordination History

### Participating Agents
1. **deployment-engineer**: Created GitHub repo, initial WSL build
2. **cpp-pro**: Fixed griffin.cc issues, created Windows build
3. **python-pro**: Developed CLI wrapper and demonstration scripts
4. **test-automator**: Generated comprehensive test suite
5. **code-reviewer**: Critical assessment and gap analysis
6. **Multiple build attempts**: build2, build_new, build_fresh, build_vs directories

### Key Milestones
- Initial WSL build successful (Linux binary)
- Windows build attempts with various compilers
- Griffin support disabled to achieve compilation
- Python CLI wrapper completed
- GitHub repository created and organized

## 6. Technical Details

### Build Commands
```bash
# Configure
cmake --preset windows

# Build
cmake --build --preset windows -j 4

# Alternative with DLL
cmake --preset windows-dll
cmake --build --preset windows-dll -j 4
```

### Runtime Configuration
```python
RUNTIME_CONFIG = {
    'max_seq_len': 32768,
    'temperature': 0.7,
    'top_k': 40,
    'decode_qbatch_size': 32,
    'prefill_tbatch_size': 64
}
```

### Error Codes
- **3221226356**: Model loading failure (current blocker)
- **0xC0000005**: Access violation (related to above)

## 7. Future Roadmap

### Immediate Priorities
1. **Fix Model Loading**: Resolve error 3221226356
   - Check Visual C++ runtime dependencies
   - Verify model format compatibility
   - Test with different compression formats

2. **Testing**: Execute comprehensive test suite with working models

3. **Griffin Support**: Investigate MSVC-compatible implementation

### Medium-term Goals
- Model download automation from Kaggle
- PTY implementation for better interactive experience
- Performance benchmarking against WSL version
- Docker containerization for distribution

### Long-term Vision
- GUI interface using Qt or web UI
- Model quantization tools
- Fine-tuning capabilities
- Integration with other tools (VS Code, etc.)

## 8. Critical Information

### Model Weights Setup
Models must be obtained from:
- Kaggle: https://www.kaggle.com/models/google/gemma-2/gemmaCpp
- Hugging Face: Alternative source

Extract to: `C:\codedev\llm\.models\`

### Dependencies
- Visual Studio 2022 with C++ workload
- CMake 3.27+
- Python 3.10+
- uv package manager
- Visual C++ Redistributables (potentially missing)

### Known Constraints
- MSVC template instantiation limitations
- Windows path length limitations (use short paths)
- Memory requirements: 8GB+ RAM recommended
- CPU: AVX2 support required for SIMD optimizations

## 9. Testing Strategy

### Unit Tests
- Located in gemma.cpp/test/
- Require working model for execution
- Cover tokenization, inference, streaming

### Integration Tests
- Python CLI tests in test_results.md
- End-to-end inference validation
- Performance benchmarks

### Manual Testing
```bash
# Basic execution test
gemma.exe --help

# Model loading test (currently failing)
gemma.exe --weights C:\codedev\llm\.models\gemma2-2b-it-sfp.sbs --tokenizer C:\codedev\llm\.models\tokenizer.spm

# Python CLI test
python gemma-cli.py --model gemma2-2b-it
```

## 10. Contact and Resources

### Documentation
- GEMMA_CLI_USAGE.md: User guide
- WINDOWS_BUILD_SUCCESS.md: Build instructions
- CRITICAL_REVIEW.md: Architecture assessment
- FINAL_WORKING_STATUS.md: Current state

### External Resources
- Gemma.cpp GitHub: https://github.com/google/gemma.cpp
- Highway Library: https://github.com/google/highway
- Model Downloads: Kaggle/Hugging Face

---
*This context should be used to restore project state and continue development. Priority: Fix model loading issue to achieve working inference.*