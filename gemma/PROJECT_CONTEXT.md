# Gemma.cpp Project Context
*Saved: 2025-09-22*

## Quick Restore Command
To restore this context in a new session, ask Claude to:
"Load the Gemma.cpp project context from memory - search for 'Gemma.cpp Project' entities"

## Project Summary
**Gemma.cpp** - Lightweight C++ inference engine for Google's Gemma foundation models
- **Location**: C:\codedev\llm\gemma
- **Status**: Build issues RESOLVED, CMake configuration FIXED
- **Working Executable**: C:/codedev/llm/gemma/gemma.cpp/gemma.exe

## Key Achievements
✅ Resolved 15.9GB of redundant failed builds
✅ Fixed CMake duplicate Highway target issues
✅ Cleaned 12GB of build artifacts
✅ Established clean dependency resolution chain
✅ Updated all documentation files

## Critical Paths & Commands

### Build Command
```bash
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4
```

### Model Locations
- Models: C:\codedev\llm\.models\
- Example: C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\
- Formats: .sbs (weights), .spm (tokenizer)

### Key Files Modified
1. `CMakeLists.txt` (lines 249-265) - Highway detection logic
2. `cmake/Dependencies.cmake` (lines 20-35) - Local Highway handling
3. `CLAUDE.md` - Simplified build instructions
4. `.claude/CLAUDE.md` - Developer guide

## Technical Decisions
- **Dependency Strategy**: Local Highway → vcpkg → FetchContent
- **Build System**: Visual Studio 2022 with v143 toolset
- **CMake Policies**: CMP0126, CMP0169 set to NEW
- **Removed**: highway_scalar_fallback.h (BF16 errors)

## Agent Coordination Lessons
- Used parallel specialized agents (deployment-engineer, cpp-pro, docs-architect)
- Emphasized 60-120 second timeouts for builds
- All agents used MCP tools for efficiency
- User preference: Multiple agents working simultaneously

## Next Steps (Roadmap)
1. Test inference with 2B and 4B models
2. GPU acceleration (CUDA/SYCL currently disabled)
3. MCP server integration
4. Advanced sampling algorithms
5. Intel oneAPI optimization

## Environment Details
- **CMake**: 4.1.1 at C:\Program Files\CMake\bin\cmake.exe
- **vcpkg**: Installed but detection problematic
- **Bazel**: Available as alternative
- **Compiler**: MSVC 2022 / Visual Studio 17

## Context Retrieval
This context has been saved to the memory system under:
- Entity: "Gemma.cpp Project"
- Related entities for build state, design decisions, configuration, etc.

Use memory search to retrieve: `search_nodes("Gemma.cpp")`