# Gemma.cpp Project Context - 2025-09-22

## Project Overview

**Project**: Gemma.cpp - Lightweight C++ inference engine for Google's Gemma foundation models
**Status**: Build system fixed, executables now properly output to build/bin/
**Technology Stack**: C++20, CMake 3.14+, Visual Studio 2022, Highway SIMD, vcpkg (optional)
**Critical Achievement**: Fixed CMake output structure preventing executables from polluting source tree

## Critical Issue Resolution

### Problem Identified
Executables were being generated in the source directory (`gemma.cpp/`) instead of the proper build output directory (`build/bin/`). This was causing:
- Source tree pollution with .exe files
- Confusion about which executables to run
- Improper separation of source and build artifacts

### Solution Implemented
Complete restructuring of CMake output directories:
```cmake
# Main CMakeLists.txt
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Per-configuration for MSVC
if(MSVC)
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} ${CMAKE_BINARY_DIR}/bin/${config})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} ${CMAKE_BINARY_DIR}/lib/${config})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} ${CMAKE_BINARY_DIR}/lib/${config})
  endforeach()
endif()
```

## Current Project State

### Directory Structure
```
C:\codedev\llm\gemma\
├── build/                      # Build directory
│   ├── bin/                    # Executable output (NEW - properly configured)
│   │   ├── Debug/              # Debug executables
│   │   └── Release/            # Release executables
│   ├── lib/                    # Library output
│   └── CMakeCache.txt          # CMake configuration
├── gemma.cpp/                  # Source directory
│   ├── gemma.exe              # OLD executable (to be removed)
│   ├── CMakeLists.txt         # Fixed with output properties
│   └── ops/
│       └── highway_scalar_fallback.h  # RESTORED from backup
├── cmake/
│   └── Dependencies.cmake      # Fixed with hwy target alias
└── CMakeLists.txt             # Main build file (FIXED)
```

### Build Status
- **8 background builds running** with mixed results
- **CMake configuration timeouts** occurring (needs investigation)
- **Working executable** exists at old location: `C:/codedev/llm/gemma/gemma.cpp/gemma.exe`
- **New executables** should appear in: `build/bin/Release/gemma.exe`

## Design Decisions

### 1. Output Directory Strategy
- **Executables**: `build/bin/[CONFIG]/` for clean separation
- **Libraries**: `build/lib/[CONFIG]/` following standards
- **Multi-config support**: Separate Debug/Release directories for MSVC
- **Install rules**: Using GNUInstallDirs for standard compliance

### 2. Dependency Resolution
Priority order established:
1. **Local Highway** (if exists in tree)
2. **vcpkg Highway** (if available)
3. **FetchContent** (fallback from GitHub)

Target alias added for compatibility:
```cmake
if(NOT TARGET hwy AND TARGET hwy::hwy)
  add_library(hwy ALIAS hwy::hwy)
endif()
```

### 3. Missing Header Recovery
`highway_scalar_fallback.h` was missing, causing BF16 conversion errors. Restored from backup at:
`C:\codedev\llm\gemma\gemma.cpp\ops\highway_scalar_fallback.h`

## Code Patterns Established

### CMake Target Configuration
```cmake
# For each executable target
set_target_properties(gemma PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
)
```

### Include Path Fixes
```cpp
// Fixed in c_api.h
#include "gemma.cpp/gemma.h"  // Changed from "gemma/gemma.h"
```

### .gitignore Updates
```gitignore
# Exclude executables from source tree
*.exe
*.dll
*.so
*.dylib
```

## Agent Coordination Summary

### Agents Involved
1. **cpp-pro**: Initial analysis, identified output directory misconfiguration
2. **debugger**: Found missing headers, traced BF16 conversion errors
3. **deployment-engineer**: Implemented proper CMake output configuration
4. **context-manager**: Consolidated findings and saved state

### Key Findings (Unanimous)
- All agents identified executables in wrong location as critical issue
- Missing `highway_scalar_fallback.h` was blocking successful builds
- CMake output directories needed complete restructuring
- Target alias required for hwy dependency compatibility

## Files Modified

### Critical Changes
1. **C:\codedev\llm\gemma\CMakeLists.txt**
   - Added complete output directory configuration
   - Set per-configuration directories for MSVC

2. **C:\codedev\llm\gemma\gemma.cpp\CMakeLists.txt**
   - Fixed all executable targets with RUNTIME_OUTPUT_DIRECTORY
   - Ensured proper output paths

3. **C:\codedev\llm\gemma\cmake\Dependencies.cmake**
   - Added hwy target alias for compatibility
   - Fixed dependency resolution order

4. **C:\codedev\llm\gemma\gemma.cpp\ops\highway_scalar_fallback.h**
   - Restored from backup to fix BF16 conversion errors

5. **C:\codedev\llm\gemma\gemma.cpp\c_api.h**
   - Fixed include path from "gemma/gemma.h" to "gemma.cpp/gemma.h"

6. **C:\codedev\llm\gemma\.gitignore**
   - Updated to exclude executables from source tree

## Known Issues

### Current Problems
1. **Background builds**: 8 builds running, some timing out
2. **CMake timeouts**: Configuration phase taking too long
3. **Old executable**: Still present in source directory
4. **Build verification**: Need to confirm new output directories work

### Required Actions
1. Kill all background builds: `taskkill /F /IM cmake.exe /IM cl.exe /IM MSBuild.exe`
2. Clean build directory and start fresh
3. Test CPU-only build to verify output directories
4. Remove old executables from source tree

## Future Roadmap

### Immediate Tasks
1. **Kill background processes** and clean state
2. **Fresh build** with fixed configuration
3. **Verify** executables appear in `build/bin/Release/`
4. **Test** inference with proper paths
5. **Document** working build commands

### Short-term Goals
1. **Simplify** build system if possible
2. **Fix** CMake configuration timeouts
3. **Optimize** dependency resolution
4. **Create** build presets for common configurations
5. **Add** CI/CD validation

### Long-term Vision
1. **GPU support** (CUDA, SYCL, Vulkan)
2. **MCP integration** for agent communication
3. **Performance** optimization with profiling
4. **Package** management improvements
5. **Cross-platform** testing automation

## Working Commands

### Build (After Fixes)
```batch
cd C:\codedev\llm\gemma
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4
```

### Run (From Proper Location)
```batch
.\build\bin\Release\gemma.exe ^
  --tokenizer C:\codedev\llm\.models\tokenizer.spm ^
  --weights C:\codedev\llm\.models\gemma2-2b-it-sfp.sbs
```

### Clean Rebuild
```batch
rmdir /s /q build
cmake -B build -G "Visual Studio 17 2022" -T v143
cmake --build build --config Release -j 4
```

## Model Paths (Verified)

- **2B Model**: `C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs`
- **4B Model**: `C:\codedev\llm\.models\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs`
- **Tokenizer**: `C:\codedev\llm\.models\tokenizer.spm`

## Success Criteria

✅ **Completed**:
- CMake output directories configured
- Missing headers restored
- Dependency conflicts resolved
- Target aliases added
- Include paths fixed

⏳ **Pending Verification**:
- Executables in build/bin/Release/
- Clean build succeeds
- Inference runs from new location
- Background processes terminated
- Build time reasonable

## Context Saved

This comprehensive context captures the critical state after fixing the Gemma.cpp build system. The main achievement was identifying and fixing the CMake configuration that was placing executables in the source directory instead of the proper build output directory. All agents involved (cpp-pro, debugger, deployment-engineer) unanimously identified this as the critical issue and contributed to the solution.

**Timestamp**: 2025-09-22
**Version**: Post-CMake-Fix
**Status**: Build system restructured, awaiting verification