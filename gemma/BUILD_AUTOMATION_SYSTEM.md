# Gemma.cpp Build Automation System

Comprehensive versioned build, test, and release automation with hash-linked releases.

## Overview

This build automation system provides:
- **Versioned builds** with Git-based version generation
- **Hash-linked releases** for reproducibility
- **Multiple build frontends** (Just, Make, CMake)
- **CI/CD integration** via GitHub Actions
- **Automated testing** and deployment

## Quick Start

### Using Just (Recommended - Modern)
```bash
# Install Just: https://github.com/casey/just
just                 # Show all available recipes
just build           # Build with default settings
just test            # Run all tests
just deploy          # Deploy to deploy/
just release         # Create versioned release package
```

### Using Make (Traditional)
```bash
make help            # Show all targets
make build           # Build project
make test            # Run tests
make package         # Create release package
make install         # Install to system
```

### Using CMake Directly
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 10
```

## Build System Architecture

### 1. Version Management (`cmake/Version.cmake`)

Automatically generates version information from Git:

**Version Format**: `MAJOR.MINOR.PATCH[+commits][-prerelease].HASH[-dirty]`

**Examples**:
- `v1.2.3.a1b2c3d4` - Release build on tag v1.2.3
- `1.2.3+5.a1b2c3d4-dev` - 5 commits after v1.2.3 on develop branch
- `1.2.3.a1b2c3d4-dirty` - Uncommitted changes

**Generated Variables**:
- `GEMMA_VERSION_MAJOR/MINOR/PATCH` - Version components
- `GEMMA_VERSION_STRING` - Semantic version (e.g., "1.2.3")
- `GEMMA_VERSION_FULL` - Full version with hash
- `GEMMA_BUILD_IDENTIFIER` - Complete build identifier
- `GEMMA_GIT_COMMIT_HASH` - Short commit hash (8 chars)
- `GEMMA_GIT_COMMIT_HASH_FULL` - Full commit hash
- `GEMMA_GIT_BRANCH` - Current branch name
- `GEMMA_BUILD_TIMESTAMP` - ISO 8601 timestamp
- `GEMMA_BUILD_HASH` - Hash of build configuration

**Usage in CMakeLists.txt**:
```cmake
# Include version generation
include(cmake/Version.cmake)

# Use version variables
set(PROJECT_VERSION ${GEMMA_VERSION_STRING})
target_compile_definitions(gemma PRIVATE
    GEMMA_VERSION="${GEMMA_VERSION_FULL}"
)
```

### 2. Version Header (`cmake/version.h.in`)

Generates C++ header with version constants:

```cpp
#include <gemma/version.h>

// Access version at runtime
std::cout << gemma::version::VERSION << "\n";  // "1.2.3"
std::cout << gemma::version::get_build_info() << "\n";  // Full info

// Compile-time version checks
#if GEMMA_VERSION_MAJOR >= 1
    // Feature available in v1.x+
#endif
```

### 3. Justfile (Modern Task Runner)

**Key Features**:
- Cross-platform (Windows/Linux/macOS)
- Parameterized recipes
- Environment variable support
- Rich error handling
- Task dependencies

**Example Recipes**:
```bash
# Build variants
just build-msvc Release      # MSVC build
just build-oneapi perfpack   # Intel oneAPI with all optimizations

# Testing
just test-smoke              # Quick validation
just test-inference          # Model inference test
just test-session            # Session management tests

# Deployment
just deploy                  # Copy binary to deploy/
just package v1.2.3          # Create versioned package
just release                 # Build + package with auto-version

# Development
just format                  # Format C++ code
just lint                    # Run linter
just compile-db              # Generate compile_commands.json

# Utilities
just run                     # Interactive inference
just run-session dev         # Session-enabled mode
just models                  # List available models
just requirements            # Check system requirements
```

### 4. Makefile (Traditional)

**Key Targets**:
```bash
# Primary targets
make                         # Default: build
make build BUILD_TYPE=Release
make test
make clean

# Build variants
make build-debug
make build-fast              # Ninja build
make build-oneapi            # Intel oneAPI

# Testing
make test-smoke
make test-session
make benchmark

# Deployment
make deploy
make package
make install PREFIX=/usr/local

# Development
make format
make lint
make compile-db

# Utilities
make version
make status
make requirements
```

### 5. GitHub Actions CI/CD

**Workflow**: `.github/workflows/build-test-release.yml`

**Stages**:

#### 1. Version Generation
- Extracts version from Git tags
- Generates build identifier
- Exports for later stages

#### 2. Build Matrix
- **Windows MSVC** - Visual Studio 2022 v143
- **Windows Intel oneAPI** - ICX compiler with MKL/IPP/TBB
- **Ubuntu GCC** - Latest GCC
- **macOS Clang** - Latest Clang

**Build Features**:
- Artifact caching (ccache/sccache)
- Parallel builds (10 cores max)
- Version embedding in binary
- Upload build artifacts with version

#### 3. Testing
- Smoke tests (binary execution)
- Version verification
- Unit tests (via CTest)
- Cross-platform validation

#### 4. Packaging & Release
- **Triggered on**: Git tags matching `v*`
- Creates platform-specific packages:
  - Windows: `gemma-VERSION-windows-x64.zip`
  - Linux: `gemma-VERSION-linux-x86_64.tar.gz`
  - macOS: `gemma-VERSION-macos-arm64.tar.gz`
- Generates SHA256 checksums
- Creates GitHub release with notes
- Marks pre-releases (`alpha`, `beta`, `rc`)

#### 5. Notifications
- Build status reporting
- Failure notifications
- Release announcements

## Hash-Linked Release Framework

### Build Hash Generation

Each build gets a unique hash based on configuration:

```
BUILD_HASH = MD5(
    BUILD_TYPE +
    COMPILER_ID +
    SYSTEM_NAME +
    SYSTEM_PROCESSOR
)[:8]
```

### Build Identifier Format

```
BUILD_IDENTIFIER = VERSION_FULL-BUILD_VARIANT-COMPILER_NAME-BUILD_HASH
```

**Example**: `1.2.3+5.a1b2c3d4-dev-release-icx-e4d3b2a1`

**Components**:
- `1.2.3+5` - Version 1.2.3, 5 commits ahead
- `a1b2c3d4` - Git commit hash
- `dev` - Branch suffix (develop branch)
- `release` - Build type
- `icx` - Intel ICX compiler
- `e4d3b2a1` - Build configuration hash

### Reproducible Builds

Given a build identifier, you can reproduce the exact build:

```bash
# Extract components from identifier
VERSION_FULL="1.2.3+5.a1b2c3d4-dev"
BUILD_VARIANT="release"
COMPILER="icx"
BUILD_HASH="e4d3b2a1"

# Checkout exact commit
git checkout a1b2c3d4

# Match build configuration
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=icx \
    # ... additional flags to match BUILD_HASH

# Verify hash matches
cmake -B build -DGEMMA_VERIFY_BUILD_HASH=e4d3b2a1
```

### Release Artifacts

Each release includes:
- **Binary**: `gemma.exe` or `gemma`
- **Configuration**: `gemma.config.toml`
- **Documentation**: `DEPLOYMENT_GUIDE.md`, `README.txt`
- **Examples**: Sample session files
- **Version Info**: Embedded in binary
- **Checksum**: SHA256 verification file
- **Build Metadata**: CMake cache, version header

## Build Variants

### 1. Debug Build
```bash
just build-msvc Debug
# OR
make build-debug
```
- Optimizations: `-O0 -g`
- Assertions: Enabled
- Debug symbols: Full
- Use case: Development, debugging

### 2. Release Build
```bash
just build-msvc Release
# OR
make build
```
- Optimizations: `-O3`
- Assertions: Disabled
- Debug symbols: Stripped
- Use case: Production deployment

### 3. RelWithDebInfo Build
```bash
just build RelWithDebInfo
```
- Optimizations: `-O2 -g`
- Assertions: Disabled
- Debug symbols: Preserved
- Use case: Performance profiling

### 4. Intel oneAPI Optimized
```bash
just build-oneapi perfpack
```
- Compiler: Intel ICX
- Libraries: MKL, IPP, TBB, DNNL
- Optimizations: `-O3 -xHost -march=native -mavx2 -flto`
- Use case: Maximum performance

## Version Management Workflow

### 1. Development (Continuous)
```bash
# On feature branch or develop
git commit -m "feat: add new feature"

# Build shows: 1.2.3+12.a1b2c3d4-dev
just build
```

### 2. Pre-Release
```bash
# Create release branch
git checkout -b release/v1.3.0

# Tag release candidate
git tag -a v1.3.0-rc1 -m "Release candidate 1"

# Build shows: 1.3.0-rc1.d4e5f6a7-rc
just release
```

### 3. Release
```bash
# Tag final release
git tag -a v1.3.0 -m "Release v1.3.0"

# Push tag (triggers CI/CD)
git push origin v1.3.0

# GitHub Actions:
# - Builds all platforms
# - Runs tests
# - Creates packages
# - Publishes GitHub release
```

### 4. Hotfix
```bash
# Create hotfix branch
git checkout -b hotfix/v1.3.1 v1.3.0

# Make fixes
git commit -m "fix: critical bug"

# Tag hotfix
git tag -a v1.3.1 -m "Hotfix v1.3.1"

# Push and deploy
git push origin v1.3.1
```

## Testing Framework

### Test Hierarchy
```
1. Smoke Tests (seconds)
   - Binary execution
   - Help flag
   - Version display

2. Unit Tests (minutes)
   - Component testing
   - Isolated functionality

3. Integration Tests (minutes)
   - Session management
   - Model loading
   - Inference pipeline

4. Performance Tests (minutes)
   - Benchmarks
   - Profiling
   - Regression detection
```

### Running Tests

**All Tests**:
```bash
just test           # Complete test suite
make test           # Traditional make
```

**Selective Testing**:
```bash
# Quick validation
just test-smoke

# Specific feature
just test-session

# Performance
just benchmark
```

### CI/CD Testing

GitHub Actions runs:
- Smoke tests on all platforms
- Unit tests (via CTest)
- Integration tests (session management)
- Performance regression detection

## Build Configuration Reference

### Environment Variables

```bash
# Build configuration
BUILD_TYPE=Release|Debug|RelWithDebInfo
JOBS=10                    # Parallel build jobs
PREFIX=/usr/local          # Install prefix

# Compiler selection
CXX=g++|clang++|icx
CC=gcc|clang|icx

# Cache locations
CCACHE_DIR=~/.cache/ccache
SCCACHE_DIR=~/.cache/sccache

# Model paths
MODEL_PATH=/path/to/models
```

### CMake Options

```bash
# Build type
-DCMAKE_BUILD_TYPE=Release

# Compiler selection
-DCMAKE_CXX_COMPILER=icx
-DCMAKE_C_COMPILER=icx

# oneAPI features
-DGEMMA_USE_ONEAPI_LIBS=ON
-DGEMMA_USE_TBB=ON
-DGEMMA_USE_IPP=ON
-DGEMMA_USE_DNNL=ON

# Backends
-DGEMMA_ENABLE_SYCL=ON
-DGEMMA_ENABLE_CUDA=ON
-DGEMMA_ENABLE_VULKAN=ON

# Advanced
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
-DCMAKE_VERBOSE_MAKEFILE=ON
-DCMAKE_COLOR_DIAGNOSTICS=ON
```

## Troubleshooting

### Version Not Detected
```bash
# Ensure Git repository initialized
git init

# Create initial tag
git tag -a v0.1.0 -m "Initial version"

# Verify
just version
```

### Build Hash Mismatch
```bash
# Verify exact compiler version
$CXX --version

# Check system information
uname -a

# Regenerate hash
cmake -B build -DGEMMA_REGENERATE_HASH=ON
```

### CI/CD Failures
```bash
# Check workflow syntax
gh workflow view build-test-release

# View recent runs
gh run list

# Debug failed run
gh run view <run-id> --log-failed
```

## Best Practices

### 1. Version Tags
- ✅ Use semantic versioning: `v1.2.3`
- ✅ Tag on main branch for releases
- ✅ Use pre-release tags: `v1.2.3-rc1`
- ❌ Don't tag feature branches

### 2. Build Reproducibility
- ✅ Commit lock files (if any)
- ✅ Document compiler versions
- ✅ Pin dependency versions
- ✅ Use build hashes for verification

### 3. CI/CD
- ✅ Run on all commits to main
- ✅ Test on multiple platforms
- ✅ Cache build artifacts
- ✅ Limit parallel jobs (max 10)

### 4. Release Process
- ✅ Update CHANGELOG.md
- ✅ Test release candidates
- ✅ Generate checksums
- ✅ Sign releases (future)

## Integration with Existing Tools

### VS Code
```json
{
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "cmake.configureArgs": [
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  ],
  "C_Cpp.default.compileCommands": "${workspaceFolder}/compile_commands.json"
}
```

### CLion
- Automatically detects CMakeLists.txt
- Supports custom build profiles
- Integrates with CTest

### Visual Studio 2022
- Open folder → CMakeLists.txt
- Configure with VS generator
- Build from IDE

## Future Enhancements

### Planned
- [ ] Signed releases (GPG)
- [ ] Docker container builds
- [ ] Cross-compilation support
- [ ] Automated changelog generation
- [ ] Performance regression tracking
- [ ] Binary size monitoring
- [ ] Dependency vulnerability scanning

### Under Consideration
- [ ] Nix/Guix reproducible builds
- [ ] SBOM (Software Bill of Materials)
- [ ] SLSA provenance
- [ ] Multi-architecture releases (ARM)

## Support

### Documentation
- Build system: This file
- CMake: `CLAUDE.md`
- Deployment: `deploy/DEPLOYMENT_GUIDE.md`

### Getting Help
- Build issues: Check `BUILD_TROUBLESHOOTING.md`
- CI/CD: Review GitHub Actions logs
- Versions: Run `just version` or `make version`

### Reporting Issues
Include:
- Build identifier from binary
- CMake configuration output
- Build logs
- System information (`just requirements`)
